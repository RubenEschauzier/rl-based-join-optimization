from _pytest._io import terminalwriter
import pdb
from collections import defaultdict
import os
import sys
from time import sleep

import diskcache
import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader

from src.datastructures.query_cardinality_dataset import QueryCardinalityDataset
from src.models.epistemic_neural_network import MultiHeadEpistemicNetwork, prepare_epinet_model
from src.pretrain_procedure import DualMetricScheduler
from src.utils.epinet_utils.calibration_plot import compute_calibration_measures, calculate_calibration_metrics
from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.training_tracking import TrainSummary, ExperimentWriter

# Get the path of the parent directory (the root of the project)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.models.model_instantiator import ModelFactory
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset, prepare_data
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask
import torch


class MetricsTracker:
    """Tracks and aggregates validation metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.p_values = []
        self.distribution_variances = []

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def update_calibration(self, p_vals, dist_vars):
        self.p_values.extend(p_vals)
        self.distribution_variances.extend(dist_vars)

    def reset_calibration_stats(self):
        self.p_values = []
        self.distribution_variances = []

    def summarize(self):
        """Returns the mean for all scalar metrics."""
        return {k: np.mean(v).item() for k, v in self.metrics.items()}




def prepare_cardinality_estimator(model_config, model_directory=None):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    if model_directory:
        load_weights_from_pretraining(gine_conv_model, model_directory,
                                      "embedding_model.pt",
                                      ["head_cardinality.pt"],
                                      float_weights=True)
    return gine_conv_model


def print_param_count(epinet_cost_estimation, train_epi_network):
    params_cost_estimate = list(epinet_cost_estimation.cost_estimation_model.parameters())

    total_params_cost_estimation = 0
    for param in params_cost_estimate:
        total_params_cost_estimation += param.numel()
    print(f"Cost estimation model has {total_params_cost_estimation} parameters")

    if train_epi_network:
        total_params = 0
        for param in epinet_cost_estimation.parameters():
            total_params += param.numel()
        print(f"Epinet model has {total_params - total_params_cost_estimation} parameters")


def fetch_or_compute_priors(query_batch, valid_indices, query_plans, cache, epinet, precomputed_indexes,
                            precomputed_masks, device):
    """Retrieves priors for all heads from cache or computes and stores them if missing."""
    priors = {}
    missing_indices = []

    for i in valid_indices:
        q_id = query_batch.query[i]
        cached_priors = cache.get(q_id)
        if cached_priors is not None:
            # Retrieve all cached heads and move to target device
            priors[i] = {head: tensor.to(device) for head, tensor in cached_priors.items()}
        else:
            missing_indices.append(i)

    if missing_indices:
        with torch.no_grad():
            embedded_prior = epinet.embed_query_batched_prior(query_batch.to(device))
            for i in missing_indices:
                q_id = query_batch.query[i]
                plans_current_query = query_plans[q_id]

                unweighted_ensemble_prior = epinet.compute_ensemble_prior(
                    plans_current_query, embedded_prior, precomputed_indexes, precomputed_masks, i
                )

                # Separate CPU tensors for caching and GPU tensors for immediate use
                cpu_priors = {head: tensor.cpu() for head, tensor in unweighted_ensemble_prior.items()}
                device_priors = {head: tensor.to(device) for head, tensor in unweighted_ensemble_prior.items()}

                cache[q_id] = cpu_priors
                priors[i] = device_priors

    return [priors[i] for i in valid_indices]

def compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_epi_indexes,
                                      mean_cost, std_cost, joint_loss, head_name):
    n_total = repeated_target.shape[0]
    n_plans = n_total // n_epi_indexes

    pred_flat = epinet_cost_estimates.detach().cpu().numpy().reshape(-1, 1)
    targets_flat = repeated_target.detach().cpu().numpy().reshape(-1, 1)

    pred_unscaled = (pred_flat * std_cost) + mean_cost
    y_scaled = targets_flat[:n_plans].flatten()
    y_true = (y_scaled * std_cost) + mean_cost

    pred_matrix = pred_unscaled.reshape(n_epi_indexes, n_plans)
    pred_matrix_scaled = pred_flat.reshape(n_epi_indexes, n_plans)

    y_pred_mean = pred_matrix.mean(axis=0)
    y_pred_mean_scaled = pred_matrix_scaled.mean(axis=0)
    y_pred_std = pred_matrix.std(axis=0)

    p_values, epinet_distribution_variance = compute_calibration_measures(
        y_true,
        pred_matrix.reshape(n_plans, n_epi_indexes),
    )

    joint_gaussian_nll = joint_loss(torch.tensor(pred_matrix_scaled), torch.tensor(y_scaled))

    mse = np.mean((y_pred_mean - y_true) ** 2)
    mse_scaled = np.mean((y_pred_mean_scaled - y_scaled) ** 2)

    return {
        f"val_epi_mse_{head_name}": mse,
        f"val_epi_mse_{head_name}_scaled": mse_scaled,
        f"val_epi_avg_std_{head_name}": np.mean(y_pred_std),
        f"val_observed_p_values_{head_name}": p_values,
        f"val_distribution_variance_{head_name}": epinet_distribution_variance,
        f"val_joint_gaussian_nll_{head_name}": joint_gaussian_nll,
    }


def validate_cached(val_loader, query_plans_val, targets, epinet_cost_estimation, val_cache,
                    mean_vals, std_vals, train_loss, device, n_val_epi_indexes,
                    sigma, alpha_mlp, alpha_ensemble, precomputed_indexes, precomputed_masks,
                    head_names_to_val = ("plan_cost",)):
    mape = MeanAbsolutePercentageError().to(device)
    joint_loss = GaussianJointLogLoss()
    tracker = MetricsTracker()
    generator = torch.Generator(device=device)

    total_val_queries = len(val_loader.dataset)

    pbar = tqdm(total=total_val_queries, desc="Validating", leave=False, position=1, dynamic_ncols=True)

    n_empty_plans = 0
    n_non_empty_plans = 0
    processed = 0

    for query_batch in val_loader:
        processed += len(query_batch.query)
        valid_indices = [i for i, q_id in enumerate(query_batch.query)
                         if q_id in query_plans_val and len(query_plans_val[q_id]) > 0]
        n_empty_plans += (len(query_batch.query) - len(valid_indices))
        n_non_empty_plans += len(valid_indices)
        # Update pbar with stats to track if expected number of empty plans are produced
        pbar.set_postfix(
            empty_plans=n_empty_plans,
            non_empty_plans=n_non_empty_plans,
            queries=processed
        )

        if not valid_indices:
            pbar.update(1)
            continue

        unweighted_ensemble_priors = fetch_or_compute_priors(
            query_batch, valid_indices, query_plans_val, val_cache, epinet_cost_estimation, precomputed_indexes,
            precomputed_masks, device
        )

        with torch.no_grad():
            embedded = epinet_cost_estimation.embed_query_batched(query_batch.to(device))

            for valid_idx, unweighted_ensemble_prior in zip(valid_indices, unweighted_ensemble_priors):
                q_id = query_batch.query[valid_idx]
                plans_query = query_plans_val[q_id]

                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
                )
                for head_name in head_names_to_val:
                    estimated_head_val = estimated_cost[head_name]
                    prior_for_head = unweighted_ensemble_prior[head_name]

                    val_loss_epinet, repeated_target, epinet_cost_estimates = loss_epinet(
                        prior_for_head,
                        epinet_cost_estimation,
                        train_loss,
                        estimated_head_val,
                        last_feature,
                        plans_query,
                        n_val_epi_indexes,
                        sigma, alpha_mlp, alpha_ensemble,
                        generator,
                        device,
                        head_name=head_name
                    )

                    mean_val = mean_vals[head_name]
                    std_val = std_vals[head_name]

                    val_metrics = compute_validation_metrics_epinet(
                        epinet_cost_estimates, repeated_target, n_val_epi_indexes, mean_val, std_val, joint_loss,
                        head_name = head_name
                    )

                    tracker.update_calibration(
                        val_metrics.pop(f"val_observed_p_values_{head_name}"),
                        val_metrics.pop(f"val_distribution_variance_{head_name}"),
                    )

                    estimated_head_val = estimated_head_val.view(-1)
                    head_target = targets[q_id][head_name].view(-1)

                    original_cost = (estimated_head_val * std_val) + mean_val
                    original_target = (head_target * std_val) + mean_val

                    tracker.update(
                        **val_metrics,
                        **{
                            f"val_loss_{head_name}_scaled": train_loss(estimated_head_val, head_target).item(),
                            f"val_loss_{head_name}_unscaled": train_loss(original_cost, original_target).item(),
                            f"val_mape_{head_name}_scaled": mape(estimated_head_val, head_target).item(),
                            f"val_mape_{head_name}_unscaled": mape(original_cost, original_target).item(),
                            f"val_joint_nll_{head_name}_no_epinet": joint_loss(estimated_head_val.unsqueeze(0),
                                                                               head_target).item(),
                            f"val_loss_{head_name}_epinet": val_loss_epinet.cpu().item()
                        }
                    )

                pbar.update(1)
    pbar.close()
    return tracker


def train_on_batch_cached(query_batch, valid_indices, query_plans, cache,
                          precomputed_indexes, precomputed_masks,
                          epinet_cost_estimation,
                          optimizer, loss,
                          n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble,
                          generator, device):
    acc_loss = torch.tensor(0.0, device=device)

    unweighted_ensemble_priors = fetch_or_compute_priors(
        query_batch, valid_indices, query_plans, cache, epinet_cost_estimation, precomputed_indexes, precomputed_masks, device
    )

    embedded = epinet_cost_estimation.embed_query_batched(query_batch.to(device))

    for valid_idx, unweighted_ensemble_prior in zip(valid_indices, unweighted_ensemble_priors):
        embedded_numpy = embedded[valid_idx].detach().cpu().numpy()
        q_id = query_batch.query[valid_idx]
        plans_query = query_plans[q_id]

        estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
            plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
        )
        estimated_cost = estimated_cost["plan_cost"]

        loss_epinet_val, _, _ = loss_epinet(
            unweighted_ensemble_prior["plan_cost"],
            epinet_cost_estimation,
            loss,
            estimated_cost,
            last_feature,
            plans_query,
            n_epi_indexes_train,
            sigma, alpha_mlp, alpha_ensemble,
            generator, device
        )
        acc_loss += loss_epinet_val

    acc_loss /= len(valid_indices)
    acc_loss.backward()
    optimizer.step()
    return acc_loss.detach().cpu().item()

def loss_epinet(unweighted_ensemble_priors,
                epinet_cost_estimation,
                loss,
                estimated_cost, last_feature,
                plans, n_epi_indexes,
                sigma, alpha_mlp, alpha_ensemble,
                generator,
                device,
                head_name = "plan_cost"):
    last_feature_detached = last_feature.detach()
    n_plans = estimated_cost.shape[0]

    generator.manual_seed(plans[0][2])
    # (n_plans, epi_index_dim)
    c_vectors = torch.randn((n_plans, epinet_cost_estimation.epi_index_dim), generator=generator, device=device)
    c_vectors = torch.nn.functional.normalize(c_vectors, dim=-1)

    # (n_epi_indexes, epi_index_dim)
    epinet_indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(n_epi_indexes)
    # (n_epi_indexes, n_plans)
    ensemble_prior = torch.matmul(epinet_indexes, unweighted_ensemble_priors)
    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Row-major contiguous flattening.
    # Elements are grouped by epistemic index (ei), iterating through all plans for a given index before moving to the next:
    # (plan_1_ei_1, plan_2_ei_1, ..., plan_n_ei_1,  plan_1_ei_2, plan_2_ei_2, ..., plan_n_ei_2,  ..., plan_n_ei_k)
    # where ei = epistemic index sampled during training.
    ensemble_prior_flat = ensemble_prior.view(-1, 1)

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Row-major contiguous block. Grouped by epistemic index: evaluates all plans
    # using the first sampled index, then all plans using the second index, etc.
    mlp_prior = epinet_cost_estimation.compute_mlp_prior_batched(last_feature_detached, epinet_indexes)
    mlp_prior = mlp_prior[head_name]

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: Matches the row-major, index-grouped layout of the priors.
    learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp_batched(last_feature_detached, epinet_indexes)
    learnable_mlp_prior = learnable_mlp_prior[head_name]

    # Shape: (n_epi_indexes * n_plans, 1)
    # Layout: repeat(n_epi_indexes, 1) copies the full block of base plan costs K times.
    # This aligns the base costs identically with the row-major, index-grouped layout of the Epinet.
    estimated_cost_exp = estimated_cost.repeat(n_epi_indexes, 1)
    

    # Shape: (n_epi_indexes * n_plans)
    # Layout: Row-major contiguous flattening.
    # Grouped by epistemic index: calculates the perturbation (z * c) for all plans using
    # the first sampled index, then all plans using the second index, etc.
    
    # TODO: Maybe this c_vectors should be correlated within a query. Not completely but a little
    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    anchor_term_flat = anchor_matrix.view(-1)

    raw_targets = torch.tensor([plan[1] for plan in plans], device=device)

    # Base model loss (purely deterministic, no variance from z)
    loss_base = loss(estimated_cost.squeeze(-1), raw_targets.float())

    # Epinet loss (we detach the base model so it doesn't receive z-variance gradients)
    epinet_estimated_cost_detached = estimated_cost_exp.detach() + (
            learnable_mlp_prior + alpha_mlp * mlp_prior + alpha_ensemble * ensemble_prior_flat
    )

    # (n_epi_indexes * n_plans)
    raw_targets_exp = raw_targets.repeat(n_epi_indexes)

    # TODO: We removed the epinet perturbation terms. To see if now the epinet works
    # perturbed_targets = raw_targets_exp + sigma * anchor_term_flat
    perturbed_targets = raw_targets_exp

    unperturbed_target = raw_targets_exp

    
    loss_epinet_only = loss(epinet_estimated_cost_detached.squeeze(-1), perturbed_targets)
    loss_total = loss_base + loss_epinet_only

    return loss_total, unperturbed_target, epinet_estimated_cost_detached


def train_simulated_epinet_cached(queries_train: QueryCardinalityDataset, query_plans_train,
                                  mean_train, std_train,
                                  queries_val, query_plans_val,
                                  model_builder_fn, model_kwargs, model_state_dict,
                                  epinet_cost_estimation: MultiHeadEpistemicNetwork,
                                  device,
                                  query_batch_size, n_epi_indexes_train,
                                  sigma, alpha_mlp, alpha_ensemble, lr, weight_decay, n_epochs,
                                  n_epi_indexes_val,
                                  writer,
                                  cache_directory,
                                  debug_single_batch = False,
                                  trial: optuna.Trial = None):

    import shutil
    shutil.rmtree(cache_directory, ignore_errors=True)
    os.makedirs(cache_directory, exist_ok=True)
    train_cache = diskcache.Cache(os.path.join(cache_directory, "train_cache"), size_limit=50 * 1024 ** 3)
    val_cache = diskcache.Cache(os.path.join(cache_directory, "val_cache"), size_limit=50 * 1024 ** 3)

    # Actively clear the cache to prevent stale priors between runs, as weights are randomly initialized
    train_cache.clear()
    val_cache.clear()

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    train_summary = TrainSummary([ ("val_epi_mse_plan_cost", "min"), ("val_epi_mse_plan_cost_scaled", "min"),
                                   ("val_epi_avg_std_plan_cost", "min"), ("val_joint_gaussian_nll_plan_cost", "min"),
                                   ("val_loss_plan_cost_scaled", "min"), ("val_loss_plan_cost_unscaled", "min"),
                                   ("val_mape_plan_cost_scaled", "min"), ("val_mape_plan_cost_unscaled", "min"),
                                   ("val_joint_nll_plan_cost_no_epinet", "min"), ("val_loss_plan_cost_epinet", "min"),
                                   ("train_loss", "min"), ("val_calibration_error", "min"), ("val_sharpness", "min")])

    # Predefine a generator so the perturbation vectors are consistent among epochs
    generator = torch.Generator(device=device)
    epinet_cost_estimation.to(device)

    # TODO: Temp removal shuffle = True for debug
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)
    loader_val = DataLoader(queries_val, batch_size=1, shuffle=False)

    val_targets = {query: {"plan_cost": torch.tensor([plan[1] for plan in plans_val], device=device)}
                   for query, plans_val in query_plans_val.items()}

    # Flag to overfit on a single batch to validate that the model works on very small data
    if debug_single_batch:
        print("DEBUG MODE: Fitting on a single training batch subset.")

        # Create single valid batch from training data
        valid_indices = []
        for idx in range(len(queries_train)):
            q_id = queries_train[idx].query
            if q_id in query_plans_train:
                valid_indices.append(idx)
            if len(valid_indices) == query_batch_size:
                break

        if not valid_indices:
            raise ValueError("No valid training queries found to debug.")

        # Take subset with valid data batch
        subset_dataset = queries_train[valid_indices]

        # Overwrite data loaders to only use the one training batch
        loader = DataLoader(subset_dataset, batch_size=query_batch_size, shuffle=False)
        loader_val = DataLoader(subset_dataset, batch_size=1, shuffle=False)

        query_plans_val = query_plans_train
        val_targets = {
            query: {"plan_cost": torch.tensor([plan[1] for plan in plans_train], device=device)}
            for query, plans_train in query_plans_train.items()
        }

    # TODO: Temp removal of freezing of cost estimation model parameters for debugging
    # Freeze cost estimation model parameters for epinet training
    # for param in epinet_cost_estimation.cost_estimation_model.parameters():
    #     param.requires_grad = False
    # epinet_cost_estimation.cost_estimation_model.eval()

    params_all = list(epinet_cost_estimation.parameters())
    params_trainable = [p for p in epinet_cost_estimation.parameters() if p.requires_grad]

    num_trainable = sum(p.numel() for p in params_trainable)
    num_fixed = sum(p.numel() for p in params_all if not p.requires_grad)

    print( f"Parameter counts -> Trainable: {num_trainable:,} | Fixed: {num_fixed:,} "
           f"| Total: {num_trainable + num_fixed:,}")

    optimizer = torch.optim.AdamW(params_trainable, lr=lr, weight_decay=weight_decay)
    scheduler = DualMetricScheduler(optimizer,
                                    patience=3,
                                    threshold=1e-2
                                    )
    previous_lr = scheduler.get_last_lr()

    print_param_count(epinet_cost_estimation, True)
    sleep(1)

    loss = torch.nn.MSELoss(reduction='mean')

    # noinspection PyTypeChecker
    total_train_queries = len(loader.dataset)
    pbar = tqdm(total=total_train_queries, position=0, leave=True, dynamic_ncols=True)


    for epoch in range(1, n_epochs + 1):
        batch_losses = []
        n_empty_plans = 0
        n_non_empty_plans = 0
        processed = 0

        pbar.reset()
        pbar.set_description(f"Epoch {epoch}/{n_epochs} [Train]")

        for query_batch in loader:
            optimizer.zero_grad()

            valid_indices = [i for i, q_id in enumerate(query_batch.query)
                             if q_id in query_plans_train and len(query_plans_train[q_id]) > 0]

            n_empty_plans += (len(query_batch.query) - len(valid_indices))
            n_non_empty_plans += len(valid_indices)
            processed += len(query_batch.query)

            pbar.set_postfix(
                empty_plans=n_empty_plans,
                non_empty_plans=n_non_empty_plans,
                queries=processed
            )

            if not valid_indices:
                continue

            batch_loss = train_on_batch_cached(
                query_batch, valid_indices, query_plans_train, train_cache,
                precomputed_indexes, precomputed_masks,
                epinet_cost_estimation, optimizer, loss,
                n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble, generator, device
            )

            batch_losses.append(batch_loss)
            pbar.update(len(query_batch.query))

        epoch_train_loss = np.mean(batch_losses)

        tracker = validate_cached(
            loader_val, query_plans_val, val_targets, epinet_cost_estimation, val_cache, mean_train, std_train,
            loss, device, n_epi_indexes_val, sigma, alpha_mlp, alpha_ensemble,
            precomputed_indexes, precomputed_masks
        )

        mean_metrics_val = tracker.summarize()
        calibration_error, sharpness = calculate_calibration_metrics(
            tracker.p_values, tracker.distribution_variances, 100,
            os.path.join(writer.get_epoch_dir(epoch), 'calibration_plot.pdf')
        )

        mean_metrics_val.update({
            "train_loss": epoch_train_loss.item(),
            "val_calibration_error": calibration_error.item(),
            "val_sharpness": sharpness.item()
        })

        train_summary.update(mean_metrics_val, epoch)

        scheduler.step(mean_metrics_val['val_loss_plan_cost_unscaled'], mean_metrics_val['train_loss'])

        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file([], best, per_epoch, epinet_cost_estimation, epoch)

        tracker.reset_calibration_stats()
        if scheduler.get_last_lr() != previous_lr:
            print(f"INFO: Lr Updated from {previous_lr} to {scheduler.get_last_lr()}")
            previous_lr = scheduler.get_last_lr()

        if trial:
            trial.report(mean_metrics_val["val_joint_gaussian_nll_plan_cost"], epoch)
            if trial.should_prune():
                train_cache.close()
                val_cache.close()
                raise optuna.TrialPruned()

    pbar.close()
    train_cache.close()
    val_cache.close()
    return train_summary.best_values["val_joint_gaussian_nll_plan_cost"]


def main_simulated_epinet_training(cfg: DictConfig,
                                   train_dataset,
                                   val_dataset,
                                   oracle_model,
                                   epinet_cost_estimation,
                                   model_kwargs,
                                   model_builder_fn,
                                   device,
                                   writer):

    debug_single_batch = OmegaConf.select(cfg, "debug.debug_single_batch", default=False)
    writer.create_experiment_directory()

    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated,
                                     debug_single_batch=debug_single_batch,
                                     query_batch_size=cfg.hyperparameters.query_batch_size)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val,
                                         debug_single_batch=debug_single_batch,
                                         query_batch_size=cfg.hyperparameters.query_batch_size)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train_cost, std_train_cost = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

    mean_train = {"plan_cost": mean_train_cost}
    std_train = {"plan_cost": std_train_cost}

    model_state_dict = epinet_cost_estimation.state_dict()

    # Configure explicit cache directory from configuration or default to a fixed path
    cache_directory = getattr(cfg.dataset, 'prior_cache_dir', os.path.join(os.getcwd(), ".prior_cache"))

    return train_simulated_epinet_cached(
        queries_train=train_dataset,
        query_plans_train=train_plans,
        mean_train=mean_train,
        std_train=std_train,
        queries_val=val_dataset,
        query_plans_val=val_plans,
        model_builder_fn=model_builder_fn,
        model_kwargs=model_kwargs,
        model_state_dict=model_state_dict,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer,
        cache_directory=cache_directory,
        query_batch_size=cfg.hyperparameters.query_batch_size,
        n_epi_indexes_train=cfg.hyperparameters.n_epi_indexes_train,
        n_epi_indexes_val=cfg.hyperparameters.n_epi_indexes_val,
        sigma=cfg.hyperparameters.sigma,
        alpha_mlp=cfg.hyperparameters.alpha_mlp,
        alpha_ensemble=cfg.hyperparameters.alpha_ensemble,
        lr=cfg.hyperparameters.lr,
        weight_decay=cfg.hyperparameters.weight_decay,
        n_epochs=cfg.hyperparameters.n_epochs,
        debug_single_batch=debug_single_batch,
    )


def main_supervised_value_estimation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_dataset, val_dataset = prepare_data(
        cfg.dataset.endpoint_location,
        cfg.dataset.queries_train,
        cfg.dataset.queries_val,
        cfg.dataset.rdf2vec_vector_location,
        cfg.dataset.occurrences_location,
        cfg.dataset.tp_cardinality_location
    )

    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config, model_directory=cfg.models.oracle.dir
    )

    model_kwargs = {
        "full_gnn_config": cfg.models.embedder.config,
        "config_ensemble_prior": cfg.models.epinet.prior_config,
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "mlp_dimension": cfg.hyperparameters.mlp_dimension,
        "model_weights": cfg.models.epinet.model_file,
        "cost_only": True,
    }

    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(cfg.hyperparameters.mlp_dimension, 1),
        }
    }
    heads_config_prior = {
        'plan_cost': {
            'layer': torch.nn.Linear(5, 1),
        }
    }

    epinet_cost_estimation = prepare_epinet_model(**model_kwargs, device=device,
                                                  heads_config=heads_config, heads_config_prior=heads_config_prior)
    experiment_base_dir = cfg.models.output.experiment_base_dir

    writer = ExperimentWriter(experiment_base_dir, "simulated_cost_epinet_training",
                              OmegaConf.to_container(cfg, resolve=True),
                              {k: v for k, v in model_kwargs.items() if
                               k not in ("model_weights", "heads_config", "heads_config_prior")})

    return main_simulated_epinet_training(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        oracle_model=oracle_model,
        epinet_cost_estimation=epinet_cost_estimation,
        model_kwargs=model_kwargs,
        model_builder_fn=prepare_epinet_model,
        device=device,
        writer=writer
    )


@hydra.main(version_base=None,
            config_path="../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="simulated_supervised_cost_estimation_mixed_yago_train_epinet.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    best_epinet_dir = find_best_epoch_directory(cfg.models.epinet.experiment_dir, "val_loss_cost_unscaled")
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_p99_q_error")

    cfg.models.embedder.dir = str(best_embedder_dir)
    cfg.models.oracle.dir = str(best_oracle_dir)
    cfg.models.epinet.dir = str(best_epinet_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    OmegaConf.set_struct(cfg, True)
    return main_supervised_value_estimation(cfg)


if __name__ == "__main__":
    main()