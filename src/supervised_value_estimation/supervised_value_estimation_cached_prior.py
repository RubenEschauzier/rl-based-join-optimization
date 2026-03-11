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
from src.models.epistemic_neural_network import EpistemicNetwork, prepare_epinet_model
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
    """Retrieves priors from cache or computes and stores them if missing."""
    priors = {}
    missing_indices = []

    for i in valid_indices:
        q_id = query_batch.query[i]
        if q_id in cache:
            priors[i] = cache[q_id].to(device)
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
                cache[q_id] = unweighted_ensemble_prior.cpu()
                priors[i] = unweighted_ensemble_prior.to(device)

    return [priors[i] for i in valid_indices]


def compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_epi_indexes,
                                      mean_cost, std_cost, joint_loss):
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
        "val_epi_mse": mse,
        "val_epi_mse_scaled": mse_scaled,
        "val_epi_avg_std": np.mean(y_pred_std),
        "val_observed_p_values": p_values,
        "val_distribution_variance": epinet_distribution_variance,
        "val_joint_gaussian_nll": joint_gaussian_nll,
    }


def validate_cached(val_loader, query_plans_val, epinet_cost_estimation, val_cache,
                    mean_cost, std_cost, train_loss, device, n_val_epi_indexes,
                    sigma, alpha_mlp, alpha_ensemble, precomputed_indexes, precomputed_masks):
    mape = MeanAbsolutePercentageError().to(device)
    joint_loss = GaussianJointLogLoss()
    tracker = MetricsTracker()
    generator = torch.Generator(device=device)

    total_val_queries = len(val_loader.dataset)

    with tqdm(total=total_val_queries, desc="Validating") as pbar:
        for query_batch in val_loader:
            valid_indices = [i for i, q_id in enumerate(query_batch.query) if q_id in query_plans_val]
            if not valid_indices:
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

                    val_loss_epinet, repeated_target, epinet_cost_estimates = loss_epinet(
                        unweighted_ensemble_prior,
                        epinet_cost_estimation,
                        train_loss,
                        estimated_cost,
                        last_feature,
                        plans_query,
                        n_val_epi_indexes,
                        sigma, alpha_mlp, alpha_ensemble,
                        generator,
                        device
                    )

                    val_metrics = compute_validation_metrics_epinet(
                        epinet_cost_estimates, repeated_target, n_val_epi_indexes, mean_cost, std_cost, joint_loss
                    )

                    tracker.update_calibration(
                        val_metrics.pop("val_observed_p_values"),
                        val_metrics.pop("val_distribution_variance")
                    )

                    estimated_cost = estimated_cost.squeeze()
                    target = torch.tensor([plan[1] for plan in plans_query], device=device)

                    original_cost = (estimated_cost * std_cost) + mean_cost
                    original_target = (target * std_cost) + mean_cost

                    tracker.update(
                        **val_metrics,
                        val_loss_cost_scaled=train_loss(estimated_cost, target).item(),
                        val_loss_cost_unscaled=train_loss(original_cost, original_target).item(),
                        val_mape_cost_scaled=mape(estimated_cost, target).item(),
                        val_mape_cost_unscaled=mape(original_cost, original_target).item(),
                        val_joint_nll_no_epinet=joint_loss(estimated_cost.unsqueeze(0), target).item(),
                        val_loss_epinet=val_loss_epinet.cpu().item()
                    )

                    pbar.update(1)

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
        q_id = query_batch.query[valid_idx]
        plans_query = query_plans[q_id]

        estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
            plans_query, embedded[valid_idx], precomputed_indexes, precomputed_masks
        )

        loss_epinet_val, _, _ = loss_epinet(
            unweighted_ensemble_prior,
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
                device):
    last_feature = last_feature.detach()
    n_plans = estimated_cost.shape[0]

    generator.manual_seed(plans[0][2])
    c_vectors = torch.randn((n_plans, epinet_cost_estimation.epi_index_dim), generator=generator, device=device)
    c_vectors = torch.nn.functional.normalize(c_vectors, dim=1)

    epinet_indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(n_epi_indexes)
    ensemble_prior = torch.matmul(epinet_indexes, unweighted_ensemble_priors)

    ensemble_prior_flat = ensemble_prior.view(-1, 1)

    mlp_prior = epinet_cost_estimation.compute_mlp_prior_batched(last_feature, epinet_indexes)
    learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp_batched(last_feature, epinet_indexes)

    estimated_cost_exp = estimated_cost.repeat(n_epi_indexes, 1)
    epinet_estimated_cost = estimated_cost_exp + (
            learnable_mlp_prior + alpha_mlp * mlp_prior + alpha_ensemble * ensemble_prior_flat
    )

    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    anchor_term_flat = anchor_matrix.view(-1)

    raw_targets = torch.tensor([plan[1] for plan in plans], device=device)
    raw_targets_exp = raw_targets.repeat(n_epi_indexes)

    perturbed_targets = raw_targets_exp + sigma * anchor_term_flat
    unperturbed_target = raw_targets_exp

    query_loss = loss(epinet_estimated_cost.squeeze(), perturbed_targets)
    return query_loss, unperturbed_target, epinet_estimated_cost


def train_simulated_epinet_cached(queries_train: QueryCardinalityDataset, query_plans_train,
                                  mean_train, std_train,
                                  queries_val, query_plans_val,
                                  model_builder_fn, model_kwargs, model_state_dict,
                                  epinet_cost_estimation: EpistemicNetwork,
                                  device,
                                  query_batch_size, n_epi_indexes_train,
                                  sigma, alpha_mlp, alpha_ensemble, lr, weight_decay, n_epochs,
                                  n_epi_indexes_val,
                                  writer,
                                  cache_directory,
                                  trial: optuna.Trial = None):
    os.makedirs(cache_directory, exist_ok=True)
    train_cache = diskcache.Cache(os.path.join(cache_directory, "train_cache"), size_limit=50 * 1024 ** 3)
    val_cache = diskcache.Cache(os.path.join(cache_directory, "val_cache"), size_limit=50 * 1024 ** 3)

    # Actively clear the cache to prevent stale priors between runs, as weights are randomly initialized
    train_cache.clear()
    val_cache.clear()

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    train_summary = TrainSummary([ ("val_epi_mse", "min"), ("val_epi_mse_scaled", "min"),
                                   ("val_epi_avg_std", "min"), ("val_joint_gaussian_nll", "min"),
                                   ("val_loss_cost_scaled", "min"), ("val_loss_cost_unscaled", "min"),
                                   ("val_mape_cost_scaled", "min"), ("val_mape_cost_unscaled", "min"),
                                   ("val_joint_nll_no_epinet", "min"), ("val_loss_epinet", "min"),
                                   ("train_loss", "min"), ("val_calibration_error", "min"), ("val_sharpness", "min")])

    # Predefine a generator so the perturbation vectors are consistent among epochs
    generator = torch.Generator(device=device)
    epinet_cost_estimation.to(device)

    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)
    loader_val = DataLoader(queries_val, batch_size=1, shuffle=False)

    # Freeze cost estimation model parameters for epinet training
    for param in epinet_cost_estimation.cost_estimation_model.parameters():
        param.requires_grad = False
    epinet_cost_estimation.cost_estimation_model.eval()

    params = list(epinet_cost_estimation.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=1e-2)
    previous_lr = scheduler.get_last_lr()

    print_param_count(epinet_cost_estimation, True)
    sleep(1)

    loss = torch.nn.MSELoss(reduction='mean')

    # noinspection PyTypeChecker
    total_train_queries = len(loader.dataset)

    for epoch in range(1, n_epochs + 1):
        batch_losses = []

        with tqdm(total=total_train_queries, desc=f"Epoch {epoch}/{n_epochs} [Train]") as pbar:
            for query_batch in loader:
                optimizer.zero_grad()

                valid_indices = [i for i, q_id in enumerate(query_batch.query) if q_id in query_plans_train]
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
            loader_val, query_plans_val, epinet_cost_estimation, val_cache, mean_train, std_train,
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

        scheduler.step(mean_metrics_val['val_loss_cost_unscaled'])

        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file([], best, per_epoch, epinet_cost_estimation, epoch)

        tracker.reset_calibration_stats()
        if scheduler.get_last_lr() != previous_lr:
            print(f"INFO: Lr Updated from {previous_lr} to {scheduler.get_last_lr()}")
            previous_lr = scheduler.get_last_lr()

        if trial:
            trial.report(mean_metrics_val["val_joint_gaussian_nll"], epoch)
            if trial.should_prune():
                train_cache.close()
                val_cache.close()
                raise optuna.TrialPruned()

    train_cache.close()
    val_cache.close()
    return train_summary.best_values["val_joint_gaussian_nll"]


def main_simulated_epinet_training(cfg: DictConfig,
                                   train_dataset,
                                   val_dataset,
                                   oracle_model,
                                   epinet_cost_estimation,
                                   model_kwargs,
                                   model_builder_fn,
                                   device,
                                   writer):
    writer.create_experiment_directory()

    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

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
        n_epochs=cfg.hyperparameters.n_epochs
    )


def main_supervised_value_estimation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    experiment_base_dir = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training"

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
            config_path="../../experiments/experiment_configs/epinet_cost_estimation",
            config_name="simulated_supervised_cost_estimation_train_epinet.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    best_epinet_dir = find_best_epoch_directory(cfg.models.epinet.experiment_dir, "val_loss_cost_unscaled")
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_q_error")

    cfg.models.embedder.dir = str(best_embedder_dir)
    cfg.models.oracle.dir = str(best_oracle_dir)
    cfg.models.epinet.dir = str(best_epinet_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    OmegaConf.set_struct(cfg, True)
    return main_supervised_value_estimation(cfg)


if __name__ == "__main__":
    main()