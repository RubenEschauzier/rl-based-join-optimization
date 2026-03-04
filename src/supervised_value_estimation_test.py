from collections import defaultdict
import os
import sys

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader

from src.models.epistemic_neural_network import EpistemicNetwork, prepare_epinet_model
from src.utils.epinet_utils.calibration_plot import compute_calibration_measures, calculate_calibration_metrics
from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.training_tracking import TrainSummary, ExperimentWriter

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.models.model_instantiator import ModelFactory
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset, prepare_data
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask
import torch
import torch.multiprocessing as mp


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




def epinet_data_prep_worker(input_queue, output_queue, model_builder_fn, model_kwargs, state_dict,
                            precomputed_indexes, precomputed_masks):
    torch.set_num_threads(1)
    # Initialize heads config here as initializing it on the mean thread with cuda gives problems


    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(model_kwargs["mlp_dimension"], 1),
        }
    }
    heads_config_prior = {
        'plan_cost': {
            'layer': torch.nn.Linear(5, 1),
        }
    }

    # Initialize CPU model and freeze parameters
    model = model_builder_fn(**model_kwargs, device=torch.device('cpu'),
                             heads_config=heads_config, heads_config_prior=heads_config_prior)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    while True:
        payload = input_queue.get()
        if payload is None:
            break

        embedded, plans, query = payload

        # Prepare tree representations
        prep_trees, prep_idx, prep_masks = model.prepare_cost_estimation_inputs(
            plans, embedded, precomputed_indexes, precomputed_masks,
            target_device=torch.device('cpu')
        )

        # Embed the queries and calculate the priors on cpu
        embedded_prior = model.embed_query_batched_prior(query)
        unweighted_ensemble_priors = model.compute_ensemble_prior(
            plans, embedded_prior, precomputed_indexes, precomputed_masks, 0
        )

        output_queue.put({
            "prep_trees": prep_trees,
            "prep_idx": prep_idx,
            "prep_masks": prep_masks,
            "plans": plans,
            "unweighted_priors": unweighted_ensemble_priors,
        })


def validate_worker_based(val_loader, query_plans_val, epinet_cost_estimation,
                          mean_cost, std_cost, train_loss, device, n_val_epi_indexes,
                          sigma, alpha_mlp, alpha_ensemble, generator,
                          input_queue, output_queue):
    mape = MeanAbsolutePercentageError().to(device)
    joint_loss = GaussianJointLogLoss()
    tracker = MetricsTracker()

    # Feed validation queries to the workers
    num_val_queries = fill_input_query_queue(val_loader, epinet_cost_estimation, query_plans_val, input_queue)

    completed = 0
    with tqdm(total=num_val_queries, desc="Validating") as pbar:
        while completed < num_val_queries:
            batch = output_queue.get()

            with torch.no_grad():
                prep_trees = batch["prep_trees"].to(device, non_blocking=True)
                prep_idx = batch["prep_idx"].to(device, non_blocking=True)
                prep_masks = batch["prep_masks"].to(device, non_blocking=True)
                unweighted_priors = batch["unweighted_priors"].to(device, non_blocking=True)
                plans = batch["plans"]

                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_from_prepared(
                    prep_trees, prep_idx, prep_masks
                )

                # Execute Epinet loss calculation
                _, repeated_target, epinet_cost_estimates = loss_epinet_prepared_priors(
                    unweighted_priors, epinet_cost_estimation, train_loss, estimated_cost, last_feature,
                    plans, n_val_epi_indexes, sigma, alpha_mlp, alpha_ensemble, generator, device
                )

                # Calculate metrics
                val_metrics = compute_validation_metrics_epinet(
                    epinet_cost_estimates, repeated_target, n_val_epi_indexes, mean_cost, std_cost, joint_loss
                )

                tracker.update_calibration(
                    val_metrics.pop("val_observed_p_values"),
                    val_metrics.pop("val_distribution_variance")
                )

                estimated_cost = estimated_cost.squeeze()
                target = torch.tensor([plan[1] for plan in plans], device=device)

                original_cost = (estimated_cost * std_cost) + mean_cost
                original_target = (target * std_cost) + mean_cost

                # Accumulate scalar metrics
                tracker.update(
                    **val_metrics,
                    val_loss_cost_scaled=train_loss(estimated_cost, target).item(),
                    val_loss_cost_unscaled=train_loss(original_cost, original_target).item(),
                    val_mape_cost_scaled=mape(estimated_cost, target).item(),
                    val_mape_cost_unscaled=mape(original_cost, original_target).item(),
                    val_joint_nll_no_epinet=joint_loss(estimated_cost.unsqueeze(0), target).item()
                )

            completed += 1
            pbar.update(1)

    return tracker

def compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_epi_indexes,
                                      mean_cost, std_cost, joint_loss):
    """
    Computes validation metrics for epinet_cost_estimates. Expects all predictions of a single query to be passed
    :param epinet_cost_estimates:
    :param repeated_target:
    :param n_epi_indexes:
    :param mean_cost:
    :param std_cost:
    :param joint_loss: Class to compute the gaussian nll
    :return:
    """
    n_total = repeated_target.shape[0]
    n_plans = n_total // n_epi_indexes

    # 1. Move to CPU and Numpy
    # Reshape to (-1, 1) because scaler expects 2D array [samples, features]
    pred_flat = epinet_cost_estimates.detach().cpu().numpy().reshape(-1, 1)
    targets_flat = repeated_target.detach().cpu().numpy().reshape(-1, 1)

    pred_unscaled = (pred_flat * std_cost) + mean_cost
    y_scaled = targets_flat[:n_plans].flatten()
    y_true = (y_scaled * std_cost) + mean_cost

    # Shape: (n_epi_indexes, n_plans)
    # Column j contains all 'n_epi_indexes' predictions for Plan j
    pred_matrix = pred_unscaled.reshape(n_epi_indexes, n_plans)
    pred_matrix_scaled = pred_flat.reshape(n_epi_indexes, n_plans)

    y_pred_mean = pred_matrix.mean(axis=0)
    y_pred_mean_scaled = pred_matrix_scaled.mean(axis=0)
    y_pred_std = pred_matrix.std(axis=0)

    p_values, epinet_distribution_variance = compute_calibration_measures(
        y_true,
        pred_matrix.reshape(n_plans, n_epi_indexes),
    )

    # Calculates the joint loss over all plans
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


def summarize_epistemic_metrics(metrics_dict, train_loss, calibration_error, sharpness):
    keys = list(metrics_dict.values())[0].keys()

    summary = {
        k: np.mean([q_metrics[k] for q_metrics in metrics_dict.values()]).item()
        for k in keys
    }
    print(f"Cost estimation train loss: {train_loss:.4f}\n")
    print(
        "Cost estimation metrics\n"
        f"  Validation Loss Scaled : {summary['val_loss_cost_scaled']:.4f}\n"
        f"  Validation MAPE Scaled : {summary['val_mape_cost_scaled']:.4f}\n"
        f"  Validation Loss True : {summary['val_loss_cost_unscaled']:.4f}\n"
        f"  Validation MAPE True : {summary['val_mape_cost_unscaled']:.4f}\n"
        f"  Validation joint NLL base: {summary['val_joint_nll_no_epinet']:.4f}"

    )
    if "val_epi_mse" in summary:
        print(
            "[Epistemic metrics — mean over queries]\n"
            f"  MSE                    : {summary['val_epi_mse']:.4f}\n"
            f"  Scaled MSE             : {summary['val_epi_mse_scaled']:.4f}\n"
            f"  Avg Std                : {summary['val_epi_avg_std']:.4f}\n"
            f"  Calibration Error      : {calibration_error:.3f}\n",
            f"  Sharpness              : {sharpness:.3f}\n",
            f"  Join Gaussian NLL      : {summary['val_joint_gaussian_nll']:.4f}"
        )
    return summary


def train_on_batch(queries_batch,
                   epinet_cost_estimation,
                   optimizer, loss,
                   n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble,
                   generator,
                   device):
    acc_loss = torch.tensor(0.0, device=device)
    for prepared_query in queries_batch:
        estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_from_prepared(
            prepared_query["prep_trees"], prepared_query["prep_idx"], prepared_query["prep_masks"]
        )
        loss_epinet, unperturbed_target, epinet_estimated_cost = (
            loss_epinet_prepared_priors(prepared_query["unweighted_priors"].to(device),
                                        epinet_cost_estimation,
                                        loss,
                                        estimated_cost,
                                        last_feature,
                                        prepared_query["plans"],
                                        n_epi_indexes_train,
                                        sigma, alpha_mlp, alpha_ensemble,
                                        generator,
                                        device))
        acc_loss += loss_epinet
    acc_loss.backward()
    optimizer.step()
    return acc_loss.detach().cpu().item()


def fill_input_query_queue(loader, epinet_cost_estimation, plans_dict, input_queue):
    """Embeds queries on GPU and pushes them to the CPU workers."""
    valid_query_count = 0
    for i, query in enumerate(loader):
        q_id = query.query[0]
        if q_id not in plans_dict:
            continue

        with torch.no_grad():
            embedded = epinet_cost_estimation.embed_query_batched(query)
        input_queue.put((embedded[0].cpu(), plans_dict[q_id], query))
        valid_query_count += 1
    return valid_query_count

def train_simulated_epinet_worker_based(queries_train, query_plans_train,
                                        mean_train, std_train,
                                        queries_val, query_plans_val,
                                        model_builder_fn, model_kwargs, model_state_dict,
                                        epinet_cost_estimation: EpistemicNetwork,
                                        device,
                                        query_batch_size, n_epi_indexes_train,
                                        # Hyperparameters
                                        sigma, alpha_mlp, alpha_ensemble, lr, weight_decay, n_epochs,
                                        n_epi_indexes_val,
                                        writer,
                                        num_workers,
                                        trial: optuna.Trial = None
                                        ):
    #TODO: Check if correctly set entire cost model frozen

    train_summary = TrainSummary([("train_loss", "min"), ("val_loss_cost_scaled", "min"),
                                  ("val_loss_cost_unscaled", "min"), ("val_mape_cost_scaled", "min"),
                                  ("val_mape_cost_unscaled", "min"), ("val_epi_mse", "min"),
                                  ("val_epi_mse_scaled", "min"),
                                  ("val_epi_avg_std", "min"),
                                  ("val_calibration_error", "min"), ("val_sharpness", "min"),
                                  ("val_joint_gaussian_nll", "min"), ("val_joint_nll_no_epinet", "min")])
    epinet_cost_estimation.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)
    loader = DataLoader(queries_train, batch_size=1, shuffle=True)
    loader_val = DataLoader(queries_val, batch_size=1, shuffle=False)

    # Freeze base cost model when training the epistemic network
    for param in epinet_cost_estimation.cost_estimation_model.parameters():
        param.requires_grad = False
    epinet_cost_estimation.cost_estimation_model.eval()

    params = list(epinet_cost_estimation.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2)

    previous_lr = scheduler.get_last_lr()
    print_param_count(epinet_cost_estimation, True)

    loss = torch.nn.MSELoss(reduction='mean')
    generator = torch.Generator(device=device)

    # create queue passing computed plan representations to GPU for forward pass
    query_queue = mp.Queue()
    # create queues passing prepared queries to the main process
    prepared_queries_queue = mp.Queue()

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=epinet_data_prep_worker,
            args=(query_queue, prepared_queries_queue, model_builder_fn, model_kwargs, model_state_dict,
                  precomputed_indexes, precomputed_masks)
        )
        p.start()
        workers.append(p)


    for epoch in range(1, n_epochs + 1):
        # Fill the queue again, so the CPU can process the queries while validation runs
        print(f"Filling input queries... (size queue {query_queue.qsize()})")
        n_queries = fill_input_query_queue(loader, epinet_cost_estimation, query_plans_train, query_queue)
        print(f"Filled queue with {n_queries} queries")
        batch_losses = []
        queries_batch = []
        retrieved_queries = 0

        pbar = tqdm(total=n_queries)
        while retrieved_queries < n_queries:
            # Get from queue and form a batch
            queries_batch.append(prepared_queries_queue.get())
            retrieved_queries += 1
            if len(queries_batch) >= query_batch_size or retrieved_queries == n_queries:
                batch_loss = train_on_batch(queries_batch,
                                            epinet_cost_estimation,
                                            optimizer, loss,
                                            n_epi_indexes_train, sigma, alpha_mlp, alpha_ensemble,
                                            generator,
                                            device)
                batch_losses.append(batch_loss)
                queries_batch = []
            pbar.update(1)

        # Perform validation run
        epoch_train_loss = np.mean(batch_losses)
        tracker = validate_worker_based(
            loader_val, query_plans_val, epinet_cost_estimation, mean_train, std_train,
            loss, device, n_epi_indexes_val, sigma, alpha_mlp, alpha_ensemble, generator,
            query_queue, prepared_queries_queue
        )

        # Extract summaries and calibration metrics
        mean_metrics_val = tracker.summarize()
        calibration_error, sharpness = calculate_calibration_metrics(
            tracker.p_values, tracker.distribution_variances, 100,
            os.path.join(writer.get_epoch_dir(epoch), 'calibration_plot.pdf')
        )

        mean_metrics_val.update({
            "train_loss": epoch_train_loss,
            "val_calibration_error": calibration_error.item(),
            "val_sharpness": sharpness.item()
        })

        train_summary.update(mean_metrics_val, epoch)
        scheduler.step(mean_metrics_val['val_loss_cost_unscaled'])

        optimizer.zero_grad()

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        # Pruning for optuna hyperparameter search based on nll
        if trial:
            trial.report(mean_metrics_val["val_joint_gaussian_nll"], epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return train_summary.best_values["val_joint_gaussian_nll"]


def loss_epinet_prepared_priors(unweighted_ensemble_priors,
                                epinet_cost_estimation,
                                loss,
                                estimated_cost, last_feature,
                                plans, n_epi_indexes,
                                # Hyperparameters
                                sigma, alpha_mlp, alpha_ensemble,
                                generator, device
                                ):
    # Apply stop gradient operator to last feature to serve as input to epinet
    last_feature = last_feature.detach()
    n_plans = estimated_cost.shape[0]

    generator.manual_seed(plans[0][2])
    c_vectors = torch.randn((n_plans, epinet_cost_estimation.epi_index_dim), generator=generator, device=device)
    c_vectors = torch.nn.functional.normalize(c_vectors, dim=1)

    # Sample all K epistemic indexes at once in a single tensor [K, epi_index_dim]
    epinet_indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(n_epi_indexes)
    ensemble_prior = torch.matmul(epinet_indexes, unweighted_ensemble_priors)

    # Flatten to match the target shape: [n_epi_indexes * n_plans, 1]
    ensemble_prior_flat = ensemble_prior.view(-1, 1)

    # Both return shape: [n_epi_indexes * n_plans, 1]
    mlp_prior = epinet_cost_estimation.compute_mlp_prior_batched(last_feature, epinet_indexes)
    learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp_batched(last_feature, epinet_indexes)

    # Repeat base network estimates: [n_plans, 1] -> [n_epi_indexes * n_plans, 1]
    estimated_cost_exp = estimated_cost.repeat(n_epi_indexes, 1)

    epinet_estimated_cost = estimated_cost_exp + (
            learnable_mlp_prior + alpha_mlp * mlp_prior + alpha_ensemble * ensemble_prior_flat
    )

    # anchor_matrix: [n_epi_indexes, epi_index_dim] @ [epi_index_dim, n_plans] -> [n_epi_indexes, n_plans]
    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    anchor_term_flat = anchor_matrix.view(-1)  # [n_epi_indexes * n_plans]

    raw_targets = torch.tensor([plan[1] for plan in plans], device=device)
    # Repeat raw targets: [n_plans] -> [n_epi_indexes * n_plans]
    raw_targets_exp = raw_targets.repeat(n_epi_indexes)

    # Final perturbed targets
    perturbed_targets = raw_targets_exp + sigma * anchor_term_flat
    unperturbed_target = raw_targets_exp

    query_loss = loss(epinet_estimated_cost.squeeze(), perturbed_targets)
    return query_loss, unperturbed_target, epinet_estimated_cost


def main_simulated_epinet_training(cfg: DictConfig,
                                   train_dataset,
                                   val_dataset,
                                   oracle_model,
                                   epinet_cost_estimation,
                                   model_kwargs,
                                   model_builder_fn,
                                   device,
                                   writer):
    mp.set_start_method('spawn', force=True)
    writer.create_experiment_directory()

    # Prepare datasets
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

    model_state_dict = epinet_cost_estimation.state_dict()

    # Execute training
    train_simulated_epinet_worker_based(
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
        num_workers=cfg.hyperparameters.num_workers,
        # Hyperparameters
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

    # Prepare large (40 million parameters) oracle model to estimate cardinality of join plans (performs slightly better)
    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config, model_directory=cfg.models.oracle.dir
    )


    model_kwargs = {
        "full_gnn_config": cfg.models.embedder.config,
        "config_ensemble_prior": cfg.models.epinet.prior_config,
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "mlp_dimension": cfg.hyperparameters.mlp_dimension,
        "model_weights": cfg.models.epinet.model_file,
        # For epinet training we only want to load from a trained cost model, rest is useless
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
                              { k: v for k,v in model_kwargs.items() if
                                 k != "model_weights" and k != "heads_config" and k != "heads_config_prior" })

    main_simulated_epinet_training(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        oracle_model=oracle_model,
        epinet_cost_estimation=epinet_cost_estimation,
        model_kwargs = model_kwargs,
        model_builder_fn = prepare_epinet_model,
        device=device,
        writer=writer
    )


@hydra.main(version_base=None,
            config_path="../experiments/experiment_configs/epinet_cost_estimation",
            config_name="supervised_simulated_cost_estimation_train_epinet.yaml")
def main(cfg: DictConfig):
    # Temporarily unlock the config to allow dynamic updates
    OmegaConf.set_struct(cfg, False)

    # Locate the best trained cost model dynamically
    best_epinet_dir = find_best_epoch_directory(cfg.models.epinet.experiment_dir, "val_loss_cost_unscaled")
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_q_error")

    # Inject the resolved path directly into the config state
    cfg.models.embedder.dir = str(best_embedder_dir)
    cfg.models.oracle.dir = str(best_oracle_dir)
    cfg.models.epinet.dir = str(best_epinet_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    # Relock the config to prevent accidental downstream modifications
    OmegaConf.set_struct(cfg, True)

    # Pass the unified config to the main setup function
    main_supervised_value_estimation(cfg)


if __name__ == "__main__":
    main()
