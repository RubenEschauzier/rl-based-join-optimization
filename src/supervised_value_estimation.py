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

from src.models.epistemic_neural_network import EpistemicNetwork
from src.models.query_plan_prediction_model import PlanCostEstimatorFull, QueryPlansPredictionModel
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


def prepare_cardinality_estimator(model_config, model_directory=None):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    if model_directory:
        load_weights_from_pretraining(gine_conv_model, model_directory,
                                      "embedding_model.pt",
                                      ["head_cardinality.pt"],
                                      float_weights=True)
    return gine_conv_model


def validate(queries_val, query_plans_val,
             precomputed_indexes, precomputed_masks,
             mean_cost, std_cost,
             train_loss,
             epinet_cost_estimation,
             device,
             validate_epi_network, n_val_epi_indexes,
             sigma, alpha_mlp, alpha_ensemble,
             generator,
             image_save_loc):
    mape = MeanAbsolutePercentageError()
    joint_loss = GaussianJointLogLoss()
    mape.to(device)

    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)

    query_to_val_metrics = {}
    validation_p_values = []
    validation_predicted_distribution_variances = []
    for queries in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            embedded = epinet_cost_estimation.embed_query_batched(queries)

            embedded_prior = None
            if validate_epi_network:
                embedded_prior = epinet_cost_estimation.embed_query_batched_prior(queries)

            plans = query_plans_val[queries.query[0]]

            estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                plans, embedded[0], precomputed_indexes, precomputed_masks
            )

            query_metrics = {}

            if validate_epi_network:
                loss_epinet, repeated_target, epinet_cost_estimates = calculate_loss_epinet(
                    epinet_cost_estimation=epinet_cost_estimation,
                    loss=train_loss,
                    estimated_cost=estimated_cost, last_feature=last_feature,
                    embedded_prior=embedded_prior,
                    plans=plans, precomputed_indexes=precomputed_indexes, precomputed_masks=precomputed_masks, i=0,
                    n_epi_indexes=n_val_epi_indexes, device=device,
                    sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                    generator=generator,
                )
                validation_metrics = compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_val_epi_indexes,
                                                  mean_cost, std_cost, joint_loss)
                validation_p_values.extend(validation_metrics["val_observed_p_values"])
                validation_predicted_distribution_variances.extend(validation_metrics["val_distribution_variance"])
                validation_metrics.pop("val_observed_p_values")
                validation_metrics.pop("val_distribution_variance")

                query_metrics.update(validation_metrics)

            estimated_cost = estimated_cost.squeeze()
            original_cost = (estimated_cost * std_cost) + mean_cost

            target = torch.tensor([plan[1] for plan in plans], device=device)
            original_target = (target * std_cost) + mean_cost

            mape_val_scaled = mape(estimated_cost, target)
            mape_val_unscaled = mape(original_cost, original_target)

            val_loss_scaled = train_loss(estimated_cost, target)
            val_loss_unscaled = train_loss(original_cost, original_target)

            joint_nll_no_epinet = joint_loss(estimated_cost.unsqueeze(0), target)

            query_metrics["val_loss_cost_scaled"] = val_loss_scaled.cpu().item()
            query_metrics["val_loss_cost_unscaled"] = val_loss_unscaled.cpu().item()

            query_metrics["val_mape_cost_scaled"] = mape_val_scaled.cpu().item()
            query_metrics["val_mape_cost_unscaled"] = mape_val_unscaled.cpu().item()
            query_metrics["val_joint_nll_no_epinet"] = joint_nll_no_epinet.cpu().item()

            query_to_val_metrics[queries.query[0]] = query_metrics
    calibration_error, sharpness = calculate_calibration_metrics(
        validation_p_values,
        validation_predicted_distribution_variances,
        100,
        image_save_loc
    )
    return query_to_val_metrics, calibration_error, sharpness


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

def validation_step(epoch_train_loss, epoch,
                    train_summary, writer,
                    queries_val, query_plans_val,
                    precomputed_indexes, precomputed_masks,
                    mean_train, std_train,
                    loss,
                    epinet_cost_estimation,
                    device,
                    train_epi_network, n_epi_indexes_val,
                    sigma, alpha_mlp, alpha_ensemble,
                    generator):
    image_save_loc = os.path.join(writer.get_epoch_dir(epoch), 'calibration_plot.pdf')
    query_to_val_cost, calibration_error, sharpness = validate(queries_val, query_plans_val,
                                                               precomputed_indexes, precomputed_masks,
                                                               mean_train, std_train,
                                                               loss,
                                                               epinet_cost_estimation,
                                                               device,
                                                               train_epi_network, n_epi_indexes_val,
                                                               sigma=sigma, alpha_mlp=alpha_mlp,
                                                               alpha_ensemble=alpha_ensemble,
                                                               generator=generator,
                                                               image_save_loc=image_save_loc)

    mean_metrics_val = summarize_epistemic_metrics(query_to_val_cost, epoch_train_loss.item(),
                                                   calibration_error, sharpness)
    mean_metrics_val["train_loss"] = epoch_train_loss.item()
    mean_metrics_val["val_calibration_error"] = calibration_error.item()
    mean_metrics_val["val_sharpness"] = sharpness.item()
    train_summary.update(mean_metrics_val, epoch)

    best, per_epoch = train_summary.summary()
    writer.write_epoch_to_file([], best, per_epoch, epinet_cost_estimation, epoch)
    if train_epi_network:
        return mean_metrics_val['val_loss_cost_scaled'], mean_metrics_val['val_joint_gaussian_nll']
    else:
        return mean_metrics_val['val_loss_cost_scaled'], None

def train_simulated_cost_model(queries_train, query_plans_train,
                               mean_train, std_train,
                               queries_val, query_plans_val,
                               epinet_cost_estimation: EpistemicNetwork,
                               device,
                               query_batch_size, n_epi_indexes_train,
                               # Hyperparameters
                               sigma, alpha_mlp, alpha_ensemble, lr, weight_decay, n_epochs,
                               n_epi_indexes_val,
                               train_epi_network: bool,
                               writer,
                               first_validate = False,
                               trial: optuna.Trial = None):
    train_summary = TrainSummary([("train_loss", "min"), ("val_loss_cost_scaled", "min"),
                                  ("val_loss_cost_unscaled", "min"),  ("val_mape_cost_scaled", "min"),
                                  ("val_mape_cost_unscaled", "min"),  ("val_epi_mse", "min"),
                                  ("val_epi_mse_scaled", "min"),
                                  ("val_epi_avg_std", "min"),
                                  ("val_calibration_error", "min"), ("val_sharpness", "min"),
                                  ("val_joint_gaussian_nll", "min"), ("val_joint_nll_no_epinet", "min")])
    epinet_cost_estimation.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)

    # Freeze base cost model when training the epistemic network
    if train_epi_network:
        for param in epinet_cost_estimation.cost_estimation_model.parameters():
            param.requires_grad = False
        epinet_cost_estimation.cost_estimation_model.eval()

    params = list(epinet_cost_estimation.parameters())
    params_cost_estimate = list(epinet_cost_estimation.cost_estimation_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2
                                  )
    previous_lr = scheduler.get_last_lr()

    total_params_cost_estimation = 0
    for param in params_cost_estimate:
        total_params_cost_estimation += param.numel()
    print(f"Cost estimation model has {total_params_cost_estimation} parameters")

    if train_epi_network:
        total_params = 0
        for param in epinet_cost_estimation.parameters():
            total_params += param.numel()
        print(f"Epinet model has {total_params - total_params_cost_estimation} parameters")

    loss = torch.nn.MSELoss(reduction='mean')
    generator = torch.Generator(device=device)

    # First validation run to see performance of randomly initialized model
    if first_validate:
        validation_step(np.array(-1), 0, train_summary, writer,
                        queries_val, query_plans_val,
                        precomputed_indexes, precomputed_masks,
                        mean_train, std_train,
                        loss,
                        epinet_cost_estimation,
                        device,
                        train_epi_network, n_epi_indexes_val,
                        sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                        generator=generator)

    for epoch in range(1, n_epochs+1):
        query_loss_epoch = []
        for k, queries in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()

            embedded = epinet_cost_estimation.embed_query_batched(queries)

            embedded_prior = None
            if train_epi_network:
                with torch.no_grad():
                    embedded_prior = epinet_cost_estimation.embed_query_batched_prior(queries)

            total_loss_tensor = torch.tensor(0.0, device=device)
            for i in range(len(queries.query)):
                if queries.query[i] not in query_plans_train:
                    continue
                plans = query_plans_train[queries.query[i]]
                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans, embedded[i], precomputed_indexes, precomputed_masks
                )

                if train_epi_network:
                    loss_epinet, _, _ = calculate_loss_epinet(
                        epinet_cost_estimation=epinet_cost_estimation,
                        loss = loss,
                        estimated_cost=estimated_cost, last_feature=last_feature,
                        embedded_prior=embedded_prior,
                        plans=plans, precomputed_indexes=precomputed_indexes, precomputed_masks=precomputed_masks, i=i,
                        n_epi_indexes=n_epi_indexes_train, device=device,
                        sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                        generator=generator
                    )
                    total_loss_tensor += loss_epinet
                else:
                    target = torch.tensor([plan[1] for plan in plans], device=device).squeeze()
                    total_loss_tensor += loss(estimated_cost.squeeze(), target)

            total_loss_tensor.backward()
            optimizer.step()

            query_loss_epoch.append(total_loss_tensor.detach().cpu().item() / query_batch_size)


        epoch_train_loss = np.mean(query_loss_epoch)
        val_loss, joint_nll = validation_step(epoch_train_loss, epoch, train_summary, writer,
                        queries_val, query_plans_val,
                        precomputed_indexes, precomputed_masks,
                        mean_train, std_train,
                        loss,
                        epinet_cost_estimation,
                        device,
                        train_epi_network, n_epi_indexes_val,
                        sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                        generator=generator)

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(val_loss)

        # Pruning for optuna hyperparameter search based on nll
        if trial:
            trial.report(joint_nll, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return train_summary.best_values["val_joint_gaussian_nll"]

def calculate_loss_epinet(epinet_cost_estimation,
                          loss,
                          estimated_cost, last_feature, embedded_prior,
                          plans, precomputed_indexes, precomputed_masks, i, n_epi_indexes,
                          # Hyperparameters
                          sigma, alpha_mlp, alpha_ensemble,
                          generator, device):
    # Apply stop gradient operator to last feature to serve as input to epinet
    last_feature = last_feature.detach()
    n_plans = estimated_cost.shape[0]

    # The ensemble prior with GNNs only depends on input, not the sampled epinet index,
    # so we precompute it.
    unweighted_ensemble_priors = epinet_cost_estimation.compute_ensemble_prior(
        plans, embedded_prior, precomputed_indexes, precomputed_masks, i
    )

    # Sample uniformly from the unit sphere, use unique index of first plan to seed the generator to ensure
    # the same datapoint gets the same perturbation in each epoch
    generator.manual_seed(plans[0][2])
    c_vectors = torch.randn((n_plans, epinet_cost_estimation.epi_index_dim ), generator=generator, device=device)
    c_vectors = torch.nn.functional.normalize(c_vectors, dim=1)

    # Sample all K epistemic indexes at once in a single tensor [K, epi_index_dim]
    epinet_indexes = epinet_cost_estimation.sample_epistemic_indexes_batched(n_epi_indexes)
    # epinet_indexes = torch.randn((n_epi_indexes, epinet_cost_estimation.epi_index_dim), device=device)

    # [n_epi_indexes, epi_index_dim] @ [epi_index_dim, n_plans] -> [n_epi_indexes, n_plans]
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


def main_simulated_training(cfg: DictConfig,
                            train_dataset,
                            val_dataset,
                            oracle_model,
                            epinet_cost_estimation,
                            device,
                            writer):
    writer.create_experiment_directory()

    # Prepare datasets
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

    # Execute training
    train_simulated_cost_model(
        queries_train=train_dataset,
        query_plans_train=train_plans,
        mean_train=mean_train,
        std_train=std_train,
        queries_val=val_dataset,
        query_plans_val=val_plans,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer,

        # Hyperparameters
        query_batch_size=cfg.hyperparameters.query_batch_size,
        n_epi_indexes_train=cfg.hyperparameters.n_epi_indexes_train,
        n_epi_indexes_val=cfg.hyperparameters.n_epi_indexes_val,
        train_epi_network=cfg.execution.train_epi_network,
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

    embedding_model = prepare_cardinality_estimator(
        model_config=cfg.models.embedder.config,
        model_directory=cfg.models.embedder.dir
    )

    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(cfg.hyperparameters.mlp_dimension, 1),
        }
    }

    # Prepare large (20 million parameters) oracle model to estimate cardinality of join plans
    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config, model_directory=cfg.models.oracle.dir
    )

    # Prepare plan cost estimation models and epinet
    cost_net_attention_pooling = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=cfg.hyperparameters.mlp_dimension
    )
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, device)
    epinet_cost_estimation = EpistemicNetwork(
        cfg.hyperparameters.epinet_index_dim, cfg.models.epinet.prior_config, combined_model, device=device
    )

    experiment_base_dir = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training"

    if cfg.execution.train_epi_network:
        if not cfg.models.epinet.trained_cost_model_file:
            raise ValueError("Training epinet requires a pretrained cost model.")

        epinet_cost_estimation.load_epinet(
            cfg.models.epinet.trained_cost_model_file,
            load_only_cost_model=True
        )
        experiment_name = "simulated_cost_epinet_training"
    else:
        experiment_name = "simulated_cost"

    writer = ExperimentWriter(experiment_base_dir, experiment_name, {}, {})


    main_simulated_training(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        oracle_model=oracle_model,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer
    )


@hydra.main(version_base=None,
            config_path="../experiments/experiment_configs/epinet_cost_estimation",
            config_name="simulated_supervised_cost_estimation.yaml")
def main(cfg: DictConfig):
    # Temporarily unlock the config to allow dynamic updates
    OmegaConf.set_struct(cfg, False)

    # Locate the best embedder and oracle directory dynamically
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_q_error")

    # Inject the resolved path directly into the config state
    cfg.models.embedder.dir = best_embedder_dir
    cfg.models.oracle.dir = best_oracle_dir

    # Relock the config to prevent accidental downstream modifications
    OmegaConf.set_struct(cfg, True)

    # Pass the unified config to the main setup function
    main_supervised_value_estimation(cfg)

if __name__ == "__main__":
    main()