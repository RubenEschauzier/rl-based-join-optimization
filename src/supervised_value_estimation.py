# First train the model on (query, plan, estimated_cost) tuples (like we already do), use those epistemic neural nets
#   - Create an entire dataset in advance and then normalize it to be between 0 and 1
#   - Use only optimal reward for a given sub plan after augmentation
# Then create candidate plans in batches for real query latency prediction based on quantiles of the epistemic
# neural nets and quantile beam search.
#   - Select k beams per quantile with z full plans per beam
#   - Select n quantiles to search, so for n=3 highest 25 quantile performance, highest 50 quantile,
#   and highest 75 quantile and highest average value.
# We will use an adapted version of safe exploration from balsa:
#   - Prefer plans with highest 75 quantile performance
#   - Then investigate plans with 50 quantile
#   - Etc
# Execute these queries, record latency.
#   - Cache (query, plan, latency)
#   - Track execution times found for normalization between 0 and 1
#   - Augment data for sub plans, but ensure the best plan is used for reward of that sub plan
#   - Do adaptive timeouts by tracking execution times per query
import time
from datetime import datetime
import itertools
import os
import sys

import numpy as np
import optuna
from joblib import Parallel, delayed

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
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask
import torch



def prepare_data(endpoint_location,
                 queries_location_train, queries_location_val,
                 rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location_train, queries_location_val,
                                                           endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           shuffle_train=True, load_mappings=False
                                                           )
    return train_dataset, val_dataset


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
        return mean_metrics_val['val_joint_gaussian_nll']
    else:
        return mean_metrics_val['val_loss_cost_scaled']

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

    for epoch in range(1,n_epochs+1):
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
        joint_nll = validation_step(epoch_train_loss, epoch, train_summary, writer,
                        queries_val, query_plans_val,
                        precomputed_indexes, precomputed_masks,
                        mean_train, std_train,
                        loss,
                        epinet_cost_estimation,
                        device,
                        train_epi_network, n_epi_indexes_val,
                        sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                        generator=generator)

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

def main_simulated_training(train_dataset, val_dataset,
                            oracle_model,
                            epinet_cost_estimation,
                            train_epi_network,
                            device, query_batch_size,
                            save_loc_simulated_dataset, save_loc_simulated_dataset_val,
                            writer):

    # Hyperparameter results from tuning runs on partial data
    n_epi_indexes_train = 16
    n_epi_indexes_val = 1000
    sigma = 0.60
    alpha_mlp = 0.08
    alpha_ensemble = 0.30
    lr = .0001
    weight_decay = 0.04
    n_epochs = 25

    writer.create_experiment_directory()

    # oracle_model = oracle_model.to(device)
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, save_loc_simulated_dataset)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, save_loc_simulated_dataset_val)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

    train_simulated_cost_model(queries_train=train_dataset, query_plans_train=train_plans,
                               mean_train=mean_train, std_train=std_train,
                               queries_val=val_dataset, query_plans_val=val_plans,
                               epinet_cost_estimation=epinet_cost_estimation,
                               device=device,
                               query_batch_size=query_batch_size,
                               n_epi_indexes_train=n_epi_indexes_train, n_epi_indexes_val=n_epi_indexes_train,
                               train_epi_network=train_epi_network,
                               writer=writer,
                               sigma=sigma, alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble,
                               lr=lr, weight_decay=weight_decay, n_epochs=n_epochs)

def main_latency_training(train_dataset, val_dataset,
                          epinet_cost_estimation,
                          device, query_batch_size,
                          n_beams, max_plans,
                          writer
                          ):
    pass


def main_supervised_value_estimation(endpoint_location,
                                     queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated_dataset, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_directory_oracle,
                                     model_config_embedder, model_directory_embedder,
                                     model_config_epistemic_prior,
                                     train_epi_network,
                                     trained_cost_model_loc,
                                     ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    # oracle_model = prepare_cardinality_estimator(
    #     model_config=model_config_oracle, model_directory=model_directory_oracle
    # )
    mlp_dimension = 64
    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(mlp_dimension, 1),
        }
    }
    cost_net_attention_pooling = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=mlp_dimension
    )
    embedding_model = prepare_cardinality_estimator(model_config=model_config_embedder,
                                                    model_directory=model_directory_embedder)
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, device)

    epinet_cost_estimation = EpistemicNetwork(32, model_config_epistemic_prior, combined_model, device=device)
    if train_epi_network:
        print("Training epinet network")
        if not trained_cost_model_loc:
            raise ValueError("Training epinet requires a pretrained cost model")
        epinet_cost_estimation.load_epinet(trained_cost_model_loc, load_only_cost_model=True)
        writer = ExperimentWriter("experiments/experiment_outputs/yago_gnce/supervised_epinet_training",
                                  "simulated_cost_epinet_training",
                                  {}, {})
    else:
        writer = ExperimentWriter("experiments/experiment_outputs/yago_gnce/supervised_epinet_training",
                                  "simulated_cost",
                                  {}, {})

    main_simulated_training(train_dataset, val_dataset,
                            "Should be oracle currently not exist",
                            epinet_cost_estimation,
                            train_epi_network,
                            device,
                            8,
                            save_loc_simulated_dataset, save_loc_simulated_val,
                            writer)


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/yago/sparql"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_oracle = "experiments/model_configs/policy_networks/t_cv_repr_huge.yaml"
    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"
    model_config_emb_pair_norm = "experiments/model_configs/policy_networks/t_cv_repr_pair_norm_cardinality_head_own_embeddings.yaml"
    model_config_emb_graph_norm = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_cardinality_head_own_embeddings.yaml"

    model_config_prior_tiny = "experiments/model_configs/prior_networks/prior_t_cv_tiny.yaml"
    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"

    oracle_experiment_dir = "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce_large_pretrain-05-10-2025-18-13-40"
    # trained_cost_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-12-02-2026-17-17-13/epoch-25/model/epinet_model.pt"
    trained_cost_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-24-02-2026-17-00-05/epoch-1/model/epinet_model.pt"

    emb_experiment_dir = ("experiments/experiment_outputs/yago_gnce/pretrained_models/"
                      "pretrain_experiment_triple_conv-15-12-2025-11-10-45")
    emb_experiment_dir_pair_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_pair_norm-15-12-2025-10-00-26")

    emb_experiment_dir_graph_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_graph_norm-15-12-2025-09-12-57")

    save_loc_simulated = "data/simulated_query_plan_data/star_yago_gnce/data"
    save_loc_simulated_val = "data/simulated_query_plan_data/star_yago_gnce/val_data"

    model_dir_oracle = os.path.join(oracle_experiment_dir, "epoch-39/model")
    model_dir_embedder = find_best_epoch_directory(emb_experiment_dir, "val_q_error")
    main_supervised_value_estimation(endpoint_location, queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_dir_oracle,
                                     model_config_emb, model_dir_embedder,
                                     model_config_prior,
                                     True,
                                     trained_cost_model_file
                                     )