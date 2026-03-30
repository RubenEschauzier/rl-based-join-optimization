import json
import os
from functools import partial

import torch
from torch_geometric.data import DataLoader

from main import find_best_epoch_directory
from src.models.epistemic_neural_network import  prepare_epinet_model
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.supervised_value_estimation.agents.EpinetCostEstimatorAgent import EpinetCostEstimatorAgent
from src.supervised_value_estimation.validation.validation_runner import multiprocess_validate_agent
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset_single
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask


def build_cost_agent(model_kwargs, epinet_hyperparam_kwargs, precomputed_indexes, precomputed_masks, device):
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

    epinet_cost_estimation = prepare_epinet_model(**model_kwargs,
                                                  device=device,
                                                  heads_config=heads_config,
                                                  heads_config_prior=heads_config_prior)
    epinet_cost_estimation.eval()
    agent = EpinetCostEstimatorAgent(
        epinet_cost_estimation,
        precomputed_indexes, precomputed_masks,
        **epinet_hyperparam_kwargs,
        head_name="plan_cost"
    )
    return agent

def run_validation(model_kwargs, epinet_hyperparams, locations_dict):
    query_env_data_prep = BlazeGraphQueryEnvironment(locations_dict["endpoint"])
    client = QLeverOptimizerClient(locations_dict["endpoint_query_execution"])

    val_dataset = load_queries_into_dataset_single(
        locations_dict["queries_loc"],
        locations_dict["endpoint"],
        locations_dict["embeddings"],
        query_env_data_prep,
        "predicate_edge",
        to_load=None,
        occurrences_location=locations_dict["occurrences_location"],
        tp_cardinality_location=locations_dict["tp_cardinality_location"],
        load_mappings=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Set weights dir to best epoch result
    best = find_best_epoch_directory(model_kwargs["model_weights"], "val_joint_gaussian_nll")
    model_kwargs["model_weights"] = str(os.path.join(best, "epinet_model.pt"))

    # Precompute some fixed values
    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    agent_kwargs = {
        "model_kwargs": model_kwargs,
        "epinet_hyperparam_kwargs": epinet_hyperparams,
        "precomputed_indexes": precomputed_indexes,
        "precomputed_masks": precomputed_masks,
        "device": torch.device("cpu"),
    }

    # Execute validation
    metrics = multiprocess_validate_agent(
        val_loader=val_loader,
        client=client,
        agent_builder_fn=build_cost_agent,
        agent_kwargs=agent_kwargs,
        beam_width=8,
        num_workers=4,
        max_concurrent_db_queries=4,
        samples_per_execution_batch=32,
        client_default_timeout=30
    )

    return metrics

if __name__ == "__main__":
    locations = {
        "endpoint": "http://localhost:9999/blazegraph/namespace/yago/sparql",
        "endpoint_query_execution": "http://localhost:8888",
        "queries_loc": "data/generated_queries/star_yago_gnce/dataset_val",
        "embeddings": "data/rdf2vec_embeddings/yago_gnce/model.json",
        "occurrences_location": "data/term_occurrences/yago_gnce/occurrences.json",
        "tp_cardinality_location": "data/term_occurrences/yago_gnce/tp_cardinalities.json",
    }
    output_location = "experiments/experiment_outputs/yago_gnce/validation/epinet_cost_model.json"
    model_kwargs_dict = {
        "full_gnn_config": "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_separate_head.yaml",
        "config_ensemble_prior": "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml",
        "model_weights": "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/"
                         "simulated_cost_epinet_training-25-03-2026-22-26-33",
        "mlp_dimension": 64,
        "epinet_index_dim": 32,
    }
    epinet_hyperparams_dict = {
        "alpha_mlp": 0.08,
        "alpha_ensemble": .3,
        "n_epinet_samples": 64,
    }
    metrics_val = run_validation(model_kwargs_dict, epinet_hyperparams_dict, locations)
    with open(os.path.join(output_location), 'w') as f:
        json.dump(metrics_val, f)
