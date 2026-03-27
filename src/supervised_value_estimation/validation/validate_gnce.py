from functools import partial

import torch
from torch_geometric.data import DataLoader

from main import find_best_epoch_directory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.supervised_value_estimation.supervised_value_estimation import prepare_cardinality_estimator
from src.supervised_value_estimation.agents.CardinalityEstimatorAgent import CardinalityEstimatorValidationAgent
from src.supervised_value_estimation.validation.validation_runner import multiprocess_validate_agent
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset_single


def estimator_fn_builder(config_loc, trained_model_dir, device):
    """
    Initializes the embedding model and returns a callable estimator function.
    Must be executed within the worker process to avoid pickling issues.
    """
    embedding_model = prepare_cardinality_estimator(
        model_config=config_loc,
        model_directory=trained_model_dir
    )
    embedding_model.eval()

    def estimator_fn(query, model, estimation_device):
        pred = model.forward(
            x=query.x.to(estimation_device),
            edge_index=query.edge_index.to(estimation_device),
            edge_attr=query.edge_attr.to(estimation_device),
            batch=query.batch.to(estimation_device)
        )
        pred = pred[0]['output']
        return torch.exp(pred)

    return partial(estimator_fn, model=embedding_model, estimation_device=device)


def build_cardinality_agent(estimator_builder_fn, estimator_kwargs, estimator_requires_features):
    """
    Constructs the CardinalityEstimatorValidationAgent inside the worker process.
    This guarantees the model initializes locally per process.
    """
    estimator_fn = estimator_builder_fn(**estimator_kwargs)

    return CardinalityEstimatorValidationAgent(
        estimator_fn=estimator_fn,
        estimator_requires_features=estimator_requires_features
    )


def run_validation(model_parameters, locations_dict):
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

    best_embedder_dir = find_best_epoch_directory(model_parameters["trained_model_dir"], "val_q_error")

    # Structure arguments for the generic agent builder
    agent_kwargs = {
        "estimator_builder_fn": estimator_fn_builder,
        "estimator_kwargs": {
            "config_loc": model_parameters["config_loc"],
            "trained_model_dir": best_embedder_dir,
            "device": torch.device('cpu')
        },
        "estimator_requires_features": True
    }

    # Execute validation using the generic pipeline
    metrics = multiprocess_validate_agent(
        val_loader=val_loader,
        client=client,
        agent_builder_fn=build_cardinality_agent,
        agent_kwargs=agent_kwargs,
        beam_width=8,
        num_workers=8,
        max_concurrent_db_queries=4,
        samples_per_execution_batch=32,
        client_default_timeout=30.0
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
    model_configs = {
        "config_loc": "experiments/model_configs/pretrain_model/t_cv_repr_graph_norm_separate_head.yaml",
        "trained_model_dir": "experiments/experiment_outputs/yago_gnce/pretrained_models/"
                             "pretrain_experiment_triple_conv_graph_norm-02-03-2026-10-52-01",
    }
    run_validation(model_configs, locations)