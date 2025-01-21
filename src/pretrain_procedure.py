import functools
import math
from os import path
import logging
from typing import Literal

import numpy as np
from sklearn.utils import shuffle

import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datastructures.query import Query
from src.datastructures.query_pytorch_dataset import QueryCardinalityDataset
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_edge_labeled_graph import QueryToEdgeLabeledGraph
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.utils.training_utils.query_loading_utils import load_queries_and_cardinalities, load_and_prepare_queries, \
    prepare_pretraining_queries
from src.utils.training_utils.utils import initialize_graph_models, get_parameters_model, embed_query_graphs, \
    load_watdiv_queries_pickle, save_checkpoint, register_debugging_hooks, q_error_fn


def prepare_watdiv_test_data(env, test_queries_location, test_cardinalities_location, rdf2vec_vector_location, device):
    queries = load_watdiv_queries_pickle(test_queries_location)

    cardinalities = load_watdiv_queries_pickle(test_cardinalities_location)
    for key in cardinalities.keys():
        cardinalities[key] = [math.log(int(card) + 1) for card in cardinalities[key]]

    for key, value in queries.items():
        queries[key] = [Query(query) for query in value]
        queries[key] = prepare_pretraining_queries(env, queries[key], rdf2vec_vector_location,
                                                   disable_progress_bar=True)
        for query in queries[key]:
            query.features = query.features.to(device)
            query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]
    return queries, cardinalities


def validate_model(graph_embedding_models, cardinality_estimation_head,
                   val_queries, val_cardinalities, batch_size,
                   device):
    cardinality_estimation_head.eval()
    loss_fn = torch.nn.L1Loss(reduction="none")

    validation_run_stats = {}

    n_processed = 0
    n_batches = 0
    predictions = []
    q_errors = []
    maes = []
    losses = []
    # TODO: PROPER BATCHED ITERATION WE'RE MISSING OUT ON ELEMENTS
    for b in range(0, len(val_queries), batch_size):
        queries_batch = val_queries[b:b + batch_size]
        cardinalities_batch = val_cardinalities[b:b + batch_size]

        for query in queries_batch:
            query.features = query.features.to(device)
            query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]

        embedded_features = embed_query_graphs(queries_batch, graph_embedding_models, training=False)
        embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

        pred = cardinality_estimation_head.forward(embedded_features_batched)

        n_processed += len(queries_batch)
        n_batches += 1
        loss = loss_fn(pred.squeeze(), cardinalities_batch)
        losses.extend(loss.cpu().detach().numpy())
        predictions.extend(pred.cpu().detach().numpy())
        # Prepare statistics list
        for i in range(pred.shape[0]):
            # Scale back to normal cardinality
            prediction_query = pred[i].cpu().detach().item()
            cardinality_query = cardinalities_batch[i].cpu().item()
            q_error_query = q_error_fn(torch.exp(pred[i]), torch.exp(cardinalities_batch[i])). \
                cpu().detach().item()
            loss_query = loss[i].cpu().detach().item()
            mae_query = torch.abs(torch.exp(pred[i]).cpu().detach() - torch.exp(cardinalities_batch[i]).cpu().detach())

            validation_run_stats[queries_batch[i].query_string] = {
                "prediction": prediction_query,
                "actual": cardinality_query,
                "loss": loss_query,
                "mae": mae_query,
                "q_error": q_error_query
            }

            q_errors.append(q_error_query)
            maes.append(mae_query)
    cardinality_estimation_head.train()
    return losses, maes, q_errors, validation_run_stats

def test_model_watdiv(queries, cardinalities,
                      graph_embedding_models, cardinality_estimation_head,
                      device):
    cardinality_estimation_head.eval()
    loss_fn = torch.nn.L1Loss(reduction="none")

    test_run_stats = {}

    for template in queries.keys():
        # Ensure proper shapes when only one query is in a template
        cardinalities_template = torch.tensor(cardinalities[template], device=device)
        queries_template = queries[template]

        embedded_features = embed_query_graphs(queries_template, graph_embedding_models, training=False)
        embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

        predictions = cardinality_estimation_head.forward(embedded_features_batched)
        loss = loss_fn(predictions.squeeze(), cardinalities_template.squeeze())

        q_errors = []
        # In case loss has no shape, we have single query for a template
        if len(loss.shape) == 0:
            q_error_query = q_error_fn(torch.exp(predictions.detach().squeeze()),
                                       torch.exp(cardinalities_template.detach().squeeze())).cpu().item()
            prediction_query = predictions.cpu().detach().item()
            cardinality_query = cardinalities_template.cpu().item()
            loss_query = loss.cpu().detach().item()
            test_run_stats[queries_template[0].query_string] = {
                "prediction": prediction_query,
                "actual": cardinality_query,
                "loss": loss_query,
                "q_error": q_error_query
            }
        # Otherwise we iterate over the predictions for all queries
        else:
            for i in range(predictions.shape[0]):
                q_error_query = q_error_fn(torch.exp(predictions[i].detach().squeeze()),
                                           torch.exp(cardinalities_template[i].detach().squeeze())).cpu().item()
                prediction_query = predictions[i].cpu().detach().item()
                cardinality_query = cardinalities_template[i].cpu().item()
                loss_query = loss[i].cpu().detach().item()

                q_errors.append(q_error_query)

                test_run_stats[queries_template[i].query_string] = {
                    "prediction": prediction_query,
                    "actual": cardinality_query,
                    "loss": loss_query,
                    "q_error": q_error_query
                }
    cardinality_estimation_head.train()
    return test_run_stats


def load_queries(env, device,
                 train_queries_location, rdf2vec_vector_location,
                 load_prepared_queries_location=None,
                 train_cardinalities_location=None,
                 test_query_location=None, test_cardinalities_location=None,
                 save_prepared_queries_location=None,
                 to_load = None
                 ):
    # If no cardinality location is given, this function can load JSON formatted queries, with the query in the field
    # 'query' and cardinality in 'y'. Otherwise, it assumes a txt file with cardinalities and queries separately
    test_q = None
    test_c = None
    if test_query_location and test_cardinalities_location:
        test_q, test_c = prepare_watdiv_test_data(env=env,
                                                  test_queries_location=test_query_location,
                                                  test_cardinalities_location=test_cardinalities_location,
                                                  rdf2vec_vector_location=rdf2vec_vector_location,
                                                  device=device)

    if load_prepared_queries_location:
        feature_dict = torch.load(load_prepared_queries_location)

        num_queries = len(feature_dict.keys())
        queries, cardinalities = load_queries_and_cardinalities(train_queries_location,
                                                                cardinalities_location=train_cardinalities_location,
                                                                to_load=to_load)

        # Iterate over queries to fill with pickled dictionary containing initialized features
        for query in queries:
            query.set_features_graph_views_from_dict(feature_dict)
        cardinalities = cardinalities[:num_queries]
        return queries, cardinalities, test_q, test_c
    else:
        queries, cardinalities = load_and_prepare_queries(env=env,
                                                          queries_location=train_queries_location,
                                                          cardinalities_location=train_cardinalities_location,
                                                          rdf2vec_vector_location=rdf2vec_vector_location,
                                                          prepared_queries_location=save_prepared_queries_location,
                                                          to_load=to_load
                                                          )
        return queries, cardinalities, test_q, test_c




def run_pretraining(queries, cardinalities,
                    device, n_epoch, batch_size, lr, seed,
                    ckp_dir=None, test_queries=None, test_cardinalities=None
                    ):
    # Build graph embedding models from config
    model_factory_graph_emb = ModelFactory("experiments/model_configs/graph_embedding_model.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory_graph_emb, 4)])

    # Build cardinality estimator from config
    model_factory_card_est_head = ModelFactory("experiments/model_configs/cardinality_estimation_head.yaml")
    cardinality_estimation_head = model_factory_card_est_head.load_mlp()

    # # Register hooks for all layers
    # for layer in cardinality_estimation_head:
    #     register_debugging_hooks(layer)

    # Move models to device
    [graph_model.to(device) for graph_model in graph_embedding_models]
    cardinality_estimation_head.to(device)

    # Get parameters and set optimizer
    parameters = get_parameters_model(graph_embedding_models)
    parameters.extend(cardinality_estimation_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2
                                  )

    train_queries, val_queries, train_card, val_card = train_test_split(queries, cardinalities,
                                                                        test_size=.2, random_state=seed)
    train_card = torch.tensor(train_card, device=device)
    val_card = torch.tensor(val_card, device=device)

    # Use L1 loss
    loss_fn = torch.nn.L1Loss(reduction="none")

    previous_lr = scheduler.get_last_lr()

    # Per epoch stats tracking
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_q_error_per_epoch = []

    for i in range(n_epoch):
        cardinality_estimation_head.train()
        # Dictionary {query string: {prediction, actual, q_error, loss}}
        train_epoch_stats = {}
        loss_epoch = []

        train_queries, train_card = shuffle(train_queries, train_card, random_state=seed)

        for b in range(0, len(train_queries), batch_size):
            queries_batch = train_queries[b:b + batch_size]
            cardinalities_batch = train_card[b:b + batch_size]

            # Move to gpu per batch to save GPU memory
            for query in queries_batch:
                query.features = query.features.to(device)
                query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]

            optimizer.zero_grad()

            embedded_features = embed_query_graphs(queries_batch, graph_embedding_models)
            embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

            pred = cardinality_estimation_head.forward(embedded_features_batched)
            loss = loss_fn(pred.squeeze(), cardinalities_batch)

            torch.mean(loss).backward()

            optimizer.step()

            # Track statistics
            for k in range(pred.shape[0]):
                loss_float = loss[k].cpu().detach().item()
                loss_epoch.append(loss_float)
                train_epoch_stats[queries_batch[k].query_string] = {
                    "prediction": pred[k].cpu().detach().item(),
                    "actual": cardinalities_batch[k].cpu().detach().item(),
                    "loss": loss_float,
                    "q_error": q_error_fn(torch.exp(pred[k].detach().squeeze()),
                                          torch.exp(cardinalities_batch[k].detach().squeeze())).cpu().item()
                }

        val_loss, val_mae, val_q_error, val_stats = validate_model(
            graph_embedding_models=graph_embedding_models,
            cardinality_estimation_head=cardinality_estimation_head,
            val_queries=val_queries,
            val_cardinalities=val_card,
            batch_size=batch_size,
            device=device
        )

        train_loss_per_epoch.append(np.mean(loss_epoch))
        val_loss_per_epoch.append(val_loss)
        val_q_error_per_epoch.append(val_q_error)

        if test_queries:
            test_stats = test_model_watdiv(test_queries, test_cardinalities,
                                           graph_embedding_models,
                                           cardinality_estimation_head, device)

            average_test_loss = np.mean([stat['loss'] for stat in test_stats.values()])
            average_test_q_error = np.mean([stat['q_error'] for stat in test_stats.values()])
        else:
            test_stats = None
            average_test_loss = None
            average_test_q_error = None
        print("Epoch {}/{}: Train loss: {}, Val loss: {}, Val MAE: {} Val q-error: {}, Test loss: {}, q-error: {}".format(
            i + 1,
            n_epoch,
            train_loss_per_epoch[i],
            sum(val_loss) / len(val_loss),
            sum(val_mae) / len(val_mae),
            np.sum(np.abs(np.array(val_q_error))) / len(val_q_error),
            average_test_loss,
            average_test_q_error
        ))

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(sum(val_loss) / len(val_loss))

        # Save checkpoint to directory if it is specified
        if ckp_dir:
            # Prepare checkpoint data
            statistics_dict = {
                'train_stats': train_epoch_stats,
                'val_stats': val_stats,
                'test_stats': test_stats,
            }

            models_to_save = [cardinality_estimation_head]
            models_to_save.extend(graph_embedding_models)
            model_file_names = ["cardinality_head", "graph_emb_0", "graph_emb_1", "graph_emb_2", "graph_emb_3"]

            save_checkpoint(path.join(ckp_dir, "ckp_{}".format(i)),
                            optimizer,
                            models_to_save, model_file_names,
                            statistics_dict
                            )

def main_pretraining(train_queries_location, rdf2vec_vector_location, endpoint_uri,
                     n_epoch, batch_size, lr, seed,
                     ckp_dir=None,
                     load_prepared_queries_location=None,
                     train_cardinalities_location=None,
                     test_query_location=None, test_cardinalities_location=None,
                     save_prepared_queries_location=None,
                     ):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on: {}".format(device))

    # Initialize the query environment
    env = BlazeGraphQueryEnvironment(endpoint_uri)

    queries, cardinalities, test_q, test_c = load_queries(env, device,
                                                          train_queries_location, rdf2vec_vector_location,
                                                          load_prepared_queries_location,
                                                          train_cardinalities_location,
                                                          test_query_location, test_cardinalities_location,
                                                          save_prepared_queries_location,
                                                          to_load=500
                                                          )
    cardinalities = [np.log(card) for card in cardinalities]
    if test_c:
        test_c = [np.log(t_card) for t_card in test_c]

    run_pretraining(queries, cardinalities, device, n_epoch, batch_size, lr, seed,
                    ckp_dir=ckp_dir, test_queries=test_q, test_cardinalities=test_c)

    pass


