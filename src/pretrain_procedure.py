import math
from os import path
import logging
import numpy as np
from sklearn.utils import shuffle

import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.datastructures.query import Query
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_graph_featurizers.quad_views import FeaturizeQueryGraphQuadViews
from src.utils.training_utils.utils import initialize_graph_models, get_parameters_model, embed_query_graphs, \
    load_watdiv_queries_pickle, save_checkpoint


def load_data_txt_file(location, to_load=None):
    data = []
    num_processed = 0
    with open(location, 'r') as f:
        for line in f.readlines():
            # Skip white spaces
            if len(line.strip()) > 0:
                data.append(line.strip())
                num_processed += 1
            if to_load and len(data) == to_load:
                break
    return data


def prepare_pretraining_queries(env, queries, rdf2vec_vector_location, save_prepared_queries=False, save_location=None,
                                disable_progress_bar=False):
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    rdf2vec_featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    view_creator = FeaturizeQueryGraphQuadViews()

    queries = rdf2vec_featurizer.run(queries, disable_progress_bar=disable_progress_bar)
    queries = view_creator.run(queries, "edge_index", disable_progress_bar=disable_progress_bar)

    # When preparing many queries this can be used to reuse the computations
    if save_prepared_queries:
        features_dict = {}
        for query in queries:
            features_dict[query.query_string] = [query.features, query.query_graph_representations]
        torch.save(features_dict, save_location)
    return queries


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


def load_raw_queries(queries_location, to_load=None):
    raw_queries = load_data_txt_file(queries_location, to_load)

    queries = []
    for query in tqdm(raw_queries):
        queries.append(Query(query))
    return queries


def load_and_prepare_queries(env, queries_location, rdf2vec_vector_location,
                             save_prepared_queries, prepared_queries_location):
    queries = load_raw_queries(queries_location)

    # Prepare features of queries
    queries = prepare_pretraining_queries(env, queries, rdf2vec_vector_location,
                                          save_prepared_queries=save_prepared_queries,
                                          save_location=prepared_queries_location
                                          )
    return queries


def compute_q_errors(predictions, true_cardinalities):
    total_q_error = 0
    for pred, actual in zip(predictions.cpu(), true_cardinalities):
        pred = pred.detach()
        ratios = np.array([actual / pred, pred / actual])
        total_q_error += ratios[np.isfinite(ratios)].max()
    return total_q_error


def q_error_function(pred, true, eps=1e-7):
    # print("Pred: {}, True: {}, Q-error: {}".
    #       format(pred.item(), true, torch.max(true / (pred + eps), pred / (true + eps)).item())
    #       )
    return torch.max(true / (pred + eps), pred / (true + eps))


def validate_model(graph_embedding_models, cardinality_estimation_head,
                   val_queries, val_cardinalities, batch_size,
                   device):
    loss_fn = torch.nn.L1Loss(reduction="none")

    validation_run_stats = {}

    n_processed = 0
    n_batches = 0
    predictions = []
    q_errors = []
    losses = []
    # TODO: PROPER BATCHED ITERATION WE'RE MISSING OUT ON ELEMENTS
    for b in range(0, len(val_queries), batch_size):
        queries_batch = val_queries[b:b + batch_size]
        cardinalities_batch = val_cardinalities[b:b + batch_size]

        for query in queries_batch:
            query.features = query.features.to(device)
            query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]

        embedded_features = embed_query_graphs(queries_batch, graph_embedding_models)
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
            q_error_query = q_error_function(torch.exp(pred[i]), torch.exp(cardinalities_batch[i])). \
                cpu().detach().item()
            loss_query = loss[i].cpu().detach().item()

            validation_run_stats[queries_batch[i].query_string] = {
                "prediction": prediction_query,
                "actual": cardinality_query,
                "loss": loss_query,
                "q_error": q_error_query
            }

            q_errors.append(q_error_query)
    return losses, q_errors, validation_run_stats


def test_model_watdiv(queries, cardinalities,
                      graph_embedding_models, cardinality_estimation_head,
                      device):
    loss_fn = torch.nn.L1Loss(reduction="none")

    test_run_stats = {}

    for template in queries.keys():
        # Ensure proper shapes when only one query is in a template
        cardinalities_template = torch.tensor(cardinalities[template], device=device)
        queries_template = queries[template]

        embedded_features = embed_query_graphs(queries_template, graph_embedding_models)
        embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

        predictions = cardinality_estimation_head.forward(embedded_features_batched)
        loss = loss_fn(predictions.squeeze(), cardinalities_template.squeeze())

        q_errors = []
        # In case loss has no shape, we have single query for a template
        if len(loss.shape) == 0:
            q_error_query = q_error_function(torch.exp(predictions.detach().squeeze()),
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
                q_error_query = q_error_function(torch.exp(predictions[i].detach().squeeze()),
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

    return test_run_stats


def run_pretraining(queries_location, cardinalities_location, rdf2vec_vector_location,
                    test_query_location, test_cardinalities_location,
                    endpoint_uri, n_epoch, batch_size, lr, seed,
                    ckp_dir=None,
                    save_prepared_queries=False, save_prepared_queries_location=None,
                    load_prepared_queries=False, load_prepared_queries_location=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on: {}".format(device))

    # Build graph embedding models from config
    model_factory_graph_emb = ModelFactory("experiments/model_configs/graph_embedding_model.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory_graph_emb, 4)])

    # Build cardinality estimator from config
    model_factory_card_est_head = ModelFactory("experiments/model_configs/cardinality_estimation_head.yaml")
    cardinality_estimation_head = model_factory_card_est_head.load_mlp()

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

    # Initialize the query environment
    env = BlazeGraphQueryEnvironment(endpoint_uri)

    test_queries, test_cardinalities = prepare_watdiv_test_data(env=env,
                                                                test_queries_location=test_query_location,
                                                                test_cardinalities_location=test_cardinalities_location,
                                                                rdf2vec_vector_location=rdf2vec_vector_location,
                                                                device=device)

    cardinalities = [math.log(int(card) + 1) for card in load_data_txt_file(cardinalities_location)]

    if load_prepared_queries:
        feature_dict = torch.load(load_prepared_queries_location)

        num_queries = len(feature_dict.keys())
        queries = load_raw_queries(queries_location, to_load=num_queries)

        # Iterate over queries to fill with pickled dictionary containing initialized features
        for query in queries:
            query.set_features_graph_views_from_dict(feature_dict)
        cardinalities = cardinalities[:num_queries]
    else:
        queries = load_and_prepare_queries(env=env,
                                           queries_location=queries_location,
                                           rdf2vec_vector_location=rdf2vec_vector_location,
                                           save_prepared_queries=save_prepared_queries,
                                           prepared_queries_location=save_prepared_queries_location
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
                    "q_error": q_error_function(torch.exp(pred[k].detach().squeeze()),
                                                torch.exp(cardinalities_batch[k].detach().squeeze())).cpu().item()
                }

        val_loss, val_q_error, val_stats = validate_model(
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

        test_stats = test_model_watdiv(test_queries, test_cardinalities,
                                       graph_embedding_models,
                                       cardinality_estimation_head, device)

        average_test_loss = np.mean([stat['loss'] for stat in test_stats.values()])
        average_test_q_error = np.mean([stat['q_error'] for stat in test_stats.values()])

        print("Epoch {}/{}: Train loss: {}, Validation loss: {}, q-error: {}, Test loss: {}, q-error: {}".format(
            i + 1,
            n_epoch,
            train_loss_per_epoch[i],
            sum(val_loss) / len(val_loss),
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
