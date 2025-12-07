import os.path
import pickle
from os import listdir, path
from os.path import isfile, join

import torch

from src.datastructures.query import Query


def initialize_graph_models(factories: [tuple]):
    """
    Initializes graph models in order specified in factories. This order should align with the order of generated graphs
    to run over
    :param factories: list of tuples of model factories and number of model should be created from this factory
    :return: list with created models
    """
    models = []
    for (factory, n_instances) in factories:
        for i in range(n_instances):
            models.append(factory.build_model_from_config())
    return models


def run_models(query, embedding_models, pool=False, training=True):
    embeddings_list = []
    for graph, model in zip(query.query_graph_representations, embedding_models):
        embedding = model.forward(query.features, graph, training=training)
        embeddings_list.append(embedding)
    if pool:
        # Average pool of four graph embeddings
        embeddings_tensor = torch.stack(embeddings_list)
        return torch.mean(embeddings_tensor, dim=0)
    else:
        embeddings_tensor = torch.concat(embeddings_list, dim=1)
        return embeddings_tensor


def load_watdiv_queries(location, per_template=False):
    files = [f for f in listdir(location) if isfile(join(location, f))]
    if per_template:
        return read_watdiv_template_dict(location, files)
    else:
        return read_watdiv_single_list(location, files)


def load_watdiv_queries_pickle(location):
    with open(location, 'rb') as f:
        return pickle.load(f)


def read_watdiv_single_list(location, files):
    raw_queries = []
    for file in files:
        with open(join(location, file), 'r') as f:
            raw_queries.extend(f.read().strip().split('\n\n'))

    queries = [Query(query) for query in raw_queries]
    return queries


def read_watdiv_template_dict(location, files):
    queries = {}
    for file in files:
        with open(join(location, file), 'r') as f:
            queries[file.split('.')[0]] = [Query(query) for query in f.read().strip().split('\n\n')]
    return queries


def get_parameters_model(all_models):
    parameters = []
    for model in all_models:
        parameters.extend(list(model.parameters()))
    return parameters


def embed_query_graphs(queries, embedding_models, training=True):
    query_graph_embeddings = []
    for query in queries:
        query_emb = run_models(query, embedding_models, training=training)
        query_graph_embeddings.append(query_emb)
    return query_graph_embeddings


def q_error_fn(pred, true, eps=1e-7):
    return torch.max(true / (pred + eps), pred / (true + eps))


def mixed_mae_q_error_loss(alpha, pred, true):
    mae = alpha * torch.nn.L1Loss(reduction='mean')(pred, true)
    q_error = (1 - alpha) * torch.mean(q_error_fn(torch.exp(pred), torch.exp(true)))
    return mae + q_error


def save_checkpoint(ckp_dir, optimizer, models, model_file_names, statistics):
    if not os.path.isdir(ckp_dir):
        os.mkdir(ckp_dir)

    torch.save(optimizer.state_dict(), path.join(ckp_dir, "optimizer"))
    with open(path.join(ckp_dir, "statistics"), "wb") as f:
        pickle.dump(statistics, f)
    for model, filename in zip(models, model_file_names):
        torch.save(model.state_dict(), path.join(ckp_dir, filename))


def reset_value_head_only(model):
    # Get the last layer of the value network (the actual value head)
    value_head = model.policy.value_net  # Usually the last Linear layer

    if isinstance(value_head, torch.nn.Linear):
        torch.nn.init.orthogonal_(value_head.weight, gain=1)
        torch.nn.init.constant_(value_head.bias, 0)
