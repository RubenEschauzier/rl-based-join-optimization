from os import listdir
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


def run_models(query, embedding_models, pool = False):
    embeddings_list = []
    for graph, model in zip(query.query_graph_representations, embedding_models):
        embedding = model.run(query.features, graph)
        embeddings_list.append(embedding)
    if pool:
        # Average pool of four graph embeddings
        embeddings_tensor = torch.stack(embeddings_list)
        return torch.mean(embeddings_tensor, dim=0)
    else:
        embeddings_tensor = torch.concat(embeddings_list, dim=1)
        return embeddings_tensor


def load_watdiv_queries(location):
    raw_queries = []
    files = [f for f in listdir(location) if isfile(join(location, f))]
    for file in files:
        with open(join(location, file), 'r') as f:
            raw_queries.extend(f.read().strip().split('\n\n'))

    queries = [Query(query) for query in raw_queries]
    return queries


def get_parameters_model(all_models):
    parameters = []
    for model in all_models:
        parameters.extend(list(model.parameters()))
    return parameters


def embed_query_graphs(queries, embedding_models):
    query_graph_embeddings = []
    for query in queries:
        query_emb = run_models(query, embedding_models)
        query_graph_embeddings.append(query_emb)
    return query_graph_embeddings
