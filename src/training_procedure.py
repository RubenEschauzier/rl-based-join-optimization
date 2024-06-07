from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn

from src.datastructures.query import Query
from src.models.model_instantiator import ModelFactory
from src.models.pointer_network import PointerNet
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_graph_featurizers.quad_views import FeaturizeQueryGraphQuadViews


def load_queries(location):
    raw_queries = []
    files = [f for f in listdir(location) if isfile(join(location, f))]
    for file in files:
        with open(join(location, file), 'r') as f:
            raw_queries.extend(f.read().strip().split('\n\n'))

    queries = [Query(query) for query in raw_queries]
    return queries


# Not sure how to batch this yet, might not need it due to small model size
def embed_query_graphs(queries, embedding_models):
    query_graph_embeddings = []
    for query in queries:
        query_emb = run_models(query, embedding_models)
        query_graph_embeddings.append(query_emb)
    return query_graph_embeddings


def run_models(query, embedding_models):
    embeddings_list = []
    for graph, model in zip(query.query_graph_representations, embedding_models):
        embedding = model.run(query.features, graph)
        embeddings_list.append(embedding)

    return torch.cat(embeddings_list, dim=1)


def prepare(env, queries_location, rdf2vec_vector_location):
    queries = load_queries(queries_location)[0::20]

    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    rdf2vec_featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    view_creator = FeaturizeQueryGraphQuadViews()

    queries = rdf2vec_featurizer.run(queries)
    queries = view_creator.run(queries, "edge_index")
    return queries


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


def run_training(endpoint, queries_location, rdf2vec_vector_location,
                 n_epoch, batch_size, max_seq_length, seed):
    # Build graph embedding models from config
    model_factory = ModelFactory("experiments/configs/test_config.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory, 4)])

    # Pointer network
    test = PointerNet(8192, 8192, 8192)

    # Initializes the query environments
    env = BlazeGraphQueryEnvironment(endpoint)

    # Prepare features of queries
    queries = prepare(env, queries_location, rdf2vec_vector_location)

    # Split queries
    train_queries, test_queries = train_test_split(queries, test_size=.2, random_state=seed)
    for i in range(n_epoch):
        train_queries = shuffle(train_queries, random_state=seed)
        for b in range(0, len(train_queries), batch_size):
            embedded_features = embed_query_graphs(train_queries[b:b + batch_size], graph_embedding_models)
            sequence_lengths = torch.Tensor([seq.shape[0] for seq in embedded_features])
            # embedded_features[0] = nn.functional.pad(embedded_features[0],
            #                                          (0, 0, 0, max_seq_length - embedded_features[0].shape[0]),
            #                                          value=0.0)
            padded_features = nn.utils.rnn.pad_sequence(embedded_features, batch_first=True)

            log_pointer_score, argmax_pointer, mask = test.forward(padded_features, sequence_lengths)
            print(argmax_pointer.shape)
            print(list(argmax_pointer))
            # print(len(embedded_features))
            # print(embedded_features[0].shape)
            # print(embedded_features)
        pass
