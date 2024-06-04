from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.datastructures.query import Query
from src.models.model_instantiator import ModelFactory
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


def featurize_queries(featurizer, graph_view_creator):
    pass


def prepare(env, queries_location, rdf2vec_vector_location):
    queries = load_queries(queries_location)

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
                 n_epoch, batch_size, seed):
    model_factory = ModelFactory("experiments/configs/test_config.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory, 4)])
    env = BlazeGraphQueryEnvironment(endpoint)

    queries = prepare(env, queries_location, rdf2vec_vector_location)
    train_queries, test_queries = train_test_split(queries, test_size=.2, random_state=seed)

    for i in range(n_epoch):
        train_queries = shuffle(train_queries, random_state=seed)
        embedded_features = featurize_queries()
        pass

