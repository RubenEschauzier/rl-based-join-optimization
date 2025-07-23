import functools
import math
import typing
from time import sleep

import torch
import json
from tqdm import tqdm

from src.datastructures.query import Query
from src.datastructures.query_pytorch_dataset import QueryCardinalityDataset
from src.query_featurizers.featurize_edge_labeled_graph import QueryToEdgeLabeledGraph
from src.query_featurizers.featurize_predicate_edges import QueryToEdgePredicateGraph
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_featurizers.quad_views import FeaturizeQueryGraphQuadViews


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

def load_raw_queries(queries_location, to_load=None):
    raw_queries = load_data_txt_file(queries_location, to_load)

    queries = []
    for query in tqdm(raw_queries):
        queries.append(Query(query))
    return queries


def load_queries_and_cardinalities(queries_location, cardinalities_location=None, to_load=None):
    if cardinalities_location:
        tqdm.write("Converting query strings to Query data structure...")
        queries = load_raw_queries(queries_location, to_load=to_load)
        cardinalities = load_data_txt_file(cardinalities_location, to_load=to_load)
        pass
    else:
        with open(queries_location, 'r') as f:
            queries_json = json.load(f)
        tqdm.write("Converting query strings to Query data structure...")
        sleep(.5)
        if to_load:
            queries = [Query(query['query']) for i, query in tqdm(enumerate(queries_json)) if i < to_load]
            cardinalities = [query['y'] for i, query in enumerate(queries_json) if i < to_load]
        else:
            queries = [Query(query['query']) for query in tqdm(queries_json)]
            cardinalities = [query['y'] for query in queries_json ]

    return queries, cardinalities

def load_queries_into_dataset(queries_location, endpoint_location, rdf2vec_vector_location,
                              env,
                              feature_type: typing.Literal["labeled_edge", "predicate_edge"],
                              validation_size=.2,
                              load_mappings = True,
                              to_load=None, occurrences_location=None, tp_cardinality_location=None,
                              shuffle=True):
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)

    featurizer_edge_labeled_graph = load_featurizer(feature_type,
                                                    vectors, env,
                                                    rdf2vec_vector_location, endpoint_location,
                                                    occurrences_location, tp_cardinality_location)
    dataset = QueryCardinalityDataset(root=queries_location,
                                      featurizer=featurizer_edge_labeled_graph,
                                      to_load=to_load,
                                      load_mappings=load_mappings,
                                      )
    if shuffle:
        dataset = dataset.shuffle()
    train_dataset = dataset[math.floor(len(dataset)*validation_size):]
    validation_dataset = dataset[:math.floor(len(dataset)*validation_size)]

    return train_dataset, validation_dataset

def load_featurizer(featurizer_type: typing.Literal["labeled_edge", "predicate_edge"],
                    vectors, query_env,
                    rdf2vec_vector_location, endpoint_location, occurrences_location=None, tp_cardinality_location=None):

    occurrences = None
    tp_cardinality = None
    if occurrences_location:
        with open(occurrences_location, 'r') as f:
            occurrences = json.load(f)
    if tp_cardinality_location:
        with open(tp_cardinality_location, 'r') as f:
            tp_cardinality = json.load(f)

    if featurizer_type == "labeled_edge":
        query_to_graph = QueryToEdgeLabeledGraph(vectors, query_env, tp_cardinalities=tp_cardinality)
    elif featurizer_type == "predicate_edge":
        query_to_graph = QueryToEdgePredicateGraph(vectors, query_env, term_occurrences=occurrences)
    else:
        raise NotImplementedError
    return functools.partial(query_to_graph.transform_undirected)

if __name__ == '__main__':
    load_queries_and_cardinalities(r"C:\Users\ruben\projects\rl-based-join-optimization\data\pretrain_data\generated_queries_subsampler\stars_2.json")
