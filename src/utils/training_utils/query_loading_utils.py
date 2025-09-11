import functools
import math
import os
import typing
from time import sleep

import torch
import json

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datastructures.post_process_data import filter_duplicate_subject_predicate_combinations, query_post_processor, \
    filter_failed_cardinality_queries
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

def split_raw_queries(query_dir, val_proportion, train_dir, val_dir):
    for file in os.listdir(query_dir):
        query_location = os.path.join(query_dir, file)

        if os.path.isdir(query_location):
            continue

        with open(query_location, "r") as f:
            data = json.load(f)
        print(f"Total records: {len(data)}")

        train, val = train_test_split(data, test_size=val_proportion, random_state=42)

        print(f"Train: {len(train)}, Val: {len(val)}")

        with open(os.path.join(train_dir, file), "w") as f:
            json.dump(train, f, indent=2)

        with open(os.path.join(val_dir, file), "w") as f:
            json.dump(val, f, indent=2)


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

def load_queries_into_dataset(queries_location_train, queries_location_val,
                              endpoint_location, rdf2vec_vector_location,
                              env,
                              feature_type: typing.Literal["labeled_edge", "predicate_edge"],
                              load_mappings = True,
                              to_load=None, occurrences_location=None, tp_cardinality_location=None,
                              shuffle_train=True, shuffle_val=False):
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)

    featurizer_edge_labeled_graph = load_featurizer(feature_type,
                                                    vectors, env,
                                                    rdf2vec_vector_location, endpoint_location,
                                                    occurrences_location, tp_cardinality_location)
    post_processor = filter_failed_cardinality_queries
    
    train_dataset = QueryCardinalityDataset(root=queries_location_train,
                                      featurizer=featurizer_edge_labeled_graph,
                                      post_processor=post_processor,
                                      to_load=to_load,
                                      load_mappings=load_mappings,
                                      )
    val_dataset = QueryCardinalityDataset(root=queries_location_val,
                                      featurizer=featurizer_edge_labeled_graph,
                                      post_processor=post_processor,
                                      to_load=to_load,
                                      load_mappings=load_mappings,
                                      )

    if shuffle_train:
        train_dataset = train_dataset.shuffle()
    if shuffle_val:
        val_dataset = val_dataset.shuffle()

    return train_dataset, val_dataset

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
    project_root = os.getcwd()
    dataset_name = "star_yago_gnce"
    raw_queries_loc = os.path.join(project_root, f"/data/generated_queries/{dataset_name}")
    train_queries_loc = os.path.join(raw_queries_loc, os.path.join("dataset_train", "raw"))
    val_queries_loc = os.path.join(raw_queries_loc, os.path.join("dataset_val", "raw"))

    split_raw_queries(raw_queries_loc, .1, train_queries_loc, val_queries_loc)