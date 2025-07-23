import json
import os
import pickle
from typing import Type, Tuple

import torch
from torch_geometric.data import InMemoryDataset, download_url, Dataset
from torch_geometric.data.data import BaseData, Data
from torch_geometric.io import fs

from src.datastructures.query import Query, ProcessQuery


class QueryCardinalityDataset(InMemoryDataset):

    def __init__(self, root, featurizer, load_mappings=True,
                 to_load=None, transform=None, pre_transform=None, pre_filter=None ):
        self.featurizer = featurizer
        self.to_load = to_load
        self.load_mappings = load_mappings
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        if os.path.exists(os.path.join(self.processed_dir, "node_mappings.json")) and load_mappings:
            with open(os.path.join(self.processed_dir, "node_mappings.json"), "r") as fr:
                mappings = json.load(fr)
            self.data_mappings = mappings

    def get(self, idx: int):
        data = super().get(idx)
        if self.load_mappings:
            data.term_to_id = self.data_mappings[idx]

        return data

    def raw_file_names(self):
        return [
            "fixed_stars_2025-03-30_18-49-33_3.json",
            "fixed_stars_2025-03-30_19-10-42_5.json",
            # "fixed_stars_2025-04-13_14-17-45_8.json",
        ]

    def processed_file_names(self):
        return [
            "processed_stars.pt"
        ]

    def process(self):
        raw_queries_list = []

        for file in self.raw_file_names():
            queries = []
            path = os.path.join(self.raw_dir, file)
            with open(path, 'r') as f:
                raw_data = json.load(f)
            for i, data in enumerate(raw_data):
                if not self.to_load or i < self.to_load:
                    tp_str, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])
                    # Temp fix for wrong generated queries during testing
                    if "type" not in data:
                        data["type"] = "star"
                    queries.append({
                        "query": data['query'],
                        "cardinality": data['y'],
                        "triple_patterns": tp_str,
                        "rdflib_patterns": tp_rdflib,
                        "type": data['type'],
                    })
            raw_queries_list.append(queries)

        data_list = []
        for i, query_type in enumerate(raw_queries_list):
            data_list.extend([self.featurizer(query) for query in query_type])

        # Dictionary cannot be collated so it is saved seperately
        node_mappings = []
        for data in data_list:
            node_mappings.append(data.id_to_term)
            del data.id_to_term
        self.data_mappings = node_mappings

        with open(self.processed_dir + '/node_mappings.json', 'w') as fm:
            # noinspection PyTypeChecker
            json.dump(node_mappings, fm, indent=2)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list ]

        if self.transform is not None:
            data_list = [self.transform(data) for data in data_list ]

        print("Loaded: {} queries".format(sum(len(sublist) for sublist in data_list)))
        self.save(data_list, self.processed_paths[0])

def get_size_data(data):
    return sum([v.element_size() * v.numel() for k, v in data if type(v) != str])



if __name__ == "__main__":
    pass
    # path_to_data = os.path.join(ROOT_DIR, 'data', 'pretrain_data', 'generated_queries',
    #                             'sub_sampled_predicate_edge_undirected')
    # endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"
    #
    # rdf2vec_vector_location = os.path.join(ROOT_DIR, "data/input/rdf2vec_vectors_gnce/vectors_gnce.json")
    # vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    # query_env = BlazeGraphQueryEnvironment(endpoint_location)
    #
    # query_to_graph = QueryToEdgePredicateGraph(vectors, query_env)
    # featurizer_edge_labeled_graph = functools.partial(query_to_graph.transform_undirected)
    # dataset = QueryCardinalityDataset(root=path_to_data,
    #                                   featurizer=featurizer_edge_labeled_graph,
    #                                   to_load=500
    #                                   )
