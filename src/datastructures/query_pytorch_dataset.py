import faulthandler
import functools
import json
import os
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url

from src.datastructures.query import Query, ProcessQuery
from src.query_featurizers.featurize_edge_labeled_graph import QueryToEdgeLabeledGraph
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_predicate_edges import QueryToEdgePredicateGraph
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec


class QueryCardinalityDataset(InMemoryDataset):
    def __init__(self, root, featurizer, to_load=None, transform=None, pre_transform=None, ):
        self.featurizer = featurizer
        self.to_load = to_load
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])


    def raw_file_names(self):
        return ['star_queries_2_3_5.json']

    def processed_file_names(self):
        return ['star_queries_2_3_5_edge_graph.pt']

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
                    queries.append({
                        "query": data['query'],
                        "cardinality": data['y'],
                        "triple_patterns": tp_str,
                        "rdflib_patterns": tp_rdflib
                    })
            raw_queries_list.extend(queries)

        data_list = []
        for i, query in enumerate(raw_queries_list):
            data_list.append(self.featurizer(query))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.transform is not None:
            data_list = [self.transform(data) for data in data_list]

        print("Loaded: {} queries".format(len(data_list)))

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
