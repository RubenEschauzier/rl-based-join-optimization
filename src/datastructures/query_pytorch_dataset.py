import json
import os
from torch_geometric.data import InMemoryDataset, download_url

from src.datastructures.query import Query, ProcessQuery


class QueryCardinalityDataset(InMemoryDataset):
    def __init__(self, root, featurizer, to_load=None, transform=None, pre_transform=None, ):
        self.featurizer = featurizer
        self.to_load = to_load
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])


    def raw_file_names(self):
        return [
            "watdiv_stars_2025-01-27_09-32-26_2.json",
            "watdiv_stars_2025-01-27_09-37-29_3.json",
            "watdiv_stars_2025-01-27_09-39-53_5.json",
            "watdiv_stars_2025-01-27_10-42-20_8.json"
        ]

    def processed_file_names(self):
        return [
            "watdiv_stars_2025-01-27_09-32-26_2.pt",
            "watdiv_stars_2025-01-27_09-37-29_3.pt",
            "watdiv_stars_2025-01-27_09-39-53_5.pt",
            "watdiv_stars_2025-01-27_10-42-20_8.pt"
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
            data_list.append([self.featurizer(query) for query in query_type])

        if self.pre_transform is not None:
            data_list = [[self.pre_transform(data) for data in data_type ] for data_type in data_list]

        if self.transform is not None:
            data_list = [[self.transform(data) for data in data_type ] for data_type in data_list]

        print("Loaded: {} queries".format(sum(len(sublist) for sublist in data_list)))

        for i, data_type in enumerate(data_list):
            self.save(data_type, self.processed_paths[i])

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
