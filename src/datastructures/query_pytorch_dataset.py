import json
import os
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url

from main import ROOT_DIR
from src.datastructures.query import Query, ProcessQuery


class QueryCardinalityDataset(InMemoryDataset):
    def __init__(self, root, featurizer, to_load=None, transform=None, pre_transform=None, ):
        self.to_load = to_load
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])


    def raw_file_names(self):
        return ['star_queries_2_3_5.json']

    def processed_file_names(self):
        return ['star_queries_2_3_5.pt']

    def process(self):
        data_list = []

        for file in self.raw_file_names():
            queries = []
            path = os.path.join(self.raw_dir, file)
            with open(path, 'r') as f:
                raw_data = json.load(f)
            for i, data in enumerate(raw_data):
                if self.to_load and i < self.to_load:
                    queries.append({
                        "query": data['query'],
                        "cardinality": data['y'],
                        "triple_patterns": ProcessQuery.deconstruct_to_triple_pattern(data['query'])
                    })
            data_list.extend(queries)
        print(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    path_to_data = os.path.join(ROOT_DIR, 'data', 'pretrain_data', 'generated_queries_sub_sampler')
    dataset = QueryCardinalityDataset(root=path_to_data, to_load=10)
