import json
import os
from torch_geometric.data import InMemoryDataset
import torch
from tqdm import tqdm
import shutil
import torch
from src.datastructures.query import Query, ProcessQuery


class QueryCardinalityDataset(InMemoryDataset):

    def __init__(self, root, featurizer, post_processor=None, load_mappings=True,
                 to_load=None, file_list=None, transform=None, pre_transform=None, pre_filter=None):
        self.featurizer = featurizer
        self.post_processor = post_processor
        self.to_load = to_load
        self.load_mappings = load_mappings
        self._file_list = file_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        if os.path.exists(os.path.join(self.processed_dir, ""
                                                           "node_mappings.json")) and load_mappings:
            with open(os.path.join(self.processed_dir, "node_mappings.json"), "r") as fr:
                mappings = json.load(fr)
            self.data_mappings = mappings

    def get(self, idx: int):
        data = super().get(idx)
        if self.load_mappings:
            data.term_to_id = self.data_mappings[idx]

        return data

    def raw_file_names(self):
        # If explicit list is given, use it
        if self._file_list is not None:
            return self._file_list

        # Otherwise, discover all files in the raw directory
        return sorted(os.listdir(self.raw_dir))

    def processed_file_names(self):
        return [
            "processed_queries.pt"
        ]

    def process(self):
        raw_queries_list = []
        temp_dir = os.path.join(self.processed_dir, "temp_cache")
        os.makedirs(temp_dir, exist_ok=True)

        print(f"Processing {len(self.raw_file_names())} files")
        for file in self.raw_file_names():
            temp_file_path = os.path.join(temp_dir, f"{file}.pt")

            # Load from cache if it exists
            if os.path.exists(temp_file_path):
                queries = torch.load(temp_file_path)
            else:
                queries = []
                path = os.path.join(self.raw_dir, file)
                with open(path, 'r') as f:
                    raw_data = json.load(f)

                for i, data in tqdm(enumerate(raw_data), desc=file):
                    # Optimize loop to break early if the limit is reached
                    if self.to_load and i >= self.to_load:
                        break

                    tp_str, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])

                    # Temp fix for wrong generated queries during testing
                    if "type" not in data:
                        data["type"] = file

                    queries.append({
                        "query": data['query'],
                        "cardinality": data['y'],
                        "triple_patterns": tp_str,
                        "rdflib_patterns": tp_rdflib,
                        "type": data['type'],
                    })

                # Cache the processed raw queries for this file
                torch.save(queries, temp_file_path)

            raw_queries_list.append(queries)

        data_list = []
        for query_type in raw_queries_list:
            data_list.extend([self.featurizer(query) for query in query_type])

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.transform is not None:
            data_list = [self.transform(data) for data in data_list]

        print(f"Before post_processor: {len(data_list)} queries")

        if self.post_processor is not None:
            filtered_data = []
            for data in data_list:
                processed = self.post_processor(data)
                if processed is not None:
                    filtered_data.append(processed)
            data_list = filtered_data

        # Dictionary cannot be collated so it is saved separately
        node_mappings = []
        for data in data_list:
            node_mappings.append(data.id_to_term)
            del data.id_to_term

        if self.load_mappings:
            self.data_mappings = node_mappings

        mappings_path = os.path.join(self.processed_dir, 'node_mappings.json')
        with open(mappings_path, 'w') as fm:
            json.dump(node_mappings, fm, indent=2)

        print(f"Total: {len(data_list)} queries")
        self.save(data_list, self.processed_paths[0])

        # Remove the temporary cache directory after successful completion
        shutil.rmtree(temp_dir)

def get_size_data(data):
    return sum([v.element_size() * v.numel() for k, v in data if type(v) != str])


if __name__ == "__main__":
    pass