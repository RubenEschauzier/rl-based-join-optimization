import json

from tqdm import tqdm
import rdflib.term
import torch

from src.datastructures.query import Query


class FeaturizeQueriesRdf2Vec:
    def __init__(self, query_env, rdf2vec_vectors: dict):
        self.env = query_env
        self.vectors = rdf2vec_vectors
        self.vector_size = len(list(self.vectors.values())[0])
        pass

    def run(self, queries: [Query], disable_progress_bar=False):
        estimates = self.get_cardinality_estimates_query_set(queries, disable_progress_bar=disable_progress_bar)
        for i in tqdm(range(len(queries)), disable=disable_progress_bar):
            featurized = self.featurize_query(queries[i], estimates[i])
            queries[i].set_features(torch.Tensor(featurized))

        return queries

    def featurize_query(self, query: Query, cardinality_estimates_tp: list[int]):
        var_to_id = self.map_variables_to_ids(query)
        query_embedding = []
        for i in range(len(query.rdflib_tp)):
            tp_emb = [cardinality_estimates_tp[i]]
            for entity in query.rdflib_tp[i]:
                # Embedding strategy from
                # "Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks"
                if type(entity) == rdflib.term.Variable:
                    var_embedding = [1] * self.vector_size
                    var_embedding[0] = var_to_id.get(entity)
                    tp_emb.extend(var_embedding)
                    continue

                if type(entity) == rdflib.term.URIRef or type(entity) == rdflib.term.Literal:
                    if self.vectors.get(str(entity)):
                        entity_embedding = self.vectors.get(str(entity))
                    else:
                        entity_embedding = [0] * self.vector_size
                    tp_emb.extend(entity_embedding)
                    continue

                else:
                    raise NotImplementedError("Entities other than Variables, URIRefs, or Literals are not yet "
                                              "supported")
            query_embedding.append(tp_emb)
        return torch.Tensor(query_embedding)

    def get_cardinality_estimates_query_set(self, queries, disable_progress_bar):
        estimates = []
        for query in tqdm(queries, disable=disable_progress_bar):
            estimates.append(self.get_cardinality_estimate(query))
        return estimates

    def get_cardinality_estimate(self, query: Query):
        estimates_tp = []
        for tp_string in query.string_tp:
            estimates_tp.append(int(self.env.cardinality_triple_pattern(tp_string)))
        return estimates_tp

    @staticmethod
    def map_variables_to_ids(query: Query):
        var_to_id = {}
        var_id = 0
        for tp in query.rdflib_tp:
            for entity in tp:
                if type(entity) == rdflib.term.Variable and entity not in var_to_id:
                    var_to_id[entity] = var_id
                    var_id += 1
        return var_to_id

    @staticmethod
    def load_vectors(location):
        with open(location, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def convert_text_to_json(text_file_location, output_location):
        vector_dict = {}
        with open(text_file_location, 'r') as f:
            data = f.read().strip().split('\n')

        for entity in data:
            split = entity.split('[sep]')
            vector = [float(x) for x in split[1].strip().split(' ')]
            vector_dict[split[0]] = vector

        with open(output_location, 'w', encoding='utf-8') as f:
            json.dump(vector_dict, f, ensure_ascii=False, indent=4)
