import rdflib
import torch_geometric
from rdflib import Variable
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# TODO: Run this again and see what it does to performance to have undirected graph
class QueryToEdgeLabeledGraph:
    def __init__(self, entity_embeddings, env, tp_cardinalities = None):
        self.entity_embeddings = entity_embeddings
        self.tp_cardinalities = tp_cardinalities
        self.env = env
        self.vector_size = len(entity_embeddings[next(iter(entity_embeddings))])

    def transform(self, json_query):
        tp_features = self.encode_tp_rdf2vec(json_query['triple_patterns'], json_query['rdflib_patterns'])
        edge_index, edge_features = self.to_edge_index(json_query['rdflib_patterns'])
        edge_index, y, edge_features, tp_features = (torch_geometric.EdgeIndex(edge_index), torch.tensor(json_query['cardinality']),
                                                     torch.tensor(edge_features, dtype=torch.float32),
                                                     torch.tensor(tp_features))
        edge_index_undirected, edge_features_undirected = to_undirected(edge_index, edge_features)
        data_query = Data(x=tp_features, edge_index=edge_index_undirected,
                          edge_attr=edge_features_undirected, y=y,
                          query=json_query['query'],
                          triple_patterns=json_query['triple_patterns'],
                          type=json_query['type'])
        return data_query

    def to_edge_index(self, rdflib_tps):
        edge_index = [[], []]
        edge_features = []
        for i in range(len(rdflib_tps)):
            outer_pattern = rdflib_tps[i]
            for j in range(i+1, len(rdflib_tps)):

                inner_pattern = rdflib_tps[j]
                if self.is_connected(outer_pattern, inner_pattern):
                    edge_feature = self.get_edge_feature_connection_type(outer_pattern, inner_pattern)
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_features.append(edge_feature)

        return edge_index, edge_features

    def is_connected(self, rdflib_tp1, rdflib_tp2):
        vars_tp1 = self.get_variables_tp(rdflib_tp1)
        vars_tp2 = self.get_variables_tp(rdflib_tp2)
        return not vars_tp1.isdisjoint(vars_tp2)

    def get_variables_tp(self, rdflib_tp):
        # map variables expects list of triple patterns, we use it to get variables of single tp
        return set([var.n3() for var in self.map_variables_to_ids([rdflib_tp]).keys()])

    def encode_tp_rdf2vec(self, string_tp, rdflib_tp):
        var_to_id = self.map_variables_to_ids(rdflib_tp)
        tp_embeddings = []
        for i in range(len(rdflib_tp)):
            if self.tp_cardinalities and self.tp_cardinalities[string_tp[i]]:
                tp_emb = [int(self.tp_cardinalities[string_tp[i]])]
            else:
                tp_emb = [int(self.env.cardinality_triple_pattern(string_tp[i]))]
            for entity in rdflib_tp[i]:
                # Embedding strategy from
                # "Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks"
                if type(entity) == rdflib.term.Variable:
                    var_embedding = [1] * self.vector_size
                    var_embedding[0] = var_to_id.get(entity)
                    tp_emb.extend(var_embedding)
                    continue

                if type(entity) == rdflib.term.URIRef or type(entity) == rdflib.term.Literal:
                    if self.entity_embeddings.get(str(entity)):
                        entity_embedding = self.entity_embeddings.get(str(entity))
                    else:
                        # print("Unknown entity")
                        # print(entity)
                        entity_embedding = [0] * self.vector_size
                    tp_emb.extend(entity_embedding)
                    continue

                else:
                    raise NotImplementedError("Entities other than Variables, URIRefs, or Literals are not yet "
                                              "supported")
            tp_embeddings.append(tp_emb)
        return tp_embeddings

    @staticmethod
    def get_edge_feature_connection_type(rdflib_tp1, rdflib_tp2):
        edge_feature = []
        for i in range(len(rdflib_tp1)):
            for j in range(len(rdflib_tp2)):
                if (type(rdflib_tp1[i]) == Variable and type(rdflib_tp2[j]) == Variable
                        and rdflib_tp1[i] == rdflib_tp2[j]):
                   edge_feature.append(1)
                else:
                    edge_feature.append(0)
        return edge_feature

    @staticmethod
    def map_variables_to_ids(rdflib_tp):
        var_to_id = {}
        var_id = 0
        for tp in rdflib_tp:
            for entity in tp:
                if type(entity) == rdflib.term.Variable and entity not in var_to_id:
                    var_to_id[entity] = var_id
                    var_id += 1
        return var_to_id