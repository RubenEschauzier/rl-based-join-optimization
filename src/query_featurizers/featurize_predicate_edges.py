import rdflib
import torch
import torch_geometric
from torch_geometric.data import Data

# TODO Make it undirected edge graph (Also for the edge labeled graph as this should work better).
# TODO Add -1 to end of edge feature if its a feature not in the original query graph 1 else. (So the extra edges added
# TODO to make it an undirected graph
class QueryToEdgePredicateGraph:
    def __init__(self, entity_embeddings, env, add_direction=False, term_occurrences=None):
        self.entity_embeddings = entity_embeddings
        self.term_occurrences = term_occurrences
        self.env = env
        self.vector_size = len(entity_embeddings[next(iter(entity_embeddings))])
        self.add_direction_feature = add_direction

    def transform(self, json_query):
        edge_index = [[], []]
        edge_attr = []
        term_to_id = self.map_terms_to_ids(json_query['rdflib_patterns'])
        for tp in json_query['rdflib_patterns']:
            edge_index[0].append(term_to_id[tp[0]])
            edge_index[1].append(term_to_id[tp[2]])
            edge_attr.append(self.term_to_embedding(tp[1], term_to_id))

        node_features = self.get_node_features(json_query['rdflib_patterns'], term_to_id)
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch_geometric.EdgeIndex(edge_index)
        y = torch.tensor(json_query['cardinality'])

        data_query = Data(x=node_features, edge_index=edge_index,
                          edge_attr=edge_attr, y=y,
                          query=json_query['query'],
                          type=json_query['type'])
        return data_query

    def transform_undirected(self, json_query):
        edge_index = [[], []]
        edge_attr = []
        # Used to create edge_index
        term_to_id_nodes = self.map_terms_to_ids(
            [[row[0], row[2]] for row in json_query['rdflib_patterns']]
        )
        # Used to assign index value to node / edge representations of variables
        variable_to_id = self.map_variables_to_ids(json_query['rdflib_patterns'])
        for tp in json_query['rdflib_patterns']:
            edge_index[0].append(term_to_id_nodes[tp[0]])
            edge_index[0].append(term_to_id_nodes[tp[2]])
            edge_index[1].append(term_to_id_nodes[tp[2]])
            edge_index[1].append(term_to_id_nodes[tp[0]])
            edge_emb = self.term_to_embedding(tp[1], variable_to_id)
            edge_attr.append(edge_emb + [1])
            edge_attr.append(edge_emb + [-1])

        node_features = self.get_node_features(json_query['rdflib_patterns'], variable_to_id)
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch_geometric.EdgeIndex(edge_index)
        y = torch.tensor(json_query['cardinality'])

        data_query = Data(x=node_features, edge_index=edge_index,
                          edge_attr=edge_attr, y=y,
                          query=json_query['query'],
                          type=json_query['type'])
        return data_query

    def get_node_features(self, rdflib_tp, term_to_id):
        processed = set()
        node_features = []
        for tp in rdflib_tp:
            for term in [tp[0], tp[2]]:
                if term not in processed:
                    node_features.append(self.term_to_embedding(term, term_to_id))
                    processed.add(term)
        return torch.tensor(node_features)

    def term_to_embedding(self, term, term_to_id):
        if type(term) == rdflib.term.Variable:
            var_embedding = [term_to_id[term]]
            var_embedding.extend([1] * self.vector_size)
            return var_embedding
        if type(term) == rdflib.term.URIRef or type(term) == rdflib.term.Literal:
            if self.term_occurrences and term.n3() in self.term_occurrences:
                entity_embedding = [self.term_occurrences[term.n3()]]
            else:
                entity_embedding = [self.get_term_count(term.n3())]
            if self.entity_embeddings.get(str(term)):
                entity_embedding.extend(self.entity_embeddings.get(str(term)))
            else:
                entity_embedding.extend([0] * self.vector_size)
            return entity_embedding
        else:
            raise NotImplementedError("Entities other than Variables, URIRefs, or Literals are not yet "
                                      "supported")

    def get_term_count(self, term):
        return self.env.cardinality_term(term)

    def map_terms_to_ids(self, rdflib_tp):
        term_to_id = {}
        term_id = 0
        for tp in rdflib_tp:
            for entity in tp:
                if type(entity) == rdflib.term.BNode:
                    raise NotImplementedError
                if entity not in term_to_id:
                    term_to_id[entity] = term_id
                    term_id += 1
        return term_to_id

    def map_variables_to_ids(self, rdflib_tp):
        var_to_id = {}
        var_id = 0
        for tp in rdflib_tp:
            for entity in tp:
                if type(entity) == rdflib.term.Variable and entity not in var_to_id:
                    var_to_id[entity] = var_id
                    var_id += 1
        return var_to_id


class QueryToTermGraph:
    def __init__(self, entity_embeddings, env):
        self.entity_embeddings = entity_embeddings
        self.env = env
        self.vector_size = len(entity_embeddings[next(iter(entity_embeddings))])

    def transform(self, json_query):
        pass

