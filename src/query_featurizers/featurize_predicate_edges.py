import math
import warnings

import numpy as np
import rdflib
import torch
import torch_geometric
from torch_geometric.data import Data


class QueryToEdgePredicateGraph:
    def __init__(self, entity_embeddings, env,
                 add_direction=False, term_occurrences=None,
                 predicate_multiplicities=None):
        self.entity_embeddings = entity_embeddings
        self.term_occurrences = term_occurrences or {}
        self.predicate_multiplicities = predicate_multiplicities
        self.env = env

        self.vector_size = len(next(iter(self.entity_embeddings.values())))
        self.add_direction_feature = add_direction

        self.n_multiplicities = 0
        if self.predicate_multiplicities:
            self.n_multiplicities = len(next(iter(self.predicate_multiplicities.values())))

        self.embedding_stats = {"fail": 0, "succeed": 0}
        self.occurrences_stats = {"fail": 0, "succeed": 0}

    def transform(self, json_query):
        edge_index_src, edge_index_dst, edge_attr = [], [], []
        rdflib_patterns = json_query['rdflib_patterns']

        term_to_id = self.map_terms_to_ids(rdflib_patterns)

        for s, p, o in rdflib_patterns:
            edge_index_src.append(term_to_id[s])
            edge_index_dst.append(term_to_id[o])
            edge_attr.append(self.term_to_embedding(p, term_to_id, is_predicate=True))

        node_features = self.get_node_features(rdflib_patterns, term_to_id)

        edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(json_query['cardinality'], dtype=torch.float)

        data_query = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            query=json_query['query'],
            triple_patterns=json_query['triple_patterns'],
            type=json_query['type']
        )
        data_query.id_to_term = {key.n3(): value for key, value in term_to_id.items()}

        return data_query

    def transform_undirected(self, json_query):
        edge_index_src, edge_index_dst, edge_attr = [], [], []
        rdflib_patterns = json_query['rdflib_patterns']

        # Maps unique nodes to structural IDs for edge_index construction
        term_to_id_nodes = self.map_terms_to_ids([[row[0], row[2]] for row in rdflib_patterns])

        # Maps variables to feature IDs for the embedding function
        variable_to_id = self.map_variables_to_ids(rdflib_patterns)

        for s, p, o in rdflib_patterns:
            src_id, dst_id = term_to_id_nodes[s], term_to_id_nodes[o]

            edge_index_src.extend([src_id, dst_id])
            edge_index_dst.extend([dst_id, src_id])

            edge_emb = self.term_to_embedding(p, variable_to_id, is_predicate=True)
            edge_attr.append(edge_emb + [1.0])
            edge_attr.append(edge_emb + [-1.0])

        node_features = self.get_node_features(rdflib_patterns, variable_to_id)

        edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(json_query['cardinality'], dtype=torch.float)

        data_query = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            query=json_query['query'],
            triple_patterns=json_query['triple_patterns'],
            type=json_query['type']
        )
        data_query.id_to_term = {key.n3(): value for key, value in term_to_id_nodes.items()}

        return data_query

    def get_node_features(self, rdflib_tp, term_to_id):
        processed = set()
        node_features = []
        for tp in rdflib_tp:
            for term in (tp[0], tp[2]):
                if term not in processed:
                    node_features.append(self.term_to_embedding(term, term_to_id, is_predicate=False))
                    processed.add(term)
        return torch.tensor(node_features, dtype=torch.float)

    def term_to_embedding(self, term, term_to_id, is_predicate, log_features=True):
        feature_vector = []

        if isinstance(term, rdflib.term.Variable):
            # Count/ID Segment
            feature_vector.extend([float(term_to_id[term]), 0.0])

            # Multiplicity Segment
            if self.predicate_multiplicities:
                feature_vector.extend([0.0] * self.n_multiplicities)
                feature_vector.append(0.0)  # Mask

            # Embedding Segment
            # Mask indicating it's a variable
            feature_vector.extend([1.0] * self.vector_size)
            feature_vector.extend([1.0, 0.0, 0.0])

            return feature_vector

        elif isinstance(term, (rdflib.term.URIRef, rdflib.term.Literal)):
            term_n3 = term.n3()

            # Count/ID Segment
            if term_n3 in self.term_occurrences:
                self.occurrences_stats["succeed"] += 1
                term_count = self.term_occurrences[term_n3]
            else:
                self.occurrences_stats["fail"] += 1
                warnings.warn(f"Precomputed count does not exist: {term_n3}")
                term_count = self.get_term_count(term_n3)
                print(f"Actual term count: {term_count}")

            count_val = math.log(term_count) if log_features else term_count
            feature_vector.extend([count_val, 1.0])

            # Multiplicity Segment
            if self.predicate_multiplicities:
                if is_predicate:
                    if term_n3 not in self.predicate_multiplicities:
                        raise ValueError(f"Missing predicate multiplicity: {term_n3}")
                    multiplicity = self.predicate_multiplicities[term_n3]
                    if log_features:
                        multiplicity = np.log1p(multiplicity).tolist()
                    feature_vector.extend(multiplicity)
                    feature_vector.append(1.0)  # Mask
                else:
                    feature_vector.extend([0.0] * self.n_multiplicities)
                    feature_vector.append(0.0)  # Mask

            # Embedding Segment
            term_str = str(term)
            if term_str in self.entity_embeddings:
                self.embedding_stats["succeed"] += 1
                feature_vector.extend(self.entity_embeddings[term_str])
                # Mask: Valid Embedding
                feature_vector.extend([0.0, 0.0, 1.0])
            else:
                if isinstance(term, rdflib.term.URIRef):
                    self.embedding_stats["fail"] += 1
                    warnings.warn(f"Embedding for URI does not exist: {term_n3}")
                # Mask: Missing Embedding
                feature_vector.extend([0.0, 1.0, 0.0])
                feature_vector.extend([0.0] * self.vector_size)

            return feature_vector

        else:
            raise NotImplementedError("Entities other than Variables, URIRefs, or Literals are not supported.")

    def get_term_count(self, term):
        return self.env.cardinality_term(term)

    def map_terms_to_ids(self, rdflib_tp):
        term_to_id = {}
        for tp in rdflib_tp:
            for entity in tp:
                if isinstance(entity, rdflib.term.BNode):
                    raise NotImplementedError("BNodes are not supported.")
                if entity not in term_to_id:
                    term_to_id[entity] = len(term_to_id)
        return term_to_id

    def map_variables_to_ids(self, rdflib_tp):
        var_to_id = {}
        for tp in rdflib_tp:
            for entity in tp:
                if isinstance(entity, rdflib.term.Variable) and entity not in var_to_id:
                    var_to_id[entity] = len(var_to_id)
        return var_to_id


class QueryToTermGraph:
    def __init__(self, entity_embeddings, env):
        self.entity_embeddings = entity_embeddings
        self.env = env
        self.vector_size = len(next(iter(self.entity_embeddings.values())))

    def transform(self, json_query):
        pass