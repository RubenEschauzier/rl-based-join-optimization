import base64
import math
import pickle
import warnings

import numpy as np
import rdflib
import torch
from datasketch import HyperLogLog
from torch_geometric.data import Data


class QueryToEdgePredicateGraphHll:
    def __init__(self, entity_embeddings, env,
                 add_direction=False, term_occurrences=None,
                 predicate_multiplicities=None, loaded_hll_json=None,
                 tp_cardinalities=None):
        self.entity_embeddings = entity_embeddings
        self.term_occurrences = term_occurrences or {}
        self.predicate_multiplicities = predicate_multiplicities
        self.env = env
        self.tp_cardinalities = tp_cardinalities or {}

        self.vector_size = len(next(iter(self.entity_embeddings.values())))
        self.add_direction_feature = add_direction

        self.n_multiplicities = 0
        if self.predicate_multiplicities:
            self.n_multiplicities = len(next(iter(self.predicate_multiplicities.values())))

        self.embedding_stats = {"fail": 0, "succeed": 0}
        self.occurrences_stats = {"fail": 0, "succeed": 0}
        self.embedding_unique_terms_failed = set()

        self.hll_store = {}
        if loaded_hll_json:
            for pred, sketches in loaded_hll_json.items():
                self.hll_store[pred] = {
                    "domain": pickle.loads(base64.b64decode(sketches["domain"])),
                    "range": pickle.loads(base64.b64decode(sketches["range"]))
                }

    def get_pattern_count(self, s, p, o):
        """Retrieves exact pattern cardinality from cache or environment."""
        pattern_tuple = (s, p, o)
        if pattern_tuple in self.tp_cardinalities:
            return self.tp_cardinalities[pattern_tuple]

        pattern_str = f"{s.n3()} {p.n3()} {o.n3()} ."
        if pattern_str in self.tp_cardinalities:
            return self.tp_cardinalities[pattern_str]

        return self.env.cardinality_pattern(pattern_tuple)

    def transform(self, json_query):
        edge_index_src, edge_index_dst, edge_attr = [], [], []
        rdflib_patterns = json_query['rdflib_patterns']

        term_to_id = self.map_terms_to_ids(rdflib_patterns)

        for s, p, o in rdflib_patterns:
            edge_index_src.append(term_to_id[s])
            edge_index_dst.append(term_to_id[o])

            edge_emb = self.term_to_embedding(p, term_to_id, is_predicate=True)

            pattern_count = self.get_pattern_count(s, p, o)
            edge_emb.extend([math.log1p(pattern_count)])

            edge_attr.append(edge_emb)

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

        term_to_id_nodes = self.map_terms_to_ids([[row[0], row[2]] for row in rdflib_patterns])
        variable_to_id = self.map_variables_to_ids(rdflib_patterns)

        for s, p, o in rdflib_patterns:
            src_id, dst_id = term_to_id_nodes[s], term_to_id_nodes[o]

            edge_index_src.extend([src_id, dst_id])
            edge_index_dst.extend([dst_id, src_id])

            edge_emb = self.term_to_embedding(p, variable_to_id, is_predicate=True)

            pattern_count = self.get_pattern_count(s, p, o)
            smoothed_count = math.log1p(pattern_count)

            edge_attr.append(edge_emb + [smoothed_count, 1.0])
            edge_attr.append(edge_emb + [smoothed_count, -1.0])

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

    def compute_variable_hll_intersections(self, rdflib_tp):
        var_sketches = {}
        for s, p, o in rdflib_tp:
            p_str = f"<{str(p)}>"
            if p_str in self.hll_store:
                if isinstance(s, rdflib.term.Variable):
                    var_sketches.setdefault(s, []).append(self.hll_store[p_str]["domain"])
                if isinstance(o, rdflib.term.Variable):
                    var_sketches.setdefault(o, []).append(self.hll_store[p_str]["range"])

        var_features = {}
        for var, sketches in var_sketches.items():
            if len(sketches) < 2:
                var_features[var] = 0.0
            else:
                min_intersection = float('inf')
                for i in range(len(sketches)):
                    for j in range(i + 1, len(sketches)):
                        hll_a = sketches[i]
                        hll_b = sketches[j]

                        hll_union = HyperLogLog(p=12)
                        hll_union.merge(hll_a)
                        hll_union.merge(hll_b)

                        intersect = max(0, hll_a.count() + hll_b.count() - hll_union.count())
                        min_intersection = min(min_intersection, intersect)

                var_features[var] = math.log1p(min_intersection)
        return var_features

    def get_node_features(self, rdflib_tp, term_to_id):
        processed = set()
        node_features = []
        var_hll_features = self.compute_variable_hll_intersections(rdflib_tp)

        for tp in rdflib_tp:
            for term in (tp[0], tp[2]):
                if term not in processed:
                    hll_val = var_hll_features.get(term, 0.0) if isinstance(term, rdflib.term.Variable) else 0.0
                    node_features.append(self.term_to_embedding(term, term_to_id, is_predicate=False, hll_val=hll_val))
                    processed.add(term)
        return torch.tensor(node_features, dtype=torch.float)

    def term_to_embedding(self, term, term_to_id, is_predicate, log_features=True, hll_val=0.0):
        feature_vector = []

        if isinstance(term, rdflib.term.Variable):
            feature_vector.extend([float(term_to_id[term]), hll_val])

            if self.predicate_multiplicities:
                feature_vector.extend([0.0] * self.n_multiplicities)
                feature_vector.append(0.0)

            feature_vector.extend([1.0, 0.0, 0.0])
            feature_vector.extend([1.0] * self.vector_size)

            return feature_vector

        elif isinstance(term, (rdflib.term.URIRef, rdflib.term.Literal)):
            term_n3 = term.n3()

            if term_n3 in self.term_occurrences:
                self.occurrences_stats["succeed"] += 1
                term_count = self.term_occurrences[term_n3]
            else:
                self.occurrences_stats["fail"] += 1
                warnings.warn(f"Precomputed count does not exist: {term_n3}")
                term_count = self.env.cardinality_term(term_n3)

            count_val = math.log1p(term_count) if log_features else term_count
            feature_vector.extend([count_val, 1.0])

            if self.predicate_multiplicities:
                if is_predicate:
                    if term_n3 not in self.predicate_multiplicities:
                        raise ValueError(f"Missing predicate multiplicity: {term_n3}")
                    multiplicity = self.predicate_multiplicities[term_n3]
                    if log_features:
                        multiplicity = np.log1p(multiplicity).tolist()
                    feature_vector.extend(multiplicity)
                    feature_vector.append(1.0)
                else:
                    feature_vector.extend([0.0] * self.n_multiplicities)
                    feature_vector.append(0.0)

            term_str = str(term)
            if term_str in self.entity_embeddings:
                self.embedding_stats["succeed"] += 1
                feature_vector.extend([0.0, 0.0, 1.0])
                feature_vector.extend(self.entity_embeddings[term_str])
            else:
                if isinstance(term, rdflib.term.URIRef):
                    self.embedding_stats["fail"] += 1
                    self.embedding_unique_terms_failed.add(term_n3)
                feature_vector.extend([0.0, 1.0, 0.0])
                feature_vector.extend([0.0] * self.vector_size)

            return feature_vector

        else:
            raise NotImplementedError("Entities other than Variables, URIRefs, or Literals are not supported.")

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

    def get_unique_failed_terms(self):
        return self.embedding_unique_terms_failed