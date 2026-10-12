"""predicate_edge_hll features with the rdf2vec part replaced by a frozen contextualizer.

The layout is identical to QueryToEdgePredicateGraphHll (same node / edge dimensions, same
statistics features), so the same estimator model configs apply. Only the embedding slot of
every feature vector changes:
  constant nodes   rdf2vec vector  -> contextual vector of the term in this query
  variable nodes   all-ones vector -> contextual vector of the variable in this query
  predicate edges  rdf2vec vector  -> the contextualizer's (pretrained) predicate vector
The contextualizer runs once per query (src/contextual_embeddings/inference.py).
"""
from src.query_featurizers.featurize_predicate_edges_tp_hll import QueryToEdgePredicateGraphHll


class QueryToEdgePredicateGraphHllContextual(QueryToEdgePredicateGraphHll):
    def __init__(self, *args, contextualizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if contextualizer is None:
            raise ValueError("a FrozenContextualizer is required")
        if contextualizer.dim != self.vector_size:
            raise ValueError(f"contextualizer dim {contextualizer.dim} != embedding size {self.vector_size}")
        self.contextualizer = contextualizer
        self._context = None

    def transform_undirected(self, json_query):
        self._context = self.contextualizer.embed_query(json_query['rdflib_patterns'])
        try:
            return super().transform_undirected(json_query)
        finally:
            self._context = None

    def term_to_embedding(self, term, term_to_id, is_predicate, log_features=True, hll_val=0.0):
        vector = super().term_to_embedding(term, term_to_id, is_predicate, log_features, hll_val)
        if self._context is not None:
            nodes, predicates = self._context
            replacement = predicates.get(term) if is_predicate else nodes.get(term)
            if replacement is not None:
                vector[-self.vector_size:] = replacement.tolist()
        return vector
