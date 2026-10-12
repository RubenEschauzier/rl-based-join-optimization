"""Contextual embeddings trained on counterfactual (near-miss) queries.

Learned cardinality estimators that read static term embeddings (rdf2vec) have a blind spot:
queries in which every pair of patterns has matches but the whole query has none (hard
zeros). A static vector of a constant cannot say that it does not fit *together with* the
other patterns of a particular query.

This package learns a small contextualizer that does:
  corpus.py      query-shaped instances (star, path, tree, cycle, ...) sampled from the data
                 graph, with constants and variables like queries. Positives are non-empty by
                 construction; negatives replace one arm so the instance becomes empty -- near
                 misses built like the hard zeros of the workload generator (every pair still
                 matches) plus easy random replacements.
  vocabulary.py  term vocabulary, warm-started from the rdf2vec vectors.
  model.py       token table (the rdf2vec vectors, trained further) + a small relational GNN
                 over the instance graph; linear pretraining heads predict which arm does not
                 belong and whether the instance is empty.
  pretrain.py    pretraining, validated on instances with held-out constants.
  inference.py   the frozen contextualizer: one forward pass per query gives a contextual
                 vector per term. Its output replaces the rdf2vec part of the estimator's input
                 features (src/query_featurizers/featurize_predicate_edges_tp_hll_contextual.py),
                 the pretraining heads are discarded.

Config: experiments/experiment_configs/contextual_embeddings/contextualizer_mixed_yago_v2.yaml
"""
