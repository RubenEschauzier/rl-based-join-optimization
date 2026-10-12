"""Term vocabulary of the contextualizer and the rdf2vec warm start of its token table.

Terms are IRIs without angle brackets (the key format of rdf2vec model.json and of
`str(rdflib.URIRef)`). Index 0 is the unknown token. The vocabulary holds every predicate of
the graph, every entity of the corpus and every term of the workload queries; terms with an
rdf2vec vector start from it, the others from random vectors of the same scale.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.random_query_generation.statistics_based_generation.graph import IRI_PREFIX

UNKNOWN = "<unk>"


def entity_term(entity_id):
    return f"{IRI_PREFIX}{int(entity_id)}"


def predicate_terms(graph_npz):
    """Graph predicate index -> term, read from the graph cache without loading the graph."""
    return [entity_term(n) for n in np.load(graph_npz)["predicate_iris"]]


def load_corpus(corpus_directory):
    instances = []
    for shard in sorted(Path(corpus_directory).glob("shard_*.pt")):
        instances.extend(torch.load(shard, weights_only=False))
    return instances


def build(instances, predicates, queries_directory, rdf2vec_path, out_directory, min_count=1, seed=0):
    """Writes vocabulary.json and init_vectors.pt to out_directory; returns (terms, vectors)."""
    counts = Counter(entity_term(e) for instance in instances for e in instance["nodes"] if e >= 0)
    terms = {t for t, c in counts.items() if c >= min_count} | set(predicates)
    for path in glob.glob(os.path.join(queries_directory, "*.json")):
        for record in json.load(open(path)):
            terms.update(record["x"])
    terms = [UNKNOWN] + sorted(terms)
    print(f"[vocabulary] {len(terms):,} terms; loading rdf2vec from {rdf2vec_path} ...", flush=True)
    rdf2vec = json.load(open(rdf2vec_path))
    dim = len(next(iter(rdf2vec.values())))
    scale = float(np.std(np.asarray(list(rdf2vec.values())[:20000], dtype=np.float32)))
    rng = np.random.default_rng(seed)
    vectors = rng.normal(0.0, scale, size=(len(terms), dim)).astype(np.float32)
    vectors[0] = 0.0
    found = 0
    for i, term in enumerate(terms):
        vector = rdf2vec.get(term)
        if vector is not None:
            vectors[i] = vector
            found += 1
    print(f"[vocabulary] {found:,}/{len(terms):,} terms start from rdf2vec", flush=True)
    out = Path(out_directory)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vocabulary.json", "w") as f:
        json.dump(terms, f)
    vectors = torch.from_numpy(vectors)
    torch.save(vectors, out / "init_vectors.pt")
    return terms, vectors


def load(directory):
    directory = Path(directory)
    with open(directory / "vocabulary.json") as f:
        terms = json.load(f)
    return terms, torch.load(directory / "init_vectors.pt")
