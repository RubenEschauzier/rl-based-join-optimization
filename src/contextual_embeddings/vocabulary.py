"""Term vocabulary of the contextualizer and the rdf2vec warm start of its token table.

Terms are IRIs without angle brackets (the key format of rdf2vec model.json and of
`str(rdflib.URIRef)`) and literals in quoted N3 form ("MOLLIE", as rdflib's Literal.n3() and the
generator's `x` lists write them). Graph entities and predicates are named through GraphTerms: the
number of <http://example.com/N> for anonymised YAGO, the graph cache's vocabulary for a
dictionary-encoded dump (WatDiv). Index 0 is the unknown token. The vocabulary holds every predicate of
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

from src.random_query_generation.statistics_based_generation.graph import IRI_PREFIX, Terms

UNKNOWN = "<unk>"


def entity_term(entity_id):
    return f"{IRI_PREFIX}{int(entity_id)}"


def term_key(raw):
    """Vocabulary key of an N-Triples term: <iri> -> iri, a literal as it is."""
    return raw[1:-1] if raw.startswith("<") else raw


class GraphTerms:
    """Vocabulary keys of a generator cache's entities and predicates, without loading the graph."""

    def __init__(self, cache_directory):
        directory = Path(cache_directory)
        self.directory = str(directory)
        self.predicate_iris = np.load(directory / "graph.npz")["predicate_iris"]
        self.entities = self.predicates = None
        if (directory / "vocabulary.npz").exists():
            v = np.load(directory / "vocabulary.npz")
            self.entities = Terms(v["entity_data"], v["entity_offsets"])
            self.predicates = Terms(v["predicate_data"], v["predicate_offsets"])

    def entity(self, entity_id):
        if self.entities is None:
            return entity_term(entity_id)
        return term_key(self.entities[int(entity_id)])

    def predicate_terms(self):
        """Graph predicate index -> key."""
        if self.predicates is None:
            return [entity_term(n) for n in self.predicate_iris]
        return [term_key(self.predicates[int(n)]) for n in self.predicate_iris]

    def entity_ids(self, keys):
        """{key: graph entity id} for the given keys that are graph entities."""
        if self.entities is None:
            out = {}
            for key in keys:
                suffix = key[len(IRI_PREFIX):] if key.startswith(IRI_PREFIX) else ""
                if suffix.isdigit():
                    out[key] = int(suffix)
            return out
        from src.query_featurizers.row_store import term_index
        raw = {(key if key.startswith('"') else f"<{key}>"): key for key in keys}
        return {raw[t]: i for t, i in term_index(self.entities, set(raw)).items()}


def predicate_terms(graph_npz):
    """Graph predicate index -> term, read from the graph cache without loading the graph."""
    return GraphTerms(Path(graph_npz).parent).predicate_terms()


def load_corpus(corpus_directory):
    instances = []
    for shard in sorted(Path(corpus_directory).glob("shard_*.pt")):
        instances.extend(torch.load(shard, weights_only=False))
    return instances


def build(instances, predicates, queries_directory, rdf2vec_path, out_directory, min_count=1, seed=0,
          entity_name=entity_term):
    """Writes vocabulary.json and init_vectors.pt to out_directory; returns (terms, vectors).
    entity_name: graph entity id -> key (GraphTerms.entity)."""
    counts = Counter(entity_name(e) for instance in instances for e in instance["nodes"] if e >= 0)
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
