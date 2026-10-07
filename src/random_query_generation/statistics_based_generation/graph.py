"""The data graph in memory: every triple as integers, indexed both ways.

The anonymised YAGO dump has only IRIs of the form <http://example.com/<int>>, so a triple is
three integers. Entities keep their numeric id; predicates are renumbered 0..P-1. Two CSR
indexes over entity ids:

    out[s]  the edges (p, o) of subject s, sorted by (p, o)
    in[o]   the edges (p, s) into object o, sorted by (p, s)

so the number of matches of any triple pattern with at most one constant, and the neighbours
of an entity, are binary searches. ~1 GB for the 58M YAGO triples; cached as .npz.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IRI_PREFIX = "http://example.com/"
_NUMBER = re.compile(rb"<http://example\.com/(\d+)>")


def parse_ntriples(path, chunk_bytes=1 << 28):
    """(s, p, o) int64 arrays from an N-Triples file of numeric example.com IRIs."""
    parts = []
    with open(path, "rb") as f:
        rest = b""
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            chunk = rest + chunk
            cut = chunk.rfind(b"\n") + 1
            rest, chunk = chunk[cut:], chunk[:cut]
            parts.append(np.array(_NUMBER.findall(chunk), dtype=np.int64))
        if rest.strip():
            parts.append(np.array(_NUMBER.findall(rest), dtype=np.int64))
    flat = np.concatenate(parts)
    if len(flat) % 3:
        raise ValueError(f"{path}: {len(flat)} terms is not a multiple of 3; not all terms are numeric IRIs")
    triples = flat.reshape(-1, 3)
    return triples[:, 0], triples[:, 1], triples[:, 2]


def _csr(keys, values_a, values_b, n_keys):
    order = np.lexsort((values_b, values_a, keys))
    counts = np.bincount(keys, minlength=n_keys)
    indptr = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, values_a[order].astype(np.int32), values_b[order].astype(np.int32)


@dataclass
class DataGraph:
    predicate_iris: np.ndarray     # (P,) original predicate number, index = predicate id
    out_ptr: np.ndarray            # (E + 1,)
    out_p: np.ndarray              # (T,) predicate ids, per subject sorted by (p, o)
    out_o: np.ndarray              # (T,)
    in_ptr: np.ndarray
    in_p: np.ndarray
    in_s: np.ndarray

    @property
    def n_entities(self):
        return len(self.out_ptr) - 1

    @property
    def n_triples(self):
        return len(self.out_p)

    @property
    def n_predicates(self):
        return len(self.predicate_iris)

    # --- neighbours ------------------------------------------------------------------------
    def out_edges(self, s):
        a, b = self.out_ptr[s], self.out_ptr[s + 1]
        return self.out_p[a:b], self.out_o[a:b]

    def in_edges(self, o):
        a, b = self.in_ptr[o], self.in_ptr[o + 1]
        return self.in_p[a:b], self.in_s[a:b]

    def out_degree(self, s):
        return int(self.out_ptr[s + 1] - self.out_ptr[s])

    def in_degree(self, o):
        return int(self.in_ptr[o + 1] - self.in_ptr[o])

    # --- pattern sizes ---------------------------------------------------------------------
    @staticmethod
    def _range(predicates, p):
        return np.searchsorted(predicates, p, "left"), np.searchsorted(predicates, p, "right")

    def objects_of(self, s, p):
        """Objects o with (s, p, o)."""
        predicates, objects = self.out_edges(s)
        a, b = self._range(predicates, p)
        return objects[a:b]

    def subjects_of(self, o, p):
        """Subjects s with (s, p, o)."""
        predicates, subjects = self.in_edges(o)
        a, b = self._range(predicates, p)
        return subjects[a:b]

    def count_bound_subject(self, s, p):
        predicates, _ = self.out_edges(s)
        a, b = self._range(predicates, p)
        return int(b - a)

    def count_bound_object(self, o, p):
        predicates, _ = self.in_edges(o)
        a, b = self._range(predicates, p)
        return int(b - a)

    def has_triple(self, s, p, o):
        objects = self.objects_of(s, p)
        i = np.searchsorted(objects, o)
        return bool(i < len(objects) and objects[i] == o)

    def iri(self, entity):
        return f"<{IRI_PREFIX}{int(entity)}>"

    def predicate_iri(self, predicate):
        return f"<{IRI_PREFIX}{int(self.predicate_iris[predicate])}>"

    # --- construction ----------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, s, p, o):
        predicate_iris, p = np.unique(p, return_inverse=True)
        n_entities = int(max(s.max(), o.max())) + 1
        out_ptr, out_p, out_o = _csr(s, p, o, n_entities)
        in_ptr, in_p, in_s = _csr(o, p, s, n_entities)
        return cls(predicate_iris, out_ptr, out_p, out_o, in_ptr, in_p, in_s)

    def save(self, path):
        np.savez(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def load_graph(ntriples_path, cache_dir):
    """The graph of `ntriples_path`, built once and cached in cache_dir/graph.npz."""
    cache = Path(cache_dir) / "graph.npz"
    if cache.exists():
        return DataGraph.load(cache)
    start = time.perf_counter()
    print(f"Parsing {ntriples_path} ...")
    graph = DataGraph.from_arrays(*parse_ntriples(ntriples_path))
    os.makedirs(cache_dir, exist_ok=True)
    graph.save(cache)
    print(f"Graph: {graph.n_triples:,} triples, {graph.n_predicates} predicates, "
          f"{graph.n_entities:,} entity ids ({time.perf_counter() - start:.0f}s) -> {cache}")
    return graph
