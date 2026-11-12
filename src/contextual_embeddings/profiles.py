"""Neighbour profiles: an entity described by what its neighbours look like, not by who it is.

    python -m src.contextual_embeddings.profiles [hydra overrides]     # builds profiles.directory

Index-time statistics, exact and inductive (any entity of the graph has one):
  type         the characteristic set of an entity: its distinct (predicate, direction) features
               (YAGO: 18,502 distinct types over 13M entities).
  histogram    per entity e and feature f = (predicate, direction), how many neighbours reached
               through f are of each type: e.g. "c' has, via incoming p, 40 neighbours of type T7".
The size of an entity's histogram is bounded by the number of types, not by its degree.

The contextualizer encodes a type from its feature set and an entity's profile as the
count-weighted sum over its histogram (model.py), so "a neighbour that has q1 AND q2" stays
visible after pooling -- the information constant-made hard zeros hinge on.

Files: types.npz (type_features: n_types x n_features multi-hot, entity_type), histograms.npz
(CSR over entities: ptr, feature, type, count).
"""
from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.random_query_generation.statistics_based_generation.generate import load_generation_config
from src.random_query_generation.statistics_based_generation.graph import load_graph


def build(graph, out_directory, chunk_edges=20_000_000):
    start = time.perf_counter()
    n, n_predicates = graph.n_entities, graph.n_predicates
    n_features = 2 * n_predicates                # feature f: p (outgoing) or n_predicates + p (incoming)
    s = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.out_ptr))
    p = graph.out_p.astype(np.int64)
    o = graph.out_o.astype(np.int64)

    # characteristic set (type) per entity
    keys = np.unique(np.concatenate([s * n_features + p, o * n_features + n_predicates + p]))
    entity, feature = keys // n_features, keys % n_features
    starts = np.r_[0, np.flatnonzero(np.diff(entity)) + 1]
    owners = entity[starts]
    signatures = [feature[a:b].astype(np.int16).tobytes() for a, b in zip(starts, np.r_[starts[1:], len(feature)])]
    type_ids, inverse = np.unique(np.array(signatures, dtype=object), return_inverse=True)
    entity_type = np.full(n, -1, dtype=np.int32)
    entity_type[owners] = inverse.astype(np.int32)
    type_features = np.zeros((len(type_ids), n_features), dtype=np.uint8)
    for t, signature in enumerate(type_ids):
        type_features[t, np.frombuffer(signature, dtype=np.int16)] = 1
    print(f"[profiles] {len(type_ids):,} types over {len(owners):,} entities ({time.perf_counter() - start:.0f}s)",
          flush=True)

    # histograms: (entity, feature, neighbour type) -> count, over ranges of entities of about `chunk_edges`
    # edges (their out- and in-edges from the CSR), so the peak is one range, not the whole graph (WatDiv:
    # 436M edge ends). Ranges ascend, so the concatenation is sorted by (entity, feature, type) as a whole.
    n_types = len(type_ids)
    del keys, entity, feature, signatures, type_ids, inverse, s, p, o
    edge_ends = np.cumsum(np.diff(graph.out_ptr) + np.diff(graph.in_ptr))     # edges of entities 0..e
    feature_parts, type_parts, count_parts = [], [], []
    hist_counts = np.zeros(n, dtype=np.int64)
    a = 0
    while a < n:
        done = edge_ends[a - 1] if a else 0
        b = min(n, max(a + 1, int(np.searchsorted(edge_ends, done + chunk_edges, "right"))))
        out_a, out_b, in_a, in_b = graph.out_ptr[a], graph.out_ptr[b], graph.in_ptr[a], graph.in_ptr[b]
        s = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(graph.out_ptr[a:b + 1]))
        o = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(graph.in_ptr[a:b + 1]))
        role = np.concatenate([s * n_features + graph.out_p[out_a:out_b],
                               o * n_features + n_predicates + graph.in_p[in_a:in_b]])
        neighbour_type = np.concatenate([entity_type[graph.out_o[out_a:out_b]], entity_type[graph.in_s[in_a:in_b]]])
        combined, counts = np.unique(role * n_types + neighbour_type, return_counts=True)
        role = combined // n_types
        hist_counts[a:b] = np.bincount(role // n_features - a, minlength=b - a)
        feature_parts.append((role % n_features).astype(np.int16))
        type_parts.append((combined % n_types).astype(np.int32))
        count_parts.append(counts.astype(np.int32))
        a = b
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(hist_counts, out=ptr[1:])
    out = Path(out_directory)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "types.npz", type_features=type_features, entity_type=entity_type)
    np.savez(out / "histograms.npz", ptr=ptr, feature=np.concatenate(feature_parts),
             type=np.concatenate(type_parts), count=np.concatenate(count_parts))
    print(f"[profiles] {ptr[-1]:,} histogram entries, max {np.diff(ptr).max():,} per entity "
          f"({time.perf_counter() - start:.0f}s) -> {out}", flush=True)


class ProfileTable:
    """Read access to the profiles (memory-mapped). rows(entity) -> (feature, type, count)."""

    def __init__(self, directory):
        directory = Path(directory)
        types = np.load(directory / "types.npz")
        self.type_features = types["type_features"]
        histograms = np.load(directory / "histograms.npz", mmap_mode="r")
        self.ptr, self.feature, self.type, self.count = (histograms[k] for k in ("ptr", "feature", "type", "count"))
        self.n_types, self.n_features = self.type_features.shape

    def rows(self, entity):
        if entity < 0 or entity + 1 >= len(self.ptr):
            return np.empty(0, np.int16), np.empty(0, np.int32), np.empty(0, np.int32)
        a, b = self.ptr[entity], self.ptr[entity + 1]
        return self.feature[a:b], self.type[a:b], self.count[a:b]


@hydra.main(version_base=None, config_path="../../experiments/experiment_configs/contextual_embeddings",
            config_name="contextualizer_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    gen = load_generation_config(cfg.corpus.generation_config)
    build(load_graph(gen.ntriples, gen.cache_directory), cfg.profiles.directory)


if __name__ == "__main__":
    main()
