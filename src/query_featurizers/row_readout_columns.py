"""Adds the RowReadout input columns (src/models/model_layers/row_readout.py) to a featurised dataset.

    python -m src.query_featurizers.row_readout_columns SOURCE_ROOT TARGET_ROOT [STORE [GRAPH_DIR]]

SOURCE_ROOT is a processed QueryCardinalityDataset root; TARGET_ROOT gets a copy with
    x[:, :k]         the node's entity id, -1 for variables: one column, or two (id >> 12, id & 4095) when the
                     store's graph has 2^24 entities or more (row_readout.store_layout; float32 is exact to 2^24)
    edge_attr[:, 0]  the source node's feature in the pattern: predicate id p if the source is the pattern's
                     subject (edge 2i), P + p if its object (edge 2i + 1); predicate ids index STORE/refs.npz's
                     predicate_iris (-1 for a predicate outside it), P the store's predicates
prepended, so columns counted from the end (direction, pattern count) keep their meaning. raw/ is linked to the
source's, node_mappings.json copied. Entity and predicate ids: the number of <http://example.com/N> (anonymised
YAGO), or, with GRAPH_DIR (a dictionary-encoded generator cache, e.g. WatDiv's), the term's id in its vocabulary.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PREFIX = "<http://example.com/"


def numeric_ids(iris):
    """(entity id of a term or -1, store predicate index of a term or -1) for anonymised YAGO."""
    lookup = {int(v): k for k, v in enumerate(iris.tolist())}
    entity = lambda t: int(t[len(PREFIX):-1]) if t.startswith(PREFIX) else -1
    predicate = lambda t: lookup.get(int(t[len(PREFIX):-1]), -1) if t.startswith(PREFIX) else -1
    return entity, predicate


def vocabulary_ids(graph_dir, iris, terms):
    """The same through a generator cache's vocabulary (graph.py Terms: <iri>, "literal")."""
    from src.query_featurizers.row_store import term_index
    from src.random_query_generation.statistics_based_generation.graph import load_graph
    graph = load_graph(None, graph_dir)
    index = term_index(graph.entity_terms, set(terms))
    lookup = {graph.predicate_terms[int(v)]: k for k, v in enumerate(iris.tolist())}
    return (lambda t: index.get(t, -1)), (lambda t: lookup.get(t, -1))


def columns(processed, mappings, entity, predicate, n_predicates):
    """(entity id per node, source feature per edge) for every graph, concatenated in graph order."""
    data, slices, _ = processed
    xs, es = slices["x"].tolist(), slices["edge_attr"].tolist()
    edge_index = data["edge_index"]
    ent = np.full(xs[-1], -1, dtype=np.int64)
    feat = np.full(es[-1], -1, dtype=np.float32)
    for i, (patterns, term_to_id) in enumerate(zip(data["triple_patterns"], mappings)):
        x0, e0 = xs[i], es[i]
        for term, node in term_to_id.items():
            if not term.startswith("?"):
                ent[x0 + int(node)] = entity(term)
        for j, tp in enumerate(patterns):
            s, p = tp.split(" ")[:2]                      # subjects and predicates never contain spaces
            assert int(edge_index[0, e0 + 2 * j]) == int(term_to_id[s]), f"graph {i}: edge order differs from its patterns"
            pid = predicate(p)
            if pid >= 0:
                feat[e0 + 2 * j], feat[e0 + 2 * j + 1] = pid, n_predicates + pid
    return ent, feat


def id_columns(ent, k):
    """Entity ids as k float32 columns (row_readout.entity_ids reads them back)."""
    from src.models.model_layers.row_readout import ID_SPLIT
    if k == 1:
        assert ent.max(initial=-1) < 2 ** 24, "ids beyond float32's exact range need two columns"
        return torch.from_numpy(ent.astype(np.float32))[:, None]
    hi = np.where(ent >= 0, ent >> ID_SPLIT, -1)
    lo = np.where(ent >= 0, ent & ((1 << ID_SPLIT) - 1), -1)
    return torch.from_numpy(np.stack([hi, lo], 1).astype(np.float32))


def main(source, target, store="data/constant_embeddings/yago", graph_dir=None):
    import json
    from src.models.model_layers.row_readout import store_layout
    source, target = Path(source), Path(target)
    iris = np.load(Path(store) / "refs.npz")["predicate_iris"]
    n_predicates, k = store_layout(store)
    processed = torch.load(source / "processed/processed_queries.pt", weights_only=False)
    mappings = json.load(open(source / "processed/node_mappings.json"))
    if graph_dir is None:
        entity, predicate = numeric_ids(iris)
    else:
        terms = {t for m in mappings for t in m if not t.startswith("?")}
        entity, predicate = vocabulary_ids(graph_dir, iris, terms)
    ent, feat = columns(processed, mappings, entity, predicate, n_predicates)
    data, slices, cls = processed
    data["x"] = torch.cat([id_columns(ent, k), data["x"]], 1)
    data["edge_attr"] = torch.cat([torch.from_numpy(feat)[:, None], data["edge_attr"]], 1)
    (target / "processed").mkdir(parents=True, exist_ok=True)
    torch.save((data, slices, cls), target / "processed/processed_queries.pt")
    for name in ("node_mappings.json", "pre_filter.pt", "pre_transform.pt"):
        if (source / "processed" / name).exists():
            shutil.copy(source / "processed" / name, target / "processed" / name)
    if not (target / "raw").exists():
        (target / "raw").symlink_to((source / "raw").resolve())
    print(f"{len(mappings):,} queries: {int((ent >= 0).sum()):,} constant nodes of {len(ent):,} "
          f"({sum(1 for m in mappings for t in m if not t.startswith('?')):,} constant terms), {k} id column(s), "
          f"{int((feat < 0).sum()):,} edges without a store predicate -> {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
