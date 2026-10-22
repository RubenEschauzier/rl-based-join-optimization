"""Adds the RowReadout input columns (src/models/model_layers/row_readout.py) to a featurised dataset.

    python -m src.query_featurizers.row_readout_columns SOURCE_ROOT TARGET_ROOT [STORE]

SOURCE_ROOT is a processed QueryCardinalityDataset root; TARGET_ROOT gets a copy with
    x[:, 0]          the node's entity id (the number of <http://example.com/N>), -1 for variables
    edge_attr[:, 0]  the source node's feature in the pattern: predicate id p if the source is the pattern's
                     subject (edge 2i), 92 + p if its object (edge 2i + 1); predicate ids index STORE/refs.npz's
                     predicate_iris (-1 for a predicate outside it)
prepended, so columns counted from the end (direction, pattern count) keep their meaning. raw/ is linked to the
source's, node_mappings.json copied.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PREFIX = "<http://example.com/"
NP_ = 92


def columns(processed, mappings, iris):
    """(entity id per node, source feature per edge) for every graph, concatenated in graph order."""
    data, slices, _ = processed
    xs, es = slices["x"].tolist(), slices["edge_attr"].tolist()
    edge_index = data["edge_index"]
    ent = np.full(xs[-1], -1, dtype=np.float32)
    feat = np.full(es[-1], -1, dtype=np.float32)
    lookup = {int(v): k for k, v in enumerate(iris.tolist())}
    for i, (patterns, term_to_id) in enumerate(zip(data["triple_patterns"], mappings)):
        x0, e0 = xs[i], es[i]
        for term, node in term_to_id.items():
            if term.startswith(PREFIX):
                ent[x0 + int(node)] = int(term[len(PREFIX):-1])
        for j, tp in enumerate(patterns):
            s, p, o = tp.split(" ")[:3]
            assert int(edge_index[0, e0 + 2 * j]) == int(term_to_id[s]), f"graph {i}: edge order differs from its patterns"
            pid = lookup.get(int(p[len(PREFIX):-1]), -1) if p.startswith(PREFIX) else -1
            if pid >= 0:
                feat[e0 + 2 * j], feat[e0 + 2 * j + 1] = pid, NP_ + pid
    return ent, feat


def main(source, target, store="data/constant_embeddings/yago"):
    import json
    source, target = Path(source), Path(target)
    iris = np.load(Path(store) / "refs.npz")["predicate_iris"]
    processed = torch.load(source / "processed/processed_queries.pt", weights_only=False)
    mappings = json.load(open(source / "processed/node_mappings.json"))
    ent, feat = columns(processed, mappings, iris)
    data, slices, cls = processed
    data["x"] = torch.cat([torch.from_numpy(ent)[:, None], data["x"]], 1)
    data["edge_attr"] = torch.cat([torch.from_numpy(feat)[:, None], data["edge_attr"]], 1)
    (target / "processed").mkdir(parents=True, exist_ok=True)
    torch.save((data, slices, cls), target / "processed/processed_queries.pt")
    for name in ("node_mappings.json", "pre_filter.pt", "pre_transform.pt"):
        if (source / "processed" / name).exists():
            shutil.copy(source / "processed" / name, target / "processed" / name)
    if not (target / "raw").exists():
        (target / "raw").symlink_to((source / "raw").resolve())
    print(f"{len(mappings):,} queries: {int((ent >= 0).sum()):,} constant nodes of {len(ent):,}, "
          f"{int((feat < 0).sum()):,} edges without a store predicate -> {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
