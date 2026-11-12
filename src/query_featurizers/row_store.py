"""The row store of RowReadout (src/models/model_layers/row_readout.py), built from a generator graph cache.

    python -m src.query_featurizers.row_store GRAPH_DIR STORE_DIR [CONSTANTS]

GRAPH_DIR is a statistics_based_generation cache (graph.npz, and vocabulary.npz for a dictionary-encoded dump);
CONSTANTS optionally a JSON list of the terms whose rows to keep (the workload's constants; every entity if
omitted). Feature f in [0, F), F = 2 * P for P predicates: f < P the entity is the subject of predicate f, else the
object of predicate f - P; inv(f) is the same predicate seen from the other end.

  dptr, df, ddeg   per entity, its features f and degrees deg_f (rows of kept entities only; dptr spans every entity)
  rptr, eg, e0-e2  per row (c, f), n = deg_f(c) neighbours x reached through f, and per feature g of those
                   neighbours: M0 = #x with deg_g(x) > 0, M1 = sum deg_g(x), M2 = sum deg_g(x)^2
  refs.npz         ref[k, f, g]: M_k summed over ALL rows of f per f-edge (the predicate-level profile), T[f] the
                   f-edges, n_ent[f] the entities with f, predicate_iris, n_entities

The references need no rows: summed over every row of f, M_k[f, g] = sum over the f-edges (c, x) of h_k(deg_g(x))
with h = (1, deg, deg^2), and the f-edges reaching x are deg_inv(f)(x) of them, so ref_k[f, g] * T[f] =
sum_x deg_inv(f)(x) h_k(deg_g(x)): one pass over every entity's pairs of features. Only the kept rows are expanded.
The same store as experiments/experiment_outputs/mixed_yago_v2/work_2026-10-19/constant_embedding.py build (YAGO,
every entity), which RowReadout loads to the GPU whole: keeping only the workload's constants keeps WatDiv's (69M
rows) small. Entity ids are the graph's (row_readout_columns.py writes them into the dataset).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def _ranges(start, count):
    count = np.asarray(count, np.int64)
    return np.repeat(np.asarray(start, np.int64) - (np.cumsum(count) - count), count) + np.arange(int(count.sum()))


def degrees(out_ptr, out_p, out_o, n_predicates):
    """Stage 1: (dptr, df, ddeg) over every entity: its features and their degrees, sorted by (entity, feature)."""
    E, F = len(out_ptr) - 1, 2 * n_predicates
    s = np.repeat(np.arange(E, dtype=np.int64), np.diff(out_ptr))
    dkey, ddeg = np.unique(np.concatenate([s * F + out_p, out_o.astype(np.int64) * F + n_predicates + out_p]),
                           return_counts=True)
    del s
    dptr = np.zeros(E + 1, np.int64)
    np.cumsum(np.bincount(dkey // F, minlength=E), out=dptr[1:])
    return dptr, (dkey % F).astype(np.int16), ddeg.astype(np.int32)


def references(dptr, df, ddeg, n_predicates, chunk=20_000_000):
    """ref[k, f, g] = sum over rows of f of M_k at g, per f-edge; T[f]; n_ent[f] (see the module docstring)."""
    F = 2 * n_predicates
    ref = np.zeros((3, F, F))
    E = len(dptr) - 1
    start = 0
    while start < E:                                    # chunks of entities with about `chunk` feature pairs
        sizes = np.diff(dptr[start:])
        end = start + max(1, int(np.searchsorted(np.cumsum(sizes.astype(np.int64) ** 2), chunk)))
        end = min(end, E)
        n = np.diff(dptr[start:end + 1])
        a = _ranges(np.repeat(dptr[start:end], n), np.repeat(n, n))                   # (a, b): every pair of
        b = np.repeat(np.arange(dptr[start], dptr[end]), np.repeat(n, n))            # features of one entity
        f = (df[b].astype(np.int64) + n_predicates) % F                                 # the row's feature inv(f_b)
        key = f * F + df[a]
        mult, d = ddeg[b].astype(np.float64), ddeg[a].astype(np.float64)
        for k, h in enumerate((np.ones_like(d), d, d * d)):
            ref[k] += np.bincount(key, mult * h, minlength=F * F).reshape(F, F)
        start = end
    T = np.bincount(df, ddeg.astype(np.float64), minlength=F)
    n_ent = np.bincount(df, minlength=F)
    return ref / np.maximum(T, 1)[None, :, None], T, n_ent


def rows(dptr, df, ddeg, out_ptr, out_p, out_o, in_ptr, in_p, in_s, n_predicates, keep, chunk=2_000_000):
    """Stage 2 for the kept entities (sorted ids): (rptr, eg, e0, e1, e2) in row order (entity, feature, g)."""
    F = 2 * n_predicates
    parts = []
    for begin in range(0, len(keep), chunk):
        c = keep[begin:begin + chunk]
        n = dptr[c + 1] - dptr[c]
        row = _ranges(dptr[c], n)                                    # the kept rows, in (entity, feature) order
        rc, rf = np.repeat(c, n), df[row].astype(np.int64)
        # neighbours of every row: c's out-edges with predicate f (f < P) or in-edges with predicate f - P
        lo, hi = np.empty(len(row), np.int64), np.empty(len(row), np.int64)
        out = rf < n_predicates
        for mask, ptr, preds, p in ((out, out_ptr, out_p, rf), (~out, in_ptr, in_p, rf - n_predicates)):
            for i in np.flatnonzero(mask):                         # per row: c's edges with that predicate
                a, b = ptr[rc[i]], ptr[rc[i] + 1]
                seg = preds[a:b]
                lo[i] = a + np.searchsorted(seg, p[i], "left")
                hi[i] = a + np.searchsorted(seg, p[i], "right")
        cnt = hi - lo
        edge = _ranges(lo, cnt)
        x = np.where(np.repeat(out, cnt), out_o[np.minimum(edge, len(out_o) - 1)], in_s[np.minimum(edge, len(in_s) - 1)])
        rid = np.repeat(np.arange(len(row)), cnt)                    # row index within this chunk
        nx = dptr[x + 1] - dptr[x]
        feat = _ranges(dptr[x], nx)
        key = np.repeat(rid, nx) * F + df[feat]
        d = ddeg[feat].astype(np.float64)
        u, inverse = np.unique(key, return_inverse=True)
        parts.append((len(row), u // F, (u % F).astype(np.int16), np.bincount(inverse).astype(np.int32),
                      np.bincount(inverse, d).astype(np.float32), np.bincount(inverse, d * d).astype(np.float32)))
    counts, eg, e0, e1, e2 = [], [], [], [], []
    for n_rows, r, g, m0, m1, m2 in parts:
        counts.append(np.bincount(r, minlength=n_rows))
        eg.append(g), e0.append(m0), e1.append(m1), e2.append(m2)
    counts = np.concatenate(counts) if counts else np.zeros(0, np.int64)
    rptr = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=rptr[1:])
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)
    return rptr, cat(eg, np.int16), cat(e0, np.int32), cat(e1, np.float32), cat(e2, np.float32)


def constant_ids(graph_dir, terms):
    """Graph entity ids of the given terms (N-Triples form: <iri>, "literal"); unknown terms are skipped."""
    from src.random_query_generation.statistics_based_generation.graph import IRI_PREFIX, load_graph
    graph = load_graph(None, graph_dir)
    if graph.entity_terms is None:                                  # numeric YAGO: <http://example.com/N>
        prefix = f"<{IRI_PREFIX}"
        return np.unique([int(t[len(prefix):-1]) for t in terms if t.startswith(prefix)]).astype(np.int64)
    index = term_index(graph.entity_terms)
    return np.unique([index[t] for t in terms if t in index]).astype(np.int64)


def term_index(entity_terms, wanted=None):
    """{term: entity id} over a Terms table (only the `wanted` terms, if given)."""
    data, offsets = entity_terms.data, entity_terms.offsets
    raw = data.tobytes()
    index = {}
    for i in range(len(offsets) - 1):
        t = raw[offsets[i]:offsets[i + 1]].decode()
        if wanted is None or t in wanted:
            index[t] = i
    return index


def build(graph_dir, store_dir, constants=None):
    t0 = time.perf_counter()
    graph_dir, store_dir = Path(graph_dir), Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    g = np.load(graph_dir / "graph.npz")
    P = len(g["predicate_iris"])
    out_ptr, out_p, out_o = g["out_ptr"], g["out_p"].astype(np.int64), g["out_o"]
    dptr, df, ddeg = degrees(out_ptr, out_p, out_o, P)
    E = len(dptr) - 1
    print(f"stage 1: {len(df):,} (entity, feature) degrees of {E:,} entities ({time.perf_counter() - t0:.0f}s)", flush=True)
    ref, T, n_ent = references(dptr, df, ddeg, P)
    print(f"references ({time.perf_counter() - t0:.0f}s)", flush=True)
    keep = np.arange(E, dtype=np.int64) if constants is None else np.asarray(constants, np.int64)
    rptr, eg, e0, e1, e2 = rows(dptr, df, ddeg, out_ptr, g["out_p"], out_o, g["in_ptr"], g["in_p"], g["in_s"], P, keep)
    # the kept rows' degrees; dptr over every entity (0 rows for the others), as RowReadout indexes it
    n_keep = dptr[keep + 1] - dptr[keep]
    kept_rows = _ranges(dptr[keep], n_keep)
    new_dptr = np.zeros(E + 1, np.int64)
    new_dptr[keep + 1] = n_keep
    np.cumsum(new_dptr, out=new_dptr)
    # check: the entry of a row at its own back-edge feature inv(f) has M0 = deg_f(c)
    row = np.repeat(np.arange(len(kept_rows)), np.diff(rptr))
    back = eg.astype(np.int64) == (df[kept_rows][row].astype(np.int64) + P) % (2 * P)
    assert back.sum() == len(kept_rows) and np.array_equal(e0[back], ddeg[kept_rows]), "back-edge moment differs"
    for name, a in (("dptr", new_dptr), ("df", df[kept_rows]), ("ddeg", ddeg[kept_rows]), ("rptr", rptr), ("eg", eg),
                    ("e0", e0), ("e1", e1), ("e2", e2)):
        np.save(store_dir / f"{name}.npy", a)
    np.savez(store_dir / "refs.npz", ref=ref, T=T, n_ent=n_ent, predicate_iris=g["predicate_iris"],
             n_entities=np.int64(E))
    print(f"stage 2: {len(kept_rows):,} rows of {len(keep):,} entities, {len(eg):,} entries -> {store_dir} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    graph, store = sys.argv[1], sys.argv[2]
    kept = None
    if len(sys.argv) > 3:
        kept = constant_ids(graph, json.load(open(sys.argv[3])))
    build(graph, store, kept)
