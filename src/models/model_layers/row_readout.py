"""Query-conditioned readout of the constants' degree-moment rows, as the first layer of a backbone.

The row store (data/constant_embeddings/yago, built by
experiments/experiment_outputs/mixed_yago_v2/work_2026-10-19/constant_embedding.py build) holds, per entity c and
feature f (predicate and direction, f < 92 c is the subject, else the object of predicate f - 92) it has, a ROW:
n = deg_f(c) neighbours reached through f, and per feature g of those neighbours M0 = #neighbours with g,
M1 = sum deg_g, M2 = sum deg_g^2. R_k[f, g] are the predicate-level moments per f-edge (refs.npz).

For every pattern with a constant c, leaving c through f to the node v, the layer reads row (c, f):
  exactly  at the features g of the other patterns on v in this (sub)graph: the entry (c, f, g) as logs, ratios and
           deviations from R_k[f, g], or a missing flag (no neighbour of c has g: the star is empty)
  softly   attention over all entries of the row, the query built from f and those g
and appends `out_dim` features to both directed edges of the pattern (zeros for variable-only patterns).
Sub-queries see only their own patterns, so the readout of a subset is that subset's.

Inputs (src/query_featurizers/row_readout_columns.py adds them to a featurised dataset): the node's entity id in
the first ID_COLUMNS columns of x (-1 for variables), edge_attr[:, 0] the source node's feature in the pattern (-1 if
unknown). A float32 column holds integers exactly only up to 2^24, so a graph with that many entities (WatDiv: 20.3M)
splits the id over two columns, id >> 12 and id & 4095 (store_layout: the store's entity count decides, for the
dataset and the layer alike). The id and feature columns are stripped: the outputs are x[:, id columns:] and
edge_attr[:, 1:-1] + readout + the direction column, so the following TripleGineConv layers take
edge_dim = (edge columns - 2) + out_dim. The number of predicates P (features F = 2P) comes from the store too.
The store (~3 GB) is loaded to the input's device on the first forward; it is not part of the state dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ID_SPLIT = 12                     # two id columns: id >> ID_SPLIT and id & (2^ID_SPLIT - 1)


def store_layout(directory):
    """(predicates P, id columns) of a row store: two id columns when its graph has 2^24 entities or more."""
    d = Path(directory)
    refs = np.load(d / "refs.npz")
    n = int(refs["n_entities"]) if "n_entities" in refs else len(np.load(d / "dptr.npy", mmap_mode="r")) - 1
    return len(refs["predicate_iris"]), 1 if n < 2 ** 24 else 2


def entity_ids(x, id_columns):
    """The entity id of every node (-1 for variables) from its first id_columns columns."""
    if id_columns == 1:
        return x[:, 0].long()
    hi, lo = x[:, 0].long(), x[:, 1].long()
    return torch.where(hi >= 0, (hi << ID_SPLIT) + lo, torch.full_like(hi, -1))


def _ranges(start, count):
    return torch.repeat_interleave(start - (torch.cumsum(count, 0) - count), count) + \
        torch.arange(int(count.sum()), device=start.device)


def _segment_max(v, seg, n):
    return torch.full((n, v.shape[1]), -1e9, device=v.device, dtype=v.dtype).scatter_reduce(
        0, seg[:, None].expand_as(v), v, "amax", include_self=True)


class _Store:
    def __init__(self, directory, device, F):
        d = Path(directory)
        ld = lambda n, dt: torch.from_numpy(np.load(d / f"{n}.npy").astype(dt)).to(device)
        dptr = np.load(d / "dptr.npy")
        df = np.load(d / "df.npy").astype(np.int64)
        self.dkey = torch.from_numpy(np.repeat(np.arange(len(dptr) - 1, dtype=np.int64), np.diff(dptr)) * F + df).to(device)
        del dptr, df
        rptr = np.load(d / "rptr.npy")
        eg = np.load(d / "eg.npy")
        self.ekey = torch.from_numpy(np.repeat(np.arange(len(rptr) - 1, dtype=np.int64), np.diff(rptr)) * F + eg).to(device)
        self.rptr = torch.from_numpy(rptr).to(device)
        self.eg = torch.from_numpy(eg.astype(np.int32)).to(device)
        del rptr, eg
        self.n = ld("ddeg", np.float32)
        self.e0, self.e1, self.e2 = ld("e0", np.float32), ld("e1", np.float32), ld("e2", np.float32)
        r = np.load(d / "refs.npz")
        self.lref = torch.from_numpy(np.log(np.maximum(r["ref"], 1e-30))).float().to(device)
        self.lavg = torch.from_numpy(np.log(np.maximum(r["T"], 1) / np.maximum(r["n_ent"], 1))).float().to(device)
        self.device = device

    @staticmethod
    def find(keys, query):
        i = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
        return i, keys[i] == query


class RowReadout(nn.Module):
    def __init__(self, store, out_dim=16, hidden=32, emb=16):
        super().__init__()
        self.store_directory, self.out_dim = str(store), out_dim
        self._store = None
        n_predicates, self.id_columns = store_layout(store)
        self.F = F = 2 * n_predicates
        self.f_emb, self.g_emb = nn.Embedding(F, emb), nn.Embedding(F, emb)
        self.pair = nn.Sequential(nn.Linear(10 + 2 * emb, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.entry = nn.Sequential(nn.Linear(9 + emb, hidden), nn.ReLU(), nn.Linear(hidden, 2 * hidden))
        self.query = nn.Linear(emb, hidden)
        self.out = nn.Sequential(nn.Linear(3 + 3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.hidden = hidden

    def store(self, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:      # "cuda" and "cuda:0" are the same store: one load
            device = torch.device("cuda", torch.cuda.current_device())
        if self._store is None or self._store.device != device:
            self._store = _Store(self.store_directory, device, self.F)
        return self._store

    @staticmethod
    def moments(st, f, g, m0, m1, m2, n):
        """9 features of entries (M0, M1, M2 > 0) of rows (f, n) at features g."""
        ln = torch.log(n)
        return torch.stack([torch.log1p(m0) / 10, torch.log1p(m1) / 10, torch.log1p(m2) / 20,
                            torch.log(m0) - ln, torch.log(m1 / m0), torch.log(m2 / m1.clamp_min(1)) / 2,
                            torch.log(m0) - ln - st.lref[0, f, g], torch.log(m1) - ln - st.lref[1, f, g],
                            (torch.log(m2) - ln - st.lref[2, f, g]) / 2], 1)

    def forward(self, x, edge_index, edge_attr):
        device, F = x.device, self.F
        ent, fs = entity_ids(x, self.id_columns), edge_attr[:, 0].long()
        x, base = x[:, self.id_columns:], edge_attr[:, 1:]
        E = len(fs)
        readout = torch.zeros(E, self.out_dim, device=device, dtype=base.dtype)
        src, dst = edge_index
        eid = torch.nonzero((ent[src] >= 0) & (fs >= 0), as_tuple=True)[0]          # edges leaving a constant
        if len(eid):
            st = self.store(device)
            f = fs[eid]
            rows, found = st.find(st.dkey, ent[src[eid]] * F + f)
            K = len(eid)
            # pairs (constant edge e, other edge e' leaving e's target, not e's own reverse e ^ 1)
            order = torch.argsort(src)
            ss = src[order]
            lo = torch.searchsorted(ss, dst[eid])
            cnt = torch.searchsorted(ss, dst[eid], right=True) - lo
            pe = torch.repeat_interleave(torch.arange(K, device=device), cnt)
            pj = order[_ranges(lo, cnt)]
            keep = (pj != (eid[pe] ^ 1)) & (fs[pj] >= 0)
            pe, pj = pe[keep], pj[keep]
            g = fs[pj]
            pf = f[pe]
            ok = found[pe]
            e, hit = st.find(st.ekey, rows[pe] * F + g)
            hit = hit & ok
            one = torch.ones_like(g, dtype=torch.float32)
            m0, m1, m2 = (torch.where(hit, v[e], one) for v in (st.e0, st.e1, st.e2))
            feats = self.moments(st, pf, g, m0, m1, m2, torch.where(ok, st.n[rows[pe]], one)) * hit[:, None]
            h = self.pair(torch.cat([feats, hit[:, None].float(), self.f_emb(pf), self.g_emb(g)], 1))
            h_sum = torch.zeros(K, self.hidden, device=device).index_add_(0, pe, h)
            h_max = _segment_max(h, pe, K).clamp_min(0) if len(pe) else torch.zeros(K, self.hidden, device=device)
            h_max = torch.where(torch.bincount(pe, minlength=K)[:, None] > 0, h_max, torch.zeros_like(h_max))
            # soft read of the whole row
            q = self.f_emb(f) + torch.zeros(K, self.g_emb.embedding_dim, device=device).index_add_(0, pe, self.g_emb(g)) / \
                torch.bincount(pe, minlength=K).clamp_min(1)[:, None]
            q = self.query(q)
            fr = rows[found]
            owner = torch.nonzero(found, as_tuple=True)[0]
            ne = st.rptr[fr + 1] - st.rptr[fr]
            ent_ix = _ranges(st.rptr[fr], ne)
            eo = torch.repeat_interleave(owner, ne)
            eg = st.eg[ent_ix].long()
            ef = f[eo]
            kv = self.entry(torch.cat([self.moments(st, ef, eg, st.e0[ent_ix], st.e1[ent_ix], st.e2[ent_ix],
                                                    st.n[torch.repeat_interleave(fr, ne)]), self.g_emb(eg)], 1))
            k, v = kv[:, :self.hidden], kv[:, self.hidden:]
            sc = (k * q[eo]).sum(1, keepdim=True) / self.hidden ** 0.5
            sc = torch.exp(sc - _segment_max(sc, eo, K)[eo])
            w = sc / torch.zeros(K, 1, device=device).index_add_(0, eo, sc)[eo]
            soft = torch.zeros(K, self.hidden, device=device).index_add_(0, eo, w * v)
            n = torch.where(found, st.n[rows], torch.ones_like(f, dtype=torch.float32))
            rx = torch.stack([torch.log1p(n) / 10 * found, (torch.log(n) - st.lavg[f]) * found, found.float()], 1)
            readout[eid] = self.out(torch.cat([rx, h_sum, h_max, soft], 1)).to(base.dtype)
            readout = readout + readout[torch.arange(E, device=device) ^ 1]          # both directions of a pattern
        return x, torch.cat([base[:, :-1], readout, base[:, -1:]], 1)


class QueryReadout:
    """RowReadout over the connected subsets of ONE query, with the subset-independent work done once per query.

    A subset's graph keeps the query's patterns in its mask (soft_dp.subset_batch: pattern i's edges 2i, 2i + 1, in
    pattern order), so its constant edges, their rows and their pairs (e, e') are those of the full query whose patterns
    are both in the subset. Built once (on the first call, so a timed first call includes it): the rows, the pair MLP
    output and g embedding of every pair of the full query, the entries' keys and values, and the row features rx.
    A constant edge's readout depends only on which of its pairs are present (its context), so the readout of every
    context of every constant edge with at most `max_pairs` pairs is computed once too, and a call computes each
    constant edge's context from the masks and looks it up; edges with more pairs are read per call. The same numbers
    as RowReadout.forward on soft_dp.subset_batch; inference only."""

    def __init__(self, module: RowReadout, query, device, max_pairs: int = 10):
        self.m, self.query, self.device, self.ready, self.max_pairs = module, query, device, False, max_pairs

    @torch.no_grad()
    def _build(self):
        m, dev = self.m, self.device
        x, ei, ea = self.query.x.to(dev), torch.as_tensor(self.query.edge_index).to(dev), self.query.edge_attr.to(dev)
        ent, fs, F = entity_ids(x, m.id_columns), ea[:, 0].long(), m.F
        src, dst = ei
        self.n = len(self.query.triple_patterns)
        eid = torch.nonzero((ent[src] >= 0) & (fs >= 0), as_tuple=True)[0]
        self.eid, self.K = eid, len(eid)
        if not self.K:
            self.ready = True
            return
        st = m.store(dev)
        f = fs[eid]
        rows, found = st.find(st.dkey, ent[src[eid]] * F + f)
        order = torch.argsort(src)
        ss = src[order]
        lo = torch.searchsorted(ss, dst[eid])
        cnt = torch.searchsorted(ss, dst[eid], right=True) - lo
        pe = torch.repeat_interleave(torch.arange(self.K, device=dev), cnt)
        pj = order[_ranges(lo, cnt)]
        keep = (pj != (eid[pe] ^ 1)) & (fs[pj] >= 0)
        pe, pj = pe[keep], pj[keep]
        g, pf, ok = fs[pj], f[pe], found[pe]
        e, hit = st.find(st.ekey, rows[pe] * F + g)
        hit = hit & ok
        one = torch.ones_like(g, dtype=torch.float32)
        m0, m1, m2 = (torch.where(hit, v[e], one) for v in (st.e0, st.e1, st.e2))
        feats = m.moments(st, pf, g, m0, m1, m2, torch.where(ok, st.n[rows[pe]], one)) * hit[:, None]
        self.h = m.pair(torch.cat([feats, hit[:, None].float(), m.f_emb(pf), m.g_emb(g)], 1))
        self.g_e = m.g_emb(g)
        self.pe, self.pat_c, self.pat_j = pe, eid // 2, pj // 2
        self.f_e = m.f_emb(f)
        fr = rows[found]
        owner = torch.nonzero(found, as_tuple=True)[0]
        ne = st.rptr[fr + 1] - st.rptr[fr]
        ent_ix = _ranges(st.rptr[fr], ne)
        eg = st.eg[ent_ix].long()
        ef = f[torch.repeat_interleave(owner, ne)]
        kv = m.entry(torch.cat([m.moments(st, ef, eg, st.e0[ent_ix], st.e1[ent_ix], st.e2[ent_ix],
                                          st.n[torch.repeat_interleave(fr, ne)]), m.g_emb(eg)], 1))
        self.k, self.v = kv[:, :m.hidden], kv[:, m.hidden:]
        self.ne = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.ne[owner] = ne                              # entries per constant edge (0 when its row is missing)
        self.eptr = torch.cumsum(self.ne, 0) - self.ne
        n = torch.where(found, st.n[rows], torch.ones_like(f, dtype=torch.float32))
        self.rx = torch.stack([torch.log1p(n) / 10 * found, (torch.log(n) - st.lavg[f]) * found, found.float()], 1)
        # contexts: pair b of constant edge c is bit b of c's context; code = bits @ W over the subset's patterns
        npair = torch.bincount(pe, minlength=self.K)
        bit = torch.arange(len(pe), device=dev) - (torch.cumsum(npair, 0) - npair)[pe]      # pe is sorted
        self.wv = 2.0 ** bit                             # W's entries, per pair (merge_readouts)
        self.W = torch.zeros(self.n, self.K, device=dev).index_put_((self.pat_j, pe), self.wv, accumulate=True)
        self.tabled = npair <= self.max_pairs
        size = torch.where(self.tabled, 2 ** npair, torch.zeros_like(npair))
        self.tptr = torch.cumsum(size, 0) - size
        tc = torch.repeat_interleave(torch.arange(self.K, device=dev), size)             # table row -> constant edge
        code = torch.arange(len(tc), device=dev) - self.tptr[tc]
        p_all = _ranges((torch.cumsum(npair, 0) - npair)[tc], npair[tc])                 # (row, pair of its edge)
        r_all = torch.repeat_interleave(torch.arange(len(tc), device=dev), npair[tc])
        on = ((code[r_all] >> bit[p_all]) & 1).bool()
        self.table = self._out(tc, r_all[on], p_all[on])
        self.ready = True

    def _out(self, c_i, pin, p_i):
        """RowReadout's output for instances (constant edge c_i[r]) whose present pairs are p_i (instance pin)."""
        m, dev, H, I = self.m, self.device, self.m.hidden, len(c_i)
        h = self.h[p_i]
        h_sum = torch.zeros(I, H, device=dev).index_add_(0, pin, h)
        cnt = torch.bincount(pin, minlength=I)
        h_max = _segment_max(h, pin, I).clamp_min(0) if len(pin) else torch.zeros(I, H, device=dev)
        h_max = torch.where(cnt[:, None] > 0, h_max, torch.zeros_like(h_max))
        q = self.f_e[c_i] + torch.zeros(I, self.g_e.shape[1], device=dev).index_add_(0, pin, self.g_e[p_i]) / \
            cnt.clamp_min(1)[:, None]
        q = m.query(q)
        ne = self.ne[c_i]
        eo = torch.repeat_interleave(torch.arange(I, device=dev), ne)
        ix = _ranges(self.eptr[c_i], ne)
        sc = (self.k[ix] * q[eo]).sum(1, keepdim=True) / H ** 0.5
        sc = torch.exp(sc - _segment_max(sc, eo, I)[eo])
        w = sc / torch.zeros(I, 1, device=dev).index_add_(0, eo, sc)[eo]
        soft = torch.zeros(I, H, device=dev).index_add_(0, eo, w * self.v[ix])
        return m.out(torch.cat([self.rx[c_i], h_sum, h_max, soft], 1))

    @torch.no_grad()
    def __call__(self, masks, x, edge_attr, bits=None):
        """RowReadout.forward's outputs for soft_dp.subset_batch(query, masks): (x without its id columns, edge_attr
        with the readout).
        bits [M, n]: the subsets' patterns in place of masks (a merged readout: soft_dp.subset_batch_many's)."""
        if not self.ready:
            self._build()
        dev = self.device
        base = edge_attr[:, 1:]
        readout = torch.zeros(len(edge_attr), self.m.out_dim, device=dev, dtype=base.dtype)
        if self.K:
            if bits is None:
                bits = ((masks.to(dev)[:, None] >> torch.arange(self.n, device=dev)) & 1).bool()
            inst = bits[:, self.pat_c]                                         # [M, K]: constant edge in the subset
            s_i, c_i = torch.nonzero(inst, as_tuple=True)
            code = (bits.float() @ self.W).long()                              # [M, K] contexts
            out = self.table[(self.tptr[c_i] + code[s_i, c_i]).clamp_max(max(len(self.table) - 1, 0))] \
                if len(self.table) else torch.zeros(len(c_i), self.m.out_dim, device=dev)
            wide = ~self.tabled[c_i]
            if bool(wide.any()):                                               # too many pairs to table: per call
                w_s, w_c = s_i[wide], c_i[wide]
                I = len(w_c)
                inst_w = torch.full_like(inst, -1, dtype=torch.long)
                inst_w[w_s, w_c] = torch.arange(I, device=dev)
                s_p, p_i = torch.nonzero(inst[:, self.pe] & bits[:, self.pat_j], as_tuple=True)
                pin = inst_w[s_p, self.pe[p_i]]
                keep = pin >= 0
                out[wide] = self._out(w_c, pin[keep], p_i[keep])
            edges = bits.repeat_interleave(2, dim=1)                           # [M, 2n], subset_batch's edge order
            pos = (torch.cumsum(edges.reshape(-1).long(), 0) - 1).view_as(edges)
            readout[pos[s_i, self.eid[c_i]]] = out.to(base.dtype)
            readout = readout + readout[torch.arange(len(readout), device=dev) ^ 1]
        return x[:, self.m.id_columns:], torch.cat([base[:, :-1], readout, base[:, -1:]], 1)


def packed_to(dev, fields):
    """fields {name: tensor}: the ones not on `dev` copied there in one transfer per dtype, side by side (pinned and
    non-blocking to a GPU: a pageable host-to-GPU copy waits for every kernel queued before it); the others as
    they are."""
    out, by = {}, {}
    for k, t in fields.items():
        if t.device == dev:
            out[k] = t
        else:
            by.setdefault(t.dtype, []).append(k)
    for keys in by.values():
        flat = torch.cat([fields[k].reshape(-1) for k in keys])
        flat = flat.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else flat.to(dev)
        s = 0
        for k in keys:
            m = fields[k].numel()
            out[k] = flat[s:s + m].view(fields[k].shape)
            s += m
    return out


def _repeat(values, counts):
    return torch.from_numpy(np.repeat(np.asarray(values, dtype=np.int64), counts))


def merge_readouts(readouts, ns):
    """One QueryReadout over the disjoint union of several queries (n_q patterns each, in order), for
    soft_dp.subset_batch_many's graphs and bits: their built tables side by side, the indexes offset (patterns,
    edges, constant edges, entries, table rows), W from the pairs' entries. Called with those bits, the same numbers
    as each query's own readout on its own subsets. One query: its own readout."""
    for r in readouts:
        if not r.ready:
            r._build()
    if len(readouts) == 1:
        return readouts[0]
    first = readouts[0]
    out = QueryReadout(first.m, None, first.device, first.max_pairs)
    out.ready, out.n = True, sum(ns)
    built = [(r, sum(ns[:x])) for x, r in enumerate(readouts) if r.K]
    out.K = sum(r.K for r, _ in built)
    if not out.K:
        return out
    dev = torch.device(first.device)
    po, Ks, pairs = [b for _, b in built], [r.K for r, _ in built], [len(r.pe) for r, _ in built]
    excl = lambda xs: np.concatenate([[0], np.cumsum(xs)[:-1]]).astype(np.int64)
    g = packed_to(dev, {"po_c": _repeat(po, Ks), "table_c": _repeat(excl([len(r.table) for r, _ in built]), Ks),
                        "entry_c": _repeat(excl([len(r.k) for r, _ in built]), Ks),
                        "K_p": _repeat(excl(Ks), pairs), "po_p": _repeat(po, pairs)})
    cat = lambda k: torch.cat([getattr(r, k) for r, _ in built])
    out.pat_c, out.eid = cat("pat_c") + g["po_c"], cat("eid") + 2 * g["po_c"]
    out.tptr, out.eptr = cat("tptr") + g["table_c"], cat("eptr") + g["entry_c"]
    out.pe, out.pat_j = cat("pe") + g["K_p"], cat("pat_j") + g["po_p"]
    for k in ("tabled", "f_e", "ne", "rx", "h", "g_e", "k", "v", "table"):
        setattr(out, k, cat(k))
    out.W = torch.zeros(out.n, out.K, device=dev).index_put_((out.pat_j, out.pe), cat("wv"), accumulate=True)
    return out
