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

Inputs (src/query_featurizers/row_readout_columns.py adds them to a featurised dataset): x[:, 0] the node's entity
id (-1 for variables), edge_attr[:, 0] the source node's feature in the pattern (-1 if unknown). Both columns are
stripped: the outputs are x[:, 1:] and edge_attr[:, 1:-1] + readout + the direction column, so the following
TripleGineConv layers take edge_dim = (edge columns - 2) + out_dim.
The store (~3 GB) is loaded to the input's device on the first forward; it is not part of the state dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

NP_, F = 92, 184


def _ranges(start, count):
    return torch.repeat_interleave(start - (torch.cumsum(count, 0) - count), count) + \
        torch.arange(int(count.sum()), device=start.device)


def _segment_max(v, seg, n):
    return torch.full((n, v.shape[1]), -1e9, device=v.device, dtype=v.dtype).scatter_reduce(
        0, seg[:, None].expand_as(v), v, "amax", include_self=True)


class _Store:
    def __init__(self, directory, device):
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
        self.f_emb, self.g_emb = nn.Embedding(F, emb), nn.Embedding(F, emb)
        self.pair = nn.Sequential(nn.Linear(10 + 2 * emb, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.entry = nn.Sequential(nn.Linear(9 + emb, hidden), nn.ReLU(), nn.Linear(hidden, 2 * hidden))
        self.query = nn.Linear(emb, hidden)
        self.out = nn.Sequential(nn.Linear(3 + 3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.hidden = hidden

    def store(self, device):
        if self._store is None or self._store.device != device:
            self._store = _Store(self.store_directory, device)
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
        device = x.device
        ent, fs = x[:, 0].long(), edge_attr[:, 0].long()
        x, base = x[:, 1:], edge_attr[:, 1:]
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
    Per call: the sum / max of the present pairs, the attention query, the soft read and the output MLP. The same
    numbers as RowReadout.forward on soft_dp.subset_batch; inference only."""

    def __init__(self, module: RowReadout, query, device):
        self.m, self.query, self.device, self.ready = module, query, device, False

    @torch.no_grad()
    def _build(self):
        m, dev = self.m, self.device
        x, ei, ea = self.query.x.to(dev), torch.as_tensor(self.query.edge_index).to(dev), self.query.edge_attr.to(dev)
        ent, fs = x[:, 0].long(), ea[:, 0].long()
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
        self.ready = True

    @torch.no_grad()
    def __call__(self, masks, x, edge_attr):
        """RowReadout.forward's outputs for soft_dp.subset_batch(query, masks): (x[:, 1:], edge_attr with the readout)."""
        if not self.ready:
            self._build()
        m, dev = self.m, self.device
        base = edge_attr[:, 1:]
        readout = torch.zeros(len(edge_attr), m.out_dim, device=dev, dtype=base.dtype)
        if self.K:
            H = m.hidden
            bits = ((masks.to(dev)[:, None] >> torch.arange(self.n, device=dev)) & 1).bool()
            inst = bits[:, self.pat_c]                                         # [M, K]: constant edge in the subset
            s_i, c_i = torch.nonzero(inst, as_tuple=True)
            I = len(c_i)
            inst_id = (torch.cumsum(inst.reshape(-1).long(), 0) - 1).view_as(inst)
            s_p, p_i = torch.nonzero(inst[:, self.pe] & bits[:, self.pat_j], as_tuple=True)
            pin = inst_id[s_p, self.pe[p_i]]
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
            edges = bits.repeat_interleave(2, dim=1)                           # [M, 2n], subset_batch's edge order
            pos = (torch.cumsum(edges.reshape(-1).long(), 0) - 1).view_as(edges)
            readout[pos[s_i, self.eid[c_i]]] = m.out(torch.cat([self.rx[c_i], h_sum, h_max, soft], 1)).to(base.dtype)
            readout = readout + readout[torch.arange(len(readout), device=dev) ^ 1]
        return x[:, 1:], torch.cat([base[:, :-1], readout, base[:, -1:]], 1)
