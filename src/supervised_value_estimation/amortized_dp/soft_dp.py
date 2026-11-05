"""Differentiable bushy C_out DP over all connected subsets, vectorised on the GPU.

The exact bushy DP (bushy.optimal_bushy_plan) is
    B(U) = card(U) + min over valid splits U = U1 + U2 of [B(U1) + B(U2)],   B({p}) = 0,
over the connected pattern subsets U. Here the min is relaxed to a distribution over the splits,
    p(split | U) = softmax over U's splits of -log(B(U1) + B(U2)) / tau,
and B(U) is the EXPECTED cost of building U when every subset picks its split from p:
    B(U) = card(U) + sum over splits p(split | U) [B(U1) + B(U2)]
(so B >= the hard minimum, with no softmin bias; tau -> 0 gives the hard DP). Top-down from the full
query, p defines a distribution over complete bushy plans; mu(U) = P(U is built by the plan), and since
C_out is a sum over the built subsets, the expected cost of the plan distribution under ANY cards is
    E[C_out] = sum over non-singleton U of mu(U) card(U).
With cards from a model and card_true from counts, log E[C_out true] is a differentiable surrogate of
the true cost of the plan the model's estimates choose (Flow-Loss, Negi et al. PVLDB 2021, for
left-deep plan graphs; here the bushy subset hypergraph).

Everything is in log space: logcard per subset, log B, log mu. Subsets are indexed in ascending mask
order (singletons included); SubsetStructure holds them with the splits grouped by subset size, so the
DP runs as n - 1 batched levels. A `family` restricts the DP to plans built from the given subsets only
(e.g. the subsets with observed true cardinalities: bushy.StitchTable's plans).
subset_batch builds the cardinality model's input for many subsets at once (the disjoint union of their
subgraphs; QueryGymEstimatedCost.reduced_form_query one subset at a time).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

FLOOR = -1e4      # log B of a single pattern (cost 0)
UNBUILT = 1e4     # log B of a subset no allowed split builds: finite sentinels keep values and gradients NaN-free


def connected_masks(neighbour_masks: Sequence[int], device) -> torch.Tensor:
    """Bool [2^n]: is mask m a non-empty connected pattern set (grown from its lowest pattern)."""
    n = len(neighbour_masks)
    m = torch.arange(1 << n, device=device, dtype=torch.int64)
    nb = torch.tensor(list(neighbour_masks), device=device, dtype=torch.int64)
    reach_of = torch.zeros_like(m)                       # union of the neighbour masks of m's patterns
    for i in range(n):
        reach_of |= ((m >> i) & 1) * nb[i]
    reach = m & -m
    for _ in range(n - 1):
        reach = reach | (reach_of[reach] & m)
    out = reach == m
    out[0] = False
    return out


def split_pairs(allowed: torch.Tensor, n: int, device, max_elements: int = 1 << 27):
    """{k: (u, s1, s2)} int64 mask tensors of all unordered splits of the allowed subsets of size k into
    two allowed parts (s1 holds u's lowest pattern). With allowed = connected_masks, these are exactly the
    csg-cmp pairs of DPccp (two connected parts of a connected set are adjacent). The 2^(k-1) candidate splits
    of a size's subsets are built in chunks of at most max_elements entries (memory at 20+ patterns)."""
    masks = torch.nonzero(allowed).squeeze(1)
    size = torch.zeros_like(masks)
    for i in range(n):
        size += (masks >> i) & 1
    bit = torch.arange(n, device=device)
    levels = {}
    for k in range(2, n + 1):
        level = masks[size == k]
        if not len(level):
            continue
        j = torch.arange((1 << (k - 1)) - 1, device=device)               # never all others: s2 non-empty
        take = (j[None, :, None] >> torch.arange(k - 1, device=device)) & 1
        rows = max(1, max_elements // (len(j) * max(k - 1, 1)))
        parts = []
        for start in range(0, len(level), rows):
            u = level[start:start + rows]
            bits = (u[:, None] >> bit) & 1
            pos = torch.argsort(1 - bits, dim=1, stable=True)[:, :k]      # u's set bits, ascending
            low, others = pos[:, :1], pos[:, 1:]
            s1 = (1 << low) + (take * (1 << others)[:, None, :]).sum(-1)
            s2 = u[:, None] ^ s1
            ok = allowed[s1] & allowed[s2]
            parts.append((u[:, None].expand_as(s1)[ok], s1[ok], s2[ok]))
        levels[k] = tuple(torch.cat(t) for t in zip(*parts))
    return levels


class SubsetStructure:
    """One query's connected subsets (optionally only `family` and the singletons) and their splits, as
    index tensors on `device`. Build once per query and reuse: it does not depend on the cardinalities."""

    def __init__(self, neighbour_masks: Sequence[int], device, family: Iterable[int] | None = None):
        self.n = n = len(neighbour_masks)
        self.device = device
        allowed = connected_masks(neighbour_masks, device)
        if family is not None:
            keep = torch.zeros_like(allowed)
            keep[1 << torch.arange(n, device=device)] = True
            family = torch.tensor(sorted(set(family)), device=device, dtype=torch.int64)
            if len(family):
                keep[family] = True
            allowed = allowed & keep
        self.masks = torch.nonzero(allowed).squeeze(1)                  # ascending
        index = torch.full((1 << n,), -1, device=device, dtype=torch.int64)   # mask -> position (2^n: not kept)
        index[self.masks] = torch.arange(len(self.masks), device=device)
        self.singleton = (self.masks & (self.masks - 1)) == 0
        self.full = int(index[(1 << n) - 1])                            # -1: the family has no full set
        self.levels = []                                                # ascending size: (subsets, u, s1, s2)
        for k, (u, s1, s2) in sorted(split_pairs(allowed, n, device).items()):
            ui = index[u]
            self.levels.append((torch.unique(ui), ui, index[s1], index[s2]))

    def __len__(self):
        return len(self.masks)

    def n_splits(self) -> int:
        return sum(len(u) for _, u, _, _ in self.levels)


def _segment_logsumexp(x, segment, size):
    """logsumexp of x over equal segment ids -> [size] (-inf for empty segments)."""
    peak = torch.full((size,), -math.inf, device=x.device, dtype=x.dtype).scatter_reduce(0, segment, x, "amax")
    safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    total = torch.zeros(size, device=x.device, dtype=x.dtype).index_add(0, segment, torch.exp(x - safe[segment]))
    return safe + torch.log(total)


def soft_dp(structure: SubsetStructure, logcard: torch.Tensor, tau: float):
    """Expected-cost DP. logcard: [S] log card per subset of the structure (singletons ignored).
    Returns (log B [S], log p per level [splits of that level], aligned with structure.levels).
    tau = 0 is the hard DP (p one-hot on the first cheapest split). log B is about UNBUILT for a subset
    no allowed split can build (and its splits get p = 0 wherever another split exists)."""
    S = len(structure)
    logb = torch.where(structure.singleton, torch.full_like(logcard, FLOOR), torch.full_like(logcard, UNBUILT))
    log_p = []
    for subsets, u, s1, s2 in structure.levels:
        v = torch.logaddexp(logb[s1], logb[s2])                             # log(B(U1) + B(U2)) per split
        if tau > 0:
            score = -v / tau
            lp = score - _segment_logsumexp(score, u, S)[u]
        else:
            best = torch.full((S,), math.inf, device=v.device, dtype=v.dtype).scatter_reduce(0, u, v.detach(), "amin")
            order = torch.arange(len(u), device=v.device)
            hit = torch.where(v.detach() == best[u], order, torch.full_like(order, len(u)))
            first = torch.full((S,), len(u), device=v.device, dtype=order.dtype).scatter_reduce(0, u, hit, "amin")
            lp = torch.where(order == first[u], torch.zeros_like(v), torch.full_like(v, -math.inf))
        log_p.append(lp)
        build = _segment_logsumexp(lp + v, u, S)[subsets]                    # log E over p of B(U1) + B(U2)
        logb = logb.index_put((subsets,), torch.logaddexp(logcard[subsets], build))
    return logb, log_p


def plan_marginals(structure: SubsetStructure, log_p) -> torch.Tensor:
    """mu [S]: P(subset is built by a plan drawn top-down from the full query with p). Singletons get 1
    (every plan has every pattern as a leaf)."""
    if structure.full < 0:
        raise ValueError("the family holds no full set: no complete plan")
    S = len(structure)
    mu = torch.zeros(S, device=structure.device, dtype=log_p[0].dtype if log_p else torch.float32)
    mu = mu.index_put((torch.tensor([structure.full], device=structure.device),), torch.ones(1, device=structure.device, dtype=mu.dtype))
    for (subsets, u, s1, s2), lp in zip(reversed(structure.levels), reversed(log_p)):   # parents before children
        flow = mu[u] * torch.exp(lp)
        mu = mu.index_add(0, s1, flow).index_add(0, s2, flow)
    return mu


def expected_log_cost(structure: SubsetStructure, mu: torch.Tensor, logcard: torch.Tensor) -> torch.Tensor:
    """log E[C_out] = log sum over non-singletons of mu(U) card(U), under the cards `logcard`."""
    weight = torch.where(structure.singleton, torch.zeros_like(mu), mu)
    peak = torch.where(weight > 0, logcard, torch.full_like(logcard, -math.inf)).max().detach()
    return peak + torch.log((weight * torch.exp(torch.clamp(logcard - peak, max=0.0))).sum())


def hard_plan(structure: SubsetStructure, log_p):
    """The plan of a hard DP's (tau = 0) one-hot split choices, as a bushy.py tree."""
    if structure.full < 0:
        raise ValueError("the family holds no full set: no complete plan")
    choice = {}
    for (_, u, s1, s2), lp in zip(structure.levels, log_p):
        picked = torch.nonzero(lp == 0).squeeze(1)
        choice.update(zip(u[picked].tolist(), zip(s1[picked].tolist(), s2[picked].tolist())))
    masks = structure.masks.tolist()

    def decode(i):
        m = masks[i]
        if not m & (m - 1):
            return m.bit_length() - 1
        a, b = choice[i]
        return (decode(a), decode(b))
    return decode(structure.full)


def subset_batch(query, masks: torch.Tensor, device):
    """The disjoint union of the subgraphs of `masks` (int64 tensor), in mask order: (x, edge_index,
    edge_attr, batch) for a GNN forward. Same graphs as Batch.from_data_list of reduced_form_query
    (pattern i's term nodes and its edges 2i, 2i + 1), nodes in ascending node id within a subset."""
    n = len(query.triple_patterns)
    x = query.x.to(device)
    ei = torch.as_tensor(query.edge_index).to(device)
    ea = query.edge_attr.to(device)
    masks = masks.to(device)
    p = torch.arange(n, device=device)
    incidence = torch.zeros(n, x.shape[0], dtype=torch.bool, device=device)   # pattern i touches node v
    incidence[p, ei[0, 2 * p]] = True
    incidence[p, ei[1, 2 * p]] = True
    bits = ((masks[:, None] >> p) & 1).bool()
    node_in = (bits.float() @ incidence.float()) > 0
    counts = node_in.sum(1)
    local = torch.cumsum(node_in.long(), 1) - 1 + (torch.cumsum(counts, 0) - counts)[:, None]
    s_idx, v_idx = torch.nonzero(node_in, as_tuple=True)
    es, ee = torch.nonzero(bits.repeat_interleave(2, dim=1), as_tuple=True)
    edge_index = torch.stack([local[es, ei[0, ee]], local[es, ei[1, ee]]])
    return x[v_idx], edge_index, ea[ee], s_idx


def subset_batch_many(queries, masks_list, device):
    """subset_batch of several queries at once, over the disjoint union of the queries (patterns, nodes and edges
    side by side, in order): (x, edge_index, edge_attr, batch, bits [M, sum n] the subsets' patterns in the union),
    the same graphs as torch.cat of each query's subset_batch with node and graph offsets. queries: objects with x,
    edge_index, edge_attr and triple_patterns (on the device: no copies); masks on the CPU or the device."""
    import numpy as np
    from src.models.model_layers.row_readout import packed_to
    device = torch.device(device)
    ns = [len(q.triple_patterns) for q in queries]
    xs = [q.x.to(device) for q in queries]
    eis = [torch.as_tensor(q.edge_index).to(device) for q in queries]
    vo = np.concatenate([[0], np.cumsum([len(t) for t in xs])[:-1]]).astype(np.int64)
    po = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
    Q = np.arange(len(ns))
    g = packed_to(device, {"masks": torch.cat([m.reshape(-1) for m in masks_list]),
                           "local": torch.from_numpy(np.concatenate([np.arange(n) for n in ns]).astype(np.int64)),
                           "q_of_col": torch.from_numpy(np.repeat(Q, ns)),
                           "q_of_row": torch.from_numpy(np.repeat(Q, [len(m) for m in masks_list])),
                           "edge_off": torch.from_numpy(np.repeat(vo, [e.shape[1] for e in eis]))})
    x = torch.cat(xs)
    ei = torch.cat(eis, 1) + g["edge_off"]
    ea = torch.cat([q.edge_attr.to(device) for q in queries])
    N = sum(ns)
    bits = (((g["masks"][:, None] >> g["local"][None, :]) & 1) != 0) & (g["q_of_row"][:, None] == g["q_of_col"][None, :])
    p = torch.arange(N, device=device)
    incidence = torch.zeros(N, x.shape[0], dtype=torch.bool, device=device)   # pattern i touches node v
    incidence[p, ei[0, 2 * p]] = True
    incidence[p, ei[1, 2 * p]] = True
    node_in = (bits.float() @ incidence.float()) > 0
    counts = node_in.sum(1)
    local = torch.cumsum(node_in.long(), 1) - 1 + (torch.cumsum(counts, 0) - counts)[:, None]
    s_idx, v_idx = torch.nonzero(node_in, as_tuple=True)
    es, ee = torch.nonzero(bits.repeat_interleave(2, dim=1), as_tuple=True)
    edge_index = torch.stack([local[es, ei[0, ee]], local[es, ei[1, ee]]])
    return x[v_idx], edge_index, ea[ee], s_idx, bits


def model_subset_logcards(model, query, masks: torch.Tensor, device, chunk: int = 16384) -> torch.Tensor:
    """The cardinality model's log card for every subset in `masks`, in chunks (keeps the graph for
    backward; wrap in torch.no_grad() for estimates only)."""
    out = []
    for start in range(0, len(masks), chunk):
        x, edge_index, edge_attr, batch = subset_batch(query, masks[start:start + chunk], device)
        heads = model.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        out.append(next(h["output"] for h in heads if h["output_type"] == "cardinality").reshape(-1))
    return torch.cat(out)
