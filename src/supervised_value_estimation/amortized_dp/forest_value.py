"""Forest value network on top of the per-subset cardinality model, for bottom-up bushy C_out planning.

State = a forest F: the components built so far (bushy.py), each with the subtree that built it. Planning merges two
adjacent components per step, as bushy.estimator_beam_plan, but ranks the child forests by

    total(F) = log( cost so far(F) + G(F) ),    G(F) = the cost of finishing F (all remaining join results),

with G predicted by the value network instead of being left out. Three parts:

1. Cardinality of a merge U = A | B: the backbone's per-subset estimate l(U) (single patterns: exact counts), bounded
   by the bounded estimates l''(A), l''(B) of the components already built (their PREVIOUS predictions):
       hard   Bmax(U) = min( l''(A) + l''(B),  l''(A) + log maxdeg(B),  l''(B) + log maxdeg(A) )
              (a single pattern p joining on one variable adds at most maxdeg(p) rows per row; both ends shared or
              the other end a constant: at most 1; the deg terms only when that side is a single pattern)
       soft   B~(U)   the same with log deg = a learned convex mix of log {mean, size-biased mean, max} degree
       l'(U)  = l(U) - g(U) * s * softplus((l(U) - B~(U)) / s)        g = sigmoid(MLP) in [0, 1], s > 0 learned
       l''(U) = min( l'(U), Bmax(U) )                                  the hard bound stays the outer safety net
   B~ <= Bmax always, so the learned part can only tighten. At init g ~ 0.05: l'' ~ min(l, Bmax).
2. Value: V(F) ~ log G(F). The final result is paid by every plan, so G(F) >= card(Q) by construction:
       V(F) = -inf (nothing left, 1 component),  l(Q) (2 components: only the final join),
              logaddexp( l(Q), W(F) ) otherwise,   W = attention over the components
   Tokens: per component its subset embedding phi, l'', the log cost of its subtree (both previous predictions),
   single-pattern flag and size; plus a query token (phi(Q), l(Q), components left). Attention has a learned bias
   per head between components that share a variable. W starts at 0 (last layer zero): V is then constant within
   a query and the planner is the estimator beam with the bound.
3. Planner: beam over forests by total(F) (beam_plan); width 1 is greedy.

Training (losses, on counted plans): per state F on a counted plan, its candidate merges m with child forests F_m
       Q^(m) = logaddexp( l''(U_m), V(F_m) )          estimated cost-to-go of F through m
       Q(m)  = log( card(U_m) + G(F_m) )              true, G stitched from counted subsets (bushy.StitchTable)
   plan     log E_{p}[exp Q] - min Q over the candidates with a known Q,  p = softmax(-Q^ / tau) over them:
            the per-step soft Bellman form of the diff-DP plan loss; gradients reach l'' (backbone, bound) and V
   value    |V(F) - log G(F)| on forests with a known G (3+ components)
   l1       |l - log1p count| on counted subsets, and the same on l'' of counted nodes (one-sided for failed counts)
   consistency  |V(F) - min_m Q^(m)| (target detached) on states with 3+ components (no counts needed)
(build_query_data, forward_batch, losses). Everything is in log1p space, as the cardinality model.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.labels import members

NEG = -1e4        # log of a zero cost (a single pattern costs nothing; nothing left to build)
BIG = 1e4         # log degree of a side with no bound
STATS = ("mean", "size-biased mean", "max")


def _logaddexp(a, b):
    return torch.logaddexp(a, b)


# --- per-query constants -------------------------------------------------------------------------------------

def pattern_fanouts(patterns: Sequence[str], degrees: dict):
    """(touch_s, touch_o, fan): per pattern the patterns sharing its subject / object variable (bitmasks) and
    fan [n, 2, 3] log degree statistics (STATS) of joining it on its subject (side 0: objects per subject) or
    object (side 1: subjects per object); 0 when the other end is a constant (at most one triple per binding),
    BIG when the predicate has no statistics."""
    spo = [p.strip().rstrip(" .").split() for p in patterns]
    spo = [(t[0], t[1], " ".join(t[2:])) for t in spo]
    touch_s, touch_o, fan = [], [], []
    for i, (s, p, o) in enumerate(spo):
        touch = lambda v: sum(1 << j for j, (s2, _, o2) in enumerate(spo) if j != i and v.startswith("?") and v in (s2, o2))
        touch_s.append(touch(s)), touch_o.append(touch(o))
        d = degrees.get(p)
        row = []
        for side, other in (("s", o), ("o", s)):
            if not other.startswith("?"):
                row.append([0.0, 0.0, 0.0])
            elif d is None or side not in d:
                row.append([BIG, BIG, BIG])
            else:
                st = d[side]
                mean = max(float(st["mean"]), 1.0)
                row.append([math.log(mean), math.log(max(mean * float(st["skew"]), 1.0)), math.log(max(float(st["max"]), 1.0))])
        fan.append(row)
    return touch_s, touch_o, torch.tensor(fan, dtype=torch.float32)


def merge_fan(a: int, b: int, touch_s, touch_o, fan) -> torch.Tensor:
    """[3] log degree statistics of `a` joining `b` when `a` is a single pattern (BIG otherwise, or when it shares
    no subject / object variable with b): 0 when both its ends are bound by b."""
    if a & (a - 1):
        return torch.full((3,), BIG)
    i = a.bit_length() - 1
    on_s, on_o = bool(b & touch_s[i]), bool(b & touch_o[i])
    if on_s and on_o:
        return torch.zeros(3)
    if on_s:
        return fan[i, 0]
    if on_o:
        return fan[i, 1]
    return torch.full((3,), BIG)


def _reach(mask, neighbour_masks):
    r = 0
    for i in members(mask):
        r |= neighbour_masks[i]
    return r & ~mask


def forest_adjacency(masks: Sequence[int], neighbour_masks, size: int, reach_cache: dict | None = None) -> torch.Tensor:
    """[size, size] bool: components i, j share a variable (padded with False). reach_cache: {mask: reach}."""
    if reach_cache is None:
        reach_cache = {}
    reach = []
    for m in masks:
        if m not in reach_cache:
            reach_cache[m] = _reach(m, neighbour_masks)
        reach.append(reach_cache[m])
    rows = [[bool(r & o) for o in masks] + [False] * (size - len(masks)) for r in reach]
    adj = torch.zeros(size, size, dtype=torch.bool)
    adj[:len(masks)] = torch.tensor(rows, dtype=torch.bool).reshape(len(masks), size)
    return adj


# --- modules -------------------------------------------------------------------------------------------------

class SoftBound(nn.Module):
    """l''(U) of a merge from l(U), the components' l'' and the degree statistics (module text, part 1)."""

    def __init__(self, phi_dim=200, hidden=32):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.mix[-1].weight)
        with torch.no_grad():
            self.mix[-1].bias.copy_(torch.tensor([0.0, 0.0, 4.0]))         # starts ~ the max degree
        self.gate = nn.Sequential(nn.Linear(phi_dim + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)                         # g ~ 0.05: starts ~ the hard bound alone
        self.log_s = nn.Parameter(torch.tensor(math.log(0.5)))

    def soft_fan(self, f):
        ok = f[:, 0] < BIG / 2
        w = torch.softmax(self.mix(torch.where(ok[:, None], f, torch.zeros_like(f))), dim=1)
        return torch.where(ok, (w * f).sum(1), torch.full_like(f[:, 0], BIG))

    def forward(self, l, la, lb, fa, fb, phi, size):
        """l, la, lb [M]: l(U), l''(A), l''(B); fa, fb [M, 3]: merge_fan(A, B), merge_fan(B, A); phi [M, D]
        standardised phi(U); size [M]: |U| / n. Returns (l''(U), Bmax(U))."""
        cross = la + lb
        hard = torch.minimum(cross, torch.minimum(lb + fa[:, 2], la + fb[:, 2]))
        tight = torch.minimum(cross, torch.minimum(lb + self.soft_fan(fa), la + self.soft_fan(fb)))
        gin = torch.cat([phi, ((l - tight) / 10).clamp(-3, 3)[:, None], ((l - hard) / 10).clamp(-3, 3)[:, None], size[:, None]], 1)
        g = torch.sigmoid(self.gate(gin)).squeeze(1)
        s = torch.exp(self.log_s)
        soft = l - g * s * F.softplus((l - tight) / s)
        return torch.minimum(soft, hard), hard


class _Layer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.adjacent = nn.Parameter(torch.zeros(heads))

    def forward(self, x, pad, adj):
        B, L, _ = x.shape
        bias = adj.float()[:, None] * self.adjacent[None, :, None, None]                       # [B, H, L, L]
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=bias.reshape(B * self.heads, L, L), need_weights=False)[0]
        return x + self.ff(self.n2(x))


class ForestValue(nn.Module):
    """W(F) from the forest's component tokens and a query token (module text, part 2)."""

    def __init__(self, phi_dim=200, d=64, heads=4, layers=2):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(phi_dim + 4, d), nn.ReLU(), nn.Linear(d, d))
        self.query = nn.Sequential(nn.Linear(phi_dim + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([_Layer(d, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, tokens, pad, adj, query):
        """tokens [B, K, D+4], pad [B, K] (True = no component), adj [B, K, K], query [B, D+2] -> W [B]."""
        x = torch.cat([self.query(query)[:, None], self.token(tokens)], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        adj = F.pad(adj, (1, 0, 1, 0))
        for layer in self.layers:
            x = layer(x, pad, adj)
        return self.out(x[:, 0]).squeeze(1)


class ForestPlanner(nn.Module):
    """The bound and the value network, with the phi standardisation (set from a sample of subsets)."""

    def __init__(self, phi_dim=200, d=64, heads=4, layers=2, bound_hidden=32):
        super().__init__()
        self.config = {"phi_dim": phi_dim, "d": d, "heads": heads, "layers": layers, "bound_hidden": bound_hidden}
        self.bound = SoftBound(phi_dim, bound_hidden)
        self.value = ForestValue(phi_dim, d, heads, layers)
        self.register_buffer("mean", torch.zeros(phi_dim))
        self.register_buffer("std", torch.ones(phi_dim))

    def standardise(self, phi):
        return (phi - self.mean) / self.std

    def values(self, tokens, pad, adj, query, lq, k):
        """V(F) [B] from the value network's inputs, l(Q) [B] and the number of components k [B]."""
        W = self.value(tokens, pad, adj, query)
        v = torch.logaddexp(lq, W)
        v = torch.where(k == 2, lq, v)
        return torch.where(k <= 1, torch.full_like(v, NEG), v)

    def save(self, path, extra=None):
        torch.save({"config": self.config, "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device):
        c = torch.load(path, map_location=device)
        net = cls(**c["config"])
        net.load_state_dict(c["state"])
        return net.to(device)


def token_features(phi_std, l2, cost, single, size):
    """[.., D+4] component tokens: standardised phi, l'' / 10, log subtree cost / 10 (0 for a single pattern),
    single-pattern flag, |C| / n."""
    c = torch.where(single, torch.zeros_like(cost), cost / 10)
    return torch.cat([phi_std, (l2 / 10)[..., None], c[..., None], single.float()[..., None], size[..., None]], -1)


# --- planning ------------------------------------------------------------------------------------------------

class QueryContext:
    """One query for planning: the encoder's (phi, l) per subset mask, cached (encode(masks) -> (phi [M, D], l [M]),
    exact counts for single patterns), and the per-query constants."""

    def __init__(self, n, neighbour_masks, patterns, degrees, encode: Callable):
        self.n, self.neighbour_masks, self.encode = n, list(neighbour_masks), encode
        self.touch_s, self.touch_o, self.fan = pattern_fanouts(patterns, degrees)
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reach: dict[int, int] = {}

    def encoded(self, masks):
        missing = [m for m in dict.fromkeys(masks) if m not in self.cache]
        if missing:
            phi, l = self.encode(missing)
            for i, m in enumerate(missing):
                self.cache[m] = (phi[i], l[i])
        return torch.stack([self.cache[m][0] for m in masks]), torch.stack([self.cache[m][1] for m in masks])


@torch.no_grad()
def beam_plan(net: ForestPlanner, ctx: QueryContext, width: int, all_trees=False, noise=0.0, generator=None, trace=None):
    """Bottom-up beam by total(F) = log(cost so far + G(F)) (float64; ties broken by the cost so far). Returns
    (tree(s), number of subsets encoded): the best tree, or with all_trees the beam's final trees, cheapest first (as
    bushy.estimator_beam_plan). noise: sigma of Gaussian noise on each candidate's score (exploration). trace: a list
    that receives (child forest masks, l''(U), V(child)) of every candidate (tests)."""
    n, nb = ctx.n, ctx.neighbour_masks
    full = (1 << n) - 1
    phi_q, lq = ctx.encoded([full])
    singles = [1 << i for i in range(n)]
    _, ls = ctx.encoded(singles)
    device = lq.device
    q_std = net.standardise(phi_q)
    # beam entry: (forest, log cost so far, {mask: (l'', log subtree cost, tree)})
    beams = [(bushy.initial_forest(n), NEG, {m: (float(ls[i]), NEG, i) for i, m in enumerate(singles)})]
    while len(beams[0][0]) > 1:
        cands = [(bi, a, b) for bi, (forest, _, _) in enumerate(beams) for a, b in bushy.candidate_merges(forest, nb)]
        U = [a | b for _, a, b in cands]
        phi_u, l_u = ctx.encoded(U)
        comps = [beams[bi][2] for bi, _, _ in cands]
        la = torch.tensor([c[a][0] for c, (_, a, _) in zip(comps, cands)], device=device)
        lb = torch.tensor([c[b][0] for c, (_, _, b) in zip(comps, cands)], device=device)
        fa = torch.stack([merge_fan(a, b, ctx.touch_s, ctx.touch_o, ctx.fan) for _, a, b in cands]).to(device)
        fb = torch.stack([merge_fan(b, a, ctx.touch_s, ctx.touch_o, ctx.fan) for _, a, b in cands]).to(device)
        size = torch.tensor([bin(u).count("1") / n for u in U], device=device)
        l2, _ = net.bound(l_u, la, lb, fa, fb, net.standardise(phi_u), size)
        ca = torch.tensor([c[a][1] for c, (_, a, _) in zip(comps, cands)], device=device)
        cb = torch.tensor([c[b][1] for c, (_, _, b) in zip(comps, cands)], device=device)
        cost_u = torch.logaddexp(l2, torch.logaddexp(ca, cb))
        so_far = torch.logaddexp(torch.tensor([beams[bi][1] for bi, _, _ in cands], device=device), l2)
        children = [bushy.merge(beams[bi][0], a, b) for bi, a, b in cands]
        k = len(children[0])
        if k == 1:
            total = so_far.double()
            if trace is not None:
                trace.extend((tuple(sorted(c)), float(l2[j]), NEG) for j, c in enumerate(children))
        else:
            masks = [[c for c in child if c != u] + [u] for child, u in zip(children, U)]          # new component last
            old = [m for ms in masks for m in ms[:-1]]
            phi_old, _ = ctx.encoded(old)
            l_old = torch.tensor([comps[j][m][0] for j, ms in enumerate(masks) for m in ms[:-1]], device=device)
            c_old = torch.tensor([comps[j][m][1] for j, ms in enumerate(masks) for m in ms[:-1]], device=device)
            sz_old = torch.tensor([bin(m).count("1") / n for m in old], device=device)
            tok_old = token_features(net.standardise(phi_old), l_old, c_old, sz_old * n < 1.5, sz_old).view(len(cands), k - 1, -1)
            tok_new = token_features(net.standardise(phi_u), l2, cost_u, torch.zeros_like(l2, dtype=torch.bool), size)
            tokens = torch.cat([tok_old, tok_new[:, None]], 1)
            adj = torch.stack([forest_adjacency(ms, nb, k, ctx.reach) for ms in masks]).to(device)
            pad = torch.zeros(len(cands), k, dtype=torch.bool, device=device)
            query = torch.cat([q_std.expand(len(cands), -1), (lq / 10).expand(len(cands))[:, None],
                               torch.full((len(cands), 1), k / n, device=device)], 1)
            kk = torch.full((len(cands),), k, device=device)
            V = net.values(tokens, pad, adj, query, lq.expand(len(cands)), kk)
            total = torch.logaddexp(so_far.double(), V.double())
            if trace is not None:
                trace.extend((tuple(sorted(c)), float(l2[j]), float(V[j])) for j, c in enumerate(children))
        if noise > 0:
            total = total + noise * torch.randn(total.shape, device=device, generator=generator)
        best: dict = {}
        for j, child in enumerate(children):
            key = (j,) if all_trees and k == 1 else child
            score = (float(total[j]), float(so_far[j]))
            if key not in best or score < best[key][0]:
                best[key] = (score, j)
        next_beams = []
        for _, (_, j) in sorted(best.items(), key=lambda item: item[1][0])[:width]:
            bi, a, b = cands[j]
            info = dict(beams[bi][2])
            ta, tb = info.pop(a)[2], info.pop(b)[2]
            info[a | b] = (float(l2[j]), float(cost_u[j]), (ta, tb))
            next_beams.append((children[j], float(so_far[j]), info))
        beams = next_beams
    trees = [next(iter(beam[2].values()))[2] for beam in beams]
    return (trees if all_trees else trees[0]), len(ctx.cache)


# --- training data -------------------------------------------------------------------------------------------

def build_query_data(n, neighbour_masks, patterns, degrees, plans, counts, failed_count):
    """The CPU tensors of one counted query for forward_batch: every subtree of the plans and of their candidate
    merges is a NODE (l'' depends on how a set was built); every forest on a plan is a STATE, with its candidate
    merges and their child forests (FORESTS = states and children, deduplicated). counts: {mask: count or None
    (failed: costs `failed_count`)}. True values: log1p counts, G stitched from the counted subsets."""
    touch_s, touch_o, fan = pattern_fanouts(patterns, degrees)
    nodes: dict = {}
    node_rows = []                                       # (mask, c1, c2, height)

    def node(tree):
        key = bushy.canonical(tree)
        if key not in nodes:
            if bushy.is_leaf(key):
                row = (1 << key, -1, -1, 0)
            else:
                i1, i2 = node(key[0]), node(key[1])
                row = (node_rows[i1][0] | node_rows[i2][0], i1, i2, 1 + max(node_rows[i1][3], node_rows[i2][3]))
            nodes[key] = len(node_rows)
            node_rows.append(row)
        return nodes[key]

    forests: dict = {}                                   # sorted node ids -> forest id
    forest_rows = []

    def forest(ids):
        key = tuple(sorted(ids))
        if key not in forests:
            forests[key] = len(forest_rows)
            forest_rows.append(key)
        return forests[key]

    states: dict = {}                                    # forest id -> state id
    cand_rows = []                                       # (state id, candidate node, child forest)
    for tree in plans:
        trees = {1 << i: i for i in range(n)}
        for a, b in bushy.tree_to_merges(tree):
            ids = {m: node(t) for m, t in trees.items()}
            f = forest(ids.values())
            if f not in states:
                states[f] = len(states)
                for x, y in bushy.candidate_merges(tuple(sorted(trees)), neighbour_masks):
                    c = node((trees[x], trees[y]))
                    child = forest([i for m, i in ids.items() if m not in (x, y)] + [c])
                    cand_rows.append((states[f], c, child))
            trees[a | b] = (trees.pop(a), trees.pop(b))

    true_of = {m: math.log1p(failed_count if v is None else v) for m, v in counts.items()}
    stitch = bushy.StitchTable(list(true_of), true_of)
    masks_needed = sorted({r[0] for r in node_rows} | {(1 << n) - 1})
    index = {m: i for i, m in enumerate(masks_needed)}
    N = len(node_rows)
    fa = torch.full((N, 3), BIG)
    fb = torch.full((N, 3), BIG)
    for j, (m, c1, c2, _) in enumerate(node_rows):
        if c1 >= 0:
            fa[j] = merge_fan(node_rows[c1][0], node_rows[c2][0], touch_s, touch_o, fan)
            fb[j] = merge_fan(node_rows[c2][0], node_rows[c1][0], touch_s, touch_o, fan)
    node_true = torch.tensor([true_of.get(r[0], math.nan) if r[1] >= 0 else math.nan for r in node_rows])
    node_failed = torch.tensor([r[1] >= 0 and r[0] in counts and counts[r[0]] is None for r in node_rows])
    Fn = len(forest_rows)
    forest_nodes = torch.full((Fn, n), -1, dtype=torch.long)
    forest_adj = torch.zeros(Fn, n, n, dtype=torch.bool)
    k = torch.tensor([len(r) for r in forest_rows])
    g_true = torch.full((Fn,), math.nan)
    needed = set(states)
    for s_id, c, child in cand_rows:
        if not math.isnan(float(node_true[c])):
            needed.add(child)
    for f, ids in enumerate(forest_rows):
        forest_nodes[f, :len(ids)] = torch.tensor(ids)
        forest_adj[f] = forest_adjacency([node_rows[i][0] for i in ids], neighbour_masks, n)
        if f in needed:
            g = stitch.cost_to_go([node_rows[i][0] for i in ids])
            g_true[f] = NEG if g == bushy.NEG_INF else (math.nan if g == math.inf else g)
    cand = torch.tensor(cand_rows, dtype=torch.long).reshape(-1, 3)
    q_true = torch.full((len(cand_rows),), math.nan)
    for j, (_, c, child) in enumerate(cand_rows):
        if not math.isnan(float(node_true[c])) and not math.isnan(float(g_true[child])):
            q_true[j] = float(torch.logaddexp(node_true[c], g_true[child]))
    state_forest = torch.tensor(sorted(states, key=states.get), dtype=torch.long)
    raw_masks = [m for m in masks_needed if m & (m - 1) and m in counts]
    return {"n": n, "masks": torch.tensor(masks_needed, dtype=torch.long), "full": index[(1 << n) - 1],
            "node_mask": torch.tensor([index[r[0]] for r in node_rows]), "node_c1": torch.tensor([r[1] for r in node_rows]),
            "node_c2": torch.tensor([r[2] for r in node_rows]), "node_height": torch.tensor([r[3] for r in node_rows]),
            "node_size": torch.tensor([bin(r[0]).count("1") / n for r in node_rows]), "fa": fa, "fb": fb,
            "node_true": node_true, "node_failed": node_failed,
            "forest_nodes": forest_nodes, "forest_adj": forest_adj, "forest_k": k, "g_true": g_true,
            "state_forest": state_forest, "cand": cand, "q_true": q_true,
            "raw_index": torch.tensor([index[m] for m in raw_masks], dtype=torch.long),
            "raw_true": torch.tensor([true_of[m] for m in raw_masks]),
            "raw_failed": torch.tensor([counts[m] is None for m in raw_masks])}


def forward_batch(net: ForestPlanner, parts, phi, l, device, tau):
    """Everything the losses need for a batch of build_query_data parts, given the encoder's phi [M, D] and l [M]
    for the concatenation of the parts' `masks` (single patterns: exact counts). Returns a dict of tensors."""
    off_m = off_n = off_f = off_s = 0
    width = max(p["n"] for p in parts)
    cat = {key: [] for key in ("node_mask", "node_c1", "node_c2", "node_height", "node_size", "fa", "fb",
                               "node_true", "node_failed", "forest_nodes", "forest_adj", "forest_k", "g_true", "forest_q",
                               "state_forest", "cand", "q_true", "raw_index", "raw_true", "raw_failed", "forest_n")}
    for p in parts:
        N, Fn, n = len(p["node_mask"]), len(p["forest_nodes"]), p["n"]
        cat["node_mask"].append(p["node_mask"] + off_m)
        cat["node_c1"].append(torch.where(p["node_c1"] >= 0, p["node_c1"] + off_n, p["node_c1"]))
        cat["node_c2"].append(torch.where(p["node_c2"] >= 0, p["node_c2"] + off_n, p["node_c2"]))
        for key in ("node_height", "node_size", "fa", "fb", "node_true", "node_failed", "forest_k", "g_true", "q_true",
                    "raw_true", "raw_failed"):
            cat[key].append(p[key])
        fn = torch.full((Fn, width), -1, dtype=torch.long)
        fn[:, :n] = torch.where(p["forest_nodes"] >= 0, p["forest_nodes"] + off_n, p["forest_nodes"])
        cat["forest_nodes"].append(fn)
        adj = torch.zeros(Fn, width, width, dtype=torch.bool)
        adj[:, :n, :n] = p["forest_adj"]
        cat["forest_adj"].append(adj)
        cat["forest_q"].append(torch.full((Fn,), p["full"] + off_m))
        cat["forest_n"].append(torch.full((Fn,), n))
        cat["state_forest"].append(p["state_forest"] + off_f)
        c = p["cand"].clone()
        c[:, 0] += off_s
        c[:, 1] += off_n
        c[:, 2] += off_f
        cat["cand"].append(c)
        cat["raw_index"].append(p["raw_index"] + off_m)
        off_m += len(p["masks"])
        off_n += N
        off_f += Fn
        off_s += len(p["state_forest"])
    b = {key: torch.cat(v).to(device) for key, v in cat.items()}
    phi_std = net.standardise(phi)

    # l'' and log subtree cost per node, level by level (single patterns: exact counts, cost 0)
    l_node = l[b["node_mask"]]
    cost = torch.full_like(l_node, NEG)
    l2 = l_node
    for h in range(1, int(b["node_height"].max()) + 1 if len(b["node_height"]) else 1):
        idx = torch.nonzero(b["node_height"] == h).squeeze(1)
        if not len(idx):
            continue
        c1, c2 = b["node_c1"][idx], b["node_c2"][idx]
        lu, _ = net.bound(l_node[idx], l2[c1], l2[c2], b["fa"][idx], b["fb"][idx], phi_std[b["node_mask"][idx]], b["node_size"][idx])
        l2 = l2.index_put((idx,), lu)
        cost = cost.index_put((idx,), torch.logaddexp(lu, torch.logaddexp(cost[c1], cost[c2])))

    # V of every forest
    fn = b["forest_nodes"]
    valid = fn >= 0
    safe = fn.clamp(min=0)
    single = b["node_height"][safe] == 0
    tokens = token_features(phi_std[b["node_mask"][safe]], l2[safe], cost[safe], single, b["node_size"][safe])
    tokens = tokens * valid[..., None]
    k = b["forest_k"]
    query = torch.cat([phi_std[b["forest_q"]], (l[b["forest_q"]] / 10)[:, None], (k / b["forest_n"])[:, None]], 1)
    V = net.values(tokens, ~valid, b["forest_adj"], query, l[b["forest_q"]], k)

    # per candidate: estimated and true cost-to-go from its state
    cand = b["cand"]
    q_hat = torch.logaddexp(l2[cand[:, 1]], V[cand[:, 2]])
    return {"l": l, "l2": l2, "V": V, "q_hat": q_hat, **b}


def _segment_logsumexp(x, seg, size):
    peak = torch.full((size,), -math.inf, device=x.device, dtype=x.dtype).scatter_reduce(0, seg, x, "amax")
    safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    total = torch.zeros(size, device=x.device, dtype=x.dtype).index_add(0, seg, torch.exp(x - safe[seg]))
    return safe + torch.log(total)


def losses(out, tau):
    """{l1_raw, l1_bound, value, plan, floor, consistency} means (0 when empty) and coverage counts."""
    zero = out["l"].sum() * 0
    res = {}

    def l1(est, true, failed):
        err = torch.where(failed, torch.relu(true - est), (est - true).abs())
        return err.mean() if len(err) else zero
    res["l1_raw"] = l1(out["l"][out["raw_index"]], out["raw_true"], out["raw_failed"])
    known = ~torch.isnan(out["node_true"])
    res["l1_bound"] = l1(out["l2"][known], out["node_true"][known], out["node_failed"][known])
    vk = ~torch.isnan(out["g_true"]) & (out["forest_k"] >= 3)
    res["value"] = (out["V"][vk] - out["g_true"][vk]).abs().mean() if bool(vk.any()) else zero

    cand = out["cand"]
    S = len(out["state_forest"])
    qk = ~torch.isnan(out["q_true"])
    n_known = torch.zeros(S, device=cand.device, dtype=torch.long).index_add(0, cand[:, 0], qk.long())
    use = qk & (n_known[cand[:, 0]] >= 2)
    if bool(use.any()):
        seg, qh, qt = cand[use, 0], out["q_hat"][use], out["q_true"][use]
        best = torch.full((S,), math.inf, device=qt.device).scatter_reduce(0, seg, qt, "amin")
        has = torch.isfinite(best)

        def regret(scores):
            lp = -scores / tau - _segment_logsumexp(-scores / tau, seg, S)[seg]
            return (_segment_logsumexp(lp + qt, seg, S) - best)[has]
        res["plan"] = regret(qh).mean()
        with torch.no_grad():
            res["floor"] = regret(qt).mean()
    else:
        res["plan"], res["floor"] = zero, zero
    sk = out["forest_k"][out["state_forest"]] >= 3
    if bool(sk.any()):
        target = torch.full((S,), math.inf, device=cand.device).scatter_reduce(0, cand[:, 0], out["q_hat"].detach(), "amin")
        res["consistency"] = (out["V"][out["state_forest"]][sk] - target[sk]).abs().mean()
    else:
        res["consistency"] = zero
    res["states_with_choice"] = int((n_known >= 2).sum())
    res["states"] = S
    return res
