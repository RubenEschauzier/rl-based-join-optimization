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
3. Planner: beam over forests by total(F) (beam_plan); width 1 is greedy. beam_plans runs several beams of one query
   (the deployed beam and the exploration beams, or the validation widths) as one batch. Several candidates can
   build the same child forest (the same component sets, built in another order); the beam keeps the best
   `per_forest` of them (1: one per forest).
4. Cost (ForestPlanner.cost): None is C_out, the cost of a join U = A |><| B is its result, c^(U) = l''(U). With a
   work_function.QLeverCost the cost is QLever's planning cost in rows of C_out (work_function's module text):
       c^(U) = logsumexp( log out + l''(U),  log read + r^(S, p) for an operand p that is a lazily read pattern
                          (its exact count >= lazy_rows) joined with the other operand S,
                          log sort + log(n log2 n) for an operand that is a join result whose join variables differ
                          from this join's (n from its l'') )                                      (join_cost)
   r^(S, p) = log1p |p| + logsigmoid(h(phi(S), phi(p), l''(S), log1p |p|)) is a learned rows-read head (ReadHead):
   at most the whole scan, starting at about all of it. Costs so far, subtree costs and G are then in these rows,
   and a forest with 2 components has V = c^ of its final join. C_out only: epinet, robust beams, start forests and
   plan_scores.

Training (losses, on counted plans): per state F on a counted plan, its candidate merges m with child forests F_m
       Q^(m) = logaddexp( l''(U_m), V(F_m) )          estimated cost-to-go of F through m
       Q(m)  = log( card(U_m) + G(F_m) )              true, G stitched from counted subsets (bushy.StitchTable)
   plan     log E_{p}[exp Q] - min Q over the candidates with a known Q,  p = softmax(-Q^ / tau) over them:
            the per-step soft Bellman form of the diff-DP plan loss; gradients reach l'' (backbone, bound) and V
   value    |V(F) - log G(F)| on forests with a known G (3+ components)
   w        |W(F) - W*(F)|, W*(F) = log( G(F) - (c(Q) + 1) ) the true remaining joins without the final result:
            W's own target (through V, W gets almost no gradient where the final result dominates G);
            forests with a known G, a counted Q and G(F) > c(Q) + 1
   l1       |l - log1p count| on counted subsets, and the same on l'' of counted nodes (one-sided for failed counts)
   consistency  |V(F) - min_m Q^(m)| (target detached) on states with 3+ components (no counts needed)
   l1_read  |r^(S, p) - log1p rows read| on joins with a lazily read pattern and a measured read (cost QLever)
   beam     the beam-aware loss (build_query_data's `beam`: a traced beam of width k, beam_plans `kept`): per step of
            that beam, the POOL of every candidate of every forest it kept, scored as the planner does,
            total^ = logaddexp(log cost so far^, V). A candidate is ON PATH when its true cost so far plus its
            stitched G equals the best stitched plan of the query (some best known plan still goes through it).
            In a pool with an on-path candidate, its best-scored one c* must survive the beam's selection:
              its own forest's builds   tau softplus((total^(c*) - total^(b-th best other build)) / tau)
              the pool                  tau softplus((total^(c*) - total^(k-th best other kept candidate)) / tau)
            (b = per_forest; the k-th among the candidates that pass the per-forest cut; a term only when there
            are that many). At the last step (complete plans, the cheapest is deployed) k = 1 and every plan is its
            own forest. Pools after the best plan left the beam have no on-path candidate (Daume and Marcu,
            LaSO, 2005; Wiseman and Rush 2016, beam search optimization).
(build_query_data, forward_batch, losses; per-query weights, e.g. importance weights of prioritized replay, scale
each query's terms). Everything is in log1p space, as the cardinality model. With stop_gradient
the planner reads the encoder's phi and l detached: only the L1 on l trains the encoder.
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.labels import members
from src.supervised_value_estimation.amortized_dp.work_function import (QLeverCost, WorkStitchTable, join_work,
                                                                        variable_masks)

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

    def forward(self, tokens, pad, adj, query, return_hidden=False):
        """tokens [B, K, D+4], pad [B, K] (True = no component), adj [B, K, K], query [B, D+2] -> W [B]; with
        return_hidden (W, h), h [B, d] the normalised query-token state the output layers read (the epinet's value
        features)."""
        x = torch.cat([self.query(query)[:, None], self.token(tokens)], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        adj = F.pad(adj, (1, 0, 1, 0))
        for layer in self.layers:
            x = layer(x, pad, adj)
        h = self.out[0](x[:, 0])
        W = self.out[1:](h).squeeze(1)
        return (W, h) if return_hidden else W


class ReadHead(nn.Module):
    """r^(S, p) [M]: log1p rows a lazily read pattern p reads when joined with S (module text, part 4), from
    standardised phi(S), phi(p) and l''(S), log1p |p|: log1p |p| + logsigmoid(h), at most the whole scan; h starts at
    4 (reads ~ all of it)."""

    def __init__(self, phi_dim=200, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * phi_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 4.0)

    def forward(self, phi_s, phi_p, l_s, l_p):
        h = self.mlp(torch.cat([phi_s, phi_p, (l_s / 10)[:, None], (l_p / 10)[:, None]], 1)).squeeze(1)
        return l_p + F.logsigmoid(h)


class ForestPlanner(nn.Module):
    """The bound and the value network, with the phi standardisation (set from a sample of subsets). cost: None
    (C_out) or the fields of a work_function.QLeverCost (module text, part 4), with the rows-read head."""

    def __init__(self, phi_dim=200, d=64, heads=4, layers=2, bound_hidden=32, cost=None, read_hidden=64):
        super().__init__()
        self.config = {"phi_dim": phi_dim, "d": d, "heads": heads, "layers": layers, "bound_hidden": bound_hidden,
                       "cost": dict(cost) if cost else None, "read_hidden": read_hidden}
        self.bound = SoftBound(phi_dim, bound_hidden)
        self.value = ForestValue(phi_dim, d, heads, layers)
        self.cost = QLeverCost(**cost) if cost else None
        self.read = ReadHead(phi_dim, read_hidden) if cost else None
        self.register_buffer("mean", torch.zeros(phi_dim))
        self.register_buffer("std", torch.ones(phi_dim))

    def standardise(self, phi):
        return (phi - self.mean) / self.std

    def values(self, tokens, pad, adj, query, lq, k, w_shift=None, return_hidden=False):
        """V(F) [B] from the value network's inputs, l(Q) [B] and the number of components k [B]. w_shift: h -> [B]
        added to W (an epinet sample, forest_epinet.py); return_hidden: (V, W, h)."""
        W, h = self.value(tokens, pad, adj, query, return_hidden=True)
        if w_shift is not None:
            W = W + w_shift(h)
        v = torch.logaddexp(lq, W)
        v = torch.where(k == 2, lq, v)
        v = torch.where(k <= 1, torch.full_like(v, NEG), v)
        return (v, W, h) if return_hidden else v

    def save(self, path, extra=None):
        torch.save({"config": self.config, "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device):
        c = torch.load(path, map_location=device)
        net = cls(**c["config"])
        net.load_state_dict(c["state"])
        return net.to(device)


def _log_weight(w):
    return math.log(w) if w > 0 else NEG


def sort_log(l):
    """log(n log2 n) of sorting n = expm1(l) rows (l in log1p space; at least 1 row)."""
    ln = torch.log(torch.expm1(l.clamp(max=80.0)).clamp(min=1.0))
    return ln + torch.log((ln / math.log(2)).clamp(min=1.0))


def join_cost(net, l2, la, lb, lazy_a, lazy_b, ra, rb, sort_a, sort_b):
    """log c^(U) [M] of joins U = A |><| B (module text, part 4): l''(U) for C_out (net.cost None); with a QLeverCost
    the logsumexp of log out + l''(U), log read + ra / rb (r^ of A read against B / of B against A) where A / B is
    a lazily read pattern, log sort + log(n log2 n) from l''(A) / l''(B) where that operand is sorted first."""
    c = net.cost
    if c is None:
        return l2
    neg = torch.full_like(l2, NEG)
    terms = [l2 + _log_weight(c.out),
             torch.where(lazy_a, ra + _log_weight(c.read), neg), torch.where(lazy_b, rb + _log_weight(c.read), neg),
             torch.where(sort_a, sort_log(la) + _log_weight(c.sort), neg),
             torch.where(sort_b, sort_log(lb) + _log_weight(c.sort), neg)]
    return torch.logsumexp(torch.stack(terms), 0)


def operand_reads(net, phi_a, phi_b, la, lb):
    """(r^(B, A), r^(A, B)) [M]: the rows read by A read against B and by B against A (standardised phi)."""
    return net.read(phi_b, phi_a, lb, la), net.read(phi_a, phi_b, la, lb)


def token_features(phi_std, l2, cost, single, size):
    """[.., D+4] component tokens: standardised phi, l'' / 10, log subtree cost / 10 (0 for a single pattern),
    single-pattern flag, |C| / n."""
    c = torch.where(single, torch.zeros_like(cost), cost / 10)
    return torch.cat([phi_std, (l2 / 10)[..., None], c[..., None], single.float()[..., None], size[..., None]], -1)


# --- planning ------------------------------------------------------------------------------------------------

class QueryContext:
    """One query for planning: the encoder's (phi, l) per subset mask, cached in one table (encode(masks) -> (phi [M, D],
    l [M]), exact counts for single patterns), and the per-query constants. step_timer: a context manager around each
    beam step's network part (encode the new subsets, bound, value net; evaluate_planners times it as one call)."""

    def __init__(self, n, neighbour_masks, patterns, degrees, encode: Callable):
        self.n, self.neighbour_masks, self.encode = n, list(neighbour_masks), encode
        self.touch_s, self.touch_o, self.fan = pattern_fanouts(patterns, degrees)
        self.var_masks = variable_masks(patterns)       # per pattern its variables (the QLever cost's sort rule)
        self.cache: dict[int, int] = {}                  # mask -> row of phi / l
        self.phi = self.l = None
        self.reach: dict[int, int] = {}
        self.step_timer: Callable = contextlib.nullcontext

    def rows(self, masks):
        """[M] rows of phi / l for the masks (list or tensor), encoding the ones not seen yet."""
        keys = masks.tolist() if torch.is_tensor(masks) else list(masks)
        missing = [m for m in dict.fromkeys(keys) if m not in self.cache]
        if missing:
            phi, l = self.encode(missing)
            base = len(self.cache)
            self.cache.update((m, base + i) for i, m in enumerate(missing))
            self.phi = phi if self.phi is None else torch.cat([self.phi, phi])
            self.l = l if self.l is None else torch.cat([self.l, l])
        return torch.tensor([self.cache[m] for m in keys], dtype=torch.long, device=self.l.device)

    def encoded(self, masks):
        r = self.rows(masks)
        return self.phi[r], self.l[r]


def _popcount(m, n):
    return ((m[..., None] >> torch.arange(n, device=m.device)) & 1).sum(-1)


def _merge_fans(a, b, n, touch_s, touch_o, fan):
    """merge_fan over tensors of masks: [M, 3]."""
    i = _popcount(a - 1, n).clamp(max=n - 1)            # the pattern of a single-pattern a
    on_s, on_o = (b & touch_s[i]) != 0, (b & touch_o[i]) != 0
    big = torch.full_like(fan[i, 0], BIG)
    out = torch.where((on_s & on_o)[:, None], torch.zeros_like(big),
                      torch.where(on_s[:, None], fan[i, 0], torch.where(on_o[:, None], fan[i, 1], big)))
    return torch.where(((a & (a - 1)) == 0)[:, None], out, big)


@torch.no_grad()
def plan_scores(net: ForestPlanner, ctx: QueryContext, trees, epinet=None, indexes=None, with_root=False):
    """[K, T] the estimated log C_out of finished plans under each index: logsumexp over a tree's joins U of l''_z(U),
    with l_z(U) = l(U) + epinet.card_shift(phi(U), z) and the bound applied bottom-up along the tree, exactly as the
    beam does (single patterns: exact counts, not shifted). No index (or z = 0): the planner's own estimate, K = 1.
    The value net is not involved: it only guides the search; a finished plan's cost is its joins' estimates.
    with_root: also [K, T] l'' of each tree's root (trees of 2+ patterns; the start of beam_plans)."""
    if net.cost is not None:
        raise NotImplementedError("plan_scores scores C_out; the planner has a QLever cost")
    n = ctx.n
    device = ctx.encoded([1 << i for i in range(n)])[1].device
    K = 1 if epinet is None or indexes is None else len(indexes)
    long = dict(dtype=torch.long, device=device)
    touch_s, touch_o, fan = torch.tensor(ctx.touch_s, **long), torch.tensor(ctx.touch_o, **long), ctx.fan.to(device)
    node_id: dict = {}                                   # subtree -> node (shared subtrees scored once)
    nodes = []                                           # (U, child a, child b, height); children -1 = single pattern

    def visit(t):
        if isinstance(t, int):
            return ("leaf", t)
        a, b = visit(t[0]), visit(t[1])
        key = (a, b)
        if key not in node_id:
            ma = 1 << a[1] if a[0] == "leaf" else nodes[a[1]][0]
            mb = 1 << b[1] if b[0] == "leaf" else nodes[b[1]][0]
            h = 1 + max(0 if a[0] == "leaf" else nodes[a[1]][3], 0 if b[0] == "leaf" else nodes[b[1]][3])
            node_id[key] = len(nodes)
            nodes.append((ma | mb, a, b, h))
        return ("node", node_id[key])
    roots = [visit(bushy.canonical(t))[1] for t in trees]
    singles = ctx.encoded([1 << i for i in range(n)])[1]                   # [n] exact log1p counts
    l2 = torch.zeros(K, len(nodes), device=device)
    for h in range(1, max(x[3] for x in nodes) + 1):
        idx = [i for i, x in enumerate(nodes) if x[3] == h]
        U = torch.tensor([nodes[i][0] for i in idx], **long)
        phi_u, l_u = ctx.encoded(U)
        phi_std = net.standardise(phi_u)
        l_u = l_u.expand(K, -1)
        if K > 1 or (epinet is not None and indexes is not None):
            l_u = l_u + epinet.scale_card * epinet.card_samples(phi_std, indexes.to(device))
        child = lambda c: singles[c[1]].expand(K) if c[0] == "leaf" else l2[:, c[1]]
        la = torch.stack([child(nodes[i][1]) for i in idx], 1)
        lb = torch.stack([child(nodes[i][2]) for i in idx], 1)
        a = torch.tensor([1 << c[1] if c[0] == "leaf" else nodes[c[1]][0] for c in (nodes[i][1] for i in idx)], **long)
        b = torch.tensor([1 << c[1] if c[0] == "leaf" else nodes[c[1]][0] for c in (nodes[i][2] for i in idx)], **long)
        fa, fb = _merge_fans(a, b, n, touch_s, touch_o, fan), _merge_fans(b, a, n, touch_s, touch_o, fan)
        size = _popcount(U, n) / n
        out, _ = net.bound(l_u.reshape(-1), la.reshape(-1), lb.reshape(-1), fa.repeat(K, 1), fb.repeat(K, 1),
                           phi_std.repeat(K, 1), size.repeat(K))
        l2[:, idx] = out.view(K, len(idx))

    def members_of(i, acc):
        acc.append(i)
        for c in (nodes[i][1], nodes[i][2]):
            if c[0] == "node":
                members_of(c[1], acc)
        return acc
    scores = torch.stack([torch.logsumexp(l2[:, members_of(r, [])].double(), 1) for r in roots], 1).float()
    return (scores, l2[:, roots]) if with_root else scores


@torch.no_grad()
def beam_plans(net: ForestPlanner, ctx: QueryContext, widths, noises=None, all_trees=None, generator=None, trace=None,
               epinet=None, indexes=None, robust=None, start=None, trace_h=None, per_forest=1, kept=None):
    """Independent beams over one query in one batch: group g keeps widths[g] forests per step, ranked by
    total(F) = log(cost so far + G(F)) (float64; ties broken by the cost so far, then by first appearance), with
    N(0, noises[g]) on each candidate's score (exploration). Returns ([trees of group g, cheapest first: all the
    beam's final trees with all_trees[g], else the best one], number of subsets encoded). All forests of a step have
    the same number of components, so a step is a few tensor operations over every candidate of every group. trace:
    a list that receives (child forest masks, l''(U), V(child)) of every candidate (tests). epinet, indexes [G, D]:
    Thompson sampling (forest_epinet.ForestEpinet): group g plans with l_z(U) = l(U) + epinet.card_shift and
    W_z = W + epinet.value_shift at its index z_g, one sample for the whole plan; z = 0 is the plain planner.
    robust {"epinet", "indexes" [K, D], "stat" "mean" | "p90"}: risk-aware beams. Every entry carries K worlds, one per
    sampled model z_k: its own l''_z of each component, log subtree costs and cost so far, with
    l_z(U) = l(U) + card_shift(phi(U), z_k) through the bound. Candidates are ranked by the statistic over the worlds of
    total_z(F) = logaddexp(log cost so far_z, V_z(F)), V_z = logaddexp(l_z(Q), W + value_shift(h, z_k)), with W and h
    from the value net ONCE on the plain (z = 0) tokens. "mean": E_z[total_z] (the expected log cost, as the geomean
    metric); "p90": the 90% quantile over z (conservative). A finished plan's score is the statistic of its
    plan_scores. Not combined with indexes (Thompson beams).
    start: the component trees of a forest to plan from (every group; default the single patterns), each component
    with its l'' and log subtree cost along its tree as plan_scores computes them. Not combined with robust.
    trace_h: a list that receives, aligned with trace, the value net's state h of every candidate's child forest (a
    [d] tensor; None for a complete forest).
    per_forest: per group and child forest the best this many candidates (builds) are kept, instead of one; the
    final step keeps one complete plan per group either way (or all, with all_trees). kept: a list that receives,
    per step, [(group, component trees)] of every forest the beams keep (the beam loss's pools, build_query_data).
    With a QLever cost (net.cost) the joins cost join_cost (module text, part 4); not with epinet, robust or start."""
    G = len(widths)
    qcost = net.cost is not None
    if qcost and (epinet is not None or robust is not None or start is not None):
        raise NotImplementedError("beam_plans with a QLever cost: no epinet, robust beams or start forest")
    noises = list(noises) if noises is not None else [0.0] * G
    all_trees = list(all_trees) if all_trees is not None else [False] * G
    n = ctx.n
    rq = ctx.rows([(1 << n) - 1])
    rs = ctx.rows([1 << i for i in range(n)])
    phi_q, lq = ctx.phi[rq], ctx.l[rq]
    device = lq.device
    q_std = net.standardise(phi_q)
    lq_g = lq.expand(G)
    if epinet is not None:
        indexes = indexes.to(device)
        lq_g = lq + epinet.card_shift(q_std.expand(G, -1), indexes)
    long = dict(dtype=torch.long, device=device)
    bit = torch.ones(n, **long) << torch.arange(n, **long)
    nb = torch.tensor(ctx.neighbour_masks, **long)
    touch_s, touch_o, fan = torch.tensor(ctx.touch_s, **long), torch.tensor(ctx.touch_o, **long), ctx.fan.to(device)
    width_t = torch.tensor(widths, **long)
    if robust is not None:
        assert epinet is None, "robust beams and Thompson indexes do not mix"
        r_epi, Zr, stat = robust["epinet"], robust["indexes"].to(device), robust["stat"]
        assert stat in ("mean", "p90"), stat
        Kr = len(Zr)
        lq_r = lq + r_epi.scale_card * r_epi.card_samples(q_std, Zr)[:, 0]               # [K] l_z(Q)
    noise_t = torch.tensor(noises, dtype=torch.float32, device=device)
    all_t = torch.tensor(all_trees, dtype=torch.bool, device=device)
    # beam entries: components sorted by mask; mask, reach, cache row, l'', log subtree cost; per entry the cost so far
    grp = torch.arange(G, **long)
    cm = bit.expand(G, n).clone()
    cr = (nb & ~bit).expand(G, n).clone()
    crow = rs.expand(G, n).clone()
    cl = ctx.l[rs].expand(G, n).clone()
    cc = torch.full((G, n), NEG, device=device)
    so = torch.full((G,), NEG, device=device)
    if robust is not None:                               # per entry and world: l'', log subtree cost, cost so far
        clw = cl[..., None].expand(G, n, Kr).clone()
        ccw = torch.full((G, n, Kr), NEG, device=device)
        sow = torch.full((G, Kr), NEG, device=device)
    trees = [list(range(n)) for _ in range(G)]
    if per_forest > 1:                                   # per component a hash of its tree: the same build twice
        ch = torch.tensor([(i + 1) * 0x9E3779B97F4A7C15 % (1 << 62) for i in range(n)], **long).expand(G, n).clone()
    if qcost:                                            # per component its variables and sort key (-1: a scan)
        lazy_l = math.log1p(net.cost.lazy_rows)
        cv = torch.tensor(ctx.var_masks, **long).expand(G, n).clone()
        cs = torch.full((G, n), -1, **long)
    if start is not None:
        assert robust is None, "a start forest is not combined with robust beams"
        comps = sorted((bushy.tree_mask(t), bushy.canonical(t)) for t in start)
        masks = torch.tensor([m for m, _ in comps], **long)
        reach_of = [0] * len(comps)                      # union of the neighbour masks of a component's patterns
        for idx, (m, _) in enumerate(comps):
            for p in bushy.members(m):
                reach_of[idx] |= ctx.neighbour_masks[p]
        reach = torch.tensor(reach_of, **long)
        built = [i for i, (m, _) in enumerate(comps) if m & (m - 1)]
        rows = ctx.rows(masks)                           # first: encoding new masks replaces ctx.l
        l_start = ctx.l[rows].clone()
        c_start = torch.full((len(comps),), NEG, device=device)
        if built:
            cost, root = plan_scores(net, ctx, [comps[i][1] for i in built], with_root=True)
            l_start[built], c_start[built] = root[0], cost[0]
        cm = masks.expand(G, -1).clone()
        cr = (reach & ~masks).expand(G, -1).clone()
        crow = rows.expand(G, -1).clone()
        cl = l_start.expand(G, -1).clone()
        cc = c_start.expand(G, -1).clone()
        so = torch.logsumexp(c_start, 0).expand(G).clone() if built else so
        trees = [[t for _, t in comps] for _ in range(G)]
    for k in range(len(trees[0]), 1, -1):
        I, J = torch.triu_indices(k, k, 1, device=device)
        e, p = ((cr[:, I] & cm[:, J]) != 0).nonzero(as_tuple=True)            # by entry, then pair (i < j)
        i, j = I[p], J[p]
        a, b = cm[e, i], cm[e, j]
        U = a | b
        with ctx.step_timer():                           # the step's network part: encode, bound, value net
            ru = ctx.rows(U)
            phi_u, l_u = ctx.phi[ru], ctx.l[ru]
            phi_std = net.standardise(phi_u)
            gc = grp[e]
            if epinet is not None:
                l_u = l_u + epinet.card_shift(phi_std, indexes[gc])
            size = _popcount(U, n) / n
            fa, fb = _merge_fans(torch.cat([a, b]), torch.cat([b, a]), n, touch_s, touch_o, fan).split(len(a))
            l2, _ = net.bound(l_u, cl[e, i], cl[e, j], fa, fb, phi_std, size)
            c_u = l2
            if qcost:
                jv = cv[e, i] & cv[e, j]
                sort_a, sort_b = (cs[e, i] >= 0) & (cs[e, i] != jv), (cs[e, j] >= 0) & (cs[e, j] != jv)
                lazy_a, lazy_b = ((a & (a - 1)) == 0) & (cl[e, i] >= lazy_l), ((b & (b - 1)) == 0) & (cl[e, j] >= lazy_l)
                ra, rb = operand_reads(net, net.standardise(ctx.phi[crow[e, i]]), net.standardise(ctx.phi[crow[e, j]]),
                                       cl[e, i], cl[e, j])
                c_u = join_cost(net, l2, cl[e, i], cl[e, j], lazy_a, lazy_b, ra, rb, sort_a, sort_b)
            cost_u = torch.logaddexp(c_u, torch.logaddexp(cc[e, i], cc[e, j]))
            so_far = torch.logaddexp(so[e], c_u)
            if robust is not None:
                l_uw = l_u[:, None] + r_epi.scale_card * r_epi.card_samples(phi_std, Zr).T            # [C, K]
                Cn = len(e)
                l2w, _ = net.bound(l_uw.reshape(-1), clw[e, i].reshape(-1), clw[e, j].reshape(-1),
                                   fa.repeat_interleave(Kr, 0), fb.repeat_interleave(Kr, 0),
                                   phi_std.repeat_interleave(Kr, 0), size.repeat_interleave(Kr))
                l2w = l2w.view(Cn, Kr)
                cost_uw = torch.logaddexp(l2w, torch.logaddexp(ccw[e, i], ccw[e, j]))
                so_w = torch.logaddexp(sow[e], l2w)
            C, kc = len(e), k - 1
            ar = torch.arange(k, **long)
            old = ((ar != i[:, None]) & (ar != j[:, None])).nonzero(as_tuple=True)[1].view(C, k - 2)
            eo = e[:, None]
            child_m = torch.cat([cm[eo, old], U[:, None]], 1)                       # the new component last
            child_r = torch.cat([cr[eo, old], ((cr[e, i] | cr[e, j]) & ~U)[:, None]], 1)
            if qcost:
                child_v = torch.cat([cv[eo, old], (cv[e, i] | cv[e, j])[:, None]], 1)
                child_s = torch.cat([cs[eo, old], jv[:, None]], 1)
            if per_forest > 1:                           # symmetric in the two parts (int64 arithmetic wraps)
                lo, hi = torch.minimum(ch[e, i], ch[e, j]), torch.maximum(ch[e, i], ch[e, j])
                child_h = torch.cat([ch[eo, old], ((lo * 1000003) ^ (hi * 998244353) ^ (lo + hi))[:, None]], 1)
            if kc == 1:
                V = torch.full_like(l2, NEG)
                hv = None
                total = so_far.double()
                if robust is not None:
                    total_w = so_w.double()
            else:
                sz_old = _popcount(cm[eo, old], n) / n
                tok_old = token_features(net.standardise(ctx.phi[crow[eo, old]]), cl[eo, old], cc[eo, old], sz_old * n < 1.5, sz_old)
                tok_new = token_features(phi_std, l2, cost_u, torch.zeros_like(l2, dtype=torch.bool), size)
                tokens = torch.cat([tok_old, tok_new[:, None]], 1)
                adj = (child_r[:, :, None] & child_m[:, None, :]) != 0
                query = torch.cat([q_std.expand(C, -1), (lq_g[gc] / 10)[:, None], torch.full((C, 1), kc / n, device=device)], 1)
                shift = (lambda h: epinet.value_shift(h, indexes[gc])) if epinet is not None else None
                lq_v = lq_g[gc] + _log_weight(net.cost.out) if qcost else lq_g[gc]
                V, W, hv = net.values(tokens, torch.zeros(C, kc, dtype=torch.bool, device=device), adj, query, lq_v,
                                      torch.full((C,), kc, **long), w_shift=shift, return_hidden=True)
                if qcost and kc == 2:                    # the cost of the final join of the two components
                    rows2 = torch.cat([crow[eo, old], ru[:, None]], 1)
                    l2c = torch.cat([cl[eo, old], l2[:, None]], 1)
                    jq = child_v[:, 0] & child_v[:, 1]
                    sq = (child_s >= 0) & (child_s != jq[:, None])
                    lz = ((child_m & (child_m - 1)) == 0) & (l2c >= lazy_l)
                    r0, r1 = operand_reads(net, net.standardise(ctx.phi[rows2[:, 0]]), net.standardise(ctx.phi[rows2[:, 1]]),
                                           l2c[:, 0], l2c[:, 1])
                    V = join_cost(net, lq_g[gc], l2c[:, 0], l2c[:, 1], lz[:, 0], lz[:, 1], r0, r1, sq[:, 0], sq[:, 1])
                total = torch.logaddexp(so_far.double(), V.double())
                if robust is not None:
                    Vw = lq_r.expand(C, Kr) if kc == 2 else torch.logaddexp(
                        lq_r.expand(C, Kr), W[:, None] + r_epi.scale_value * r_epi.value_samples(hv, Zr).T)
                    total_w = torch.logaddexp(so_w.double(), Vw.double())
        if robust is not None:                           # rank by the statistic over the worlds
            total = total_w.mean(1) if stat == "mean" else torch.quantile(total_w, 0.9, dim=1)
        if trace is not None:
            trace.extend((tuple(sorted(m)), x, v) for m, x, v in zip(child_m.tolist(), l2.tolist(), V.tolist()))
        if trace_h is not None:
            trace_h.extend([None] * len(l2) if hv is None else list(hv.detach()))
        if bool((noise_t > 0).any()):
            total = total + noise_t[gc] * torch.randn(total.shape, device=device, generator=generator)

        # per group and child forest the best candidate (final step with all_trees: every candidate), then per group
        # the best widths[g] of those
        idx = torch.arange(C, **long)
        distinct = torch.where(all_t[gc] & (kc == 1), idx, torch.zeros_like(idx))
        _, inv = torch.unique(torch.cat([gc[:, None], child_m.sort(1).values, distinct[:, None]], 1), dim=0, return_inverse=True)
        K = int(inv.max()) + 1
        order = torch.sort(so_far, stable=True).indices
        order = order[torch.sort(total[order], stable=True).indices]               # by (total, so far, index)
        pos = torch.full((K,), C, **long).scatter_reduce(0, inv[order], idx, "amin")
        win = order[pos]
        first = torch.full((K,), C, **long).scatter_reduce(0, inv, idx, "amin")
        o = torch.argsort(first, stable=True)
        o = o[torch.sort(so_far[win[o]], stable=True).indices]
        o = o[torch.sort(total[win[o]], stable=True).indices]                     # by (total, so far, first appearance)
        win = win[o]
        if per_forest > 1 and kc > 1:                    # the best per_forest builds of each child forest
            _, inv_b = torch.unique(torch.cat([gc[:, None], child_m.sort(1).values, child_h.sum(1)[:, None]], 1),
                                    dim=0, return_inverse=True)
            first_b = torch.full((int(inv_b.max()) + 1,), C, **long).scatter_reduce(0, inv_b[order], idx, "amin")
            ordered = order[torch.sort(first_b).values]  # the best candidate of each distinct build, in order
            g_ord = inv[ordered]
            srt = torch.sort(g_ord, stable=True).indices
            rank = torch.empty_like(srt)
            rank[srt] = torch.arange(len(srt), **long) - torch.searchsorted(g_ord[srt], g_ord[srt])
            win = ordered[rank < per_forest]                                        # by (total, so far, index)
        win = win[torch.sort(gc[win], stable=True).indices]
        gw = gc[win]
        counts = torch.bincount(gw, minlength=G)
        rank = torch.arange(len(win), **long) - (torch.cumsum(counts, 0) - counts)[gw]
        sel = win[rank < width_t[gw]]

        es, olds, perm = e[sel], old[sel], child_m[sel].argsort(1)
        cm = child_m[sel].gather(1, perm)
        cr = child_r[sel].gather(1, perm)
        crow = torch.cat([crow[es[:, None], olds], ru[sel][:, None]], 1).gather(1, perm)
        cl = torch.cat([cl[es[:, None], olds], l2[sel][:, None]], 1).gather(1, perm)
        cc = torch.cat([cc[es[:, None], olds], cost_u[sel][:, None]], 1).gather(1, perm)
        so, grp = so_far[sel], gc[sel]
        if qcost:
            cv = child_v[sel].gather(1, perm)
            cs = child_s[sel].gather(1, perm)
        if per_forest > 1:
            ch = child_h[sel].gather(1, perm)
        if robust is not None:
            pw = perm[..., None].expand(-1, -1, Kr)
            clw = torch.cat([clw[es[:, None], olds], l2w[sel][:, None]], 1).gather(1, pw)
            ccw = torch.cat([ccw[es[:, None], olds], cost_uw[sel][:, None]], 1).gather(1, pw)
            sow = so_w[sel]
        built = [[trees[ee][x] for x in ol] + [(trees[ee][ii], trees[ee][jj])]
                 for ee, ii, jj, ol in zip(es.tolist(), i[sel].tolist(), j[sel].tolist(), olds.tolist())]
        trees = [[t[x] for x in pm] for t, pm in zip(built, perm.tolist())]
        if kept is not None:
            kept.append([(g, tuple(t)) for g, t in zip(grp.tolist(), trees)])
    out = [[] for _ in range(G)]
    for g, t in zip(grp.tolist(), trees):
        out[g].append(t[0])
    return [ts if keep else ts[:1] for ts, keep in zip(out, all_trees)], len(ctx.cache)


def beam_plan(net: ForestPlanner, ctx: QueryContext, width: int, all_trees=False, noise=0.0, generator=None, trace=None,
              per_forest=1):
    """One beam (beam_plans): (the best tree, or with all_trees the beam's final trees, cheapest first, as
    bushy.estimator_beam_plan; number of subsets encoded). Width 1 is greedy."""
    groups, encoded = beam_plans(net, ctx, [width], [noise], [all_trees], generator, trace, per_forest=per_forest)
    return (groups[0] if all_trees else groups[0][0]), encoded


# --- training data -------------------------------------------------------------------------------------------

def build_query_data(n, neighbour_masks, patterns, degrees, plans, counts, failed_count, sizes=None, reads=None,
                     cost=None, beam=None, per_forest=1):
    """The CPU tensors of one counted query for forward_batch: every subtree of the plans and of their candidate
    merges is a NODE (l'' depends on how a set was built); every forest on a plan is a STATE, with its candidate
    merges and their child forests (FORESTS = states and children, deduplicated). counts: {mask: count or None
    (failed: costs `failed_count`)}. True values: log1p counts, G stitched from the counted subsets.
    cost: None (C_out) or a work_function.QLeverCost: true join costs and G in its rows (WorkStitchTable), with
    sizes (rows per pattern) and reads ({(other operand mask, pattern index): rows read or None}); every node with a
    lazily read pattern operand carries that read (log1p, nan unknown) for the l1_read loss. beam: {"width": k,
    "kept": beam_plans' kept of one group (per step its forests' component trees)} adds the beam loss's pools
    (module text), with b = per_forest."""
    touch_s, touch_o, fan = pattern_fanouts(patterns, degrees)
    var_masks = variable_masks(patterns)
    nodes: dict = {}
    node_rows = []                                       # (mask, c1, c2, height)
    node_vars = []                                       # variables of the node's patterns (bitmask)

    def node(tree):
        key = bushy.canonical(tree)
        if key not in nodes:
            if bushy.is_leaf(key):
                row, v = (1 << key, -1, -1, 0), var_masks[key]
            else:
                i1, i2 = node(key[0]), node(key[1])
                row = (node_rows[i1][0] | node_rows[i2][0], i1, i2, 1 + max(node_rows[i1][3], node_rows[i2][3]))
                v = node_vars[i1] | node_vars[i2]
            nodes[key] = len(node_rows)
            node_rows.append(row)
            node_vars.append(v)
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
    pools = []                                           # per beam step: (child forest ids, their component sets)
    if beam is not None:
        for parents in [[tuple(range(n))]] + list(beam["kept"]):
            if len(parents[0]) < 2:
                break
            children, sets, seen = [], [], set()
            for comps in parents:
                trees = {bushy.tree_mask(t): t for t in comps}
                ids = {m: node(t) for m, t in trees.items()}
                for x, y in bushy.candidate_merges(tuple(sorted(trees)), neighbour_masks):
                    child = forest([i for m, i in ids.items() if m not in (x, y)] + [node((trees[x], trees[y]))])
                    if child not in seen:                # the same build from two parents: one candidate
                        seen.add(child)
                        children.append(child)
                        sets.append(tuple(sorted([m for m in trees if m not in (x, y)] + [x | y])))
            if len(parents[0]) == 2:                     # the last step: complete plans, the cheapest is deployed
                sets = [(t,) for t in range(len(children))]
            pools.append((children, sets))

    true_of = {m: math.log1p(failed_count if v is None else v) for m, v in counts.items()}
    N = len(node_rows)
    node_jv = [node_vars[r[1]] & node_vars[r[2]] if r[1] >= 0 else -1 for r in node_rows]
    node_true = torch.tensor([true_of.get(r[0], math.nan) if r[1] >= 0 else math.nan for r in node_rows])
    node_failed = torch.tensor([r[1] >= 0 and r[0] in counts and counts[r[0]] is None for r in node_rows])
    node_lazy = torch.zeros(N, dtype=torch.bool)
    read1, read2 = torch.full((N,), math.nan), torch.full((N,), math.nan)
    if cost is None:
        stitch = bushy.StitchTable(list(true_of), true_of)
        sorted_by = None
        join_true = node_true                            # a join costs its result
    else:
        assert sizes is not None, "a QLever cost needs the patterns' sizes"
        reads = {k: v for k, v in (reads or {}).items() if v is not None}
        stitch = WorkStitchTable(list(true_of), true_of, sizes, var_masks, reads, cost)
        node_lazy = torch.tensor([r[1] < 0 and sizes[r[0].bit_length() - 1] >= cost.lazy_rows for r in node_rows])
        sorted_by = {}
        join_true = torch.full((N,), math.nan)
        for j, (m, c1, c2, _) in enumerate(node_rows):
            if c1 < 0:
                continue
            m1, m2 = node_rows[c1][0], node_rows[c2][0]
            for c, other, out in ((c1, m2, read1), (c2, m1, read2)):
                if node_lazy[c] and (other, node_rows[c][0].bit_length() - 1) in reads:
                    out[j] = math.log1p(reads[(other, node_rows[c][0].bit_length() - 1)])
            if m not in true_of:
                continue
            operands = [(node_rows[c][0], math.expm1(true_of[node_rows[c][0]]) if node_rows[c][0] in true_of else None,
                         node_rows[c][1] >= 0 and node_jv[c] != node_jv[j]) for c in (c1, c2)]
            w = join_work(cost, math.exp(true_of[m]), operands, reads, sizes)
            if w is not None:
                join_true[j] = math.log(w)

    def cost_to_go(ids):
        masks = [node_rows[i][0] for i in ids]
        if sorted_by is None:
            return stitch.cost_to_go(masks)
        return stitch.cost_to_go(masks, {node_rows[i][0]: node_jv[i] for i in ids if node_rows[i][1] >= 0})

    masks_needed = sorted({r[0] for r in node_rows} | {(1 << n) - 1})
    index = {m: i for i, m in enumerate(masks_needed)}
    fa = torch.full((N, 3), BIG)
    fb = torch.full((N, 3), BIG)
    for j, (m, c1, c2, _) in enumerate(node_rows):
        if c1 >= 0:
            fa[j] = merge_fan(node_rows[c1][0], node_rows[c2][0], touch_s, touch_o, fan)
            fb[j] = merge_fan(node_rows[c2][0], node_rows[c1][0], touch_s, touch_o, fan)
    Fn = len(forest_rows)
    forest_nodes = torch.full((Fn, n), -1, dtype=torch.long)
    forest_adj = torch.zeros(Fn, n, n, dtype=torch.bool)
    k = torch.tensor([len(r) for r in forest_rows])
    g_true = torch.full((Fn,), math.nan)
    needed = set(states)
    for s_id, c, child in cand_rows:
        if not math.isnan(float(join_true[c])):
            needed.add(child)
    for f, ids in enumerate(forest_rows):
        forest_nodes[f, :len(ids)] = torch.tensor(ids)
        forest_adj[f] = forest_adjacency([node_rows[i][0] for i in ids], neighbour_masks, n)
        if f in needed:
            g = cost_to_go(ids)
            g_true[f] = NEG if g == bushy.NEG_INF else (math.nan if g == math.inf else g)
    cand = torch.tensor(cand_rows, dtype=torch.long).reshape(-1, 3)
    q_true = torch.full((len(cand_rows),), math.nan)
    for j, (_, c, child) in enumerate(cand_rows):
        if not math.isnan(float(join_true[c])) and not math.isnan(float(g_true[child])):
            q_true[j] = float(torch.logaddexp(join_true[c], g_true[child]))

    pool_forest, pool_index, pool_set, pool_on, pool_k = [], [], [], [], []
    if pools:                                            # on path: true cost so far + G(child) = the best plan's
        best = stitch.cost_to_go(bushy.initial_forest(n))
        subtree: dict = {}                               # node -> true cost of its subtree (rows; nan unknown)

        def subtree_cost(i):
            if i not in subtree:
                r = node_rows[i]
                subtree[i] = 0.0 if r[1] < 0 else math.exp(float(join_true[i])) + subtree_cost(r[1]) + subtree_cost(r[2])
            return subtree[i]
        limit = math.exp(best) * (1 + 1e-6) if math.isfinite(best) else -1.0
        sets_seen = n_pools = 0
        for children, sets in pools:
            on = []
            for child in children:
                so_far = sum(subtree_cost(i) for i in forest_rows[child])
                ok = False
                if so_far <= limit:                      # nan compares False
                    if math.isnan(float(g_true[child])):
                        g = cost_to_go(forest_rows[child])
                        g_true[child] = NEG if g == bushy.NEG_INF else (math.nan if g == math.inf else g)
                    g = float(g_true[child])
                    ok = not math.isnan(g) and so_far + (0.0 if g <= NEG / 2 else math.exp(g)) <= limit
                on.append(ok)
            if not any(on):
                continue
            ids = {key: sets_seen + t for t, key in enumerate(dict.fromkeys(sets))}
            sets_seen += len(ids)
            pool_forest += children
            pool_index += [n_pools] * len(children)
            pool_set += [ids[key] for key in sets]
            pool_on += on
            pool_k.append(1 if forest_rows[children[0]] and len(forest_rows[children[0]]) == 1 else int(beam["width"]))
            n_pools += 1
    state_forest = torch.tensor(sorted(states, key=states.get), dtype=torch.long)
    raw_masks = [m for m in masks_needed if m & (m - 1) and m in counts]
    return {"n": n, "masks": torch.tensor(masks_needed, dtype=torch.long), "full": index[(1 << n) - 1],
            "node_mask": torch.tensor([index[r[0]] for r in node_rows]), "node_c1": torch.tensor([r[1] for r in node_rows]),
            "node_c2": torch.tensor([r[2] for r in node_rows]), "node_height": torch.tensor([r[3] for r in node_rows]),
            "node_size": torch.tensor([bin(r[0]).count("1") / n for r in node_rows]), "fa": fa, "fb": fb,
            "node_true": node_true, "node_failed": node_failed, "node_join_true": join_true,
            "node_vars": torch.tensor(node_vars, dtype=torch.long), "node_jv": torch.tensor(node_jv, dtype=torch.long),
            "node_lazy": node_lazy, "node_read1_true": read1, "node_read2_true": read2,
            "forest_nodes": forest_nodes, "forest_adj": forest_adj, "forest_k": k, "g_true": g_true,
            "state_forest": state_forest, "cand": cand, "q_true": q_true,
            "pool_forest": torch.tensor(pool_forest, dtype=torch.long), "pool_index": torch.tensor(pool_index, dtype=torch.long),
            "pool_set": torch.tensor(pool_set, dtype=torch.long), "pool_on": torch.tensor(pool_on, dtype=torch.bool),
            "pool_k": torch.tensor(pool_k, dtype=torch.long),
            "pool_b": torch.full((len(pool_k),), int(per_forest), dtype=torch.long),
            "raw_index": torch.tensor([index[m] for m in raw_masks], dtype=torch.long),
            "raw_true": torch.tensor([true_of[m] for m in raw_masks]),
            "raw_failed": torch.tensor([counts[m] is None for m in raw_masks]),
            "full_true": torch.tensor(math.log1p(counts[(1 << n) - 1]) if counts.get((1 << n) - 1) is not None else math.nan)}


def _take(x, idx):
    """x[idx] along dim 0 through index_select: its backward is an index_add (atomic adds), where x[idx]'s is a
    sort-based accumulating index_put, which took most of a training step's GPU time."""
    return x.index_select(0, idx.reshape(-1)).view(*idx.shape, *x.shape[1:])


def forward_batch(net: ForestPlanner, parts, phi, l, device, tau, stop_gradient=False, value_chunk=None, hidden=False):
    """Everything the losses need for a batch of build_query_data parts, given the encoder's phi [M, D] and l [M]
    for the concatenation of the parts' `masks` (single patterns: exact counts). Returns a dict of tensors.
    stop_gradient: the planner reads phi and l detached, so the plan, value, consistency and bounded-L1 losses stop
    at its inputs and only the L1 on l ("l" in the result) reaches the encoder. value_chunk: V in chunks of this many forests, with gradients under
    activation checkpointing (same result; memory of one chunk instead of the batch; also keeps a pass below the
    fused attention kernel's batch limit). Always "W" and "w_true" (W*, nan where undefined) per forest; hidden: also
    "H" (the value net's query-token state), for the epinet; parts' "raw_keys" / "forest_keys" pass through."""
    l_raw = l
    if stop_gradient:
        phi, l = phi.detach(), l.detach()
    off_m = off_n = off_f = off_s = 0
    width = max(p["n"] for p in parts)
    node_keys = ("node_height", "node_size", "fa", "fb", "node_true", "node_failed", "node_join_true", "node_vars",
                 "node_jv", "node_lazy", "node_read1_true", "node_read2_true")
    cat = {key: [] for key in ("node_mask", "node_c1", "node_c2", *node_keys,
                               "forest_nodes", "forest_adj", "forest_k", "g_true", "forest_q",
                               "state_forest", "cand", "q_true", "raw_index", "raw_true", "raw_failed", "forest_n",
                               "forest_full_true", "raw_part", "node_part", "forest_part", "state_part",
                               "pool_forest", "pool_index", "pool_set", "pool_on", "pool_k", "pool_b", "pool_part")}
    off_p = off_set = 0
    for i, p in enumerate(parts):
        N, Fn, n = len(p["node_mask"]), len(p["forest_nodes"]), p["n"]
        cat["node_mask"].append(p["node_mask"] + off_m)
        cat["node_c1"].append(torch.where(p["node_c1"] >= 0, p["node_c1"] + off_n, p["node_c1"]))
        cat["node_c2"].append(torch.where(p["node_c2"] >= 0, p["node_c2"] + off_n, p["node_c2"]))
        for key in (*node_keys, "forest_k", "g_true", "q_true", "raw_true", "raw_failed", "pool_on", "pool_k", "pool_b"):
            cat[key].append(p[key])
        cat["pool_forest"].append(p["pool_forest"] + off_f)
        cat["pool_index"].append(p["pool_index"] + off_p)
        cat["pool_set"].append(p["pool_set"] + off_set)
        cat["pool_part"].append(torch.full((len(p["pool_k"]),), i))
        off_p += len(p["pool_k"])
        off_set += int(p["pool_set"].max()) + 1 if len(p["pool_set"]) else 0
        fn = torch.full((Fn, width), -1, dtype=torch.long)
        fn[:, :n] = torch.where(p["forest_nodes"] >= 0, p["forest_nodes"] + off_n, p["forest_nodes"])
        cat["forest_nodes"].append(fn)
        adj = torch.zeros(Fn, width, width, dtype=torch.bool)
        adj[:, :n, :n] = p["forest_adj"]
        cat["forest_adj"].append(adj)
        cat["forest_q"].append(torch.full((Fn,), p["full"] + off_m))
        cat["forest_full_true"].append(p["full_true"].float().expand(Fn) if "full_true" in p else torch.full((Fn,), math.nan))
        cat["raw_part"].append(torch.full((len(p["raw_index"]),), i))
        cat["node_part"].append(torch.full((N,), i))
        cat["forest_part"].append(torch.full((Fn,), i))
        cat["state_part"].append(torch.full((len(p["state_forest"]),), i))
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

    # l'' and log subtree cost per node, level by level (single patterns: exact counts, cost 0); jc the log cost of
    # the node's own join (l'' for C_out; join_cost with the rows-read head for a QLever cost)
    qcost = net.cost is not None
    l_node = _take(l, b["node_mask"])
    cost = torch.full_like(l_node, NEG)
    l2 = l_node
    jc = l_node
    read1 = read2 = torch.full_like(l_node, NEG)
    for h in range(1, int(b["node_height"].max()) + 1 if len(b["node_height"]) else 1):
        idx = torch.nonzero(b["node_height"] == h).squeeze(1)
        if not len(idx):
            continue
        c1, c2 = b["node_c1"][idx], b["node_c2"][idx]
        lu, _ = net.bound(_take(l_node, idx), _take(l2, c1), _take(l2, c2), b["fa"][idx], b["fb"][idx],
                          _take(phi_std, b["node_mask"][idx]), b["node_size"][idx])
        l2 = l2.index_put((idx,), lu)
        ju = lu
        if qcost:
            jv = b["node_jv"][idx]
            sort1 = (b["node_jv"][c1] >= 0) & (b["node_jv"][c1] != jv)
            sort2 = (b["node_jv"][c2] >= 0) & (b["node_jv"][c2] != jv)
            r1, r2 = operand_reads(net, _take(phi_std, b["node_mask"][c1]), _take(phi_std, b["node_mask"][c2]),
                                   _take(l2, c1), _take(l2, c2))
            ju = join_cost(net, lu, _take(l2, c1), _take(l2, c2), b["node_lazy"][c1], b["node_lazy"][c2], r1, r2, sort1, sort2)
            read1, read2 = read1.index_put((idx,), r1), read2.index_put((idx,), r2)
        jc = jc.index_put((idx,), ju)
        cost = cost.index_put((idx,), torch.logaddexp(ju, torch.logaddexp(_take(cost, c1), _take(cost, c2))))

    # V of every forest
    fn = b["forest_nodes"]
    valid = fn >= 0
    safe = fn.clamp(min=0)
    single = b["node_height"][safe] == 0
    tokens = token_features(_take(phi_std, b["node_mask"][safe]), _take(l2, safe), _take(cost, safe), single, b["node_size"][safe])
    tokens = tokens * valid[..., None]
    k = b["forest_k"]
    lq = _take(l, b["forest_q"])
    query = torch.cat([_take(phi_std, b["forest_q"]), (lq / 10)[:, None], (k / b["forest_n"])[:, None]], 1)
    args = (tokens, ~valid, b["forest_adj"], query, lq + _log_weight(net.cost.out) if qcost else lq, k)
    call = lambda *a: net.values(*a, return_hidden=True)
    if value_chunk and len(fn) > value_chunk:
        run = (lambda *a: checkpoint(call, *a, use_reentrant=False)) if torch.is_grad_enabled() else call
        outs = [run(*(a[s:s + value_chunk] for a in args)) for s in range(0, len(fn), value_chunk)]
        V, W, H = (torch.cat(t) for t in zip(*outs))
    else:
        V, W, H = call(*args)
    if qcost:                                            # 2 components: V is the cost of their final join
        two = torch.nonzero(k == 2).squeeze(1)
        if len(two):
            n0, n1 = fn[two, 0], fn[two, 1]
            jq = b["node_vars"][n0] & b["node_vars"][n1]
            s0 = (b["node_jv"][n0] >= 0) & (b["node_jv"][n0] != jq)
            s1 = (b["node_jv"][n1] >= 0) & (b["node_jv"][n1] != jq)
            r0, r1 = operand_reads(net, _take(phi_std, b["node_mask"][n0]), _take(phi_std, b["node_mask"][n1]),
                                   _take(l2, n0), _take(l2, n1))
            final = join_cost(net, _take(lq, two), _take(l2, n0), _take(l2, n1), b["node_lazy"][n0], b["node_lazy"][n1],
                              r0, r1, s0, s1)
            V = V.index_put((two,), final)
    b["W"] = W
    if hidden:
        b["H"] = H
    g, ft = b["g_true"], b["forest_full_true"]                  # W*(F) = log(G(F) - (c(Q) + 1)), where defined
    ok = ~torch.isnan(g) & ~torch.isnan(ft) & (k >= 3) & (g - ft > 1e-3)
    b["w_true"] = torch.where(ok, g + torch.log(-torch.expm1(-(g - ft).clamp(min=1e-3))), torch.full_like(g, math.nan))
    for key in ("raw_keys", "forest_keys"):
        if all(key in p for p in parts):
            b[key] = torch.cat([p[key] for p in parts]).to(device)

    # per candidate: estimated and true cost-to-go from its state
    cand = b["cand"]
    q_hat = torch.logaddexp(_take(jc, cand[:, 1]), _take(V, cand[:, 2]))
    # per pool candidate (beam loss): the planner's score total^ = logaddexp(log cost so far^, V)
    pool_score = None
    if len(b["pool_forest"]):
        pf = fn[b["pool_forest"]]
        sub = torch.where(pf >= 0, _take(cost, pf.clamp(min=0)), torch.full_like(pf, NEG, dtype=cost.dtype))
        pool_score = torch.logaddexp(torch.logsumexp(sub, 1), _take(V, b["pool_forest"]))
    return {"l": l_raw, "l2": l2, "V": V, "q_hat": q_hat, "jc": jc, "read1": read1, "read2": read2,
            "pool_score": pool_score, **b}


def _segment_logsumexp(x, seg, size):
    peak = torch.full((size,), -math.inf, device=x.device, dtype=x.dtype).scatter_reduce(0, seg, x, "amax")
    safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    total = torch.zeros(size, device=x.device, dtype=x.dtype).index_add(0, seg, torch.exp(x - safe[seg]))
    return safe + torch.log(total)


def _beam_pairs(score, pool, sets, on, k_of, b_of):
    """The beam loss's (c*, comparator) index pairs (module text) from detached scores, as numpy arrays: per pool
    with an on-path candidate, c* its best-scored one, against the b-th best other build of c*'s forest and the k-th
    best other candidate passing the per-forest cut. Also (pools with an on-path candidate, pools where c* survives
    the beam's selection under these scores)."""
    import numpy as np
    order = np.lexsort((np.arange(len(score)), score, pool))            # by pool, then score, then index
    groups = np.split(order, np.flatnonzero(np.diff(pool[order])) + 1) if len(order) else []
    stars, comps, n_pools, survived = [], [], 0, 0
    for grp in groups:
        on_g = on[grp]
        if not on_g.any():
            continue
        n_pools += 1
        k, b = int(k_of[pool[grp[0]]]), int(b_of[pool[grp[0]]])
        pos = int(np.argmax(on_g))
        sets_g = sets[grp]
        rank, seen = np.zeros(len(grp), dtype=np.int64), {}
        for t, st in enumerate(sets_g.tolist()):
            rank[t] = seen.get(st, 0)
            seen[st] = rank[t] + 1
        same = [t for t in range(len(grp)) if sets_g[t] == sets_g[pos] and t != pos]
        if len(same) >= b:
            stars.append(grp[pos]), comps.append(grp[same[b - 1]])
        passing = [t for t in range(len(grp)) if rank[t] < b and t != pos]
        if len(passing) >= k:
            stars.append(grp[pos]), comps.append(grp[passing[k - 1]])
        survived += int(rank[pos] < b and sum(t < pos for t in passing) < k)
    return np.array(stars, dtype=np.int64), np.array(comps, dtype=np.int64), n_pools, survived


def losses(out, tau, weights=None):
    """{l1_raw, l1_bound, value, w, plan, floor, consistency, l1_read, beam} means (0 when empty) and coverage
    counts. weights [P]: per query (part) of the batch, each term a weighted mean over its rows (None: all 1)."""
    zero = out["l"].sum() * 0
    res = {}

    def mean(x, part):
        if not len(x):
            return zero
        if weights is None:
            return x.mean()
        w = weights[part]
        return (x * w).sum() / w.sum().clamp(min=1e-12)

    def l1(est, true, failed, part):
        return mean(torch.where(failed, torch.relu(true - est), (est - true).abs()), part)
    res["l1_raw"] = l1(out["l"][out["raw_index"]], out["raw_true"], out["raw_failed"], out["raw_part"])
    known = ~torch.isnan(out["node_true"])
    res["l1_bound"] = l1(out["l2"][known], out["node_true"][known], out["node_failed"][known], out["node_part"][known])
    vk = ~torch.isnan(out["g_true"]) & (out["forest_k"] >= 3)
    res["value"] = mean((out["V"][vk] - out["g_true"][vk]).abs(), out["forest_part"][vk])
    wk = ~torch.isnan(out["w_true"])
    res["w"] = mean((out["W"][wk] - out["w_true"][wk]).abs(), out["forest_part"][wk])

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
        part = out["state_part"][has]
        res["plan"] = mean(regret(qh), part)
        with torch.no_grad():
            res["floor"] = mean(regret(qt), part)
    else:
        res["plan"], res["floor"] = zero, zero
    sk = out["forest_k"][out["state_forest"]] >= 3
    if bool(sk.any()):
        target = torch.full((S,), math.inf, device=cand.device).scatter_reduce(0, cand[:, 0], out["q_hat"].detach(), "amin")
        res["consistency"] = mean((out["V"][out["state_forest"]][sk] - target[sk]).abs(), out["state_part"][sk])
    else:
        res["consistency"] = zero
    res["states_with_choice"] = int((n_known >= 2).sum())
    res["states"] = S

    # rows read by lazily read pattern operands (QLever cost): r^ against the measured read
    rows_r, est_r, true_r = [], [], []
    for side, est in ((1, out["read1"]), (2, out["read2"])):
        known_r = ~torch.isnan(out[f"node_read{side}_true"])
        rows_r.append(out["node_part"][known_r]), est_r.append(est[known_r]), true_r.append(out[f"node_read{side}_true"][known_r])
    res["l1_read"] = mean((torch.cat(est_r) - torch.cat(true_r)).abs(), torch.cat(rows_r))
    res["read_labels"] = int(sum(len(r) for r in rows_r))

    # the beam loss over the pools of traced beams
    res["beam"], res["pools"], res["pools_kept"] = zero, 0, 0
    if out.get("pool_score") is not None:
        score = out["pool_score"]
        stars, comps, res["pools"], res["pools_kept"] = _beam_pairs(
            score.detach().double().cpu().numpy(), out["pool_index"].cpu().numpy(), out["pool_set"].cpu().numpy(),
            out["pool_on"].cpu().numpy(), out["pool_k"].cpu().numpy(), out["pool_b"].cpu().numpy())
        if len(stars):
            st = torch.as_tensor(stars, device=score.device)
            ct = torch.as_tensor(comps, device=score.device)
            hinge = tau * F.softplus((score[st] - score[ct]) / tau)
            res["beam"] = mean(hinge, out["pool_part"][out["pool_index"][st]])
    return res
