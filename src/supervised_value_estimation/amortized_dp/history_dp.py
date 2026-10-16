"""History-conditioned cardinality estimation inside the bushy DP (soft_dp.py): one model, one loss.

The DP decides each subset's history. Bottom-up by size, for every subset U:
    p(split | U)  = softmax over U's splits of -log(B(A) + B(B)) / tau        (does not involve card(U))
    M_U           = sum_splits p * (M_A + M_B) + phi(k_U) v_U^T               linear-attention memory (Katharopoulos et
                                                                             al. 2020): attention over U's whole plan,
                                                                             earlier nodes weighted by P(in U's plan)
    L(U)          = direct(U) + sum_splits p * [(L(A) - L0(A)) + (L(B) - L0(B))]
                    + delta(phi_U, readout(q_U, M_U), expected child correction, direct(U))
    B(U)          = card(U) + E_p[B(A) + B(B)]
L = log1p card; direct = the cardinality model's estimate from its pooled embedding phi_U; L0 = direct with the exact
counts of single patterns. The children's corrections flow up (a child corrected towards 0 rows pulls U with it) and
delta reads U's history to keep or undo them. delta = 0 at init: the model starts as the cardinality model itself.
At tau = 0 p is one-hot and U's history is exactly the bushy subtree the DP chose for U (the plan so far).

At inference the estimates are clamped by `degree_bound`: L'(U) = min(L(U), L'(U minus p) + log maxdeg(p)) over the
single patterns p whose removal leaves U connected, with p's predicate's largest degree on the shared side
(1 if p's other end is a constant or both ends are shared). It is a true upper bound when L'(U minus p) is exact.

Training (online/train_plan_loss.py with `history.enabled`): L1 on counted subsets + the plan loss through this DP,
gradients into both the backbone (phi, direct) and the head.
"""
from __future__ import annotations

import math

import torch

from src.supervised_value_estimation.amortized_dp import soft_dp

SPLIT_CHUNK = 1 << 19                   # splits per index_add when building history memories


def popcount(x: torch.Tensor) -> torch.Tensor:
    c = torch.zeros_like(x)
    for i in range(16):
        c += (x >> i) & 1
    return c


def single_pattern_logcounts(data, masks: torch.Tensor) -> torch.Tensor:
    """log1p exact count of each single-pattern mask (edge feature -2 of the pattern's forward edge); other masks get
    an arbitrary value (they are not used)."""
    n = len(data.triple_patterns)
    exact = data.edge_attr[0::2, -2].to(masks.device).float()
    return exact[torch.log2(masks.double()).round().long().clamp(min=0, max=n - 1)]


def merge(parts, device) -> dict:
    """One batch from several queries' structures (disjoint): parts = [{"structure": SubsetStructure,
    "exact": [S] (single patterns' log1p counts), optional "label" [S] (log1p count, nan = none), "failed" [S] bool,
    "true" [S] (log1p count, failed = the bound), "l1" [S] bool (which subsets give L1 terms)}]. Levels are merged by
    subset size; phi and direct are added by the caller (aligned with the concatenated masks)."""
    offset, levels, fulls, qid = 0, {}, [], []
    cols = {k: [] for k in ("single", "exact", "label", "failed", "true", "l1")}
    for j, part in enumerate(parts):
        s = part["structure"]
        S = len(s)
        cols["single"].append(s.singleton)
        cols["exact"].append(part["exact"].float())
        cols["label"].append(part.get("label", torch.full((S,), math.nan, device=device)))
        cols["failed"].append(part.get("failed", torch.zeros(S, dtype=torch.bool, device=device)))
        cols["true"].append(part.get("true", torch.zeros(S, device=device)))
        cols["l1"].append(part.get("l1", torch.ones(S, dtype=torch.bool, device=device)))
        for subsets, u, s1, s2 in s.levels:
            size = int(popcount(s.masks[subsets[:1]])[0])
            levels.setdefault(size, []).append((subsets + offset, u + offset, s1 + offset, s2 + offset))
        fulls.append(s.full + offset if s.full >= 0 else -1)
        qid.append(torch.full((S,), j, dtype=torch.long, device=device))
        offset += S
    b = {k: torch.cat([x.to(device) for x in v]) for k, v in cols.items()}
    b["levels"] = [tuple(torch.cat([x[i] for x in levels[k]]) for i in range(4)) for k in sorted(levels)]
    b["fulls"] = torch.tensor(fulls, device=device)
    b["qid"] = torch.cat(qid)
    b["n_queries"] = len(parts)
    return b


def _segment_log_softmax(score, u, size):
    peak = torch.full((size,), -math.inf, device=score.device, dtype=score.dtype).scatter_reduce(0, u, score, "amax")
    total = torch.zeros(size, device=score.device, dtype=score.dtype).index_add(0, u, torch.exp(score - peak[u]))
    return score - peak[u] - torch.log(total[u])


class HistoryDP(torch.nn.Module):
    def __init__(self, phi_dim=200, hidden=128, memory=16):
        super().__init__()
        self.config = {"phi_dim": phi_dim, "hidden": hidden, "memory": memory}
        self.d = memory
        self.key = torch.nn.Linear(phi_dim + 2, memory)
        self.value = torch.nn.Linear(phi_dim + 2, memory)
        self.query = torch.nn.Linear(phi_dim, memory)
        self.delta = torch.nn.Sequential(torch.nn.Linear(phi_dim + memory + 2, hidden), torch.nn.ReLU(),
                                         torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.delta[-1].weight)          # delta = 0: starts as the cardinality model
        torch.nn.init.zeros_(self.delta[-1].bias)
        self.register_buffer("mean", torch.zeros(phi_dim))   # phi standardisation (set from a sample)
        self.register_buffer("std", torch.ones(phi_dim))

    @staticmethod
    def feature(x):
        return torch.nn.functional.elu(x) + 1

    def own(self, x, L, direct):
        inp = torch.cat([x, L[:, None] / 10, direct[:, None] / 10], 1)
        k, v = self.feature(self.key(inp)), self.value(inp)
        return (k[:, :, None] * v[:, None, :]).reshape(len(x), -1), k

    def forward(self, b, phi, direct, tau):
        """(L [S], log B [S], log p per level) for the batch `b` (merge) with the subsets' phi and direct estimates."""
        x = (phi - self.mean) / self.std
        single = b["single"]
        L0 = torch.where(single, b["exact"].to(direct.dtype), direct)
        S = len(L0)
        L = L0
        M, z = self.own(x, L0, direct)
        logb = torch.where(single, torch.full_like(L0, soft_dp.FLOOR), torch.full_like(L0, soft_dp.UNBUILT))
        log_p = []
        for subsets, u, s1, s2 in b["levels"]:
            v = torch.logaddexp(logb[s1], logb[s2])
            if tau > 0:
                lp = _segment_log_softmax(-v / tau, u, S)
            else:                                            # hard DP: one-hot on the first cheapest split (soft_dp)
                best = torch.full((S,), math.inf, device=v.device, dtype=v.dtype).scatter_reduce(0, u, v.detach(), "amin")
                order = torch.arange(len(u), device=v.device)
                hit = torch.where(v.detach() == best[u], order, torch.full_like(order, len(u)))
                first = torch.full((S,), len(u), device=v.device, dtype=order.dtype).scatter_reduce(0, u, hit, "amin")
                lp = torch.where(order == first[u], torch.zeros_like(v), torch.full_like(v, -math.inf))
            log_p.append(lp)
            pw = torch.exp(lp)
            corr = L - L0
            ecorr = torch.zeros(S, device=L.device, dtype=L.dtype).index_add(0, u, pw * (corr[s1] + corr[s2]))
            mh = torch.zeros(S, self.d * self.d, device=L.device, dtype=M.dtype)
            zh = torch.zeros(S, self.d, device=L.device, dtype=z.dtype)
            for c in range(0, len(u), SPLIT_CHUNK):          # history: sum over splits of p (M_A + M_B)
                sl = slice(c, c + SPLIT_CHUNK)
                w = pw[sl][:, None].to(M.dtype)
                mh = mh.index_add(0, u[sl], w * (M[s1[sl]] + M[s2[sl]]))
                zh = zh.index_add(0, u[sl], w * (z[s1[sl]] + z[s2[sl]]))
            mh, zh = mh[subsets], zh[subsets]
            q = self.feature(self.query(x[subsets]))
            read = torch.bmm(q[:, None, :], mh.view(-1, self.d, self.d)).squeeze(1) / ((q * zh).sum(1, keepdim=True) + 1e-6)
            delta = self.delta(torch.cat([x[subsets], read, ecorr[subsets, None].to(x.dtype) / 10,
                                          direct[subsets, None].to(x.dtype) / 10], 1)).squeeze(1)
            Lu = direct[subsets] + ecorr[subsets] + delta.to(L.dtype)
            L = L.index_put((subsets,), Lu)
            m_own, k_own = self.own(x[subsets], Lu.to(x.dtype), direct[subsets].to(x.dtype))
            M = M.index_put((subsets,), mh + m_own)
            z = z.index_put((subsets,), zh + k_own)
            build = soft_dp._segment_logsumexp(lp + v, u, S)[subsets]
            logb = logb.index_put((subsets,), torch.logaddexp(Lu, build))
        return L, logb, log_p

    def save(self, path, extra=None):
        torch.save({"config": self.config, "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        net = cls(**checkpoint["config"])
        net.load_state_dict(checkpoint["state"])
        return net.to(device)


def marginals(b, log_p) -> torch.Tensor:
    """Plan marginals of every query's full set (soft_dp.plan_marginals, batched over the merged queries)."""
    S = len(b["single"])
    mu = torch.zeros(S, device=b["single"].device, dtype=log_p[0].dtype if log_p else torch.float32)
    roots = b["fulls"][b["fulls"] >= 0]
    mu = mu.index_put((roots,), torch.ones(len(roots), device=mu.device, dtype=mu.dtype))
    for (subsets, u, s1, s2), lp in zip(reversed(b["levels"]), reversed(log_p)):
        flow = mu[u] * torch.exp(lp)
        mu = mu.index_add(0, s1, flow).index_add(0, s2, flow)
    return mu


def l1_terms(L, b):
    """Per-subset L1 terms on log1p counts (a failed count: one-sided, only estimates below its bound) and the mask
    of the counted joins that give one (and are selected by b["l1"])."""
    joins = ~b["single"]
    labelled = ~torch.isnan(b["label"]) & joins
    err = torch.where(b["failed"], torch.relu(b["true"] - L), (L - torch.nan_to_num(b["label"])).abs())
    return err, (labelled | (b["failed"] & joins)) & b["l1"]


def plan_losses(b, log_p, best) -> tuple[torch.Tensor, torch.Tensor]:
    """Per query log E_p[true C_out] - log(the optimum `best`), and which queries have one (a complete plan from
    their structure and a finite optimum): train_plan_loss.query_loss's plan term, batched."""
    joins = ~b["single"]
    mu = marginals(b, log_p)
    w = mu * joins
    val = torch.where(w > 0, torch.log(w.clamp(min=1e-30)) + b["true"].to(w.dtype), torch.full_like(w, -math.inf))
    nq = b["n_queries"]
    peak = torch.full((nq,), -math.inf, device=w.device, dtype=w.dtype).scatter_reduce(0, b["qid"], val, "amax")
    safe = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    total = torch.zeros(nq, device=w.device, dtype=w.dtype).index_add(0, b["qid"], torch.exp(val - safe[b["qid"]]))
    ok = (b["fulls"] >= 0) & torch.isfinite(best) & (total > 0)
    return safe + torch.log(total.clamp(min=1e-30)) - best.to(w.dtype), ok


def degree_bound(patterns, structure, logcard, degrees) -> torch.Tensor:
    """min(logcard(U), bound(U minus p) + log maxdeg(p)) bottom-up over single-pattern splits (see the module text).
    degrees: {predicate N3: {"s": {"max": ...}, "o": {"max": ...}}} (objects per subject / subjects per object)."""
    spo = [p.strip().rstrip(" .").split() for p in patterns]
    spo = [(t[0], t[1], " ".join(t[2:])) for t in spo]
    touch_s, touch_o, fan_s, fan_o = [], [], [], []
    for i, (s, p, o) in enumerate(spo):
        touch = lambda v: sum(1 << j for j, (s2, _, o2) in enumerate(spo) if j != i and v.startswith("?") and v in (s2, o2))
        touch_s.append(touch(s)), touch_o.append(touch(o))
        d = degrees.get(p, {})
        fan_s.append(0.0 if not o.startswith("?") else math.log(max(d.get("s", {}).get("max", 1.0), 1.0)))
        fan_o.append(0.0 if not s.startswith("?") else math.log(max(d.get("o", {}).get("max", 1.0), 1.0)))
    device = logcard.device
    touch_s, touch_o = torch.tensor(touch_s, device=device), torch.tensor(touch_o, device=device)
    fan_s = torch.tensor(fan_s, device=device, dtype=logcard.dtype)
    fan_o = torch.tensor(fan_o, device=device, dtype=logcard.dtype)
    s = structure
    out = logcard.clone()
    for subsets, u, s1, s2 in s.levels:
        cands, targets = [], []
        for single, rest in ((s1, s2), (s2, s1)):
            one = s.singleton[single]
            if not one.any():
                continue
            i = torch.log2(s.masks[single[one]].double()).round().long()
            a = s.masks[rest[one]]
            shared_s, shared_o = (a & touch_s[i]) != 0, (a & touch_o[i]) != 0
            fan = torch.where(shared_s & shared_o, torch.zeros_like(fan_s[i]), torch.where(shared_s, fan_s[i], fan_o[i]))
            cands.append(out[rest[one]] + fan)
            targets.append(u[one])
        if cands:
            best = torch.full_like(out, math.inf).scatter_reduce(0, torch.cat(targets), torch.cat(cands), "amin")
            out[subsets] = torch.minimum(out[subsets], best[subsets])
    return out
