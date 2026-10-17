"""An epinet on the forest value planner (forest_value.py): epistemic uncertainty of its subset estimates and of
its value, for Thompson sampling of plans and for choosing which queries to count and train on.

Following Osband et al., "Epistemic Neural Networks" (2023) and Wen et al., "Approximate Thompson Sampling via
Epistemic Neural Networks" (2023), one epistemic index z ~ N(0, I_D) drives two heads:
    l_z(U) = l(U) + s_card  * e_card(sg[phi(U)], z)          the backbone's log-card estimate of a subset U
    W_z(F) = W(F) + s_value * e_value(sg[h(F)], z)           the value net's output, V_z(F) = logaddexp(l_z(Q), W_z(F))
    e(x, z) = mlp_learnable([x, z]) . z + alpha * mlp_prior([x, z]) . z      (src/models/mlp_epinet.py)
phi is the standardised subset embedding, h the value net's query-token state; the prior MLP is frozen, alpha sets
its spread over z to `prior_scale_target` log units, and the learnable MLP's last layer starts at zero. One z per
plan is a joint sample: every subset and forest of that plan is judged by the same plausible model (the property
approximate Thompson sampling needs, Wen et al.). z = 0 gives the plain planner exactly; s_card, s_value scale the
samples (1, or fitted each round, `calibrate`).

Training (ENN paper eq. 9, Gaussian bootstrap; epinet parameters only, features and base detached):
    card   |l_z(U) - (log1p count(U) + sigma c_U . z)|    counted subsets, one-sided for failed counts
    value  |V_z(F) - (log G(F)   + sigma c_F . z)|        forests with a stitched G (3+ components)
averaged over K indexes, with c a fixed unit vector per (query, subset) / (query, forest) (anchor_keys).

Priority (Osband et al., "Fine-Tuning Language Models via Epistemic Neural Networks", 2023, the "variance"
priority): Var_z V_z(F0), the epistemic variance of the predicted log cost of the whole query (its start forest
F0 of single patterns), query_priorities.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from src.models.mlp_epinet import MLPEpinet
from src.supervised_value_estimation.amortized_dp import forest_value as fv
from src.utils.epinet_utils.epinet_loss import anchor_vectors, stable_key

Z90 = 1.6448536269514722                        # standard normal 95% quantile: a central 90% interval is +- Z90 std


class ForestEpinet(nn.Module):
    def __init__(self, phi_dim=200, value_dim=64, index_dim=16, hidden_dim=64, prior_hidden_dim=128):
        super().__init__()
        self.config = {"phi_dim": phi_dim, "value_dim": value_dim, "index_dim": index_dim, "hidden_dim": hidden_dim,
                       "prior_hidden_dim": prior_hidden_dim}
        self.index_dim = index_dim
        self.card = MLPEpinet(phi_dim, index_dim, ["card"], hidden_dim, prior_hidden_dim)
        self.value = MLPEpinet(value_dim, index_dim, ["value"], hidden_dim, prior_hidden_dim)
        for name in ("alpha_card", "alpha_value", "scale_card", "scale_value"):
            self.register_buffer(name, torch.tensor(1.0))

    def trainable(self):
        return self.card.get_learnable_epinet_params() + self.value.get_learnable_epinet_params()

    @staticmethod
    def _rows(epinet, head, alpha, x, z):
        """Per-row correction e(x_i, z_i) [N] (one index per row)."""
        inp = torch.cat([x, z], 1)
        learnable = (epinet.learnable_epinet_heads[head](epinet.learnable_epinet_features(inp)) * z).sum(1)
        prior = (epinet.prior_epinet_heads[head](epinet.prior_epinet_features(inp)) * z).sum(1)
        return learnable + alpha * prior

    def card_shift(self, phi_std, z):
        """s_card * e_card(phi, z) per row [N]: phi_std [N, phi_dim] standardised, z [N, D]."""
        return self.scale_card * self._rows(self.card, "card", self.alpha_card, phi_std.detach(), z)

    def value_shift(self, h, z):
        return self.scale_value * self._rows(self.value, "value", self.alpha_value, h.detach(), z)

    def card_samples(self, phi_std, Z):
        """e_card [K, N] for every index of Z [K, D] (unscaled)."""
        return self.card.sample(torch.zeros(len(phi_std), device=Z.device), phi_std.detach(), Z, "card", self.alpha_card)

    def value_samples(self, h, Z):
        return self.value.sample(torch.zeros(len(h), device=Z.device), h.detach(), Z, "value", self.alpha_value)

    @torch.no_grad()
    def set_prior_scale(self, phi_std, h, target):
        """alpha so that each prior's spread over z is `target` log units on these features."""
        self.alpha_card.fill_(target / max(self.card.prior_scale(phi_std, head_name="card"), 1e-6))
        self.alpha_value.fill_(target / max(self.value.prior_scale(h, head_name="value"), 1e-6))

    def save(self, path):
        torch.save({"config": self.config, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device):
        c = torch.load(path, map_location=device)
        epi = cls(**c["config"])
        epi.load_state_dict(c["state"])
        return epi.to(device)


def anchor_keys(part, query_key):
    """{"raw_keys", "forest_keys"} int64 [R] / [F] for a build_query_data part: stable keys of (query, counted subset)
    and (query, forest as its set of component masks), so a subset's / forest's anchor c is the same every round."""
    masks = part["masks"].tolist()
    raw = [stable_key(query_key, masks[i]) for i in part["raw_index"].tolist()]
    node_mask = part["node_mask"].tolist()
    forests = [stable_key(query_key, *sorted(masks[node_mask[i]] for i in row if i >= 0))
               for row in part["forest_nodes"].tolist()]
    return {"raw_keys": torch.tensor(raw, dtype=torch.long), "forest_keys": torch.tensor(forests, dtype=torch.long)}


def _anchors(keys, dim, device):
    return anchor_vectors(keys.cpu().numpy(), dim, device=device)


def losses(epi, net, out, phi, sigma, n_indexes, max_rows, generator=None):
    """{"card", "value"}: the epinet losses on a forward_batch(..., hidden=True) result whose parts carry anchor
    keys (anchor_keys); phi the encoder's embeddings it was given. At most max_rows subsets / forests per head (a
    random subset of them)."""
    device = out["l"].device
    D = epi.index_dim
    Z = torch.randn(n_indexes, D, device=device, generator=generator)
    zero = out["l"].sum() * 0
    res = {}

    def pick(n):
        if n <= max_rows:
            return torch.arange(n, device=device)
        return torch.randperm(n, device=device, generator=generator)[:max_rows]

    raw = out["raw_index"]
    if len(raw):
        r = pick(len(raw))
        phi_r = net.standardise(phi[raw[r]])
        base = out["l"][raw[r]].detach()
        true, failed = out["raw_true"][r], out["raw_failed"][r]
        target = true + sigma * (Z @ _anchors(out["raw_keys"][r], D, device).T)                      # [K, R]
        pred = base + epi.card_samples(phi_r, Z)
        res["card"] = torch.where(failed, torch.relu(target - pred), (pred - target).abs()).mean()
    else:
        res["card"] = zero
    known = (~torch.isnan(out["g_true"]) & (out["forest_k"] >= 3)).nonzero().squeeze(1)
    if len(known):
        f = known[pick(len(known))]
        W, lq = out["W"][f].detach(), out["l"][out["forest_q"][f]].detach()
        target = out["g_true"][f] + sigma * (Z @ _anchors(out["forest_keys"][f], D, device).T)
        pred = torch.logaddexp(lq, W + epi.value_samples(out["H"][f], Z))
        res["value"] = (pred - target).abs().mean()
    else:
        res["value"] = zero
    return res


@torch.no_grad()
def start_forests(net, items, device):
    """The value net on each query's start forest F0 (every pattern its own component). items: [(phi [n + 1, D_phi],
    l [n + 1], neighbour_masks)], single patterns first and the full query last (exact counts for singles).
    Returns (q_std [B, D_phi], l(Q) [B], k = n [B], W [B], h [B, d])."""
    width = max(len(nb) for _, _, nb in items)
    B = len(items)
    D_phi = items[0][0].shape[1]
    phi_s = torch.zeros(B, width, D_phi, device=device)
    l_s = torch.zeros(B, width, device=device)
    pad = torch.ones(B, width, dtype=torch.bool, device=device)
    adj = torch.zeros(B, width, width, dtype=torch.bool, device=device)
    phi_q = torch.zeros(B, D_phi, device=device)
    lq = torch.zeros(B, device=device)
    k = torch.zeros(B, dtype=torch.long, device=device)
    for b, (phi, l, nb) in enumerate(items):
        n = len(nb)
        phi_s[b, :n], l_s[b, :n], pad[b, :n], k[b] = phi[:n].to(device), l[:n].to(device), False, n
        phi_q[b], lq[b] = phi[n].to(device), l[n].to(device)
        adj[b, :n, :n] = torch.tensor([[i != j and bool(nb[i] >> j & 1) for j in range(n)] for i in range(n)], device=device)
    on = ~pad
    size = on.float() / k[:, None]
    tokens = fv.token_features(net.standardise(phi_s), l_s, torch.full_like(l_s, fv.NEG), on, size) * on[..., None]
    q_std = net.standardise(phi_q)
    query = torch.cat([q_std, (lq / 10)[:, None], torch.ones(B, 1, device=device)], 1)
    _, W, h = net.values(tokens, pad, adj, query, lq, k, return_hidden=True)
    return q_std, lq, k, W, h


@torch.no_grad()
def query_priorities(net, epi, items, device, n_indexes=32, generator=None, chunk=1024):
    """Var_z V_z(F0) per query (items as start_forests): the epistemic variance of the predicted log cost of the
    whole query. Larger = the model is less sure what the query costs."""
    out = []
    Z = torch.randn(n_indexes, epi.index_dim, device=device, generator=generator)
    for s in range(0, len(items), chunk):
        q_std, lq, k, W, h = start_forests(net, items[s:s + chunk], device)
        lq_z = lq + epi.scale_card * epi.card_samples(q_std, Z)                                   # [K, B]
        v = torch.logaddexp(lq_z, W + epi.scale_value * epi.value_samples(h, Z))
        out.append(torch.where(k == 2, lq_z, v).var(0))
    return torch.cat(out) if out else torch.zeros(0, device=device)


@torch.no_grad()
def calibration(samples, base, true):
    """Out-of-sample check of an epinet head: samples [K, N] of the estimate, base [N] the deployed estimate, true
    [N]. Coverage of the central 90% interval of the samples, Spearman of the samples' std with |base - true|, and
    the scale that would give a Gaussian 90% interval of +- Z90 std that covers 90%: q90(|error| / std) / Z90."""
    if samples.shape[1] < 10:
        return None
    lo, hi = torch.quantile(samples, 0.05, dim=0), torch.quantile(samples, 0.95, dim=0)
    std = samples.std(0).clamp(min=1e-6)
    err = (base - true).abs()
    rho = spearmanr(std.cpu().numpy(), err.cpu().numpy()).statistic
    return {"n": int(samples.shape[1]), "coverage90": round(float(((true >= lo) & (true <= hi)).float().mean()), 4),
            "spearman": round(float(rho), 4) if not math.isnan(rho) else None,
            "mean_std": round(float(std.mean()), 4), "mean_abs_error": round(float(err.mean()), 4),
            "fitted_scale": round(float(torch.quantile(err / std, 0.9) / Z90), 4)}


def scaled_card_samples(epi, net, phi, l, Z):
    """[K, N] l_z of subsets at the current sample scale."""
    return l[None] + epi.scale_card * epi.card_samples(net.standardise(phi), Z)


def fit_scale(current, cal, lo=0.5, hi=20.0, step=0.5):
    """The sample scale after a calibration: a step of `step` in log space from the current scale towards the
    fitted one (the epinet trains meanwhile, so a full step overshoots), clamped to [lo, hi]."""
    if cal is None or not np.isfinite(cal["fitted_scale"]) or cal["fitted_scale"] <= 0:
        return current
    return float(min(max(current * cal["fitted_scale"] ** step, lo), hi))
