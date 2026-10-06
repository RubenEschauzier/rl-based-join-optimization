"""The epinet training term, independent of the base network and of how data points are keyed.

For a data point x with target y, its fixed anchor vector c (unit norm, so z.c ~ N(0, 1)) and
K sampled indices z, the epinet is fit to perturbed targets:

    pred_z = sg[base(x)] + learnable(sg[phi(x)], z) + alpha_mlp * prior(sg[phi(x)], z)
             (+ alpha_ensemble * ensemble_prior(x, z), if the caller has one)
    loss   = loss_fn(pred_z, y + sigma * (z . c))

The base receives no gradient from this term; the caller adds its own base loss.

Anchors must stay fixed per data point across steps. `loss_epinet` derives them from a seeded
generator in plan order (one seed per query); `anchor_vectors` derives them from integer keys
instead, so any data point with a stable identity (e.g. a query + a set of joined patterns)
gets the same anchor wherever and whenever it appears in a batch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def stable_key(*parts) -> int:
    """A 63-bit integer key from strings / integers, stable across processes (unlike hash())."""
    digest = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def _splitmix64(values):
    with np.errstate(over="ignore"):
        z = values + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def anchor_vectors(keys, epi_index_dim, anchor_seed=0, device=torch.device("cpu")):
    """(n, D) unit-norm Gaussian anchors, a deterministic function of (key, anchor_seed).

    Counter-based (splitmix64 + Box-Muller), so it is vectorised and needs no generator state.
    """
    keys = np.asarray(keys, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        base = _splitmix64(keys ^ _splitmix64(np.uint64(anchor_seed) * np.uint64(0x2545F4914F6CDD1D)))
        counters = base[:, None] * np.uint64(2 * epi_index_dim + 1) + np.arange(2 * epi_index_dim, dtype=np.uint64)
    bits = _splitmix64(counters) >> np.uint64(11)                     # 53 random bits
    uniform = (bits.astype(np.float64) + 0.5) / float(1 << 53)       # in (0, 1)
    u1, u2 = uniform[:, :epi_index_dim], uniform[:, epi_index_dim:]
    normal = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    vectors = torch.from_numpy(normal).float().to(device)
    return torch.nn.functional.normalize(vectors, dim=-1)


def epinet_term_loss(epinet, head_name, estimated, features, targets, epinet_indexes, c_vectors,
                     sigma, alpha_mlp, loss, alpha_ensemble=0.0, ensemble_prior_flat=None):
    """The epinet term for one head.

    estimated (n, 1) base predictions, features (n, F) base features (detached here), targets (n,),
    epinet_indexes (K, D), c_vectors (n, D), ensemble_prior_flat (K*n, 1) or None. `loss` is any
    loss_fn(prediction (K*n,), target (K*n,)) -> scalar. Rows are grouped by index (all n points
    for z_1, then for z_2, ...). Returns (loss, repeated unperturbed targets, epinet predictions).
    """
    features = features.detach()
    n_epi_indexes = epinet_indexes.shape[0]
    mlp_prior = epinet.compute_mlp_prior_batched(features, epinet_indexes)[head_name]
    learnable_mlp_prior = epinet.compute_learnable_mlp_batched(features, epinet_indexes)[head_name]
    correction = learnable_mlp_prior + alpha_mlp * mlp_prior
    if ensemble_prior_flat is not None:
        correction = correction + alpha_ensemble * ensemble_prior_flat
    epinet_estimated = estimated.repeat(n_epi_indexes, 1).detach() + correction
    anchor_term_flat = torch.matmul(epinet_indexes, c_vectors.T).view(-1)
    targets_exp = targets.repeat(n_epi_indexes)
    perturbed_targets = targets_exp + sigma * anchor_term_flat
    return loss(epinet_estimated.squeeze(-1), perturbed_targets), targets_exp, epinet_estimated
