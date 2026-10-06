"""Risk measures over epinet samples."""
from __future__ import annotations

import torch


def cvar_upper_tail(samples, alpha):
    """CVaR_alpha of costs: per row, the mean of the worst (highest) (1 - alpha) share of the
    samples in that row. samples: (n_candidates, n_samples) -> (n_candidates,).
    At least one sample is kept, so alpha close to 1 is the worst case."""
    tail_length = max(1, int((1 - alpha) * samples.shape[1]))
    sorted_samples, _ = torch.sort(samples, dim=1)
    return sorted_samples[:, -tail_length:].mean(dim=1)
