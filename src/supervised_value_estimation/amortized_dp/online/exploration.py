"""Which plans to execute for a training query: Thompson sampling over the epinet.

Each plan is the greedy plan under ONE sampled value function (one epistemic index z):
where the model is certain, the samples agree and the plans coincide; where it is uncertain,
they differ, so the execution budget goes where it teaches the most (Approximate Thompson
Sampling via Epistemic Neural Networks, Osband et al. 2023).

All samples are decoded together: `oversample * n_plans` indices, one greedy path per index,
and at every depth ONE evaluation of the union of all paths' candidates (one agent holding all
indices, its sampled scores read per index). Each path is exactly the plan beam_search with
beam width 1 finds under that single index. The first n_plans distinct plans are kept, so a
query whose samples all agree simply gets fewer plans.

The alternative, partial_observation.deviation_plans (greedy plan plus one-step deviations at
the closest decisions), stays available as online.plan_selection = deviation.
"""
from __future__ import annotations

import numpy as np
from torch_geometric.data import Batch

from src.supervised_value_estimation.amortized_dp.agents import EpinetAmortizedDPAgent


def _candidates(prefix, adjacency, n_tp):
    """Next prefixes as beam_search generates them: connected pairs (i < j) first, then any
    pattern sharing a variable with the prefix (no cartesian products)."""
    if prefix is None:
        return [(i, j) for i in range(n_tp) for j in range(i + 1, n_tp) if adjacency[i, j]]
    reach = adjacency[list(prefix)].any(axis=0)
    return [prefix + (a,) for a in range(n_tp) if reach[a] and a not in prefix]


def thompson_plans(value_net, data, n_plans, embeddings, device, generator, objective="cout", shared=None,
                   oversample=2):
    """Up to n_plans distinct greedy plans, each under its own sampled index."""
    batch = Batch.from_data_list([data])
    indexes = value_net.epinet_state.sample_epistemic_indexes_batched(oversample * n_plans, generator=generator)
    agent = EpinetAmortizedDPAgent(value_net, embed_fn=None, epistemic_indexes=indexes, device=device,
                                   embedding_cache=embeddings, objective=objective, shared=shared)
    episode = agent.setup_episode(batch)
    n_tp = episode["n_tp"]
    adjacency = episode["adjacency"].cpu().numpy()
    paths = [None] * len(indexes)
    for _ in range(n_tp - 1):
        options = [_candidates(path, adjacency, n_tp) for path in paths]
        union = list(dict.fromkeys(candidate for candidates in options for candidate in candidates))
        agent.prepare(union, episode)
        scores = {candidate: agent.sampled_scores(list(candidate), episode) for candidate in union}
        # min over a path's candidates under its own index; ties go to the first, as in beam_search
        paths = [candidates[int(np.argmin([scores[c][k] for c in candidates]))] for k, candidates in enumerate(options)]
    plans = []
    for path in paths:
        if list(path) not in plans:
            plans.append(list(path))
        if len(plans) == n_plans:
            break
    return plans
