"""Offline training of the bushy (forest-state) value net, and the greedy bottom-up optimizer.

    python -m src.supervised_value_estimation.amortized_dp.bushy_train [hydra overrides]

Data (an option next to the left-deep train_amortized_dp.py, which is unchanged):
  labels       the oracle's log cardinality of every connected subset of each training query
               (labels.label_query, the DP's own sub-query slicing), cached;
  embeddings   per triple pattern, from the best backbone (its TripleGINE layers loaded into the
               policy-network config, whose triple_embedding head pools a pattern's two nodes);
  states       forests (bushy.py) along the oracle-optimal bushy plan, along random merge
               rollouts, and one random merge off the optimal path; every candidate merge of a
               state is one training row: the child forest with card(merged component) and the
               exact constrained cost-to-go G(child) (bushy.forest_cost_to_go) as targets.
Losses (standardised log space), as in the left-deep trainer:
  card  MSE on the merged component's log card;   G  MSE on log G of non-final children;
  rank  listwise distillation over a state's children of softmax(-log Q / T),
        log Q(child) = log(card(child component) + G(child)): exactly what the greedy agent ranks.
Greedy optimizer: from the singleton forest, repeatedly take the candidate merge with the smallest
predicted Q until one component is left -> a bushy join tree.
Writes output_directory/{labels_*.pkl, embeddings_*.pt, states_*.pt, value_net.pt, metrics.json}.
"""
from __future__ import annotations

import gc
import json
import math
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.datastructures.query_cardinality_dataset import QueryCardinalityDataset
from src.models.model_instantiator import ModelFactory
from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.bushy_model import ForestValueNet, forest_inputs
from src.supervised_value_estimation.amortized_dp.labels import (
    JOIN_FEATURE_DIM, NEG_INF, QueryLabels, family_of, join_position_features, label_query, mask_of, members,
    model_log_cardinalities, variable_adjacency,
)

_STATE = {}


# --- models -----------------------------------------------------------------------------------

def load_backbone(config, run, select, device):
    """A pretrained cardinality backbone at the epoch with the best `select` value."""
    progress = json.load(open(Path(run) / "train_progress.json"))[select]
    epoch = min(range(len(progress)), key=progress.__getitem__)
    model = ModelFactory(config).load_gine_conv()
    model.embedding_model.load_state_dict(torch.load(Path(run) / f"epoch-{epoch}/model/embedding_model.pt", map_location="cpu"))
    model.heads[0].load_state_dict(torch.load(Path(run) / f"epoch-{epoch}/model/head_cardinality.pt", map_location="cpu"))
    return model.to(device).eval(), epoch


def load_embedder(policy_config, run, select, device):
    """The backbone's TripleGINE layers in the policy-network config (same layer ids), whose
    triple_embedding head gives one vector per triple pattern."""
    progress = json.load(open(Path(run) / "train_progress.json"))[select]
    epoch = min(range(len(progress)), key=progress.__getitem__)
    model = ModelFactory(policy_config).load_gine_conv()
    state = torch.load(Path(run) / f"epoch-{epoch}/model/embedding_model.pt", map_location="cpu")
    missing, unexpected = model.embedding_model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"embedder layers missing from the backbone checkpoint: {missing}")
    return model.to(device).eval(), epoch


@torch.no_grad()
def embed_patterns(embedder, query, device):
    batch = torch.zeros(query.x.shape[0], dtype=torch.long, device=device)
    outputs = embedder.forward(x=query.x.to(device), edge_index=query.edge_index.to(device),
                               edge_attr=query.edge_attr.to(device), batch=batch)
    embedded, _ = next(head["output"] for head in outputs if head["output_type"] == "triple_embedding")
    return embedded.float().cpu()


# --- queries, labels, embeddings ----------------------------------------------------------------

def load_split(directory):
    return QueryCardinalityDataset(root=str(directory), featurizer=None, load_mappings=True)


def count_connected_subsets(neighbour_masks, cap):
    """Number of connected pattern subsets, or cap + 1 as soon as there are more than cap
    (EnumerateCsg with an early stop: a 16-pattern star has 65,535)."""
    n, count, stack = len(neighbour_masks), 0, []
    for v in range(n - 1, -1, -1):
        stack.append((1 << v, (1 << (v + 1)) - 1))
        count += 1
        while stack:
            s, x = stack.pop()
            reach = 0
            for i in range(n):
                if s >> i & 1:
                    reach |= neighbour_masks[i]
            frontier = reach & ~s & ~x
            sub = 0
            while True:
                sub = (sub - frontier) & frontier
                if not sub:
                    break
                count += 1
                if count > cap:
                    return cap + 1
                stack.append((s | sub, x | frontier))
    return count


def _neighbour_masks(triple_patterns):
    variables = [{t for t in tp.split()[:3:2] if t.startswith("?")} for tp in triple_patterns]
    return [sum(1 << j for j in range(len(variables)) if j != i and variables[i] & variables[j]) for i in range(len(variables))]


def select_queries(dataset, max_subsets, n, seed):
    """A random sample of n queries (None: all) among those with at most max_subsets connected
    subsets (all their subsets get labels); candidates are checked in random order until n are found."""
    order = np.random.default_rng(seed).permutation(len(dataset))
    picked = []
    for i in order:
        if count_connected_subsets(_neighbour_masks(dataset[int(i)].triple_patterns), max_subsets) <= max_subsets:
            picked.append(int(i))
            if n is not None and len(picked) >= n:
                break
    return sorted(picked)


def build_labels(dataset, indices, oracle, device, path):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            labels = pickle.load(f)
        if all(" ".join(dataset[i].query.split()) in labels for i in indices):
            return labels
    labels = {}
    for i in tqdm(indices, desc=f"labels {path.stem}", mininterval=60):
        query = dataset[i]
        labels[" ".join(query.query.split())] = label_query(query, oracle, device)
    with open(path.with_suffix(".tmp"), "wb") as f:
        pickle.dump(labels, f)
    os.replace(path.with_suffix(".tmp"), path)
    return labels


# --- states and exact targets --------------------------------------------------------------------

def sample_forests(tree, n, neighbour_masks, rng, n_rollouts, max_states):
    """Non-final forests: along `tree`'s path, one random merge off each of its states, and random
    rollouts. Returns (path, deviations, rollouts) as forest lists (each deduplicated, path first,
    at most max_states in all: the path is always kept, the rest sampled) and the rollouts' trees."""
    path = [bushy.forest_after(tree, n, k) for k in range(n - 1)]
    deviations, rollouts, rollout_trees = [], [], []
    for forest in path:
        merges = bushy.candidate_merges(forest, neighbour_masks)
        if len(merges) > 1:
            deviations.append(bushy.merge(forest, *merges[int(rng.integers(len(merges)))]))
    for _ in range(n_rollouts):
        forest, steps = bushy.initial_forest(n), []
        while len(forest) > 1:
            rollouts.append(forest)
            merges = bushy.candidate_merges(forest, neighbour_masks)
            steps.append(merges[int(rng.integers(len(merges)))])
            forest = bushy.merge(forest, *steps[-1])
        rollout_trees.append(bushy.merges_to_tree(n, steps))
    unique = list(dict.fromkeys(f for f in path + deviations + rollouts if len(f) > 1))
    if len(unique) > max_states:                     # keep the path, sample the rest
        keep = set(path)
        rest = [f for f in unique if f not in keep]
        picked = rng.choice(len(rest), size=max(0, max_states - len(keep)), replace=False) if rest else []
        unique = [f for f in unique if f in keep] + [rest[i] for i in picked]
    kept, seen = set(unique), set()
    groups = []
    for group in (path, deviations, rollouts):
        groups.append([f for f in dict.fromkeys(group) if f in kept and f not in seen and len(f) > 1])
        seen.update(groups[-1])
    return groups, rollout_trees


def sample_states(labels, rng, n_rollouts, max_states):
    """Non-final forests: along the oracle-optimal bushy plan, random rollouts, and one random
    merge off the optimal path."""
    tree, cost = bushy.optimal_bushy_plan(labels)
    if tree is None or not math.isfinite(cost):
        return []
    (path, deviations, rollouts), _ = sample_forests(tree, labels.n_tp, labels.neighbour_masks, rng, n_rollouts, max_states)
    return path + deviations + rollouts


def _exact_rows(labels, forests, state_offset=0):
    """Every candidate merge of the given forests as a row (state, child forest, merged component,
    log card, log G, full, bound) with exact targets (bound False: G is the constrained DP)."""
    rows = []
    for k, forest in enumerate(forests):
        for a, b in bushy.candidate_merges(forest, labels.neighbour_masks):
            child = bushy.merge(forest, a, b)
            full = len(child) == 1
            g = NEG_INF if full else bushy.forest_cost_to_go(labels, child)
            if not full and not math.isfinite(g):
                continue
            rows.append((state_offset + k, child, a | b, labels.logcard[a | b], g, full, False))
    return rows


def _state_rows(job):
    query_id, seed = job
    labels = _STATE["labels"][query_id]
    rng = np.random.default_rng([seed, query_id])
    forests = sample_states(labels, rng, _STATE["n_rollouts"], _STATE["max_states"])
    return [(query_id, *row[:6]) for row in _exact_rows(labels, forests)]


def _forest_rows(job):
    """Rows (every candidate merge with exact targets) for given forests of one query."""
    query_id, state_offset, forests = job
    return [(query_id, *row[:6]) for row in _exact_rows(_STATE["labels"][query_id], forests, state_offset)]


def dagger_rows(model, tables, labels_list, query_ids, round_index, device, workers):
    """DAgger: the states the current greedy policy visits on these training queries, labelled
    with the exact constrained-DP targets (state ids offset per round, so they stay distinct)."""
    model.eval()
    jobs = []
    for q in query_ids:
        n = labels_list[q].n_tp
        _, visited = greedy_plan(model, tables.embeddings[q, :n], tables.adjacency[q, :n, :n], tables.edge_features[q, :n, :n],
                                 labels_list[q].neighbour_masks, device, return_states=True)
        jobs.append((q, 100_000 * (round_index + 1), visited))
    model.train()
    _STATE.update(labels=labels_list)
    rows = []
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for part in pool.imap_unordered(_forest_rows, jobs, chunksize=16):
            rows.extend(part)
    return rows


def build_states(labels_list, n_rollouts, max_states, seed, workers):
    _STATE.update(labels=labels_list, n_rollouts=n_rollouts, max_states=max_states)
    jobs = [(q, seed) for q in range(len(labels_list))]
    rows = []
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for part in tqdm(pool.imap_unordered(_state_rows, jobs, chunksize=16), total=len(jobs),
                         desc="states", mininterval=60):
            rows.extend(part)
    return rows


# --- hybrid targets: every training query ------------------------------------------------------

def _connected(neighbour_masks):
    reach, frontier = 1, 1
    while frontier:
        grown = reach
        for i in members(frontier):
            grown |= neighbour_masks[i]
        frontier, reach = grown & ~reach, grown
    return reach == (1 << len(neighbour_masks)) - 1


def noisy_greedy_plans(n, neighbour_masks, logcard, score, fits, rng, count, temperature):
    """`count` bottom-up plans that each merge with probability proportional to
    exp(-log card(merge) / temperature) (temperature -> 0: GOO), in lockstep: one `score` call per
    step for the candidate merges of all of them. Before each step, plans are dropped (those with the
    most new subsets first) until the step fits: fits(masks) reserves them in the budget or refuses."""
    plans = [(bushy.initial_forest(n), []) for _ in range(count)]
    for _ in range(n - 1):
        candidates = [bushy.candidate_merges(forest, neighbour_masks) for forest, _ in plans]
        needs = [{a | b for a, b in merges if a | b not in logcard} for merges in candidates]
        keep = sorted(range(len(plans)), key=lambda k: len(needs[k]))
        while keep and not fits(set().union(*(needs[k] for k in keep))):
            keep.pop()
        if not keep:
            return []
        keep.sort()
        plans, candidates = [plans[k] for k in keep], [candidates[k] for k in keep]
        score(list(set().union(*(needs[k] for k in keep))))
        next_plans = []
        for (forest, steps), merges in zip(plans, candidates):
            logits = -np.array([logcard[a | b] for a, b in merges]) / temperature
            weights = np.exp(logits - logits.max())
            a, b = merges[int(rng.choice(len(merges), p=weights / weights.sum()))]
            next_plans.append((bushy.merge(forest, a, b), steps + [(a, b)]))
        plans = next_plans
    return [bushy.merges_to_tree(n, steps) for _, steps in plans]


def sparse_rows(query, oracle, device, rng, width, n_rollouts, max_states, budget, greedy_plans=0, greedy_temperature=1.0):
    """Online-style targets for a query with too many connected subsets to label them all: the
    oracle scores at most `budget` subsets, chosen as an online learner would observe them.
      plans   the oracle's own beam search (bushy.estimator_beam_plan, `width`): its final trees,
              best T first;
      states  T's path, one random merge off each of its states, random rollouts (sample_forests);
              a state is taken, in that priority, only while everything it needs fits the budget:
              its components, the merged component of each candidate merge (card targets), and per
              non-final child the merges of its repair completion (bushy.repair_completion along T);
      more    the rest of the budget goes to up to `greedy_plans` noisy greedy plans under the oracle
              (noisy_greedy_plans): more complete plans to stitch bounds from;
      G       the stitched upper bound (bushy.StitchTable) over ALL scored subsets: at most the cost
              of the child's own repair completion, so finite for every child; the more complete plans
              (beam, rollouts, greedy) were scored, the tighter.
    Returns the partial labels (logcard of the scored subsets) and rows as in _exact_rows, with
    bound True (the target G is an upper bound)."""
    n = len(query.triple_patterns)
    adjacency = variable_adjacency(query.triple_patterns)
    labels = QueryLabels(query=query.query if isinstance(query.query, str) else query.query[0],
                         family=family_of(query.type if isinstance(query.type, str) else query.type[0]),
                         n_tp=n, neighbour_masks=[mask_of(adjacency[i]) for i in range(n)], logcard={})
    if not _connected(labels.neighbour_masks):
        return labels, []
    logcard, neighbour_masks = labels.logcard, labels.neighbour_masks
    pending = set()                                  # reserved in the budget, not scored yet

    def score(masks):
        missing = list(dict.fromkeys(m for m in masks if m not in logcard))
        if missing:
            logcard.update(zip(missing, model_log_cardinalities(oracle, query, missing, device)))
            pending.difference_update(missing)
        return [logcard[m] for m in masks]

    def fits(needed):
        new = {m for m in needed if m not in logcard and m not in pending}
        if len(logcard) + len(pending) + len(new) > budget:
            return False
        pending.update(new)
        return True

    trees, _ = bushy.estimator_beam_plan(n, neighbour_masks, score, width, all_trees=True)
    tree = trees[0]
    (path, deviations, rollouts), rollout_trees = sample_forests(tree, n, neighbour_masks, rng, n_rollouts, max_states)
    repairs, taken = {}, []

    def take(forest):
        needed = [c for c in forest if c & (c - 1)]
        for a, b in bushy.candidate_merges(forest, neighbour_masks):
            needed.append(a | b)
            child = bushy.merge(forest, a, b)
            if len(child) > 1 and child not in repairs:
                repairs[child] = [x | y for x, y in bushy.repair_completion(tree, child)]
            needed += repairs.get(child, [])
        if fits(needed):
            taken.append(forest)
    for forest in path + deviations:
        take(forest)
    if rollouts and fits([m for t in rollout_trees for m in bushy.internal_masks(t)]):
        for forest in rollouts:
            take(forest)
    score(list(pending))
    if greedy_plans:
        noisy_greedy_plans(n, neighbour_masks, logcard, score, fits, rng, greedy_plans, greedy_temperature)
    table = bushy.StitchTable(logcard, logcard)
    rows = []
    for state, forest in enumerate(taken):
        for a, b in bushy.candidate_merges(forest, neighbour_masks):
            child = bushy.merge(forest, a, b)
            full = len(child) == 1
            g = NEG_INF if full else table.cost_to_go(child)
            if not full and not math.isfinite(g):
                continue
            rows.append((state, child, a | b, logcard[a | b], g, full, True))
    return labels, rows


def row_arrays(rows, n_max):
    """Rows (state, child forest, merged component, log card, log G, full, bound) as compact arrays:
    per row each pattern's component index in the sorted child forest and the newest one's index."""
    component = np.zeros((len(rows), n_max), dtype=np.int8)
    newest = np.zeros(len(rows), dtype=np.int64)
    for k, (_, child, merged, *_) in enumerate(rows):
        ordered = sorted(child)
        for index, mask in enumerate(ordered):
            for p in members(mask):
                component[k, p] = index
        newest[k] = ordered.index(merged)
    return {"state": np.array([r[0] for r in rows], dtype=np.int64), "component": component, "newest": newest,
            "card": np.array([r[3] for r in rows], dtype=np.float32),
            "g": np.array([0.0 if r[5] else r[4] for r in rows], dtype=np.float32),
            "full": np.array([r[5] for r in rows], dtype=bool), "bound": np.array([r[6] for r in rows], dtype=bool)}


def _hybrid_job(job):
    """One training query in a labelling worker (forked; its own oracle on the GPU)."""
    position, query, exact = job
    s = _STATE
    if "oracle" not in s:
        torch.set_num_threads(1)                     # many workers: no intra-op thread oversubscription
        s["oracle"], _ = load_backbone(s["oracle_config"].config, s["oracle_config"].run, s["oracle_config"].select, s["device"])
    rng = np.random.default_rng([s["seed"], position])
    if exact:
        labels = label_query(query, s["oracle"], s["device"])
        rows = _exact_rows(labels, sample_states(labels, rng, s["n_rollouts"], s["max_states"]))
    else:
        labels, rows = sparse_rows(query, s["oracle"], s["device"], rng, s["width"], s["sparse_rollouts"],
                                   s["sparse_max_states"], s["budget"], s["greedy_plans"], s["greedy_temperature"])
    return position, labels, row_arrays(rows, s["n_max"])


def _subset_count(job):
    neighbour_masks, cap = job
    return count_connected_subsets(neighbour_masks, cap)


def prepare_hybrid(cfg, out):
    """Every training query (or n_train of them) with hybrid targets: queries with <= max_subsets
    connected subsets get all subsets labelled and exact DP targets (as prepare + build_states),
    the others sparse_rows. Labelled in shards by forked workers, each with its own oracle on the
    GPU, so this must run before the parent process touches CUDA. The workers are forked before the
    dataset is loaded and get each query with its job (forked after, each would end up with a private
    copy of the dataset: copy-on-write pages touched by reference counting). Cached under train_hybrid/:
    selection.pkl, rows_<shard>.npz, labels_<shard>.pkl (if data.save_labels), embeddings.pt.
    Returns keys, labels (n_tp and neighbour masks only), embeddings, patterns and row arrays."""
    directory = out / "train_hybrid"
    directory.mkdir(parents=True, exist_ok=True)
    selection_path = directory / "selection.pkl"
    workers = int(cfg.data.get("label_workers", cfg.data.workers))
    if selection_path.exists():
        with open(selection_path, "rb") as f:
            selection = pickle.load(f)
    else:
        dataset = load_split(cfg.data.train_dir)
        neighbour_masks = []
        for i in range(len(dataset)):
            adjacency = variable_adjacency(dataset[i].triple_patterns)
            neighbour_masks.append([mask_of(adjacency[j]) for j in range(len(adjacency))])
        cap = int(cfg.data.max_subsets)
        with multiprocessing.get_context("fork").Pool(int(cfg.data.workers)) as pool:
            counts = pool.map(_subset_count, [(m, cap) for m in neighbour_masks], chunksize=256)
        indices = list(range(len(dataset)))
        if cfg.data.n_train is not None:
            indices = sorted(np.random.default_rng(int(cfg.training.seed)).permutation(len(dataset))[:int(cfg.data.n_train)].tolist())
        selection = {"indices": indices, "exact": [counts[i] <= cap for i in indices],
                     "neighbour_masks": [neighbour_masks[i] for i in indices]}
        with open(selection_path, "wb") as f:
            pickle.dump(selection, f)
        del dataset
        gc.collect()
    indices, exact = selection["indices"], selection["exact"]
    n_max = max(len(m) for m in selection["neighbour_masks"])
    print(f"[bushy] hybrid: {len(indices):,} training queries, {sum(exact):,} exact (<= {int(cfg.data.max_subsets):,} "
          f"connected subsets), {len(indices) - sum(exact):,} sparse", flush=True)
    _STATE.update(oracle_config=cfg.oracle, device=cfg.training.device, seed=int(cfg.training.seed),
                  n_rollouts=int(cfg.data.rollouts), max_states=int(cfg.data.max_states), n_max=n_max,
                  width=int(cfg.data.sparse_beam_width), sparse_rollouts=int(cfg.data.sparse_rollouts),
                  sparse_max_states=int(cfg.data.sparse_max_states), budget=int(cfg.data.sparse_budget),
                  greedy_plans=int(cfg.data.get("sparse_greedy_plans", 0)),
                  greedy_temperature=float(cfg.data.get("sparse_greedy_temperature", 1.0)))
    shard_size = int(cfg.data.shard_size)
    shards = [range(start, min(start + shard_size, len(indices))) for start in range(0, len(indices), shard_size)]
    todo = [k for k in range(len(shards)) if not (directory / f"rows_{k:03d}.npz").exists()]
    if todo:
        # Workers are recycled (their host memory creeps up); later forks happen after the dataset
        # is loaded, so it is frozen out of the garbage collector, whose traversals would copy it.
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=int(cfg.data.get("tasks_per_worker", 1000))) as pool:
            dataset = load_split(cfg.data.train_dir)
            gc.freeze()
            for k in todo:
                start = time.perf_counter()
                jobs = [(position, dataset[indices[position]], exact[position]) for position in shards[k]]
                arrays, labels = [], {}
                for position, query_labels, part in tqdm(pool.imap_unordered(_hybrid_job, jobs, chunksize=4), total=len(jobs),
                                                         desc=f"hybrid shard {k + 1}/{len(shards)}", mininterval=60):
                    part["query"] = np.full(len(part["state"]), position, dtype=np.int64)
                    arrays.append(part)
                    labels[" ".join(query_labels.query.split())] = query_labels
                merged = {name: np.concatenate([a[name] for a in arrays]) for name in arrays[0]}
                np.savez(directory / f"rows_{k:03d}.tmp.npz", **merged)
                os.replace(directory / f"rows_{k:03d}.tmp.npz", directory / f"rows_{k:03d}.npz")
                if cfg.data.get("save_labels", True):
                    with open(directory / f"labels_{k:03d}.pkl", "wb") as f:
                        pickle.dump(labels, f)
                sparse = ~merged["bound"]
                print(f"[bushy] hybrid shard {k + 1}/{len(shards)}: {len(merged['state']):,} rows "
                      f"({int((~sparse).sum()):,} with bounds), {sum(len(l.logcard) for l in labels.values()):,} "
                      f"oracle subsets ({time.perf_counter() - start:.0f}s)", flush=True)
    else:
        dataset = load_split(cfg.data.train_dir)
    parts = [dict(np.load(directory / f"rows_{k:03d}.npz")) for k in range(len(shards))]
    arrays = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    order = np.lexsort((arrays["state"], arrays["query"]))
    arrays = {name: value[order] for name, value in arrays.items()}
    _, arrays["state"] = np.unique(arrays["query"] * 100_000 + arrays["state"], return_inverse=True)
    keys = [" ".join(dataset[i].query.split()) for i in indices]
    patterns = [list(dataset[i].triple_patterns) for i in indices]
    labels_list = [QueryLabels(query=key, family="", n_tp=len(m), neighbour_masks=m, logcard={})
                   for key, m in zip(keys, selection["neighbour_masks"])]
    path = directory / "embeddings.pt"
    if path.exists():
        embeddings = torch.load(path)
    else:
        embedder, _ = load_embedder(cfg.embedder.policy_config, cfg.embedder.run, cfg.embedder.select, cfg.training.device)
        embeddings = [embed_patterns(embedder, dataset[i], cfg.training.device).half()
                      for i in tqdm(indices, desc="embed train", mininterval=60)]
        torch.save(embeddings, path)
        del embedder
    _STATE.clear()
    return keys, labels_list, embeddings, patterns, arrays


def rows_to_arrays(rows, n_max):
    """Row tuples (query, state, child, merged, log card, log G, full) -> arrays sorted by state, with
    global state ids (one per distinct (query, state))."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    arrays = row_arrays([(r[1], r[2], r[3], r[4], r[5], r[6], False) for r in rows], n_max)
    arrays["query"] = np.array([r[0] for r in rows], dtype=np.int64)
    if rows:
        _, arrays["state"] = np.unique(np.stack([arrays["query"], arrays["state"]], axis=1), axis=0, return_inverse=True)
    arrays["state"] = arrays["state"].reshape(-1)
    return arrays


class ForestTables:
    """GPU tables: per query the pattern embeddings (float16), adjacency and join-position features
    (uint8 counts); per row (state, candidate merge) the child forest's component index per pattern
    (int8), the merged component, the targets (bound: G is an upper bound), and the state it belongs
    to. `rows` are row tuples (query, state, ...) or the arrays of rows_to_arrays / prepare_hybrid."""

    def __init__(self, labels_list, embeddings_list, patterns_list, rows, device):
        n_max = max(labels.n_tp for labels in labels_list)
        dim = embeddings_list[0].shape[1]
        q = len(labels_list)
        self.n_max, self.device = n_max, device
        self.embeddings = torch.zeros(q, n_max, dim, dtype=torch.float16)
        self.pattern_mask = torch.zeros(q, n_max, dtype=torch.bool)
        self.adjacency = torch.zeros(q, n_max, n_max, dtype=torch.bool)
        self.edge_features = torch.zeros(q, n_max, n_max, JOIN_FEATURE_DIM, dtype=torch.uint8)
        for i, (labels, embeddings, patterns) in enumerate(zip(labels_list, embeddings_list, patterns_list)):
            n = labels.n_tp
            self.embeddings[i, :n] = embeddings
            self.pattern_mask[i, :n] = True
            self.adjacency[i, :n, :n] = torch.from_numpy((np.array(labels.neighbour_masks)[:, None] >> np.arange(n)) & 1 > 0)
            self.edge_features[i, :n, :n] = torch.from_numpy(join_position_features(patterns).astype(np.uint8))
        arrays = rows if isinstance(rows, dict) else rows_to_arrays(rows, n_max)
        self.rows = {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}
        order = torch.argsort(self.rows["state"], stable=True)
        self.rows = {k: v[order] for k, v in self.rows.items()}
        self.n_states = int(self.rows["state"].max()) + 1 if len(self.rows["state"]) else 0
        self.state_offsets = torch.zeros(self.n_states + 1, dtype=torch.long)
        self.state_offsets[1:] = torch.cumsum(torch.bincount(self.rows["state"], minlength=self.n_states), 0)
        for name in ("embeddings", "pattern_mask", "adjacency", "edge_features", "state_offsets"):
            setattr(self, name, getattr(self, name).to(device))
        self.rows = {k: v.to(device) for k, v in self.rows.items()}

    def statistics(self):
        g = self.rows["g"][~self.rows["full"]]
        return float(self.rows["card"].mean()), float(self.rows["card"].std()), float(g.mean()), float(g.std())

    def batch_rows(self, state_ids):
        starts, ends = self.state_offsets[state_ids], self.state_offsets[state_ids + 1]
        counts = ends - starts
        local = torch.repeat_interleave(torch.arange(len(state_ids), device=self.device), counts)
        within = torch.arange(int(counts.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        return torch.repeat_interleave(starts, counts) + within, local


def _group_log_softmax(logits, groups, n_groups):
    peak = torch.full((n_groups,), -math.inf, device=logits.device).scatter_reduce(0, groups, logits, reduce="amax", include_self=True)
    shifted = torch.exp(logits - peak[groups])
    normaliser = torch.zeros(n_groups, device=logits.device).index_add(0, groups, shifted)
    return logits - peak[groups] - torch.log(normaliser[groups])


def forward_rows(model, tables, r):
    q = r["query"]
    return model(tables.embeddings[q].float(), tables.pattern_mask[q], tables.adjacency[q], r["component"].long(), r["newest"],
                 tables.edge_features[q].float() if model.edge_feature_dim else None)


def training_loss(model, tables, state_ids, rank_weight, temperature, under_weight=0.1, balance_weight=0.0):
    rows, local = tables.batch_rows(state_ids)
    r = {k: v[rows] for k, v in tables.rows.items()}
    card_std, g_std = forward_rows(model, tables, r)
    card_target, g_target = model.standardise(r["card"], r["g"])
    loss_card = torch.mean((card_std - card_target) ** 2)
    not_full = ~r["full"]
    g_error = g_std[not_full] - g_target[not_full]
    # A bound target is the cheapest KNOWN completion, so the truth can only be lower: predicting
    # above it is always wrong, below it only weakly penalised (as in train_amortized_dp).
    g_error = g_error * torch.where(r["bound"][not_full] & (g_error < 0), under_weight ** 0.5, 1.0)
    loss_g = torch.mean(g_error ** 2) if not_full.any() else card_std.sum() * 0
    card, g = model.unstandardise(card_std, g_std)
    g_term = torch.where(r["full"], torch.full_like(g, -math.inf), g)
    logq = torch.logaddexp(card, g_term)
    true_logq = torch.logaddexp(r["card"], torch.where(r["full"], torch.full_like(r["g"], -math.inf), r["g"]))
    n = len(state_ids)
    predicted = _group_log_softmax(-logq / temperature, local, n)
    target = torch.exp(_group_log_softmax(-true_logq / temperature, local, n))
    loss_rank = -torch.zeros(n, device=card.device).index_add(0, local, target * predicted).mean()
    total = loss_card + loss_g + rank_weight * loss_rank
    parts = {"card": loss_card.item(), "g": loss_g.item(), "rank": loss_rank.item()}
    if model.moe_config:
        balance = model.load_balancing_loss()
        total = total + balance_weight * balance
        parts["balance"] = balance.item()
    return total, parts


OBJECTIVES = ("cout", "latency")


def _merge_scorer(model, embeddings, adjacency, edge_features, n, device, objective="cout"):
    """scores(children, merges) -> per candidate merge (a, b) the predicted (log cost of the merge
    itself, log cost-to-go of the child forest), with log cost-to-go = -inf for a complete forest:
        cout     (log card(a | b), log G(child))                       -- C_out
        latency  (log step ms of the merge, log latency-to-go(child) ms) -- the online latency heads
    A plan's cost is then the logsumexp of its merges' first terms, as C_out sums cards."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "latency" and not (model.has_latency_head and model.has_step_head):
        raise ValueError("The latency objective needs the latency-to-go and step heads.")
    emb = embeddings.unsqueeze(0).to(device).float()
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    adj = adjacency.unsqueeze(0).to(device)
    edges = edge_features.unsqueeze(0).to(device).float() if model.edge_feature_dim else None

    def scores(children, merges):
        component, newest_index, side = forest_inputs(children, n, merges=merges, device=device)
        b = len(children)
        args = (emb.expand(b, -1, -1), mask.expand(b, -1), adj.expand(b, -1, -1), component, newest_index,
                edges.expand(b, -1, -1, -1) if edges is not None else None)
        full = torch.tensor([len(c) == 1 for c in children], device=device)
        if objective == "cout":
            card_std, g_std = model(*args)
            now, to_go = model.unstandardise(card_std, g_std)
        else:
            out = model.forward_heads(*args, merge_side=side, heads=("card", "cost_to_go", "latency", "step"))
            # log1p ms -> log ms (a 1 us floor: a 0 ms step must not dominate the logsumexp)
            now = torch.log(torch.expm1(model.unstandardise_step(out["step"])).clamp(min=1e-3))
            to_go = torch.log(torch.expm1(model.unstandardise_latency(out["latency"])).clamp(min=1e-3))
        return now, torch.where(full, torch.full_like(to_go, -math.inf), to_go)
    return scores


def _ranked_merges(scores, forest, neighbour_masks):
    """Candidate merges of a forest, children and log Q (merge + cost-to-go), best first."""
    merges = bushy.candidate_merges(forest, neighbour_masks)
    children = [bushy.merge(forest, a, b) for a, b in merges]
    now, to_go = scores(children, merges)
    q = torch.logaddexp(now, to_go).tolist()
    order = sorted(range(len(merges)), key=q.__getitem__)
    return [merges[k] for k in order], [children[k] for k in order], [q[k] for k in order]


@torch.no_grad()
def greedy_plan(model, embeddings, adjacency, edge_features, neighbour_masks, device, return_states=False,
                objective="cout"):
    """Bottom-up greedy: merge the pair with the smallest predicted log Q (merge + cost-to-go of
    the child, under `objective`) until one component. With return_states also the (non-final)
    forests it visited."""
    n = len(neighbour_masks)
    scores = _merge_scorer(model, embeddings, adjacency, edge_features, n, device, objective)
    forest = bushy.initial_forest(n)
    visited = []
    trees = {1 << i: i for i in range(n)}
    while len(forest) > 1:
        visited.append(forest)
        merges, children, _ = _ranked_merges(scores, forest, neighbour_masks)
        a, b2 = merges[0]
        trees[a | b2] = (trees.pop(a), trees.pop(b2))
        forest = children[0]
    tree = next(iter(trees.values()))
    return (tree, visited) if return_states else tree


@torch.no_grad()
def deviation_plans(model, embeddings, adjacency, edge_features, neighbour_masks, device, n_plans, objective="cout"):
    """The greedy plan plus up to n_plans - 1 one-step deviations from it (as
    partial_observation.deviation_plans for left-deep plans): at the greedy path's closest calls
    (smallest gap in log Q between its best and second-best merge) take the second-best merge
    and continue greedily. Returns distinct trees, greedy first."""
    n = len(neighbour_masks)
    scores = _merge_scorer(model, embeddings, adjacency, edge_features, n, device, objective)

    def greedy_from(forest, trees):
        while len(forest) > 1:
            merges, children, _ = _ranked_merges(scores, forest, neighbour_masks)
            a, b = merges[0]
            trees[a | b] = (trees.pop(a), trees.pop(b))
            forest = children[0]
        return next(iter(trees.values()))
    forest, trees, decisions = bushy.initial_forest(n), {1 << i: i for i in range(n)}, []
    while len(forest) > 1:
        merges, children, q = _ranked_merges(scores, forest, neighbour_masks)
        decisions.append((dict(trees), merges, children, q))
        a, b = merges[0]
        trees[a | b] = (trees.pop(a), trees.pop(b))
        forest = children[0]
    plans = [next(iter(trees.values()))]
    branchable = sorted((k for k, d in enumerate(decisions) if len(d[1]) > 1), key=lambda k: decisions[k][3][1] - decisions[k][3][0])
    for k in branchable:
        if len(plans) >= n_plans:
            break
        trees, merges, children, _ = decisions[k]
        trees = dict(trees)
        a, b = merges[1]
        trees[a | b] = (trees.pop(a), trees.pop(b))
        plan = greedy_from(children[1], trees)
        if bushy.canonical(plan) not in {bushy.canonical(p) for p in plans}:
            plans.append(plan)
    return plans


@torch.no_grad()
def beam_plan(model, embeddings, adjacency, edge_features, neighbour_masks, device, width, objective="cout"):
    """Bottom-up beam search: keep the `width` forests with the smallest predicted total
    log(cost so far + merge + cost-to-go of the child), under `objective` (cout: cards and G;
    latency: step latencies and latency-to-go), the cost so far summing the predictions for the
    beam's own merges. A forest reached along two paths keeps the cheaper one (its cost-to-go is
    the same). Width 1 is greedy_plan; with cout an unbounded width is the exact DP over the
    predicted cards. Returns the complete tree with the smallest predicted cost."""
    n = len(neighbour_masks)
    scores = _merge_scorer(model, embeddings, adjacency, edge_features, n, device, objective)
    beams = [(bushy.initial_forest(n), NEG_INF, {1 << i: i for i in range(n)})]   # (forest, log cost so far, trees)
    while len(beams[0][0]) > 1:
        candidates = [(index, a, b) for index, (forest, _, _) in enumerate(beams)
                      for a, b in bushy.candidate_merges(forest, neighbour_masks)]
        children = [bushy.merge(beams[index][0], a, b) for index, a, b in candidates]
        now, to_go = scores(children, [(a, b) for _, a, b in candidates])
        so_far = torch.logaddexp(torch.tensor([beams[index][1] for index, _, _ in candidates], device=now.device), now)
        total = torch.logaddexp(so_far, to_go).tolist()
        so_far = so_far.tolist()
        best = {}                                       # child forest -> candidate with the cheapest cost so far
        for k, child in enumerate(children):
            if child not in best or so_far[k] < so_far[best[child]]:
                best[child] = k
        kept = sorted(best.values(), key=lambda k: total[k])[:width]
        next_beams = []
        for k in kept:
            index, a, b = candidates[k]
            trees = dict(beams[index][2])
            trees[a | b] = (trees.pop(a), trees.pop(b))
            next_beams.append((children[k], so_far[k], trees))
        beams = next_beams
    return next(iter(min(beams, key=lambda beam: beam[1])[2].values()))


@torch.no_grad()
def greedy_quality(model, tables, labels_list, device, limit=300):
    """Model selection: log cost ratio (oracle cards) of the greedy plan to the oracle-optimal
    bushy plan, on validation queries."""
    model.eval()
    ratios = []
    for i, labels in enumerate(labels_list[:limit]):
        n = labels.n_tp
        tree = greedy_plan(model, tables.embeddings[i, :n], tables.adjacency[i, :n, :n], tables.edge_features[i, :n, :n],
                           labels.neighbour_masks, device)
        _, best = bushy.optimal_bushy_plan(labels)
        ratios.append(bushy.tree_log_cost(tree, labels.logcard) - best)
    model.train()
    ratios = np.array(ratios)
    return {"greedy_log_ratio_mean": float(ratios.mean()), "greedy_optimal_share": float(np.mean(ratios < 1e-6)),
            "greedy_within_2x": float(np.mean(ratios < math.log(2)))}


def prepare(cfg, split_dir, n, seed, oracle, embedder, device, out, name, max_subsets):
    dataset = load_split(split_dir)
    indices = select_queries(dataset, max_subsets, n, seed)
    labels = build_labels(dataset, indices, oracle, device, out / f"labels_{name}.pkl")
    keys = [" ".join(dataset[i].query.split()) for i in indices]
    path = out / f"embeddings_{name}.pt"
    if path.exists():
        embeddings = torch.load(path)
    else:
        embeddings = {k: embed_patterns(embedder, dataset[i], device) for k, i in tqdm(list(zip(keys, indices)), desc=f"embed {name}", mininterval=60)}
        torch.save(embeddings, path)
    labels_list = [labels[k] for k in keys]
    patterns = [list(dataset[i].triple_patterns) for i in indices]
    return keys, labels_list, [embeddings[k] for k in keys], patterns


def train(cfg):
    torch.manual_seed(int(cfg.training.seed))
    out = Path(cfg.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    hybrid = bool(cfg.data.get("hybrid", False))
    if hybrid and int(cfg.training.get("dagger_rounds", 0)):
        raise ValueError("DAgger rounds need the exact labels of every training query; not with data.hybrid")
    train_data = prepare_hybrid(cfg, out) if hybrid else None    # forks CUDA workers: before CUDA is used here
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    oracle, oracle_epoch = load_backbone(cfg.oracle.config, cfg.oracle.run, cfg.oracle.select, device)
    embedder, embedder_epoch = load_embedder(cfg.embedder.policy_config, cfg.embedder.run, cfg.embedder.select, device)
    print(f"[bushy] oracle epoch {oracle_epoch}, embedder epoch {embedder_epoch}", flush=True)
    data = {}
    for name, split_dir, n in (("train", cfg.data.train_dir, cfg.data.n_train), ("val", cfg.data.val_dir, cfg.data.n_val)):
        if name == "train" and hybrid:
            keys, labels_list, embeddings, patterns, arrays = train_data
            data[name] = (keys, labels_list, embeddings, patterns, ForestTables(labels_list, embeddings, patterns, arrays, device))
            print(f"[bushy] train: {len(arrays['state']):,} rows ({int(arrays['bound'].sum()):,} with bound targets) "
                  f"from {len(labels_list):,} queries", flush=True)
            continue
        max_subsets = int(cfg.data.get("val_max_subsets", cfg.data.max_subsets)) if name == "val" else int(cfg.data.max_subsets)
        keys, labels_list, embeddings, patterns = prepare(cfg, split_dir, n, int(cfg.training.seed), oracle, embedder, device,
                                                          out, name, max_subsets)
        path = out / f"states_{name}.pt"
        if path.exists():
            rows = torch.load(path)
        else:
            start = time.perf_counter()
            rows = build_states(labels_list, int(cfg.data.rollouts), int(cfg.data.max_states), int(cfg.training.seed),
                                int(cfg.data.workers))
            torch.save(rows, path)
            print(f"[bushy] {name}: {len(rows):,} rows from {len(labels_list):,} queries ({time.perf_counter() - start:.0f}s)", flush=True)
        data[name] = (keys, labels_list, embeddings, patterns, ForestTables(labels_list, embeddings, patterns, rows, device))
    del oracle
    tables = data["train"][4]
    kwargs = dict(embedding_dim=tables.embeddings.shape[2], hidden_dim=int(cfg.model.hidden), n_message_layers=int(cfg.model.layers),
                  max_triple_patterns=max(32, tables.n_max), message_layer=cfg.model.message_layer, edge_feature_dim=JOIN_FEATURE_DIM,
                  moe=OmegaConf.to_container(cfg.model.moe) if cfg.model.get("moe") else None)
    model = ForestValueNet(**kwargs).to(device)
    if cfg.training.get("init_from"):            # warm start (e.g. for DAgger rounds), with its target statistics
        model.load_state_dict(torch.load(cfg.training.init_from, map_location=device, weights_only=False)["state_dict"])
        print(f"[bushy] initialised from {cfg.training.init_from}", flush=True)
    else:
        model.set_target_statistics(*tables.statistics())
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    tag = str(cfg.training.get("tag", ""))
    print(f"[bushy] train {tables.n_states:,} states ({len(tables.rows['state']):,} rows), "
          f"val {data['val'][4].n_states:,} states; {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    history, best = [], math.inf
    batch = int(cfg.training.batch_states)
    loss_settings = dict(under_weight=float(cfg.training.get("upper_bound_under_weight", 0.1)),
                         balance_weight=float(cfg.model.moe.get("balance_weight", 0.01)) if cfg.model.get("moe") else 0.0)

    def run_epoch(epoch, tables, phase):
        nonlocal best
        model.train()
        start, parts = time.perf_counter(), []
        for ids in torch.randperm(tables.n_states, device=device).split(batch):
            loss, part = training_loss(model, tables, ids, float(cfg.training.rank_weight), float(cfg.training.temperature),
                                       **loss_settings)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            parts.append(part)
        model.eval()
        with torch.no_grad():
            val_tables = data["val"][4]
            val_parts = [training_loss(model, val_tables, ids, float(cfg.training.rank_weight), float(cfg.training.temperature),
                                       **loss_settings)[1]
                         for ids in torch.arange(val_tables.n_states, device=device).split(batch)]
        metrics = {"epoch": epoch, "phase": phase, "seconds": time.perf_counter() - start,
                   **{f"train_{k}": float(np.mean([p[k] for p in parts])) for k in parts[0]},
                   **{f"val_{k}": float(np.mean([p[k] for p in val_parts])) for k in val_parts[0]},
                   **greedy_quality(model, val_tables, data["val"][1], device, int(cfg.training.selection_queries))}
        history.append(metrics)
        print(f"[bushy] {json.dumps(metrics)}", flush=True)
        if metrics["greedy_log_ratio_mean"] < best:
            best = metrics["greedy_log_ratio_mean"]
            torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs, "metrics": metrics,
                        "embedder": dict(cfg.embedder), "oracle": dict(cfg.oracle)}, out / f"value_net{tag}.pt")
        with open(out / f"metrics{tag}.json", "w") as f:
            json.dump(history, f, indent=2)

    epoch = 0
    for _ in range(int(cfg.training.epochs)):
        run_epoch(epoch, tables, "base")
        epoch += 1
    # DAgger: add the states the greedy policy itself visits (exact targets), keep training.
    rng = np.random.default_rng(int(cfg.training.seed))
    n_train = len(data["train"][1])
    train_rows = torch.load(out / "states_train.pt") if int(cfg.training.get("dagger_rounds", 0)) else None
    for round_index in range(int(cfg.training.get("dagger_rounds", 0))):
        query_ids = rng.choice(n_train, size=min(int(cfg.training.dagger_queries), n_train), replace=False).tolist()
        start = time.perf_counter()
        new_rows = dagger_rows(model, tables, data["train"][1], query_ids, round_index, device, int(cfg.data.workers))
        train_rows.extend(new_rows)
        _, labels_list, embeddings, patterns, _ = data["train"]
        tables = ForestTables(labels_list, embeddings, patterns, train_rows, device)
        print(f"[bushy] DAgger round {round_index}: +{len(new_rows):,} rows from greedy states of {len(query_ids):,} queries "
              f"({time.perf_counter() - start:.0f}s); now {tables.n_states:,} states", flush=True)
        for _ in range(int(cfg.training.dagger_epochs)):
            run_epoch(epoch, tables, f"dagger{round_index}")
            epoch += 1
    print(f"[bushy] best validation greedy log ratio {best:.3f} -> {out / f'value_net{tag}.pt'}", flush=True)


@hydra.main(version_base=None, config_path="../../../experiments/experiment_configs/amortized_dp",
            config_name="bushy_greedy_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
