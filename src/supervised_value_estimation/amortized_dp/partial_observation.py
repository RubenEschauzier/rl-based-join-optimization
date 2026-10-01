"""Offline simulation of learning from executed plans only (no full-oracle teacher).

    python -m src.supervised_value_estimation.amortized_dp.partial_observation
    python -m src.supervised_value_estimation.amortized_dp.partial_observation \
        partial_observation.plans_per_round=8 partial_observation.plan_selection=random

Each ROUND, for every training query, the current model plans K plans, which are then
"executed": the oracle reveals the cardinality of every intermediate each plan produces
(its prefix sets) and of every scanned pattern, exactly what QLever's runtime tree reports.
Observations accumulate per query in an observed table that only grows.

Training targets come from that table alone -- never from model predictions:
    card        exact, for every observed subset
    cost-to-go  DP over OBSERVED entries only ("stitching"): the cheapest completion that
                can be assembled from observed pieces, including plans nobody executed. An
                upper bound on the true cost-to-go, trained with a one-sided loss.
    rank        listwise over a state's observed children, by their known completion cost

Plan selection per query (K plans):
    deviation   the model's greedy plan, plus K-1 one-step deviations: follow the greedy
                plan to some depth, take the second-best join there, finish greedily.
                `deviation_steps` picks the depths: closest (smallest score gap first) or
                random.
    random      K random connected left-deep plans (a no-model baseline).
    oracle      no plans: the full oracle table is revealed in round 1 (exact cost-to-go, plain
                MSE). The like-for-like upper bound: same queries, same training schedule.

After each round the model is scored by greedy decoding on validation queries against the
exact optimum, and the run records how much of each query's subset table has been
observed, how good the best EXECUTED plan is, and how good the best STITCHED plan is.
Outputs: amortized_dp.output_directory/partial-<selection>-K<k>-<time>/.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from src.supervised_value_estimation.amortized_dp.agents import AmortizedDPAgent, load_cardinality_gnn
from src.supervised_value_estimation.amortized_dp.data import (
    compute_embeddings, load_datasets, load_or_build_labels, load_split, query_index, sample_indices,
)
from src.supervised_value_estimation.amortized_dp.labels import (
    JOIN_FEATURE_DIM, cost_to_go, logsumexp, mask_of, optimal_left_deep_plan, plan_log_cost,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.train_amortized_dp import (
    StateTables, greedy_selection_metrics, training_step,
)
from src.supervised_value_estimation.optuna_epinet_sweep import _resolve_model_paths


# --- plan selection ----------------------------------------------------------------------

def _candidates(prefix, labels):
    """Next prefixes, as beam_search generates them (first pair in one orientation only)."""
    if not prefix:
        return [[i, j] for i, j in labels.connected_pairs()]
    reach = labels.neighbours(mask_of(prefix))
    return [prefix + [a] for a in range(labels.n_tp) if reach >> a & 1]


def _ranked(agent, episode, prefix, labels):
    candidates = _candidates(prefix, labels)
    scores, _ = agent.estimate_costs(candidates, episode)
    order = np.argsort(scores, kind="stable")
    return [candidates[k] for k in order], [scores[k] for k in order]


def _greedy_from(agent, episode, prefix, labels):
    while len(prefix) < labels.n_tp:
        ranked, _ = _ranked(agent, episode, prefix, labels)
        prefix = ranked[0]
    return prefix


def deviation_plans(agent, query_batch, labels, n_plans, deviation_steps, rng):
    """The greedy plan plus up to n_plans-1 one-step deviations from it."""
    episode = agent.setup_episode(query_batch)
    prefix, decisions = [], []
    while len(prefix) < labels.n_tp:
        ranked, scores = _ranked(agent, episode, prefix, labels)
        decisions.append((ranked, scores))
        prefix = ranked[0]
    plans = [prefix]
    branchable = [depth for depth, (ranked, _) in enumerate(decisions) if len(ranked) > 1]
    if deviation_steps == "closest":
        branchable.sort(key=lambda depth: decisions[depth][1][1] - decisions[depth][1][0])
    else:
        rng.shuffle(branchable)
    for depth in branchable:
        if len(plans) >= n_plans:
            break
        plan = _greedy_from(agent, episode, decisions[depth][0][1], labels)
        if plan not in plans:
            plans.append(plan)
    return plans


def random_plan(labels, rng):
    i, j = labels.connected_pairs()[rng.integers(len(labels.connected_pairs()))]
    plan = [i, j]
    while len(plan) < labels.n_tp:
        reach = labels.neighbours(mask_of(plan))
        plan.append(int(rng.choice([a for a in range(labels.n_tp) if reach >> a & 1])))
    return plan


# --- observation and stitching -----------------------------------------------------------

def observe(plan, labels, table):
    """What executing `plan` reveals: every scanned pattern and every intermediate result."""
    for pattern in plan:
        table[1 << pattern] = labels.logcard[1 << pattern]
    for size in range(2, len(plan) + 1):
        mask = mask_of(plan[:size])
        table[mask] = labels.logcard[mask]


def best_stitched_log_cost(labels, table):
    """Cheapest complete plan whose every prefix is observed (possibly never executed)."""
    log_g = cost_to_go(labels, table)
    best = math.inf
    for i, j in labels.connected_pairs():
        pair = (1 << i) | (1 << j)
        # -inf (log 0) means the pair already is the whole query; +inf means no known completion.
        if pair in table and log_g.get(pair, math.inf) < math.inf:
            best = min(best, logsumexp([min(table[1 << i], table[1 << j]), table[pair], log_g[pair]]))
    return best


def _geomean_ratio(log_costs, optimal):
    ratios = [c - o for c, o in zip(log_costs, optimal) if math.isfinite(c)]
    return float(math.exp(np.mean(ratios))) if ratios else math.nan


@hydra.main(version_base=None,
            config_path="../../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="partial_observation_mixed_yago.yaml")
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    settings, cfg_training, sim = cfg.amortized_dp, cfg.amortized_dp.training, cfg.partial_observation
    torch.manual_seed(sim.seed)
    rng = np.random.default_rng(sim.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"partial-{sim.plan_selection}-K{sim.plans_per_round}-{settings.model.message_layer}"
    run_directory = Path(settings.output_directory) / f"{name}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}")

    train_dataset, val_dataset = load_datasets(cfg)
    val_queries, _ = load_split(cfg)
    val_index = query_index(val_dataset)
    # A subset of the queries the full-oracle run labelled, so the label cache is reused.
    labelled_indices = sample_indices(range(len(train_dataset)), settings.n_train_queries, settings.train_query_seed)
    selection_indices = sample_indices([val_index[q] for q in val_queries if q in val_index],
                                       settings.n_selection_queries, settings.train_query_seed + 1)
    oracle = load_cardinality_gnn(cfg.models.oracle.config, cfg.models.oracle.dir, device)
    label_directory = Path(settings.label_directory)
    train_labels = load_or_build_labels(label_directory / "train.pkl", train_dataset, labelled_indices, oracle, device)
    selection_labels = load_or_build_labels(label_directory / "val.pkl", val_dataset, selection_indices, oracle, device)
    del oracle
    train_indices = [i for i in labelled_indices if train_labels[train_dataset[i].query].is_connected]
    train_indices = sample_indices(train_indices, sim.n_train_queries, sim.seed)
    selection_indices = [i for i in selection_indices if selection_labels[val_dataset[i].query].is_connected]

    embedder = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, device)
    train_embeddings = compute_embeddings(embedder, train_dataset, train_indices, device)
    selection_embeddings = compute_embeddings(embedder, val_dataset, selection_indices, device)
    del embedder

    queries = [train_dataset[i].query for i in train_indices]
    labels_list = [train_labels[q] for q in queries]
    patterns = [list(train_dataset[i].triple_patterns) for i in train_indices]
    optimal = [optimal_left_deep_plan(labels)[1] for labels in labels_list]
    observed = [{} for _ in queries]
    best_executed = [math.inf for _ in queries]

    model_kwargs = {"embedding_dim": train_embeddings[queries[0]].shape[1], "hidden_dim": settings.model.hidden_dim,
                    "n_message_layers": settings.model.n_message_layers,
                    "message_layer": settings.model.message_layer,
                    "edge_feature_dim": JOIN_FEATURE_DIM if settings.model.edge_features else 0}
    model = ContractedJoinGraphValueNet(**model_kwargs).to(device)
    loss_settings = _TrainingSettings(cfg_training, sim.upper_bound_under_weight)
    statistics_set = False

    for round_index in range(1, sim.rounds + 1):
        start = time.perf_counter()
        model.eval()
        agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=train_embeddings)
        for k, i in enumerate(train_indices):
            labels = labels_list[k]
            if sim.plan_selection == "oracle":
                observed[k] = dict(labels.logcard)
                best_executed[k] = optimal[k]
                continue
            if sim.plan_selection == "random":
                plans = [random_plan(labels, rng) for _ in range(sim.plans_per_round)]
            else:
                plans = deviation_plans(agent, Batch.from_data_list([train_dataset[i]]), labels,
                                        sim.plans_per_round, sim.deviation_steps, rng)
            for plan in plans:
                observe(plan, labels, observed[k])
                best_executed[k] = min(best_executed[k], plan_log_cost(plan, labels))
        planning_seconds = time.perf_counter() - start

        tables = StateTables(labels_list, [train_embeddings[q] for q in queries], device, patterns_list=patterns,
                             observed_logcards=observed,
                             cost_to_go_upper_bound=sim.plan_selection != "oracle")
        if not statistics_set:
            # Standardisation from the first round's observations, then kept fixed so the
            # model's outputs do not shift under it between rounds.
            model.set_target_statistics(*tables.statistics())
            statistics_set = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_training.lr, weight_decay=cfg_training.weight_decay)
        model.train()
        running = []
        for _ in range(sim.epochs_per_round):
            order = torch.randperm(tables.n_states, device=device)
            for batch_start in range(0, tables.n_states, cfg_training.states_per_batch):
                loss, parts = training_step(model, tables, order[batch_start:batch_start + cfg_training.states_per_batch],
                                            loss_settings)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
                optimizer.step()
                running.append(parts)
        train_seconds = time.perf_counter() - start - planning_seconds

        selection = greedy_selection_metrics(model, val_dataset, selection_indices, selection_embeddings,
                                             selection_labels, device)
        coverage = float(np.mean([len(t) / len(l.logcard) for t, l in zip(observed, labels_list)]))
        stitched = [best_stitched_log_cost(l, t) for l, t in zip(labels_list, observed)]
        metrics = {
            "round": round_index,
            "executed_plans_per_query": round_index * sim.plans_per_round,
            "subset_coverage": coverage,
            "train_best_executed_geomean_ratio": _geomean_ratio(best_executed, optimal),
            "train_best_stitched_geomean_ratio": _geomean_ratio(stitched, optimal),
            "train_frac_stitched_better_than_executed": float(np.mean(
                [s < e - 1e-9 for s, e in zip(stitched, best_executed)])),
            "states": tables.n_states, "rows": int(len(tables.rows["mask"])),
            **{key: float(np.mean([r[key] for r in running])) for key in running[0]},
            "planning_seconds": planning_seconds, "train_seconds": train_seconds,
            "val_greedy": selection,
        }
        with open(run_directory / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"Round {round_index}: {metrics['executed_plans_per_query']} plans/query, coverage {coverage:.1%} | "
              f"train best executed {metrics['train_best_executed_geomean_ratio']:.3f}, stitched "
              f"{metrics['train_best_stitched_geomean_ratio']:.3f} (better on "
              f"{metrics['train_frac_stitched_better_than_executed']:.1%}) | val greedy "
              f"{selection['overall']['geomean_ratio']:.4f}, optimal {selection['overall']['frac_optimal']:.1%} "
              f"| {planning_seconds:.0f}s plan, {train_seconds:.0f}s train")
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "round": round_index,
                    "metrics": metrics}, run_directory / "model.pt")

    with open(run_directory / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"final_round": sim.rounds, "model": str(run_directory / "model.pt")}, f, indent=2)


class _TrainingSettings:
    """cfg.amortized_dp.training plus the one-sided loss weight, readable like the config."""

    def __init__(self, cfg_training, upper_bound_under_weight):
        self._cfg = cfg_training
        self.upper_bound_under_weight = upper_bound_under_weight

    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def get(self, name, default=None):
        return self.upper_bound_under_weight if name == "upper_bound_under_weight" else self._cfg.get(name, default)


if __name__ == "__main__":
    main()
