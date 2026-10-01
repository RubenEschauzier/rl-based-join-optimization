"""Online training of the amortised-DP planner on real executions (no epinet yet).

    python -m src.supervised_value_estimation.amortized_dp.online.train_online
    python -m src.supervised_value_estimation.amortized_dp.online.train_online online.execution.mode=simulated

Starts from an offline-trained value net and adds a latency cost-to-go head. Each ROUND:
  1. take the next `queries_per_round` training queries;
  2. plan K plans per query with the current model: its greedy plan plus K-1 one-step
     deviations at the closest calls (partial_observation.deviation_plans), ranked by
     predicted C_out cost-to-go;
  3. execute them, each with a timeout fixed before the round:
         min(default, max(min, multiplier * fastest known latency of that query));
  4. add what each runtime tree reveals to the query's observation store: exact
     cardinalities of every intermediate and scan, and per-step times;
  5. train on the most recent `replay_queries` observed queries: card, stitched C_out
     cost-to-go and stitched latency cost-to-go (both upper bounds: one-sided losses), and the
     listwise loss over observed children.
Every `validation.every_rounds`, the greedy plan of each validation query is executed once,
back to back with QLever's own plan (join order left to QLever), and the
latency, timeout rate and speedup are logged. Plans are chosen by C_out in this version; the
latency head is trained alongside but not yet used for decisions.

Outputs in online.output_directory/online-<mode>-<time>/: config.yaml, rounds.jsonl,
validation.jsonl, model-<round>.pt, and state.pkl (observations, for resuming).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from src.supervised_value_estimation.amortized_dp.agents import AmortizedDPAgent, load_cardinality_gnn
from src.supervised_value_estimation.amortized_dp.compare_agents import _resolve_checkpoint
from src.supervised_value_estimation.amortized_dp.data import (
    compute_embeddings, load_datasets, load_split, query_index,
)
from src.supervised_value_estimation.amortized_dp.labels import connected_subset_masks, family_of
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.online.executor import RayPlanExecutor, SimulatedExecutor
from src.supervised_value_estimation.amortized_dp.online.observations import QueryObservations
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import parse_execution
from src.supervised_value_estimation.amortized_dp.partial_observation import _TrainingSettings, deviation_plans
from src.supervised_value_estimation.amortized_dp.train_amortized_dp import StateTables, training_step
from src.supervised_value_estimation.optuna_epinet_sweep import _resolve_model_paths
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search


def _is_connected(neighbour_masks):
    reached, frontier = 1, 1
    while frontier:
        new = 0
        for i in range(len(neighbour_masks)):
            if frontier >> i & 1:
                new |= neighbour_masks[i]
        frontier = new & ~reached
        reached |= new
    return reached == (1 << len(neighbour_masks)) - 1


def _observations_for(data):
    masks, neighbour_masks = connected_subset_masks(data)
    observations = QueryObservations(data.query, family_of(data.type), len(data.triple_patterns), neighbour_masks)
    observations.triple_patterns = list(data.triple_patterns)
    return observations


def _timeout(observations, timeouts):
    best = observations.best_latency_s
    if best is None:
        return float(timeouts.default_s)
    return float(min(timeouts.default_s, max(timeouts.min_s, timeouts.multiplier * best)))


def _load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device)
    model = ContractedJoinGraphValueNet(**{**saved["model_kwargs"], "latency_head": True})
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    new = sorted(k for k in missing if not k.startswith("latency"))
    if new or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing {new}, unexpected {unexpected}")
    return model.to(device), {**saved["model_kwargs"], "latency_head": True}


def _summarise_latencies(latencies, censored, native=None):
    # QLever reports whole milliseconds, so ratios of sub-millisecond latencies are noise.
    latencies = np.maximum(np.asarray(latencies, dtype=float), 0.001)
    summary = {"n": int(len(latencies)), "mean_s": float(latencies.mean()), "median_s": float(np.median(latencies)),
               "p90_s": float(np.percentile(latencies, 90)), "p99_s": float(np.percentile(latencies, 99)),
               "timeout_rate": float(np.mean(censored))}
    if native is not None:
        native = np.maximum(np.asarray(native, dtype=float), 0.001)
        ratio = native / latencies
        summary.update({"native_mean_s": float(native.mean()), "native_median_s": float(np.median(native)),
                        "geomean_speedup_vs_native": float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-6))))),
                        "frac_faster_than_native": float(np.mean(latencies < native)),
                        "frac_slower_than_2x_native": float(np.mean(latencies > 2 * native))})
    return summary


@hydra.main(version_base=None,
            config_path="../../../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="online_amortized_dp_mixed_yago.yaml")
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    online, cfg_training = cfg.online, cfg.amortized_dp.training
    rng = np.random.default_rng(online.seed)
    torch.manual_seed(online.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_directory = Path(online.output_directory) / (
        f"online-{online.execution.mode}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}")

    train_dataset, val_dataset = load_datasets(cfg)
    val_queries, _ = load_split(cfg)
    train_pool = [i for i in range(len(train_dataset))
                  if _is_connected(connected_subset_masks(train_dataset[i])[1])]
    rng.shuffle(train_pool)
    val_index = query_index(val_dataset)
    validation_pool = [val_index[q] for q in val_queries if q in val_index
                       and _is_connected(connected_subset_masks(val_dataset[val_index[q]])[1])]
    validation_pool = [validation_pool[k] for k in
                       sorted(rng.choice(len(validation_pool), min(online.validation.n_queries, len(validation_pool)),
                                         replace=False))]
    print(f"Train pool: {len(train_pool)} variable-connected queries | validation: {len(validation_pool)}")

    checkpoint = _resolve_checkpoint(online.init_checkpoint, cfg.amortized_dp.output_directory)
    model, model_kwargs = _load_model(checkpoint, device)
    print(f"Initialised from {checkpoint} (+ new latency head)")
    embedder = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, device)

    if online.execution.mode == "ray":
        executor = RayPlanExecutor(list(online.execution.endpoints), n_actors=online.execution.n_actors,
                                   in_flight_per_endpoint=online.execution.in_flight_per_endpoint,
                                   ray_address=online.execution.get("ray_address"))
    elif online.execution.mode == "simulated":
        # No database: the oracle's cardinalities stand in for execution. Needs oracle labels.
        from src.supervised_value_estimation.amortized_dp.data import load_or_build_labels
        oracle = load_cardinality_gnn(cfg.models.oracle.config, cfg.models.oracle.dir, device)
        wanted = train_pool[:online.rounds * online.queries_per_round]
        labels = load_or_build_labels(Path(cfg.amortized_dp.label_directory) / "online_simulated_train.pkl",
                                      train_dataset, wanted, oracle, device)
        labels.update(load_or_build_labels(Path(cfg.amortized_dp.label_directory) / "online_simulated_val.pkl",
                                           val_dataset, validation_pool, oracle, device))
        del oracle
        executor = SimulatedExecutor(labels, seed=online.seed)
    else:
        raise ValueError(f"online.execution.mode must be ray or simulated, got {online.execution.mode}")

    observations: dict[str, QueryObservations] = {}
    embeddings: dict[str, torch.Tensor] = {}
    order_seen: list[str] = []            # queries in the order they were first executed
    loss_settings = _TrainingSettings(cfg_training, online.upper_bound_under_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=online.lr, weight_decay=cfg_training.weight_decay)
    latency_statistics_set = False

    def ensure_embedded(dataset, indices):
        missing = [i for i in indices if dataset[i].query not in embeddings]
        if missing:
            embeddings.update(compute_embeddings(embedder, dataset, missing, device))

    def validate(round_index):
        ensure_embedded(val_dataset, validation_pool)
        model.eval()
        agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=embeddings)
        # Our plan and QLever's own plan are executed back to back for every query, in
        # alternating order, at EVERY validation. Measuring QLever's plans once and reusing
        # them would compare a cold first run against later warm (page-cached) runs.
        requests, kinds = [], []
        for k, i in enumerate(validation_pool):
            data = val_dataset[i]
            plan = beam_search(Batch.from_data_list([data]), agent, 1)[0]["plan"]
            pair = [("ours", plan)] + ([("native", None)] if online.validation.include_native else [])
            if k % 2:
                pair.reverse()
            for kind, chosen in pair:
                requests.append({"query": data.query, "triple_patterns": list(data.triple_patterns), "plan": chosen,
                                 "timeout_s": float(online.timeouts.default_s)})
                kinds.append(kind)
        results = executor.execute(requests)
        ours = [r for r, kind in zip(results, kinds) if kind == "ours"]
        native = ([r["latency_s"] for r, kind in zip(results, kinds) if kind == "native"]
                  if online.validation.include_native else None)
        summary = {"round": round_index, "executed_plans_total": sum(len(o.executions) for o in observations.values()),
                   **_summarise_latencies([r["latency_s"] for r in ours], [r["censored"] for r in ours], native)}
        with open(run_directory / "validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        speed = (f", geomean speedup vs QLever {summary['geomean_speedup_vs_native']:.3f}, faster on "
                 f"{summary['frac_faster_than_native']:.1%}") if native is not None else ""
        print(f"[validation round {round_index}] mean {summary['mean_s']:.3f}s, median {summary['median_s']:.3f}s, "
              f"p90 {summary['p90_s']:.3f}s, timeouts {summary['timeout_rate']:.1%}{speed}")
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "round": round_index},
                   run_directory / f"model-{round_index}.pt")
        with open(run_directory / "state.pkl", "wb") as f:
            pickle.dump({"observations": observations, "order_seen": order_seen, "round": round_index}, f)
        model.train()

    if online.validation.at_start:
        validate(0)

    cursor = 0
    for round_index in range(1, online.rounds + 1):
        start = time.perf_counter()
        batch = [train_pool[(cursor + k) % len(train_pool)] for k in range(online.queries_per_round)]
        cursor += online.queries_per_round
        ensure_embedded(train_dataset, batch)
        model.eval()
        agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=embeddings)
        requests, is_greedy = [], []
        for i in batch:
            data = train_dataset[i]
            if data.query not in observations:
                observations[data.query] = _observations_for(data)
                order_seen.append(data.query)
            obs = observations[data.query]
            timeout_s = _timeout(obs, online.timeouts)
            plans = deviation_plans(agent, Batch.from_data_list([data]), obs.labels, online.plans_per_query,
                                    online.deviation_steps, rng)
            for k, plan in enumerate(plans):
                requests.append({"query": data.query, "triple_patterns": list(data.triple_patterns),
                                 "plan": plan, "timeout_s": timeout_s})
                is_greedy.append(k == 0)
        planning_seconds = time.perf_counter() - start

        results = executor.execute(requests)
        execution_seconds = time.perf_counter() - start - planning_seconds
        aligned = cached = 0
        for request, result in zip(requests, results):
            parsed = parse_execution(result["tree"], request["plan"], request["triple_patterns"])
            aligned += parsed["aligned"]
            cached += parsed["any_cached"]
            observations[request["query"]].add(request["plan"], parsed, result["latency_s"], result["censored"],
                                               result["timeout_s"], cached=parsed["any_cached"])

        # Train on the most recently executed queries.
        window = order_seen[-online.replay_queries:]
        window_obs = [observations[q] for q in window]
        tables = StateTables([o.labels for o in window_obs], [embeddings[q] for q in window], device,
                             patterns_list=[o.triple_patterns for o in window_obs],
                             observed_logcards=[o.logcard for o in window_obs], cost_to_go_upper_bound=True,
                             latency_to_go=[o.latency_to_go() for o in window_obs])
        if not latency_statistics_set:
            model.set_latency_statistics(*tables.latency_statistics())
            latency_statistics_set = True
        model.train()
        running = []
        for _ in range(online.gradient_steps_per_round):
            state_ids = torch.randint(0, tables.n_states, (cfg_training.states_per_batch,), device=device)
            loss, parts = training_step(model, tables, state_ids, loss_settings)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
            optimizer.step()
            running.append(parts)
        train_seconds = time.perf_counter() - start - planning_seconds - execution_seconds

        greedy = [r for r, first in zip(results, is_greedy) if first]
        summary = {
            "round": round_index, "queries_seen": len(order_seen), "plans_executed": len(requests),
            "timeout_rate": float(np.mean([r["censored"] for r in results])),
            "failed_rate": float(np.mean([bool(r.get("failed")) for r in results])),
            "aligned_rate": aligned / len(results), "cached_rate": cached / len(results),
            "greedy_mean_latency_s": float(np.mean([r["latency_s"] for r in greedy])),
            "greedy_timeout_rate": float(np.mean([r["censored"] for r in greedy])),
            "all_plans_mean_latency_s": float(np.mean([r["latency_s"] for r in results])),
            **{key: float(np.mean([p[key] for p in running if key in p])) for key in running[0]},
            "planning_seconds": planning_seconds, "execution_seconds": execution_seconds,
            "train_seconds": train_seconds, "window_states": tables.n_states,
        }
        with open(run_directory / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"Round {round_index}: {len(requests)} plans ({summary['timeout_rate']:.1%} timeouts, "
              f"{summary['aligned_rate']:.1%} aligned, {summary['cached_rate']:.1%} cached), loss {summary['loss']:.3f} "
              f"| plan {planning_seconds:.0f}s, execute {execution_seconds:.0f}s, train {train_seconds:.0f}s")
        if round_index % online.validation.every_rounds == 0 or round_index == online.rounds:
            validate(round_index)
    executor.close()


if __name__ == "__main__":
    main()
