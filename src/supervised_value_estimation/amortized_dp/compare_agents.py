"""Compare planning agents on the held-out test half, scored against the exact DP optimum.

    python -m src.supervised_value_estimation.amortized_dp.compare_agents

Every agent runs through the existing multiprocess_validate_agent + beam_search, with
SimulatedCostExecutionStrategy in place of a database: a plan's cost is its C_out under the
oracle's cardinalities, divided by the cost of the exact left-deep DP optimum under the
same cardinalities (1.0 = optimal).

Agents (comparison.agents in the config):
    learned_cardinality   existing CardinalityEstimatorValidationAgent + the learned GNN
                          cardinality model (the same frozen GNN that embeds for amortized_dp)
    plan_cost_model       the trial-85 plan-cost network's base head (as CostEstimatorAgent)
    amortized_dp          the contracted-join-graph value net
    oracle_cardinality    CardinalityEstimatorValidationAgent with the oracle (optional)
plus, computed in-process: exact DP over the LEARNED cardinality model (a classical
optimizer with learned estimates), with its planning time measured single-threaded.

Outputs: amortized_dp.output_directory/comparison-<time>/{summary.json, summary.md,
<agent>.jsonl}.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from src.supervised_value_estimation.amortized_dp.agents import (
    build_amortized_dp_agent, build_learned_cardinality_agent, build_plan_cost_model_agent, load_cardinality_gnn,
)
from src.supervised_value_estimation.amortized_dp.data import (
    load_datasets, load_or_build_labels, load_split, query_index, sample_indices,
)
from src.supervised_value_estimation.amortized_dp.labels import (
    QueryLabels, connected_subset_masks, family_of, model_log_cardinalities, optimal_left_deep_plan,
)
from src.supervised_value_estimation.amortized_dp.simulated_execution import (
    SimulatedCostExecutionStrategy, summarize,
)
from src.supervised_value_estimation.optuna_epinet_sweep import _resolve_model_paths
from src.supervised_value_estimation.summarize_epinet_rerun import _find_runs
from src.supervised_value_estimation.validation.validation_runner import multiprocess_validate_agent


def _latest_value_net_checkpoint(output_directory, message_layer=None):
    """Newest finished run's best checkpoint, optionally restricted to one message layer."""
    candidates = [path for path in glob.glob(os.path.join(output_directory, "run-*", "best_model.pt"))
                  if os.path.exists(os.path.join(os.path.dirname(path), "summary.json"))]
    if message_layer is not None:
        candidates = [path for path in candidates
                      if torch.load(path, map_location="cpu")["model_kwargs"].get("message_layer", "mean")
                      == message_layer]
    if not candidates:
        raise FileNotFoundError(f"No finished amortized_dp run{'' if message_layer is None else f' with {message_layer}'} "
                                f"under {output_directory}; train one first or give a checkpoint path.")
    return max(candidates, key=os.path.getmtime)


def _resolve_checkpoint(spec, output_directory):
    """A checkpoint path, 'latest', or 'latest:<message layer>'."""
    if spec is None or spec == "latest":
        return _latest_value_net_checkpoint(output_directory)
    if str(spec).startswith("latest:"):
        return _latest_value_net_checkpoint(output_directory, str(spec).split(":", 1)[1])
    return str(spec)


def _plan_cost_checkpoint(cfg, seed):
    runs = _find_runs(cfg.rerun.output_directory)
    if seed not in runs or not runs[seed][1]:
        raise FileNotFoundError(f"No finished trial-85 rerun for seed {seed} under {cfg.rerun.output_directory}.")
    run_directory = runs[seed][0]
    with open(os.path.join(run_directory, "final_summary.json"), encoding="utf-8") as f:
        best_epoch = json.load(f)["best_epoch"]
    return os.path.join(run_directory, f"epoch-{best_epoch}", "model", "epinet_model.pt")


def _agent_spec(kind, cfg, value_net_checkpoint, plan_cost_checkpoint, agent_checkpoint=None):
    if kind == "learned_cardinality":
        return build_learned_cardinality_agent, {"model_config": cfg.models.embedder.config,
                                                 "model_dir": cfg.models.embedder.dir}
    if kind == "oracle_cardinality":
        return build_learned_cardinality_agent, {"model_config": cfg.models.oracle.config,
                                                 "model_dir": cfg.models.oracle.dir}
    if kind == "plan_cost_model":
        return build_plan_cost_model_agent, {"run_config": OmegaConf.to_container(cfg, resolve=True),
                                             "checkpoint_path": plan_cost_checkpoint,
                                             "seed": cfg.comparison.plan_cost_seed}
    if kind == "amortized_dp":
        checkpoint = (value_net_checkpoint if agent_checkpoint is None
                      else _resolve_checkpoint(agent_checkpoint, cfg.amortized_dp.output_directory))
        return build_amortized_dp_agent, {"value_net_checkpoint": checkpoint,
                                          "embedder_config": cfg.models.embedder.config,
                                          "embedder_dir": cfg.models.embedder.dir}
    raise ValueError(f"Unknown agent kind: {kind}")


def _dp_with_learned_cardinality(labels_by_query, strategy):
    results = []
    for query, labels in labels_by_query.items():
        order, _ = optimal_left_deep_plan(labels, labels.estimated_logcard)
        results.append(strategy.score(query, order))
    return results


def _time_dp_with_learned_cardinality(cfg, dataset, indices):
    """End-to-end planning time: estimate every connected subset, then exact bitmask DP."""
    torch.set_num_threads(1)
    gnn = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, torch.device("cpu"))
    times = []
    for i in indices:
        query = dataset[i]
        start = time.perf_counter()
        masks, neighbour_masks = connected_subset_masks(query)
        estimates = model_log_cardinalities(gnn, query, masks, torch.device("cpu"))
        labels = QueryLabels(query=query.query, family=family_of(query.type), n_tp=len(query.triple_patterns),
                             neighbour_masks=neighbour_masks, logcard=dict(zip(masks, estimates)))
        optimal_left_deep_plan(labels)
        times.append(time.perf_counter() - start)
    times = np.asarray(times) * 1000
    return {"planning_time_mean_ms": times.mean(), "planning_time_median_ms": np.median(times),
            "planning_time_p90_ms": np.percentile(times, 90), "planning_time_p99_ms": np.percentile(times, 99),
            "planning_time_max_ms": times.max()}


def _markdown(summaries, test_count):
    columns = ["geomean_ratio", "median_ratio", "p90_ratio", "p95_ratio", "p99_ratio", "max_ratio",
               "frac_optimal", "frac_within_1.1x", "frac_within_2x"]
    lines = [f"# Planning agents vs exact DP optimum ({test_count} held-out test queries)", "",
             "Cost ratio = C_out of the chosen plan / C_out of the optimal left-deep plan, both under the "
             "oracle's cardinalities (1.0 = optimal). Planning time per query, single CPU thread per worker.", "",
             "| agent | " + " | ".join(columns) + " | plan ms mean | plan ms p90 |",
             "|" + "---|" * (len(columns) + 3)]
    for name, summary in summaries.items():
        overall, planning = summary["overall"], summary.get("planning", {})
        cells = [f"{overall[c]:.1%}" if c.startswith("frac") else f"{overall[c]:.3f}" for c in columns]
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {planning.get('planning_time_mean_ms', math.nan):.1f} "
                     f"| {planning.get('planning_time_p90_ms', math.nan):.1f} |")
    sizes = sorted({int(k) for s in summaries.values() for k in s["by_n_tp"]})
    lines += ["", "## Geometric-mean cost ratio by query size (triple patterns)", "",
              "| agent | " + " | ".join(str(n) for n in sizes) + " |", "|" + "---|" * (len(sizes) + 1)]
    for name, summary in summaries.items():
        by_size = summary["by_n_tp"]
        lines.append(f"| {name} | " + " | ".join(
            f"{by_size[str(n)]['geomean_ratio']:.3f}" if str(n) in by_size else "–" for n in sizes) + " |")
    families = sorted({k for s in summaries.values() for k in s["by_family"]})
    lines += ["", "## Geometric-mean cost ratio by query family", "",
              "| agent | " + " | ".join(families) + " |", "|" + "---|" * (len(families) + 1)]
    for name, summary in summaries.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{summary['by_family'][f]['geomean_ratio']:.3f}" if f in summary["by_family"] else "–"
            for f in families) + " |")
    return "\n".join(lines) + "\n"


@hydra.main(version_base=None,
            config_path="../../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="amortized_dp_mixed_yago.yaml")
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings, comparison = cfg.amortized_dp, cfg.comparison
    output_directory = Path(settings.output_directory) / f"comparison-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    output_directory.mkdir(parents=True, exist_ok=False)

    _, val_dataset = load_datasets(cfg)
    _, test_queries = load_split(cfg)
    index = query_index(val_dataset)
    test_indices = sample_indices([index[q] for q in test_queries if q in index],
                                  comparison.max_test_queries, comparison.test_query_seed)
    
    oracle = load_cardinality_gnn(cfg.models.oracle.config, cfg.models.oracle.dir, device)
    estimator = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, device)

    labels = load_or_build_labels(Path(settings.label_directory) / "test.pkl", val_dataset, test_indices,
                                  oracle, device, estimator=estimator)
    del oracle, estimator

    # No agent can plan a query that is not variable-connected without a cartesian product
    # (beam_search returns nothing and the runner silently skips it), so score everyone on
    # the same, plannable queries and report how many were excluded.
    n_labelled = len(test_indices)
    test_indices = [i for i in test_indices if labels[val_dataset[i].query].is_connected]
    labels = {query: value for query, value in labels.items() if value.is_connected}
    n_excluded = n_labelled - len(test_indices)
    print(f"Excluded {n_excluded} of {n_labelled} test queries that are not variable-connected")
    
    strategy = SimulatedCostExecutionStrategy(labels)

    value_net_checkpoint = _resolve_checkpoint(comparison.value_net_checkpoint, settings.output_directory)
    plan_cost_checkpoint = _plan_cost_checkpoint(cfg, comparison.plan_cost_seed)
    print(f"Value net: {value_net_checkpoint}\nPlan-cost model: {plan_cost_checkpoint}\n"
          f"Test queries: {len(test_indices)}")
    loader = DataLoader(val_dataset[test_indices], batch_size=1, shuffle=False)

    summaries = {}
    dp_results = _dp_with_learned_cardinality(labels, strategy)
    timing = _time_dp_with_learned_cardinality(
        cfg, val_dataset, test_indices[:comparison.dp_timing_queries])
    summaries["exact_dp_learned_cardinality"] = summarize(dp_results, timing)
    with open(output_directory / "exact_dp_learned_cardinality.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in dp_results)

    for agent in comparison.agents:
        if not agent.get("enabled", True):
            continue
        builder, kwargs = _agent_spec(agent.kind, cfg, value_net_checkpoint, plan_cost_checkpoint,
                                      agent.get("checkpoint"))
        if agent.kind == "amortized_dp":
            print(f"{agent.name}: value net {kwargs['value_net_checkpoint']}")
        print(f"\n=== {agent.name} (beam {agent.beam_width}) ===")
        metrics, results = multiprocess_validate_agent(
            val_loader=loader, execution_strategy=strategy, agent_builder_fn=builder, agent_kwargs=kwargs,
            beam_width=agent.beam_width, num_workers=comparison.num_workers,
            samples_per_execution_batch=comparison.samples_per_execution_batch,
        )
        summaries[agent.name] = summarize(results, metrics)
        # The runner drops queries for which beam_search returns no plan; never let an agent
        # be scored on an easier subset without saying so.
        summaries[agent.name]["n_planned"] = len(results)
        if len(results) != len(test_indices):
            print(f"WARNING: {agent.name} planned {len(results)} of {len(test_indices)} queries.")
        with open(output_directory / f"{agent.name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
        overall = summaries[agent.name]["overall"]
        print(f"{agent.name}: geomean ratio {overall['geomean_ratio']:.4f} | optimal {overall['frac_optimal']:.1%} "
              f"| p95 {overall['p95_ratio']:.3f} | planning {metrics['planning_time_mean_ms']:.1f} ms")

    with open(output_directory / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"value_net_checkpoint": value_net_checkpoint,
                   "agent_checkpoints": {a.name: _agent_spec(a.kind, cfg, value_net_checkpoint, plan_cost_checkpoint,
                                                             a.get("checkpoint"))[1].get("value_net_checkpoint")
                                         for a in comparison.agents
                                         if a.get("enabled", True) and a.kind == "amortized_dp"}, "plan_cost_checkpoint": plan_cost_checkpoint,
                   "n_test_queries": len(test_indices), "n_excluded_not_connected": n_excluded,
                   "agents": summaries}, f, indent=2)
    markdown = _markdown(summaries, len(test_indices))
    with open(output_directory / "summary.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    print("\n" + markdown)


if __name__ == "__main__":
    main()
