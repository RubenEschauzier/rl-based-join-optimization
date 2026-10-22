"""Online training of the BUSHY value net (bushy_model.ForestValueNet) on real QLever executions,
the bushy counterpart of train_online.py (left-deep), which is unchanged.

    python -m src.supervised_value_estimation.amortized_dp.online.train_online_bushy [overrides]

Starts from an offline-trained forest value net (e.g. the hybrid-target, fine-tuned one) and adds
the latency-to-go and step-latency heads, which start fresh. Plans are bushy trees forced on QLever
with nested groups (runtime_tree.bushy_sparql). Each ROUND:
  1. take the next `queries_per_round` training queries;
  2. plan K plans per query: the greedy plan plus K - 1 one-step deviations at its closest calls
     (bushy_train.deviation_plans), under `decision_objective` (latency: step latencies plus
     latency-to-go, from the online heads; cout for the first `cout_warmup_rounds` rounds, while
     those heads are untrained);
  3. execute them, each with a timeout fixed before the round:
         min(default, max(min, multiplier * fastest known latency of that query));
  4. add what each runtime tree reveals (runtime_tree.parse_bushy_execution) to the query's
     ForestObservations: exact cardinality of every join result and scan, and every merge's time;
  5. train on the most recent `replay_queries` executed queries. States: the forests along every
     executed plan; rows: their candidate merges whose result was observed, with targets card
     (observed), C_out cost-to-go and latency-to-go (stitched upper bounds: one-sided losses), step
     latency (observed), plus the listwise C_out loss over a state's observed children.
Every `validation.every_rounds` (and at the start), each validation query is executed under every
planner, back to back in rotating order:
    ours (latency) beam-k   the value net with the latency heads, width k in validation.widths
                            (width 1 = greedy); from the end of the C_out warm-up on
    ours (C_out) greedy     the value net's C_out heads (the offline planner)
    GNCE GOO / beam-k       bottom-up planning on the GNCE cardinality model alone
                            (bushy.estimator_beam_plan: width 1 = GOO, wider = beam over estimated
                            C_out); planned once
    QLever                  QLever's own optimizer
reporting latency (mean, median, p90, p99, timeouts), the geometric-mean speedup vs QLever and vs
GNCE at the same width, executed C_out, and planning time; per-query results are saved.

Outputs in online.output_directory/online-bushy-<time>/: config.yaml, rounds.jsonl,
validation.jsonl, validation-<round>.json (per query), model-<round>.pt, state.pkl.
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

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.bushy_model import ForestValueNet
from src.supervised_value_estimation.amortized_dp.bushy_train import (
    ForestTables, _group_log_softmax, beam_plan, deviation_plans, embed_patterns, greedy_plan, load_backbone,
    load_embedder, load_split, row_arrays,
)
from src.supervised_value_estimation.amortized_dp.labels import (
    NEG_INF, family_of, join_position_features, mask_of, model_log_cardinalities, variable_adjacency,
)
from src.supervised_value_estimation.amortized_dp.online.executor import RayPlanExecutor
from src.supervised_value_estimation.amortized_dp.online.observations import ForestObservations
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import bushy_sparql, parse_bushy_execution
from src.supervised_value_estimation.amortized_dp.online.train_online import (
    _endpoints, _geomean_speedup, _summarise_latencies, _time_summary, _timeout, _timing_speedups, _timing_summary,
)

FRESH_HEADS = ("latency_head", "step_head", "latency_statistics", "step_statistics")
N_MAX = 16          # row arrays are built this wide, then cut to the replay window's largest query


def load_model(checkpoint, device):
    """The offline forest value net plus fresh latency-to-go and step-latency heads."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kwargs = {**saved["model_kwargs"], "latency_head": True, "step_latency_head": True}
    model = ForestValueNet(**kwargs)
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    new = [k for k in missing if not k.startswith(FRESH_HEADS)]
    if new or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing {new}, unexpected {unexpected}")
    return model.to(device), kwargs


def neighbour_masks_of(triple_patterns):
    adjacency = variable_adjacency(triple_patterns)
    return [mask_of(adjacency[i]) for i in range(len(triple_patterns))]


def connected(neighbour_masks):
    reach, frontier = 1, 1
    while frontier:
        grown = reach
        for i in range(len(neighbour_masks)):
            if frontier >> i & 1:
                grown |= neighbour_masks[i]
        frontier, reach = grown & ~reach, grown
    return reach == (1 << len(neighbour_masks)) - 1


def query_rows(obs: ForestObservations, n_max: int):
    """Training rows of one query from its observations, as row arrays (bushy_train.row_arrays)
    plus merge sides and the latency targets (log1p ms; *_valid where known)."""
    table = bushy.StitchTable(obs.logcard, obs.logcard)
    nbr = obs.labels.neighbour_masks
    states = []
    for execution in obs.executions:
        forest = bushy.initial_forest(obs.labels.n_tp)
        for a, b in bushy.tree_to_merges(execution["tree"])[:-1]:
            states.append(forest)
            forest = bushy.merge(forest, a, b)
        states.append(forest)
    states = list(dict.fromkeys(f for f in states if len(f) > 1))
    rows, sides, latency, step = [], [], [], []
    for state_id, forest in enumerate(states):
        for a, b in bushy.candidate_merges(forest, nbr):
            if a | b not in obs.logcard:
                continue
            child = bushy.merge(forest, a, b)
            full = len(child) == 1
            g = NEG_INF if full else table.cost_to_go(child)
            if not full and not math.isfinite(g):
                continue
            rows.append((state_id, child, a | b, obs.logcard[a | b], g, full, True))
            side = np.full(n_max, -1, dtype=np.int8)
            side[[p for p in range(n_max) if a >> p & 1]] = 0
            side[[p for p in range(n_max) if b >> p & 1]] = 1
            sides.append(side)
            remaining = 0.0 if full else obs.latency_to_go(child)
            latency.append(math.log1p(remaining) if math.isfinite(remaining) else math.nan)
            observed = obs.step_ms.get((min(a, b), max(a, b)))
            step.append(math.log1p(observed) if observed is not None else math.nan)
    arrays = row_arrays(rows, n_max)
    latency, step = np.array(latency, dtype=np.float32), np.array(step, dtype=np.float32)
    arrays.update(side=np.stack(sides) if sides else np.zeros((0, n_max), dtype=np.int8),
                  latency=np.nan_to_num(latency), latency_valid=~np.isnan(latency),
                  step=np.nan_to_num(step), step_valid=~np.isnan(step))
    return arrays


def online_loss(model, tables, state_ids, settings):
    """card (MSE), C_out cost-to-go and latency-to-go (one-sided: stitched upper bounds), step
    latency (MSE), and the listwise C_out loss over each state's observed children."""
    rows, local = tables.batch_rows(state_ids)
    r = {k: v[rows] for k, v in tables.rows.items()}
    q = r["query"]
    out = model.forward_heads(tables.embeddings[q].float(), tables.pattern_mask[q], tables.adjacency[q],
                              r["component"].long(), r["newest"],
                              tables.edge_features[q].float() if model.edge_feature_dim else None,
                              merge_side=r["side"].long(), heads=("card", "cost_to_go", "latency", "step"))
    under = settings["under_weight"] ** 0.5

    def one_sided(error):                          # predicting above an upper bound is always wrong
        return torch.mean((error * torch.where(error < 0, under, 1.0)) ** 2)
    card_target, g_target = model.standardise(r["card"], r["g"])
    loss_card = torch.mean((out["card"] - card_target) ** 2)
    not_full = ~r["full"]
    zero = out["card"].sum() * 0
    loss_g = one_sided(out["cost_to_go"][not_full] - g_target[not_full]) if not_full.any() else zero
    card, g = model.unstandardise(out["card"], out["cost_to_go"])
    logq = torch.logaddexp(card, torch.where(r["full"], torch.full_like(g, -math.inf), g))
    true_logq = torch.logaddexp(r["card"], torch.where(r["full"], torch.full_like(r["g"], -math.inf), r["g"]))
    n = len(state_ids)
    predicted = _group_log_softmax(-logq / settings["temperature"], local, n)
    target = torch.exp(_group_log_softmax(-true_logq / settings["temperature"], local, n))
    loss_rank = -torch.zeros(n, device=card.device).index_add(0, local, target * predicted).mean()
    valid = r["latency_valid"]
    loss_latency = one_sided(out["latency"][valid] - model.standardise_latency(r["latency"][valid])) if valid.any() else zero
    valid = r["step_valid"]
    loss_step = torch.mean((out["step"][valid] - model.standardise_step(r["step"][valid])) ** 2) if valid.any() else zero
    total = (loss_card + loss_g + settings["rank_weight"] * loss_rank
             + settings["latency_weight"] * loss_latency + settings["step_weight"] * loss_step)
    if model.moe_config:
        total = total + settings["balance_weight"] * model.load_balancing_loss()
    return total, {"loss": total.item(), "card": loss_card.item(), "g": loss_g.item(), "rank": loss_rank.item(),
                   "latency": loss_latency.item(), "step": loss_step.item()}


def executed_cout(tree):
    """C_out of an executed plan from its runtime tree: the sum of its join results' rows."""
    total, stack = 0, [tree]
    while stack:
        node = stack.pop()
        if not node:
            continue
        stack.extend(node.get("children", []) or [])
        if str(node.get("description", "")).startswith(("Join", "MultiColumnJoin")):
            total += int(node.get("result_rows") or 0)
    return total


class QueryInputs:
    """Per query: patterns, join graph, and the value net's input tensors."""

    def __init__(self, data, embeddings):
        self.query = " ".join(data.query.split())
        self.raw_query = data.query
        self.triple_patterns = list(data.triple_patterns)
        self.n = len(self.triple_patterns)
        self.family = family_of(data.type if isinstance(data.type, str) else data.type[0])
        self.neighbour_masks = neighbour_masks_of(self.triple_patterns)
        self.embeddings = embeddings
        self.adjacency = torch.from_numpy((np.array(self.neighbour_masks)[:, None] >> np.arange(self.n)) & 1 > 0)
        self.edge_features = torch.from_numpy(join_position_features(self.triple_patterns))


@hydra.main(version_base=None, config_path="../../../../experiments/experiment_configs/amortized_dp",
            config_name="bushy_online_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    online = cfg.online
    rng = np.random.default_rng(int(online.seed))
    torch.manual_seed(int(online.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_directory = Path(online.output_directory) / f"online-bushy-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}", flush=True)

    model, model_kwargs = load_model(online.init_checkpoint, device)
    print(f"Initialised from {online.init_checkpoint} (+ fresh latency-to-go and step heads); "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    embedder, _ = load_embedder(cfg.embedder.policy_config, cfg.embedder.run, cfg.embedder.select, device)

    # --- queries ---------------------------------------------------------------------------
    train_dataset = load_split(cfg.data.train_dir)
    with open(Path(cfg.data.train_embeddings).parent / "selection.pkl", "rb") as f:
        selection = pickle.load(f)
    cached = dict(zip(selection["indices"], torch.load(cfg.data.train_embeddings)))
    train_pool = [i for i in range(len(train_dataset)) if connected(neighbour_masks_of(train_dataset[i].triple_patterns))]
    rng.shuffle(train_pool)

    def train_inputs(i):
        data = train_dataset[i]
        embeddings = cached[i] if i in cached else embed_patterns(embedder, data, device).half()
        return QueryInputs(data, embeddings)

    val_dataset = load_split(cfg.data.val_dir)
    gnce_dataset = load_split(cfg.gnce.dataset)
    gnce_index = {" ".join(gnce_dataset[i].query.split()): i for i in range(len(gnce_dataset))}
    candidates = [i for i in range(len(val_dataset))
                  if " ".join(val_dataset[i].query.split()) in gnce_index
                  and list(gnce_dataset[gnce_index[" ".join(val_dataset[i].query.split())]].triple_patterns)
                  == list(val_dataset[i].triple_patterns)
                  and connected(neighbour_masks_of(val_dataset[i].triple_patterns))]
    picked = sorted(rng.choice(len(candidates), min(int(online.validation.n_queries), len(candidates)), replace=False))
    validation = [QueryInputs(val_dataset[candidates[k]], embed_patterns(embedder, val_dataset[candidates[k]], device).half())
                  for k in picked]
    print(f"Train pool: {len(train_pool):,} connected queries | validation: {len(validation)} "
          f"(sizes {np.bincount([v.n for v in validation]).tolist()})", flush=True)

    # --- fixed baselines: GNCE planners --------------------------------------------------------
    widths = [int(w) for w in online.validation.widths]
    gnce, _ = load_backbone(cfg.gnce.config, cfg.gnce.run, cfg.gnce.select, device)
    gnce_plans, gnce_ms = {w: {} for w in widths}, {w: [] for w in widths}
    for v in validation:
        data = gnce_dataset[gnce_index[v.query]]
        for w in widths:
            start = time.perf_counter()
            tree, _ = bushy.estimator_beam_plan(v.n, v.neighbour_masks,
                                                lambda masks: model_log_cardinalities(gnce, data, masks, device), w)
            gnce_ms[w].append(1000 * (time.perf_counter() - start))
            gnce_plans[w][v.query] = tree
    del gnce
    print("GNCE planning median ms: " + ", ".join(f"width {w}: {np.median(gnce_ms[w]):.1f}" for w in widths), flush=True)

    endpoints = _endpoints(online.execution)
    print(f"{len(endpoints)} QLever endpoints: {endpoints}", flush=True)
    executor = RayPlanExecutor(endpoints, n_actors=int(online.execution.n_actors),
                               in_flight_per_endpoint=int(online.execution.in_flight_per_endpoint))

    observations: dict[str, ForestObservations] = {}
    inputs: dict[str, QueryInputs] = {}
    row_cache: dict[str, tuple[int, dict]] = {}            # query -> (observations version, arrays)
    order_seen: dict[str, None] = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(online.lr), weight_decay=float(online.weight_decay))
    settings = {"under_weight": float(online.upper_bound_under_weight), "temperature": float(online.rank_temperature),
                "rank_weight": float(online.rank_weight), "latency_weight": float(online.latency_weight),
                "step_weight": float(online.step_weight),
                "balance_weight": float((model.moe_config or {}).get("balance_weight", 0.0))}
    latency_ready = False

    def request(v, tree, timeout_s):
        if tree is None:                                   # QLever's own optimizer
            return {"query": v.raw_query, "triple_patterns": v.triple_patterns, "plan": None, "timeout_s": timeout_s}
        return {"query": bushy_sparql(tree, v.triple_patterns), "triple_patterns": v.triple_patterns, "plan": None,
                "timeout_s": timeout_s}

    def plan(v, objective, width):
        args = (model, v.embeddings, v.adjacency, v.edge_features, v.neighbour_masks, device)
        if width == 1:
            return greedy_plan(*args, objective=objective)
        return beam_plan(*args, width, objective=objective)

    def validate(round_index):
        model.eval()
        planners, planning = {}, {}
        if latency_ready and round_index >= int(online.cout_warmup_rounds):     # the latency heads have trained
            for w in widths:
                planners[f"ours (latency) {'greedy' if w == 1 else f'beam-{w}'}"] = ("latency", w)
        planners["ours (C_out) greedy"] = ("cout", 1)
        plans = {name: {} for name in planners}
        for name, (objective, w) in planners.items():
            planning[name] = []
            for v in validation:
                start = time.perf_counter()
                plans[name][v.query] = plan(v, objective, w)
                planning[name].append(1000 * (time.perf_counter() - start))
        for w in widths:
            name = f"GNCE {'GOO' if w == 1 else f'beam-{w}'}"
            plans[name], planning[name] = gnce_plans[w], gnce_ms[w]
        plans["QLever"] = {v.query: None for v in validation}
        names = list(plans)
        requests, tags = [], []
        for k, v in enumerate(validation):
            rotation = k % len(names)
            for name in names[rotation:] + names[:rotation]:
                requests.append(request(v, plans[name][v.query], float(online.validation.timeout_s)))
                tags.append((name, v.query))
        results = executor.execute(requests)
        per_query = {v.query: {"n": v.n, "family": v.family} for v in validation}
        for (name, query), result in zip(tags, results):
            per_query[query][name] = {"latency_s": result["latency_s"], "censored": result["censored"],
                                      "planning_ms": result.get("planning_ms"), "exec_ms": result.get("exec_ms"),
                                      "cout": executed_cout(result["tree"]) if not result["censored"] else None,
                                      "plan": bushy.to_json(plans[name][query]) if plans[name][query] is not None else None}
        queries = [v.query for v in validation]
        summary = {"round": round_index, "planners": {}}
        results_of = lambda name: [per_query[q][name] for q in queries]
        model_ms_of = lambda name: planning.get(name)       # None for QLever: its planning is inside its latency
        for name in names:
            lat = [per_query[q][name]["latency_s"] for q in queries]
            summary["planners"][name] = {"latency": _summarise_latencies(lat, [per_query[q][name]["censored"] for q in queries]),
                                         "speedup_vs_qlever": _geomean_speedup([per_query[q]["QLever"]["latency_s"] for q in queries], lat),
                                         "timing": _timing_summary(results_of(name), model_ms_of(name)),
                                         "speedup_vs_qlever_split": _timing_speedups(results_of("QLever"), results_of(name),
                                                                                     None, model_ms_of(name))}
            if name in planning:
                summary["planners"][name]["planning"] = _time_summary(planning[name])
            couts = [(per_query[q][name]["cout"], per_query[q]["QLever"]["cout"]) for q in queries]
            couts = [(a, b) for a, b in couts if a is not None and b is not None]
            if couts:
                summary["planners"][name]["cout_geomean_vs_qlever"] = float(np.exp(np.mean([np.log(max(a, 1) / max(b, 1)) for a, b in couts])))
        for w in widths:
            ours, theirs = f"ours (latency) {'greedy' if w == 1 else f'beam-{w}'}", f"GNCE {'GOO' if w == 1 else f'beam-{w}'}"
            if ours in names:
                summary["planners"][ours]["speedup_vs_gnce_same_width"] = _geomean_speedup(
                    [per_query[q][theirs]["latency_s"] for q in queries], [per_query[q][ours]["latency_s"] for q in queries])
                summary["planners"][ours]["speedup_vs_gnce_same_width_split"] = _timing_speedups(
                    results_of(theirs), results_of(ours), model_ms_of(theirs), model_ms_of(ours))
        with open(run_directory / "validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        with open(run_directory / f"validation-{round_index}.json", "w", encoding="utf-8") as f:
            json.dump(per_query, f)
        print(f"[validation round {round_index}] {len(queries)} queries", flush=True)
        for name in names:
            s = summary["planners"][name]
            lat = s["latency"]
            line = (f"  {name:26s} latency mean {lat['mean_s'] * 1000:7.1f} ms, median {lat['median_s'] * 1000:6.1f}, "
                    f"p90 {lat['p90_s'] * 1000:7.1f}, timeouts {lat['timeout_rate']:5.1%} | speedup vs QLever "
                    f"{s['speedup_vs_qlever']:.3f}")
            if "speedup_vs_gnce_same_width" in s:
                line += f", vs GNCE same width {s['speedup_vs_gnce_same_width']:.3f}"
            if "cout_geomean_vs_qlever" in s:
                line += f" | C_out/QLever {s['cout_geomean_vs_qlever']:.3f}"
            if "planning" in s:
                line += f" | planning median {s['planning']['median_ms']:.1f} ms"
            print(line, flush=True)
            t, split = s["timing"], s["speedup_vs_qlever_split"]
            qp = t["qlever_planning"]
            line = (f"  {'':26s} execution only mean {t['execution']['mean_s'] * 1000:7.1f} ms, median "
                    f"{t['execution']['median_s'] * 1000:6.1f}, p90 {t['execution']['p90_s'] * 1000:7.1f} | QLever planning "
                    f"median {qp['median_ms'] if qp['n'] else float('nan'):.1f} ms | end to end mean "
                    f"{t['end_to_end']['mean_s'] * 1000:7.1f} ms | speedup vs QLever: execution {split['execution']:.3f}, "
                    f"end to end {split['end_to_end']:.3f}")
            if "speedup_vs_gnce_same_width_split" in s:
                g = s["speedup_vs_gnce_same_width_split"]
                line += f"; vs GNCE same width: execution {g['execution']:.3f}, end to end {g['end_to_end']:.3f}"
            print(line, flush=True)
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "round": round_index},
                   run_directory / f"model-{round_index}.pt")
        with open(run_directory / "state.pkl", "wb") as f:
            pickle.dump({"observations": observations, "order_seen": order_seen, "round": round_index}, f)
        model.train()

    if online.validation.at_start:
        validate(0)

    cursor = 0
    for round_index in range(1, int(online.rounds) + 1):
        start = time.perf_counter()
        objective = online.decision_objective if latency_ready and round_index > int(online.cout_warmup_rounds) else "cout"
        model.eval()
        requests, owners, is_greedy = [], [], []
        for k in range(int(online.queries_per_round)):
            v = train_inputs(train_pool[(cursor + k) % len(train_pool)])
            if v.query not in observations:
                observations[v.query] = ForestObservations(v.query, v.family, v.n, v.neighbour_masks, v.triple_patterns)
            inputs[v.query] = v
            order_seen.pop(v.query, None)
            order_seen[v.query] = None
            timeout_s = _timeout(observations[v.query], online.timeouts)
            for tree in deviation_plans(model, v.embeddings, v.adjacency, v.edge_features, v.neighbour_masks, device,
                                        int(online.plans_per_query), objective):
                requests.append(request(v, tree, timeout_s))
                is_greedy.append(not owners or owners[-1][0] != v.query)
                owners.append((v.query, tree))
        cursor += int(online.queries_per_round)
        planning_seconds = time.perf_counter() - start

        results = executor.execute(requests)
        execution_seconds = time.perf_counter() - start - planning_seconds
        aligned = 0
        for (query, tree), req, result in zip(owners, requests, results):
            parsed = parse_bushy_execution(result["tree"], bushy.internal_masks(tree), req["triple_patterns"])
            aligned += parsed["aligned"]
            observations[query].add(tree, parsed, result["latency_s"], result["censored"], result["timeout_s"])

        window = list(order_seen)[-int(online.replay_queries):]
        parts = []
        for q in window:
            version = observations[q].version
            if q not in row_cache or row_cache[q][0] != version:
                row_cache[q] = (version, query_rows(observations[q], N_MAX))
            parts.append(row_cache[q][1])
        arrays = {}
        for position, part in enumerate(parts):
            for name, value in part.items():
                arrays.setdefault(name, []).append(value)
            arrays.setdefault("query", []).append(np.full(len(part["state"]), position, dtype=np.int64))
        arrays = {name: np.concatenate(value) for name, value in arrays.items()}
        _, arrays["state"] = np.unique(arrays["query"] * 100_000 + arrays["state"], return_inverse=True)
        window_inputs = [inputs[q] for q in window]
        width = max(v.n for v in window_inputs)
        arrays["component"], arrays["side"] = arrays["component"][:, :width], arrays["side"][:, :width]
        tables = ForestTables([observations[q].labels for q in window], [v.embeddings for v in window_inputs],
                              [v.triple_patterns for v in window_inputs], arrays, device)
        if not latency_ready:
            latency = tables.rows["latency"][tables.rows["latency_valid"]]
            step = tables.rows["step"][tables.rows["step_valid"]]
            if len(latency) > 1 and len(step) > 1:
                model.set_latency_statistics(float(latency.mean()), float(latency.std()))
                model.set_step_statistics(float(step.mean()), float(step.std()))
                latency_ready = True
        model.train()
        steps = max(1, math.ceil(float(online.gradient_passes_per_round) * tables.n_states / int(online.states_per_batch)))
        running = []
        for _ in range(steps):
            ids = torch.randint(0, tables.n_states, (int(online.states_per_batch),), device=device)
            loss, part = online_loss(model, tables, ids, settings)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(online.grad_clip))
            optimizer.step()
            running.append(part)
        train_seconds = time.perf_counter() - start - planning_seconds - execution_seconds
        greedy = [r for r, first in zip(results, is_greedy) if first]
        summary = {"round": round_index, "objective": objective, "queries_seen": len(order_seen), "plans_executed": len(requests),
                   "timeout_rate": float(np.mean([r["censored"] for r in results])),
                   "failed_rate": float(np.mean([bool(r.get("failed")) for r in results])),
                   "aligned_rate": aligned / len(results),
                   "greedy_mean_latency_s": float(np.mean([r["latency_s"] for r in greedy])),
                   "greedy_timeout_rate": float(np.mean([r["censored"] for r in greedy])),
                   "all_plans_mean_latency_s": float(np.mean([r["latency_s"] for r in results])),
                   **{key: float(np.mean([p[key] for p in running])) for key in running[0]},
                   "planning_seconds": planning_seconds, "execution_seconds": execution_seconds,
                   "train_seconds": train_seconds, "window_states": tables.n_states,
                   "window_rows": int(len(tables.rows["state"])), "gradient_steps": steps}
        with open(run_directory / "rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"Round {round_index} ({objective}): {len(requests)} plans ({summary['timeout_rate']:.1%} timeouts, "
              f"{summary['aligned_rate']:.1%} aligned), loss {summary['loss']:.3f} (card {summary['card']:.3f}, "
              f"g {summary['g']:.3f}, rank {summary['rank']:.3f}, latency {summary['latency']:.3f}, step {summary['step']:.3f}) "
              f"| plan {planning_seconds:.0f}s, execute {execution_seconds:.0f}s, train {train_seconds:.0f}s "
              f"({steps} steps over {tables.n_states} states)", flush=True)
        if round_index % int(online.validation.every_rounds) == 0 or round_index == int(online.rounds):
            validate(round_index)
    executor.close()


if __name__ == "__main__":
    main()
