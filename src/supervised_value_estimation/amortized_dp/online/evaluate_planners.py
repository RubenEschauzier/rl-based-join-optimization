"""Baseline evaluation of a forest-value run: every planner's plans executed on QLever, next to QLever's own.

    python -m src.supervised_value_estimation.amortized_dp.online.evaluate_planners plan     RUN_DIR [--round R] [--queries N]
    python -m src.supervised_value_estimation.amortized_dp.online.evaluate_planners time-cpu RUN_DIR [--round R] [--cpu-threads 1]
    python -m src.supervised_value_estimation.amortized_dp.online.evaluate_planners execute RUN_DIR [--round R] --endpoints URL,URL,...
                                                                                     (or --hosts H,H --ports-per-host 15 --base-port 7000)
    python -m src.supervised_value_estimation.amortized_dp.online.evaluate_planners report  RUN_DIR [--round R]

RUN_DIR is a train_forest_value run directory (config.yaml, validation-<R>.json, model-<R>.pt, planner-<R>.pt);
the outputs go to RUN_DIR/evaluation-<R>/.

plan     the run's validation queries (validation-<R>.json, all or N spread over it), planned by
             GNCE greedy / beam-8 / DP      the GNCE cardinality model (cfg.gnce)
             est greedy / beam-8 / DP       the run's cardinality model at round R (model-<R>.pt; it trained with the
                                            planner unless forest.stop_gradient), "est DP + bound" with the hard
                                            degree bound and exact single-pattern counts as in the run's validation
             value greedy / beam-8          the forest value planner at round R (planner-<R>.pt)
         greedy = GOO / beam-8 = bushy.estimator_beam_plan for the estimators, fv.beam_plans for the value planner;
         DP = exact bushy DP over every connected subset minimising the estimated C_out (soft_dp at tau 0).
         Every planner builds the cardinality model's input with the same batched builder (soft_dp.subset_batch); the
         step-wise planners (greedy, beam-8) read a RowReadout backbone through a per-query row_readout.QueryReadout.
         Planned on the GPU for speed; its timings are not reported. -> plans.json
time-cpu the same planning on the CPU (--cpu-threads torch threads, default 1 as QLever's single-threaded planner):
         the inference cost, the time inside the network calls only (the cardinality model, its subset-graph
         construction included; for the value planner each beam step's network part as one block: encoding the new
         subsets, the bound, the value net's tokens and the value net), not the search or DP bookkeeping, one query
         at a time after a warm-up; plus the number of subsets estimated and forests valued, and the CPU's own plans
         (the report counts the ones that differ from plans.json). -> inference-cpu.json (plans.json untouched; a
         separate, slower run)
execute  every plan forced on QLever (runtime_tree.bushy_sparql) and every query unforced ("QLever"), `--repeats`
         times, the planners' order rotated per query and repetition, one request at a time per endpoint (a thread
         each), QLever's cache cleared before every execution (cmd=clear-cache with the servers' access token, by
         default from data/qlever/qlever_yago/qlever_yago.env): even with a 0-byte cache QLever keeps EMPTY results
         (size 0), so a repeat, or another planner's plan with the same empty subtree, would get it for free. Per
         execution: QLever's planning ms (meta.time_query_planning), execution ms (the
         root operator), time.total, censoring, cache status, executed C_out (sum of the join results' rows) and the
         result size. Resumable: executions.jsonl is appended per chunk and finished (query, planner, repeat) skipped.
report   per query and planner the median over the uncensored, uncached repeats (if none: the timeout, a lower
         bound, when a repeat timed out, else unmeasured and left out), then per planner quantiles (50 / 90 / 95 / 99 / max) of
             execution ms          and its ratio to QLever's per query (geomean speedup = QLever / planner)
             end to end ms         QLever's time.total + the planner's CPU inference ms (QLever: its time.total, its
                                   own planning inside)
             C_out error           C_out / the cheapest C_out any planner (QLever included) reached on the query
             C_out vs QLever       C_out / QLever's C_out
             inference ms          on the CPU (inference-cpu.json; without it inference and end to end are left
                                   empty), and QLever's own planning ms
         for all queries, empty and non-empty results separately, per query shape, per size and per QLever execution
         time bucket. C_out of a censored plan: from the run's counts.pkl where every join is counted, else unknown
         (left out of the C_out quantiles, counted under "C_out unknown"). Plots: C_out against execution time per
         planner, quantile curves of execution time and C_out error, violins of both per query shape, inference cost
         against query size. -> report.md, report.json, *.png
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.supervised_value_estimation.amortized_dp import bushy, forest_value as fv, history_dp, soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import load_backbone, load_split
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import bushy_sparql
from src.supervised_value_estimation.amortized_dp.online.train_forest_value import encoder
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import executed_cout
from src.supervised_value_estimation.amortized_dp.online.train_plan_loss import (Query, failed_bound, plan_cost,
                                                                                 query_readout, subset_phi)

WIDTH = 8
QUANTILES = (50, 90, 95, 99, 100)
SIZE_BUCKETS = {"<=6": (0, 6), "7-12": (7, 12), "13+": (13, 99)}
NATIVE_BUCKETS = {"<10 ms": (0, 10), "10-100 ms": (10, 100), "100 ms-1 s": (100, 1000), ">1 s": (1000, math.inf)}


def shape_of(data):
    t = data.type if isinstance(data.type, str) else data.type[0]
    return str(t).replace("statistics_based_", "").replace(".json", "")


class InferenceClock:
    """Sums the (GPU-synchronised) time of the wrapped network calls and blocks and counts their rows; a call inside
    a timed block or call is part of it, not counted again."""

    def __init__(self, device):
        self.device, self.ms, self.subsets, self.forests, self.depth = device, 0.0, 0, 0, 0

    def reset(self):
        self.ms, self.subsets, self.forests = 0.0, 0, 0

    @contextlib.contextmanager
    def block(self):
        outer = self.depth == 0
        if outer and self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if outer:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self.ms += 1000 * (time.perf_counter() - start)

    def timed(self, fn, count=None):
        def wrap(*args, **kwargs):
            with self.block():
                out = fn(*args, **kwargs)
            if count is not None:
                count(self, args)
            return out
        return wrap


def count_subsets(clock, args):
    clock.subsets += len(args[0])


def count_forests(clock, args):
    clock.forests += len(args[0])


def plan_all(args, device=None):
    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    out = run / f"evaluation-{args.round}"
    out.mkdir(exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_backbone(cfg.estimator.config, cfg.estimator.run, cfg.estimator.select, device)
    model.load_state_dict(torch.load(run / f"model-{args.round}.pt", map_location=device))
    gnce, _ = load_backbone(cfg.gnce.config, cfg.gnce.run, cfg.gnce.select, device)
    net = fv.ForestPlanner.load(run / f"planner-{args.round}.pt", device)
    model.eval(), gnce.eval(), net.eval()
    degrees = json.load(open(cfg.forest.degrees))
    val, gnce_val = load_split(cfg.data.val_dir), load_split(cfg.gnce.dataset)
    index = {" ".join(val[i].query.split()): i for i in range(len(val))}
    gnce_index = {" ".join(gnce_val[i].query.split()): i for i in range(len(gnce_val))}
    keys = list(json.load(open(run / f"validation-{args.round}.json")))
    if args.queries:                                     # evenly spread: the file is ordered by query shape
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(args.queries, len(keys))).round().astype(int)]
    if device.type == "cpu" and (out / "plans.json").exists():      # time-cpu: the queries plan_all planned
        keys = list(json.load(open(out / "plans.json")))

    clock = InferenceClock(device)
    net.values = clock.timed(net.values, count_forests)
    net.bound.forward = clock.timed(net.bound.forward)

    def estimator_plans(est_model, data, q):
        for name, width in (("greedy", 1), (f"beam-{WIDTH}", WIDTH)):
            clock.reset()
            readout = query_readout(est_model, data, device)     # as the value planner's encoder
            est = clock.timed(lambda masks: subset_phi(est_model, data, torch.tensor(masks), device, readout=readout)[1]
                              .double().cpu().tolist(), count_subsets)
            tree, _ = bushy.estimator_beam_plan(q.n, q.neighbour_masks, est, width)
            yield name, tree
        clock.reset()
        s = soft_dp.SubsetStructure(q.neighbour_masks, device)
        logcard = clock.timed(lambda m: soft_dp.model_subset_logcards(est_model, data, m, device), count_subsets)(s.masks)
        yield "DP", soft_dp.hard_plan(s, soft_dp.soft_dp(s, logcard, 0.0)[1])
        if est_model is model:                           # as the run's "est DP + bound" reference
            clock.reset()
            logcard = clock.timed(lambda m: soft_dp.model_subset_logcards(est_model, data, m, device), count_subsets)(s.masks)
            exact = history_dp.single_pattern_logcounts(data, s.masks)
            bounded = history_dp.degree_bound(q.patterns, s, torch.where(s.singleton, exact, logcard).clamp(min=0), degrees)
            yield "DP + bound", soft_dp.hard_plan(s, soft_dp.soft_dp(s, bounded, 0.0)[1])

    def value_plans(q):
        for name, width in (("greedy", 1), (f"beam-{WIDTH}", WIDTH)):
            clock.reset()
            ctx = fv.QueryContext(q.n, q.neighbour_masks, q.patterns, degrees, clock.timed(encoder(model, q, device), count_subsets))
            ctx.step_timer = clock.block                 # each beam step's network part as one call
            yield name, fv.beam_plans(net, ctx, [width])[0][0][0]

    plans = {}
    with torch.no_grad():
        for k, key in enumerate(keys[:3] + keys):            # the first three twice: warm-up, overwritten
            q = Query(val[index[key]])
            record = {"n": q.n, "shape": shape_of(q.data), "patterns": q.patterns, "query": q.data.query, "planners": {}}
            for prefix, generator in (("GNCE", estimator_plans(gnce, gnce_val[gnce_index[key]], q)),
                                      ("est", estimator_plans(model, q.data, q)), ("value", value_plans(q))):
                for name, tree in generator:
                    record["planners"][f"{prefix} {name}"] = {"plan": bushy.to_json(tree), "inference_ms": round(clock.ms, 3),
                                                              "subsets": clock.subsets, "forests": clock.forests}
            plans[key] = record
            if k % 100 == 0:
                print(f"planned {max(k - 2, 0)} / {len(keys)}", flush=True)
    if device.type == "cpu":
        json.dump({key: rec["planners"] for key, rec in plans.items()}, open(out / "inference-cpu.json", "w"))
        print(f"{len(plans)} queries, CPU inference ({torch.get_num_threads()} threads) -> {out / 'inference-cpu.json'}")
        return
    json.dump(plans, open(out / "plans.json", "w"))
    print(f"{len(plans)} queries x {len(next(iter(plans.values()))['planners'])} planners -> {out / 'plans.json'}")


def time_cpu(args):
    torch.set_num_threads(args.cpu_threads)
    plan_all(args, torch.device("cpu"))


def endpoints_of(args):
    if args.endpoints:
        return args.endpoints.split(",")
    return [f"{h if h.startswith('http') else 'http://' + h}:{args.base_port + p}"
            for h in args.hosts.split(",") for p in range(args.ports_per_host)]


def access_token(args):
    if args.access_token:
        return args.access_token
    for line in open(args.env_file):
        if line.startswith("QLEVER_ACCESS_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"no QLEVER_ACCESS_TOKEN in {args.env_file}: pass --access-token")


def run_on_endpoints(jobs, endpoints, token, timeout_s):
    """[(tag, sparql)] -> [(tag, executor._interpret result)], one thread per endpoint, the cache cleared before each."""
    import queue
    import threading

    import requests

    from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
    from src.supervised_value_estimation.amortized_dp.online.executor import _interpret

    work, results, lock = queue.Queue(), [], threading.Lock()
    for job in jobs:
        work.put(job)
    sent = QLeverOptimizerClient.format_latency(timeout_s)
    effective = QLeverOptimizerClient.decode_to_seconds(sent)
    headers = {"Accept": "application/qlever-results+json", "Content-Type": "application/sparql-query"}

    def worker(endpoint):
        session = requests.Session()
        while True:
            try:
                tag, text = work.get_nowait()
            except queue.Empty:
                return
            try:
                session.get(endpoint, params={"cmd": "clear-cache", "access-token": token}, timeout=30).raise_for_status()
                r = session.post(endpoint, params={"timeout": sent, "send": 0}, headers=headers, data=text, timeout=effective + 30)
                if r.status_code == 200:
                    body = r.json()
                    raw = {"success": True, "runtime_info": body.get("runtimeInformation", {}),
                           "time_total": body.get("time", {}).get("total", "0ms")}
                else:
                    raw = {"success": False, "error": r.text}
            except Exception as error:                  # a failed request is censored, the run goes on
                raw = {"success": False, "error": str(error)}
            res = _interpret(raw, effective)
            with lock:
                results.append((tag, res))

    threads = [threading.Thread(target=worker, args=(e,), daemon=True) for e in endpoints]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def execute_all(args):
    out = Path(args.run) / f"evaluation-{args.round}"
    plans = json.load(open(out / "plans.json"))
    log = out / "executions.jsonl"
    done = set()
    if log.exists():
        done = {(r["key"], r["planner"], r["repeat"]) for r in map(json.loads, open(log))}
    names = list(next(iter(plans.values()))["planners"]) + ["QLever"]
    jobs = []
    for k, (key, rec) in enumerate(plans.items()):
        for rep in range(args.repeats):
            rot = (k + rep) % len(names)
            for name in names[rot:] + names[:rot]:
                if (key, name, rep) in done:
                    continue
                text = rec["query"] if name == "QLever" else bushy_sparql(bushy.from_json(rec["planners"][name]["plan"]), rec["patterns"])
                jobs.append(((key, name, rep), text))
    endpoints, token = endpoints_of(args), access_token(args)
    print(f"{len(jobs):,} executions to go ({len(done):,} done) on {len(endpoints)} endpoints", flush=True)
    for s in range(0, len(jobs), args.chunk):
        results = run_on_endpoints(jobs[s:s + args.chunk], endpoints, token, args.timeout)
        with open(log, "a") as f:
            for (key, name, rep), res in results:
                tree = res.get("tree") or {}
                f.write(json.dumps({"key": key, "planner": name, "repeat": rep, "latency_ms": 1000 * res["latency_s"],
                                    "planning_ms": res.get("planning_ms"), "exec_ms": res.get("exec_ms"),
                                    "censored": res["censored"], "cached": _any_cached(tree),
                                    "cout": None if res["censored"] else executed_cout(tree),
                                    "rows": None if res["censored"] else tree.get("result_rows"),
                                    "error": res.get("error")}) + "\n")
        print(f"executed {min(s + args.chunk, len(jobs)):,} / {len(jobs):,}", flush=True)


def _any_cached(tree):
    stack = [tree]
    while stack:
        node = stack.pop()
        if not node:
            continue
        if str(node.get("cache_status", "")).lower().startswith("cached"):
            return True
        stack.extend(node.get("children", []) or [])
    return False


# --- report --------------------------------------------------------------------------------------

def per_query(plans, executions, counts, timeout_ms, cpu):
    """{key: {planner: {exec, e2e, qlever_planning, inference, cout, censored}}} and {key: empty result?}; inference
    from `cpu` (inference-cpu.json, {} without it)."""
    by = {}
    for r in executions:
        by.setdefault((r["key"], r["planner"]), []).append(r)
    largest = max([c for t in counts.values() for c in t.values() if c is not None] or [1])
    table, empty = {}, {}
    for key, rec in plans.items():
        row = {}
        for name in list(rec["planners"]) + ["QLever"]:
            runs = by.get((key, name), [])
            ok = [r for r in runs if not r["censored"] and not r["cached"] and r["exec_ms"] is not None]
            inference = cpu.get(key, {}).get(name, {}).get("inference_ms") if name != "QLever" else 0.0
            if ok:
                exec_ms = statistics.median(r["exec_ms"] for r in ok)
                total_ms = statistics.median(r["latency_ms"] for r in ok)
                planning = statistics.median(r["planning_ms"] for r in ok if r["planning_ms"] is not None) if any(
                    r["planning_ms"] is not None for r in ok) else math.nan
                cout = next(r["cout"] for r in ok)
                if ok[0]["rows"] is not None:
                    empty[key] = ok[0]["rows"] == 0
            else:                                        # a timeout (a lower bound), or unmeasured (only cached repeats)
                timed_out = any(r["censored"] for r in runs)
                exec_ms = total_ms = timeout_ms if timed_out else None
                planning = math.nan
                cout = None
                if name != "QLever" and key in counts:
                    cout, _ = plan_cost(bushy.from_json(rec["planners"][name]["plan"]), counts[key], failed_bound(counts[key], largest))
            e2e = None if total_ms is None or inference is None else total_ms + inference
            row[name] = {"exec": exec_ms, "e2e": e2e, "qlever_planning": planning,
                         "inference": inference, "cout": cout, "censored": not ok and exec_ms is not None,
                         "unmeasured": exec_ms is None, "subsets": rec["planners"].get(name, {}).get("subsets"),
                         "forests": rec["planners"].get(name, {}).get("forests")}
        table[key] = row
        if key not in empty and key in counts and counts[key].get((1 << rec["n"]) - 1) is not None:
            empty[key] = counts[key][(1 << rec["n"]) - 1] == 0
    return table, empty


def qs(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return {f"p{q}" if q < 100 else "max": (float(np.percentile(v, q)) if len(v) else None) for q in QUANTILES}


def geomean(values):
    v = np.asarray([x for x in values if x is not None and x > 0], dtype=float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) else None


def summarise(table, keys, names):
    out = {}
    for name in names:
        rows = [table[k][name] for k in keys]
        qrows = [table[k]["QLever"] for k in keys]
        best = [min(max(r["cout"], 1) for r in table[k].values() if r["cout"] is not None)
                if any(r["cout"] is not None for r in table[k].values()) else None for k in keys]
        err = [max(r["cout"], 1) / b if r["cout"] is not None and b else None for r, b in zip(rows, best)]
        vs_q = [max(r["cout"], 1) / max(q["cout"], 1) if r["cout"] is not None and q["cout"] is not None else None
                for r, q in zip(rows, qrows)]
        out[name] = {
            "queries": len(rows), "timeouts": sum(r["censored"] for r in rows), "unmeasured": sum(r["unmeasured"] for r in rows),
            "exec_ms": qs(r["exec"] for r in rows), "e2e_ms": qs(r["e2e"] for r in rows),
            "exec_speedup_vs_qlever": geomean(max(q["exec"], 1) / max(r["exec"], 1) for r, q in zip(rows, qrows)
                                              if r["exec"] is not None and q["exec"] is not None),
            "e2e_speedup_vs_qlever": geomean(max(q["e2e"], 1) / max(r["e2e"], 1) for r, q in zip(rows, qrows)
                                             if r["e2e"] is not None and q["e2e"] is not None),
            "cout_error": qs(err), "cout_error_geomean": geomean(err), "cout_unknown": sum(e is None for e in err),
            "cout_vs_qlever_geomean": geomean(vs_q), "cout_vs_qlever": qs(vs_q),
            "inference_ms": qs(r["inference"] for r in rows) if name != "QLever" else None,
            "qlever_planning_ms": qs(r["qlever_planning"] for r in rows),
            "subsets_mean": float(np.mean([r["subsets"] for r in rows])) if name != "QLever" else None,
            "forests_mean": float(np.mean([r["forests"] for r in rows])) if name != "QLever" else None}
    return out


def splits(plans, table, empty):
    keys = list(plans)
    out = {"all": keys, "non-empty result": [k for k in keys if empty.get(k) is False],
           "empty result": [k for k in keys if empty.get(k) is True]}
    for shape in sorted({plans[k]["shape"] for k in keys}):
        out[f"shape {shape}"] = [k for k in keys if plans[k]["shape"] == shape]
    for b, (lo, hi) in SIZE_BUCKETS.items():
        out[f"size {b}"] = [k for k in keys if lo <= plans[k]["n"] <= hi]
    for b, (lo, hi) in NATIVE_BUCKETS.items():
        out[f"QLever execution {b}"] = [k for k in keys if table[k]["QLever"]["exec"] is not None
                                        and lo <= table[k]["QLever"]["exec"] < hi]
    return {k: v for k, v in out.items() if v}


def fmt(x, spec=".2f"):
    return "-" if x is None else format(x, spec)


def markdown(summary):
    lines = []
    for split, (n, s) in summary.items():
        lines.append(f"\n## {split} ({n} queries)\n")
        lines.append("| planner | exec p50 | p90 | p95 | p99 | max | timeouts | exec speedup vs QLever | e2e p50 | e2e p99 | "
                     "e2e speedup | C_out err p50 | p90 | p95 | p99 | max | C_out vs QLever (geo) | CPU inference p50 / p99 ms | "
                     "QLever planning p50 ms |")
        lines.append("|" + "---|" * 20)
        for name, r in s.items():
            e, t, c = r["exec_ms"], r["e2e_ms"], r["cout_error"]
            inf = r["inference_ms"]
            inf = "-" if not inf or inf["p50"] is None else f"{fmt(inf['p50'], '.1f')} / {fmt(inf['p99'], '.1f')}"
            timeouts = f"{r['timeouts']}" + (f" (+{r['unmeasured']} unmeasured)" if r["unmeasured"] else "")
            lines.append(f"| {name} | {fmt(e['p50'], '.1f')} | {fmt(e['p90'], '.1f')} | {fmt(e['p95'], '.1f')} | {fmt(e['p99'], '.1f')} | "
                         f"{fmt(e['max'], '.0f')} | {timeouts} | {fmt(r['exec_speedup_vs_qlever'], '.3f')} | {fmt(t['p50'], '.1f')} | "
                         f"{fmt(t['p99'], '.1f')} | {fmt(r['e2e_speedup_vs_qlever'], '.3f')} | {fmt(c['p50'])} | {fmt(c['p90'])} | "
                         f"{fmt(c['p95'])} | {fmt(c['p99'])} | {fmt(c['max'], '.1f')} | {fmt(r['cout_vs_qlever_geomean'], '.3f')} | {inf} | "
                         f"{fmt(r['qlever_planning_ms']['p50'], '.1f')} |")
    return "\n".join(lines)


def report(args):
    from src.supervised_value_estimation.amortized_dp.online import evaluate_plots

    run = Path(args.run)
    out = run / f"evaluation-{args.round}"
    plans = json.load(open(out / "plans.json"))
    executions = [json.loads(line) for line in open(out / "executions.jsonl")]
    counts = pickle.load(open(run / "counts.pkl", "rb")) if (run / "counts.pkl").exists() else {}
    cpu = json.load(open(out / "inference-cpu.json")) if (out / "inference-cpu.json").exists() else {}
    if not cpu:
        print("no inference-cpu.json (run time-cpu): inference and end-to-end times are left empty")
    table, empty = per_query(plans, executions, counts, 1000 * args.timeout, cpu)
    names = list(next(iter(plans.values()))["planners"]) + ["QLever"]
    summary = {split: (len(keys), summarise(table, keys, names)) for split, keys in splits(plans, table, empty).items()}
    differ = {name: sum(cpu[k][name]["plan"] != rec["planners"][name]["plan"] for k, rec in plans.items()
                        if "plan" in cpu.get(k, {}).get(name, {})) for name in names[:-1]}
    json.dump({"per_query": table, "empty": empty, "summary": summary, "cpu_plans_differ": differ},
              open(out / "report.json", "w"))
    cpu_note = ("plans that differ from the executed GPU plans: " + ", ".join(f"{n} {c}" for n, c in differ.items())
                if any("plan" in r for v in cpu.values() for r in v.values()) else "no CPU plans to compare")
    header = (f"# Planner evaluation: {run.name}, round {args.round}\n\nExecution ms: the root operator's time (median of the "
              f"uncensored, uncached repeats; a timeout counts {args.timeout:.0f} s). End to end: QLever's time.total + the "
              f"planner's inference ms on the CPU ({cpu_note}). C_out error: C_out / the cheapest C_out of any planner on the query. Speedups: "
              f"geomean of QLever / planner (1 ms floor).\n")
    open(out / "report.md", "w").write(header + markdown(summary))
    evaluate_plots.plot_all(plans, table, names, out)
    print(header + markdown({k: v for k, v in summary.items() if k in ("all", "non-empty result", "empty result")}))
    print(f"\n-> {out / 'report.md'}, report.json, plots in {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("plan", "time-cpu", "execute", "report"))
    p.add_argument("run")
    p.add_argument("--round", type=int, default=40)
    p.add_argument("--queries", type=int, default=0, help="plan N validation queries spread over the file (0: all)")
    p.add_argument("--endpoints", default="", help="comma-separated QLever URLs")
    p.add_argument("--hosts", default="", help="comma-separated hosts, with --ports-per-host and --base-port")
    p.add_argument("--ports-per-host", type=int, default=15)
    p.add_argument("--base-port", type=int, default=7000)
    p.add_argument("--access-token", default="", help="QLever access token (for clear-cache); default from --env-file")
    p.add_argument("--env-file", default="data/qlever/qlever_yago/qlever_yago.env")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per execution")
    p.add_argument("--chunk", type=int, default=2000, help="executions per checkpoint")
    p.add_argument("--cpu-threads", type=int, default=1, help="time-cpu: torch threads")
    args = p.parse_args()
    {"plan": plan_all, "time-cpu": time_cpu, "execute": execute_all, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
