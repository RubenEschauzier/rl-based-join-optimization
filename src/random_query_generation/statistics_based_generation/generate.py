"""Generate a join-ordering-hard workload from data statistics.

    python -m src.random_query_generation.statistics_based_generation.generate                      # pilot
    python -m src.random_query_generation.statistics_based_generation.generate generation.mode=generate

Per shape (shapes.py): sizes and choke points (instantiate.py) are drawn evenly, candidates
are embedded in the in-memory graph (graph.py; worker processes share it copy-on-write),
a zero_share of them is perturbed into hard-zero candidates (hard_zero.py), and every
candidate is certified with a few ground-truth COUNTs on QLever (certificate.py). Nothing in
the loop uses an optimizer's estimates or a learned model: selection depends only on the data.

Outputs in output_directory:
    pilot mode     pilot_<shape>.jsonl (every candidate + certificate), pilot_summary.json
    generate mode  statistics_based_<shape>.json in the format of the existing generators
                   ({"x", "y", "query", "triples", "type"} + "generation" metadata), and
                   generation_summary.json
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import queue
import time
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.random_query_generation.statistics_based_generation.certificate import (
    BackgroundCounter, certify_in_memory, finish_certificate,
)
from src.random_query_generation.statistics_based_generation.counting import ExactCounter, PredicateIndex
from src.random_query_generation.statistics_based_generation.graph import load_graph
from src.random_query_generation.statistics_based_generation.hard_zero import HardZeroMaker
from src.random_query_generation.statistics_based_generation.instantiate import (
    ChokePoint, Instantiator, binding_plan, tighten_to_budget,
)
from src.random_query_generation.statistics_based_generation.shapes import build_shape
from src.random_query_generation.statistics_based_generation.statistics import load_statistics

# Set in the parent before the worker pool forks, so workers share them copy-on-write.
_STATE = {}


def _endpoints(settings):
    if settings.get("endpoints"):
        return list(settings.endpoints)
    hosts = [str(h).split("://", 1)[-1].rstrip("/") for h in settings.endpoint_hosts]
    return [f"http://{h}:{p}" for h in hosts
            for p in range(int(settings.base_port), int(settings.base_port) + int(settings.ports_per_host))]


def _choke_points(settings):
    catalogue = {}
    for entry in settings:
        entry = OmegaConf.to_container(entry, resolve=True)
        name, weight = entry.pop("name"), entry.pop("weight", 1.0)
        for key in ("n_constants", "strata_weights"):
            if key in entry:
                entry[key] = tuple(entry[key])
        catalogue[name] = (ChokePoint(name=name, **entry), float(weight))
    return catalogue


def _make_candidate(job):
    """Worker: one candidate (dict) or None. job = (shape, n, choke point, hard_zero, seed)."""
    kind, n, choke_name, hard_zero, seed = job
    graph, statistics = _STATE["graph"], _STATE["statistics"]
    instantiator = _STATE["instantiators"][kind]
    choke_point = _STATE["choke_points"][choke_name][0]
    rng = np.random.default_rng(seed)
    shape = build_shape(kind, n, rng)
    plan = binding_plan(shape, choke_point, rng)
    instance = instantiator.embed(shape, plan, choke_point, rng)
    if instance is None:
        return None
    planned_constants = len(instance.plan.constants)
    if _STATE["max_result"] is not None:
        instance = tighten_to_budget(instance, graph, statistics, _STATE["counter"], _STATE["max_result"], rng,
                                     max_conditioning=_STATE["budget_conditioning"])
        if instance is None:
            return None
    perturbation = None
    if hard_zero:
        instance = _STATE["hard_zero"].perturb(instance, rng)
        if instance is None:
            return None
        perturbation = instance.perturbation
    patterns = instance.triple_patterns(graph)
    strata = sorted(instance.plan.constants.values())
    candidate = {
        "shape": kind, "n_patterns": n, "choke_point": choke_name, "hard_zero": hard_zero,
        "perturbation": perturbation, "triple_patterns": patterns,
        "pattern_sizes": instance.pattern_sizes(graph, statistics),
        "n_constants": len(instance.plan.constants), "constant_strata": strata,
        "added_constants": len(instance.plan.constants) - planned_constants,
        "template": f"{kind}|{n}|{choke_name}|{'zero' if hard_zero else 'nonzero'}|{'-'.join(strata)}",
        "entities": sorted({t.strip("<>") for p in patterns for t in p.split() if t.startswith("<")}),
        "seed": int(seed),
    }
    if not certify_in_memory(candidate, instance.structured_patterns(), _STATE["counter"],
                             _STATE["random_plans"], rng, _STATE["max_empty_pairs"]):
        return None
    return candidate


def _accept(candidate, acceptance, max_empty_pairs=0):
    certificate = candidate["certificate"]
    result = certificate.get("result")
    if result is None or "spread" not in certificate:
        return False, "uncertified"
    if candidate["hard_zero"]:
        empty_pairs = certificate.get("empty_pairs")
        if result != 0 or empty_pairs is None or empty_pairs > max_empty_pairs:
            return False, "not_hard_zero"
    elif result == 0:
        return False, "zero"                    # unplanned zeros: the zero share is controlled
    elif acceptance.max_result is not None and result > acceptance.max_result:
        return False, "result_too_large"
    if acceptance.min_spread is not None and certificate["spread"] < acceptance.min_spread:
        return False, "low_spread"
    if acceptance.max_min_cost is not None and certificate["min_cost"] > acceptance.max_min_cost:
        return False, "expensive_optimum"
    return True, "accepted"


def _canonical(patterns):
    return " . ".join(sorted(patterns))


def _timed_candidate(job):
    start = time.perf_counter()
    return _make_candidate(job), time.perf_counter() - start


def _job_keys(cells):
    """A cell is (size, choke point, hard zero, spread class); the spread class is only known
    after certification, so jobs are drawn per (size, choke point, hard zero), for the sum of
    what its classes need."""
    keys = Counter()
    for (n, choke_name, hard_zero, _), count in cells.items():
        keys[(n, choke_name, hard_zero)] += count
    return keys


def _job_key(candidate):
    return candidate["n_patterns"], candidate["choke_point"], candidate["hard_zero"]


def spread_class(spread, edges):
    """0 .. len(edges): the spread class of a query (e.g. edges [2, 10] -> <2, 2-10, >=10)."""
    return int(np.searchsorted(np.asarray(edges, float), spread, side="right"))


def _summary(candidates):
    def quantiles(values):
        values = np.asarray([float(v) for v in values if v is not None])     # counts can exceed int64
        return (np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if len(values) else None)
    by_group = defaultdict(list)
    for c in candidates:
        by_group[(c["choke_point"], c["hard_zero"])].append(c)
    out = {}
    for (choke, zero), group in sorted(by_group.items()):
        certs = [c["certificate"] for c in group]
        out[f"{choke}{' (hard zero)' if zero else ''}"] = {
            "n": len(group),
            "certified": sum("spread" in c for c in certs),
            "result_zero": sum(c.get("result") == 0 for c in certs),
            "pairs_nonempty": sum(bool(c.get("pairs_nonempty")) for c in certs) if zero else None,
            "one_empty_pair": sum(c.get("empty_pairs") == 1 for c in certs) if zero else None,
            "spread_q10_25_50_75_90": quantiles([c.get("spread") for c in certs]),
            "min_cost_q": quantiles([c.get("min_cost") for c in certs]),
            "result_q": quantiles([c.get("result") for c in certs]),
            "largest_pattern_q": quantiles([max(c_["pattern_sizes"]) for c_ in group]),
            "unknown_plans_mean": float(np.mean([c.get("n_unknown_plans", 0) for c in certs])),
        }
    return out


def _run_shape(kind, gen, endpoints, pool, rng, seed_counter, output_directory):
    sizes = list(gen.sizes)
    catalogue = _STATE["choke_points"]
    pilot = gen.mode == "pilot"
    total = int(gen.pilot.candidates_per_shape) if pilot else int(gen.queries_per_shape)
    n_zero = int(round(total * float(gen.zero_share)))
    weights = np.array([catalogue[name][1] for name in catalogue], dtype=float)
    shares = weights / weights.sum()
    # Non-zero queries are also balanced over spread classes (generation.spread_strata), so the
    # workload is not only queries whose best order is far better than a typical one.
    strata = gen.get("spread_strata")
    stratify = strata is not None and not pilot
    class_shares = list(strata.shares) if stratify else [1.0]
    wanted = Counter()
    for n in sizes:                              # non-zero cells: size x choke point x spread class
        for name, share in zip(catalogue, shares):
            for k, class_share in enumerate(class_shares):
                wanted[(n, name, False, k if stratify else None)] += (
                    (total - n_zero) * share * class_share / len(sizes))
    bases = list(gen.hard_zero.base_choke_points)
    for n in sizes:                              # hard-zero cells: size x base choke point
        for name in bases:
            wanted[(n, name, True, None)] += n_zero / (len(sizes) * len(bases))
    wanted = Counter({cell: int(math.ceil(v)) for cell, v in wanted.items()})

    # Continuous: a fixed number of jobs is in flight in the pool; each finished job is replaced
    # at once by a job for a (size, choke point, hard zero) drawn by what its cells still need,
    # and candidates with subsets to COUNT on QLever are counted in background threads. Workers
    # never wait for QLever or for the slowest job of a batch. Which candidate fills a cell
    # depends on finishing order, so a run is not exactly reproducible from the seed.
    accepted, kept, reasons, seen = defaultdict(list), [], Counter(), set()
    budget = total if pilot else int(float(gen.acceptance.max_candidate_factor) * total)
    in_pool_limit = 2 * int(gen.workers)
    progress_every = int(gen.get("progress_every", 256))
    events = queue.Queue()        # ("made", job key, candidate, seconds) | ("counted", candidate, counts) | ("error", e)
    counter = BackgroundCounter(endpoints, float(gen.certificate.timeout_s), events)
    submitted, outstanding = Counter(), Counter()     # per job key; outstanding = not yet resolved
    status = {"submitted": 0, "finished": 0, "in_pool": 0, "at_qlever": 0, "busy": 0.0}
    start = time.perf_counter()

    def draw_weights():
        if pilot:                 # exactly the wanted number of jobs per key, like a fixed sample
            return {key: count - submitted[key] for key, count in _job_keys(wanted).items() if count > submitted[key]}
        need = Counter({cell: count - len(accepted[cell]) for cell, count in wanted.items()
                        if count > len(accepted[cell])})
        # damped by the jobs already under way for a key, so the last few places of a cell
        # do not pull in a flood of jobs
        return {key: count / (1 + outstanding[key]) for key, count in _job_keys(need).items()}

    def submit():
        weights = draw_weights()
        if not weights or status["submitted"] >= budget:
            return False
        keys = list(weights)
        p = np.array([weights[key] for key in keys], dtype=float)
        key = keys[int(rng.choice(len(keys), p=p / p.sum()))]
        submitted[key] += 1
        outstanding[key] += 1
        status["submitted"] += 1
        status["in_pool"] += 1
        pool.apply_async(_timed_candidate, ((kind, *key, next(seed_counter)),),
                         callback=lambda result, key=key: events.put(("made", key, *result)),
                         error_callback=lambda error: events.put(("error", error)))
        return True

    def resolve(c):
        outstanding[_job_key(c)] -= 1
        spread = c["certificate"].get("spread")
        k = (spread_class(spread, strata.edges) if stratify and not c["hard_zero"] and spread is not None
             else None)
        c["spread_class"] = k
        cell = (c["n_patterns"], c["choke_point"], c["hard_zero"], k)
        if pilot:
            kept.append(c)
            accepted[cell].append(c)
            return
        ok, reason = _accept(c, gen.acceptance, int(gen.hard_zero.max_empty_pairs))
        reasons[reason] += 1
        if ok and len(accepted[cell]) < wanted[cell]:
            accepted[cell].append(c)

    def progress():
        elapsed = time.perf_counter() - start
        done = sum(min(len(accepted[cell]), count) for cell, count in wanted.items())
        with counter.lock:
            counted, seconds, queued = counter.counted, counter.seconds, counter.pending
        print(f"[{kind}] candidates {status['finished']:,} | kept {done:,}/{sum(wanted.values()):,} | "
              f"{elapsed:.0f}s | {status['finished'] / max(elapsed, 1e-9):.1f} candidates/s | "
              f"workers {100 * status['busy'] / max(elapsed * int(gen.workers), 1e-9):.0f}% busy | "
              f"qlever {counted:,} counts ({seconds / max(counted, 1):.1f}s mean), {queued} queued | "
              f"{dict(reasons) if reasons else ''}", flush=True)

    while status["in_pool"] < in_pool_limit and submit():
        pass
    while status["in_pool"] or status["at_qlever"]:
        event = events.get()
        if event[0] == "error":
            counter.close()
            raise event[1]
        if event[0] == "made":
            _, key, c, seconds = event
            status["in_pool"] -= 1
            status["finished"] += 1
            status["busy"] += seconds
            if c is None or _canonical(c["triple_patterns"]) in seen:
                outstanding[key] -= 1
            else:
                seen.add(_canonical(c["triple_patterns"]))
                if c["qlever_masks"] and counter.threads:
                    status["at_qlever"] += 1
                    counter.submit(c)
                else:
                    resolve(finish_certificate(c))
            if status["finished"] % progress_every == 0:
                progress()
        else:
            _, c, counts = event
            status["at_qlever"] -= 1
            resolve(finish_certificate(c, counts))
        while status["in_pool"] < in_pool_limit and submit():
            pass
    counter.close()
    progress()
    queries = [c for cell in accepted.values() for c in cell]
    if pilot:
        with open(output_directory / f"pilot_{kind}.jsonl", "w") as f:
            for c in kept:
                f.write(json.dumps(c) + "\n")
        return _summary(kept), {}
    return _summary(queries), _write_shape(kind, queries, output_directory)


def _write_shape(kind, queries, output_directory):
    from src.utils.generation_utils.generation_utils import filter_isomorphic_queries
    records = []
    for c in queries:
        body = " ".join(f"{p} ." for p in c["triple_patterns"])
        records.append({
            "x": c["entities"],
            "y": int(c["certificate"]["result"]),
            "query": f"SELECT * WHERE {{ {body} }}",
            "triples": [p.split() + ["."] for p in c["triple_patterns"]],
            "type": f"statistics_based_{kind}",
            "generation": {k: c[k] for k in ("shape", "n_patterns", "choke_point", "hard_zero", "perturbation",
                                             "pattern_sizes", "n_constants", "constant_strata", "added_constants",
                                             "spread_class", "template",
                                             "seed", "certificate", "orders")},
        })
    records = filter_isomorphic_queries(records)
    path = output_directory / f"statistics_based_{kind}.json"
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"[{kind}] wrote {len(records):,} queries -> {path}")
    return {"written": len(records)}


@hydra.main(version_base=None,
            config_path="../../../experiments/experiment_configs/query_generation",
            config_name="statistics_based_yago.yaml")
def main(cfg: DictConfig):
    gen = cfg.generation
    output_directory = Path(gen.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / f"config_{gen.mode}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    graph = load_graph(gen.ntriples, gen.cache_directory)
    statistics = load_statistics(graph, gen.cache_directory)
    settings = gen.instantiation
    by_shape = dict(settings.get("max_restarts_by_shape") or {})
    _STATE.update(
        graph=graph, statistics=statistics, choke_points=_choke_points(gen.choke_points),
        instantiators={kind: Instantiator(graph, statistics, int(settings.max_pattern_matches),
                                          int(settings.max_candidates),
                                          int(by_shape.get(kind, settings.max_restarts)))
                       for kind in gen.shapes},
        counter=ExactCounter(graph, PredicateIndex(graph),
                             max_assignments=int(gen.certificate.get("max_conditioning_values", 2000))),
        random_plans=int(gen.certificate.random_plans),
        max_empty_pairs=int(gen.hard_zero.max_empty_pairs),
        max_result=gen.result_budget.max_result,
        budget_conditioning=int(gen.result_budget.get("max_conditioning", 200)),
    )
    _STATE["hard_zero"] = HardZeroMaker(graph, statistics, int(settings.max_pattern_matches),
                                        float(gen.hard_zero.anticorrelation_gamma), counter=_STATE["counter"],
                                        max_empty_pairs=int(gen.hard_zero.max_empty_pairs))
    endpoints = _endpoints(gen.endpoints)
    print(f"{len(endpoints)} counting endpoints: {endpoints[0]} ... {endpoints[-1]}")
    rng = np.random.default_rng(int(gen.seed))
    seed_counter = iter(range(int(gen.seed) * 10_000_000, 10 ** 12))
    summaries = {}
    context = multiprocessing.get_context("fork")
    with context.Pool(int(gen.workers)) as pool:
        for kind in gen.shapes:
            summary, written = _run_shape(kind, gen, endpoints, pool, rng, seed_counter, output_directory)
            summaries[kind] = {"by_choke_point": summary, **written}
            name = "pilot_summary.json" if gen.mode == "pilot" else "generation_summary.json"
            with open(output_directory / name, "w") as f:
                json.dump(summaries, f, indent=2)
    print(f"Done: {output_directory}")


if __name__ == "__main__":
    main()
