"""Execute plans on QLever (through Ray) or in a simulator, returning raw runtime trees.

Both backends take a list of requests
    {"query": str, "triple_patterns": [str], "plan": [int] | None, "timeout_s": float}
(plan None = let QLever's own optimizer choose) and return, per request,
    {"tree": dict, "latency_s": float, "censored": bool, "timeout_s": float, "error": str | None}
Parsing into observations happens in the trainer (runtime_tree.parse_execution), not in the
actors, so the parsing rules live in one tested place.

Differences from RayExecutionStrategy in online_supervised_value_estimation.py:
  * the timeout of every request is fixed by the caller before the batch and returned with
    the result, so a censored latency is known to be ">= timeout_s". (There the timeout of a
    query tightens while its own plans are still in flight and is never recorded.)
  * at most `in_flight_per_endpoint` requests per QLever endpoint, default 1: each endpoint
    is a single-core server, and concurrent queries on it would inflate each other's latency.
  * an actor or task failure is logged and that request reported as failed; the run goes on.
    (There it calls sys.exit(1).)

SimulatedExecutor builds trees in the same format as the real QLever output kept in
utils/visualize_qlever_runtime_output.py, from oracle cardinalities and a simple latency
model, so the whole online loop can be run and tested without a QLever server.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys

import numpy as np
from tqdm import tqdm

from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.supervised_value_estimation.amortized_dp.labels import mask_of, optimal_left_deep_plan


def _interpret(raw, timeout_s):
    """Raw QLeverOptimizerClient result (parse_local=False) -> tree, latency, censoring."""
    if raw.get("success"):
        tree = (raw.get("runtime_info") or {}).get("query_execution_tree", {}) or {}
        latency = QLeverOptimizerClient.decode_to_seconds(raw.get("time_total", "0ms"))
        return {"tree": tree, "latency_s": latency, "censored": False, "timeout_s": timeout_s, "error": None}
    error = str(raw.get("error", ""))
    tree, latency = {}, timeout_s
    try:
        parsed = json.loads(error)
        tree = parsed.get("runtimeInformation", {}) or {}
        latency = max(float(parsed.get("time", {}).get("total", timeout_s * 1000)) / 1000, timeout_s)
    except (ValueError, TypeError, AttributeError):
        pass
    # Every failure is treated as censored at its timeout: the plan took at least that long
    # or could not run, which for ranking plans means the same thing.
    return {"tree": tree, "latency_s": latency, "censored": True, "timeout_s": timeout_s, "error": error[:300]}


class RayPlanExecutor:
    def __init__(self, endpoints, n_actors=4, in_flight_per_endpoint=1, ray_address=None,
                 log_path=os.path.join("logs", "amortized_dp_online_execution.log")):
        import ray
        from src.query_environments.qlever.qlever_multi_execute_ray import MultiEndpointWorker

        os.makedirs("logs", exist_ok=True)       # QLeverOptimizerClient logs to logs/
        self.ray = ray
        if not ray.is_initialized():
            ray.init(address=ray_address) if ray_address else ray.init(num_cpus=n_actors)
        endpoints = list(dict.fromkeys(endpoints))   # a duplicated endpoint would double its load
        n_actors = max(1, min(n_actors, len(endpoints)))
        chunks = [endpoints[i::n_actors] for i in range(n_actors)]
        self.actors = [MultiEndpointWorker.remote(chunk) for chunk in chunks]
        self.capacity = {actor: max(1, len(chunk) * in_flight_per_endpoint) for actor, chunk in zip(self.actors, chunks)}
        self.logger = logging.getLogger("AmortizedDPExecution")
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def execute(self, requests):
        results = [None] * len(requests)
        pending = list(range(len(requests)))[::-1]
        in_flight = {}
        load = {actor: 0 for actor in self.actors}
        alive = list(self.actors)

        # QLever receives the timeout as a string rounded UP (format_latency: whole ms below
        # 1 s, whole seconds above), so a censored plan ran for at least the ROUNDED value.
        sent = [QLeverOptimizerClient.format_latency(r["timeout_s"]) for r in requests]
        effective = [QLeverOptimizerClient.decode_to_seconds(t) for t in sent]

        def submit(actor):
            while pending and load[actor] < self.capacity[actor]:
                index = pending.pop()
                request = requests[index]
                future = actor.execute_plan.remote(
                    {"query": request["query"], "triple_patterns": list(request["triple_patterns"])},
                    join_order=request["plan"], parse_local=False, timeout=sent[index])
                in_flight[future] = (actor, index)
                load[actor] += 1

        for actor in alive:
            submit(actor)
        with tqdm(total=len(requests), desc="Executing plans", leave=False, file=sys.stdout, mininterval=30) as bar:
            while in_flight:
                done, _ = self.ray.wait(list(in_flight), num_returns=1, timeout=1.0)
                for future in done:
                    actor, index = in_flight.pop(future)
                    load[actor] -= 1
                    timeout_s = effective[index]
                    try:
                        results[index] = _interpret(self.ray.get(future), timeout_s)
                    except self.ray.exceptions.RayActorError as error:
                        self.logger.error(f"Actor died on request {index}; dropping the actor. {error}")
                        results[index] = self._failed(timeout_s, error)
                        if actor in alive:
                            alive.remove(actor)
                        if not alive:
                            raise RuntimeError("Every execution actor has died.") from error
                    except Exception as error:  # a bad response must not end a long run
                        self.logger.error(f"Request {index} failed: {error}")
                        results[index] = self._failed(timeout_s, error)
                    bar.update(1)
                    if actor in alive:
                        submit(actor)
                for actor in alive:        # keep every actor busy if others died
                    submit(actor)
        return results

    def close(self):
        """Close every actor's HTTP sessions (MultiEndpointWorker.teardown)."""
        self.ray.get([actor.teardown.remote() for actor in self.actors])

    @staticmethod
    def _failed(timeout_s, error):
        return {"tree": {}, "latency_s": timeout_s, "censored": True, "timeout_s": timeout_s,
                "error": f"execution failed: {error}"[:300], "failed": True}


class SimulatedExecutor:
    """QLever-shaped trees from oracle cardinalities and a C_out-driven latency model.

    step time (ms) = (rows_left + rows_scanned + 2 * rows_out) / rows_per_ms, times a
    log-normal noise factor, rounded to whole milliseconds like QLever. A plan whose running
    total exceeds its timeout is cut off there, reported as an error, and its unfinished
    joins get status "cancelled". plan None stands in for QLever's own optimizer: an exact
    DP over noisy cardinalities (log-normal noise of `native_noise` on every subset).
    """

    def close(self):
        pass

    def __init__(self, labels_by_query, rows_per_ms=2e5, noise=0.15, native_noise=0.8, seed=0):
        self.labels_by_query = labels_by_query
        self.rows_per_ms = rows_per_ms
        self.noise = noise
        self.native_noise = native_noise
        self.rng = np.random.default_rng(seed)

    def _native_plan(self, labels):
        noisy = {mask: value + self.rng.normal(0.0, self.native_noise) for mask, value in labels.logcard.items()}
        return optimal_left_deep_plan(labels, noisy)[0]

    def _scan(self, pattern, rows):
        return {"cache_status": "computed", "children": [], "description": f"IndexScan PSO {pattern.rstrip(' .')}",
                "operation_time": 0, "total_time": 0, "result_rows": rows, "status": "fully materialized completed"}

    def execute(self, requests):
        results = []
        for request in requests:
            labels = self.labels_by_query[request["query"]]
            plan = request["plan"] if request["plan"] is not None else self._native_plan(labels)
            patterns = request["triple_patterns"]
            rows = lambda mask: int(round(math.exp(labels.logcard[mask])))
            timeout_ms = request["timeout_s"] * 1000
            node = self._scan(patterns[plan[0]], rows(1 << plan[0]))
            left_rows, total_ms, cut_off = node["result_rows"], 0.0, False
            for size in range(2, len(plan) + 1):
                scanned = rows(1 << plan[size - 1])
                out = rows(mask_of(plan[:size]))
                step = (left_rows + scanned + 2 * out) / self.rows_per_ms * math.exp(self.rng.normal(0.0, self.noise))
                cut_off = cut_off or total_ms + step > timeout_ms
                total_ms = total_ms + step if not cut_off else total_ms
                node = {"cache_status": "computed",
                        "children": [node, self._scan(patterns[plan[size - 1]], scanned)],
                        "description": "Join on ?x", "operation_time": int(round(step)),
                        "total_time": int(round(total_ms)), "result_rows": out,
                        "status": "cancelled" if cut_off else "fully materialized completed"}
                left_rows = out
            latency_s = request["timeout_s"] if cut_off else total_ms / 1000
            results.append({"tree": node, "latency_s": latency_s, "censored": cut_off,
                            "timeout_s": request["timeout_s"], "error": "Query timed out" if cut_off else None,
                            "plan_used": plan})
        return results
