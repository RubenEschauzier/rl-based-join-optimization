"""Do generated queries work on QLever the way training and evaluation use them?

    python -m src.random_query_generation.statistics_based_generation.qlever_checks \\
        <pilot_*.jsonl or statistics_based_*.json> ... [--per-file 50] [--endpoints http://localhost:7000,...]

For a sample of queries per file:
  1. every query was counted (its certificate has a result), i.e. QLever parses and runs it;
  2. the in-memory single-pattern sizes equal QLever's COUNT(*) of each pattern, so the graph
     the generator walks and the index QLever answers from are the same data;
  3. executing the certificate's first plan with a FORCED join order (QLeverOptimizerClient,
     as in online training) gives a runtime tree that runtime_tree.parse_execution aligns,
     and whose prefix row counts add up to the certificate's C_out for that plan.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import parse_execution
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_query, count_subsets


def load_queries(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in open(path)]
    return [{**record["generation"], "triple_patterns": [" ".join(t[:3]) for t in record["triples"]]}
            for record in json.load(open(path))]


async def _forced_runs(queries, endpoints, timeout):
    os.makedirs("logs", exist_ok=True)                 # QLeverOptimizerClient logs to logs/
    clients = [QLeverOptimizerClient(endpoint) for endpoint in endpoints]
    queue = asyncio.Queue()
    for index, query in enumerate(queries):
        queue.put_nowait(index)
    results = [None] * len(queries)

    async def worker(client):
        while not queue.empty():
            index = queue.get_nowait()
            query = queries[index]
            body = " ".join(f"{p} ." for p in query["triple_patterns"])
            results[index] = await client.execute_plan(
                {"query": f"SELECT * WHERE {{ {body} }}", "triple_patterns": query["triple_patterns"]},
                join_order=query["orders"][0], timeout=timeout, parse_local=False)

    await asyncio.gather(*(worker(client) for client in clients))
    for client in clients:
        await client.close()
    return results


def check(paths, endpoints, per_file, timeout_s, seed=0):
    rng = random.Random(seed)
    report = {}
    for path in paths:
        queries = [q for q in load_queries(path) if q.get("certificate", {}).get("result") is not None]
        total = len(load_queries(path))
        sample = rng.sample(queries, min(per_file, len(queries)))
        # 2. single-pattern sizes
        tasks = [((i, k), count_query(q["triple_patterns"], 1 << k))
                 for i, q in enumerate(sample) for k in range(len(q["triple_patterns"]))]
        counts = count_subsets(tasks, endpoints, timeout_s)
        size_mismatch = [(i, k, sample[i]["pattern_sizes"][k], counts[(i, k)]) for (i, k) in counts
                         if counts[(i, k)] is not None and counts[(i, k)] != sample[i]["pattern_sizes"][k]]
        # 3. forced join order
        runs = asyncio.run(_forced_runs(sample, endpoints, f"{int(timeout_s)}s"))
        aligned = cost_match = executed = 0
        for query, run in zip(sample, runs):
            if not run or not run.get("success"):
                continue
            executed += 1
            tree = (run.get("runtime_info") or {}).get("query_execution_tree", {}) or {}
            parsed = parse_execution(tree, query["orders"][0], query["triple_patterns"])
            aligned += parsed["aligned"]
            expected = query["certificate"].get("heuristic_cost")
            if parsed["aligned"] and expected is not None and len(parsed["prefix_rows"]) == len(query["orders"][0]) - 1:
                cost_match += sum(parsed["prefix_rows"].values()) == expected
        report[str(path)] = {
            "queries": total, "counted": len(queries), "sampled": len(sample),
            "pattern_size_mismatches": len(size_mismatch), "mismatch_examples": size_mismatch[:5],
            "forced_plan_executed": executed, "aligned": aligned, "prefix_rows_equal_certificate": cost_match,
        }
        print(json.dumps({Path(path).name: report[str(path)]}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--per-file", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--endpoints", default="http://localhost:7000,http://localhost:7001,"
                                               "http://localhost:7002,http://localhost:7003")
    args = parser.parse_args()
    check(args.paths, args.endpoints.split(","), args.per_file, args.timeout)


if __name__ == "__main__":
    main()
