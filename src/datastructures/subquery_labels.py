"""True cardinalities of random connected sub-queries, counted on QLever (labels for training an
estimator on the sub-queries a DP asks about, not only on whole queries).

    # locally: sample sub-queries of a dataset's train / val splits into one task file
    python -m src.datastructures.subquery_labels sample --dataset data/generated_queries/mixed_yago_v2 \\
        --out data/generated_queries/mixed_yago_v2/subquery_labels/tasks.jsonl.gz

    # on the slice (see dockerfiles/virtual_wall/run_subquery_counts.sh): count them
    python -m src.datastructures.subquery_labels count --tasks ~/subquery_tasks.jsonl.gz --out ~/subquery_results.jsonl

    # locally again: per split, {query: {mask: count}}
    python -m src.datastructures.subquery_labels merge --tasks ... --results ... --out-dir .../subquery_labels

Sampling (independent of any model or plan): per query `pairs` random connected pattern pairs and
`larger` random connected subsets whose size is drawn uniformly from 3..n-1 (then grown from a random
pattern), so every size is represented, not mostly the middle ones. Masks are over the query's
triple patterns in ProcessQuery.deconstruct_to_triple_pattern order -- the order of the processed
datasets, whose node features the DP slices sub-queries from.

Counting: plain SELECT (COUNT(*)) per sub-query (QLever plans it). One thread per endpoint pulls
from a shared queue, so a long count occupies one endpoint while the others keep going. Results are
appended to --out as they finish (one JSON line per task); a rerun skips what is already there.
A count that fails or times out is stored as null and not retried.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

import requests

VARIABLE_PREFIXES = ("?", "$")


def _adjacency(triple_patterns):
    variables = [{t for t in tp.split()[:3:2] if t.startswith(VARIABLE_PREFIXES)} for tp in triple_patterns]
    return [[j for j in range(len(variables)) if j != i and variables[i] & variables[j]]
            for i in range(len(variables))]


def sample_masks(triple_patterns, rng, pairs=2, larger=4):
    """Random connected sub-queries (bitmasks), see the module docstring."""
    n = len(triple_patterns)
    adjacency = _adjacency(triple_patterns)
    masks = set()
    all_pairs = [(i, j) for i in range(n) for j in adjacency[i] if i < j]
    for i, j in rng.sample(all_pairs, min(pairs, len(all_pairs))):
        masks.add((1 << i) | (1 << j))
    attempts = 0
    while n >= 4 and len(masks) < min(pairs, len(all_pairs)) + larger and attempts < 20 * larger:
        attempts += 1
        size = rng.randint(3, n - 1)
        chosen = {rng.randrange(n)}
        while len(chosen) < size:
            frontier = sorted({j for i in chosen for j in adjacency[i]} - chosen)
            if not frontier:
                break
            chosen.add(rng.choice(frontier))
        if len(chosen) == size:
            masks.add(sum(1 << i for i in chosen))
    return sorted(masks)


def count_query(triple_patterns, mask):
    body = " ".join(tp if tp.rstrip().endswith(".") else tp + " ." for i, tp in enumerate(triple_patterns) if mask >> i & 1)
    return f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"


def _open(path, mode):
    return gzip.open(path, mode + "t") if str(path).endswith(".gz") else open(path, mode)


def sample(args):
    from src.datastructures.query import ProcessQuery
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True) if out.suffix == "" else out.parent.mkdir(parents=True, exist_ok=True)
    n_tasks = n_queries = 0
    with _open(out, "w") as f:
        for split in ("train", "val"):
            for path in sorted(glob.glob(os.path.join(args.dataset, f"dataset_{split}", "raw", "*.json"))):
                for record in json.load(open(path)):
                    triple_patterns, _ = ProcessQuery.deconstruct_to_triple_pattern(record["query"])
                    masks = sample_masks(triple_patterns, rng, args.pairs, args.larger)
                    if not masks:
                        continue
                    f.write(json.dumps({"id": n_queries, "split": split, "query": record["query"],
                                        "triple_patterns": triple_patterns, "masks": masks}) + "\n")
                    n_queries += 1
                    n_tasks += len(masks)
    print(f"[sample] {n_queries:,} queries, {n_tasks:,} sub-queries -> {out}")


def _endpoints(args):
    if args.endpoints:
        return [e.strip() for e in args.endpoints.split(",") if e.strip()]
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    execution = config.online.execution if "online" in config else config.execution    # online config: online.execution
    hosts = [str(h).split("://", 1)[-1].rstrip("/") for h in execution.endpoint_hosts]
    return [f"http://{h}:{p}" for h in hosts
            for p in range(int(execution.base_port), int(execution.base_port) + int(execution.ports_per_host))]


def _access_token(args):
    if os.environ.get("QLEVER_ACCESS_TOKEN"):
        return os.environ["QLEVER_ACCESS_TOKEN"]
    env = Path(args.qlever_env)
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("QLEVER_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _count(session, endpoint, sparql, timeout_s, token):
    params = {"timeout": f"{int(timeout_s)}s"}
    if token:
        params["access-token"] = token
    response = session.post(endpoint, params=params, data=sparql, timeout=timeout_s + 60,
                            headers={"Accept": "application/sparql-results+json",
                                     "Content-Type": "application/sparql-query"})
    if response.status_code != 200:
        return None
    bindings = response.json().get("results", {}).get("bindings", [])
    return int(bindings[0]["count"]["value"]) if bindings else 0


def count(args):
    endpoints, token = _endpoints(args), _access_token(args)
    alive = []
    for endpoint in endpoints:          # health check: drop endpoints that do not answer
        try:
            if _count(requests.Session(), endpoint, "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }", 30, token):
                alive.append(endpoint)
        except requests.RequestException:
            pass
    print(f"[count] {len(alive)}/{len(endpoints)} endpoints answer; access token {'set' if token else 'NOT set'}",
          flush=True)
    if not alive:
        raise SystemExit("no QLever endpoint answers")
    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                record = json.loads(line)
                done.add((record["id"], record["mask"]))
    work = queue.Queue()
    total = 0
    with _open(args.tasks, "r") as f:
        tasks = [json.loads(line) for line in f]
    random.Random(0).shuffle(tasks)            # long sub-queries spread over the run
    for task in tasks:
        for mask in task["masks"]:
            if (task["id"], mask) not in done:
                work.put((task["id"], mask, count_query(task["triple_patterns"], mask)))
                total += 1
    print(f"[count] {total:,} sub-queries to count ({len(done):,} already done), timeout {args.timeout:.0f}s",
          flush=True)
    lock, state = threading.Lock(), {"done": 0, "failed": 0, "start": time.perf_counter()}
    out = open(args.out, "a", buffering=1)      # line-buffered: an interruption loses nothing

    def worker(endpoint):
        session = requests.Session()
        while True:
            try:
                query_id, mask, sparql = work.get_nowait()
            except queue.Empty:
                return
            start = time.perf_counter()
            try:
                value = _count(session, endpoint, sparql, args.timeout, token)
            except (requests.RequestException, ValueError, KeyError):
                value = None
            record = {"id": query_id, "mask": mask, "count": value, "ms": round((time.perf_counter() - start) * 1000)}
            with lock:
                out.write(json.dumps(record) + "\n")
                state["done"] += 1
                state["failed"] += value is None
                if state["done"] % 5000 == 0:
                    elapsed = time.perf_counter() - state["start"]
                    print(f"[count] {state['done']:,}/{total:,} ({state['done'] / elapsed:.0f}/s, "
                          f"{state['failed']:,} failed or timed out, {elapsed / 60:.0f} min)", flush=True)

    threads = [threading.Thread(target=worker, args=(e,), daemon=True) for e in alive]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    out.close()
    print(f"[count] done: {state['done']:,} counted, {state['failed']:,} failed or timed out -> {args.out}")


def merge(args):
    with _open(args.tasks, "r") as f:
        tasks = {t["id"]: t for t in map(json.loads, f)}
    labels = {"train": {}, "val": {}}
    failed = 0
    with open(args.results) as f:
        for record in map(json.loads, f):
            task = tasks[record["id"]]
            if record["count"] is None:
                failed += 1
                continue
            entry = labels[task["split"]].setdefault(task["query"], {"triple_patterns": task["triple_patterns"], "counts": {}})
            entry["counts"][str(record["mask"])] = record["count"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, table in labels.items():
        with open(out / f"subquery_labels_{split}.json", "w") as f:
            json.dump(table, f)
        print(f"[merge] {split}: {len(table):,} queries, {sum(len(e['counts']) for e in table.values()):,} sub-query labels")
    print(f"[merge] {failed:,} sub-queries without a count (failed or timed out)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--dataset", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--pairs", type=int, default=2)
    s.add_argument("--larger", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("count")
    c.add_argument("--tasks", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--timeout", type=float, default=600)
    c.add_argument("--endpoints", default=None, help="comma-separated; default: execution section of --config")
    c.add_argument("--config", default="experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed/"
                                       "online_amortized_dp_mixed_yago.yaml")
    c.add_argument("--qlever-env", default="data/qlever/qlever_yago/qlever_yago.env")
    m = sub.add_parser("merge")
    m.add_argument("--tasks", required=True)
    m.add_argument("--results", required=True)
    m.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    {"sample": sample, "count": count, "merge": merge}[args.command](args)


if __name__ == "__main__":
    main()
