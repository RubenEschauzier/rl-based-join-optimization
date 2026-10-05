"""Exact row counts of every connected subset of the validation queries, from QLever.

The p-error needs the optimal plan under TRUE cardinalities, i.e. the true size of every
connected subset, not only of the prefixes some executed plan happened to reveal. Each
subset is counted once with
    SELECT (COUNT(*) AS ?count) WHERE { <its triple patterns> }
spread over the QLever endpoints, one request in flight per endpoint, and cached on disk
(query string -> {subset mask: rows, None if the count failed or timed out}). The cache is
extended, never rebuilt, so later runs on the same validation queries count nothing.
"""
from __future__ import annotations

import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.supervised_value_estimation.amortized_dp.labels import members


def count_query(triple_patterns, mask):
    body = " ".join(p.strip() if p.strip().endswith(".") else p.strip() + " ."
                    for p in (triple_patterns[i] for i in members(mask)))
    return f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"


def _count(session, endpoint, query, timeout_s):
    response = session.post(endpoint, params={"timeout": f"{int(timeout_s)}s"}, data=query,
                            headers={"Accept": "application/sparql-results+json",
                                     "Content-Type": "application/sparql-query"},
                            timeout=timeout_s + 30)
    if response.status_code != 200:
        return None
    bindings = response.json().get("results", {}).get("bindings", [])
    return int(bindings[0]["count"]["value"]) if bindings else 0


def count_subsets(tasks, endpoints, timeout_s):
    """tasks: [(key, sparql count query)] -> {key: rows or None}. One thread per endpoint,
    so each single-core QLever instance runs one count at a time."""
    work = queue.Queue()
    for task in tasks:
        work.put(task)
    results, lock = {}, threading.Lock()
    bar = tqdm(total=len(tasks), desc="Counting true cardinalities", file=sys.stdout, mininterval=30)

    def worker(endpoint):
        session = requests.Session()
        while True:
            try:
                key, query = work.get_nowait()
            except queue.Empty:
                return
            try:
                value = _count(session, endpoint, query, timeout_s)
            except (requests.RequestException, ValueError, KeyError):
                value = None
            with lock:
                results[key] = value
                bar.update(1)

    threads = [threading.Thread(target=worker, args=(endpoint,), daemon=True) for endpoint in dict.fromkeys(endpoints)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    bar.close()
    return results


def load_or_count(path, queries, endpoints, timeout_s):
    """queries: {query string: (triple patterns, connected subset masks)} ->
    {query string: {mask: rows or None}}, counting only what the cache at `path` lacks."""
    path = Path(path)
    table = {}
    if path.exists():
        with open(path, "rb") as f:
            table = pickle.load(f)
    tasks = [((query, mask), count_query(patterns, mask))
             for query, (patterns, masks) in queries.items()
             for mask in masks if mask not in table.get(query, {})]
    if tasks:
        start = time.perf_counter()
        print(f"Counting {len(tasks)} subsets of {len(queries)} validation queries on {len(endpoints)} endpoints...")
        for (query, mask), value in count_subsets(tasks, endpoints, timeout_s).items():
            table.setdefault(query, {})[mask] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(temporary, "wb") as f:
            pickle.dump(table, f)
        os.replace(temporary, path)
        failed = sum(1 for (query, mask), _ in tasks if table[query][mask] is None)
        print(f"Counted in {time.perf_counter() - start:.0f}s; {failed} counts failed or timed out.")
    return {query: {mask: table[query][mask] for mask in masks} for query, (_, masks) in queries.items()}
