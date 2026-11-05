"""Exact row counts of every connected subset of the validation queries, from QLever.

The p-error needs the optimal plan under TRUE cardinalities, i.e. the true size of every
connected subset, not only of the prefixes some executed plan happened to reveal. Each
subset is counted once with
    SELECT (COUNT(*) AS ?count) WHERE { <its triple patterns> }
spread over the QLever endpoints, one request in flight per endpoint, and cached on disk
(query string -> {subset mask: rows, None if the count failed or timed out}). The cache is
extended, never rebuilt, so later runs on the same validation queries count nothing.

ReadTask measures what a lazily read pattern operand reads (the QLever planning cost, work_function.QLeverCost):
the join of a split (A, B) runs with A and B as separate groups, so a single pattern is a scan directly under the
join; its num-elements-read in the runtime tree is the label. Count tasks and read tasks run on the same workers
(count_subsets, CountStream: a task's query is a SPARQL count string or a ReadTask).
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


def count_on_endpoint(session, endpoint, query, timeout_s):
    response = session.post(endpoint, params={"timeout": f"{int(timeout_s)}s"}, data=query,
                            headers={"Accept": "application/sparql-results+json",
                                     "Content-Type": "application/sparql-query"},
                            timeout=timeout_s + 30)
    if response.status_code != 200:
        return None
    bindings = response.json().get("results", {}).get("bindings", [])
    return int(bindings[0]["count"]["value"]) if bindings else 0


class ReadTask:
    """The split (a, b) of a join: run(session, endpoint, timeout_s) -> {pattern index: rows read} for its operands
    that are single patterns (num-elements-read of their scans, under the join), or None if the query failed."""

    def __init__(self, triple_patterns, a, b):
        self.a, self.b = a, b
        self.members = list(members(a)) + list(members(b))
        self.patterns = [triple_patterns[i] for i in self.members]

        def group(mask):
            return "{ " + " ".join(p.strip() if p.strip().endswith(".") else p.strip() + " ."
                                   for p in (triple_patterns[i] for i in members(mask))) + " }"
        self.query = f"SELECT * WHERE {{ {group(a)} {group(b)} }}"

    def run(self, session, endpoint, timeout_s):
        from src.supervised_value_estimation.amortized_dp.online.runtime_tree import parse_bushy_execution
        response = session.post(endpoint, params={"timeout": f"{int(timeout_s)}s", "send": 0}, data=self.query,
                                headers={"Accept": "application/qlever-results+json",
                                         "Content-Type": "application/sparql-query"}, timeout=timeout_s + 30)
        if response.status_code != 200:
            return None
        tree = (response.json().get("runtimeInformation") or {}).get("query_execution_tree") or {}
        read = parse_bushy_execution(tree, [], self.patterns)["scan_read"]
        singles = [m.bit_length() - 1 for m in (self.a, self.b) if not m & (m - 1)]
        out = {p: read[self.members.index(p)] for p in singles if self.members.index(p) in read}
        return out if len(out) == len(singles) else None


def run_task(session, endpoint, query, timeout_s):
    """A count string -> rows; a ReadTask -> its reads."""
    if isinstance(query, ReadTask):
        return query.run(session, endpoint, timeout_s)
    return count_on_endpoint(session, endpoint, query, timeout_s)


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
                value = run_task(session, endpoint, query, timeout_s)
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


class CountStream:
    """count_subsets with tasks added while the counting runs (one thread per endpoint, as count_subsets): add(tasks)
    any number of times, then finish() -> {key: rows or None} once every task is counted. seconds: from the start to
    the last result."""

    def __init__(self, endpoints, timeout_s):
        self.work, self.results, self.lock = queue.Queue(), {}, threading.Lock()
        self.timeout_s, self.start = timeout_s, time.perf_counter()
        self.seconds = 0.0
        self.threads = [threading.Thread(target=self._worker, args=(endpoint,), daemon=True)
                        for endpoint in dict.fromkeys(endpoints)]
        for thread in self.threads:
            thread.start()

    def _worker(self, endpoint):
        session = requests.Session()
        while (task := self.work.get()) is not None:
            key, query = task
            try:
                value = run_task(session, endpoint, query, self.timeout_s)
            except (requests.RequestException, ValueError, KeyError):
                value = None
            with self.lock:
                self.results[key] = value
                self.seconds = time.perf_counter() - self.start

    def add(self, tasks):
        for task in tasks:
            self.work.put(task)

    def finish(self):
        for _ in self.threads:                           # after every task: a worker stops once the queue is drained
            self.work.put(None)
        for thread in self.threads:
            thread.join()
        return self.results


def _count_process(endpoints, timeout_s, tasks, results):
    while tasks.get() == "begin":
        stream = CountStream(endpoints, timeout_s)
        while (batch := tasks.get()) is not None:
            stream.add(batch)
        results.put((stream.finish(), stream.seconds))


class CountProcess:
    """CountStream in a child process, kept for the whole run: its endpoint threads no longer share the planning
    process's interpreter lock (work_2026-11-04/count_contention.py: planning 27% slower next to them). Per round
    begin() -> this, with CountStream's add(tasks), finish() -> {key: rows or None} and seconds (from begin to the
    last result)."""

    def __init__(self, endpoints, timeout_s):
        import multiprocessing as mp
        context = mp.get_context("spawn")
        self.tasks, self.results = context.Queue(), context.Queue()
        self.proc = context.Process(target=_count_process, args=(list(endpoints), timeout_s, self.tasks, self.results),
                                    daemon=True)
        self.proc.start()
        self.seconds = 0.0

    def begin(self):
        self.tasks.put("begin")
        return self

    def add(self, tasks):
        if tasks:
            self.tasks.put(list(tasks))

    def finish(self):
        self.tasks.put(None)
        while True:
            try:
                results, self.seconds = self.results.get(timeout=60)
                return results
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"the count process died (exit code {self.proc.exitcode})")

    def close(self):
        self.tasks.put("stop")
        self.proc.join(timeout=30)


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
