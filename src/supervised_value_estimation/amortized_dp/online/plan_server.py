"""Planning a training round's queries over one GPU network server (2026-11-04).

Planning is launch-bound (work_2026-11-04/plan_timing.py): a query's beams take ~5,000 small CUDA kernels, the GPU
busy 10-34% of the time; the network part of a step (encoding the new subsets, the bound, the value net over a few
hundred forests) is ~55-80% of the time, the selection between steps the rest. Here `workers` spawned processes each
plan one query at a time with fv.beam_plans on CPU tensors (the selection, in parallel) and send each step's network
part to this process, the server, which answers every waiting request with ONE fv.network_steps call (one batched
encoder pass over all their new subsets, then the bound and the value net once): the kernels are launched once per
batch of queries instead of once per query.

    pool = PlanPool(workers, degrees, net.config["cost"])
    for index, groups, kept in pool.plan(net, jobs, encode_many):
        ...                                              # in the order the queries finish

jobs [(make_state, spec)]: make_state() -> the query's fv.PlanState (built when the query is dispatched, released when
it is done); spec {n, neighbour_masks, patterns, widths, noises, all_trees, per_forest, kept, seed}: fv.beam_plans's
arguments (kept: trace the kept forests; noise from a CPU generator seeded with seed). groups: beam_plans's trees per
group; kept: its kept trace or None. The plain planner only (no epinet). Messages carry numpy arrays: a torch tensor
would travel through its own shared-memory file.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from types import SimpleNamespace

import numpy as np
import torch

from src.supervised_value_estimation.amortized_dp import forest_value as fv
from src.supervised_value_estimation.amortized_dp.work_function import QLeverCost


def _pack(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    if isinstance(x, dict):
        return {k: _pack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_pack(v) for v in x)
    return x


def _unpack(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, dict):
        return {k: _unpack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_unpack(v) for v in x)
    return x


GATHER_S = 0.010        # a batch waits up to this long for the other running queries' requests (their selection)


class _Remote:
    """beam_plans's `network` in a worker: each call a round trip to the server."""

    def __init__(self, conn, job):
        self.conn, self.job = conn, job

    def open(self):
        self.conn.send(("open", self.job))
        return _unpack(self.conn.recv())

    def step(self, req):
        self.conn.send(("step", self.job, _pack(req)))
        return _unpack(self.conn.recv())


def _worker(conn, degrees, cost):
    torch.set_num_threads(1)
    net = SimpleNamespace(cost=QLeverCost(**cost) if cost else None)     # all beam_plans reads of the net here
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):                      # the server is gone (the run ended)
            return
        if msg is None:
            return
        job, spec = msg
        try:
            ctx = fv.QueryContext(spec["n"], spec["neighbour_masks"], spec["patterns"], degrees, None)
            kept = [] if spec.get("kept") else None
            generator = torch.Generator().manual_seed(int(spec["seed"]))
            groups, _ = fv.beam_plans(net, ctx, spec["widths"], spec["noises"], spec["all_trees"], generator,
                                      per_forest=int(spec.get("per_forest", 1)), kept=kept, network=_Remote(conn, job))
            conn.send(("done", job, groups, kept))
        except Exception:                                # noqa: BLE001 -- reported by the server
            conn.send(("error", job, traceback.format_exc()))


class PlanPool:
    """`workers` planning processes (spawned: the server holds the CUDA context), kept for the whole run. stats:
    the last plan()'s network batches and requests (requests / batch = how many queries a batch serves)."""

    def __init__(self, workers, degrees, cost=None):
        context = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(workers):
            here, there = context.Pipe()
            proc = context.Process(target=_worker, args=(there, degrees, dict(cost) if cost else None), daemon=True)
            proc.start()
            there.close()
            self.conns.append(here)
            self.procs.append(proc)
        self.stats = {"batches": 0, "requests": 0}

    @torch.no_grad()
    def plan(self, net, jobs, encode_many=None):
        """Plan the jobs; yields (job index, groups, kept) as they finish. Each round of the loop answers every
        waiting worker with one network_open / network_steps call."""
        pending, idle, busy, states = deque(range(len(jobs))), list(self.conns), {}, {}
        self.stats = {"batches": 0, "requests": 0}
        while pending or busy:
            while idle and pending:
                x = pending.popleft()
                conn = idle.pop()
                states[x], busy[conn] = jobs[x][0](), x
                conn.send((x, jobs[x][1]))
            opens, steps, done = [], [], []
            ready = wait(list(busy))
            t0 = time.perf_counter()                     # gather: the other running queries are a step behind
            while len(ready) < len(busy) and (left := GATHER_S - (time.perf_counter() - t0)) > 0:
                ready += wait([c for c in busy if c not in ready], timeout=left)
            for conn in ready:
                msg = conn.recv()
                if msg[0] == "open":
                    opens.append((conn, msg[1]))
                elif msg[0] == "step":
                    steps.append((conn, msg[1], _unpack(msg[2])))
                elif msg[0] == "done":
                    del busy[conn], states[msg[1]]
                    idle.append(conn)
                    done.append(msg[1:])
                else:
                    raise RuntimeError(f"planning worker failed on job {msg[1]}:\n{msg[2]}")
            if opens:
                for (conn, _), r in zip(opens, fv.network_open(net, [states[x] for _, x in opens], encode_many)):
                    conn.send(_pack(r))
            if steps:
                for (conn, _, _), r in zip(steps, fv.network_steps(net, [(states[x], q) for _, x, q in steps], encode_many)):
                    conn.send(_pack(r))
                self.stats["batches"] += 1
                self.stats["requests"] += len(steps)
            yield from done

    def close(self):
        for conn in self.conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
