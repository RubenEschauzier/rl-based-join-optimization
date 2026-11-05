"""The planning server (plan_server.py, fv.network_steps): beams whose network part is answered in batches with
other queries' are the beams planned alone."""
import threading

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import bushy, forest_value as fv
from src.supervised_value_estimation.amortized_dp.online.plan_server import PlanPool, _pack, _unpack
from src.supervised_value_estimation.amortized_dp.test.test_forest_value import QCOST, _encoder, _net, _qnet, _query


class _Batched:
    """Beams in threads, each with its own `network`; a step waits until every running beam has sent its request,
    then all are answered by ONE network_open / network_steps call (as the server does, with requests of queries of
    different sizes and steps)."""

    def __init__(self, net, states):
        self.net, self.states = net, states
        self.cv = threading.Condition()
        self.waiting, self.running, self.batches = {}, 0, []

    def network(self, x):
        outer = self

        class Remote:
            def open(self):
                return outer._ask(x, "open", None)

            def step(self, req):                         # through numpy, as the pipes
                return _unpack(_pack(outer._ask(x, "step", _unpack(_pack(req)))))
        return Remote()

    def _ask(self, x, kind, req):
        with self.cv:
            self.waiting[x] = [kind, req, None]
            self.cv.notify_all()
            while self.waiting[x][2] is None:
                self.cv.wait()
            return self.waiting.pop(x)[2]

    def serve(self, run):
        threads = [threading.Thread(target=run, args=(x,)) for x in range(len(self.states))]
        self.running = len(threads)
        done = threading.Event()

        def finished(t):
            t.join()
            with self.cv:
                self.running -= 1
                self.cv.notify_all()
        for t in threads:
            t.start()
        watchers = [threading.Thread(target=finished, args=(t,)) for t in threads]
        for w in watchers:
            w.start()
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.running == 0 or len([1 for v in self.waiting.values() if v[2] is None]) == self.running)
                if self.running == 0:
                    break
                pend = [(x, v) for x, v in self.waiting.items() if v[2] is None]
                opens = [(x, v) for x, v in pend if v[0] == "open"]
                steps = [(x, v) for x, v in pend if v[0] == "step"]
                if opens:
                    for (x, v), r in zip(opens, fv.network_open(self.net, [self.states[x] for x, _ in opens])):
                        v[2] = tuple(t.clone() for t in r)
                if steps:
                    self.batches.append(len(steps))
                    for (x, v), r in zip(steps, fv.network_steps(self.net, [(self.states[x], v[1]) for x, v in steps])):
                        v[2] = r
                self.cv.notify_all()
        for w in watchers:
            w.join()
        done.set()


@pytest.mark.parametrize("cost", [None, QCOST])
def test_batched_server_beams_are_the_beams_alone(cost):
    """Six queries of 5-11 patterns planned at once (beam-8 with alternatives, two noise-free beams of other widths,
    2 builds per forest, the kept trace): every group's trees and the kept trace equal beam_plans alone."""
    queries = [_query(5 + s, s) for s in range(6)] + [_query(11, 9)]
    encoders = [_encoder(len(p), s)[0] for s, (p, _, _) in enumerate(queries)]
    net = _net(3, trained=True) if cost is None else _qnet(3)
    widths, alls = [8, 1, 4], [True, False, False]
    alone = []
    for (patterns, nb, degrees), enc in zip(queries, encoders):
        kept = []
        out, _ = fv.beam_plans(net, fv.QueryContext(len(patterns), nb, patterns, degrees, enc), widths, None, alls,
                               per_forest=2, kept=kept)
        alone.append((out, kept))
    states = [fv.PlanState(fv.QueryContext(len(p), nb, p, d, enc)) for (p, nb, d), enc in zip(queries, encoders)]
    server = _Batched(net, states)
    stub = type("Stub", (), {"cost": net.cost})()
    got = [None] * len(queries)

    def run(x):
        patterns, nb, degrees = queries[x]
        kept = []
        out, _ = fv.beam_plans(stub, fv.QueryContext(len(patterns), nb, patterns, degrees, None), widths, None, alls,
                               per_forest=2, kept=kept, network=server.network(x))
        got[x] = (out, kept)
    server.serve(run)
    assert max(server.batches) > 1                       # several queries per network pass
    for (out, kept), (out_b, kept_b) in zip(alone, got):
        assert out == out_b
        assert kept == kept_b


@pytest.mark.parametrize("cost", [None, QCOST])
def test_network_steps_of_several_queries_are_each_alone(cost):
    """One network_steps call over requests of queries of different sizes and steps (padded forests) gives each
    request's numbers alone."""
    net = _net(1, trained=True) if cost is None else _qnet(1)
    captured = []
    for s in range(4):
        patterns, nb, degrees = _query(6 + 2 * s, s)
        enc, _ = _encoder(len(patterns), s)
        st = fv.PlanState(fv.QueryContext(len(patterns), nb, patterns, degrees, enc))

        class Capture:
            def __init__(self, st):
                self.st, self.reqs = st, []

            def open(self):
                return fv.network_open(net, [self.st])[0]

            def step(self, req):
                self.reqs.append(req)
                return fv.network_steps(net, [(self.st, req)])[0]
        c = Capture(st)
        fv.beam_plans(type("Stub", (), {"cost": net.cost})(),
                      fv.QueryContext(len(patterns), nb, patterns, degrees, None), [4], network=c)
        captured.append(c)
    picks = [(c, c.reqs[(3 * x) % len(c.reqs)]) for x, c in enumerate(captured)]          # different steps
    assert len({r["kc"] for _, r in picks}) > 1
    together = fv.network_steps(net, [(c.st, r) for c, r in picks])
    for (c, r), t in zip(picks, together):
        a = fv.network_steps(net, [(c.st, r)])[0]
        for key in ("ru", "l2", "c_u", "cost_u", "V"):
            assert torch.allclose(a[key].double(), t[key].double(), atol=1e-5), key


def test_plan_pool_workers_match_beam_plans():
    """PlanPool's spawned workers, served by fv.network_steps here: the plans of beam_plans alone (noise-free groups)
    and valid trees for the noisy ones."""
    queries = [_query(6 + s % 5, s) for s in range(8)]
    encoders = [_encoder(len(p), s)[0] for s, (p, _, _) in enumerate(queries)]
    net = _net(5, trained=True)
    degrees = queries[0][2]                              # the same degree statistics for every query (the pool's)
    pool = PlanPool(3, degrees)
    try:
        jobs = [((lambda p=p, nb=nb, enc=enc: fv.PlanState(fv.QueryContext(len(p), nb, p, degrees, enc))),
                 {"n": len(p), "neighbour_masks": list(nb), "patterns": p, "widths": [8, 8, 1], "noises": [0.0, 1.0, 0.0],
                  "all_trees": [True, False, False], "per_forest": 1, "kept": True, "seed": x})
                for x, ((p, nb, _), enc) in enumerate(zip(queries, encoders))]
        got = {x: (groups, kept) for x, groups, kept in pool.plan(net, jobs)}
        assert pool.stats["requests"] > pool.stats["batches"]
    finally:
        pool.close()
    for x, ((p, nb, _), enc) in enumerate(zip(queries, encoders)):
        kept = []
        alone, _ = fv.beam_plans(net, fv.QueryContext(len(p), nb, p, degrees, enc), [8, 1], None, [True, False], kept=kept)
        groups, kept_b = got[x]
        assert groups[0] == alone[0] and groups[2] == alone[1]
        assert all(bushy.is_valid_tree(t, len(p), nb) for t in groups[1])
        assert [[t for g, t in step if g == 0] for step in kept_b] == [[t for g, t in step if g == 0] for step in kept]


def test_count_process_streams_rounds():
    """CountProcess: two rounds of tasks through the child process (a closed port: every count fails, None)."""
    from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import CountProcess
    counter = CountProcess(["http://127.0.0.1:9"], 1.0)
    try:
        for r in range(2):
            stream = counter.begin()
            stream.add([((f"q{r}", m), f"SELECT {m}") for m in range(5)])
            stream.add([])
            stream.add([((f"q{r}", 9), "SELECT 9")])
            results = stream.finish()
            assert results == {(f"q{r}", m): None for m in (0, 1, 2, 3, 4, 9)}
            assert stream.seconds >= 0
    finally:
        counter.close()
