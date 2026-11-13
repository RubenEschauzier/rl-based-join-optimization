import itertools
import threading
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.random_query_generation.statistics_based_generation.certificate import (
    certificate_masks, connected_pairs, pattern_adjacency, plan_orders, summarise,
)
from src.random_query_generation.statistics_based_generation.generate import _accept
from src.random_query_generation.statistics_based_generation.counting import _Message, _sum_by
from src.random_query_generation.statistics_based_generation.graph import DataGraph, load_graph, parse_ntriples, pattern_terms
from src.random_query_generation.statistics_based_generation.hard_zero import HardZeroMaker
from src.random_query_generation.statistics_based_generation.instantiate import (
    ChokePoint, Instantiator, binding_plan, variable_connected,
)
from src.random_query_generation.statistics_based_generation.shapes import SHAPES, build_shape
from src.random_query_generation.statistics_based_generation.statistics import compute_statistics
from src.supervised_value_estimation.amortized_dp.labels import mask_of


def _random_triples(seed=0, n_entities=60, n_predicates=6, n_triples=900):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n_entities, n_triples)
    o = rng.integers(0, n_entities, n_triples)
    p = 1000 + rng.integers(0, n_predicates, n_triples)
    unique = np.unique(np.stack([s, p, o], axis=1), axis=0)
    unique = unique[unique[:, 0] != unique[:, 2]]
    return unique[:, 0], unique[:, 1], unique[:, 2]


@pytest.fixture(scope="module")
def data():
    s, p, o = _random_triples()
    graph = DataGraph.from_arrays(s, p, o)
    return graph, compute_statistics(graph), (s, p, o)


def test_ntriples_parsing(tmp_path):
    path = tmp_path / "g.nt"
    path.write_text("<http://example.com/1> <http://example.com/13000080> <http://example.com/2> .\n"
                    "<http://example.com/3> <http://example.com/13000081> <http://example.com/1> .\n")
    s, p, o = parse_ntriples(path)
    assert s.tolist() == [1, 3] and p.tolist() == [13000080, 13000081] and o.tolist() == [2, 1]


def test_dump_with_real_iris_and_literals_keeps_its_terms(tmp_path):
    path = tmp_path / "g.nt"
    path.write_text('<http://x.org/User0>\t<http://x.org/name>\t"deaf \\"gauntlets\\" quoit" .\n'
                    '<http://x.org/User0>\t<http://x.org/follows>\t<http://x.org/User1> .\n'
                    '<http://x.org/User1> <http://x.org/name> "Anna"@en .\n'
                    '<http://x.org/User0>\t<http://x.org/follows>\t<http://x.org/User1> .\n')   # duplicate: kept once
    graph = load_graph(path, tmp_path / "cache")
    for g in (graph, load_graph(path, tmp_path / "cache")):          # built, then from the cache
        assert g.n_triples == 3 and g.n_predicates == 2 and g.n_entities == 4
        ids = {g.iri(e): e for e in range(g.n_entities)}
        user0, user1 = ids["<http://x.org/User0>"], ids["<http://x.org/User1>"]
        name = next(q for q in range(g.n_predicates) if g.predicate_iri(q) == "<http://x.org/name>")
        assert g.iri(g.objects_of(user0, name)[0]) == '"deaf \\"gauntlets\\" quoit"'
        assert g.iri(g.objects_of(user1, name)[0]) == '"Anna"@en' and g.iri(user1) == "<http://x.org/User1>"
    pattern = '?v0 <http://x.org/name> "deaf \\"gauntlets\\" quoit"'
    assert pattern_terms(pattern) == ["?v0", "<http://x.org/name>", '"deaf \\"gauntlets\\" quoit"']


def test_graph_indexes_match_the_triples(data):
    graph, _, (s, p, o) = data
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = set(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    assert graph.n_triples == len(triples)
    for entity in range(0, 60, 7):
        for pred in range(graph.n_predicates):
            assert graph.count_bound_subject(entity, pred) == sum(1 for a, b, _ in triples if a == entity and b == pred)
            assert graph.count_bound_object(entity, pred) == sum(1 for _, b, c in triples if c == entity and b == pred)
    a, b, c = next(iter(triples))
    missing = next(x for x in range(60) if (a, b, x) not in triples)
    assert graph.has_triple(a, b, c) and not graph.has_triple(a, b, missing)


def test_pair_join_sizes_are_exact(data):
    graph, statistics, (s, p, o) = data
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = list(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    for p1, p2 in [(0, 1), (2, 2), (3, 5)]:
        t1 = [t for t in triples if t[1] == p1]
        t2 = [t for t in triples if t[1] == p2]
        assert statistics.joins["SS"][p1, p2] == sum(1 for x in t1 for y in t2 if x[0] == y[0])
        assert statistics.joins["SO"][p1, p2] == sum(1 for x in t1 for y in t2 if x[0] == y[2])
        assert statistics.joins["OS"][p1, p2] == sum(1 for x in t1 for y in t2 if x[2] == y[0])
        assert statistics.joins["OO"][p1, p2] == sum(1 for x in t1 for y in t2 if x[2] == y[2])


@pytest.mark.parametrize("kind", SHAPES)
@pytest.mark.parametrize("n", [4, 7, 12, 16])
def test_shapes_have_n_edges_and_are_connected(kind, n):
    shape = build_shape(kind, n, np.random.default_rng(n))
    assert shape.n_edges == n and variable_connected(shape, {})
    assert all(u != v for u, v in shape.edges) and len(set(shape.edges)) == n
    if kind in ("cycle", "diamond", "flower"):
        assert shape.n_edges >= shape.n_nodes                          # has a cycle


def test_binding_plans_never_break_variable_connectivity():
    rng = np.random.default_rng(0)
    for kind in SHAPES:
        for _ in range(20):
            shape = build_shape(kind, int(rng.integers(4, 12)), rng)
            plan = binding_plan(shape, ChokePoint("x", n_constants=(1, 4)), rng)
            assert variable_connected(shape, plan.constants)
    star = build_shape("star", 4, rng)
    assert not variable_connected(star, {star.root: None})          # a bound center is a cartesian product


@pytest.mark.parametrize("kind", ["star", "path", "tree", "snowflake", "path_star", "cycle"])
def test_embeddings_are_real_non_empty_and_respect_the_guardrail(data, kind):
    graph, statistics, _ = data
    instantiator = Instantiator(graph, statistics, max_pattern_matches=200, max_restarts=300)
    rng = np.random.default_rng(1)
    found = 0
    for choke_point in (ChokePoint("random"), ChokePoint("c", "correlated", 2.0), ChokePoint("h", hub_gamma=1.0)):
        shape = build_shape(kind, 5, rng)
        instance = instantiator.embed(shape, binding_plan(shape, choke_point, rng), choke_point, rng)
        if instance is None:
            continue
        found += 1
        assert all(graph.has_triple(instance.entities[u], p, instance.entities[v])
                   for (u, v), p in zip(instance.oriented, instance.predicates))
        assert len(set(instance.entities.tolist())) == shape.n_nodes
        assert {frozenset(e) for e in instance.oriented} == {frozenset(e) for e in shape.edges}
        assert max(instance.pattern_sizes(graph, statistics)) <= 200
    assert found >= 1


def test_certificate_math():
    patterns = ["?a <p> ?b", "?b <q> ?c", "?c <r> ?d"]
    assert pattern_adjacency(patterns) == [[1], [0, 2], [1]]
    assert connected_pairs(patterns) == [mask_of((0, 1)), mask_of((1, 2))]
    orders = [[0, 1, 2], [2, 1, 0]]
    counts = {mask_of((0, 1)): 10, 0b111: 4, mask_of((1, 2)): 1000}
    cert = summarise(patterns, orders, counts)
    assert cert["plan_costs"] == [14, 1004] and cert["min_cost"] == 14 and cert["heuristic_cost"] == 14
    assert cert["spread"] == pytest.approx(np.median([14, 1004]) / 14)
    assert "spread_with_scans" not in cert
    # scans are the same in every plan: 10000 scanned rows dilute the same C_out gap to ~1.05
    scanned = summarise(patterns, orders, counts, sizes=[5000, 3000, 2000])
    assert scanned["scan_rows"] == 10000 and scanned["best_plan_work"] == 10014
    assert scanned["spread_with_scans"] == pytest.approx((10000 + np.median([14, 1004])) / 10014)
    unknown = summarise(patterns, orders, {**counts, mask_of((1, 2)): None})
    assert unknown["n_unknown_plans"] == 1 and unknown["plan_costs"] == [14]
    zero = summarise(patterns, orders, {mask_of((0, 1)): 5, mask_of((1, 2)): 7, 0b111: 0}, hard_zero=True)
    assert zero["pairs_nonempty"] and zero["result"] == 0 and zero["first_empty_prefix"] == 3
    assert set(certificate_masks(patterns, orders, hard_zero=True)) == {0b011, 0b110, 0b111}
    rng = np.random.default_rng(0)
    for order in plan_orders(["?a <p> ?b", "?b <q> ?c", "?b <r> ?d", "?d <s> ?e"], [5, 1, 9, 3], 5, rng):
        adjacency = pattern_adjacency(["?a <p> ?b", "?b <q> ?c", "?b <r> ?d", "?d <s> ?e"])
        assert all(any(j in adjacency[i] for j in order[:k]) for k, i in enumerate(order) if k)
    assert plan_orders(["?a <p> ?b", "?b <q> ?c"], [7, 2], 3, rng)[0][0] == 1          # smallest first


def test_hard_zero_perturbations_keep_every_pattern_non_empty(data):
    graph, statistics, _ = data
    instantiator = Instantiator(graph, statistics, max_pattern_matches=500, max_restarts=300)
    maker = HardZeroMaker(graph, statistics, max_pattern_matches=500)
    rng = np.random.default_rng(2)
    made = 0
    for _ in range(30):
        shape = build_shape("path", 4, rng)
        cp = ChokePoint("x", n_constants=(1, 2))
        instance = instantiator.embed(shape, binding_plan(shape, cp, rng), cp, rng)
        variant = maker.perturb(instance, rng) if instance is not None else None
        if variant is None:
            continue
        made += 1
        assert all(size > 0 for size in variant.pattern_sizes(graph, statistics))
        changed = (variant.entities != instance.entities).sum() + (variant.predicates != instance.predicates).sum()
        assert changed == 1 and variant.perturbation in ("constant", "predicate")
    assert made >= 5


def test_acceptance_rules():
    acceptance = OmegaConf.create({"min_spread": 10, "max_min_cost": 1000, "max_result": 50})
    good = {"hard_zero": False, "certificate": {"result": 5, "spread": 20, "min_cost": 100}}
    assert _accept(good, acceptance) == (True, "accepted")
    assert _accept({**good, "certificate": {**good["certificate"], "spread": 3}}, acceptance)[1] == "low_spread"
    assert _accept({**good, "certificate": {**good["certificate"], "min_cost": 5000}}, acceptance)[1] == "expensive_optimum"
    assert _accept({**good, "certificate": {**good["certificate"], "result": 0}}, acceptance)[1] == "zero"
    zero = {"hard_zero": True, "certificate": {"result": 0, "spread": 50, "min_cost": 10, "empty_pairs": 0}}
    assert _accept(zero, acceptance)[0]
    one = {**zero, "certificate": {**zero["certificate"], "empty_pairs": 1}}
    assert _accept(one, acceptance, max_empty_pairs=1)[0]
    assert _accept(one, acceptance, max_empty_pairs=0)[1] == "not_hard_zero"
    assert _accept({**zero, "certificate": {**zero["certificate"], "empty_pairs": None}}, acceptance)[1] == "not_hard_zero"


# --- in-memory counting -------------------------------------------------------------------

def _brute_force_count(triples, patterns, subset):
    """Number of homomorphisms of the sub-BGP, by enumerating assignments of its variables."""
    subset = list(subset)
    variables = sorted({t[1] for i in subset for t in (patterns[i][0], patterns[i][2]) if t[0] == "v"})
    entities = sorted({x for s, _, o in triples for x in (s, o)})
    by_predicate = {}
    for s, p, o in triples:
        by_predicate.setdefault(p, set()).add((s, o))
    total = 0
    for values in itertools.product(entities, repeat=len(variables)):
        binding = dict(zip(variables, values))
        value = lambda term: binding[term[1]] if term[0] == "v" else term[1]
        total += all((value(patterns[i][0]), value(patterns[i][2])) in by_predicate.get(patterns[i][1], ())
                     for i in subset)
    return total


def test_acyclic_counts_equal_brute_force():
    from src.random_query_generation.statistics_based_generation.counting import AcyclicCounter, PredicateIndex
    s, p, o = _random_triples(seed=3, n_entities=14, n_predicates=3, n_triples=60)
    graph = DataGraph.from_arrays(s, p, o)
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = list(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    for full_pass_share in (0.0, 2.0):                    # both the full-pass and the gather path
        counter = AcyclicCounter(graph, PredicateIndex(graph), full_pass_share=full_pass_share)
        rng = np.random.default_rng(4)
        for _ in range(25):
            # random tree-shaped query over 3-4 variables, sometimes with constants
            n_vars = int(rng.integers(2, 5))
            patterns = []
            for v in range(1, n_vars):
                u = int(rng.integers(0, v))
                a, b = (("v", u), ("v", v)) if rng.random() < 0.5 else (("v", v), ("v", u))
                patterns.append((a, int(rng.integers(0, 3)), b))
            for _ in range(int(rng.integers(0, 3))):
                v = int(rng.integers(0, n_vars))
                constant = ("c", int(rng.choice([x for x, _, _ in triples])))
                patterns.append((("v", v), int(rng.integers(0, 3)), constant) if rng.random() < 0.5
                                else (constant, int(rng.integers(0, 3)), ("v", v)))
            for size in range(1, len(patterns) + 1):
                subset = range(size)
                if not AcyclicCounter.is_acyclic(patterns, subset):
                    continue
                connected = variable_connected_patterns(patterns, subset)
                if connected:
                    assert counter.count(patterns, subset) == _brute_force_count(triples, patterns, subset)


def variable_connected_patterns(patterns, subset):
    subset = list(subset)
    seen, frontier = {subset[0]}, [subset[0]]
    while frontier:
        i = frontier.pop()
        vi = {t[1] for t in (patterns[i][0], patterns[i][2]) if t[0] == "v"}
        for j in subset:
            if j not in seen and vi & {t[1] for t in (patterns[j][0], patterns[j][2]) if t[0] == "v"}:
                seen.add(j)
                frontier.append(j)
    return len(seen) == len(subset)


def test_cycles_are_detected():
    from src.random_query_generation.statistics_based_generation.counting import AcyclicCounter
    triangle = [(("v", 0), 0, ("v", 1)), (("v", 1), 0, ("v", 2)), (("v", 2), 0, ("v", 0))]
    assert AcyclicCounter.is_acyclic(triangle, [0, 1]) and not AcyclicCounter.is_acyclic(triangle, [0, 1, 2])
    double = [(("v", 0), 0, ("v", 1)), (("v", 1), 1, ("v", 0))]
    assert not AcyclicCounter.is_acyclic(double, [0, 1])


def test_constructed_star_hard_zeros_have_non_empty_pairs_and_an_empty_result():
    from src.random_query_generation.statistics_based_generation.counting import AcyclicCounter, PredicateIndex
    from src.random_query_generation.statistics_based_generation.instantiate import BindingPlan, Instance
    from src.random_query_generation.statistics_based_generation.shapes import QueryShape
    s, p, o = _random_triples(seed=5, n_entities=40, n_predicates=3, n_triples=500)
    graph = DataGraph.from_arrays(s, p, o)
    statistics = compute_statistics(graph)
    counter = AcyclicCounter(graph, PredicateIndex(graph))
    maker = HardZeroMaker(graph, statistics, max_pattern_matches=10_000, counter=counter)
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = list(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    rng = np.random.default_rng(0)
    built = 0
    for x in range(40):
        out = [(pp, oo) for ss, pp, oo in triples if ss == x]
        if len({pp for pp, _ in out}) < 3:
            continue
        chosen = []
        for pp, oo in out:
            if pp not in [c[0] for c in chosen]:
                chosen.append((pp, oo))
        chosen = chosen[:3]
        # star: x -> three leaves, all constants (the instance is a real witness)
        shape = QueryShape("star", 4, [(0, 1), (0, 2), (0, 3)], 0)
        entities = np.array([x] + [oo for _, oo in chosen])
        instance = Instance(shape, entities, np.array([pp for pp, _ in chosen]),
                            BindingPlan({1: "middle", 2: "middle", 3: "middle"}), [(0, 1), (0, 2), (0, 3)])
        variant = maker._split_constant(instance, 1, rng)
        if variant is None:
            continue
        built += 1
        patterns = variant.structured_patterns()
        assert _brute_force_count(triples, patterns, [0, 1, 2]) == 0
        for pair in ([0, 1], [0, 2], [1, 2]):
            assert _brute_force_count(triples, patterns, pair) > 0
    assert built >= 1


def test_splits_keep_heldout_templates_out_of_training(tmp_path):
    import json as json_
    from src.random_query_generation.statistics_based_generation.splits import split
    records = [{"query": f"q{i}", "y": 1, "generation": {"template": f"star|{4 + i % 6}|random|nonzero|"}}
               for i in range(300)]
    (tmp_path / "statistics_based_star.json").write_text(json_.dumps(records))
    split(tmp_path, tmp_path / "out", heldout_templates=0.34, val=0.1, test=0.1, seed=1)
    parts = {name: json_.load(open(tmp_path / "out" / f"dataset_{name}" / "raw" / "statistics_based_star.json"))
             for name in ("train", "val", "test", "test_heldout")}
    held = {q["generation"]["template"] for q in parts["test_heldout"]}
    assert len(held) == 2
    for name in ("train", "val", "test"):
        assert not held & {q["generation"]["template"] for q in parts[name]}
    assert sum(len(v) for v in parts.values()) == 300


def test_tightening_meets_the_budget_and_keeps_the_query_non_empty(data):
    from src.random_query_generation.statistics_based_generation.counting import AcyclicCounter, PredicateIndex
    from src.random_query_generation.statistics_based_generation.instantiate import (
        cycle_nodes, result_upper_bound, tighten_to_budget,
    )
    graph, statistics, (s, p, o) = data
    counter = AcyclicCounter(graph, PredicateIndex(graph))
    instantiator = Instantiator(graph, statistics, max_pattern_matches=10_000, max_restarts=300)
    rng = np.random.default_rng(3)
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = list(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    tightened = 0
    for kind in ("star", "path", "tree", "cycle"):
        for _ in range(5):
            shape = build_shape(kind, 4, rng)
            cp = ChokePoint("x", n_constants=(0, 0))
            instance = instantiator.embed(shape, binding_plan(shape, cp, rng), cp, rng)
            if instance is None:
                continue
            result = tighten_to_budget(instance, graph, statistics, counter, 5, rng)
            if result is None:
                continue
            tightened += 1
            patterns = result.structured_patterns()
            exact = _brute_force_count(triples, patterns, range(len(patterns)))
            assert 1 <= exact <= result_upper_bound(result, counter) <= 5
            assert not set(result.plan.constants) & cycle_nodes(shape)
            assert variable_connected(shape, result.plan.constants)
    assert tightened >= 3


def test_exact_counts_of_cyclic_queries_equal_brute_force():
    from src.random_query_generation.statistics_based_generation.counting import ExactCounter, PredicateIndex
    s, p, o = _random_triples(seed=6, n_entities=12, n_predicates=2, n_triples=70)
    graph = DataGraph.from_arrays(s, p, o)
    p_ids = np.searchsorted(graph.predicate_iris, p)
    triples = list(zip(s.tolist(), p_ids.tolist(), o.tolist()))
    counter = ExactCounter(graph, PredicateIndex(graph), max_assignments=10_000)
    rng = np.random.default_rng(1)
    shapes = {
        "triangle": [(0, 1), (1, 2), (2, 0)],
        "square+stem": [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4)],
        "diamond": [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
        "flower": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0), (0, 5)],
    }
    for name, edges in shapes.items():
        for _ in range(6):
            patterns = []
            for u, v in edges:
                a, b = (("v", u), ("v", v)) if rng.random() < 0.5 else (("v", v), ("v", u))
                patterns.append((a, int(rng.integers(0, 2)), b))
            if rng.random() < 0.5:        # a constant on the last node
                last = max(max(e) for e in edges)
                constant = ("c", int(rng.choice([x for x, _, _ in triples])))
                patterns[-1] = tuple(constant if t == ("v", last) else t for t in patterns[-1])
            subset = range(len(patterns))
            assert counter.count_exact(patterns, subset) == _brute_force_count(triples, patterns, subset), name
    tiny = ExactCounter(graph, PredicateIndex(graph), max_assignments=0)
    triangle = [(("v", 0), 0, ("v", 1)), (("v", 1), 0, ("v", 2)), (("v", 2), 0, ("v", 0))]
    assert tiny.count_exact(triangle, range(3)) is None


def test_spread_classes():
    from src.random_query_generation.statistics_based_generation.generate import spread_class
    assert [spread_class(s, [2, 10]) for s in (1.0, 1.99, 2.0, 9.9, 10.0, 1e6)] == [0, 0, 1, 1, 2, 2]


def test_dense_message_paths_match_the_sparse_ones():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 1000, 5000).astype(np.int32)
    weights = rng.random(5000) + 0.1
    sparse, dense = _sum_by(keys, weights), _sum_by(keys, weights, n_entities=1000)
    assert np.array_equal(sparse.ids, dense.ids) and np.allclose(sparse.values, dense.values)
    message = _Message(np.array([3, 17, 999]), np.array([2.0, 5.0, 7.0]))
    entities = rng.integers(0, 1000, 500)
    assert np.array_equal(message.lookup(entities), message.lookup(entities, n_entities=1000))


def test_big_patterns_fit_the_row_quota(data):
    graph, statistics, _ = data
    cap, quota = 100, 400                    # every predicate of the fixture matches more than the cap
    assert (statistics.counts[statistics.counts > 0] > cap).all()
    instantiator = Instantiator(graph, statistics, max_pattern_matches=cap, max_restarts=300, big_pattern_rows=quota)
    rng = np.random.default_rng(2)
    with_big = 0
    for kind in ("star", "path", "tree", "cycle"):
        for choke_point in (ChokePoint("random"), ChokePoint("a", n_constants=(2, 3))):
            shape = build_shape(kind, 5, rng)
            instance = instantiator.embed(shape, binding_plan(shape, choke_point, rng), choke_point, rng)
            if instance is None:
                continue
            over = [size for size in instance.pattern_sizes(graph, statistics) if size > cap]
            assert sum(over) <= quota
            with_big += bool(over)
    assert with_big >= 1


def test_predicate_index_is_sorted_both_ways(data):
    from src.random_query_generation.statistics_based_generation.counting import PredicateIndex
    graph, _, (s, p, o) = data
    index = PredicateIndex(graph)
    p_ids = np.searchsorted(graph.predicate_iris, p)
    for predicate in range(graph.n_predicates):
        expected = sorted((a, b) for a, q, b in zip(s.tolist(), p_ids.tolist(), o.tolist()) if q == predicate)
        subjects, objects = index.sorted_by(predicate, by_subject=True)
        assert list(zip(subjects.tolist(), objects.tolist())) == expected
        keys, others = index.sorted_by(predicate, by_subject=False)
        assert list(zip(keys.tolist(), others.tolist())) == sorted((b, a) for a, b in expected)


def _killing_candidate(job):
    """Stand-in for _make_candidate: no candidate, and the worker dies on one job (as if OOM-killed)."""
    import os
    if job[-1] == 3:
        os._exit(1)
    return None


def test_a_killed_worker_costs_one_job_not_the_shape(tmp_path, monkeypatch):
    import multiprocessing
    from collections import Counter as C
    from src.random_query_generation.statistics_based_generation import generate as G
    monkeypatch.setattr(G, "_make_candidate", _killing_candidate)
    monkeypatch.setattr(G, "LOST_CHECK_S", 0.2)
    gen = OmegaConf.create({
        "mode": "pilot", "sizes": [4], "zero_share": 0.0, "spread_strata": None, "workers": 2,
        "pilot": {"candidates_per_shape": 12}, "hard_zero": {"base_choke_points": ["random"], "max_empty_pairs": 1},
        "acceptance": {"max_candidate_factor": 10}, "progress_every": 1000,
        "certificate": {"timeout_s": 1}, "final_count": {"timeout_s": 1}})
    monkeypatch.setitem(G._STATE, "choke_points", {"random": (None, 1.0)})
    context = multiprocessing.get_context("fork")
    monkeypatch.setitem(G._STATE, "started", context.RawArray("q", 4))
    with context.Pool(2) as pool:
        summary, _ = G._run_shape("star", gen, [], pool, np.random.default_rng(0), iter(range(10 ** 6)), tmp_path)
    assert (tmp_path / "pilot_star.jsonl").exists()        # the shape finished despite the lost job


def test_estimate_is_the_independence_formula():
    from src.random_query_generation.statistics_based_generation.estimation import Estimator
    est = Estimator(distinct_subjects=[10, 4], distinct_objects=[5, 8])
    # ?a p0 ?x . ?x p1 ?b : x is p0's object (5 distinct) and p1's subject (4 distinct)
    patterns = [(("v", 0), 0, ("v", 1)), (("v", 1), 1, ("v", 2))]
    sizes = [20, 12]
    assert est.cardinality(patterns, sizes, [0]) == 20                              # one pattern: exact
    assert est.cardinality(patterns, sizes, [0, 1]) == pytest.approx(20 * 12 * 4 / (5 * 4))
    # a constant next to x: x's distinct values are the pattern's size
    patterns = [(("c", 7), 0, ("v", 1)), (("v", 1), 1, ("v", 2))]
    assert est.cardinality(patterns, [3, 12], [0, 1]) == pytest.approx(3 * 12 * 3 / (3 * 4))
    cert = est.certificate(patterns, [3, 12], [[0, 1], [1, 0]])
    assert cert["scan_rows"] == 15 and cert["best_plan_work"] == 15 + cert["min_cost"]


def test_pair_checker_agrees_with_the_exact_counter(data):
    from src.random_query_generation.statistics_based_generation.counting import AcyclicCounter, PredicateIndex
    from src.random_query_generation.statistics_based_generation.estimation import PairChecker
    graph, statistics, (s, p, o) = data
    counter = AcyclicCounter(graph, PredicateIndex(graph))
    checker = PairChecker(graph, statistics)
    rng = np.random.default_rng(5)
    entities = np.unique(np.concatenate([s, o]))
    seen = {True: 0, False: 0}
    for _ in range(400):
        p1, p2 = (int(x) for x in rng.integers(0, graph.n_predicates, 2))
        x_subject_1, x_subject_2 = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
        ends = []
        for k in range(2):
            ends.append(("c", int(rng.choice(entities))) if rng.random() < 0.5
                        else ("v", 10 + k))
        x = ("v", 0)
        first = (x, p1, ends[0]) if x_subject_1 else (ends[0], p1, x)
        second = (x, p2, ends[1]) if x_subject_2 else (ends[1], p2, x)
        patterns = [first, second]
        exact = counter.count(patterns, [0, 1])
        value = checker.count(patterns, (0, 1))
        assert (value > 0) == (exact > 0), (patterns, value, exact)
        seen[exact > 0] += 1
    assert seen[True] > 20 and seen[False] > 20                    # both outcomes were tested


def test_predicate_pools_match_the_index(data):
    from src.random_query_generation.statistics_based_generation.counting import PredicateIndex
    from src.random_query_generation.statistics_based_generation.estimation import predicate_pools
    graph, _, _ = data
    index = PredicateIndex(graph)
    ptr, subjects, objects = predicate_pools(graph)
    assert np.array_equal(ptr, index.ptr)
    for q in range(graph.n_predicates):
        a, b = ptr[q], ptr[q + 1]
        assert sorted(zip(subjects[a:b].tolist(), objects[a:b].tolist())) == \
            sorted(zip(index.subjects[a:b].tolist(), index.objects[a:b].tolist()))


def _fake_qlever(fail_on_memory):
    """A local HTTP server answering COUNT queries like QLever: 500 "Tried to allocate" for a query
    containing BIG if `fail_on_memory`, else the count 7."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            query = self.rfile.read(int(self.headers["Content-Length"])).decode()
            if fail_on_memory and "BIG" in query:
                status, body = 500, {"exception": "Tried to allocate 800 kB, but only 1 kB were available"}
            else:
                status, body = 200, {"results": {"bindings": [{"count": {"value": "7"}}]}}
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _collect(done, n):
    """{candidate name: count} of n finished candidates, and how many retries were announced."""
    results, escalated = {}, 0
    while len(results) < n:
        tag, c, counts = done.get(timeout=10)
        if tag == "escalated":
            escalated += 1
        else:
            results[c["name"]] = counts["result"]
    return results, escalated


def test_the_retry_queue_is_capped():
    """Beyond retries_per_thread queued retries per big lane thread, a memory failure is final."""
    import queue
    from src.random_query_generation.statistics_based_generation.certificate import BackgroundCounter
    small, small_url = _fake_qlever(fail_on_memory=True)
    big, big_url = _fake_qlever(fail_on_memory=False)
    done = queue.Queue()
    counter = BackgroundCounter([small_url] * 4, done, [big_url], retries_per_thread=2)
    # keep the big lane busy so the retries queue up: its thread only starts once all are submitted
    gate = threading.Event()
    original = counter._run
    def slow_big(session, endpoint, item, lane):
        if lane == "big":
            gate.wait(10)
        return original(session, endpoint, item, lane)
    counter._run = slow_big
    candidates = [{"name": i} for i in range(12)]
    for c in candidates:
        counter.submit(c, [("result", "SELECT (COUNT(*) AS ?count) WHERE { BIG }", 5)], "final")
    time.sleep(1.0)
    gate.set()
    results, escalated = _collect(done, len(candidates))
    counter.close()
    small.shutdown()
    big.shutdown()
    # the blocked big thread holds one count (a retry or small-lane work it helped with) while two
    # retries wait: at most 3 escalated, every other memory failure is final
    assert 1 <= escalated == counter.escalated <= 3
    assert sum(v is None for v in results.values()) >= len(candidates) - 1 - escalated
    assert counter.retries_queued == 0


@pytest.mark.parametrize("with_big_lane", [True, False])
def test_memory_failures_are_retried_in_the_big_lane(with_big_lane):
    import queue
    from src.random_query_generation.statistics_based_generation.certificate import BackgroundCounter
    small, small_url = _fake_qlever(fail_on_memory=True)
    big, big_url = _fake_qlever(fail_on_memory=False)
    done = queue.Queue()
    counter = BackgroundCounter([small_url], done, [big_url] if with_big_lane else [])
    candidates = [{"name": i} for i in range(20)]
    for i, c in enumerate(candidates):
        query = "SELECT (COUNT(*) AS ?count) WHERE { BIG }" if i % 2 else "SELECT (COUNT(*) AS ?count) WHERE { }"
        counter.submit(c, [("result", query, 5)], "final")
    results, escalated = _collect(done, len(candidates))
    counter.close()
    small.shutdown()
    big.shutdown()
    assert counter.pending == 0 and counter.retries_queued == 0
    assert escalated == counter.escalated     # every retry is announced before its result
    for i, c in enumerate(candidates):
        if i % 2 and not with_big_lane:
            assert results[i] is None and c["count_log"]["result"]["error"] == "memory"
        else:
            assert results[i] == 7 and c["count_log"]["result"]["error"] is None
    if with_big_lane:
        # the big lane also takes small-lane work when it has no retries, so some heavy counts go there directly
        assert counter.escalated == counter.escalated_ok <= 10
        assert all(candidates[i]["count_log"]["result"]["lane"] == "big" for i in range(1, 20, 2))


def _closed_port():
    """A local port nothing listens on (connection refused)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


@pytest.mark.parametrize("dead_in", ["small", "big"])
def test_a_dead_endpoint_does_not_swallow_counts(dead_in):
    """A count that cannot connect goes back to the queue for another endpoint; nothing is lost."""
    import queue
    from src.random_query_generation.statistics_based_generation.certificate import BackgroundCounter
    live, live_url = _fake_qlever(fail_on_memory=False)
    dead_url = _closed_port()
    small, big = ([dead_url, live_url], []) if dead_in == "small" else ([live_url], [dead_url])
    done = queue.Queue()
    counter = BackgroundCounter(small, done, big)
    counter.REST_S = 0.2
    candidates = [{"name": i} for i in range(30)]
    for c in candidates:
        counter.submit(c, [("result", "SELECT (COUNT(*) AS ?count) WHERE { }", 5)], "final")
    results, _ = _collect(done, len(candidates))
    counter.close()
    live.shutdown()
    assert all(v == 7 for v in results.values())
    assert counter.reconnects >= 1 and counter.pending == 0


def test_a_server_dying_mid_query_is_final():
    """A connection that breaks while the query runs is server_died (the query may have killed the
    instance), not requeued like an endpoint that is down."""
    import socket
    import requests
    from src.random_query_generation.statistics_based_generation import certificate
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    def accept_then_die():
        conn, _ = server.accept()
        conn.recv(65536)
        time.sleep(certificate.SERVER_DIED_AFTER_S + 0.5)
        conn.close()
    threading.Thread(target=accept_then_die, daemon=True).start()
    url = f"http://127.0.0.1:{server.getsockname()[1]}"
    assert certificate.count_with_reason(requests.Session(), url, "SELECT * WHERE { }", 10) == (None, "server_died")
    server.close()
    assert certificate.count_with_reason(requests.Session(), _closed_port(), "SELECT * WHERE { }", 10) == (None, "connection")


def _size_four_only_candidate(job):
    """Stand-in for _make_candidate: a cheap candidate for size 4, never one for size 5."""
    kind, n, choke_name, hard_zero, seed, selectivity = job
    if n != 4:
        return None
    patterns = [f"?a <http://example.org/p{seed}> ?b", f"?b <http://example.org/q{seed}> ?c",
                f"?c <http://example.org/r{seed}> ?d", f"?d <http://example.org/s{seed}> ?e"]
    return {"shape": kind, "n_patterns": n, "choke_point": choke_name, "hard_zero": hard_zero,
            "triple_patterns": patterns, "template": f"{kind}|{n}", "entities": [], "seed": seed,
            "selectivity": selectivity, "pattern_sizes": [10, 10, 10, 10],
            "estimate": {"best_plan_work": 1.0, "spread": 1.0}}


def test_a_cell_that_cannot_fill_hands_its_places_on(tmp_path, monkeypatch):
    import multiprocessing
    from src.random_query_generation.statistics_based_generation import generate as G
    monkeypatch.setattr(G, "_make_candidate", _size_four_only_candidate)
    server, url = _fake_qlever(fail_on_memory=False)
    gen = OmegaConf.create({
        "mode": "generate", "method": "estimated", "sizes": [4, 5], "zero_share": 0.0, "spread_strata": None,
        "workers": 2, "queries_per_shape": 40, "progress_every": 10 ** 6,
        "hard_zero": {"base_choke_points": ["random"], "max_empty_pairs": 1},
        "acceptance": {"max_candidate_factor": 50, "max_best_plan_work": None, "max_result": None},
        "estimated": {"max_constant_share": 1.0},
        "adaptive": {"selectivity": {"window": 5, "target_pass": 0.3, "step": 0.5, "max": 2.0},
                     "max_attempts_per_query": 3},
        "certificate": {"timeout_s": 5}, "final_count": {"timeout_s": 5}})
    monkeypatch.setitem(G._STATE, "choke_points", {"random": (None, 1.0)})
    context = multiprocessing.get_context("fork")
    monkeypatch.setitem(G._STATE, "started", context.RawArray("q", 4))
    with context.Pool(2) as pool:
        _, written = G._run_shape("path", gen, ([url], []), pool, np.random.default_rng(0),
                                  iter(range(10 ** 6)), tmp_path)
    server.shutdown()
    assert written["written"] == 40                       # the shape still gets its total, all of size 4
    [closed] = written["closed_cells"]
    assert closed["cell"][0] == 5 and closed["kept"] == 0 and closed["handed_on"] == 20
    assert closed["attempts_of_key"] >= 3 * 20
    assert written["selectivity"]["5|random|False"] == 2.0     # size 5 never passed: beta went to its max
    assert "4|random|False" not in written["selectivity"]      # size 4 always passed: beta stayed 0
