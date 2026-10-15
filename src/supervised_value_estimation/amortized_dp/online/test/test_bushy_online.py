import math
import random

import numpy as np
import torch

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.bushy_model import ForestValueNet
from src.supervised_value_estimation.amortized_dp.bushy_train import (
    ForestTables, beam_plan, deviation_plans, greedy_plan,
)
from src.supervised_value_estimation.amortized_dp.labels import JOIN_FEATURE_DIM, join_position_features
from src.supervised_value_estimation.amortized_dp.online.observations import ForestObservations
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import bushy_sparql, parse_bushy_execution
from src.supervised_value_estimation.amortized_dp.online.train_online_bushy import (
    neighbour_masks_of, online_loss, query_rows,
)

# A 5-pattern query: 0-1-2 share ?a, 2-3 share ?b, 3-4 share ?c.
PATTERNS = ["?a <http://e/p0> ?x0 .", "?a <http://e/p1> ?x1 .", "?a <http://e/p2> ?b .",
            "?b <http://e/p3> ?c .", "?c <http://e/p4> ?x4 ."]


def _qlever_tree(tree, rows, ms, sort_every=2):
    """A QLever-shaped runtime tree for a plan: scans, joins (rows[mask], ms of the join itself) and
    a Sort node above every `sort_every`-th join; total_time is cumulative as QLever reports it."""
    counter = {"joins": 0}

    def build(node):
        if bushy.is_leaf(node):
            t = 1.0
            return {"description": f"IndexScan PSO {PATTERNS[node].rstrip(' .')}", "result_rows": 10 + node,
                    "operation_time": t, "total_time": t, "status": "fully materialized completed",
                    "cache_status": "computed", "children": []}
        left, right = build(node[0]), build(node[1])
        mask = bushy.tree_mask(node)
        own = ms[mask]
        join = {"description": "Join on ?v", "result_rows": rows[mask], "operation_time": own,
                "total_time": left["total_time"] + right["total_time"] + own, "status": "fully materialized completed",
                "cache_status": "computed", "children": [left, right]}
        counter["joins"] += 1
        if counter["joins"] % sort_every == 0:
            return {"description": "Sort (internal order) on ?v", "result_rows": rows[mask], "operation_time": 2.0,
                    "total_time": join["total_time"] + 2.0, "status": "fully materialized completed",
                    "cache_status": "computed", "children": [join]}
        return join
    return build(tree)


def _top_join(qtree):
    while not qtree["description"].startswith("Join"):
        qtree = qtree["children"][0]
    return qtree


def _plan_and_tree(tree, seed=0):
    rng = random.Random(seed)
    masks = bushy.internal_masks(tree)
    rows = {m: rng.randint(0, 1000) for m in masks}
    ms = {m: float(rng.randint(1, 20)) for m in masks}
    return rows, ms, _qlever_tree(tree, rows, ms)


def test_bushy_sparql_nests_every_subtree():
    query = bushy_sparql(((0, 1), (2, 3)), PATTERNS)
    assert query.count("{") == 1 + 7 and all(p in query for p in PATTERNS[:4])


def test_parse_recovers_rows_and_steps_that_add_up_to_the_total():
    nbr = neighbour_masks_of(PATTERNS)
    tree = (((0, 1), 2), (3, 4))
    assert bushy.is_valid_tree(tree, 5, nbr)
    rows, ms, qtree = _plan_and_tree(tree)
    parsed = parse_bushy_execution(qtree, bushy.internal_masks(tree), PATTERNS)
    assert parsed["aligned"] and parsed["rows"] == rows
    assert set(parsed["steps"]) == {tuple(sorted(m)) for m in bushy.tree_to_merges(tree)}
    # Time above the top join (a final Sort) belongs to no merge.
    assert sum(parsed["steps"].values()) == _top_join(qtree)["total_time"]
    assert parsed["scan_rows"] == {i: 10 + i for i in range(5)}
    # A different plan's masks: not aligned.
    assert not parse_bushy_execution(qtree, bushy.internal_masks(((((0, 1), 2), 3), 4)), PATTERNS)["aligned"]


def test_cached_subtree_drops_step_times_but_keeps_rows():
    tree = (((0, 1), 2), (3, 4))
    rows, ms, qtree = _plan_and_tree(tree)
    node = qtree
    while node["children"]:                          # the scan of pattern 0
        node = node["children"][0]
    node["cache_status"] = "cached_not_pinned"
    parsed = parse_bushy_execution(qtree, bushy.internal_masks(tree), PATTERNS)
    assert parsed["rows"] == rows and parsed["any_cached"]
    # Every merge above the cached scan loses its time; the other branch keeps it.
    assert set(parsed["steps"]) == {(0b1000, 0b10000)}


def test_latency_to_go_stitches_observed_steps():
    nbr = neighbour_masks_of(PATTERNS)
    obs = ForestObservations("q", "other", 5, nbr, PATTERNS)
    tree = (((0, 1), 2), (3, 4))
    rows, ms, qtree = _plan_and_tree(tree)
    parsed = parse_bushy_execution(qtree, bushy.internal_masks(tree), PATTERNS)
    obs.add(tree, parsed, qtree["total_time"] / 1000, False, 10.0)
    assert obs.latency_to_go(bushy.initial_forest(5)) == _top_join(qtree)["total_time"]
    assert obs.latency_to_go((0b11111,)) == 0.0
    assert obs.latency_to_go(bushy.merge(bushy.initial_forest(5), 0b1, 0b100)) == math.inf   # never observed


def _model():
    torch.manual_seed(0)
    return ForestValueNet(embedding_dim=8, hidden_dim=16, n_message_layers=2, message_layer="gine",
                          edge_feature_dim=JOIN_FEATURE_DIM, latency_head=True, step_latency_head=True)


def test_rows_loss_and_latency_planning():
    nbr = neighbour_masks_of(PATTERNS)
    obs = ForestObservations("q", "other", 5, nbr, PATTERNS)
    for seed, tree in enumerate([(((0, 1), 2), (3, 4)), ((((0, 1), 2), 3), 4), (((0, 2), 1), (3, 4))]):
        rows, ms, qtree = _plan_and_tree(tree, seed)
        obs.add(tree, parse_bushy_execution(qtree, bushy.internal_masks(tree), PATTERNS), 0.05, False, 10.0)
    arrays = query_rows(obs, 5)
    assert len(arrays["state"]) and arrays["latency_valid"].any() and arrays["step_valid"].any()
    assert np.all(arrays["side"][arrays["full"]].max(axis=1) == 1)
    arrays["query"] = np.zeros(len(arrays["state"]), dtype=np.int64)
    embeddings = torch.randn(5, 8)
    tables = ForestTables([obs.labels], [embeddings], [PATTERNS], arrays, "cpu")
    model = _model()
    model.set_target_statistics(*tables.statistics())
    model.set_latency_statistics(1.0, 1.0)
    model.set_step_statistics(1.0, 1.0)
    settings = {"under_weight": 0.1, "temperature": 1.0, "rank_weight": 1.0, "latency_weight": 1.0,
                "step_weight": 1.0, "balance_weight": 0.0}
    loss, parts = online_loss(model, tables, torch.arange(tables.n_states), settings)
    loss.backward()
    assert all(math.isfinite(v) for v in parts.values())
    adjacency = torch.from_numpy((np.array(nbr)[:, None] >> np.arange(5)) & 1 > 0)
    edges = torch.from_numpy(join_position_features(PATTERNS))
    model.eval()
    for objective in ("cout", "latency"):
        assert bushy.is_valid_tree(greedy_plan(model, embeddings, adjacency, edges, nbr, "cpu", objective=objective), 5, nbr)
        assert bushy.is_valid_tree(beam_plan(model, embeddings, adjacency, edges, nbr, "cpu", 4, objective=objective), 5, nbr)
        plans = deviation_plans(model, embeddings, adjacency, edges, nbr, "cpu", 4, objective)
        assert plans[0] == greedy_plan(model, embeddings, adjacency, edges, nbr, "cpu", objective=objective)
        assert len({bushy.canonical(p) for p in plans}) == len(plans) and all(bushy.is_valid_tree(p, 5, nbr) for p in plans)
