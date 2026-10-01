import ast
import json
import copy
import math
from pathlib import Path

import pytest

from src.supervised_value_estimation.amortized_dp.labels import mask_of
from src.supervised_value_estimation.amortized_dp.online.observations import QueryObservations
from src.supervised_value_estimation.amortized_dp.online.runtime_tree import parse_execution

REAL_OUTPUT = Path(__file__).resolve().parents[3] / "utils" / "visualize_qlever_runtime_output.py"


def _real_tree():
    """The real QLever runtime tree stored in utils/visualize_qlever_runtime_output.py."""
    source = REAL_OUTPUT.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "qlever_output":
            return ast.literal_eval(node.value)["query_execution_tree"]
    raise AssertionError("qlever_output not found")


# The five patterns of that query, deliberately listed in a different order than the plan.
PATTERNS = [
    "?s <http://example.com/13000087> ?o4 .",                       # scanned last
    "?s <http://example.com/13000080> <http://example.com/10724425> .",
    "?s <http://example.com/13000087> <http://example.com/2941846> .",
    "?s <http://example.com/13000080> <http://example.com/8719681> .",
    "?s <http://example.com/13000087> <http://example.com/2447036> .",
]
PLAN = [1, 3, 4, 2, 0]      # the join order the tree encodes


def test_real_tree_prefix_cardinalities_and_step_times():
    parsed = parse_execution(_real_tree(), PLAN, PATTERNS)
    assert parsed["aligned"] and parsed["n_join_nodes"] == 4 and not parsed["any_cached"]
    assert parsed["prefix_rows"] == {2: 1, 3: 1, 4: 1, 5: 12}
    # Subtree total times are 1, 2, 4, 6 ms, so each step adds 1, 1, 2, 2 ms.
    assert parsed["step_ms"] == {2: 1.0, 3: 1.0, 4: 2.0, 5: 2.0}


def test_real_tree_scans_are_matched_to_their_patterns():
    parsed = parse_execution(_real_tree(), PLAN, PATTERNS)
    assert parsed["scan_rows"] == {1: 1, 3: 1, 4: 142, 2: 67656, 0: 20724}


def test_a_wrong_join_count_drops_every_prefix_observation():
    parsed = parse_execution(_real_tree(), PLAN + [5], PATTERNS + ["?s <http://p/x> ?y ."])
    assert not parsed["aligned"] and parsed["prefix_rows"] == {} and parsed["step_ms"] == {}


def test_multi_column_joins_count_as_joins():
    tree = copy.deepcopy(_real_tree())
    tree["children"][0]["description"] = "MultiColumnJoin on ?s ?x"
    assert parse_execution(tree, PLAN, PATTERNS)["aligned"]


def test_cached_subtrees_keep_cardinalities_but_lose_their_times():
    tree = copy.deepcopy(_real_tree())
    tree["children"][0]["children"][0]["cache_status"] = "cached_not_pinned"   # the 3-pattern join
    parsed = parse_execution(tree, PLAN, PATTERNS)
    assert parsed["any_cached"]
    assert parsed["prefix_rows"] == {2: 1, 3: 1, 4: 1, 5: 12}
    assert parsed["step_ms"] == {2: 1.0}          # every step from the cached one onwards is dropped


def test_unfinished_joins_after_a_timeout_are_not_used():
    tree = copy.deepcopy(_real_tree())
    tree["status"] = "cancelled"
    parsed = parse_execution(tree, PLAN, PATTERNS)
    assert 5 not in parsed["prefix_rows"] and 5 not in parsed["step_ms"]
    assert parsed["prefix_rows"] == {2: 1, 3: 1, 4: 1}


def _observations():
    full = (1 << 5) - 1
    return QueryObservations("q", "star", 5, [full & ~(1 << i) for i in range(5)])


def test_observations_store_log1p_rows_and_stitch_latency():
    obs = _observations()
    obs.add(PLAN, parse_execution(_real_tree(), PLAN, PATTERNS), latency_s=0.006, censored=False, timeout_s=60)
    assert obs.logcard[mask_of(PLAN[:5])] == pytest.approx(math.log1p(12))
    assert obs.logcard[1 << 2] == pytest.approx(math.log1p(67656))
    to_go = obs.latency_to_go()
    assert to_go[mask_of(PLAN[:2])] == pytest.approx(math.log1p(1 + 2 + 2))   # steps 3, 4, 5
    assert obs.best_stitched_latency_ms() == pytest.approx(6.0)
    assert obs.best_latency_s == pytest.approx(0.006)


def test_censored_executions_are_not_a_best_latency():
    obs = _observations()
    obs.add(PLAN, parse_execution(_real_tree(), PLAN, PATTERNS), latency_s=60.0, censored=True, timeout_s=60)
    assert obs.best_latency_s is None


def test_simulated_trees_parse_like_real_ones():
    import random
    from src.supervised_value_estimation.amortized_dp.labels import QueryLabels, plan_log_cost
    from src.supervised_value_estimation.amortized_dp.online.executor import SimulatedExecutor

    n = 5
    rng = random.Random(0)
    full = (1 << n) - 1
    labels = QueryLabels("q", "star", n, [full & ~(1 << i) for i in range(n)],
                         {m: rng.uniform(1, 9) for m in range(1, 1 << n)})
    executor = SimulatedExecutor({"q": labels}, noise=0.0)
    result = executor.execute([{"query": "q", "triple_patterns": PATTERNS, "plan": PLAN, "timeout_s": 60}])[0]
    parsed = parse_execution(result["tree"], PLAN, PATTERNS)
    assert parsed["aligned"] and not result["censored"]
    for size, rows in parsed["prefix_rows"].items():
        assert rows == round(math.exp(labels.logcard[mask_of(PLAN[:size])]))
    assert set(parsed["scan_rows"]) == set(range(n))

    timed_out = executor.execute([{"query": "q", "triple_patterns": PATTERNS, "plan": PLAN, "timeout_s": 1e-6}])[0]
    assert timed_out["censored"] and timed_out["latency_s"] == 1e-6
    assert parse_execution(timed_out["tree"], PLAN, PATTERNS)["prefix_rows"] == {}


def test_failed_qlever_responses_are_censored_at_their_timeout():
    from src.supervised_value_estimation.amortized_dp.online.executor import _interpret

    tree = _real_tree()
    raw = {"success": False, "error": json.dumps({"runtimeInformation": tree, "time": {"total": 1500}})}
    result = _interpret(raw, timeout_s=2.0)
    assert result["censored"] and result["latency_s"] == 2.0 and result["tree"] == tree
    garbage = _interpret({"success": False, "error": "<html>502</html>"}, timeout_s=3.0)
    assert garbage["censored"] and garbage["tree"] == {} and garbage["latency_s"] == 3.0
    ok = _interpret({"success": True, "runtime_info": {"query_execution_tree": tree}, "time_total": "6ms"}, 60)
    assert not ok["censored"] and ok["latency_s"] == pytest.approx(0.006)
