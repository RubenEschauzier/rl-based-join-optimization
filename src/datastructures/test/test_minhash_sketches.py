import math
import random

from src.datastructures.minhash_sketches import K, make_sketch, pattern_key, set_query, variable_features


def _terms(ids):
    return [f"<http://example.com/{i}>" for i in ids]


def test_small_sets_are_exact():
    a, b, c = _terms(range(0, 50)), _terms(range(40, 90)), _terms(range(45, 120))
    f = variable_features([make_sketch(a), make_sketch(b), make_sketch(c)])
    assert f[:2] == [1.0, 1.0]
    assert math.isclose(f[2], math.log1p(5))            # 45..49
    assert math.isclose(f[3], math.log1p(5))            # a ∩ c = 45..49 is the smallest pair


def test_hard_zero_shape_pairwise_overlap_joint_empty():
    # every pair shares members, the three sets share none (large sets: sketched, not exact)
    a = _terms(list(range(0, 5000)) + list(range(10000, 15000)))
    b = _terms(list(range(0, 5000)) + list(range(20000, 25000)))
    c = _terms(list(range(10000, 15000)) + list(range(20000, 25000)))
    f = variable_features([make_sketch(a), make_sketch(b), make_sketch(c)])
    assert f[0] == 1.0 and f[1] == 0.0                   # never "certainly non-empty"
    assert f[2] == 0.0                                    # no common hash -> estimate 0
    assert f[3] > math.log1p(1000)                        # pairs clearly overlap


def test_large_intersection_estimate_and_certainty():
    rng = random.Random(0)
    common = list(range(0, 30000))
    a = _terms(common + rng.sample(range(100000, 200000), 30000))
    b = _terms(common + rng.sample(range(200000, 300000), 30000))
    f = variable_features([make_sketch(a), make_sketch(b)])
    assert f[1] == 1.0
    assert abs(math.expm1(f[2]) / 30000 - 1) < 0.5        # bottom-128 estimate within 50%


def test_single_pattern_variable_and_keys():
    assert variable_features([make_sketch(_terms(range(3)))]) == [0.0, 0.0, 0.0, 0.0]
    assert pattern_key("?x", "<p>", "<c>", "?x") == "s|<p>|<c>"
    assert pattern_key("<c>", "<p>", "?x", "?x") == "o|<c>|<p>"
    assert pattern_key("?x", "<p>", "?y", "?y") == "o|*|<p>"
    assert pattern_key("?x", "<p>", "?y", "?x") == "s|<p>|*"
    assert set_query("o|*|<p>") == "SELECT DISTINCT ?x WHERE { ?y <p> ?x . }"
    assert len(make_sketch(_terms(range(1000)))[1]) == K
