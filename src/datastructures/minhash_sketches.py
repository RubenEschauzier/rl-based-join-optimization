"""Bottom-k MinHash sketches of the entity sets triple patterns bind a variable to, and the
per-variable intersection features computed from them.

A variable ?x of a query gets one set per triple pattern it occurs in:
    ?x p c   -> {x : x p c}            key "s|<p>|<c>"
    c p ?x   -> {x : c p x}            key "o|<c>|<p>"
    ?x p ?y  -> all subjects of p      key "s|<p>|*"     (first order: ?y unrestricted)
    ?y p ?x  -> all objects of p       key "o|*|<p>"
The query is empty if these sets do not intersect, also when every pair does (a hard zero).

A sketch is (exact set size, the k smallest 64-bit hashes of the set's members). Cheap to keep
in an index: a set with at most k members is its own sketch (one short scan of the POS / PSO
permutation at estimation time), only the larger ones are stored -- on YAGO 9,918 of 31.6M
(predicate, constant) pairs for k = 128, 5 MB.

From the sketches of a variable's sets A_1..A_m:
  - a hash in every sketch is a common member: the intersection is CERTAINLY non-empty;
  - with U the k smallest hashes of the union of the sketches (= of the union of the sets),
    |{h in U : h in every sketch}| / |U| estimates |A_1 ∩ .. ∩ A_m| / |A_1 ∪ .. ∪ A_m|, and
    |union| is exact when |U| < k, else (k - 1) / (k-th smallest normalised hash);
  - exact when every set has at most k members.
"""
from __future__ import annotations

import hashlib
import math

K = 128
_SCALE = float(2 ** 64)


def entity_hash(term: str) -> int:
    """64-bit hash of an entity in N3 form (<iri>), independent of any engine's ids."""
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "big")


def make_sketch(members, k=K):
    """[set size, sorted k smallest hashes] of an iterable of N3 terms."""
    hashes = sorted({entity_hash(m) for m in members})
    return [len(hashes), hashes[:k]]


def pattern_key(s, p, o, variable):
    """Sketch key of the set `variable` (an N3 term) is bound to by pattern (s, p, o), all N3."""
    s_var, o_var = s.startswith("?"), o.startswith("?")
    if s == variable and o == variable:
        return None
    if s == variable:
        return f"s|{p}|{o}" if not o_var else f"s|{p}|*"
    if o == variable:
        return f"o|{s}|{p}" if not s_var else f"o|*|{p}"
    return None


def set_query(key):
    """SPARQL query listing the members of a sketch key's set."""
    side, a, b = key.split("|", 2)
    if side == "s":
        return f"SELECT DISTINCT ?x WHERE {{ ?x {a} {'?y' if b == '*' else b} . }}"
    return f"SELECT DISTINCT ?x WHERE {{ {'?y' if a == '*' else a} {b} ?x . }}"


def _intersection(sketches, k):
    """(certainly non-empty, estimated intersection size) of sets given by their sketches."""
    hash_sets = [set(h) for _, h in sketches]
    if all(size <= len(h) for size, h in sketches):              # every sketch is its whole set
        joint = len(set.intersection(*hash_sets))
        return joint > 0, float(joint)
    certain = any(all(h in other for other in hash_sets[1:]) for h in sketches[0][1])
    union = sorted(set().union(*hash_sets))[:k]
    in_all = sum(1 for h in union if all(h in s for s in hash_sets))
    if len(union) < k:
        union_size = float(len(union))
    else:
        union_size = (k - 1) / ((union[-1] + 1) / _SCALE)
    return certain, in_all / len(union) * union_size


def variable_features(sketches, k=K):
    """[has >= 2 sets, certainly non-empty, log1p estimated joint intersection,
    log1p smallest estimated pairwise intersection] of a variable's sets."""
    if len(sketches) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    certain, joint = _intersection(sketches, k)
    pairwise = min(_intersection([sketches[i], sketches[j]], k)[1]
                   for i in range(len(sketches)) for j in range(i + 1, len(sketches)))
    return [1.0, float(certain), math.log1p(joint), math.log1p(pairwise)]
