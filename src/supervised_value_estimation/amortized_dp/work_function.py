"""Engine work of a bushy plan tree (bushy.py), from the cardinalities of its subsets.

C_out (bushy.tree_join_cout) is engine-agnostic: the sum of the join results' sizes. What an engine actually does
for a join depends on how it executes it; `qlever_work` instantiates that for QLever (rules read from QLever's
source at the image commit f35a290, engine/Join.cpp, engine/IndexScan.cpp, index/CompressedRelation.cpp):

    scans      A scan whose size is below `lazy_rows` (runtime parameter lazy-index-scan-max-size-materialization,
               1M) is materialized in full and reads all its rows. A larger scan directly under a join is read
               lazily: only the index blocks (`block_rows` rows each: 250 kB per column, ~31,250 rows) whose ID range
               holds a join value of the other operand are read (getBlocksForJoin). Rows read are the measured
               `num-elements-read` when given (`read`), else the expected rows of the blocks hit by k distinct join
               values spread over B blocks, size * (1 - (1 - 1/B)^k), with k = the other operand's rows (an upper
               bound on its distinct join values).
    joins      A zipper (merge) join reads both inputs: l + r rows (a lazy scan counts with its rows read), and
               writes its result (C_out). (QLever gallops between two materialized inputs of very different sizes;
               on the plans of work_2026-11-04/work_function_check.py its fitted weight was 0, so it is left out.)
    sorts      Joins work on inputs sorted by their join variables. A scan can be read in a permutation sorted by any
               of its variables; a join result comes sorted by its own join variables, so an operand that is a join
               result on other join variables is sorted first: n log2 n for n rows (Sort).

`qlever_work` returns these components in rows; `weighted` combines them with per-row costs (e.g. fitted to QLever's
execution times), so C_out is the special case {"out": 1}.

The PLANNING cost (QLeverCost, for the forest value planner's `cost: qlever`): in a bushy plan every pattern is a leaf
directly under a join, so a pattern of at least lazy_rows rows is always read lazily, with rows read r(S, p) that
depend on the other operand S of its join; the full scans and the merge reads of full scans are the same for every
plan, and the merge reads of join results add up to C_out minus the final result. Up to plan-invariant terms the
work of a plan is then a sum over its joins U = A |><| B of

    c(U) = out * (1 + |U|)  +  read * r(other, X) for an operand X that is a lazily read pattern
                            +  sort * n log2 n    for an operand X that is a join result sorted first (n = |X|)

in rows of C_out (log1p space as the counts: 1 + |U|). The defaults are the per-row weights fitted on those plans
without galloping (measured reads; ns per row: out 14.9, lazy read 14.6, sort 4.9 per n log2 n), relative to out.
`join_work`, `plan_work` and `WorkStitchTable` give the true cost from counts and measured reads; the planner's
estimate is forest_value.join_cost.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.supervised_value_estimation.amortized_dp.bushy import NEG_INF, StitchTable, Tree, is_leaf, tree_mask

VARIABLE = re.compile(r"\?[A-Za-z0-9_]+")
COMPONENTS = ("scan_full", "scan_lazy", "merge", "out", "sort")


@dataclass(frozen=True)
class QLeverParameters:
    lazy_rows: int = 1_000_000
    block_rows: int = 31_250
    sorts: bool = True


def pattern_variables(triple_patterns: Sequence[str]) -> list[frozenset]:
    return [frozenset(VARIABLE.findall(p)) for p in triple_patterns]


def expected_rows_read(size: int, distinct_keys: int, block_rows: int) -> float:
    """Expected rows of the blocks hit by `distinct_keys` uniformly spread join values in a scan of `size` rows."""
    blocks = max(1, math.ceil(size / block_rows))
    if distinct_keys <= 0:
        return 0.0
    return size * -math.expm1(distinct_keys * math.log1p(-1.0 / blocks)) if blocks > 1 else float(size)


def qlever_work(tree: Tree, rows: Mapping[int, int], sizes: Sequence[int], variables: Sequence[frozenset],
                parameters: QLeverParameters = QLeverParameters(), read: Mapping[int, float] | None = None):
    """Work components (rows) of executing `tree` on QLever; None if a needed count is missing.

    rows: cardinality per join result mask; sizes: rows per pattern; variables: per pattern (pattern_variables);
    read: measured rows read per lazily scanned pattern (leaf index), else they are estimated."""
    work = dict.fromkeys(COMPONENTS, 0.0)
    p = parameters

    def card(node):
        return sizes[node] if is_leaf(node) else rows.get(tree_mask(node))

    def node_variables(node):
        if is_leaf(node):
            return variables[node]
        return node_variables(node[0]) | node_variables(node[1])

    def visit(node):
        """Returns (rows, join variables the result is sorted by; None for a scan), or None if a count is missing."""
        if is_leaf(node):
            return sizes[node], None
        children = [visit(node[0]), visit(node[1])]
        out = card(node)
        if out is None or None in children:
            return None
        join_variables = node_variables(node[0]) & node_variables(node[1])
        for child, other, (n, sorted_by) in ((node[0], node[1], children[0]), (node[1], node[0], children[1])):
            if is_leaf(child) and n >= p.lazy_rows:
                k = card(other) if not is_leaf(other) else sizes[other]
                r = read.get(child) if read and child in read else expected_rows_read(n, k, p.block_rows)
                work["scan_lazy"] += r
                work["merge"] += r
                continue
            if is_leaf(child):
                work["scan_full"] += n
            elif p.sorts and sorted_by != join_variables:
                work["sort"] += n * math.log2(max(n, 2))
            work["merge"] += n
        work["out"] += out
        return out, join_variables

    return work if visit(tree) is not None else None


def weighted(components: Mapping[str, float] | None, weights: Mapping[str, float]) -> float | None:
    """sum of weight * component; C_out is {"out": 1}."""
    if components is None:
        return None
    return sum(weights.get(name, 0.0) * value for name, value in components.items())


# --- the planning cost ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QLeverCost:
    """Per-join planning cost in rows of C_out (module text): weights of the result rows, the rows a lazily read
    pattern operand reads, and n log2 n of a join-result operand sorted first; patterns of at least lazy_rows rows
    are read lazily."""
    out: float = 1.0
    read: float = 0.98
    sort: float = 0.33
    lazy_rows: int = 1_000_000


def variable_masks(triple_patterns: Sequence[str]) -> list[int]:
    """Per pattern its variables as a bitmask (bit i: the i-th variable of the query in order of appearance)."""
    names: dict[str, int] = {}
    out = []
    for p in triple_patterns:
        m = 0
        for v in VARIABLE.findall(p):
            m |= 1 << names.setdefault(v, len(names))
        out.append(m)
    return out


def mask_variables(mask: int, var_masks: Sequence[int]) -> int:
    v, i = 0, 0
    while mask:
        if mask & 1:
            v |= var_masks[i]
        mask >>= 1
        i += 1
    return v


def sort_rows(n: float) -> float:
    """n log2 n of sorting n rows."""
    return n * math.log2(max(n, 2.0))


def join_work(cost: QLeverCost, out_rows: float, operands, reads: Mapping, sizes: Sequence[int]) -> float | None:
    """c(U) of one join (module text) in rows, or None if a needed read or row count is unknown. out_rows: 1 + |U|;
    operands: [(mask, rows or None, needs a sort)] of A and B; reads: {(other operand mask, pattern index): rows read
    or None}; sizes: rows per pattern."""
    total = cost.out * out_rows
    (ma, na, sa), (mb, nb, sb) = operands
    for (m, n, needs_sort), other in (((ma, na, sa), mb), ((mb, nb, sb), ma)):
        if not m & (m - 1):
            p = m.bit_length() - 1
            if sizes[p] >= cost.lazy_rows:
                r = reads.get((other, p))
                if r is None:
                    return None
                total += cost.read * r
        elif needs_sort:
            if n is None:
                return None
            total += cost.sort * sort_rows(n)
    return total


def plan_work(tree: Tree, counts: Mapping[int, int | None], sizes: Sequence[int], var_masks: Sequence[int],
              reads: Mapping, cost: QLeverCost, bound: float | None = None):
    """(true planning cost of a plan in rows: the sum of join_work over its joins; number of failed counts used),
    as train_plan_loss.plan_cost: a failed (None) count costs `bound`; cost None if a count or a read is missing."""
    imputed = 0

    def rows(m):
        nonlocal imputed
        if m not in counts or (counts[m] is None and bound is None):
            return None
        if counts[m] is None:
            imputed += 1
            return bound
        return counts[m]

    def visit(node):
        """-> (work so far, mask, rows, join variables) or None."""
        if is_leaf(node):
            return 0.0, 1 << node, sizes[node], None
        a, b = visit(node[0]), visit(node[1])
        if a is None or b is None:
            return None
        u = a[1] | b[1]
        n = rows(u)
        if n is None:
            return None
        jv = mask_variables(a[1], var_masks) & mask_variables(b[1], var_masks)
        c = join_work(cost, 1 + n, [(a[1], a[2], a[3] is not None and a[3] != jv),
                                   (b[1], b[2], b[3] is not None and b[3] != jv)], reads, sizes)
        if c is None:
            return None
        return a[0] + b[0] + c, u, n, jv

    out = visit(tree)
    return (None, imputed) if out is None else (out[0], imputed)


class WorkStitchTable(StitchTable):
    """StitchTable's stitched upper bound on G(forest) with the QLever planning cost (module text) instead of C_out:
    a join's cost depends on its split (lazily read operands, sorted operands), so the top-down DP keeps, per set,
    the cheapest build for each of the join variables its result can come sorted by. A split whose join needs a read
    that is not measured, or the rows of an uncounted operand to sort, is not used. logcard: log1p counts (as
    StitchTable); reads: {(other operand mask, pattern index): rows read or None}."""

    def __init__(self, sets: Iterable[int], logcard: dict[int, float], sizes: Sequence[int], var_masks: Sequence[int],
                 reads: Mapping, cost: QLeverCost):
        super().__init__(sets, logcard)
        self.sizes, self.var_masks, self.reads, self.cost = sizes, var_masks, reads, cost
        self.vars_of: dict[int, int] = {}

    def variables(self, m: int) -> int:
        if m not in self.vars_of:
            self.vars_of[m] = mask_variables(m, self.var_masks)
        return self.vars_of[m]

    def cost_to_go(self, forest: Sequence[int], sorted_by: Mapping[int, int] | None = None) -> float:
        """log of the stitched G(forest) in rows; -inf when complete, inf when nothing completes it. sorted_by: per
        built component (2+ patterns) the join variables its result is sorted by (a component left out: unknown,
        so it is sorted before any join that needs it)."""
        if len(forest) == 1:
            return NEG_INF
        components = set(forest)
        sorted_by = sorted_by or {}
        full = 0
        for c in forest:
            full |= c
        whole: dict[int, bool] = {}
        build: dict[int, dict] = {}
        unlisted = [c for c in forest if c & (c - 1) and c not in self.joins]

        def is_whole(u):
            if u not in whole:
                whole[u] = all(u & c in (0, c) for c in forest)
            return whole[u]

        def rows_of(u):
            return math.expm1(self.logcard[u]) if u in self.logcard else None

        def states(u):
            """{sort key (join variables; -1 for a single pattern or an unknown order): cheapest cost} of building u."""
            if u in components:
                return {sorted_by.get(u, -1): 0.0}
            if u not in build:
                best: dict[int, float] = {}
                if u in self.joins:
                    splits = [(v, w) for v, w in self.splits(u) if is_whole(v) and is_whole(w)]
                    splits += [(c, u ^ c) for c in unlisted
                               if c & ~u == 0 and (u ^ c) and ((u ^ c) in self.joins or (u ^ c) in components)
                               and is_whole(u ^ c)]
                    out_rows = math.exp(self.logcard[u])
                    for v, w in splits:
                        jv = self.variables(v) & self.variables(w)
                        for sv, cv in states(v).items():
                            for sw, cw in states(w).items():
                                c = join_work(self.cost, out_rows,
                                              [(v, rows_of(v), bool(v & (v - 1)) and sv != jv),
                                               (w, rows_of(w), bool(w & (w - 1)) and sw != jv)], self.reads, self.sizes)
                                if c is not None and cv + cw + c < best.get(jv, math.inf):
                                    best[jv] = cv + cw + c
                build[u] = best
            return build[u]

        best = states(full)
        value = min(best.values(), default=math.inf)
        return math.log(value) if value < math.inf else math.inf


def plan_internal_splits(tree: Tree):
    """(A mask, B mask) of every join of the plan, bottom-up."""
    if is_leaf(tree):
        return []
    return plan_internal_splits(tree[0]) + plan_internal_splits(tree[1]) + [(tree_mask(tree[0]), tree_mask(tree[1]))]


def lazy_splits(splits, sizes: Sequence[int], cost: QLeverCost):
    """The splits (A, B) with a lazily read pattern operand: their reads need measuring."""
    def lazy(m):
        return not m & (m - 1) and sizes[m.bit_length() - 1] >= cost.lazy_rows
    return [(a, b) for a, b in splits if lazy(a) or lazy(b)]

