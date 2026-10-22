"""Turn a QLever runtime tree for a forced left-deep plan into per-prefix observations.

For a plan o1, ..., on forced with nested groups (QLeverOptimizerClient._apply_join_order),
QLever's execution tree contains one join per prefix: the join building {o1, o2}, then the
one building {o1, o2, o3}, and so on, possibly with Sort nodes in between. A post-order
walk visits them innermost first, so the k-th join found builds the (k+1)-pattern prefix.

Per prefix this yields
    rows      exact cardinality of the intermediate result      (if the join completed)
    step_ms   time added by that join step: the join's subtree total time minus the previous
              join's, so it includes the new pattern's scan and any sort in between
and per pattern the scan's row count, matched to the pattern by its terms, and what the scan read (scan_read).
A scan of at least lazy-index-scan-max-size-materialization (1M) estimated rows is read lazily: joined with a
small side, QLever reads only the index blocks that can match (details num-blocks-read < num-blocks-all), and its
result_rows is then what it read, not the pattern's cardinality. Such a scan gives no row count, only its read
elements (num-elements-read); scans read in full give both.

Robustness rules (each observation is dropped, never guessed):
  * joins are nodes whose description starts with one of JOIN_PREFIXES; the existing
    _walk_tree only counted "Join", so a MultiColumnJoin (two shared variables) would shift
    every later cardinality onto the wrong prefix. If the number of joins is not n - 1, no
    per-prefix observation is used at all.
  * a node counts as finished if its status says completed or materialized; cardinalities of
    unfinished joins (a timeout cut them off) are not used.
  * a subtree that contains a node served from QLever's result cache has meaningless times,
    so step times through it are not used. Cardinalities from cached nodes are still exact.
"""
from __future__ import annotations

import re

JOIN_PREFIXES = ("Join", "MultiColumnJoin")
SCAN_PREFIXES = ("IndexScan", "Scan")
TERM_PATTERN = re.compile(r'\?[a-zA-Z0-9_]+|<[^>]+>|"(?:[^"\\]|\\.)*"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?')
UNCACHED = ("", "computed")


def _finished(node) -> bool:
    status = str(node.get("status", "")).lower()
    return "completed" in status or "materialized" in status


def _cached(node) -> bool:
    return str(node.get("cache_status", "")).lower() not in UNCACHED


def _collect(node, joins, scans):
    """Post-order walk; returns whether this subtree contains a cached node."""
    if not node:
        return False
    subtree_cached = _cached(node)
    for child in node.get("children", []) or []:
        subtree_cached |= _collect(child, joins, scans)
    description = str(node.get("description", ""))
    if description.startswith(JOIN_PREFIXES):
        joins.append((node, subtree_cached))
    elif description.startswith(SCAN_PREFIXES):
        scans.append(node)
    return subtree_cached


def _scan_counts(node):
    """(rows, read) of a finished scan: rows the pattern's cardinality, None when the scan skipped blocks; read
    the elements it read (num-elements-read, else its rows)."""
    if not _finished(node) or node.get("result_rows") is None:
        return None, None
    details = node.get("details") or {}
    blocks_read, blocks_all = details.get("num-blocks-read"), details.get("num-blocks-all")
    skipped = blocks_read is not None and blocks_all is not None and blocks_read < blocks_all
    return (None if skipped else int(node["result_rows"])), int(details.get("num-elements-read", node["result_rows"]))


def _match_scan(description, pattern_terms):
    """Index of the one pattern all of whose terms appear in the scan description."""
    terms = set(TERM_PATTERN.findall(description))
    hits = [i for i, pattern in enumerate(pattern_terms) if pattern and set(pattern) <= terms]
    return hits[0] if len(hits) == 1 else None


def parse_execution(tree, plan, triple_patterns):
    """Observations from one executed plan; see the module docstring for what is kept."""
    joins, scans = [], []
    _collect(tree, joins, scans)
    n = len(plan)
    aligned = len(joins) == n - 1
    prefix_rows, step_ms = {}, {}
    if aligned:
        previous_total = 0.0
        previous_timing_valid = True
        for k, (node, subtree_cached) in enumerate(joins, start=2):
            finished = _finished(node)
            if finished and node.get("result_rows") is not None:
                prefix_rows[k] = int(node["result_rows"])
            total = node.get("total_time")
            timing_valid = finished and not subtree_cached and total is not None and previous_timing_valid
            if timing_valid:
                step_ms[k] = max(float(total) - previous_total, 0.0)
                previous_total = float(total)
            previous_timing_valid = timing_valid
    pattern_terms = [TERM_PATTERN.findall(pattern)[:3] for pattern in triple_patterns]
    scan_rows, scan_read = {}, {}
    for node in scans:
        index = _match_scan(str(node.get("description", "")), pattern_terms)
        rows, read = _scan_counts(node)
        if index is not None and read is not None:
            scan_read[index] = read
            if rows is not None:
                scan_rows[index] = rows
    return {
        "aligned": aligned,
        "n_join_nodes": len(joins),
        "prefix_rows": prefix_rows,          # prefix length k -> rows of the first k patterns
        "step_ms": step_ms,                  # prefix length k -> time of the step building it
        "scan_rows": scan_rows,              # pattern index -> rows of its scan (not for scans that skipped blocks)
        "scan_read": scan_read,              # pattern index -> elements its scan read
        "any_cached": any(cached for _, cached in joins),
    }


# --- bushy plans ---------------------------------------------------------------------------

def bushy_sparql(tree, triple_patterns) -> str:
    """SPARQL forcing a bushy plan tree (bushy.py): every subtree is its own group, so QLever
    plans each group separately and joins them, e.g. ((0, 1), (2, 3)) ->
    { { { tp0 } { tp1 } } { { tp2 } { tp3 } } }. A leaf is { tp }."""
    def group(node, depth):
        pad = "  " * depth
        if isinstance(node, int):
            return f"{pad}{{ {triple_patterns[node]} }}"
        return f"{pad}{{\n{group(node[0], depth + 1)}\n{group(node[1], depth + 1)}\n{pad}}}"
    return "SELECT * WHERE {\n" + group(tree, 1) + "\n}"


def parse_bushy_execution(tree, plan_masks, triple_patterns):
    """Observations from one executed bushy plan (forced with bushy_sparql). `plan_masks` are the
    pattern masks of the plan's join results (bushy.internal_masks).

    Every join node's pattern set is the union of the scans below it (matched to patterns by their
    terms), so no assumption on child order or on Sort nodes in between is needed. Per join:
        rows   exact cardinality of its result                           (if the join completed)
        step   time of the merge: the join's subtree total time minus the totals of the nearest
               joins below each operand, so it includes the scans of single-pattern operands and
               any sort in between (as the left-deep step times do); the steps of a plan add up
               to its total time. Keyed by the two operand masks (smaller first).
    aligned: the join sets are exactly the plan's. Step times through cached subtrees, or through
    an operand whose own timing is invalid, are dropped; cardinalities are kept. Scans as in the module text:
    scan_rows only from scans read in full, scan_read from every finished scan."""
    pattern_terms = [TERM_PATTERN.findall(pattern)[:3] for pattern in triple_patterns]
    rows, steps, scan_rows, scan_read, joins = {}, {}, {}, {}, []
    state = {"unmatched": False, "cached": False}

    def walk(node):
        """-> (pattern mask, nearest join node info or None, subtree cached)."""
        children = [walk(child) for child in (node.get("children", []) or [])]
        cached = _cached(node) or any(c[2] for c in children)
        description = str(node.get("description", ""))
        if description.startswith(SCAN_PREFIXES):
            index = _match_scan(description, pattern_terms)
            if index is None:
                state["unmatched"] = True
                return 0, None, cached
            scan_count, read = _scan_counts(node)
            if read is not None:
                scan_read[index] = read
                if scan_count is not None:
                    scan_rows[index] = scan_count
            return 1 << index, None, cached
        mask = 0
        for child_mask, _, _ in children:
            mask |= child_mask
        if description.startswith(JOIN_PREFIXES):
            finished = _finished(node)
            total = node.get("total_time")
            operands = [c[0] for c in children]
            below = [c[1] for c in children]
            timing = (finished and not cached and total is not None and len(operands) == 2
                      and all(b is None or b["timing"] for b in below))
            info = {"mask": mask, "timing": timing, "total": float(total) if total is not None else None}
            if finished and node.get("result_rows") is not None:
                rows[mask] = int(node["result_rows"])
            if timing:
                key = tuple(sorted(operands))
                steps[key] = max(info["total"] - sum(b["total"] for b in below if b is not None), 0.0)
            joins.append(mask)
            state["cached"] |= cached
            return mask, info, cached
        nearest = next((c[1] for c in children if c[1] is not None), None) if len(children) == 1 else None
        return mask, nearest, cached

    if tree:
        walk(tree)
    aligned = not state["unmatched"] and sorted(joins) == sorted(plan_masks)
    return {"aligned": aligned, "n_join_nodes": len(joins), "rows": rows, "steps": steps,
            "scan_rows": scan_rows, "scan_read": scan_read, "any_cached": state["cached"]}
