"""Cardinality labels for every connected subset of a query, and the exact DP targets.

Cost model: the same C_out cost as JoinPlan (src/baselines/enumeration.py), which is also
what the simulated plan-cost dataset is built from. A left-deep order o_1, ..., o_n costs

    card(o_1) + card({o_1, o_2}) + card({o_1, o_2, o_3}) + ... + card(all)

and, because the first join is symmetric, o_1 is taken to be the cheaper of the first two
(JoinPlan's DP does the same; beam_search only emits the first pair in one order).

Subsets are bitmasks over the query's triple patterns. Everything that can be is kept in
log space: cardinalities span many orders of magnitude.

Exact cost-to-go, the DP teacher:
    G(full) = 0
    G(S)    = min over a adjacent to S of  card(S u {a}) + G(S u {a})
    optimal = min over connected pairs {i, j} of  min(card i, card j) + card({i, j}) + G({i, j})
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import Batch

from src.baselines.enumeration import JoinOrderEnumerator
from src.query_environments.gym.query_gym_estimated_cost import QueryGymEstimatedCost

NEG_INF = -math.inf
# The same test beam_search_left_deep uses to rule out cartesian products: two patterns
# join only if they share a VARIABLE. (build_adj_list also links patterns that share a
# constant IRI, but joining on a constant alone is a cartesian product in SPARQL.)
VARIABLE_PATTERN = re.compile(r"[\?\$]\w+")


def mask_of(indices) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def members(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def logsumexp(values) -> float:
    values = [v for v in values if v != NEG_INF]
    if not values:
        return NEG_INF
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


@dataclass
class QueryLabels:
    """Log-cardinalities of every connected subset of one query, plus its join graph."""
    query: str
    family: str
    n_tp: int
    neighbour_masks: list[int]                       # per triple pattern, its adjacent patterns
    logcard: dict[int, float]                        # oracle log card per connected subset
    estimated_logcard: dict[int, float] = field(default_factory=dict)  # optional second model

    @property
    def full_mask(self) -> int:
        return (1 << self.n_tp) - 1

    def neighbours(self, mask: int) -> int:
        reach = 0
        for i in members(mask):
            reach |= self.neighbour_masks[i]
        return reach & ~mask

    def children(self, mask: int) -> list[int]:
        """Connected supersets reachable by joining one more adjacent pattern."""
        reach = self.neighbours(mask)
        return [mask | (1 << a) for a in range(self.n_tp) if reach >> a & 1]

    @property
    def is_connected(self) -> bool:
        """Whether the whole query can be joined without a cartesian product."""
        return self.full_mask in self.logcard

    def connected_pairs(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.n_tp) for j in range(i + 1, self.n_tp)
                if self.neighbour_masks[i] >> j & 1]


def family_of(query_type: str) -> str:
    """Coarse query family from the source file name stored in Data.type."""
    name = str(query_type).lower()
    if "star" in name:
        return "star"
    if "path" in name:
        return "path"
    if "complex" in name:
        return "complex"
    return "other"


# Terms of a triple pattern, in order: variables, IRIs, and literals (with an optional
# language tag or datatype). Same grammar as build_adj_list in src/baselines/enumeration.py.
TERM_PATTERN = re.compile(r'\?[a-zA-Z0-9_]+|<[^>]+>|"(?:[^"\\]|\\.)*"(?:@[a-zA-Z-]+|\^\^<[^>]+>)?')
# Unordered pairs of positions (0 = subject, 1 = predicate, 2 = object) a shared variable
# can take in the two patterns: ss, sp, so, pp, po, oo.
POSITION_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
JOIN_FEATURE_DIM = len(POSITION_PAIRS) + 1


def join_position_features(triple_patterns) -> np.ndarray:
    """(n, n, 7) edge features: per pair of patterns, how many shared variables join them in
    each position combination (subject-subject, subject-object, ...) plus the total count.
    Join position matters for cardinality: an object-subject (path) join and a
    subject-subject (star) join on the same variable behave very differently."""
    positions = []
    for pattern in triple_patterns:
        terms = TERM_PATTERN.findall(pattern)[:3]
        variable_positions = {}
        for position, term in enumerate(terms):
            if VARIABLE_PATTERN.fullmatch(term):
                variable_positions.setdefault(term, []).append(position)
        positions.append(variable_positions)
    n = len(triple_patterns)
    features = np.zeros((n, n, JOIN_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for variable in positions[i].keys() & positions[j].keys():
                features[i, j, -1] += 1
                for a in positions[i][variable]:
                    for b in positions[j][variable]:
                        features[i, j, POSITION_PAIRS.index(tuple(sorted((a, b))))] += 1
    return features


def variable_adjacency(triple_patterns) -> dict[int, list[int]]:
    variables = [set(VARIABLE_PATTERN.findall(pattern)) for pattern in triple_patterns]
    return {i: [j for j in range(len(variables)) if j != i and variables[i] & variables[j]]
            for i in range(len(variables))}


def connected_subset_masks(query) -> tuple[list[int], list[int]]:
    """All variable-connected subsets (as bitmasks) and each pattern's neighbour bitmask."""
    n_tp = len(query.triple_patterns)
    adjacency = variable_adjacency(query.triple_patterns)
    subsets = JoinOrderEnumerator(adjacency, lambda key: 0.0, n_tp).enumerate_csg(n_tp)
    masks = sorted({mask_of(subset) for subset in subsets})
    neighbour_masks = [mask_of(adjacency[i]) for i in range(n_tp)]
    return masks, neighbour_masks


@torch.no_grad()
def model_log_cardinalities(model, query, masks, device, chunk_size=1024) -> list[float]:
    """Log-cardinality from a GNN cardinality model for each subset, in batched forwards.

    Sub-queries are built exactly as OrderDynamicProgramming.predict_cardinality builds
    them (QueryGymEstimatedCost.reduced_form_query), so batched and single-query estimates
    are identical.
    """
    if getattr(query, "batch", None) is None:
        query.batch = torch.zeros(query.x.shape[0], dtype=torch.long)
    values = []
    for start in range(0, len(masks), chunk_size):
        chunk = masks[start:start + chunk_size]
        sub_queries = [QueryGymEstimatedCost.reduced_form_query(query, members(mask), mask.bit_count())
                       for mask in chunk]
        batch = Batch.from_data_list(sub_queries).to(device)
        output = model.forward(x=batch.x, edge_index=batch.edge_index,
                               edge_attr=batch.edge_attr, batch=batch.batch)
        cardinality = next(head["output"] for head in output if head["output_type"] == "cardinality")
        values.extend(cardinality.reshape(-1).double().cpu().tolist())
    return values


def label_query(query, oracle, device, estimator=None) -> QueryLabels:
    masks, neighbour_masks = connected_subset_masks(query)
    labels = QueryLabels(
        query=query.query if isinstance(query.query, str) else query.query[0],
        family=family_of(query.type if isinstance(query.type, str) else query.type[0]),
        n_tp=len(query.triple_patterns),
        neighbour_masks=neighbour_masks,
        logcard=dict(zip(masks, model_log_cardinalities(oracle, query, masks, device))),
    )
    if estimator is not None:
        labels.estimated_logcard = dict(zip(masks, model_log_cardinalities(estimator, query, masks, device)))
    return labels


def cost_to_go(labels: QueryLabels, logcard: dict[int, float] | None = None) -> dict[int, float]:
    """log G(S) for every connected S; G(full) = 0 is represented as -inf (log 0)."""
    logcard = labels.logcard if logcard is None else logcard
    log_g = {labels.full_mask: NEG_INF}
    for mask in sorted(logcard, key=lambda m: -m.bit_count()):
        if mask == labels.full_mask:
            continue
        options = [logsumexp([logcard[child], log_g[child]]) for child in labels.children(mask)
                   if child in log_g]
        log_g[mask] = min(options) if options else math.inf   # inf: dead end (disconnected rest)
    return log_g


def plan_log_cost(order, labels: QueryLabels, logcard: dict[int, float] | None = None) -> float:
    """log C_out of a left-deep order, first join taken in its cheaper orientation."""
    logcard = labels.logcard if logcard is None else logcard
    order = [int(i) for i in order]
    if len(order) == 1:
        return logcard[mask_of(order)]
    terms = [min(logcard[mask_of(order[:1])], logcard[mask_of(order[1:2])])]
    for size in range(2, len(order) + 1):
        terms.append(logcard[mask_of(order[:size])])
    return logsumexp(terms)


def optimal_left_deep_plan(labels: QueryLabels, logcard: dict[int, float] | None = None
                           ) -> tuple[list[int], float]:
    """Exact DP: the cheapest left-deep order and its log cost, under `logcard`."""
    logcard = labels.logcard if logcard is None else logcard
    log_g = cost_to_go(labels, logcard)
    if labels.n_tp == 1:
        return [0], logcard[1]
    best_pair, best_cost = None, math.inf
    for i, j in labels.connected_pairs():
        pair = mask_of((i, j))
        cost = logsumexp([min(logcard[1 << i], logcard[1 << j]), logcard[pair], log_g[pair]])
        if cost < best_cost:
            best_pair, best_cost = (i, j) if logcard[1 << i] <= logcard[1 << j] else (j, i), cost
    order, mask = list(best_pair), mask_of(best_pair)
    while mask != labels.full_mask:
        mask = min(labels.children(mask), key=lambda child: logsumexp([logcard[child], log_g[child]]))
        order.append((mask ^ mask_of(order)).bit_length() - 1)
    return order, best_cost
