import itertools
import numpy as np
import math
import random

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.baselines.enumeration import JoinOrderEnumerator
from src.supervised_value_estimation.amortized_dp.agents import AmortizedDPAgent, state_tensor
from src.supervised_value_estimation.amortized_dp.labels import (
    NEG_INF,
    QueryLabels,
    cost_to_go,
    logsumexp,
    mask_of,
    members,
    optimal_left_deep_plan,
    plan_log_cost,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.simulated_execution import SimulatedCostExecutionStrategy
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search


def _random_connected_graph(n, extra_edges, rng):
    edges = {(rng.randrange(i), i) for i in range(1, n)}              # random spanning tree
    while len(edges) < n - 1 + extra_edges and len(edges) < n * (n - 1) // 2:
        i, j = sorted(rng.sample(range(n), 2))
        edges.add((i, j))
    return edges


def _synthetic_labels(n, extra_edges=2, seed=0):
    rng = random.Random(seed)
    edges = _random_connected_graph(n, extra_edges, rng)
    neighbour_masks = [0] * n
    for i, j in edges:
        neighbour_masks[i] |= 1 << j
        neighbour_masks[j] |= 1 << i
    labels = QueryLabels(query=f"q{seed}", family="complex", n_tp=n, neighbour_masks=neighbour_masks, logcard={})
    # Connected subsets by flood fill from every singleton.
    frontier = [1 << i for i in range(n)]
    seen = set(frontier)
    while frontier:
        mask = frontier.pop()
        for child in labels.children(mask):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    labels.logcard = {mask: rng.uniform(0.0, 12.0) for mask in seen}
    return labels, edges


def _all_left_deep_orders(labels):
    for order in itertools.permutations(range(labels.n_tp)):
        if all(mask_of(order[:k]) in labels.logcard for k in range(1, labels.n_tp + 1)):
            yield list(order)


@pytest.mark.parametrize("seed", range(6))
def test_dp_optimum_matches_brute_force_and_the_existing_enumerator(seed):
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=seed)
    order, log_cost = optimal_left_deep_plan(labels)
    brute_force = min(plan_log_cost(o, labels) for o in _all_left_deep_orders(labels))
    assert log_cost == pytest.approx(brute_force, abs=1e-9)
    assert plan_log_cost(order, labels) == pytest.approx(log_cost, abs=1e-9)

    adjacency = {i: members(labels.neighbour_masks[i]) for i in range(labels.n_tp)}
    enumerator = JoinOrderEnumerator(adjacency, lambda key: labels.logcard[mask_of(key)], labels.n_tp,
                                     cardinality_is_log_scale=True)
    _, left_deep = enumerator.search()
    assert math.log(left_deep.cost) == pytest.approx(log_cost, rel=1e-9)


def test_cost_to_go_is_the_cheapest_completion():
    labels, _ = _synthetic_labels(5, extra_edges=2, seed=11)
    log_g = cost_to_go(labels)
    for mask in labels.logcard:
        if mask == labels.full_mask:
            assert log_g[mask] == NEG_INF
            continue
        completions = [o for o in _all_left_deep_orders(labels) if mask_of(o[:mask.bit_count()]) == mask]
        best = min(logsumexp([labels.logcard[mask_of(o[:k])] for k in range(mask.bit_count() + 1, labels.n_tp + 1)])
                   for o in completions)
        assert log_g[mask] == pytest.approx(best, abs=1e-9)


EDGE_DIM = 7
LAYERS = ["mean", "gine", "gat"]


def _random_model_inputs(n=6, batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(batch, n, 16, generator=generator)
    adjacency = torch.rand(batch, n, n, generator=generator) > 0.5
    adjacency = adjacency | adjacency.transpose(1, 2)
    adjacency &= ~torch.eye(n, dtype=torch.bool)
    state = torch.rand(batch, n, generator=generator) > 0.5
    edges = torch.randint(0, 3, (batch, n, n, EDGE_DIM), generator=generator).float()
    edges = (edges + edges.transpose(1, 2)) * adjacency.unsqueeze(-1)
    return embeddings, torch.ones(batch, n, dtype=torch.bool), adjacency, state, edges


def _model(layer, seed):
    torch.manual_seed(seed)
    return ContractedJoinGraphValueNet(embedding_dim=16, hidden_dim=32, message_layer=layer,
                                       edge_feature_dim=EDGE_DIM).eval()


@pytest.mark.parametrize("layer", LAYERS)
def test_value_net_does_not_depend_on_how_patterns_are_numbered(layer):
    model = _model(layer, 0)
    embeddings, pattern_mask, adjacency, state, edges = _random_model_inputs()
    p = torch.randperm(embeddings.shape[1])
    permuted = model(embeddings[:, p], pattern_mask[:, p], adjacency[:, p][:, :, p], state[:, p],
                     edges[:, p][:, :, p])
    original = model(embeddings, pattern_mask, adjacency, state, edges)
    for a, b in zip(original, permuted):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("layer", LAYERS)
def test_card_head_depends_only_on_the_joined_set(layer):
    model = _model(layer, 1)
    embeddings, pattern_mask, adjacency, state, edges = _random_model_inputs(seed=3)
    changed = embeddings.clone()
    changed[~state] = torch.randn_like(changed[~state])            # only patterns outside S
    card, cost_to_go_value = model(embeddings, pattern_mask, adjacency, state, edges)
    card_changed, cost_to_go_changed = model(changed, pattern_mask, adjacency, state, edges)
    assert torch.allclose(card, card_changed, atol=1e-6)
    assert not torch.allclose(cost_to_go_value, cost_to_go_changed)   # the future does depend on them


@pytest.mark.parametrize("layer", LAYERS)
def test_padding_patterns_are_ignored(layer):
    model = _model(layer, 2)
    embeddings, pattern_mask, adjacency, state, edges = _random_model_inputs(n=5, batch=2, seed=4)
    padded = [torch.cat([embeddings, torch.randn(2, 3, 16)], dim=1),
              torch.cat([pattern_mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1),
              torch.nn.functional.pad(adjacency, (0, 3, 0, 3), value=True),
              torch.cat([state, torch.ones(2, 3, dtype=torch.bool)], dim=1),
              torch.nn.functional.pad(edges, (0, 0, 0, 3, 0, 3), value=1.0)]
    for a, b in zip(model(embeddings, pattern_mask, adjacency, state, edges), model(*padded)):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("layer", ["gine", "gat"])
def test_edge_features_reach_the_cost_to_go(layer):
    model = _model(layer, 5)
    embeddings, pattern_mask, adjacency, state, edges = _random_model_inputs(seed=6)
    _, with_edges = model(embeddings, pattern_mask, adjacency, state, edges)
    _, other_edges = model(embeddings, pattern_mask, adjacency, state, edges.flip(-1))
    assert not torch.allclose(with_edges, other_edges)


def test_mean_layer_keeps_the_original_parameter_names():
    names = set(ContractedJoinGraphValueNet(embedding_dim=16, hidden_dim=32).state_dict())
    assert "message_layers.0.0.weight" in names and not any("edge" in n for n in names)


def test_state_tensor_decodes_bitmasks():
    assert state_tensor([0b101, 0b010], 3, torch.device("cpu")).tolist() == [[True, False, True],
                                                                            [False, True, False]]


class _OracleAgent(AmortizedDPAgent):
    """AmortizedDPAgent with exact cardinalities and cost-to-go instead of a network."""

    def __init__(self, labels):
        self.labels = labels
        self.log_g = cost_to_go(labels)

    def setup_episode(self, query):
        n = self.labels.n_tp
        return {"n_tp": n, "full_mask": (1 << n) - 1, "logcard": dict(self.labels.logcard),
                "log_g": dict(self.log_g)}

    def _evaluate(self, episode, masks):
        pass


def _synthetic_query(labels, edges):
    # Give every edge its own shared variable, so beam_search's "no cartesian product"
    # check reproduces exactly the adjacency of the labels.
    terms = [[] for _ in range(labels.n_tp)]
    for index, (i, j) in enumerate(sorted(edges)):
        terms[i].append(f"?e{index}")
        terms[j].append(f"?e{index}")
    patterns = [f"?s{i} <http://p/{'_'.join(t[1:] for t in terms[i])}> {' '.join(terms[i])} ."
                for i in range(labels.n_tp)]
    data = Data(x=torch.zeros(1, 1), triple_patterns=patterns, query=labels.query)
    return Batch.from_data_list([data])


@pytest.mark.parametrize("seed", range(5))
def test_greedy_beam_search_with_exact_values_finds_the_dp_optimum(seed):
    labels, edges = _synthetic_labels(7, extra_edges=3, seed=seed + 20)
    top = beam_search(_synthetic_query(labels, edges), _OracleAgent(labels), beam_width=1)
    strategy = SimulatedCostExecutionStrategy({labels.query: labels})
    result = strategy.score(labels.query, top[0]["plan"])
    assert result["log_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert top[0]["cost"] == pytest.approx(result["log_optimal_cost"], abs=1e-9)


def _string_estimator(labels, patterns):
    """G-Care-style estimator: sub-query strings in, exact cardinalities out."""
    index = {pattern: i for i, pattern in enumerate(patterns)}

    def estimate(sub_queries):
        cards = []
        for sub_query in sub_queries:
            members_in_query = [index[line.strip()] for line in sub_query.splitlines()[1:-1]]
            cards.append(math.exp(labels.logcard[mask_of(members_in_query)]))
        return cards
    return estimate


@pytest.mark.parametrize("seed", range(4))
def test_cardinality_agent_ranks_by_prefix_c_out_and_finds_the_optimum_with_exact_cards(seed):
    from src.supervised_value_estimation.agents.CardinalityEstimatorAgent import CardinalityEstimatorValidationAgent

    labels, edges = _synthetic_labels(7, extra_edges=3, seed=seed + 40)
    query = _synthetic_query(labels, edges)
    patterns = list(query.to_data_list()[0].triple_patterns)
    agent = CardinalityEstimatorValidationAgent(estimator_fn=_string_estimator(labels, patterns),
                                                estimator_requires_features=False)
    top = beam_search(query, agent, beam_width=4)
    for candidate in top:
        # The score is the plan's C_out, not the newest intermediate's cardinality...
        assert math.log(candidate["cost"]) == pytest.approx(plan_log_cost(candidate["plan"], labels), rel=1e-9)
    # ...so complete plans no longer all tie at the last depth.
    assert len({round(c["cost"], 6) for c in top}) == len(top)


# --- partial observation (offline simulation of learning from executed plans) ------------

from src.supervised_value_estimation.amortized_dp.partial_observation import (  # noqa: E402
    best_stitched_log_cost, deviation_plans, observe, random_plan,
)
from src.supervised_value_estimation.amortized_dp.train_amortized_dp import StateTables  # noqa: E402


def _complete_graph_labels(n, logcard):
    full = (1 << n) - 1
    neighbour_masks = [full & ~(1 << i) for i in range(n)]
    return QueryLabels(query="complete", family="star", n_tp=n, neighbour_masks=neighbour_masks, logcard=logcard)


def test_observe_reveals_scans_and_every_intermediate_only():
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=50)
    plan = optimal_left_deep_plan(labels)[0]
    table = {}
    observe(plan, labels, table)
    expected = {1 << p for p in plan} | {mask_of(plan[:k]) for k in range(2, len(plan) + 1)}
    assert set(table) == expected
    assert all(table[m] == labels.logcard[m] for m in table)


def test_stitching_finds_a_cheaper_plan_nobody_executed():
    n = 5
    rng = random.Random(0)
    masks = [m for m in range(1, 1 << n)]
    logcard = {m: 10.0 + rng.random() for m in masks}
    # Make the stitched plan [0, 1, 2, 4, 3] cheap: it reuses {0,1},{0,1,2} from A and
    # {0,1,2,4} from B, but neither A nor B is that plan.
    for cheap in (mask_of([0, 1]), mask_of([0, 1, 2]), mask_of([0, 1, 2, 4])):
        logcard[cheap] = 1.0
    labels = _complete_graph_labels(n, logcard)
    plan_a, plan_b = [0, 1, 2, 3, 4], [2, 1, 0, 4, 3]
    table = {}
    for plan in (plan_a, plan_b):
        observe(plan, labels, table)
    stitched = best_stitched_log_cost(labels, table)
    assert stitched < min(plan_log_cost(plan_a, labels), plan_log_cost(plan_b, labels)) - 1e-9
    assert stitched == pytest.approx(plan_log_cost([0, 1, 2, 4, 3], labels), abs=1e-9)


def test_stitching_with_everything_observed_is_the_exact_optimum():
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=51)
    assert best_stitched_log_cost(labels, dict(labels.logcard)) == pytest.approx(
        optimal_left_deep_plan(labels)[1], abs=1e-9)


def test_deviation_plans_start_with_greedy_and_are_distinct():
    labels, edges = _synthetic_labels(7, extra_edges=3, seed=52)
    query = _synthetic_query(labels, edges)
    agent = _OracleAgent(labels)
    plans = deviation_plans(agent, query, labels, 4, "closest", np.random.default_rng(0))
    assert plans[0] == beam_search(query, agent, beam_width=1)[0]["plan"]
    assert len(plans) == len({tuple(p) for p in plans}) and 2 <= len(plans) <= 4
    assert all(sorted(p) == list(range(labels.n_tp)) for p in plans)


def test_random_plans_are_complete_and_connected():
    labels, _ = _synthetic_labels(7, extra_edges=2, seed=53)
    rng = np.random.default_rng(1)
    for _ in range(20):
        plan = random_plan(labels, rng)
        assert sorted(plan) == list(range(labels.n_tp))
        assert all(mask_of(plan[:k]) in labels.logcard for k in range(1, labels.n_tp + 1))


def test_observed_state_tables_use_observed_entries_only():
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=54)
    table = {}
    observe(optimal_left_deep_plan(labels)[0], labels, table)
    tables = StateTables([labels], [torch.zeros(labels.n_tp, 4)], torch.device("cpu"),
                         observed_logcards=[table], cost_to_go_upper_bound=True)
    assert set(tables.rows["mask"].tolist()) <= set(table)
    assert tables.cost_to_go_upper_bound and tables.n_states >= 1


def test_two_pattern_queries_keep_their_root_state_and_stitch():
    labels = _complete_graph_labels(2, {1: 3.0, 2: 4.0, 3: 5.0})
    table = {}
    observe([0, 1], labels, table)
    assert best_stitched_log_cost(labels, table) == pytest.approx(plan_log_cost([0, 1], labels))
    for observed in (None, [table]):
        tables = StateTables([labels], [torch.zeros(2, 4)], torch.device("cpu"), observed_logcards=observed)
        assert tables.n_states == 1 and tables.rows["root"].all()
