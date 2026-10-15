import math
import random

import numpy as np
import torch

from src.baselines.enumeration import connected_subgraphs
from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.bushy_model import ForestValueNet
from src.supervised_value_estimation.amortized_dp.bushy_train import (
    ForestTables, _STATE, _state_rows, beam_plan, count_connected_subsets, greedy_plan, sample_states, training_loss,
)
from src.supervised_value_estimation.amortized_dp.labels import JOIN_FEATURE_DIM
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _synthetic_labels


def _patterns(labels):
    """Triple-pattern strings whose shared variables give labels' join graph (for the join-position features)."""
    n = labels.n_tp
    patterns = []
    for i in range(n):
        neighbours = [j for j in range(n) if labels.neighbour_masks[i] >> j & 1]
        terms = [f"?v{min(i, j)}_{max(i, j)}" for j in neighbours][:2] or [f"?solo{i}"]
        subject, obj = terms[0], terms[1] if len(terms) > 1 else f"<http://example.com/{i}>"
        patterns.append(f"{subject} <http://example.com/p{i}> {obj} .")
    return patterns


def test_capped_subset_count_matches_the_enumeration():
    for seed in range(20):
        labels, _ = _synthetic_labels(7, extra_edges=seed % 5, seed=seed)
        exact = len(connected_subgraphs(7, labels.neighbour_masks))
        assert count_connected_subsets(labels.neighbour_masks, 10 ** 6) == exact
        assert count_connected_subsets(labels.neighbour_masks, 10) == min(exact, 11)


def test_states_include_the_optimal_path_and_are_never_final():
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=1)
    states = sample_states(labels, np.random.default_rng(0), n_rollouts=2, max_states=8)
    tree, _ = bushy.optimal_bushy_plan(labels)
    optimal = [bushy.forest_after(tree, 6, k) for k in range(5)]
    assert all(f in states for f in optimal) and all(len(f) > 1 for f in states)


def _tables(seeds=(0, 1, 2)):
    labels_list = [_synthetic_labels(6, extra_edges=2, seed=s)[0] for s in seeds]
    _STATE.update(labels=labels_list, n_rollouts=1, max_states=6)
    rows = [row for q in range(len(labels_list)) for row in _state_rows((q, 0))]
    torch.manual_seed(0)
    embeddings = [torch.randn(6, 8) for _ in labels_list]
    return labels_list, embeddings, ForestTables(labels_list, embeddings, [_patterns(l) for l in labels_list], rows, "cpu")


def test_rows_targets_are_the_exact_constrained_dp():
    labels_list, _, tables = _tables()
    r = {k: v for k, v in tables.rows.items()}
    k = int(torch.nonzero(~r["full"])[0])
    query = int(r["query"][k])
    component = r["component"][k].tolist()
    forest = tuple(sorted(sum(1 << p for p in range(6) if component[p] == c) for c in set(component)))
    assert abs(float(r["g"][k]) - bushy.forest_cost_to_go(labels_list[query], forest)) < 1e-4


def test_training_loss_is_finite_and_greedy_returns_a_valid_tree():
    labels_list, embeddings, tables = _tables()
    model = ForestValueNet(embedding_dim=8, hidden_dim=16, n_message_layers=2, message_layer="gine", edge_feature_dim=JOIN_FEATURE_DIM)
    model.set_target_statistics(*tables.statistics())
    loss, parts = training_loss(model, tables, torch.arange(tables.n_states), 1.0, 1.0)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"card", "g", "rank"}
    labels = labels_list[0]
    adjacency = tables.adjacency[0, :6, :6]
    tree = greedy_plan(model.eval(), embeddings[0], adjacency, tables.edge_features[0, :6, :6], labels.neighbour_masks, "cpu")
    assert bushy.is_valid_tree(tree, 6, labels.neighbour_masks)


def _model(tables):
    torch.manual_seed(1)
    model = ForestValueNet(embedding_dim=8, hidden_dim=16, n_message_layers=2, message_layer="gine", edge_feature_dim=JOIN_FEATURE_DIM)
    model.set_target_statistics(*tables.statistics())
    return model.eval()


def test_beam_of_width_one_is_greedy_and_wider_beams_give_valid_trees():
    labels_list, embeddings, tables = _tables()
    model = _model(tables)
    for q, labels in enumerate(labels_list):
        inputs = (embeddings[q], tables.adjacency[q, :6, :6], tables.edge_features[q, :6, :6], labels.neighbour_masks, "cpu")
        assert beam_plan(model, *inputs, width=1) == greedy_plan(model, *inputs)
        for width in (4, 8):
            assert bushy.is_valid_tree(beam_plan(model, *inputs, width=width), 6, labels.neighbour_masks)


def test_unbounded_beam_is_the_exact_dp_over_the_predicted_cards():
    labels_list, embeddings, tables = _tables()
    model = _model(tables)
    for q, labels in enumerate(labels_list):
        masks = connected_subgraphs(6, labels.neighbour_masks)
        sets = torch.tensor([[bool(m >> p & 1) for p in range(6)] for m in masks]).unsqueeze(0)
        with torch.no_grad():
            card_std = model.card_of_sets(embeddings[q].unsqueeze(0), torch.ones(1, 6, dtype=torch.bool), sets)[0]
        predicted = dict(zip(masks, model.unstandardise(card_std, card_std)[0].tolist()))
        tree = beam_plan(model, embeddings[q], tables.adjacency[q, :6, :6], tables.edge_features[q, :6, :6],
                         labels.neighbour_masks, "cpu", width=10 ** 6)
        _, best = bushy.optimal_bushy_plan(labels, logcard=predicted)
        assert abs(bushy.tree_log_cost(tree, predicted) - best) < 1e-4


def _star_query(n):
    from types import SimpleNamespace
    patterns = [f"?x <http://example.com/p{i}> ?o{i} ." for i in range(n)]
    return SimpleNamespace(triple_patterns=patterns, query="q", type="star")


def test_sparse_rows_respect_the_budget_and_bound_the_exact_cost_to_go(monkeypatch):
    from src.supervised_value_estimation.amortized_dp import bushy_train
    from src.supervised_value_estimation.amortized_dp.labels import QueryLabels
    n = 7
    rng = random.Random(0)
    full = {mask: rng.uniform(0.0, 12.0) for mask in connected_subgraphs(n, [(1 << n) - 1 - (1 << i) for i in range(n)])}
    calls = []
    monkeypatch.setattr(bushy_train, "model_log_cardinalities", lambda oracle, query, masks, device: calls.append(len(masks)) or [full[m] for m in masks])
    exact = QueryLabels(query="q", family="star", n_tp=n, neighbour_masks=[(1 << n) - 1 - (1 << i) for i in range(n)], logcard=full)
    for budget, greedy in ((60, 0), (127, 0), (90, 8)):
        labels, rows = bushy_train.sparse_rows(_star_query(n), None, "cpu", np.random.default_rng(0), width=2,
                                               n_rollouts=2, max_states=12, budget=budget, greedy_plans=greedy)
        assert len(labels.logcard) <= budget and sum(calls) == len(labels.logcard) and rows
        assert all(row[6] for row in rows)
        for state, child, merged, card, g, done, _ in rows:
            assert card == full[merged]
            if not done:
                assert bushy.forest_cost_to_go(exact, child) - 1e-9 <= g < math.inf
        calls.clear()


def test_moe_starts_as_the_plain_model_and_trains():
    labels_list, embeddings, tables = _tables()
    plain = ForestValueNet(embedding_dim=8, hidden_dim=16, n_message_layers=2, message_layer="gine", edge_feature_dim=JOIN_FEATURE_DIM)
    moe = ForestValueNet(embedding_dim=8, hidden_dim=16, n_message_layers=2, message_layer="gine", edge_feature_dim=JOIN_FEATURE_DIM,
                         moe={"num_experts": 4, "top_k": 2, "rank": 4})
    moe.load_state_dict(plain.state_dict(), strict=False)
    for model in (plain, moe):
        model.set_target_statistics(*tables.statistics())
    ids = torch.arange(tables.n_states)
    with torch.no_grad():
        assert torch.allclose(training_loss(plain, tables, ids, 1.0, 1.0)[0], training_loss(moe, tables, ids, 1.0, 1.0)[0])
    loss, parts = training_loss(moe, tables, ids, 1.0, 1.0, balance_weight=0.01)
    loss.backward()
    assert math.isfinite(parts["balance"]) and moe.moe_set.expert_R.grad.abs().sum() > 0
