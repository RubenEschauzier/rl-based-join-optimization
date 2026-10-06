import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mlp_epinet import MLPEpinet
from src.supervised_value_estimation.amortized_dp.agents import AmortizedDPAgent, EpinetAmortizedDPAgent
from src.supervised_value_estimation.amortized_dp.labels import mask_of, optimal_left_deep_plan
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.online.exploration import thompson_plans
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import (
    EDGE_DIM, _random_model_inputs, _synthetic_labels, _synthetic_query,
)
from src.supervised_value_estimation.amortized_dp.train_amortized_dp import StateTables, training_step
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import loss_epinet
from src.utils.epinet_utils.epinet_loss import anchor_vectors, epinet_term_loss, stable_key
from src.utils.epinet_utils.risk import cvar_upper_tail

EPINET = {"index_dim": 6, "hidden_dim": 8, "prior_hidden_dim": 10, "alpha_mlp": 1.0}
ALL_HEADS = ("card", "cost_to_go", "latency", "step")


def _epinet_model(seed, epinet=EPINET):
    torch.manual_seed(seed)
    return ContractedJoinGraphValueNet(embedding_dim=16, hidden_dim=32, message_layer="gine",
                                       edge_feature_dim=EDGE_DIM, latency_head=True, step_latency_head=True,
                                       epinet=epinet).eval()


def _setup(seed, n=6, epinet=EPINET):
    labels, edges = _synthetic_labels(n, extra_edges=3, seed=seed)
    query = _synthetic_query(labels, edges)
    cache = {labels.query: torch.randn(n, 16, generator=torch.Generator().manual_seed(seed))}
    return labels, query, cache, _epinet_model(seed, epinet)


# --- the generic epinet pieces ------------------------------------------------------------

def test_anchor_vectors_are_deterministic_unit_norm_and_standard_normal_per_coordinate():
    a, b = anchor_vectors([1, 2, 3], 16), anchor_vectors([1, 2, 3], 16)
    assert torch.equal(a, b) and not torch.allclose(a[0], a[1])
    assert torch.allclose(a.norm(dim=1), torch.ones(3))
    assert not torch.allclose(a, anchor_vectors([1, 2, 3], 16, anchor_seed=1))
    coordinates = anchor_vectors(range(20000), 8) * math.sqrt(8)       # z.c ~ N(0, 1) needs this
    assert abs(coordinates.mean()) < 0.03 and abs(coordinates.std() - 1) < 0.03
    assert stable_key("q", 3) == stable_key("q", 3) != stable_key("q", 4) and 0 <= stable_key("q", 3) < 2 ** 63


def test_epinet_term_trains_only_the_learnable_epinet():
    epinet = MLPEpinet(5, 4, ["h"], hidden_dim=6)
    epinet.learnable_epinet_heads["h"].weight.data.normal_()
    base = torch.randn(7, 1, requires_grad=True)
    features = torch.randn(7, 5, requires_grad=True)
    loss, targets, predictions = epinet_term_loss(epinet, "h", base, features, torch.randn(7), torch.randn(3, 4),
                                                  anchor_vectors(range(7), 4), 0.1, 1.0,
                                                  lambda p, t: torch.mean((p - t) ** 2))
    loss.backward()
    assert base.grad is None and features.grad is None
    assert epinet.learnable_epinet_heads["h"].weight.grad is not None
    assert all(p.grad is None for p in epinet.prior_epinet_features.parameters())
    assert predictions.shape == (21, 1) and targets.shape == (21,)


def test_loss_epinet_with_anchor_keys_uses_the_keyed_anchors():
    torch.manual_seed(0)
    epinet = MLPEpinet(4, 3, ["plan_cost"], hidden_dim=5)
    estimated, features, priors = torch.randn(5, 1), torch.randn(5, 4), torch.randn(3, 5)
    plans = [(None, float(t), 7) for t in torch.randn(5)]
    keys = [stable_key("query", i) for i in range(5)]
    total, _, predictions = loss_epinet(priors, epinet, torch.nn.MSELoss(), estimated, features, plans, 4, 0.2,
                                        1.0, 0.5, torch.Generator(), torch.device("cpu"), epistemic_seed=3,
                                        anchor_keys=keys)
    indexes = epinet.sample_epistemic_indexes_batched(4, generator=torch.Generator().manual_seed(3))
    targets = torch.tensor([p[1] for p in plans])
    term, _, expected = epinet_term_loss(epinet, "plan_cost", estimated, features, targets, indexes,
                                         anchor_vectors(keys, 3), 0.2, 1.0, torch.nn.MSELoss(), 0.5,
                                         (indexes @ priors).view(-1, 1))
    assert torch.equal(predictions, expected)
    assert total.item() == pytest.approx((torch.nn.MSELoss()(estimated.squeeze(-1), targets) + term).item())


def test_cvar_is_the_mean_of_the_worst_tail():
    samples = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 4.0, 4.0, 4.0]])
    assert torch.equal(cvar_upper_tail(samples, 0.5), torch.tensor([4.0, 4.0]))
    assert torch.equal(cvar_upper_tail(samples, 0.99), torch.tensor([5.0, 4.0]))   # at least one sample


# --- the value net -------------------------------------------------------------------------

def test_new_heads_and_epinet_leave_the_existing_predictions_unchanged():
    inputs = _random_model_inputs()
    torch.manual_seed(0)
    plain = ContractedJoinGraphValueNet(embedding_dim=16, hidden_dim=32, message_layer="gine",
                                        edge_feature_dim=EDGE_DIM, latency_head=True).eval()
    model = _epinet_model(0)
    for a, b in zip(plain(*inputs, return_latency=True), model(*inputs, return_latency=True)):
        assert torch.equal(a, b)


def test_untrained_epinet_without_prior_is_the_base_and_with_prior_it_spreads():
    inputs = _random_model_inputs(seed=1)
    added = torch.tensor([-1, 0, 1, 2])
    indexes = torch.randn(5, EPINET["index_dim"])
    flat = _epinet_model(1, {**EPINET, "alpha_mlp": 0.0})
    outputs = flat.forward_heads(*inputs, added=added, heads=ALL_HEADS)
    for head, samples in flat.epinet_samples(outputs, indexes, ALL_HEADS).items():
        assert samples.shape == (5, 4) and torch.allclose(samples, outputs[head].expand(5, -1))
    spread = _epinet_model(1)
    outputs = spread.forward_heads(*inputs, added=added, heads=ALL_HEADS)
    for samples in spread.epinet_samples(outputs, indexes, ALL_HEADS).values():
        assert (samples.std(dim=0) > 0).all()


def test_step_head_depends_on_which_pattern_was_added_but_card_does_not():
    model = _epinet_model(2)
    inputs = _random_model_inputs(seed=2)
    first = model.forward_heads(*inputs, added=torch.tensor([0, 0, 0, 0]), heads=("card", "step"))
    second = model.forward_heads(*inputs, added=torch.tensor([1, 1, 1, 1]), heads=("card", "step"))
    assert torch.equal(first["card"], second["card"]) and not torch.allclose(first["step"], second["step"])


# --- training ------------------------------------------------------------------------------

def _tables(seed):
    labels, _ = _synthetic_labels(6, extra_edges=3, seed=seed)
    plan = optimal_left_deep_plan(labels)[0]
    steps = {(0, mask_of(plan[:2])): 8.0}
    steps.update({(mask_of(plan[:k - 1]), mask_of(plan[:k])): 2.0 * k for k in range(3, len(plan) + 1)})
    latency = {mask_of(plan[:k]): math.log1p(10.0 / k) for k in range(2, len(plan) + 1)}
    tables = StateTables([labels], [torch.randn(6, 16)], torch.device("cpu"), latency_to_go=[latency],
                         step_ms=[steps], cost_to_go_upper_bound=True)
    return labels, plan, steps, tables


def test_state_tables_record_steps_and_stable_anchor_keys():
    labels, plan, steps, tables = _tables(70)
    rows = tables.rows
    root = rows["root"]
    assert (rows["parent"][root] == 0).all() and (rows["added"][root] == -1).all()
    valid = rows["step_valid"]
    assert valid.sum() == len(steps)
    for parent, child, step in zip(rows["parent"][valid].tolist(), rows["mask"][valid].tolist(), rows["step"][valid].tolist()):
        assert step == pytest.approx(math.log1p(steps[(parent, child)]))
    again = _tables(70)[3].rows
    assert torch.equal(rows["key_state"], again["key_state"]) and torch.equal(rows["key_step"], again["key_step"])


def test_training_step_trains_every_head_and_its_epinet():
    _, _, _, tables = _tables(71)
    model = _epinet_model(3).train()
    model.set_latency_statistics(*tables.latency_statistics())
    model.set_step_statistics(*tables.step_statistics())
    cfg = OmegaConf.create({"rank_temperature": 0.1, "target_temperature": 0.05, "rank_weight": 1.0})
    settings = OmegaConf.create({"sigma": 0.05, "n_indexes_train": 4})
    state_ids = torch.arange(tables.n_states)
    loss, parts = training_step(model, tables, state_ids, cfg, epinet_settings=settings)
    assert {"loss_step", "loss_latency", "loss_epinet_card", "loss_epinet_cost_to_go", "loss_epinet_latency",
            "loss_epinet_step"} <= parts.keys()
    loss.backward()
    for name in ("epinet_state.learnable_epinet_heads.card.weight", "epinet_step.learnable_epinet_heads.step.weight",
                 "step_head.0.weight", "first_join_node", "input_projection.0.weight"):
        assert dict(model.named_parameters())[name].grad is not None, name
    assert all(p.grad is None for p in model.epinet_state.prior_epinet_features.parameters())
    _, parts = training_step(model, tables, state_ids, cfg)
    assert not any(key.startswith("loss_epinet_") for key in parts)


# --- planning ------------------------------------------------------------------------------

def _candidates(labels):
    pairs = [[i, j] for i, j in labels.connected_pairs()]
    triples = [p + [a] for p in pairs for a in range(labels.n_tp) if labels.neighbours(mask_of(p)) >> a & 1]
    return pairs + triples


def test_epinet_agents_reduce_to_the_base_agent_when_the_epinet_adds_nothing():
    labels, query, cache, model = _setup(80, epinet={**EPINET, "alpha_mlp": 0.0})
    base = AmortizedDPAgent(model, embed_fn=None, embedding_cache=cache)
    thompson = EpinetAmortizedDPAgent(model, None, torch.randn(1, 6), embedding_cache=cache)
    robust = EpinetAmortizedDPAgent(model, None, torch.randn(8, 6), embedding_cache=cache, alpha_cvar=0.75)
    candidates = _candidates(labels)
    expected = base.estimate_costs(candidates, base.setup_episode(query))[0]
    for agent in (thompson, robust):
        assert agent.estimate_costs(candidates, agent.setup_episode(query))[0] == pytest.approx(expected, abs=1e-4)
    assert beam_search(query, thompson, 1)[0]["plan"] == beam_search(query, base, 1)[0]["plan"]


def test_robust_score_is_the_cvar_of_the_sampled_scores():
    labels, query, cache, model = _setup(81)
    for objective in ("cout", "latency"):
        agent = EpinetAmortizedDPAgent(model, None, torch.randn(8, 6), embedding_cache=cache, objective=objective,
                                       alpha_cvar=0.75)
        episode = agent.setup_episode(query)
        candidates = _candidates(labels)
        costs, _ = agent.estimate_costs(candidates, episode)
        sampled = torch.tensor([agent.sampled_scores(plan, episode) for plan in candidates], dtype=torch.float64)
        assert costs == pytest.approx(cvar_upper_tail(sampled, 0.75).tolist())
        assert sampled.std(dim=1).min() > 0


def test_latency_objective_sums_step_latencies_and_the_latency_to_go():
    labels, query, cache, model = _setup(82)
    agent = EpinetAmortizedDPAgent(model, None, torch.randn(1, 6), embedding_cache=cache, objective="latency")
    episode = agent.setup_episode(query)
    plan = optimal_left_deep_plan(labels)[0]
    agent.estimate_costs([plan[:3]], episode)
    agent.estimate_costs([plan], episode)
    values = episode["values"]
    partial = sum(math.expm1(values["step"][s][0]) for s in agent._steps(plan[:3]))
    partial += math.expm1(values["latency"][mask_of(plan[:3])][0])
    assert agent.sampled_scores(plan[:3], episode)[0] == pytest.approx(math.log1p(partial))
    full = sum(math.expm1(values["step"][s][0]) for s in agent._steps(plan))      # nothing left to run
    assert agent.sampled_scores(plan, episode)[0] == pytest.approx(math.log1p(full))
    assert agent._steps(plan)[0] == (mask_of(plan[:2]), -1)


def test_thompson_plans_are_distinct_complete_and_connected():
    labels, query, cache, model = _setup(83, n=7)
    data = query.to_data_list()[0]
    plans = thompson_plans(model, data, 4, cache, torch.device("cpu"), torch.Generator().manual_seed(0), shared={})
    assert 1 <= len(plans) <= 4 and len({tuple(p) for p in plans}) == len(plans)
    for plan in plans:
        assert sorted(plan) == list(range(labels.n_tp))
        assert all(mask_of(plan[:k]) in labels.logcard for k in range(1, labels.n_tp + 1))


def test_prior_calibration_sets_each_heads_prior_to_the_target_width():
    _, _, _, tables = _tables(72)
    model = _epinet_model(5)
    settings = OmegaConf.create({"prior_scale_target": 0.5, "prior_scale_indexes": 256})
    from src.supervised_value_estimation.amortized_dp.train_amortized_dp import _bits, calibrate_epinet_prior
    calibrated = calibrate_epinet_prior(model, tables, ["card", "cost_to_go"], settings)
    assert set(calibrated) == {"card", "cost_to_go"}
    r = tables.rows
    outputs = model.forward_heads(tables.embeddings[r["query"]], tables.pattern_mask[r["query"]],
                                  tables.adjacency[r["query"]], _bits(r["mask"], tables.n_max),
                                  tables.edge_features[r["query"]], added=r["added"], heads=ALL_HEADS)
    for head in ("card", "cost_to_go"):
        width = model.epinet_for(head).prior_scale(outputs["features_state"], 2048, head_name=head)
        assert width * model.epinet_alpha(head) == pytest.approx(0.5, rel=0.1)
    assert calibrate_epinet_prior(model, tables, ["card"], settings) == {}          # kept once calibrated
    assert set(calibrate_epinet_prior(model, tables, ["card", "step"], settings)) == {"step"}


@pytest.mark.parametrize("objective", ["cout", "latency"])
def test_parallel_thompson_paths_equal_one_greedy_search_per_index(objective):
    labels, query, cache, model = _setup(84, n=7)
    data = query.to_data_list()[0]
    plans = thompson_plans(model, data, 3, cache, torch.device("cpu"), torch.Generator().manual_seed(5),
                           objective=objective, shared={})
    indexes = model.epinet_state.sample_epistemic_indexes_batched(6, generator=torch.Generator().manual_seed(5))
    expected = []
    for index in indexes:
        agent = EpinetAmortizedDPAgent(model, None, index.unsqueeze(0), embedding_cache=cache, objective=objective)
        plan = [int(p) for p in beam_search(query, agent, 1)[0]["plan"]]
        if plan not in expected and len(expected) < 3:
            expected.append(plan)
    assert plans == expected
