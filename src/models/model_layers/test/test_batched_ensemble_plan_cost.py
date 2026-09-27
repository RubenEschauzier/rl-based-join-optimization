"""The grouped-op ensemble must reproduce the per-member loop it replaces.

The case that matters is *ragged* plan sets: flatten_plans emits sub-plans of differing
length for one query, so shorter plans are padded and masked. Uniform-length plans
exercise neither the padded convolution columns nor the masked attention softmax, which
is precisely where a batching bug would hide.
"""

import pytest
import torch
from torch import nn

from src.models.epistemic_neural_network import prepare_epinet_model
from src.models.model_layers.batched_ensemble_plan_cost import BatchedEnsemblePlanCost
from src.utils.epinet_utils.profile_epinet import make_query
from src.utils.tree_conv_utils import (
    precompute_left_deep_tree_conv_index,
    precompute_left_deep_tree_node_mask,
)

FULL_CONFIG = "experiments/model_configs/policy_networks/t_cv_repr_separate_head_own_embeddings_hll.yaml"
PRIOR_CONFIG = "experiments/model_configs/prior_networks/prior_t_cv_smallest_hll.yaml"


def _model(index_dim, device):
    torch.manual_seed(0)
    return prepare_epinet_model(
        full_gnn_config=FULL_CONFIG,
        config_ensemble_prior=PRIOR_CONFIG,
        epinet_index_dim=index_dim,
        mlp_dimension=64,
        heads_config={"plan_cost": {"layer": nn.Linear(64, 1)}},
        heads_config_prior={"plan_cost": {"layer": nn.Linear(5, 1)}},
        device=device,
        epinet_feature_mode="mlp_plus_plan",
        epinet_hidden_dim=50,
    ).to(device).eval()


def _relative_deviation(index_dim, plan_lengths, n_triple_patterns=8, device=None):
    device = device or torch.device("cpu")
    model = _model(index_dim, device)
    query = make_query(n_triple_patterns, device)
    conv_index = precompute_left_deep_tree_conv_index(20)
    node_mask = precompute_left_deep_tree_node_mask(20)

    generator = torch.Generator().manual_seed(1)
    plans = [
        (torch.randperm(n_triple_patterns, generator=generator)[:length].tolist(), 0.0, i)
        for i, length in enumerate(plan_lengths)
    ]

    with torch.no_grad():
        embedded_prior = model.embed_query_batched_prior(query)
        trees, indexes, masks = model.prepare_ensemble_prior_inputs(
            plans, embedded_prior, conv_index, node_mask, 0
        )
        batched = model.compute_ensemble_prior_from_prepared(
            trees, indexes, masks, use_batched=True)["plan_cost"]
        loop = model.compute_ensemble_prior_from_prepared(
            trees, indexes, masks, use_batched=False)["plan_cost"]

    assert batched.shape == (index_dim, len(plans))
    scale = loop.abs().max().item()
    assert scale > 1e-3, "priors are ~zero; the test would pass vacuously"
    return (batched - loop).abs().max().item() / scale


@pytest.mark.parametrize("index_dim", [2, 16, 30])
def test_matches_loop_on_uniform_plans(index_dim):
    assert _relative_deviation(index_dim, [8] * 12) < 1e-5


@pytest.mark.parametrize("index_dim", [2, 16, 30])
def test_matches_loop_on_ragged_plans(index_dim):
    """Sub-plans of mixed length, i.e. what flatten_plans actually produces."""
    lengths = [2, 3, 8, 4, 8, 2, 5, 6, 7, 8, 3, 2]
    assert _relative_deviation(index_dim, lengths) < 1e-5


def test_matches_loop_on_single_plan():
    assert _relative_deviation(8, [5]) < 1e-5


def test_matches_loop_on_minimum_length_plans():
    assert _relative_deviation(8, [2, 2, 2]) < 1e-5


def test_snapshot_follows_member_weights():
    """A stale snapshot would silently serve the old priors after a weight change."""
    device = torch.device("cpu")
    model = _model(8, device)
    members = [m.query_plan_model for m in model.ensemble_combined_prior_models]
    before = model.batched_ensemble_prior.head_weight.clone()

    with torch.no_grad():
        members[0].heads["plan_cost"].weight.add_(1.0)
    model.refresh_batched_ensemble()

    assert not torch.allclose(before, model.batched_ensemble_prior.head_weight)


def test_snapshot_stays_out_of_the_checkpoint():
    model = _model(4, torch.device("cpu"))
    assert not any("batched_ensemble_prior" in key for key in model.state_dict())
