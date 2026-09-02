import copy

import pytest
import torch

from src.models.model_layers.triple_gine_conv_low_rank_moe import TripleGineConvLowRankMoE


def make_layer(top_k=2):
    backbone = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    return TripleGineConvLowRankMoE(
        backbone,
        edge_dim=3,
        num_experts=4,
        top_k=top_k,
        rank=2,
        routing_level='query',
    )


def empty_graph_inputs(node_features, batch):
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 4))
    return node_features, edge_index, edge_attr, batch


def test_zero_initialized_adapters_preserve_dense_output():
    torch.manual_seed(0)
    layer = make_layer()
    node_features = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 1, 1, 1])

    output = layer(*empty_graph_inputs(node_features, batch))

    torch.testing.assert_close(output, layer.W0(node_features))


def test_query_routing_uses_one_decision_per_graph():
    torch.manual_seed(0)
    layer = make_layer()
    node_features = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 1, 1, 1])

    layer(*empty_graph_inputs(node_features, batch))

    assert layer.get_current_routing_probs().shape == (2, 4)
    assignments = layer.get_current_routing_assignments()
    assert assignments.shape == (2, 4)
    torch.testing.assert_close(assignments.sum(dim=1), torch.ones(2))


def test_online_adaptation_reduces_shift_without_changing_backbone():
    torch.manual_seed(4)
    layer = make_layer()
    layer.enable_online_adaptation()
    backbone_before = copy.deepcopy(layer.W0.state_dict())

    node_features = torch.randn(64, 4)
    batch = torch.arange(len(node_features))
    with torch.no_grad():
        baseline = layer.W0(node_features)
        regime_shift = torch.where(node_features[:, :1] > 0, 1.5, -1.5)
        target = baseline + regime_shift

    optimizer = torch.optim.Adam(layer.online_parameters(), lr=0.03)
    inputs = empty_graph_inputs(node_features, batch)
    initial_loss = torch.nn.functional.mse_loss(layer(*inputs), target).item()

    for _ in range(100):
        optimizer.zero_grad()
        prediction = layer(*inputs)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss += 0.01 * layer.load_balancing_loss()
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(layer(*inputs), target).item()
    assert final_loss < initial_loss * 0.25
    for name, value in layer.W0.state_dict().items():
        torch.testing.assert_close(value, backbone_before[name])


@pytest.mark.parametrize('top_k', [0, 5])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match='top_k'):
        make_layer(top_k=top_k)