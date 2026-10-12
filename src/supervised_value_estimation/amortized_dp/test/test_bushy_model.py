import random

import pytest
import torch

from src.supervised_value_estimation.amortized_dp import bushy
from src.supervised_value_estimation.amortized_dp.bushy_model import ForestValueNet, forest_inputs
from src.supervised_value_estimation.amortized_dp.test.test_amortized_dp import _synthetic_labels

EMBEDDING_DIM, HIDDEN, EDGE_DIM = 8, 16, 3


def _query_tensors(labels, seed=0):
    n = labels.n_tp
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(1, n, EMBEDDING_DIM, generator=generator)
    adjacency = torch.tensor([[bool(labels.neighbour_masks[i] >> j & 1) for j in range(n)] for i in range(n)]).unsqueeze(0)
    edge_features = torch.randint(0, 3, (1, n, n, EDGE_DIM), generator=generator).float() * adjacency.unsqueeze(-1)
    return embeddings, torch.ones(1, n, dtype=torch.bool), adjacency, edge_features


def _model(message_layer="gine", **kwargs):
    torch.manual_seed(0)
    return ForestValueNet(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN, n_message_layers=2,
                          message_layer=message_layer, edge_feature_dim=EDGE_DIM, **kwargs).eval()


def _forest_and_merge(labels, steps, seed):
    rng = random.Random(seed)
    forest, merge = bushy.initial_forest(labels.n_tp), None
    for _ in range(steps):
        merge = rng.choice(bushy.candidate_merges(forest, labels.neighbour_masks))
        forest = bushy.merge(forest, *merge)
    return forest, merge


@pytest.mark.parametrize("message_layer", ["mean", "gine", "gat"])
def test_outputs_do_not_depend_on_component_numbering(message_layer):
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=1)
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    model = _model(message_layer, step_latency_head=True, latency_head=True)
    forest, merge = _forest_and_merge(labels, 3, seed=2)
    component, newest, side = forest_inputs([forest], 7, merges=[merge])
    relabel = torch.randperm(7, generator=torch.Generator().manual_seed(3))
    heads = ("card", "cost_to_go", "latency", "step")
    a = model.forward_heads(embeddings, mask, adjacency, component, newest, edges, side, heads=heads)
    b = model.forward_heads(embeddings, mask, adjacency, relabel[component], relabel[newest], edges, side, heads=heads)
    for head in heads:
        assert torch.allclose(a[head], b[head], atol=1e-5), head


def test_card_is_a_set_function_of_the_newest_component():
    labels, _ = _synthetic_labels(7, extra_edges=3, seed=4)
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    model = _model()
    forest, merge = _forest_and_merge(labels, 2, seed=5)
    component, newest, _ = forest_inputs([forest], 7, merges=[merge])
    card, cost_to_go = model(embeddings, mask, adjacency, component, newest, edges)
    outside = [p for p in range(7) if not (merge[0] | merge[1]) >> p & 1]
    changed = embeddings.clone()
    changed[0, outside] += 1.0
    card_changed, cost_to_go_changed = model(changed, mask, adjacency, component, newest, edges)
    assert torch.allclose(card, card_changed, atol=1e-6)
    assert not torch.allclose(cost_to_go, cost_to_go_changed)       # the cost-to-go sees the whole forest
    union = torch.tensor([[bool((merge[0] | merge[1]) >> p & 1) for p in range(7)]]).unsqueeze(0)
    assert torch.allclose(model.card_of_sets(embeddings, mask, union)[:, 0], card, atol=1e-6)


def test_step_head_is_symmetric_in_the_merged_parts():
    labels, _ = _synthetic_labels(6, extra_edges=2, seed=6)
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    model = _model(step_latency_head=True)
    assert not hasattr(model, "first_join_node")
    forest, (left, right) = _forest_and_merge(labels, 3, seed=7)
    a = forest_inputs([forest], 6, merges=[(left, right)])
    b = forest_inputs([forest], 6, merges=[(right, left)])
    step_a = model.forward_heads(embeddings, mask, adjacency, a[0], a[1], edges, a[2], heads=("step",))["step"]
    step_b = model.forward_heads(embeddings, mask, adjacency, b[0], b[1], edges, b[2], heads=("step",))["step"]
    assert torch.allclose(step_a, step_b, atol=1e-6)


def test_padding_does_not_change_outputs():
    labels, _ = _synthetic_labels(5, extra_edges=2, seed=8)
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    model = _model()
    forest, merge = _forest_and_merge(labels, 2, seed=9)
    component, newest, _ = forest_inputs([forest], 5, merges=[merge])
    pad = lambda x, dims: torch.nn.functional.pad(x, dims)
    padded = model(pad(embeddings, (0, 0, 0, 3)), pad(mask, (0, 3)), pad(adjacency, (0, 3, 0, 3)),
                   pad(component, (0, 3)), newest, pad(edges, (0, 0, 0, 3, 0, 3)))
    plain = model(embeddings, mask, adjacency, component, newest, edges)
    for x, y in zip(padded, plain):
        assert torch.allclose(x, y, atol=1e-5)


def test_epinet_samples_on_forest_features():
    labels, _ = _synthetic_labels(5, extra_edges=2, seed=10)
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    model = _model(step_latency_head=True, epinet={"index_dim": 4, "hidden_dim": 8})
    forest, merge = _forest_and_merge(labels, 2, seed=11)
    component, newest, side = forest_inputs([forest], 5, merges=[merge])
    outputs = model.forward_heads(embeddings, mask, adjacency, component, newest, edges, side,
                                  heads=("card", "cost_to_go", "step"))
    samples = model.epinet_samples(outputs, torch.randn(3, 4), ["cost_to_go", "step"])
    assert samples["cost_to_go"].shape == (3, 1) and samples["step"].shape == (3, 1)


def test_fits_the_exact_constrained_dp_targets():
    """Every forest state of a few small queries, labelled with bushy.forest_cost_to_go and the
    newest component's card; the model must fit them (capacity / wiring sanity check)."""
    torch.manual_seed(0)
    rows = []
    for seed in range(4):
        labels, _ = _synthetic_labels(5, extra_edges=2, seed=seed)
        tensors = _query_tensors(labels, seed)
        frontier, seen = [(bushy.initial_forest(5), None)], set()
        while frontier:
            forest, merge = frontier.pop()
            if forest in seen or len(forest) == 1:
                continue
            seen.add(forest)
            card = labels.logcard[merge[0] | merge[1]] if merge else 0.0
            rows.append((tensors, forest, merge, card, bushy.forest_cost_to_go(labels, forest), merge is not None))
            frontier += [(bushy.merge(forest, a, b), (a, b)) for a, b in bushy.candidate_merges(forest, labels.neighbour_masks)]
    model = ForestValueNet(embedding_dim=EMBEDDING_DIM, hidden_dim=32, n_message_layers=2, message_layer="gine",
                           edge_feature_dim=EDGE_DIM)
    batch = [torch.cat([r[0][i] for r in rows]) for i in range(4)]
    component = torch.cat([forest_inputs([r[1]], 5, merges=[r[2]])[0] for r in rows])
    newest = torch.cat([forest_inputs([r[1]], 5, merges=[r[2]])[1] for r in rows])
    card_target = torch.tensor([r[3] for r in rows])
    g_target = torch.tensor([r[4] for r in rows])
    has_card = torch.tensor([r[5] for r in rows])
    model.set_target_statistics(card_target[has_card].mean().item(), card_target[has_card].std().item(),
                                g_target.mean().item(), g_target.std().item())
    card_z, g_z = model.standardise(card_target, g_target)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

    def loss():
        card, g = model(batch[0], batch[1], batch[2], component, newest, batch[3])
        return ((card - card_z)[has_card] ** 2).mean() + ((g - g_z) ** 2).mean()

    start = loss().item()
    for _ in range(400):
        optimiser.zero_grad()
        value = loss()
        value.backward()
        optimiser.step()
    assert loss().item() < 0.1 * start, (start, loss().item())


def test_state_heads_do_not_depend_on_the_build_order():
    """Building 01 then 23, or 23 then 01, reaches the same forest: identical G, latency-to-go and
    epinet state features; only card (of the newest component) and the step differ."""
    labels, _ = _synthetic_labels(6, extra_edges=0, seed=12)
    labels.neighbour_masks = [0b10, 0b101, 0b1010, 0b10100, 0b101000, 0b10000]   # path 0-1-2-3-4-5
    embeddings, mask, adjacency, edges = _query_tensors(labels)
    adjacency = torch.tensor([[abs(i - j) == 1 for j in range(6)] for i in range(6)]).unsqueeze(0)
    model = _model(step_latency_head=True, latency_head=True)
    start = bushy.initial_forest(6)
    first = bushy.merge(bushy.merge(start, 0b1, 0b10), 0b100, 0b1000)
    second = bushy.merge(bushy.merge(start, 0b100, 0b1000), 0b1, 0b10)
    assert first == second
    heads = ("card", "cost_to_go", "latency", "step")
    a = model.forward_heads(embeddings, mask, adjacency, *forest_inputs([first], 6, merges=[(0b100, 0b1000)])[:2],
                            edges, forest_inputs([first], 6, merges=[(0b100, 0b1000)])[2], heads=heads)
    b = model.forward_heads(embeddings, mask, adjacency, *forest_inputs([second], 6, merges=[(0b1, 0b10)])[:2],
                            edges, forest_inputs([second], 6, merges=[(0b1, 0b10)])[2], heads=heads)
    for key in ("cost_to_go", "latency", "features_state"):
        assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["card"], b["card"]) and not torch.equal(a["step"], b["step"])
