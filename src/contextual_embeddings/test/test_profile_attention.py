import numpy as np
import pytest
import torch

from src.contextual_embeddings.model import Contextualizer, collate

N_PREDICATES = 2                      # profile features: p (outgoing) and 2 + p (incoming)
TYPE_FEATURES = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=np.uint8)
ENTITY_TOKEN = {1: 1, 2: 2, 3: 3}
PREDICATE_TOKEN = np.array([4, 5])


class _Profiles:
    """entity 1: neighbours via outgoing p0 of type 1 (count 5), via incoming p1 of type 2 (count 2);
    entity 2: only via outgoing p1 (type 0)."""
    table = {1: ([0, 3], [1, 2], [5, 2]), 2: ([1], [0], [7])}

    def rows(self, entity):
        f, t, c = self.table.get(int(entity), ([], [], []))
        return np.array(f, np.int16), np.array(t, np.int32), np.array(c, np.int32)


def _model(mode):
    torch.manual_seed(0)
    return Contextualizer(torch.randn(6, 8), hidden=6, layers=2, profile_types=TYPE_FEATURES, profile_hidden=8,
                          profile_attention=mode)


def _batch(edges, nodes=(1, -1, 2)):
    instance = {"nodes": list(nodes), "edges": edges, "empty": 0, "replaced": [0] * len(edges), "kind": "positive"}
    return collate([instance], ENTITY_TOKEN, PREDICATE_TOKEN, profiles=_Profiles())


def test_edge_feature_is_the_sources_profile_feature():
    batch = _batch([(0, 1, 0), (1, 2, 1)])                    # (1 p0 ?x), (?x p1 2)
    assert batch["edge_feature"].tolist() == [0, 2, 1, 3]       # out p0, in p0, out p1, in p1


@torch.no_grad()
def test_hard_selection_uses_exactly_the_patterns_rows():
    model = _model("hard").eval()
    batch = _batch([(0, 1, 0), (1, 2, 1)])
    out = model.conditioned_profiles(3, batch["edge_index"], batch["edge_feature"], batch["profile"])
    # edge 0: entity 1 -> ?x via outgoing p0: its row (feature 0, type 1, count 5)
    assert out[0, -1] == 1.0 and torch.isclose(out[0, -2], torch.log1p(torch.tensor(5.0)))
    # edge 1: ?x -> entity 1: a variable source, nothing
    assert torch.all(out[1] == 0)
    # edge 3: entity 2 -> ?x via incoming p1: entity 2 has no such rows -> no match (the pattern is empty)
    assert out[3, -1] == 0.0 and out[3, -2] == 0.0
    # changing a row with another feature does not change edge 0 in hard mode
    _Profiles.table[1] = ([0, 3], [1, 0], [5, 9])
    other = model.conditioned_profiles(3, batch["edge_index"], batch["edge_feature"],
                                       _batch([(0, 1, 0), (1, 2, 1)])["profile"])
    _Profiles.table[1] = ([0, 3], [1, 2], [5, 2])
    assert torch.allclose(out[0], other[0])


def test_soft_attention_trains_and_prefers_the_exact_feature_at_start():
    model = _model("soft")
    batch = _batch([(0, 1, 0), (1, 2, 1)])
    nodes, arm_logits, empty_logits = model(batch)
    (empty_logits.sum() + arm_logits.sum()).backward()
    assert model.feature_key.weight.grad is not None and nodes.shape == (3, 8)
    keys = model.feature_key.weight
    assert (keys[0] @ keys[0]) > (keys[0] @ keys[3])          # self-similarity dominates at init


@pytest.mark.parametrize("mode", [None, "soft", "hard"])
def test_forward_runs_for_every_mode(mode):
    model = _model(mode) if mode else Contextualizer(torch.randn(6, 8), hidden=6, layers=2,
                                                     profile_types=TYPE_FEATURES, profile_hidden=8)
    nodes, _, _ = model(_batch([(0, 1, 0), (1, 2, 1)]))
    assert torch.isfinite(nodes).all()
