"""Regression tests for slicing sub-queries out of a featurised query (the DP's
QueryGymEstimatedCost.reduced_form_query and the sub-query training slicer).

Until 2026-10-12 every pattern contributed both its subject and its object as node rows, so a
variable shared by k patterns appeared k times; the sum readout counted every copy and estimates of
16-pattern queries were inflated up to 10^13x inside the DP."""
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.datastructures.subquery_training import CardinalityExamples
from src.models.model_instantiator import ModelFactory
from src.query_environments.gym.query_gym_estimated_cost import QueryGymEstimatedCost

CONFIG = "experiments/model_configs/pretrain_model/t_cv_repr_exact_separate_head_own_embeddings_hll.yaml"
X_DIM, EDGE_DIM = 136, 138          # the hll model config's inputs (edge_dim + 1 columns)


def _star_query(n_patterns=4, seed=0):
    """?x pi ci for i < n-1 and ?x p ?y: ?x shared by every pattern. Edges as the featurisers make
    them: per pattern two directed edges (subject -> object, object -> subject), pattern-major."""
    generator = torch.Generator().manual_seed(seed)
    terms = ["?x"] + [f"<http://example.com/{i}>" for i in range(n_patterns - 1)] + ["?y"]
    term_to_id = {t: i for i, t in enumerate(terms)}
    patterns = [f"?x <http://example.com/p{i}> {terms[1 + i]} ." for i in range(n_patterns - 1)]
    patterns.append("?x <http://example.com/pq> ?y .")
    src, dst = [], []
    for tp in patterns:
        s, _, o = tp.split(" ")[:3]
        src += [term_to_id[s], term_to_id[o]]
        dst += [term_to_id[o], term_to_id[s]]
    return Data(x=torch.randn(len(terms), X_DIM, generator=generator), edge_index=torch.tensor([src, dst]),
                edge_attr=torch.randn(2 * len(patterns), EDGE_DIM, generator=generator),
                triple_patterns=patterns, term_to_id=term_to_id, query=" ".join(patterns), type="star",
                y=torch.tensor([5.0]))


def _predict(model, data):
    batch = Batch.from_data_list([Data(x=data.x, edge_index=torch.as_tensor(data.edge_index), edge_attr=data.edge_attr)])
    outputs = model.forward(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
    return float(next(h["output"] for h in outputs if h["output_type"] == "cardinality").view(-1)[0])


def test_sub_query_has_every_node_once():
    query = _star_query(5)
    sliced = QueryGymEstimatedCost.reduced_form_query(query, [0, 2, 4], 3)
    assert sliced.x.shape[0] == 4                                   # ?x, c0, c2, ?y -- ?x once
    assert torch.equal(sliced.x[0], query.x[0])
    assert int(torch.as_tensor(sliced.edge_index).max()) == 3


@torch.no_grad()
def test_slicing_all_patterns_reproduces_the_unsliced_prediction():
    torch.manual_seed(0)
    model = ModelFactory(CONFIG).load_gine_conv().eval()
    for n in (2, 4, 8):
        query = _star_query(n, seed=n)
        sliced = QueryGymEstimatedCost.reduced_form_query(query, list(range(n)), n)
        assert abs(_predict(model, sliced) - _predict(model, query)) < 1e-5


def test_training_slicer_matches_the_dp_slicer():
    queries = [_star_query(n, seed=n) for n in (3, 5, 6)]

    class _Dataset(list):
        load_mappings, data_mappings = True, None

    examples = CardinalityExamples(_Dataset(queries))
    rng = np.random.default_rng(0)
    for i, query in enumerate(queries):
        n = len(query.triple_patterns)
        for _ in range(5):
            members = sorted(rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False).tolist())
            x, edge_index, edge_attr = examples.slice(i, members)
            reference = QueryGymEstimatedCost.reduced_form_query(query, members, len(members))
            assert torch.equal(x, reference.x) and torch.equal(edge_attr, reference.edge_attr)
            assert torch.equal(edge_index, torch.as_tensor(reference.edge_index))
