import numpy as np
import rdflib
import torch

from src.contextual_embeddings.inference import FrozenContextualizer
from src.contextual_embeddings.model import Contextualizer, collate
from src.query_featurizers.featurize_predicate_edges_tp_hll import QueryToEdgePredicateGraphHll
from src.query_featurizers.featurize_predicate_edges_tp_hll_contextual import QueryToEdgePredicateGraphHllContextual

E = "http://example.com/"
TERMS = ["<unk>", E + "1", E + "2", E + "3", E + "100", E + "101"]     # entities 1-3, predicates 100-101
INSTANCES = [
    {"nodes": [1, -1, 2], "edges": [(1, 0, 0), (1, 2, 1)], "empty": 0, "replaced": [0, 0], "kind": "positive"},
    {"nodes": [-1, 3, -1, 9], "edges": [(0, 1, 0), (0, 2, 1), (2, 3, 0)], "empty": 1, "replaced": [1, 0, 0],
     "kind": "near_miss_constant"},
]
ENTITY_TOKEN = {1: 1, 2: 2, 3: 3}                 # entity 9 is unknown -> token 0
PREDICATE_TOKEN = np.array([4, 5])


def _model():
    torch.manual_seed(0)
    return Contextualizer(torch.randn(len(TERMS), 8), hidden=6, layers=2)


def test_collate_and_forward_shapes():
    batch = collate(INSTANCES, ENTITY_TOKEN, PREDICATE_TOKEN)
    assert batch["node_token"].tolist() == [1, 0, 2, 0, 3, 0, 0]
    assert batch["node_is_variable"].tolist() == [False, True, False, True, False, True, False]
    assert batch["edge_index"].shape == (2, 10) and batch["edge_direction"].tolist()[:2] == [1.0, -1.0]
    nodes, arm_logits, empty_logits = _model()(batch)
    assert nodes.shape == (7, 8) and arm_logits.shape == (5,) and empty_logits.shape == (2,)


def test_context_changes_a_constants_vector():
    model = _model().eval()
    q1 = collate([INSTANCES[0]], ENTITY_TOKEN, PREDICATE_TOKEN)
    q2 = collate([{**INSTANCES[0], "edges": [(1, 0, 1), (1, 2, 1)]}], ENTITY_TOKEN, PREDICATE_TOKEN)
    with torch.no_grad():
        a, b = model(q1)[0], model(q2)[0]
    assert not torch.allclose(a[0], b[0])             # entity 1, different predicate next to it


def test_frozen_contextualizer_round_trip(tmp_path):
    model = _model()
    torch.save({"state_dict": model.state_dict(), "hidden": 6, "layers": 2, "terms": TERMS}, tmp_path / "c.pt")
    frozen = FrozenContextualizer(tmp_path / "c.pt")
    x, y, p = rdflib.Variable("x"), rdflib.URIRef(E + "1"), rdflib.URIRef(E + "100")
    nodes, predicates = frozen.embed_query([(x, p, y)])
    batch = collate([{"nodes": [-1, 1], "edges": [(0, 1, 0)], "empty": 0, "replaced": [0]}], ENTITY_TOKEN,
                    PREDICATE_TOKEN)
    with torch.no_grad():
        expected = model.eval()(batch)[0]
    assert np.allclose(nodes[x], expected[0].numpy(), atol=1e-6) and np.allclose(nodes[y], expected[1].numpy(), atol=1e-6)
    assert np.allclose(predicates[p], model.table.weight[4].detach().numpy())


class _Env:
    def cardinality_term(self, term):
        return 5

    def cardinality_pattern(self, pattern):
        return 7


class _FixedContext:
    dim = 3

    def embed_query(self, patterns):
        nodes = {t: np.full(3, 0.5 + i) for i, t in enumerate({t for s, _, o in patterns for t in (s, o)})}
        return nodes, {p: np.full(3, -2.0) for _, p, _ in patterns}


def test_contextual_featurizer_only_swaps_the_embedding_slot():
    vectors = {E + "1": [1.0, 1.0, 1.0], E + "100": [3.0, 3.0, 3.0]}
    x, c, p = rdflib.Variable("x"), rdflib.URIRef(E + "1"), rdflib.URIRef(E + "100")
    query = {"query": "q", "cardinality": 2.0, "triple_patterns": ["?x <p> <c> ."], "rdflib_patterns": [(x, p, c)],
             "type": "t"}
    base = QueryToEdgePredicateGraphHll(vectors, _Env()).transform_undirected(query)
    contextual = QueryToEdgePredicateGraphHllContextual(vectors, _Env(), contextualizer=_FixedContext()).transform_undirected(query)
    assert base.x.shape == contextual.x.shape and base.edge_attr.shape == contextual.edge_attr.shape
    assert torch.equal(base.x[:, :-3], contextual.x[:, :-3])
    assert torch.equal(base.edge_attr[:, :-5], contextual.edge_attr[:, :-5])
    assert torch.equal(base.edge_attr[:, -2:], contextual.edge_attr[:, -2:])
    assert torch.equal(contextual.edge_attr[:, -5:-2], torch.full((2, 3), -2.0))
    assert not torch.equal(base.x[:, -3:], contextual.x[:, -3:])


def test_profiles_count_neighbour_types_exactly(tmp_path):
    from src.contextual_embeddings.profiles import ProfileTable, build
    from src.random_query_generation.statistics_based_generation.graph import DataGraph
    # entity 0 -p0-> 1, 0 -p0-> 2, 1 -p1-> 3 : entity 1 has {p0 in, p1 out}, entity 2 has {p0 in}
    graph = DataGraph.from_arrays(np.array([0, 0, 1]), np.array([10, 10, 11]), np.array([1, 2, 3]))
    build(graph, tmp_path)
    table = ProfileTable(tmp_path)
    assert table.n_types == 4 and table.n_features == 4               # 4 distinct predicate sets, 2 predicates x 2 directions
    feature, type_, count = table.rows(0)                             # entity 0: two outgoing p0 neighbours of different types
    assert list(feature) == [0, 0] and sorted(count.tolist()) == [1, 1] and len(set(type_.tolist())) == 2
    neighbour_types = {tuple(np.flatnonzero(table.type_features[t])) for t in type_}
    assert neighbour_types == {(1, 2), (2,)}                          # {p1 out, p0 in} and {p0 in}


def test_contextualizer_with_profiles_round_trip(tmp_path):
    from src.contextual_embeddings.profiles import ProfileTable, build
    from src.random_query_generation.statistics_based_generation.graph import DataGraph
    graph = DataGraph.from_arrays(np.array([1, 1, 2]), np.array([100, 101, 100]), np.array([2, 3, 3]))
    build(graph, tmp_path / "profiles")
    table = ProfileTable(tmp_path / "profiles")
    torch.manual_seed(0)
    model = Contextualizer(torch.randn(len(TERMS), 8), hidden=6, layers=2, profile_types=table.type_features,
                           profile_hidden=5)
    batch = collate(INSTANCES[:1], ENTITY_TOKEN, PREDICATE_TOKEN, profiles=table)
    assert batch["profile"]["node"].tolist() and set(batch["profile"]["node"].tolist()) <= {0, 2}   # constants only
    nodes, _, _ = model(batch)
    assert nodes.shape == (3, 8)
    torch.save({"state_dict": model.state_dict(), "hidden": 6, "layers": 2, "terms": TERMS,
                "profiles_directory": str(tmp_path / "profiles"), "profile_hidden": 5}, tmp_path / "c.pt")
    frozen = FrozenContextualizer(tmp_path / "c.pt")
    x, c1, c2 = rdflib.Variable("x"), rdflib.URIRef(E + "1"), rdflib.URIRef(E + "2")
    out, _ = frozen.embed_query([(x, rdflib.URIRef(E + "100"), c1), (x, rdflib.URIRef(E + "101"), c2)])
    expected = model.eval()(collate([{"nodes": [-1, 1, 2], "edges": [(0, 1, 0), (0, 2, 1)], "empty": 0,
                                      "replaced": [0, 0]}], ENTITY_TOKEN, PREDICATE_TOKEN, profiles=table))[0]
    assert np.allclose(out[c1], expected[1].detach().numpy(), atol=1e-5)
