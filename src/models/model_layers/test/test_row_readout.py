"""RowReadout on a tiny synthetic store: the exact read must follow the (sub)graph's own patterns.

Store: entity 5 has one row, feature 3 (subject of predicate 3), n = 2 neighbours; their features are
g = 7 (M0 2, M1 6, M2 20) and the back edge g = 95. Query: <5> p3 ?v . ?v p7 ?y . ?v p9 ?z
(edges 2i: subject -> object with source feature p, 2i + 1 the reverse with 92 + p).
"""
import numpy as np
import pytest
import torch

from src.models.model_layers.row_readout import RowReadout, F, NP_


@pytest.fixture
def store(tmp_path):
    E = 6
    dptr = np.zeros(E + 1, np.int64)
    dptr[6] = 1                                      # entity 5 owns row 0
    np.save(tmp_path / "dptr.npy", dptr)
    np.save(tmp_path / "df.npy", np.array([3], np.int16))
    np.save(tmp_path / "ddeg.npy", np.array([2], np.int32))
    np.save(tmp_path / "rptr.npy", np.array([0, 2], np.int64))
    np.save(tmp_path / "eg.npy", np.array([7, NP_ + 3], np.int16))
    np.save(tmp_path / "e0.npy", np.array([2, 2], np.float32))
    np.save(tmp_path / "e1.npy", np.array([6, 2], np.float32))
    np.save(tmp_path / "e2.npy", np.array([20, 2], np.float32))
    ref = np.full((3, F, F), 0.5)
    np.savez(tmp_path / "refs.npz", ref=ref, T=np.ones(F), n_ent=np.ones(F), predicate_iris=np.arange(NP_))
    return tmp_path


def graph(patterns):
    """patterns [(s node, predicate, o node)] over nodes 0 = <5>, 1 = ?v, 2 = ?y, 3 = ?z."""
    x = torch.zeros(4, 3)
    x[:, 0] = torch.tensor([5.0, -1, -1, -1])
    src, dst, ea = [], [], []
    for s, p, o in patterns:
        src += [s, o]
        dst += [o, s]
        ea += [[p, 1.0, 1.0], [NP_ + p, 1.0, -1.0]]
    return x, torch.tensor([src, dst]), torch.tensor(ea)


def test_shapes_and_variable_patterns(store):
    torch.manual_seed(0)
    layer = RowReadout(store, out_dim=4, hidden=8)
    x, ei, ea = graph([(0, 3, 1), (1, 7, 2), (1, 9, 3)])
    x2, ea2 = layer(x, ei, ea)
    assert x2.shape == (4, 2) and ea2.shape == (6, 1 + 4 + 1)
    assert torch.equal(ea2[:, -1], ea[:, -1])                       # direction stays last
    assert torch.equal(ea2[0, 1:5], ea2[1, 1:5])                    # both directions of the constant pattern
    assert torch.all(ea2[2:, 1:5] == 0)                             # variable-only patterns read nothing


def test_reads_follow_the_subgraph(store):
    torch.manual_seed(0)
    layer = RowReadout(store, out_dim=4, hidden=8)
    full = layer(*graph([(0, 3, 1), (1, 7, 2), (1, 9, 3)]))[1][0, 1:5]
    only_hit = layer(*graph([(0, 3, 1), (1, 7, 2)]))[1][0, 1:5]
    only_miss = layer(*graph([(0, 3, 1), (1, 9, 3)]))[1][0, 1:5]
    alone = layer(*graph([(0, 3, 1)]))[1][0, 1:5]
    assert not torch.allclose(only_hit, only_miss)                  # g = 7 is in the row, g = 9 is not
    assert not torch.allclose(alone, only_hit)
    assert not torch.allclose(full, only_hit)


def test_exact_entry_moves_the_readout(store):
    torch.manual_seed(0)
    layer = RowReadout(store, out_dim=4, hidden=8)
    g = graph([(0, 3, 1), (1, 7, 2)])
    before = layer(*g)[1][0, 1:5]
    layer._store.e1[0] = 600.0                                      # many more g-edges behind the same neighbours
    after = layer(*g)[1][0, 1:5]
    assert not torch.allclose(before, after)


class _Query:
    def __init__(self, patterns):
        self.x, self.edge_index, self.edge_attr = graph(patterns)
        self.triple_patterns = [f"p{i}" for i in range(len(patterns))]


def test_query_readout_matches_the_layer_on_every_subset(store):
    from src.models.model_layers.row_readout import QueryReadout
    from src.supervised_value_estimation.amortized_dp import soft_dp
    torch.manual_seed(0)
    layer = RowReadout(store, out_dim=4, hidden=8)
    q = _Query([(0, 3, 1), (1, 7, 2), (1, 9, 3), (0, 3, 2)])
    masks = torch.arange(1, 1 << 4)
    x, ei, ea, _ = soft_dp.subset_batch(q, masks, "cpu")
    ref_x, ref_ea = layer(x, ei, ea)
    cache = QueryReadout(layer, q, "cpu")
    for chunk in (masks, masks[[2, 7, 12]], masks[-1:]):           # one call over all subsets, then a few
        cx, cei, cea, _ = soft_dp.subset_batch(q, chunk, "cpu")
        mx, mea = cache(chunk, cx, cea)
        rx, rea = layer(cx, cei, cea)
        assert torch.equal(mx, rx) and torch.allclose(mea, rea, atol=1e-6)
    assert torch.allclose(cache(masks, x, ea)[1], ref_ea, atol=1e-6)


def test_query_readout_matches_the_backbone_on_real_queries():
    import os
    from src.supervised_value_estimation.amortized_dp import soft_dp
    from src.supervised_value_estimation.amortized_dp.bushy_train import load_split
    from src.supervised_value_estimation.amortized_dp.online.train_plan_loss import Query, phi_and_direct, query_readout
    from src.models.model_instantiator import ModelFactory
    config = "experiments/model_configs/pretrain_model/t_cv_repr_exact_separate_head_own_embeddings_hll_rowreadout.yaml"
    data_dir = "data/generated_queries/mixed_yago_v2/hll_contextual_attention_rowreadout/dataset_val"
    if not (os.path.isdir(data_dir) and os.path.isdir("data/constant_embeddings/yago")):
        pytest.skip("row-readout dataset or store not available")
    torch.manual_seed(0)
    model = ModelFactory(config).load_gine_conv().eval()       # random weights: the computation is what is tested
    val = load_split(data_dir)
    with torch.no_grad():
        for i in range(0, len(val), max(1, len(val) // 25)):
            q = Query(val[i])
            s = soft_dp.SubsetStructure(q.neighbour_masks, "cpu")
            readout = query_readout(model, q.data, "cpu")
            for chunk in (s.masks, s.masks[torch.randperm(len(s.masks))[:7]]):
                batch = soft_dp.subset_batch(q.data, chunk, "cpu")
                phi_ref, l_ref = phi_and_direct(model, *batch)
                phi, l = phi_and_direct(model, *batch, readout=readout, masks=chunk)
                assert torch.allclose(phi, phi_ref, rtol=1e-4, atol=1e-4) and torch.allclose(l, l_ref, rtol=1e-4, atol=1e-4)
