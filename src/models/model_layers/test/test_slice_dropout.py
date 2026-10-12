import torch

from src.models.model_layers.slice_dropout import SliceDropout


def test_drops_whole_slices_only_in_training():
    torch.manual_seed(0)
    layer = SliceDropout(2, 6, p=0.5)
    x = torch.ones(1000, 8)
    out = layer.train()(x)
    assert torch.equal(out[:, :2], x[:, :2]) and torch.equal(out[:, 6:], x[:, 6:])     # outside the slice untouched
    rows = out[:, 2:6]
    assert torch.all((rows == 0).all(dim=1) | (rows == 1).all(dim=1))               # a slice is dropped as a whole, not rescaled
    assert 0.4 < (rows == 0).all(dim=1).float().mean() < 0.6
    assert torch.equal(layer.eval()(x), x)
