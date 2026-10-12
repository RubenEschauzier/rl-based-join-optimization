import torch
from torch import nn


class SliceDropout(nn.Module):
    """Drops a slice of the feature vector as a whole, per row, during training.

    Used on the embedding slot of node features (e.g. the 128 rdf2vec dimensions): with
    probability p a node's embedding is zeroed, as if the term had no embedding. A per-constant
    embedding acts almost like an ID, so without this a large estimator can memorise
    "this constant -> this cardinality", which does not transfer to unseen constants. Unlike
    nn.Dropout the kept slices are not rescaled, so they look exactly as at evaluation time.
    """

    def __init__(self, start: int, end: int, p: float = 0.2):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.start, self.end, self.p = start, end, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand(x.shape[0], 1, device=x.device) >= self.p).to(x.dtype)
        return torch.cat([x[:, :self.start], x[:, self.start:self.end] * keep, x[:, self.end:]], dim=1)

    def extra_repr(self) -> str:
        return f"start={self.start}, end={self.end}, p={self.p}"
