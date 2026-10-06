"""The MLP epinet with an MLP prior (deepmind/enn MLPEpinetWithPrior), independent of the base network.

An epinet adds an index-dependent correction to a base network's prediction:

    f(x, z) = base(x) + learnable(sg[phi(x)], z) + alpha * prior(sg[phi(x)], z)

with z ~ N(0, I_D) the epistemic index, sg[phi(x)] the base network's features (stop-gradient
is the caller's job), `learnable` a small trained MLP whose last layer starts at zero, and
`prior` a frozen, randomly initialised MLP of the same shape. Both MLPs share one hidden layer
across heads and have one output layer per head; a head's output is (head(h) . z).

MLPEpinetMixin holds the parameters and the forward computations; MLPEpinet is the standalone
module. MultiHeadEpistemicNetwork mixes it in directly, so its parameter names (and thus its
checkpoints) are unchanged: learnable_epinet_features, learnable_epinet_heads,
prior_epinet_features, prior_epinet_heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def glorot_init(module):
    """Glorot initialisation, as in the Epistemic Neural Networks paper."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPEpinetMixin:
    """Needs `epi_index_dim` and `device` attributes; `_build_mlp_epinet` creates the modules."""

    def _build_mlp_epinet(self, feature_dim, epi_index_dim, head_names, hidden_dim, prior_hidden_dim, device):
        self.learnable_epinet_features = nn.Sequential(
            nn.Linear(feature_dim + epi_index_dim, hidden_dim),
            nn.ReLU(),
        ).to(device)
        self.learnable_epinet_features.apply(glorot_init)

        self.learnable_epinet_heads = nn.ModuleDict()
        for head_name in head_names:
            head_layer = nn.Linear(hidden_dim, epi_index_dim)
            nn.init.zeros_(head_layer.weight)
            nn.init.zeros_(head_layer.bias)
            self.learnable_epinet_heads[head_name] = head_layer.to(device)

        # Prior MLP mirrors the learnable one (same architecture, different parameters),
        # as in deepmind/enn's MLPEpinetWithPrior.
        self.prior_epinet_features = nn.Sequential(
            nn.Linear(feature_dim + epi_index_dim, prior_hidden_dim),
            nn.ReLU(),
        ).to(device)
        self.prior_epinet_features.apply(glorot_init)
        for param in self.prior_epinet_features.parameters():
            param.requires_grad = False

        self.prior_epinet_heads = nn.ModuleDict()
        for head_name in head_names:
            prior_head = nn.Linear(prior_hidden_dim, epi_index_dim)
            prior_head.apply(glorot_init)
            for param in prior_head.parameters():
                param.requires_grad = False
            self.prior_epinet_heads[head_name] = prior_head.to(device)

    @staticmethod
    def _single(features_net, heads, last_feature, epi_index):
        concat_input = torch.cat([last_feature, epi_index.expand(last_feature.shape[0], -1)], dim=1)
        shared_features = features_net(concat_input)
        return {head_name: head_layer(shared_features) @ epi_index.T for head_name, head_layer in heads.items()}

    @staticmethod
    def _batched(features_net, heads, last_feature, epi_indexes):
        """Output (K*N, 1) per head, grouped by index: all N inputs for z_1, then for z_2, ..."""
        n = last_feature.shape[0]
        k = epi_indexes.shape[0]
        last_feature_exp = last_feature.repeat(k, 1)                  # [K*N, F], blocks of N per index
        epi_indexes_exp = epi_indexes.repeat_interleave(n, dim=0)     # [K*N, D], z_1 N times, z_2 N times, ...
        concat_input = torch.cat([last_feature_exp, epi_indexes_exp], dim=1)
        shared_features = features_net(concat_input)
        return {head_name: (head_layer(shared_features) * epi_indexes_exp).sum(dim=1, keepdim=True)
                for head_name, head_layer in heads.items()}

    def compute_mlp_prior(self, last_feature, epi_index):
        return self._single(self.prior_epinet_features, self.prior_epinet_heads, last_feature, epi_index)

    def compute_learnable_mlp(self, last_feature, epi_index):
        return self._single(self.learnable_epinet_features, self.learnable_epinet_heads, last_feature, epi_index)

    def compute_mlp_prior_batched(self, last_feature, epi_indexes):
        return self._batched(self.prior_epinet_features, self.prior_epinet_heads, last_feature, epi_indexes)

    def compute_learnable_mlp_batched(self, last_feature, epi_indexes):
        return self._batched(self.learnable_epinet_features, self.learnable_epinet_heads, last_feature, epi_indexes)

    def sample_epistemic_indexes(self):
        return torch.normal(0, 1, size=(1, self.epi_index_dim), device=self.device)

    def sample_epistemic_indexes_batched(self, n_epi_indexes, generator=None):
        return torch.randn((n_epi_indexes, self.epi_index_dim), device=self.device, generator=generator)

    @torch.no_grad()
    def prior_scale(self, features, n_indexes=64, head_name=None, generator=None):
        """Width of the (unweighted) MLP prior: the std over sampled indices of its output,
        averaged over the inputs (and over the heads, unless `head_name` picks one). Dividing a
        target width by this gives the alpha_mlp that makes the prior that wide."""
        indexes = self.sample_epistemic_indexes_batched(n_indexes, generator=generator)
        prior = self.compute_mlp_prior_batched(features, indexes)
        heads = [head_name] if head_name is not None else list(prior)
        return float(torch.stack([prior[h].view(n_indexes, -1).std(dim=0).mean() for h in heads]).mean())

    def get_learnable_epinet_params(self):
        """Returns parameters for the learnable MLP features and heads of the Epinet."""
        params = list(self.learnable_epinet_features.parameters())
        for head in self.learnable_epinet_heads.values():
            params.extend(list(head.parameters()))
        return params


class MLPEpinet(MLPEpinetMixin, nn.Module):
    """Standalone epinet over an arbitrary (N, feature_dim) feature tensor."""

    def __init__(self, feature_dim, epi_index_dim, head_names, hidden_dim=50, prior_hidden_dim=None,
                 device=torch.device("cpu")):
        nn.Module.__init__(self)
        self.epi_index_dim = epi_index_dim
        self.feature_dim = feature_dim
        self.head_names = list(head_names)
        self._build_mlp_epinet(feature_dim, epi_index_dim, self.head_names, hidden_dim,
                               prior_hidden_dim or hidden_dim, device)

    @property
    def device(self):
        """Where the weights are, so indices follow the module through .to(device)."""
        return self.prior_epinet_features[0].weight.device

    def sample(self, base, features, epi_indexes, head_name, alpha_mlp=1.0):
        """base (N,) + epinet correction for each index: (K, N)."""
        learnable = self.compute_learnable_mlp_batched(features, epi_indexes)[head_name]
        prior = self.compute_mlp_prior_batched(features, epi_indexes)[head_name]
        return base.reshape(1, -1) + (learnable + alpha_mlp * prior).view(epi_indexes.shape[0], -1)
