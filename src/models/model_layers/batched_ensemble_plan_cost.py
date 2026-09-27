"""Vectorized evaluation of the randomized-prior ensemble.

``MultiHeadEpistemicNetwork`` holds ``epi_index_dim`` structurally identical
``PlanCostEstimatorTiny`` members and evaluates them in a Python loop. Profiling says
that loop is ~56% of all epinet time and ~95% together with the matching loop over the
prior GNNs -- not because the members are large (they are ~100x smaller than the base
model) but because each one issues its own microsecond-sized kernels. The base model
pushes 80 plans through in 3.6 ms; thirty tiny members take 73.8 ms.

deepmind/enn avoids this by construction: its priors are a single batched einsum
(``einsum_mlp.EnsembleMLP``), never a loop over modules. This module is the same idea in
PyTorch, done with grouped convolutions and batched matmuls so it needs neither
``torch.func.vmap`` (unavailable on torch 1.x, and awkward over PyG's scatter ops) nor
any change to how the ensemble is built or checkpointed.

Construct with ``from_members`` and verify with ``max_deviation_from_loop``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedEnsemblePlanCost(nn.Module):
    """Runs an ensemble of PlanCostEstimatorTiny members as a handful of grouped ops."""

    def __init__(self, n_members: int, head_name: str = "plan_cost"):
        super().__init__()
        self.n_members = n_members
        self.head_name = head_name
        self.conv_channels: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ build
    @classmethod
    def from_members(cls, members, head_name: str = "plan_cost") -> "BatchedEnsemblePlanCost":
        """Stack the parameters of already-constructed members. Weights are copied, so
        the source modules stay authoritative for checkpointing."""
        batched = cls(len(members), head_name)
        first = members[0]

        convs = [layer for layer in first.plan_embedding_nn if hasattr(layer, "weights")]
        batched.conv_channels = [
            (conv.weights.in_channels, conv.weights.out_channels) for conv in convs
        ]
        for depth, _ in enumerate(convs):
            per_member = [
                [layer for layer in member.plan_embedding_nn if hasattr(layer, "weights")][depth]
                for member in members
            ]
            # grouped conv1d wants (groups*out, in, kernel)
            batched.register_buffer(
                f"conv_weight_{depth}",
                torch.cat([conv.weights.weight for conv in per_member], dim=0).contiguous(),
                persistent=False,
            )
            batched.register_buffer(
                f"conv_bias_{depth}",
                torch.cat([conv.weights.bias for conv in per_member], dim=0).contiguous(),
                persistent=False,
            )

        def stack(pick):
            return torch.stack([pick(member) for member in members], dim=0).contiguous()

        gate = lambda member: member.attn_pool.gate_nn
        batched.register_buffer("gate_weight_0", stack(lambda m: gate(m)[0].weight), persistent=False)
        batched.register_buffer("gate_bias_0", stack(lambda m: gate(m)[0].bias), persistent=False)
        batched.register_buffer("gate_weight_1", stack(lambda m: gate(m)[2].weight), persistent=False)
        batched.register_buffer("gate_bias_1", stack(lambda m: gate(m)[2].bias), persistent=False)
        batched.register_buffer("mlp_weight", stack(lambda m: m.mlp[0].weight), persistent=False)
        batched.register_buffer("mlp_bias", stack(lambda m: m.mlp[0].bias), persistent=False)
        batched.register_buffer("head_weight", stack(lambda m: m.heads[head_name].weight), persistent=False)
        batched.register_buffer("head_bias", stack(lambda m: m.heads[head_name].bias), persistent=False)
        return batched

    # ---------------------------------------------------------------- forward
    def _grouped_tree_conv(self, trees, indexes, depth):
        """One BinaryTreeConv applied to every member at once (groups=n_members)."""
        in_channels = self.conv_channels[depth][0] * self.n_members
        gather_index = indexes.expand(-1, -1, in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, gather_index)
        results = F.conv1d(
            expanded,
            getattr(self, f"conv_weight_{depth}"),
            getattr(self, f"conv_bias_{depth}"),
            stride=3,
            groups=self.n_members,
        )
        # BinaryTreeConv prepends a zero vector for the padding node
        zeros = results.new_zeros((results.shape[0], results.shape[1], 1))
        return torch.cat((zeros, results), dim=2)

    def _grouped_layer_norm(self, data):
        """TreeLayerNorm, normalising each member separately rather than across all."""
        n_plans, _, width = data.shape
        grouped = data.view(n_plans, self.n_members, -1, width)
        mean = grouped.mean(dim=(2, 3), keepdim=True)
        std = grouped.std(dim=(2, 3), keepdim=True)
        return ((grouped - mean) / (std + 0.00001)).view(n_plans, -1, width)

    def forward(self, trees, indexes, mask_padding):
        """trees: (n_plans, n_members * in_channels, width) -> (n_members, n_plans)."""
        hidden = trees
        for depth in range(len(self.conv_channels)):
            hidden = self._grouped_tree_conv(hidden, indexes, depth)
            if depth < len(self.conv_channels) - 1:
                hidden = torch.relu(self._grouped_layer_norm(hidden))

        n_plans, _, width = hidden.shape
        feature_dim = self.conv_channels[-1][1]
        # (n_plans, width, n_members, feature_dim)
        nodes = hidden.view(n_plans, self.n_members, feature_dim, width).permute(0, 3, 1, 2)

        gate = torch.einsum("pwmf,mhf->pwmh", nodes, self.gate_weight_0) + self.gate_bias_0
        gate = torch.relu(gate)
        gate = torch.einsum("pwmh,moh->pwmo", gate, self.gate_weight_1) + self.gate_bias_1

        # Attention is taken over valid nodes only, exactly as masking rows out before
        # AttentionalAggregation does.
        valid = (~mask_padding).view(n_plans, width, 1, 1)
        weights = torch.softmax(gate.masked_fill(~valid, float("-inf")), dim=1)
        pooled = (weights * nodes).sum(dim=1)                      # (n_plans, members, dim)

        root = nodes[:, 1, :, :]                                   # (n_plans, members, dim)
        combined = torch.cat([root, pooled], dim=2)

        hidden = torch.einsum("pmf,mhf->pmh", combined, self.mlp_weight) + self.mlp_bias
        hidden = torch.relu(hidden)
        out = torch.einsum("pmh,moh->pmo", hidden, self.head_weight) + self.head_bias
        return out.squeeze(-1).transpose(0, 1)                     # (n_members, n_plans)


@torch.no_grad()
def max_deviation_from_loop(batched, members, trees_list, indexes, masks,
                            head_name: str = "plan_cost") -> float:
    """Largest absolute difference against the original per-member loop."""
    reference = torch.stack([
        member.forward(trees_list[i], indexes, masks)[0][head_name].view(-1)
        for i, member in enumerate(members)
    ])
    stacked = torch.cat(trees_list, dim=1)
    return (batched(stacked, indexes, masks) - reference).abs().max().item()
