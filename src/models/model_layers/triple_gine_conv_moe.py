from abc import ABC

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, Module
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from typing import Optional, Union
from torch_geometric.typing import OptPairTensor, OptTensor, Size, Adj

# TODO: This is ALL chatgpt generated, so we need to validate this
class BatchedKroneckerAdapters(Module):
    """
    Vectorized Kronecker adapters using Parameter-efficient Hypercomplex Multiplication (PHM).
    Constructs all expert update matrices simultaneously.
    """

    def __init__(self, in_channels: int, out_channels: int, num_experts: int, d_phm: int, rank: int):
        super().__init__()
        self.num_experts = num_experts
        self.d_phm = d_phm

        # Ensure dimensions are divisible by d_phm
        if not in_channels % d_phm == 0 or not out_channels % d_phm == 0:
            raise ValueError(f"in_channels: {in_channels} % d_phm: {d_phm} != 0 "
                             f"or out_channels: {out_channels} % d_phm: {d_phm} != 0")
        self.in_local = in_channels // d_phm
        self.out_local = out_channels // d_phm

        # Shared PHM factor [cite: 79]
        self.shared_factor = Parameter(torch.Tensor(d_phm, d_phm, d_phm))

        # Expert-specific low-rank factors [cite: 80, 81]
        self.expert_L = Parameter(torch.Tensor(num_experts, d_phm, self.in_local, rank))
        self.expert_R = Parameter(torch.Tensor(num_experts, d_phm, rank, self.out_local))

        # Expert specific biases [cite: 70]
        self.bias = Parameter(torch.Tensor(num_experts, out_channels))
        self.dropout = torch.nn.Dropout(p=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.shared_factor, std=0.02)
        torch.nn.init.normal_(self.expert_L, std=0.02)
        torch.nn.init.zeros_(self.expert_R)
        torch.nn.init.zeros_(self.bias)

    def get_expert_matrices(self) -> Tensor:
        # Reconstruct full weight matrices for all experts simultaneously
        # H_e = sum_k S_k (x) (L_{e,k} R_{e,k}) [cite: 77, 81]

        # 1. Compute low-rank expert factors: (E, d_phm, in_local, out_local)
        expert_factors = torch.einsum('ekir, ekro -> ekio', self.expert_L, self.expert_R)

        # 2. Compute Kronecker product via einsum and reshape
        # S: (d_phm, d_phm, d_phm) -> use first dim as sum index k
        h = torch.einsum('kab, ekio -> eaibko', self.shared_factor, expert_factors)

        # Reshape to final (E, D_in, D_out)
        num_experts = self.num_experts
        d_in = self.d_phm * self.in_local
        d_out = self.d_phm * self.out_local
        print(h)
        h = h.reshape(num_experts, d_in, d_out)

        return self.dropout(h)


class TripleGineConvMoE(MessagePassing, ABC):
    def __init__(self, nn: torch.nn.Module, num_experts: int = 4,
                 d_phm: int = 4, rank: int = 4, top_k: int = 3, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.top_k = top_k
        self.initial_eps = eps
        self.DIRECTIONAL = True

        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        # Infer input/output channels from the provided neural network
        if isinstance(nn, torch.nn.Sequential):
            first_layer = nn[0]
            last_layer = nn[-1] if hasattr(nn[-1], 'out_features') else nn[-2]
        else:
            first_layer = nn
            last_layer = nn

        if hasattr(first_layer, 'in_features'):
            in_channels = first_layer.in_features
        elif hasattr(first_layer, 'in_channels'):
            in_channels = first_layer.in_channels
        else:
            raise ValueError("Could not infer input channels from `nn`.")

        out_channels = last_layer.out_features

        self.in_channels = in_channels

        if edge_dim is not None:
            self.lin = Linear(edge_dim + 2 * in_channels, in_channels)
        else:
            self.lin = None

        # Frozen backbone component [cite: 61]
        self.W0 = nn

        # MoE Components
        self.experts = BatchedKroneckerAdapters(in_channels, out_channels, num_experts, d_phm, rank)

        # Router: Projects contextualized node embeddings to expert scores [cite: 98]
        self.router = Linear(in_channels, num_experts)
        self.current_routing_probs = None

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        reset(self.W0)
        self.experts.reset_parameters()
        self.router.reset_parameters()

    def freeze_backbone(self):
        """Freezes the main neural network pathway to prevent catastrophic forgetting."""
        for param in self.W0.parameters():
            param.requires_grad = False

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # 1. Neighborhood Aggregation -> Computes H^{context} [cite: 96]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        # 2. Routing Mechanism [cite: 98]
        routing_scores = self.router(out)
        self.current_routing_probs = F.softmax(routing_scores, dim=-1)

        if self.top_k > 0:
            # Sparse Routing (Top-K)
            topk_scores, topk_indices = torch.topk(routing_scores, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_scores, dim=-1)

            # Create a zero mask and scatter the top-k weights back
            routing_weights = torch.zeros_like(routing_scores)
            routing_weights.scatter_(1, topk_indices, topk_weights)
        else:
            # Soft Routing (Dense) [cite: 101, 102]
            routing_weights = self.current_routing_probs

        # 3. Expert Evaluation (Vectorized)
        h_experts = self.experts.get_expert_matrices()  # Shape: (E, D_in, D_out)

        # Frozen pathway output [cite: 61]
        frozen_out = self.W0(out)

        # Calculate MoE output in one step
        moe_out = torch.einsum('ne, nd, edf -> nf', routing_weights, out, h_experts)
        moe_bias = torch.einsum('ne, ef -> nf', routing_weights, self.experts.bias)

        # Combine frozen and MoE pathways [cite: 70]
        return frozen_out + moe_out + moe_bias

    def message(self, x_i, x_j, edge_attr: Tensor) -> Tensor:
        reverse = edge_attr[:, -1] == -1

        if self.DIRECTIONAL:
            x_i[reverse], x_j[reverse] = x_j[reverse], x_i[reverse]

        edge_attr = edge_attr[:, :-1]
        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def get_current_routing_probs(self) -> Tensor:
        return self.current_routing_probs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(W0={self.W0})'