from abc import ABC
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class BatchedLowRankAdapters(Module):
    """Low-rank residual experts evaluated only for the selected routes."""

    def __init__(self, in_channels: int, out_channels: int, num_experts: int,
                 rank: int, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        self.expert_L = Parameter(torch.empty(num_experts, in_channels, rank))
        self.expert_R = Parameter(torch.empty(num_experts, rank, out_channels))
        self.bias = Parameter(torch.empty(num_experts, out_channels))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.expert_L, a=5 ** 0.5)
        torch.nn.init.zeros_(self.expert_R)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, expert_indices: Tensor, expert_weights: Tensor) -> Tensor:
        selected_L = self.expert_L[expert_indices]
        selected_R = self.expert_R[expert_indices]
        selected_bias = self.bias[expert_indices]

        low_rank = torch.einsum('nd,nkdr->nkr', x, selected_L)
        expert_output = torch.einsum('nkr,nkro->nko', low_rank, selected_R)
        expert_output = self.dropout(expert_output) + selected_bias
        return torch.sum(expert_weights.unsqueeze(-1) * expert_output, dim=1)


class TripleGineConvLowRankMoE(MessagePassing, ABC):
    """Triple GINE with a frozen dense path and lightweight residual experts."""

    def __init__(self, nn: torch.nn.Module, num_experts: int = 4,
                 rank: int = 4, top_k: int = 2, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 routing_level: str = 'query', router_temperature: float = 1.0,
                 adapter_scale: float = 1.0, adapter_dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if not 1 <= top_k <= num_experts:
            raise ValueError(f'top_k must be between 1 and num_experts, got {top_k}')
        if routing_level not in {'node', 'query'}:
            raise ValueError(f"routing_level must be 'node' or 'query', got {routing_level}")
        if router_temperature <= 0:
            raise ValueError('router_temperature must be positive')

        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.routing_level = routing_level
        self.router_temperature = router_temperature
        self.adapter_scale = adapter_scale
        self.initial_eps = eps
        self.DIRECTIONAL = True

        if train_eps:
            self.eps = Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

        if isinstance(nn, torch.nn.Sequential):
            first_layer = nn[0]
            last_layer = next(layer for layer in reversed(nn) if hasattr(layer, 'out_features'))
        else:
            first_layer = nn
            last_layer = nn

        if not hasattr(first_layer, 'in_features'):
            raise ValueError('Could not infer input channels from `nn`.')

        in_channels = first_layer.in_features
        out_channels = last_layer.out_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(edge_dim + 2 * in_channels, in_channels) if edge_dim is not None else None
        self.W0 = nn
        self.experts = BatchedLowRankAdapters(
            in_channels, out_channels, num_experts, rank, dropout=adapter_dropout
        )
        self.router = Linear(in_channels, num_experts)
        self.current_routing_probs: Optional[Tensor] = None
        self.current_routing_assignments: Optional[Tensor] = None
        self._online_adaptation_enabled = False
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        reset(self.W0)
        self.experts.reset_parameters()
        self.router.reset_parameters()

    def enable_online_adaptation(self, adapt_router: bool = True):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.experts.parameters():
            parameter.requires_grad = True
        if adapt_router:
            for parameter in self.router.parameters():
                parameter.requires_grad = True
        self._online_adaptation_enabled = True
        self.W0.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._online_adaptation_enabled:
            self.W0.eval()
        return self

    def freeze_backbone(self):
        self.enable_online_adaptation(adapt_router=True)

    def online_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _routing_context(self, out: Tensor, batch: Optional[Tensor]):
        if self.routing_level == 'query' and batch is not None:
            return global_mean_pool(out, batch), batch
        return out, None

    def _route(self, out: Tensor, batch: Optional[Tensor]):
        context, node_to_context = self._routing_context(out, batch)
        routing_scores = self.router(context) / self.router_temperature
        context_probs = F.softmax(routing_scores, dim=-1)
        topk_scores, context_indices = torch.topk(routing_scores, self.top_k, dim=-1)
        context_weights = F.softmax(topk_scores, dim=-1)

        hard_assignments = torch.zeros_like(context_probs)
        hard_assignments.scatter_(1, context_indices, 1.0 / self.top_k)
        self.current_routing_probs = context_probs
        self.current_routing_assignments = hard_assignments

        if node_to_context is not None:
            return context_indices[node_to_context], context_weights[node_to_context]
        return context_indices, context_weights

    def load_balancing_loss(self) -> Tensor:
        if self.current_routing_probs is None or self.current_routing_assignments is None:
            raise RuntimeError('A forward pass is required before computing routing loss')
        probability_mass = self.current_routing_probs.mean(dim=0)
        hard_usage = self.current_routing_assignments.mean(dim=0).detach()
        return self.num_experts * torch.sum(probability_mass * hard_usage)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, batch: Optional[Tensor] = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        expert_indices, expert_weights = self._route(out, batch)
        dense_output = self.W0(out)
        adapter_output = self.experts(out, expert_indices, expert_weights)
        return dense_output + self.adapter_scale * adapter_output

    def message(self, x_i, x_j, edge_attr: Tensor) -> Tensor:
        reverse = edge_attr[:, -1] == -1
        if self.DIRECTIONAL:
            x_i[reverse], x_j[reverse] = x_j[reverse], x_i[reverse]

        edge_attr = edge_attr[:, :-1]
        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def get_current_routing_probs(self) -> Tensor:
        return self.current_routing_probs

    def get_current_routing_assignments(self) -> Tensor:
        return self.current_routing_assignments

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(W0={self.W0}, experts={self.num_experts}, top_k={self.top_k})'
