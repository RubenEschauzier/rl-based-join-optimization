from abc import ABC
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, OptTensor, Size, Adj


class DirectionalGINEConv(MessagePassing, ABC):
    r"""
    Message passing function for directional message passing
    based on the GINE Conv operator
    Equation:
        .. math::
             x_i^{(k)} = h_\theta^{(k)}  \biggl( x_i^{(k-1)} \ +& \sum_{j \in \mathcal{N}^+(i)}
            \mathrm{ReLU}(x_i^{(k-1)}||e^{j,i}||x_j^{(k-1)}) \ +\\
            & \sum_{j \in \mathcal{N}^-(i)}
            \mathrm{ReLU}(x_j^{(k-1)}||e^{i,j}||x_i^{(k-1)}) \biggr)


    The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: torch.nn.Module, edge_dim: int, eps: float = 0.,
                 train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if isinstance(self.nn, torch.nn.Sequential):
            nn = self.nn[0]
        if hasattr(nn, 'in_features'):
            in_channels = nn.in_features
        elif hasattr(nn, 'in_channels'):
            in_channels = nn.in_channels
        else:
            raise ValueError("Could not infer input channels from `nn`.")

        # Separate transformation of incoming and outgoing edges
        self.lin_1 = Linear(edge_dim, in_channels)
        self.lin_2 = Linear(edge_dim, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin_1 is not None:
            self.lin_1.reset_parameters()
        if self.lin_2 is not None:
            self.lin_2.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # Switch i and j based on direction of triple:
        mask = edge_attr[:, -1] == -1

        edge_attr = edge_attr[:, :-1]

        # Apply different linear layers based on the mask
        edge_attr_transformed = torch.empty(size = x_j.shape,
                                            dtype=edge_attr.dtype, device=edge_attr.device,
                                            layout=edge_attr.layout)

        edge_attr_transformed[mask] = self.lin_1(edge_attr[mask])
        edge_attr_transformed[~mask] = self.lin_2(edge_attr[~mask])

        return (x_j + edge_attr_transformed).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'