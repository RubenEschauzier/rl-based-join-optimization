from abc import ABC

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

#https://github.com/pyg-team/pytorch_geometric/issues/121
class ChildSumTreeLSTM(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(aggr=None, flow='source_to_target')
        # Gates: W_x and U_h
        self.W_iou = nn.Linear(in_dim, 3 * out_dim)
        self.U_iou = nn.Linear(out_dim, 3 * out_dim)

        self.W_f = nn.Linear(in_dim, out_dim)
        self.U_f = nn.Linear(out_dim, out_dim)

        self.hidden_dim = out_dim
        self.partial_dense_mapping = {}

    def forward(self, x, edge_index, h, c, orders):
        for order in orders:
            self.partial_dense_mapping = {}

            mask = order[edge_index[1]]
            masked_graph = edge_index[:, mask]
            i = 0
            for target in masked_graph[1]:
                if int(target) not in self.partial_dense_mapping:
                    self.partial_dense_mapping[int(target)] = i
                    i += 1
            masked_h_prop, masked_c_prop = self.propagate(edge_index[:, mask], x=x, h=h, c=c, orders=orders)

            updated_nodes = list(self.partial_dense_mapping.keys())
            h[updated_nodes] = masked_h_prop
            c[updated_nodes] = masked_c_prop
        return h, c

    def message(self, x_j, x_i, h_j, c_j, index):
        # x_i = target nodes an x_j = source nodes
        f = torch.nn.functional.sigmoid(
            self.U_f(h_j) +
            self.W_f(x_i)
        )
        f_c = torch.mul(f, c_j)
        return h_j, f_c

    def aggregate(self, inputs, index):
        # Hidden states and forget from graph
        h_j, f_c = inputs
        mapped_index = torch.tensor([self.partial_dense_mapping[int(i.item())] for i in index], device=index.device)
        h_tilde = torch.zeros(mapped_index.max() + 1, self.hidden_dim, device=h_j.device)
        h_tilde = h_tilde.index_add(0, mapped_index, h_j)

        # ∑ f_{jk} * c_k
        f_c_sum = torch.zeros_like(h_tilde)
        f_c_sum = f_c_sum.index_add(0, mapped_index, f_c)
        return h_tilde, f_c_sum

    def update(self, aggr_out, x):
        h_tilde, f_c_sum = aggr_out
        # Take all x of nodes that will be updated in this call
        x_to_update = x[list(self.partial_dense_mapping.keys())]
        i, o, u = self.get_iou(x_to_update, h_tilde)
        return self.get_h_c(i, o, u, f_c_sum)

    def get_iou(self, x, h_tilde):
        iou = self.W_iou(x) + self.U_iou(h_tilde)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o = torch.sigmoid(i), torch.sigmoid(o)
        u = torch.tanh(u)
        return i, o, u

    @staticmethod
    def get_h_c(i, o, u, f_c_sum):
        c = torch.mul(i, u) + f_c_sum
        h = torch.mul(o, torch.tanh(c))
        return h, c

    def set_weights(self, weights_dict):
        """
        Set the weights in tree-lstm to given values. Used for testing purposes
        :param weights_dict: dictionary mapping variable names to weight tensors
        :return:
        """
        with torch.no_grad():
            if 'W_iou' in weights_dict:
                self.W_iou.weight.copy_(weights_dict['W_iou'][0])
                self.W_iou.bias.copy_(weights_dict['W_iou'][1])
            if 'U_iou' in weights_dict:
                self.U_iou.weight.copy_(weights_dict['U_iou'][0])
                self.U_iou.bias.copy_(weights_dict['U_iou'][1])
            if 'W_f' in weights_dict:
                self.W_f.weight.copy_(weights_dict['W_f'][0])
                self.W_f.bias.copy_(weights_dict['W_f'][1])
            if 'U_f' in weights_dict:
                self.U_f.weight.copy_(weights_dict['U_f'][0])
                self.U_f.bias.copy_(weights_dict['U_f'][1])


class NAryTreeLSTM(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, max_children, **kwargs):
        super().__init__(aggr=None, flow='source_to_target')
        self.W_iou = nn.Linear(in_dim, 3 * out_dim)
        # Individual U_iou for each child hidden state
        self.U_iou = nn.Parameter(torch.empty(max_children, out_dim, 3*out_dim), requires_grad=True)

        self.W_f = nn.Linear(in_dim, out_dim)
        # Fix the forget gate off-diagonal matrices to save space otherwise 16 matrices are needed.
        # Parameter tensor with size (max_child, in_dim, out_dim)
        self.U_f = nn.Parameter(torch.empty(max_children, out_dim, out_dim), requires_grad=True)

        self.hidden_dim = out_dim
        self.partial_dense_mapping = {}

    def forward(self, x, edge_index, h, c, orders):
        for order in orders:
            self.partial_dense_mapping = {}
            mask = order[edge_index[1]]
            masked_graph = edge_index[:, mask]
            i = 0
            for target in masked_graph[1]:
                if int(target) not in self.partial_dense_mapping:
                    self.partial_dense_mapping[int(target)] = i
                    i += 1
            masked_h_prop, masked_c_prop = self.propagate(edge_index[:, mask], x=x, h=h, c=c, orders=orders)

            updated_nodes = list(self.partial_dense_mapping.keys())
            h[updated_nodes] = masked_h_prop
            c[updated_nodes] = masked_c_prop
        return h, c


    def message(self, x_j, x_i, h_j, c_j, index):
        """
        In message, we partially compute i, o, u by doing the sum of hidden states
        times a weight tensor.
        Additionally, we compute the f_c elements in the sum for c_j
        :param x_j:
        :param x_i:
        :param h_j:
        :param c_j:
        :param index:
        :return:
        """
        # x_i = target nodes an x_j = source nodes
        # Chunked output for 1 chunk gives tuple with size 0
        linear_result = self.batched_chunked_linear(
            weight_x = self.W_f, weight_h = self.U_f,
            input_x = x_i, input_h = h_j,
            chunks=1, dim=1
        )[0]
        f = torch.nn.functional.sigmoid(linear_result)
        f_c = torch.mul(f, c_j)

        p_iou = NAryTreeLSTM.batched_linear(self.U_iou, h_j)

        return h_j, f_c, p_iou, x_i


    def aggregate(self, inputs, index):
        # Hidden states and forget from graph
        h_j, f_c, p_iou, x_i = inputs
        mapped_index = torch.tensor([self.partial_dense_mapping[int(i.item())] for i in index], device=index.device)
        n_parents = int(mapped_index.max() + 1)

        # Sum part of calculation of i o u
        p_iou_sum = torch.zeros(n_parents, self.hidden_dim*3, device=p_iou.device)
        p_iou_sum = p_iou_sum.index_add(0, mapped_index, p_iou)

        # ∑ f_{jk} * c_k
        f_c_sum = torch.zeros_like(torch.zeros(n_parents, self.hidden_dim, device=h_j.device))
        f_c_sum = f_c_sum.index_add(0, mapped_index, f_c)
        return f_c_sum, p_iou_sum


    #TODO Test this
    def update(self, aggr_out, x):
        f_c_sum, p_iou_sum = aggr_out
        # Take all x of nodes that will be updated in this call
        x_to_update = x[list(self.partial_dense_mapping.keys())]

        i, o, u = torch.chunk(self.W_iou(x_to_update) + p_iou_sum, chunks=3, dim=1)
        i, o = torch.sigmoid(i), torch.sigmoid(o)
        u = torch.tanh(u)

        return self.get_h_c(i, o, u, f_c_sum)

    @staticmethod
    def batched_chunked_linear(weight_x: nn.Linear, weight_h: nn.Parameter,
                               input_x: torch.Tensor, input_h: torch.Tensor,
                               chunks: int, dim: int):
        iou = weight_x(input_x) + NAryTreeLSTM.batched_linear(weight_h, input_h)
        chunked = torch.chunk(iou, chunks, dim=dim)
        return chunked

    @staticmethod
    def get_h_c(i, o, u, f_c_sum):
        c = torch.mul(i, u) + f_c_sum
        h = torch.mul(o, torch.tanh(c))
        return h, c

    @staticmethod
    def batched_linear(weight: nn.Parameter, linear_input : torch.Tensor):
        return torch.bmm(linear_input.unsqueeze(1), weight).squeeze()

    def set_weights(self, weights_dict):
        """
        Set the weights in tree-lstm to given values. Used for testing purposes
        :param weights_dict: dictionary mapping variable names to weight tensors
        :return:
        """
        with torch.no_grad():
            if 'W_iou' in weights_dict:
                self.W_iou.weight.copy_(weights_dict['W_iou'][0])
                self.W_iou.bias.copy_(weights_dict['W_iou'][1])
            if 'U_iou' in weights_dict:
                self.U_iou = nn.Parameter(weights_dict['U_iou'][0])
            if 'W_f' in weights_dict:
                self.W_f.weight.copy_(weights_dict['W_f'][0])
                self.W_f.bias.copy_(weights_dict['W_f'][1])
            if 'U_f' in weights_dict:
                self.U_f = nn.Parameter(weights_dict['U_f'][0])




if __name__ == '__main__':
    # Node features: [x0, x1, x2, x3]
    x_test = torch.tensor([
        [1.0, 0.0],  # Node 0
        [.25, .25],  # Node 1
        [0.0, 1.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [0.5, 0.5],  # Node 4
    ], dtype=torch.float)
    h_test = torch.zeros((5,2))
    c_test = torch.zeros((5,2))
    # Edges from child to parent (edge_index[0] → edge_index[1])
    # 0 → 1, 1 → 3, 2 → 3
    edge_index_test = torch.tensor([[0,1,2,3],[2,2,4,4]])
    order_child_propagate = [torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool)]
    order_parent_receive = [torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool), torch.tensor([0, 0, 0, 0, 1], dtype=torch.bool)]
    lstm = ChildSumTreeLSTM(2,2)
    lstm(x_test, edge_index_test, h=h_test, c=c_test, orders=order_parent_receive
       )