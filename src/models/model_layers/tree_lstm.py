import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

#https://github.com/pyg-team/pytorch_geometric/issues/121
class ChildSumTreeLSTM(MessagePassing):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(aggr=None, flow='source_to_target')
        # Gates: W_x and U_h
        self.W_iou = nn.Linear(in_dim, 3 * out_dim)
        self.U_iou = nn.Linear(out_dim, 3 * out_dim)

        self.W_f = nn.Linear(in_dim, out_dim)
        self.U_f = nn.Linear(out_dim, out_dim)

        self.hidden_dim = out_dim

    def forward(self, x, edge_index, h, c, orders):
        for order in orders:
            mask = order[edge_index[1]]
            x = self.propagate(edge_index[:, mask], x=x, h=h, c=c, orders=orders)
        return x

    def message(self, x_j, x_i, h_j, c_j, index, orders):
        # x_i = target nodes an x_j = source nodes
        f = torch.nn.functional.sigmoid(
            self.U_f(h_j) +
            self.W_f(x_j)
        )
        f_c = torch.mul(f, c_j)
        return h_j, f_c

    def aggregate(self, inputs, index):
        # Hidden states and forget from graph
        h_j, f_c = inputs
        h_tilde = torch.zeros(index.max() + 1, self.hidden_dim, device=h_j.device)
        h_tilde = h_tilde.index_add(0, index, h_j)

        # ∑ f_{jk} * c_k
        f_c_sum = torch.zeros_like(h_tilde)
        f_c_sum = f_c_sum.index_add(0, index, f_c)
        return h_tilde, f_c_sum

    def update(self, aggr_out, x):
        h_tilde, f_c_sum = aggr_out
        print(h_tilde.shape)
        print(self.U_iou)
        print(x.shape)
        iou = self.W_iou(x) + self.U_iou(h_tilde)
        i, o, u = torch.chunk(iou, 3, dim=1)

        i, o = torch.sigmoid(i), torch.sigmoid(o)
        u = torch.tanh(u)
        c = torch.mul(i, u) + f_c_sum
        h = torch.mul(o, torch.tanh(c))
        return h, c

class NAryTreeLSTM(MessagePassing):
    def __init__(self, **kwargs):
        pass

if __name__ == '__main__':
    # Node features: [x0, x1, x2, x3]
    x_test = torch.tensor([
        [1.0, 0.0],  # Node 0
        [0.0, 1.0],  # Node 1
        [1.0, 1.0],  # Node 2
        [0.5, 0.5],  # Node 3
    ], dtype=torch.float)
    h_test = torch.zeros((4,2))
    c_test = torch.zeros((4,2))
    # Edges from child to parent (edge_index[0] → edge_index[1])
    # 0 → 1, 1 → 3, 2 → 3
    edge_index_test = torch.tensor([[0,1,2],[1,3,3]])
    order_child_propagate = [torch.tensor([1, 0, 0, 0], dtype=torch.bool), torch.tensor([0, 1, 1, 0], dtype=torch.bool)]
    order_parent_receive = [torch.tensor([0, 1, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 0, 1], dtype=torch.bool)]
    lstm = ChildSumTreeLSTM(2,2)
    lstm(x_test, edge_index_test, h=h_test, c=c_test, orders=order_parent_receive
       )