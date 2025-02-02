import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class TriplePatternPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriplePatternPooling, self).__init__()
        self.attn = nn.Linear(in_channels, 1)  # Attention mechanism
        self.proj = nn.Linear(in_channels, out_channels)  # Projection layer

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: Node features (shape: [num_nodes, in_channels])
        batch: Batch indices mapping nodes to graphs (shape: [num_nodes])
        """
        # Triple pattern embedding is by iterating over batch edges
        print(edge_index)
        print(batch)
        # Compute attention scores
        attn_weights = self.attn(x)  # [num_nodes, 1]
        attn_weights = softmax(attn_weights, batch)  # Normalize over the batch

        # Weighted sum of node features per graph
        x_pooled = torch.zeros(batch.max().item() + 1, x.size(1), device=x.device)
        x_pooled = x_pooled.index_add(0, batch, attn_weights * x)

        return self.proj(x_pooled)  # Apply projection layer