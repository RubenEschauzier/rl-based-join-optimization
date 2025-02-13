import torch
import torch.nn as nn

class TriplePatternPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriplePatternPooling, self).__init__()

    def forward(self, x, edge_index):
        """
        x: Node features (shape: [num_nodes, in_channels])
        batch: Batch indices mapping nodes to graphs (shape: [num_nodes])
        """
        # Triple pattern embedding is by iterating over batch edges
        tp_features = x[edge_index]
        tp_features = torch.sum(tp_features, dim=0, keepdim=False)

        return tp_features  # Apply projection layer