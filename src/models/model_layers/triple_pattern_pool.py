import torch
import torch.nn as nn

class TriplePatternPooling(nn.Module):
    def __init__(self):
        super(TriplePatternPooling, self).__init__()

    def forward(self, x, edge_index, batch):
        """
        x: Node features (shape: [num_nodes, in_channels])
        batch: Batch indices mapping nodes to graphs (shape: [num_nodes])
        """
        # Sort node pairs to canonicalize direction to remove duplicate edges in edge_index
        # edge_index_undirected = torch.unique(edge_index.sort(dim=0).values.T, dim=0).T

        # Fast removal of edges, assumes reverse edge immediately follows from edge
        edge_index_undirected = edge_index[:, ::2]

        tp_features = x[edge_index_undirected].sum(dim=0, keepdims=False)
        # Batch of edge is equal to batch of start node of the edges
        edge_batch = batch[edge_index_undirected[0]]
        return tp_features, edge_batch