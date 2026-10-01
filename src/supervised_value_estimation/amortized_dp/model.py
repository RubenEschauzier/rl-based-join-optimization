"""Value network over the contracted join graph.

State = the SET S of triple patterns joined so far. After S is joined, the optimal cost of
finishing depends only on which patterns were joined, not on the order they were joined
in (the Markov property DP exploits), so the state is encoded order-invariantly:

    * the joined patterns are contracted into ONE super-node, encoded with DeepSets
      (sum of phi over members, then rho), plus an embedding of |S|;
    * the patterns still to join stay ordinary nodes, keeping their join edges; the
      super-node is connected to every remaining pattern adjacent to any member of S;
    * a few rounds of mean-aggregation message passing run on this contracted graph.

Two heads, matching the DP recursion  Q(S, a) = card(S u a) + G(S u a):
    card head  log card(S)  from the set encoding alone -- a pure set function, so it
               cannot depend on the order S was built in or on the rest of the query;
    G head     log G(S), the optimal cost-to-go, from the contextual super-node, the
               pooled remaining patterns and the fraction of the query left to join.

Message passing (`message_layer`), all dense over the <= n+1 contracted nodes:
    mean   node <- LN(node + MLP([node, mean of neighbours]))          (no edge features)
    gine   node <- LN(node + MLP((1 + eps) node + sum_u ReLU(h_u + W e_uv)))   (GINEConv)
    gat    node <- LN(node + MLP([node, sum_u a_vu (W h_u + W' e_uv)])), 4-head attention
           a_vu = softmax_u( <W_q h_v, W_k h_u + W_k' e_uv> / sqrt(d) ) over neighbours
Edge features e_uv are the join-position counts of labels.join_position_features; the
super-node's edge to a pattern sums them over the members of S. mean ignores them.

Both heads predict standardised targets; `unstandardise` maps them back to log space.
Inputs are dense and padded to the largest query in the batch (queries have <= ~20
patterns), which keeps every step a handful of batched matmuls.
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim, hidden, out_dim, layers=2):
    modules, dim = [], in_dim
    for _ in range(layers - 1):
        modules += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = hidden
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


MESSAGE_LAYERS = ("mean", "gine", "gat")


class _GINELayer(nn.Module):
    def __init__(self, hidden_dim, edge_feature_dim):
        super().__init__()
        self.edge_projection = nn.Linear(edge_feature_dim, hidden_dim) if edge_feature_dim else None
        self.eps = nn.Parameter(torch.zeros(1))
        self.update = _mlp(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, features, adjacency, edge_features):
        messages = features.unsqueeze(1).expand(-1, features.shape[1], -1, -1)     # [b, v, u] = h_u
        if self.edge_projection is not None:
            messages = messages + self.edge_projection(edge_features)
        aggregated = (torch.relu(messages) * adjacency.unsqueeze(-1)).sum(dim=2)
        return self.update((1 + self.eps) * features + aggregated)


class _GATLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feature_dim, n_heads=4):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, hidden_dim // n_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(edge_feature_dim, hidden_dim) if edge_feature_dim else None
        self.edge_value = nn.Linear(edge_feature_dim, hidden_dim) if edge_feature_dim else None
        self.update = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, features, adjacency, edge_features):
        batch, nodes, hidden = features.shape
        split = lambda x: x.view(*x.shape[:-1], self.n_heads, self.head_dim)
        query = split(self.query(features))                                        # (b, v, h, d)
        key = self.key(features).unsqueeze(1).expand(-1, nodes, -1, -1)           # [b, v, u]
        value = self.value(features).unsqueeze(1).expand(-1, nodes, -1, -1)
        if self.edge_key is not None:
            key = key + self.edge_key(edge_features)
            value = value + self.edge_value(edge_features)
        key, value = split(key), split(value)                                       # (b, v, u, h, d)
        scores = (query.unsqueeze(2) * key).sum(-1) / self.head_dim ** 0.5          # (b, v, u, h)
        scores = scores.masked_fill(~adjacency.bool().unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=2).nan_to_num(0.0)                      # isolated nodes -> 0
        aggregated = (weights.unsqueeze(-1) * value).sum(dim=2).reshape(batch, nodes, hidden)
        return self.update(torch.cat([features, aggregated], dim=-1))


class ContractedJoinGraphValueNet(nn.Module):
    def __init__(self, embedding_dim=200, hidden_dim=128, n_message_layers=3, max_triple_patterns=32,
                 message_layer="mean", edge_feature_dim=0, latency_head=False):
        super().__init__()
        if message_layer not in MESSAGE_LAYERS:
            raise ValueError(f"message_layer must be one of {MESSAGE_LAYERS}, got {message_layer!r}")
        self.hidden_dim = hidden_dim
        self.max_triple_patterns = max_triple_patterns
        self.message_layer = message_layer
        # The mean layer takes no edge features; keep the dimension 0 so callers skip them.
        self.edge_feature_dim = 0 if message_layer == "mean" else edge_feature_dim
        self.input_projection = _mlp(embedding_dim, hidden_dim, hidden_dim)
        self.set_phi = _mlp(hidden_dim, hidden_dim, hidden_dim)
        self.set_rho = _mlp(hidden_dim, hidden_dim, hidden_dim)
        self.size_embedding = nn.Embedding(max_triple_patterns + 1, hidden_dim)
        if message_layer == "mean":
            # Module names unchanged from the original model, so its checkpoints still load.
            self.message_layers = nn.ModuleList(
                [_mlp(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(n_message_layers)]
            )
        elif message_layer == "gine":
            self.message_layers = nn.ModuleList(
                [_GINELayer(hidden_dim, self.edge_feature_dim) for _ in range(n_message_layers)])
        else:
            self.message_layers = nn.ModuleList(
                [_GATLayer(hidden_dim, self.edge_feature_dim) for _ in range(n_message_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_message_layers)])
        self.card_head = _mlp(hidden_dim, hidden_dim, 1)
        self.cost_to_go_head = _mlp(3 * hidden_dim + 1, hidden_dim, 1)
        # Target standardisation, fitted on the training labels (see set_target_statistics).
        self.register_buffer("target_statistics", torch.tensor([0.0, 1.0, 0.0, 1.0]))
        # Optional latency cost-to-go head, for online training on real executions. It sees the
        # same context as the C_out head plus the model's own card and C_out predictions:
        # latency is largely a function of intermediate sizes, so it only has to learn the
        # correction on top of them. Separate buffer so older checkpoints still load.
        self.has_latency_head = latency_head
        if latency_head:
            self.latency_head = _mlp(3 * hidden_dim + 3, hidden_dim, 1)
            self.register_buffer("latency_statistics", torch.tensor([0.0, 1.0]))

    def set_target_statistics(self, card_mean, card_std, cost_to_go_mean, cost_to_go_std):
        self.target_statistics.copy_(torch.tensor(
            [card_mean, max(card_std, 1e-6), cost_to_go_mean, max(cost_to_go_std, 1e-6)]))

    def unstandardise(self, card, cost_to_go):
        card_mean, card_std, g_mean, g_std = self.target_statistics
        return card * card_std + card_mean, cost_to_go * g_std + g_mean

    def standardise(self, card, cost_to_go):
        card_mean, card_std, g_mean, g_std = self.target_statistics
        return (card - card_mean) / card_std, (cost_to_go - g_mean) / g_std

    def set_latency_statistics(self, mean, std):
        self.latency_statistics.copy_(torch.tensor([mean, max(std, 1e-6)]))

    def standardise_latency(self, latency):
        mean, std = self.latency_statistics
        return (latency - mean) / std

    def unstandardise_latency(self, latency):
        mean, std = self.latency_statistics
        return latency * std + mean

    def forward(self, embeddings, pattern_mask, adjacency, state, edge_features=None, return_latency=False):
        """
        embeddings:    (B, n, E) frozen triple-pattern embeddings, zero-padded
        pattern_mask:  (B, n) bool, True for real patterns
        adjacency:     (B, n, n) bool join graph (patterns sharing a variable)
        state:         (B, n) bool, True for patterns in S (the joined set)
        edge_features: (B, n, n, F) join-position features; required when edge_feature_dim > 0
        returns standardised (log card(S), log G(S)), each (B,); with return_latency also the
        standardised log remaining latency (requires latency_head=True)
        """
        state = state & pattern_mask
        remaining = pattern_mask & ~state
        nodes = self.input_projection(embeddings)                               # (B, n, H)

        # DeepSets encoding of S: order-invariant by construction.
        set_encoding = self.set_rho(
            (self.set_phi(nodes) * state.unsqueeze(-1)).sum(dim=1)
        ) + self.size_embedding(state.sum(dim=1).clamp(max=self.max_triple_patterns))

        # Contracted graph: node 0 is the super-node, nodes 1..n the patterns (only the
        # remaining ones are active).
        batch_size, n_patterns = pattern_mask.shape
        features = torch.cat([set_encoding.unsqueeze(1), nodes * remaining.unsqueeze(-1)], dim=1)
        active = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=state.device), remaining], dim=1)
        remaining_edges = adjacency & remaining.unsqueeze(1) & remaining.unsqueeze(2)
        super_edges = ((adjacency & state.unsqueeze(2)).any(dim=1) & remaining)      # (B, n)
        contracted = torch.zeros(batch_size, n_patterns + 1, n_patterns + 1, dtype=torch.bool,
                                 device=state.device)
        contracted[:, 1:, 1:] = remaining_edges
        contracted[:, 0, 1:] = super_edges
        contracted[:, 1:, 0] = super_edges
        contracted = contracted.float()

        if self.message_layer == "mean":
            degree = contracted.sum(dim=2, keepdim=True).clamp(min=1.0)
            for message_layer, layer_norm in zip(self.message_layers, self.layer_norms):
                aggregated = torch.bmm(contracted, features) / degree
                features = layer_norm(features + message_layer(torch.cat([features, aggregated], dim=-1)))
                features = features * active.unsqueeze(-1)
        else:
            contracted_edges = None
            if self.edge_feature_dim:
                if edge_features is None:
                    raise ValueError("This model uses edge features; pass edge_features.")
                edge_features = edge_features.float()
                # The super-node's edge to pattern j carries the features of every member of
                # S joined with j, summed. log1p keeps counts on a comparable scale.
                super_features = (edge_features * state.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)   # (B, n, F)
                contracted_edges = torch.zeros(batch_size, n_patterns + 1, n_patterns + 1, self.edge_feature_dim,
                                               device=state.device)
                contracted_edges[:, 1:, 1:] = edge_features
                contracted_edges[:, 0, 1:] = super_features
                contracted_edges[:, 1:, 0] = super_features
                contracted_edges = torch.log1p(contracted_edges) * contracted.unsqueeze(-1)
            for message_layer, layer_norm in zip(self.message_layers, self.layer_norms):
                features = layer_norm(features + message_layer(features, contracted, contracted_edges))
                features = features * active.unsqueeze(-1)

        remaining_count = remaining.sum(dim=1, keepdim=True).float()
        remaining_pool = features[:, 1:].sum(dim=1) / remaining_count.clamp(min=1.0)
        fraction_left = remaining_count / pattern_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

        card = self.card_head(set_encoding).squeeze(-1)
        context = torch.cat([features[:, 0], remaining_pool, set_encoding, fraction_left], dim=-1)
        cost_to_go = self.cost_to_go_head(context).squeeze(-1)
        if return_latency:
            if not self.has_latency_head:
                raise ValueError("This model has no latency head; build it with latency_head=True.")
            latency = self.latency_head(
                torch.cat([context, card.unsqueeze(-1), cost_to_go.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
            return card, cost_to_go, latency
        return card, cost_to_go
