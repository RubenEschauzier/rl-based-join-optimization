"""The contextualizer: token table + a small relational GNN over a query-shaped graph.

Nodes are the subject / object terms of the patterns (constants: their token; variables: one
learned vector), edges the patterns, in both directions with a direction flag (the encoding of
the estimator's featurizer and of TripleGineConv). Output per node: its static vector plus a
contextual correction, in the same 128-dimensional space as rdf2vec, so it can replace the
rdf2vec part of the estimator's input features.

Pretraining heads are linear on purpose: what they read (which arm does not belong, is the
instance empty) must be explicit in the node outputs, since only those are kept.

With neighbour profiles (profiles.py) the input of a constant is [token vector ; profile]: every
neighbour type is encoded from its whole feature set (so "has q1 AND q2" survives pooling), and an
entity's profile is the sum over its histogram of log(1 + count) * type vector * feature gate.

With query-conditioned profiles (profile_attention) every directed pattern edge c -> v whose source
c is a constant also gets the part of c's histogram that the pattern asks about: the rows reached
through the pattern's (predicate, direction) describe the candidates of v ("who reaches Ghent via
bornIn"). "soft": attention over all of c's rows, the pattern's feature as query and each row's
feature as key (one shared feature embedding, so the exact feature is preferred from the start);
"hard": exactly the rows with that feature (ablation). Added to the edge's input, together with the
log total count of the exactly matching rows and whether there are any (none: the pattern is empty).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.model_layers.triple_gine_conv import TripleGineConv


class Contextualizer(nn.Module):
    def __init__(self, init_vectors, hidden=64, layers=3, dropout=0.0, profile_types=None, profile_hidden=64,
                 profile_attention=None, attention_dim=32):
        super().__init__()
        n_terms, dim = init_vectors.shape
        self.dim = dim
        self.use_profiles = profile_types is not None
        self.profile_attention = profile_attention
        if profile_attention not in (None, "soft", "hard"):
            raise ValueError(f"profile_attention must be None, 'soft' or 'hard', got {profile_attention!r}")
        if profile_attention and not self.use_profiles:
            raise ValueError("profile_attention needs the neighbour profiles")
        if self.use_profiles:
            n_types, n_features = profile_types.shape
            self.register_buffer("type_features", torch.as_tensor(profile_types, dtype=torch.float32), persistent=False)
            self.type_encoder = nn.Sequential(nn.Linear(n_features, profile_hidden), nn.ReLU(),
                                              nn.Linear(profile_hidden, dim))
            self.feature_gate = nn.Embedding(n_features, dim)
            nn.init.normal_(self.feature_gate.weight, mean=1.0, std=0.1)
            self.profile_norm = nn.LayerNorm(dim)
            self.token_in = nn.Linear(2 * dim, dim)
            if profile_attention:
                self.feature_key = nn.Embedding(n_features, attention_dim)
                self.attention_scale = nn.Parameter(torch.tensor(1.0))
                self.conditioned_norm = nn.LayerNorm(dim)
                self.conditioned_in = nn.Sequential(nn.Linear(dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.table = nn.Embedding(n_terms, dim, sparse=True)
        self.table.weight.data.copy_(init_vectors)
        self.register_buffer("table_init", init_vectors.clone(), persistent=False)
        self.variable = nn.Parameter(torch.zeros(dim))
        self.node_in = nn.Linear(dim + 1, hidden)
        self.edge_in = nn.Linear(dim, hidden)
        self.convs = nn.ModuleList(
            TripleGineConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=hidden)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, dim)
        self.arm_head = nn.Linear(3 * dim, 1)
        self.empty_head = nn.Linear(dim, 1)

    def profiles(self, n_nodes, profile):
        """(n_nodes, dim) profile vectors from histogram rows (node, feature, type, log1p count)."""
        types = self.type_encoder(self.type_features)
        rows = profile["weight"][:, None] * types[profile["type"]] * self.feature_gate(profile["feature"])
        pooled = torch.zeros(n_nodes, self.dim, device=rows.device).index_add_(0, profile["node"], rows)
        return self.profile_norm(pooled)

    def conditioned_profiles(self, n_nodes, edge_index, edge_feature, profile):
        """(n_edges, dim + 2): per directed edge src -> dst, the part of src's histogram reached
        through the edge's (predicate, direction) feature: an attention- or selection-pooled type
        vector, the log total count of the exactly matching rows, and whether any match (zeros for
        variable sources, unknown entities or features).

        The attention key of a row is its feature, so rows are first pooled per (node, feature)
        group (sum of log1p(count) * type vector, total count); attention runs over a node's groups
        (typically 10-30) instead of its rows (up to thousands)."""
        device = edge_index.device
        n_edges = edge_index.shape[1]
        out = torch.zeros(n_edges, self.dim + 2, device=device)
        node, feature = profile["node"], profile["feature"]
        if len(node) == 0 or n_edges == 0:
            return out
        n_features = self.feature_key.num_embeddings
        group_key, group_of_row = torch.unique(node * n_features + feature, return_inverse=True)   # sorted by node
        n_groups = len(group_key)
        values = self.type_encoder(self.type_features)[profile["type"]] * profile["weight"][:, None]
        group_value = torch.zeros(n_groups, self.dim, device=device).index_add_(0, group_of_row, values)
        group_count = torch.zeros(n_groups, device=device).index_add_(0, group_of_row, torch.expm1(profile["weight"]))
        group_node, group_feature = group_key // n_features, group_key % n_features
        counts = torch.bincount(group_node, minlength=n_nodes)
        starts = torch.cumsum(counts, 0) - counts
        src = edge_index[0]
        per_edge = counts[src] * (edge_feature >= 0)
        total = int(per_edge.sum())
        if total == 0:
            return out
        pair_edge = torch.repeat_interleave(torch.arange(n_edges, device=device), per_edge)
        offsets = torch.cumsum(per_edge, 0) - per_edge
        pair_group = (torch.repeat_interleave(starts[src], per_edge)
                      + torch.arange(total, device=device) - torch.repeat_interleave(offsets, per_edge))
        query = edge_feature.clamp(min=0)[pair_edge]
        exact = (group_feature[pair_group] == query).float()
        if self.profile_attention == "hard":
            alpha = exact
        else:
            from torch_geometric.utils import softmax
            keys = self.feature_key.weight
            scores = (keys[query] * keys[group_feature[pair_group]]).sum(-1) * self.attention_scale / keys.shape[1] ** 0.5
            alpha = softmax(scores, pair_edge, num_nodes=n_edges)
        pooled = torch.zeros(n_edges, self.dim, device=device).index_add_(0, pair_edge, alpha[:, None] * group_value[pair_group])
        matched = torch.zeros(n_edges, device=device).index_add_(0, pair_edge, exact * group_count[pair_group])
        any_match = torch.zeros(n_edges, device=device).index_add_(0, pair_edge, exact).clamp(max=1.0)
        has_groups = torch.zeros(n_edges, dtype=torch.bool, device=device)
        has_groups[pair_edge] = True
        out[:, :self.dim] = torch.where(has_groups[:, None], self.conditioned_norm(pooled), pooled)
        out[:, self.dim] = torch.log1p(matched)
        out[:, self.dim + 1] = any_match
        return out

    def encode(self, node_token, node_is_variable, edge_index, edge_token, edge_direction, profile=None,
               edge_feature=None):
        """Contextual vector per node, (n_nodes, dim)."""
        static = self.table(node_token)
        if self.use_profiles:
            static = self.token_in(torch.cat([static, self.profiles(len(node_token), profile)], dim=1))
        static = torch.where(node_is_variable[:, None], self.variable.expand_as(static), static)
        h = torch.relu(self.node_in(torch.cat([static, node_is_variable[:, None].float()], dim=1)))
        edge_hidden = self.edge_in(self.table(edge_token))
        if self.profile_attention:
            edge_hidden = edge_hidden + self.conditioned_in(
                self.conditioned_profiles(len(node_token), edge_index, edge_feature, profile))
        edge_attr = torch.cat([edge_hidden, edge_direction[:, None]], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(norm(torch.relu(conv(h, edge_index, edge_attr))))
        return static + self.out(h)

    def forward(self, batch):
        nodes = self.encode(batch["node_token"], batch["node_is_variable"], batch["edge_index"],
                            batch["edge_token"], batch["edge_direction"], batch.get("profile"), batch.get("edge_feature"))
        s, o = batch["pattern_nodes"]
        arm_logits = self.arm_head(torch.cat([nodes[s], nodes[o], self.table(batch["pattern_token"])], dim=1)).view(-1)
        pooled = torch.zeros(batch["n_instances"], self.dim, device=nodes.device).index_add_(0, batch["node_batch"], nodes)
        counts = torch.bincount(batch["node_batch"], minlength=batch["n_instances"]).clamp(min=1)[:, None]
        empty_logits = self.empty_head(pooled / counts).view(-1)
        return nodes, arm_logits, empty_logits

    def table_l2(self, tokens):
        """Mean squared distance of the looked-up token vectors to their rdf2vec start."""
        return ((self.table(tokens) - self.table_init[tokens]) ** 2).mean()


def profile_rows(entity_ids, profiles, device="cpu"):
    """Histogram rows of the constant nodes: entity_ids[i] is node i's graph entity id or -1."""
    node, feature, type_, weight = [], [], [], []
    for i, e in enumerate(entity_ids):
        if e >= 0:
            f, t, c = profiles.rows(int(e))
            node.append(np.full(len(f), i))
            feature.append(f)
            type_.append(t)
            weight.append(np.log1p(c))
    cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)
    return {"node": torch.as_tensor(cat(node, np.int64), device=device),
            "feature": torch.as_tensor(cat(feature, np.int64), device=device),
            "type": torch.as_tensor(cat(type_, np.int64), device=device),
            "weight": torch.as_tensor(cat(weight, np.float32), device=device)}


def collate(instances, entity_token, predicate_token, device="cpu", profiles=None):
    """Batch of corpus instances. entity_token: {graph entity id: token}, predicate_token: array
    graph predicate index -> token; unknown entities get token 0. profiles: a ProfileTable to add
    the constants' neighbour-profile rows."""
    node_token, node_is_variable, node_batch = [], [], []
    src, dst, edge_token, edge_direction, edge_feature = [], [], [], [], []
    n_predicates = len(predicate_token)
    pattern_s, pattern_o, pattern_token, replaced, empty = [], [], [], [], []
    offset = 0
    for b, instance in enumerate(instances):
        for e in instance["nodes"]:
            node_token.append(entity_token.get(e, 0) if e >= 0 else 0)
            node_is_variable.append(e < 0)
            node_batch.append(b)
        for (u, v, p), r in zip(instance["edges"], instance["replaced"]):
            t = int(predicate_token[p])
            src += [offset + u, offset + v]
            dst += [offset + v, offset + u]
            edge_token += [t, t]
            edge_direction += [1.0, -1.0]
            # profile feature of the edge's source: outgoing p (subject source), incoming p (object source)
            edge_feature += [p, n_predicates + p]
            pattern_s.append(offset + u)
            pattern_o.append(offset + v)
            pattern_token.append(t)
            replaced.append(float(r))
        empty.append(float(instance["empty"]))
        offset += len(instance["nodes"])
    as_long = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    batch_profile = None
    if profiles is not None:
        batch_profile = profile_rows([e for instance in instances for e in instance["nodes"]], profiles, device)
    return {
        "profile": batch_profile,
        "node_token": as_long(node_token),
        "node_is_variable": torch.tensor(node_is_variable, dtype=torch.bool, device=device),
        "node_batch": as_long(node_batch),
        "edge_index": as_long([src, dst]),
        "edge_token": as_long(edge_token),
        "edge_direction": torch.tensor(edge_direction, device=device),
        "edge_feature": as_long(edge_feature),
        "pattern_nodes": (as_long(pattern_s), as_long(pattern_o)),
        "pattern_token": as_long(pattern_token),
        "replaced": torch.tensor(replaced, device=device),
        "empty": torch.tensor(empty, device=device),
        "n_instances": len(instances),
    }
