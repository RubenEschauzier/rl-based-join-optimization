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
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.model_layers.triple_gine_conv import TripleGineConv


class Contextualizer(nn.Module):
    def __init__(self, init_vectors, hidden=64, layers=3, dropout=0.0, profile_types=None, profile_hidden=64):
        super().__init__()
        n_terms, dim = init_vectors.shape
        self.dim = dim
        self.use_profiles = profile_types is not None
        if self.use_profiles:
            n_types, n_features = profile_types.shape
            self.register_buffer("type_features", torch.as_tensor(profile_types, dtype=torch.float32), persistent=False)
            self.type_encoder = nn.Sequential(nn.Linear(n_features, profile_hidden), nn.ReLU(),
                                              nn.Linear(profile_hidden, dim))
            self.feature_gate = nn.Embedding(n_features, dim)
            nn.init.normal_(self.feature_gate.weight, mean=1.0, std=0.1)
            self.profile_norm = nn.LayerNorm(dim)
            self.token_in = nn.Linear(2 * dim, dim)
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

    def encode(self, node_token, node_is_variable, edge_index, edge_token, edge_direction, profile=None):
        """Contextual vector per node, (n_nodes, dim)."""
        static = self.table(node_token)
        if self.use_profiles:
            static = self.token_in(torch.cat([static, self.profiles(len(node_token), profile)], dim=1))
        static = torch.where(node_is_variable[:, None], self.variable.expand_as(static), static)
        h = torch.relu(self.node_in(torch.cat([static, node_is_variable[:, None].float()], dim=1)))
        edge_attr = torch.cat([self.edge_in(self.table(edge_token)), edge_direction[:, None]], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(norm(torch.relu(conv(h, edge_index, edge_attr))))
        return static + self.out(h)

    def forward(self, batch):
        nodes = self.encode(batch["node_token"], batch["node_is_variable"], batch["edge_index"],
                            batch["edge_token"], batch["edge_direction"], batch.get("profile"))
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
    src, dst, edge_token, edge_direction = [], [], [], []
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
        "pattern_nodes": (as_long(pattern_s), as_long(pattern_o)),
        "pattern_token": as_long(pattern_token),
        "replaced": torch.tensor(replaced, device=device),
        "empty": torch.tensor(empty, device=device),
        "n_instances": len(instances),
    }
