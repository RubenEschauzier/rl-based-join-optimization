"""The frozen contextualizer at estimation time: one forward pass per query.

    contextualizer = FrozenContextualizer("experiments/.../contextualizer/contextualizer.pt")
    nodes, predicates = contextualizer.embed_query(rdflib_patterns)
    # nodes:      {subject / object term (rdflib): contextual vector (dim,)}
    # predicates: {predicate term (rdflib): its (pretrained) token vector (dim,)}
"""
from __future__ import annotations

import numpy as np
import rdflib
import torch

from src.contextual_embeddings.model import Contextualizer, profile_rows
from src.contextual_embeddings.profiles import ProfileTable
from src.random_query_generation.statistics_based_generation.graph import IRI_PREFIX


class FrozenContextualizer:
    def __init__(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        directory = checkpoint.get("profiles_directory")
        self.profiles = ProfileTable(directory) if directory else None
        self.model = Contextualizer(torch.zeros_like(state["table.weight"]), checkpoint["hidden"], checkpoint["layers"],
                                    checkpoint.get("dropout", 0.0),
                                    profile_types=None if self.profiles is None else self.profiles.type_features,
                                    profile_hidden=checkpoint.get("profile_hidden", 64))
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device
        self.index = {t: i for i, t in enumerate(checkpoint["terms"])}
        self.dim = self.model.dim

    def token(self, term):
        return self.index.get(str(term), 0)

    @staticmethod
    def entity_id(term):
        """Graph entity id of a constant IRI (profiles are indexed by it), -1 otherwise."""
        text = str(term)
        suffix = text[len(IRI_PREFIX):] if text.startswith(IRI_PREFIX) else ""
        return int(suffix) if suffix.isdigit() else -1

    @torch.no_grad()
    def embed_query(self, rdflib_patterns):
        node_of, node_token, node_is_variable = {}, [], []
        src, dst, edge_token, edge_direction = [], [], [], []
        for s, p, o in rdflib_patterns:
            for term in (s, o):
                if term not in node_of:
                    node_of[term] = len(node_token)
                    variable = isinstance(term, rdflib.term.Variable)
                    node_token.append(0 if variable else self.token(term))
                    node_is_variable.append(variable)
            u, v, t = node_of[s], node_of[o], self.token(p)
            src += [u, v]
            dst += [v, u]
            edge_token += [t, t]
            edge_direction += [1.0, -1.0]
        as_long = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        profile = None
        if self.profiles is not None:
            ids = [-1 if isinstance(t, rdflib.term.Variable) else self.entity_id(t) for t in node_of]
            profile = profile_rows(ids, self.profiles, self.device)
        out = self.model.encode(as_long(node_token), torch.tensor(node_is_variable, device=self.device),
                                as_long([src, dst]), as_long(edge_token),
                                torch.tensor(edge_direction, device=self.device), profile).cpu().numpy()
        predicates = {p: self.model.table.weight[self.token(p)].cpu().numpy() for _, p, _ in rdflib_patterns}
        return {term: out[i] for term, i in node_of.items()}, predicates
