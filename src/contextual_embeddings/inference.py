"""The frozen contextualizer at estimation time: one forward pass per query.

    contextualizer = FrozenContextualizer("experiments/.../contextualizer/contextualizer.pt")
    nodes, predicates = contextualizer.embed_query(rdflib_patterns)
    # nodes:      {subject / object term (rdflib): contextual vector (dim,)}
    # predicates: {predicate term (rdflib): its (pretrained) token vector (dim,)}
"""
from __future__ import annotations

from pathlib import Path

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
                                    profile_hidden=checkpoint.get("profile_hidden", 64),
                                    profile_attention=checkpoint.get("profile_attention"))
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device
        self.index = {t: i for i, t in enumerate(checkpoint["terms"])}
        self.predicate_id = {t: i for i, t in enumerate(checkpoint.get("predicate_terms") or [])}
        self.dim = self.model.dim
        # a dictionary-encoded graph (WatDiv): constants' graph ids through its vocabulary, looked up on first use
        directory = checkpoint.get("graph_directory")
        self.graph_terms = None
        if directory and (Path(directory) / "vocabulary.npz").exists():
            from src.contextual_embeddings.vocabulary import GraphTerms
            self.graph_terms = GraphTerms(directory)
        self._entity_ids = None

    @staticmethod
    def key(term):
        """Vocabulary key: an IRI without brackets, a literal in quoted N3 form."""
        return term.n3() if isinstance(term, rdflib.term.Literal) else str(term)

    def token(self, term):
        return self.index.get(self.key(term), 0)

    def entity_id(self, term):
        """Graph entity id of a constant (profiles are indexed by it), -1 otherwise."""
        if self.graph_terms is None:
            text = str(term)
            suffix = text[len(IRI_PREFIX):] if text.startswith(IRI_PREFIX) else ""
            return int(suffix) if suffix.isdigit() else -1
        if self._entity_ids is None:                     # every entity once: ~20 s for WatDiv's 20.3M terms
            from src.query_featurizers.row_store import term_index
            self._entity_ids = term_index(self.graph_terms.entities)
        raw = term.n3() if isinstance(term, (rdflib.term.Literal, rdflib.term.URIRef)) else str(term)
        return self._entity_ids.get(raw, -1)

    @torch.no_grad()
    def embed_query(self, rdflib_patterns):
        node_of, node_token, node_is_variable = {}, [], []
        src, dst, edge_token, edge_direction, edge_feature = [], [], [], [], []
        n_predicates = len(self.predicate_id)
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
            p_id = self.predicate_id.get(str(p), -1)
            edge_feature += [p_id, n_predicates + p_id] if p_id >= 0 else [-1, -1]
        as_long = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        profile = None
        if self.profiles is not None:
            ids = [-1 if isinstance(t, rdflib.term.Variable) else self.entity_id(t) for t in node_of]
            profile = profile_rows(ids, self.profiles, self.device)
        out = self.model.encode(as_long(node_token), torch.tensor(node_is_variable, device=self.device),
                                as_long([src, dst]), as_long(edge_token),
                                torch.tensor(edge_direction, device=self.device), profile,
                                as_long(edge_feature)).cpu().numpy()
        predicates = {p: self.model.table.weight[self.token(p)].cpu().numpy() for _, p, _ in rdflib_patterns}
        return {term: out[i] for term, i in node_of.items()}, predicates
