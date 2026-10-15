"""Value network over FOREST states, for bottom-up bushy planning (an option next to the left-deep
ContractedJoinGraphValueNet in model.py, which is unchanged).

State = a forest F: the components built so far (bushy.py), every pattern in exactly one of
them, unjoined patterns being singleton components. The optimal cost of finishing depends only
on which components exist, not on how they were built, so F is encoded order-invariantly:

    * every component C becomes one super-node, encoded with DeepSets over its patterns
      (rho(sum of phi over members) + an embedding of |C|) -- the left-deep super-node, now one
      per component; singletons go through the same encoder;
    * two components are adjacent when any of their patterns share a variable (no cartesian
      products), with the join-position edge features summed over the pattern pairs;
    * message passing (mean / gine / gat, the layers of model.py) on this component graph;
    * the forest is pooled with DeepSets again: mean and max over the components after message
      passing, plus the mean of their encodings before it.

Heads, matching the bushy DP recursion  Q(F, merge Ci, Cj) = card(Ci u Cj) + G(F'):
    card head  log card(C) of the NEWEST component C (the one the last merge built), from its
               set encoding alone -- a pure set function, so card(Ci u Cj) of a candidate merge
               is the card head on the union; it cannot depend on how C was built;
    G head     log G(F), the constrained optimal cost-to-go (bushy.forest_cost_to_go), from the
               forest pooling and the fraction of joins left, (k - 1) / (n - 1) for k components.
Optional heads as in model.py, for online training on real executions:
    latency head  remaining latency from F (latency-to-go), from the context, G and the largest
                  predicted card among the built components (all independent of the build order);
    step head     latency of the merge that built the newest component, from the forest context
                  and a symmetric encoding of the two merged parts (the left-deep first join of
                  two scans is just a merge of two singletons; no separate stand-in is needed).
Optional epinet, as in model.py, on the forest context (state heads) and context + merge (step).
Optional mixture of experts (`moe`): the low-rank residual experts of the backbone's
TripleGINEConvLowRankMoE (BatchedLowRankAdapters) on the set encoder (next to rho) and on every
message layer, routed per QUERY (routing_level 'query': the router sees the mean projected pattern
embedding, the same for every state of a query) to the top_k of num_experts, with the backbone's
load-balancing loss (load_balancing_loss). The experts start as zero updates (expert_R = 0).

Left-deep planning is the special case in which one component grows by singletons; its forest
has one super-node plus singleton patterns, like the contracted graph of model.py, but the
singletons are encoded as one-element sets.

Inputs are dense and padded: up to n components for n patterns, components indexed 0..n-1.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.model_layers.triple_gine_conv_low_rank_moe import BatchedLowRankAdapters
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet, _mlp


class ForestValueNet(ContractedJoinGraphValueNet):
    """Same constructor, statistics, standardisation and epinet helpers as the left-deep model;
    the forward pass encodes a forest instead of one joined set. The context has the left-deep
    model's width (3H + 1), so the heads and epinets are the same modules."""

    def __init__(self, *args, moe=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.has_step_head:
            del self.first_join_node                     # a first join is a merge of two singletons
        # Symmetric encoding of a merge's two parts: [a + b, a * b] -> H.
        self.merge_projection = _mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim)
        # Optional mixture of experts: {num_experts, top_k, rank, scale, temperature, balance_weight}.
        self.moe_config = dict(moe) if moe else None
        if self.moe_config:
            config, hidden = self.moe_config, self.hidden_dim
            if not 1 <= config["top_k"] <= config["num_experts"]:
                raise ValueError("moe top_k must be between 1 and num_experts")
            experts = lambda: BatchedLowRankAdapters(hidden, hidden, config["num_experts"], config["rank"])
            self.moe_router = nn.Linear(hidden, config["num_experts"])
            self.moe_set = experts()
            self.moe_message = nn.ModuleList([experts() for _ in self.message_layers])
            self.routing = None

    def _route(self, nodes, pattern_mask):
        """Per batch element (B, E) dense expert weights: softmax over the top_k router scores of
        the query's mean projected pattern embedding, 0 for the other experts. Keeps the routing
        probabilities and assignments for load_balancing_loss."""
        config = self.moe_config
        mask = pattern_mask.unsqueeze(-1).float()
        context = (nodes * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        scores = self.moe_router(context) / config.get("temperature", 1.0)
        top_scores, top = torch.topk(scores, config["top_k"], dim=-1)
        weights = torch.zeros_like(scores).scatter(1, top, F.softmax(top_scores, dim=-1))
        self.routing = (F.softmax(scores, dim=-1), torch.zeros_like(scores).scatter(1, top, 1.0 / config["top_k"]))
        return weights

    def load_balancing_loss(self):
        """As TripleGineConvLowRankMoE.load_balancing_loss, over the last forward's batch."""
        if self.routing is None:
            raise RuntimeError("A forward pass is required before computing the routing loss")
        probabilities, assignments = self.routing
        return self.moe_config["num_experts"] * torch.sum(probabilities.mean(dim=0) * assignments.mean(dim=0).detach())

    def _experts(self, adapters, x, route):
        """The routed low-rank expert update of (B, K, H) features: sum_e w_e (x L_e R_e + b_e), with
        each batch element's query weights (dense over the experts, so no per-node gathers)."""
        low = torch.einsum("bkh,ehr->bker", x, adapters.expert_L)
        update = torch.einsum("bker,be,ero->bko", low, route, adapters.expert_R) + (route @ adapters.bias).unsqueeze(1)
        return self.moe_config.get("scale", 1.0) * update

    def _set_encodings(self, nodes, membership, route=None):
        """DeepSets encodings of sets given as (B, K, n) float membership; empty sets -> 0. With the
        mixture of experts, `route` (from _route) adds the routed experts next to rho."""
        sizes = membership.sum(dim=-1)
        summed = torch.bmm(membership, self.set_phi(nodes))
        encoding = self.set_rho(summed) + self.size_embedding(sizes.long().clamp(max=self.max_triple_patterns))
        if route is not None:
            encoding = encoding + self._experts(self.moe_set, summed, route)
        return encoding * (sizes > 0).unsqueeze(-1)

    def forward(self, embeddings, pattern_mask, adjacency, component, newest, edge_features=None,
                return_latency=False):
        """Standardised (log card(newest component), log G(F)) and with return_latency the
        standardised log remaining latency; see forward_heads for the inputs."""
        heads = ("card", "cost_to_go", "latency") if return_latency else ("card", "cost_to_go")
        if return_latency and not self.has_latency_head:
            raise ValueError("This model has no latency head; build it with latency_head=True.")
        outputs = self.forward_heads(embeddings, pattern_mask, adjacency, component, newest, edge_features,
                                     heads=heads)
        return tuple(outputs[head] for head in heads)

    def forward_heads(self, embeddings, pattern_mask, adjacency, component, newest, edge_features=None,
                      merge_side=None, heads=("card", "cost_to_go")):
        """
        embeddings:    (B, n, E) frozen triple-pattern embeddings, zero-padded
        pattern_mask:  (B, n) bool, True for real patterns
        adjacency:     (B, n, n) bool join graph (patterns sharing a variable)
        component:     (B, n) long, the component index (0..n-1) of each pattern; padding ignored
        newest:        (B,) long, the component the last merge built (-1 for the initial forest:
                       its card output is meaningless and should be ignored)
        edge_features: (B, n, n, F) join-position features; required when edge_feature_dim > 0
        merge_side:    (B, n) long, for the step head: 0 / 1 for the patterns of the newest
                       component's left / right part, -1 elsewhere
        Returns the standardised {head: (B,)} plus epinet features "features_state" (B, 3H+1)
        and, with the step head, "features_step" (B, 4H+1).
        """
        batch_size, n_patterns = pattern_mask.shape
        device = pattern_mask.device
        nodes = self.input_projection(embeddings)                                   # (B, n, H)
        slots = torch.arange(n_patterns, device=device)
        membership = ((component.unsqueeze(1) == slots.view(1, -1, 1)) & pattern_mask.unsqueeze(1)).float()  # (B, K, n)
        active = membership.sum(dim=-1) > 0                                          # (B, K)
        route = self._route(nodes, pattern_mask) if self.moe_config else None
        encodings = self._set_encodings(nodes, membership, route)                    # (B, K, H)

        # Component graph: adjacent when any member patterns share a variable.
        linked = torch.einsum("bkn,bnm,blm->bkl", membership, adjacency.float(), membership)
        component_adjacency = ((linked > 0) & ~torch.eye(n_patterns, dtype=torch.bool, device=device)).float()

        features = encodings
        if self.message_layer == "mean":
            degree = component_adjacency.sum(dim=2, keepdim=True).clamp(min=1.0)
            for i, (message_layer, layer_norm) in enumerate(zip(self.message_layers, self.layer_norms)):
                aggregated = torch.bmm(component_adjacency, features) / degree
                update = message_layer(torch.cat([features, aggregated], dim=-1))
                if route is not None:
                    update = update + self._experts(self.moe_message[i], features, route)
                features = layer_norm(features + update)
                features = features * active.unsqueeze(-1)
        else:
            component_edges = None
            if self.edge_feature_dim:
                if edge_features is None:
                    raise ValueError("This model uses edge features; pass edge_features.")
                # Features of every pattern pair across two components, summed; log1p as in model.py.
                summed = torch.einsum("bkn,bnmf,blm->bklf", membership, edge_features.float(), membership)
                component_edges = torch.log1p(summed) * component_adjacency.unsqueeze(-1)
            for i, (message_layer, layer_norm) in enumerate(zip(self.message_layers, self.layer_norms)):
                update = message_layer(features, component_adjacency, component_edges)
                if route is not None:
                    update = update + self._experts(self.moe_message[i], features, route)
                features = layer_norm(features + update)
                features = features * active.unsqueeze(-1)

        # Forest pooling (DeepSets over the components).
        count = active.sum(dim=1, keepdim=True).float()
        mean_pool = features.sum(dim=1) / count.clamp(min=1.0)
        max_pool = features.masked_fill(~active.unsqueeze(-1), float("-inf")).max(dim=1).values
        max_pool = max_pool.nan_to_num(0.0, neginf=0.0)
        encoding_pool = encodings.sum(dim=1) / count.clamp(min=1.0)
        fraction_left = (count - 1) / (pattern_mask.sum(dim=1, keepdim=True).float() - 1).clamp(min=1.0)
        context = torch.cat([mean_pool, max_pool, encoding_pool, fraction_left], dim=-1)

        newest_encoding = encodings[torch.arange(batch_size, device=device), newest.clamp(min=0)]
        card = self.card_head(newest_encoding).squeeze(-1)
        cost_to_go = self.cost_to_go_head(context).squeeze(-1)
        outputs = {"card": card, "cost_to_go": cost_to_go, "features_state": context}
        if "latency" in heads:
            # A state quantity, so not the newest component's card (that depends on the build
            # order) but the largest predicted card among the built components (0 if none).
            built = membership.sum(dim=-1) > 1
            component_cards = self.card_head(encodings).squeeze(-1).masked_fill(~built, float("-inf"))
            largest = torch.where(built.any(dim=1), component_cards.max(dim=1).values, torch.zeros_like(card))
            outputs["latency"] = self.latency_head(
                torch.cat([context, largest.unsqueeze(-1), cost_to_go.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
        if "step" in heads:
            if not self.has_step_head:
                raise ValueError("This model has no step head; build it with step_latency_head=True.")
            if merge_side is None:
                raise ValueError("The step head needs merge_side.")
            sides = torch.stack([merge_side == 0, merge_side == 1], dim=1).float() * pattern_mask.unsqueeze(1)
            left, right = self._set_encodings(nodes, sides, route).unbind(dim=1)
            merge = self.merge_projection(torch.cat([left + right, left * right], dim=-1))
            outputs["features_step"] = torch.cat([context, merge], dim=-1)
            outputs["step"] = self.step_head(
                torch.cat([outputs["features_step"], card.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
        return outputs

    def card_of_sets(self, embeddings, pattern_mask, sets):
        """Standardised log card of arbitrary pattern sets, (B, S, n) bool -> (B, S): the card
        head on their set encodings, e.g. card(Ci u Cj) for every candidate merge of a forest."""
        nodes = self.input_projection(embeddings)
        membership = (sets & pattern_mask.unsqueeze(1)).float()
        route = self._route(nodes, pattern_mask) if self.moe_config else None
        return self.card_head(self._set_encodings(nodes, membership, route)).squeeze(-1)


def forest_inputs(forests: Sequence[Sequence[int]], n_patterns: int, newest: Optional[Sequence[int]] = None,
                  merges: Optional[Sequence[Optional[tuple[int, int]]]] = None, device=torch.device("cpu")):
    """Batch tensors for ForestValueNet from bitmask forests (bushy.py), padded to n_patterns:
    component (B, n), newest (B,) and, given the merges (left mask, right mask) that built each
    newest component, merge_side (B, n). `newest` defaults to the merge's union, else -1."""
    component = torch.zeros(len(forests), n_patterns, dtype=torch.long)
    newest_index = torch.full((len(forests),), -1, dtype=torch.long)
    merge_side = torch.full((len(forests), n_patterns), -1, dtype=torch.long)
    for b, forest in enumerate(forests):
        ordered = sorted(forest)
        for index, mask in enumerate(ordered):
            for p in range(n_patterns):
                if mask >> p & 1:
                    component[b, p] = index
        merge = merges[b] if merges is not None else None
        target = newest[b] if newest is not None else (merge[0] | merge[1] if merge else None)
        if target is not None and target >= 0:
            newest_index[b] = ordered.index(target)
        if merge:
            for side, mask in enumerate(merge):
                for p in range(n_patterns):
                    if mask >> p & 1:
                        merge_side[b, p] = side
    return component.to(device), newest_index.to(device), merge_side.to(device)
