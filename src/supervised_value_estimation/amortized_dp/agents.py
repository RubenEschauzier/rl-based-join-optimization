"""Search agents for the comparison, all behind the existing AbstractCostAgent interface.

Every agent is driven by the unchanged beam_search (search_algorithms/beam_search_left_deep)
through the unchanged multiprocess_validate_agent runner. Builders are module-level
functions so the spawn-based worker processes can construct the agents themselves.

    AmortizedDPAgent        the proposed model: ranks a candidate prefix p (joined set T) by
                            log( sum of predicted prefix cardinalities + predicted G(T) ),
                            i.e. an estimate of the TOTAL plan cost, which makes candidates
                            from different beam states comparable and greedy == Q-greedy.
    learned cardinality     the existing CardinalityEstimatorValidationAgent with a GNN
                            cardinality estimator, ranking prefixes by their estimated C_out.
    plan cost model         the trained plan-cost network's base head, as CostEstimatorAgent
                            does, minus the unused prior-ensemble embeddings.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.agents.CardinalityEstimatorAgent import CardinalityEstimatorValidationAgent
from src.supervised_value_estimation.amortized_dp.labels import (
    NEG_INF, join_position_features, logsumexp, mask_of, variable_adjacency,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet


def _single_query(query):
    return query.to_data_list()[0] if hasattr(query, "to_data_list") else query


def _query_string(query):
    return query.query if isinstance(query.query, str) else query.query[0]


@torch.no_grad()
def embed_triple_patterns(gnn, query, device):
    """(n_tp, E) frozen triple-pattern embeddings from the GNN's triple_embedding head.

    Same computation as QueryPlansPredictionModel.embed_query_batched for one query.
    """
    data = _single_query(query)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    output = gnn.forward(x=data.x.to(device), edge_index=data.edge_index.to(device),
                         edge_attr=data.edge_attr.to(device), batch=batch.to(device))
    embedded, _ = next(head["output"] for head in output if head["output_type"] == "triple_embedding")
    return embedded


def adjacency_matrix(query, n_patterns=None):
    data = _single_query(query)
    adjacency = variable_adjacency(data.triple_patterns)
    n_tp = len(data.triple_patterns)
    n_patterns = n_tp if n_patterns is None else n_patterns
    matrix = torch.zeros(n_patterns, n_patterns, dtype=torch.bool)
    for i, neighbours in adjacency.items():
        for j in neighbours:
            matrix[i, j] = True
    return matrix


def state_tensor(masks, n_patterns, device):
    bits = torch.arange(n_patterns, device=device)
    masks = torch.as_tensor(masks, dtype=torch.long, device=device)
    return (masks.unsqueeze(1) >> bits) & 1 == 1


class AmortizedDPAgent(AbstractCostAgent):
    def __init__(self, value_net: ContractedJoinGraphValueNet, embed_fn, device=torch.device("cpu"),
                 embedding_cache=None):
        self.value_net = value_net.to(device).eval()
        self.embed_fn = embed_fn
        self.device = device
        self.embedding_cache = embedding_cache

    def setup_episode(self, query):
        data = _single_query(query)
        key = _query_string(data)
        if self.embedding_cache is not None and key in self.embedding_cache:
            embeddings = self.embedding_cache[key].to(self.device)
        else:
            embeddings = self.embed_fn(query).to(self.device)
        n_tp = embeddings.shape[0]
        episode = {
            "embeddings": embeddings,
            "adjacency": adjacency_matrix(data, n_tp).to(self.device),
            "n_tp": n_tp,
            "full_mask": (1 << n_tp) - 1,
            "logcard": {},
            "log_g": {},
            "edge_features": (torch.from_numpy(join_position_features(data.triple_patterns)).to(self.device)
                              if self.value_net.edge_feature_dim else None),
        }
        self._evaluate(episode, [1 << i for i in range(n_tp)])
        return episode

    @torch.no_grad()
    def _evaluate(self, episode, masks):
        missing = [mask for mask in dict.fromkeys(masks) if mask not in episode["logcard"]]
        if not missing:
            return
        k, n_tp = len(missing), episode["n_tp"]
        edges = episode["edge_features"]
        card, cost_to_go = self.value_net(
            episode["embeddings"].unsqueeze(0).expand(k, -1, -1),
            torch.ones(k, n_tp, dtype=torch.bool, device=self.device),
            episode["adjacency"].unsqueeze(0).expand(k, -1, -1),
            state_tensor(missing, n_tp, self.device),
            None if edges is None else edges.unsqueeze(0).expand(k, -1, -1, -1),
        )
        card, cost_to_go = self.value_net.unstandardise(card, cost_to_go)
        for mask, c, g in zip(missing, card.tolist(), cost_to_go.tolist()):
            episode["logcard"][mask] = c
            episode["log_g"][mask] = NEG_INF if mask == episode["full_mask"] else g

    def plan_estimate(self, plan, episode):
        """log( predicted C_out of the prefix + predicted cost-to-go of its joined set )."""
        logcard = episode["logcard"]
        prefix_masks = [mask_of(plan[:size]) for size in range(2, len(plan) + 1)]
        terms = [min(logcard[1 << plan[0]], logcard[1 << plan[1]])] + [logcard[m] for m in prefix_masks]
        terms.append(episode["log_g"][prefix_masks[-1]])
        return logsumexp(terms)

    def estimate_costs(self, possible_next, query_state):
        needed = [mask_of(plan[:size]) for plan in possible_next for size in range(2, len(plan) + 1)]
        self._evaluate(query_state, needed)
        return ([self.plan_estimate(plan, query_state) for plan in possible_next],
                [None for _ in possible_next])


class PlanCostModelAgent(AbstractCostAgent):
    """CostEstimatorAgent without embedding the (unused) prior ensemble every episode."""

    def __init__(self, model, precomputed_indexes, precomputed_masks, head_name="plan_cost"):
        self.model = model
        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks
        self.head_name = head_name

    @torch.no_grad()
    def setup_episode(self, query):
        return {"embedded": self.model.embed_query_batched(query)[0]}

    @torch.no_grad()
    def estimate_costs(self, possible_next, query_state):
        estimated, _ = self.model.estimate_cost_full(
            [(plan,) for plan in possible_next], query_state["embedded"],
            self.precomputed_indexes, self.precomputed_masks,
        )
        return estimated[self.head_name].reshape(-1).tolist(), [None for _ in possible_next]


# --- builders (run inside the spawned worker processes) ----------------------------------

def load_cardinality_gnn(model_config, model_dir, device=torch.device("cpu")):
    from src.supervised_value_estimation.supervised_value_estimation_cached_prior import (
        prepare_cardinality_estimator,
    )
    gnn = prepare_cardinality_estimator(model_config=model_config, model_directory=model_dir)
    return gnn.to(device).eval()


def load_value_net(checkpoint_path, device=torch.device("cpu")):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    value_net = ContractedJoinGraphValueNet(**checkpoint["model_kwargs"])
    value_net.load_state_dict(checkpoint["state_dict"])
    return value_net.to(device).eval()


def build_amortized_dp_agent(value_net_checkpoint, embedder_config, embedder_dir):
    torch.set_num_threads(1)
    gnn = load_cardinality_gnn(embedder_config, embedder_dir)
    return AmortizedDPAgent(
        load_value_net(value_net_checkpoint),
        embed_fn=lambda query: embed_triple_patterns(gnn, query, torch.device("cpu")),
    )


def _cardinality_estimator_fn(model_config, model_dir):
    gnn = load_cardinality_gnn(model_config, model_dir)

    @torch.no_grad()
    def estimator_fn(batch):
        output = gnn.forward(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                             batch=batch.batch)
        return torch.exp(next(h["output"] for h in output if h["output_type"] == "cardinality")).reshape(-1)

    return estimator_fn


def build_learned_cardinality_agent(model_config, model_dir):
    """The existing CardinalityEstimatorValidationAgent, as validate_gnce.py builds it."""
    torch.set_num_threads(1)
    return CardinalityEstimatorValidationAgent(estimator_fn=_cardinality_estimator_fn(model_config, model_dir),
                                               estimator_requires_features=True)


def build_plan_cost_model_agent(run_config, checkpoint_path, seed):
    """The trained plan-cost network (base head), rebuilt exactly as the rerun built it."""
    from src.supervised_value_estimation.optuna_epinet_sweep import _build_epinet
    from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask

    torch.set_num_threads(1)
    cfg = OmegaConf.create(run_config)
    architecture = {
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "epinet_hidden_dim": cfg.hyperparameters.epinet_hidden_dim,
        "prior_epinet_hidden_dim": cfg.hyperparameters.prior_epinet_hidden_dim,
    }
    model, _ = _build_epinet(cfg, torch.device("cpu"), seed, architecture)
    model.load_epinet(checkpoint_path, load_only_cost_model=False, strict=True)
    model.eval()
    return PlanCostModelAgent(model, precompute_left_deep_tree_conv_index(20),
                              precompute_left_deep_tree_node_mask(20))
