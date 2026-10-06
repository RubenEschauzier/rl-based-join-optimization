"""Search agents for the comparison, all behind the existing AbstractCostAgent interface.

Every agent is driven by the unchanged beam_search (search_algorithms/beam_search_left_deep)
through the unchanged multiprocess_validate_agent runner. Builders are module-level
functions so the spawn-based worker processes can construct the agents themselves.

    AmortizedDPAgent        the proposed model: ranks a candidate prefix p (joined set T) by
                            log( sum of predicted prefix cardinalities + predicted G(T) ),
                            i.e. an estimate of the TOTAL plan cost, which makes candidates
                            from different beam states comparable and greedy == Q-greedy.
    EpinetAmortizedDPAgent  the same over epinet samples: one index = a Thompson sample,
                            several = robust planning on the CVaR of the sampled scores;
                            objective C_out or latency (step latencies + latency-to-go).
    learned cardinality     the existing CardinalityEstimatorValidationAgent with a GNN
                            cardinality estimator, ranking prefixes by their estimated C_out.
    plan cost model         the trained plan-cost network's base head, as CostEstimatorAgent
                            does, minus the unused prior-ensemble embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.agents.CardinalityEstimatorAgent import CardinalityEstimatorValidationAgent
from src.supervised_value_estimation.amortized_dp.labels import (
    NEG_INF, join_position_features, logsumexp, mask_of, variable_adjacency,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.utils.epinet_utils.risk import cvar_upper_tail


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


class EpinetAmortizedDPAgent(AmortizedDPAgent):
    """AmortizedDPAgent over epinet samples, for exploration and robust planning.

    epistemic_indexes (K, D): with K = 1 the agent plans greedily under ONE sampled value
    function (Thompson sampling); with K > 1 each candidate is scored by the CVaR_alpha (mean of
    the worst (1 - alpha) share) of its K sampled scores (robust planning).
    objective "cout":    log( sum of sampled cards along the prefix + sampled G ), as the base agent;
              "latency": log1p( sum of sampled step latencies along the prefix + sampled
                         latency-to-go ), in milliseconds (needs the step and latency heads).
    `shared`: optional dict for agents that plan the same queries with different indices; the
    value network then runs once per set and only the (small) epinet runs per index.
    """

    def __init__(self, value_net: ContractedJoinGraphValueNet, embed_fn, epistemic_indexes,
                 device=torch.device("cpu"), embedding_cache=None, objective="cout", alpha_cvar=0.9, shared=None):
        super().__init__(value_net, embed_fn, device, embedding_cache)
        if not value_net.has_epinet:
            raise ValueError("EpinetAmortizedDPAgent needs a value net built with an epinet.")
        if objective not in ("cout", "latency"):
            raise ValueError(f"objective must be cout or latency, got {objective!r}")
        if objective == "latency" and not (value_net.has_latency_head and value_net.has_step_head):
            raise ValueError("The latency objective needs the latency and step heads.")
        self.epistemic_indexes = epistemic_indexes.to(device)
        self.objective = objective
        self.alpha_cvar = alpha_cvar
        self.shared = shared
        self.state_heads = ("card", "cost_to_go") + (("latency",) if value_net.has_latency_head else ())

    def setup_episode(self, query):
        data = _single_query(query)
        key = _query_string(data)
        if self.embedding_cache is not None and key in self.embedding_cache:
            embeddings = self.embedding_cache[key].to(self.device)
        else:
            embeddings = self.embed_fn(query).to(self.device)
        n_tp = embeddings.shape[0]
        empty = {"state": {}, "step": {}}
        episode = {
            "embeddings": embeddings,
            "adjacency": adjacency_matrix(data, n_tp).to(self.device),
            "n_tp": n_tp,
            "full_mask": (1 << n_tp) - 1,
            "edge_features": (torch.from_numpy(join_position_features(data.triple_patterns)).to(self.device)
                              if self.value_net.edge_feature_dim else None),
            "base": self.shared.setdefault(key, empty) if self.shared is not None else empty,
            "values": {"card": {}, "cost_to_go": {}, "latency": {}, "step": {}},
        }
        self._evaluate_states(episode, [1 << i for i in range(n_tp)])
        return episode

    def _inputs(self, episode, masks):
        k, n_tp = len(masks), episode["n_tp"]
        edges = episode["edge_features"]
        return (episode["embeddings"].unsqueeze(0).expand(k, -1, -1),
                torch.ones(k, n_tp, dtype=torch.bool, device=self.device),
                episode["adjacency"].unsqueeze(0).expand(k, -1, -1),
                state_tensor(masks, n_tp, self.device),
                None if edges is None else edges.unsqueeze(0).expand(k, -1, -1, -1))

    def _sample(self, head, base_values, features):
        samples = self.value_net.epinet_for(head).sample(base_values, features, self.epistemic_indexes, head,
                                                         self.value_net.epinet_alpha(head))
        return self.value_net.unstandardise_head(head, samples).cpu().numpy()          # (K, m)

    @torch.no_grad()
    def _evaluate_states(self, episode, masks):
        base, values = episode["base"]["state"], episode["values"]
        masks = list(dict.fromkeys(masks))
        missing = [mask for mask in masks if mask not in base]
        if missing:
            outputs = self.value_net.forward_heads(*self._inputs(episode, missing), heads=self.state_heads)
            for i, mask in enumerate(missing):
                base[mask] = {head: outputs[head][i] for head in (*self.state_heads, "features_state")}
        missing = [mask for mask in masks if mask not in values["card"]]
        if not missing:
            return
        features = torch.stack([base[mask]["features_state"] for mask in missing])
        for head in self.state_heads:
            samples = self._sample(head, torch.stack([base[mask][head] for mask in missing]), features)
            for i, mask in enumerate(missing):
                values[head][mask] = samples[:, i]
        if episode["full_mask"] in missing:     # nothing left to pay
            values["cost_to_go"][episode["full_mask"]] = np.full(len(self.epistemic_indexes), NEG_INF)
            if "latency" in self.state_heads:
                values["latency"][episode["full_mask"]] = np.zeros(len(self.epistemic_indexes))

    @torch.no_grad()
    def _evaluate_steps(self, episode, steps):
        """steps: (set, added pattern or -1 for a pair's first join)."""
        base, values = episode["base"]["step"], episode["values"]["step"]
        steps = list(dict.fromkeys(steps))
        missing = [step for step in steps if step not in base]
        if missing:
            added = torch.tensor([a for _, a in missing], dtype=torch.long, device=self.device)
            outputs = self.value_net.forward_heads(*self._inputs(episode, [m for m, _ in missing]), added=added,
                                                   heads=("card", "step"))
            for i, step in enumerate(missing):
                base[step] = {"step": outputs["step"][i], "features_step": outputs["features_step"][i]}
        missing = [step for step in steps if step not in values]
        if missing:
            samples = self._sample("step", torch.stack([base[step]["step"] for step in missing]),
                                   torch.stack([base[step]["features_step"] for step in missing]))
            for i, step in enumerate(missing):
                values[step] = samples[:, i]

    @staticmethod
    def _steps(plan):
        return [(mask_of(plan[:2]), -1)] + [(mask_of(plan[:size]), plan[size - 1]) for size in range(3, len(plan) + 1)]

    def sampled_scores(self, plan, episode):
        """(K,) sampled scores of a prefix (lower is better)."""
        values = episode["values"]
        last = mask_of(plan)
        if self.objective == "cout":
            terms = [np.minimum(values["card"][1 << plan[0]], values["card"][1 << plan[1]])]
            terms += [values["card"][mask_of(plan[:size])] for size in range(2, len(plan) + 1)]
            terms.append(values["cost_to_go"][last])
            return np.logaddexp.reduce(np.stack(terms), axis=0)
        total = sum(np.expm1(values["step"][step]) for step in self._steps(plan)) + np.expm1(values["latency"][last])
        return np.log1p(np.maximum(total, 0.0))

    def prepare(self, possible_next, query_state):
        """Evaluate (in one batch) every set and step the candidates' scores need."""
        self._evaluate_states(query_state, [mask_of(plan[:size]) for plan in possible_next
                                            for size in range(2, len(plan) + 1)])
        if self.objective == "latency":
            self._evaluate_steps(query_state, [step for plan in possible_next for step in self._steps(plan)])

    def estimate_costs(self, possible_next, query_state):
        self.prepare(possible_next, query_state)
        scores = np.stack([self.sampled_scores(plan, query_state) for plan in possible_next])     # (n, K)
        if scores.shape[1] == 1:
            costs = scores[:, 0].tolist()
        else:
            costs = cvar_upper_tail(torch.from_numpy(scores), self.alpha_cvar).tolist()
        return costs, [None for _ in possible_next]


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
