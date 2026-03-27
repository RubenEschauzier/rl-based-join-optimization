import queue

import numpy as np
import torch
from typing import List
from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.typings.typings import EnvState


class EpinetCostEstimatorAgent(AbstractCostAgent):
    """
    A validation agent that does robust cost estimation using an epinet
    """
    def __init__(self, model,
                 precomputed_indexes, precomputed_masks,
                 n_epinet_samples,
                 alpha_mlp, alpha_ensemble,
                 alpha_cvar = 0.9,
                 head_name = "plan_cost",
                 ):
        self.model = model
        self.n_epinet_samples = n_epinet_samples

        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks

        self.alpha_mlp = alpha_mlp
        self.alpha_ensemble = alpha_ensemble
        self.alpha_cvar = alpha_cvar

        self.head_name = head_name

    def setup_episode(self, query):
        """
        Starts a search episode. We only embed query once and then run plan model on top of it.
        :param query:
        :return:
        """
        self.model.eval()
        embedded, embedded_prior = self.model.embed_query_batched(query)[0], self.model.embed_query_batched_prior(query)

        return {"embedded": embedded, "embedded_prior": embedded_prior, "query_idx": 0}

    def estimate_costs(self, possible_next, query_state):
        with torch.no_grad():
            # Format the plans
            formatted_plans = [(p,) for p in possible_next]

            # Compute (tiny) ensemble models on CPU while we wait for GPU
            prior_trees, prior_idx, prior_masks = self.model.prepare_ensemble_prior_inputs(
                formatted_plans, query_state["embedded_prior"], self.precomputed_indexes,
                self.precomputed_masks, query_idx=query_state["query_idx"]
            )

            unweighted_priors = self.model.compute_ensemble_prior_from_prepared(
                prior_trees, prior_idx, prior_masks
            )[self.head_name]

            estimated_cost, last_feature = self.model.estimate_cost_full(
                formatted_plans, query_state["embedded"], self.precomputed_indexes, self.precomputed_masks
            )
            estimated_cost = estimated_cost[self.head_name]

            # Compute priors
            epinet_indexes = self.model.sample_epistemic_indexes_batched(self.n_epinet_samples)

            ensemble_prior = torch.matmul(epinet_indexes, unweighted_priors)
            ensemble_prior_flat = ensemble_prior.view(-1, 1)

            mlp_prior = self.model.compute_mlp_prior_batched(last_feature, epinet_indexes)
            mlp_prior = mlp_prior[self.head_name]

            learnable_mlp_prior = self.model.compute_learnable_mlp_batched(last_feature, epinet_indexes)
            learnable_mlp_prior = learnable_mlp_prior[self.head_name]

            # Repeat the estimates and get the epinet-based cost estimates for each sampled index
            estimated_cost_exp = estimated_cost.repeat(self.n_epinet_samples, 1)
            epinet_estimated_cost = estimated_cost_exp + (
                    learnable_mlp_prior + self.alpha_mlp * mlp_prior + self.alpha_ensemble * ensemble_prior_flat
            )

            # Turn back into a distribution prediction shape
            estimated_distribution = epinet_estimated_cost.view((self.n_epinet_samples, -1)).T

            # Calculate the number of samples in the worst-case tail
            tail_length = max(1, int((1 - self.alpha_cvar) * self.n_epinet_samples))

            # Sort costs in ascending order
            sorted_dist, _ = torch.sort(estimated_distribution, dim=1)

            # Isolate the highest costs (worst outcomes) and compute their expectation
            worst_cases = sorted_dist[:, -tail_length:]

            return worst_cases.mean(dim=1), [None for _ in range(len(possible_next))]

