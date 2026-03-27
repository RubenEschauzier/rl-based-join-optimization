import queue

import numpy as np
import torch
from typing import List
from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent
from src.supervised_value_estimation.typings.typings import EnvState


class CostEstimatorAgent(AbstractCostAgent):
    """
    A validation agent that does cost estimation (with or without epinets)
    """
    def __init__(self, model,
                 precomputed_indexes, precomputed_masks,
                 head_name = "plan_cost",
                 ):
        self.model = model

        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks

        self.head_name = head_name

    def setup_episode(self, query):
        """
        Starts a search episode. Samples a single z and embeds the query based on this z
        :param query:
        :return:
        """
        self.model.eval()

        embedded, embedded_prior = self.model.embed_query_batched(query)[0], self.model.embed_query_batched_prior(query)

        return {"embedded": embedded, "embedded_prior": embedded_prior, "query_idx": 0}

    def estimate_costs(self, possible_next, query_state):
        # Format the plans
        formatted_plans = [(p,) for p in possible_next]

        estimated_cost, _ = self.model.estimate_cost_full(
            formatted_plans, query_state["embedded"], self.precomputed_indexes, self.precomputed_masks
        )

        return estimated_cost[self.head_name], [None for _ in range(len(possible_next))]