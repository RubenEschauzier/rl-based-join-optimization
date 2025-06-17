import numpy as np
import torch
from src.query_environments.gym.query_gym_execution_feedback import QueryExecutionGymExecutionFeedback


class QueryGymCardinalityEstimationFeedback(QueryExecutionGymExecutionFeedback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        if action >= self.n_triples_query or self._joined[action] == 1:
            raise ValueError("Invalid action")

        self._joined[action] = 1
        # Join order is defined from 0 to max_triples + 1 for processing purposes. 0 denotes not made joins
        self.join_order[self.join_count] = action + 1
        self.join_count += 1

        print(self._query)
        print(self.join_order)

        next_obs = self._build_obs()
        reward = self._get_reward(self._query)
        done = False
        if self.join_count >= self.n_triples_query:
            done = True

        return next_obs, 0, done, False, {}

    def _get_reward(self, query):
        return 2

    def reduced_form_query(self, query, join_order):
        join_order = join_order - 1
        triple_patterns = query.triple_patterns[join_order]
        #TODO Also update the graph features etc then pass through embedder and cardinality estimation head
        #TODO Rerun featurization as it will now have node_id to term so I can determine what nodes and edges to remove
        #to rewrite query data object
