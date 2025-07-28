import numpy as np
from SPARQLWrapper import JSON

from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_base import QueryGymBase


#https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
class QueryGymExecutionCost(QueryGymBase):
    def __init__(self, query_timeout, query_dataset, query_embedder, env, **kwargs):
        super().__init__(query_dataset, query_embedder, env, **kwargs)
        self._query_timeout = query_timeout

    def _get_reward(self):
        if self._joins_made >= self._n_triples_query:
            join_order_trimmed = self._join_order[self._join_order != -1]
            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(self._query.query,
                                                                             join_order_trimmed,
                                                                             self._query.triple_patterns)
            # Execute query to obtain selectivity
            env_result, exec_time = self.env.run_raw(rewritten, self._query_timeout, JSON, {"explain": "True"})

            units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results",
                                                                            self._query)

            if status == "OK":
                reward_per_step = QueryGymExecutionCost.query_plan_cost(units_out, counts)
                reward_per_step = np.log(reward_per_step)
                final_cost_policy = -np.sum(reward_per_step)
            else:
                # Very large negative reward when query fails.
                final_cost_policy = -70

            return final_cost_policy
        else:
            return 0

    @staticmethod
    def query_plan_cost(units_out, counts):
        """
        Get the cost of a plan based on execution. Use the first triple pattern cardinality +
        the output cardinality of each join step. Similar to cost-based reward.
        :param units_out:
        :param counts:
        :return:
        """
        # We add first count to reward query plans with small initial scans
        cost = [counts[0]]
        for i in range(units_out.shape[0] - 1):
            # Join work assuming index-based nested loop join (should include a cost for hash join)
            cost.append(units_out[i])
        return cost



