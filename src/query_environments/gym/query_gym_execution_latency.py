# Here reset with tree-lstm etc.
# Then add step logic and then abstract function that calculates the reward, (either 0 for execution based or cardinaltiy based etc)
# Then add step logic that calls abstract class get_infos (which is return {} for all except for one where it calculates best plan)
# For best plan wrapper all it does is override or change the get_infos to include: get best_plan, call _get_reward() on it
# This way we can have modular gyms that start from the same behavior.
import numpy as np
from SPARQLWrapper import JSON

from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_base import QueryGymBase


#https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
class QueryGymExecutionLatency(QueryGymBase):
    def __init__(self, query_timeout, query_dataset, query_embedder, env, n_train_episodes, cost_frac,
                 curriculum = True, **kwargs):
        super().__init__(query_dataset, query_embedder, env, **kwargs)
        self.switch_point = n_train_episodes * cost_frac
        self._query_timeout = query_timeout
        self._curriculum = curriculum

        self.n_episodes = 0

    def get_reward(self, query, join_order, joins_made):
        join_order_trimmed = join_order[join_order != -1]
        if len(join_order_trimmed) >= len(query.triple_patterns):
            self.n_episodes += 1

            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(query.query,
                                                                             join_order_trimmed,
                                                                             query.triple_patterns)
            # Execute query to obtain selectivity
            env_result, exec_time = self.env.run_raw(rewritten, self._query_timeout, JSON, {"explain": "True"})
            if self._curriculum:
                units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results",
                                                                                query)

                if status == "OK":
                    reward_per_step = QueryGymExecutionLatency.query_plan_cost(units_out, counts)
                    reward_per_step = np.log(reward_per_step)
                    final_cost_policy = -np.sum(reward_per_step)
                else:
                    # Very large negative reward when query fails.
                    final_cost_policy = -70
                alpha = self._get_alpha()
                reward = alpha * -exec_time + (1-alpha) * final_cost_policy
                return reward
            else:
                return -exec_time
        else:
            return 0
    def _get_alpha(self):
        return min(self.n_episodes / self.switch_point, 1)

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



