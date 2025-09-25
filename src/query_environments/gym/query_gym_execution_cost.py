import json
import math
import pickle
import random
import re
import warnings

import numpy as np
from SPARQLWrapper import JSON

from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_base import QueryGymBase


#https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
class QueryGymExecutionCost(QueryGymBase):
    def __init__(self, query_timeout, query_dataset, query_embedder, env,
                 timeout_reward = -50, fast_fail_reward = -10,
                 query_slow_down_patterns = "?z1 <http://xmlns.com/foaf/givenName> ?z0 .",
                 tp_occurrences=None,
                 slow_down_index_increment=5,
                 **kwargs):
        super().__init__(query_dataset, query_embedder, env, **kwargs)
        self._query_timeout = query_timeout
        self.timeout_reward = timeout_reward
        self.fast_fail_reward = fast_fail_reward
        self.min_reward = float("inf")
        self.max_reward = -float("inf")
        self.n_steps = 0
        # Passing occurrence information ensures a proper slowdown triple pattern gets added. We sort
        # the occurrences and then whenever a query is too fast, we add a slowdown triple in ascending order of
        # cardinality. This way we get the minimal cardinality needed to slow down the query sufficiently
        if tp_occurrences:
            self.sorted_tps = sorted(tp_occurrences.items(), key=lambda x: int(x[1]))
            self.slow_down_index = 0
            self.slow_down_index_increment = slow_down_index_increment
            self.query_slow_down_pattern = self.replace_vars(self.sorted_tps[self.slow_down_index][0])
            # for tp, card in tp_occurrences.items():
            #     if min_card_slow_down <= int(card) <= max_card_slow_down:
            #         tp_non_match_var = self.replace_vars(tp)
            #         self.query_slow_down_patterns = [tp_non_match_var]
            #         break
        else:
            self.query_slow_down_pattern = query_slow_down_patterns

        self.executions_cache = {}
        print("Determined slow_down_pattern: ${}".format(self.query_slow_down_pattern))

    def get_reward_cached(self, query, join_order, joins_made):
        if (query, join_order) not in self.executions_cache:
            final_cost, reward_per_step = self.get_reward(query, join_order, joins_made)
            if final_cost != self.timeout_reward:
                self.executions_cache[(query, join_order)] = final_cost, reward_per_step
            return final_cost, reward_per_step
        else:
            print("Cache hit...")
            return self.executions_cache[(query, join_order)]

    def get_reward(self, query, join_order, joins_made):
        join_order_trimmed = join_order[join_order != -1]
        if len(join_order_trimmed) >= len(query.triple_patterns):
            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(query.query,
                                                                             join_order_trimmed,
                                                                             query.triple_patterns)
            # Execute query to obtain selectivity
            try:
                env_result, exec_time = self.env.run_raw(rewritten, self._query_timeout, JSON,
                                                         {"explain": "True"},
                                                         {"X-BIGDATA-MAX-QUERY-MILLIS": str(self._query_timeout+1)}
                                                         )

                units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results")
            except Exception as e:
                print("Fail in self.env.run_raw")
                print(e)
                status = "FAIL"

            if status == "OK":
                final_cost_policy, reward_per_step = self.get_successful_execution_reward(
                    units_out, counts, join_order_trimmed
                )
                if final_cost_policy < self.min_reward:
                    self.min_reward = final_cost_policy
                    if self.n_steps > 500:
                        # Timeout reward is always worse than the worst non-timeout join plan seen so far
                        self.timeout_reward = self.min_reward - 1
                if final_cost_policy > self.max_reward:
                    self.max_reward = final_cost_policy
                    if self.n_steps > 500:
                        self.fast_fail_reward = self.max_reward
                return final_cost_policy, reward_per_step
            else:
                if status == "FAIL_FAST_QUERY_NO_STATS":
                    print("Starting slow down query...")
                    final_cost_policy, reward_per_step = (
                        self.execute_and_benchmark_slowdown_query(rewritten, join_order_trimmed))
                else:
                    print("Train mode: {}".format(self.train_mode))
                    print("Status: {}".format(status))
                    final_cost_policy = self.timeout_reward
                    print(self.timeout_reward)
                    reward_per_step = [self.timeout_reward/join_order_trimmed.shape[0]
                                       for _ in range(join_order_trimmed.shape[0])]
                return final_cost_policy, reward_per_step
        else:
            return 0, None

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

    def execute_and_benchmark_slowdown_query(self, rewritten, join_order_trimmed):
        # Execute query to obtain selectivity
        status, units_out, counts = self.execute_slow_down_query(rewritten)
        if status == "OK":
            return self.get_successful_execution_reward(units_out, counts, join_order_trimmed)
        elif status == "FAIL_FAST_QUERY_NO_STATS" or status == "TIME_OUT" and self.sorted_tps:
            if status == "FAIL_FAST_QUERY_NO_STATS":
                sign = 1
            else:
                # Time out hit so we decrement our triple pattern cardinality to a faster one.
                sign = -1
            while True:
                print(f"Incrementing slow_down index with sign {sign}...")
                self.slow_down_index += math.ceil(sign * self.slow_down_index_increment)
                print(f"New index: {self.slow_down_index}...")
                self.query_slow_down_pattern = self.replace_vars(self.sorted_tps[self.slow_down_index][0])
                status, units_out, counts = self.execute_slow_down_query(rewritten)
                if status == "OK":
                    return self.get_successful_execution_reward(units_out, counts, join_order_trimmed)
                elif status == "FAIL_FAST_QUERY_NO_STATS":
                    continue
                elif status == "TIME_OUT":
                    print(f"Time out in slow_down index: {self.slow_down_index}")
                    # We decrement by 1 to ensure when we overshoot we slowly get back to a triple pattern that does
                    # work. This is to prevent trashing when the window for successful query execution is small.
                    sign = -(1 / self.slow_down_index_increment)
                    continue
                else:
                    raise ValueError(f"Unexpected error executing slow down query {rewritten}, with "
                                     f"triple pattern {self.query_slow_down_pattern} and status {status}")

        elif status == "FAIL_FAST_QUERY_NO_STATS":
            final_cost_policy = self.fast_fail_reward
            reward_per_step = [self.fast_fail_reward / join_order_trimmed.shape[0]
                               for _ in range(join_order_trimmed.shape[0])]
            return final_cost_policy, reward_per_step
        else:
            raise ValueError(f"Unexpected error executing slow down query {rewritten}, with "
                             f"triple pattern {self.query_slow_down_pattern} and status {status}")

    def execute_slow_down_query(self, rewritten):
        slowed_down_query = QueryGymExecutionCost.insert_slowdown_triple(rewritten, [self.query_slow_down_pattern])
        try:
            env_result, exec_time = self.env.run_raw(slowed_down_query, self._query_timeout, JSON,
                                                     {"explain": "True"},
                                                     {"X-BIGDATA-MAX-QUERY-MILLIS": str(self._query_timeout + 1)}
                                                     )

            units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results")
        except:
            print("Fail in self.env.run_raw slowed down query")
            status = "FAIL"
            units_out = []
            counts = []

        return status, units_out, counts

    @staticmethod
    def get_successful_execution_reward(units_out, counts, join_order_trimmed):
        print("Succeeded query...")
        reward_per_step = QueryGymExecutionCost.query_plan_cost(
            units_out[:len(join_order_trimmed)] + 1,
            counts[:len(join_order_trimmed)] + 1)
        reward_per_step = np.log(reward_per_step)
        final_cost_policy = -np.sum(reward_per_step)
        return final_cost_policy, reward_per_step


    @staticmethod
    def insert_slowdown_triple(rewritten, slow_down_patterns):
        rewritten_with_insert = rewritten.rstrip()
        removed_bracket = rewritten_with_insert[:-1].rstrip()
        for i in range(len(slow_down_patterns)):
            removed_bracket += "\n" + slow_down_patterns[i]
        removed_bracket += "\n}"
        return removed_bracket

    @staticmethod
    def replace_vars(triple_pattern: str) -> str:
        counter = {"i": 0}
        def repl(match):
            i = counter["i"]
            counter["i"] += 1
            return f"?z{i}"
        return re.sub(r"\?\w+", repl, triple_pattern)


