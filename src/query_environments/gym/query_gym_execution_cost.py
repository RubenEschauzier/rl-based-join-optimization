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
        print(f"Initialized environment with {len(query_dataset)} queries")
        self._query_timeout = query_timeout
        self.timeout_reward = timeout_reward
        self.fast_fail_reward = fast_fail_reward
        self.n_steps = 0
        # Passing occurrence information ensures a proper slowdown triple pattern gets added. We sort
        # the occurrences and then whenever a query is too fast, we add a slowdown triple in ascending order of
        # cardinality. This way we get the minimal cardinality needed to slow down the query sufficiently
        if tp_occurrences:
            self.sorted_tps = sorted(tp_occurrences.items(), key=lambda x: int(x[1]))
            self.slow_down_index = 0
            self.slow_down_index_increment = slow_down_index_increment
            self.query_slow_down_pattern = self.replace_vars(self.sorted_tps[self.slow_down_index][0])
        else:
            self.query_slow_down_pattern = query_slow_down_patterns

        self.executions_cache = {}
        # Caches to iteratively tighten the timeout for queries
        self.min_execution_time_cache = {}
        self.min_reward_cache = {}
        print("Start slow_down_pattern: ${}".format(self.query_slow_down_pattern))

    def get_reward_cached(self, query, join_order, joins_made):
        join_order_tuple = tuple(join_order)
        if (query.query, join_order_tuple) not in self.executions_cache:
            final_cost, reward_per_step = self.get_reward(query, join_order, joins_made)
            # Only cache full plans, as partial plans require no query executions
            if final_cost != self.timeout_reward and final_cost > 0:
                self.executions_cache[(query.query, join_order_tuple)] = final_cost, reward_per_step
            return final_cost, reward_per_step
        else:
            print("Cache hit full plan...")
            return self.executions_cache[(query.query, join_order_tuple)]

    def get_reward(self, query, join_order, joins_made):
        join_order_trimmed = join_order[join_order != -1]
        if len(join_order_trimmed) >= len(query.triple_patterns):
            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(query.query,
                                                                             join_order_trimmed,
                                                                             query.triple_patterns)
            # Execute query to obtain selectivity
            try:
                time_out = self._determine_timeout(query)
                # TODO: Use timeout from previous executions if applicable
                env_result, exec_time = self.env.run_raw(rewritten, time_out, JSON,
                                                         {"explain": "True"},
                                                         {"X-BIGDATA-MAX-QUERY-MILLIS": str(self._query_timeout+1)})

                units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results")
            except Exception as e:
                print("Fail in self.env.run_raw")
                print(e)
                status = "FAIL"
            return self.process_query_execution(query, rewritten, time_out, exec_time, status, units_out, counts,
                                                join_order_trimmed, add_to_cache=True)
        else:
            return 0, None


    def execute_and_benchmark_slowdown_query(self, rewritten, join_order_trimmed):
        """
        When blazegraph gets a really fast query, it often fails to produce statistics. Thus, to prevent this
        we slow down the query with a triple pattern and then execute it. We iteratively increase the cardinality
        of the added triple pattern to prevent further failures. We decrease the cardinality when we reach time out.
        These queries are not subject to iterative bounding of the time out, as these queries are fast anyways and
        unpredictable.
        :param rewritten:
        :param join_order_trimmed:
        :return:
        """

        # Execute query to obtain selectivity
        status, units_out, counts, exec_time = self.execute_slow_down_query(rewritten)
        if status == "OK":
            final_cost_policy, cost_per_step = self.get_successful_execution_reward(
                units_out, counts, join_order_trimmed)
            return final_cost_policy, cost_per_step, exec_time
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
                status, units_out, counts, exec_time = self.execute_slow_down_query(rewritten)
                if status == "OK":
                    final_cost_policy, cost_per_step = self.get_successful_execution_reward(
                        units_out, counts, join_order_trimmed)
                    return final_cost_policy, cost_per_step, exec_time
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

        return status, units_out, counts, exec_time

    def execute_benchmark_plan_query(self, query, time_out, join_order_trimmed):
        """
        When a query times out, we execute a benchmark query that runs with default optimizer.
        Using this result, we set the time-out and maximal reward of that query, thus in future executions we have
        a tight time out bound and a notion of what is a decent query plan.
        :return:
        """
        env_result, exec_time = self.env.run_raw(query.query, time_out, JSON,
                                                 {"explain": "True"},
                                                 {"X-BIGDATA-MAX-QUERY-MILLIS": str(self._query_timeout + 1)}
                                                 )
        units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results")
        if status == "OK":
            final_cost_policy, reward_per_step = self.get_successful_execution_reward(
                units_out, counts, join_order_trimmed
            )
            # Update worst reward so far for query
            self._update_reward_cache(query, final_cost_policy)
            self._update_min_execution_time(query, exec_time)

        elif status == "FAIL_FAST_QUERY_NO_STATS":
            print("Starting slow down query...")
            final_cost_policy, reward_per_step, exec_time = (
                self.execute_and_benchmark_slowdown_query(query.query, join_order_trimmed))

            # Update worst reward so far for query
            self._update_reward_cache(query, final_cost_policy)
            self._update_min_execution_time(query, exec_time)
        elif status == "TIME_OUT":
            print("Default optimizer query timed out could not determine a reward or a tightened bound")

    def process_query_execution(self, query, rewritten, time_out,
                                exec_time, status, units_out, counts, join_order_trimmed,
                                add_to_cache):
        if status == "OK":
            final_cost_policy, reward_per_step = self.get_successful_execution_reward(
                units_out, counts, join_order_trimmed
            )
            # Update worst reward so far for query
            if add_to_cache:
                self._update_reward_cache(query, final_cost_policy)
                self._update_min_execution_time(query, exec_time)

            return final_cost_policy, reward_per_step
        elif status == "FAIL_FAST_QUERY_NO_STATS":
            print("Starting slow down query...")
            final_cost_policy, reward_per_step, exec_time = (
                self.execute_and_benchmark_slowdown_query(rewritten, join_order_trimmed))

            # Update worst reward so far for query
            if query.query not in self.min_reward_cache or self.min_reward_cache[query.query] > final_cost_policy:
                self.min_reward_cache[query.query] = final_cost_policy
            return final_cost_policy, reward_per_step

        elif status == "TIME_OUT":
            if query.query not in self.min_reward_cache:
                # Determine what reward to give and time out for next execution by using default optimizer
                print("Determining reward using default optimizer")
                self.execute_benchmark_plan_query(query, time_out, join_order_trimmed)
            time_out_reward = self._determine_timeout_reward(query)
            final_cost_policy, reward_per_step = (time_out_reward,
                                                  [time_out_reward / join_order_trimmed.shape[0]
                                                   for _ in range(join_order_trimmed.shape[0])])
            print(f"Timed out with time out {time_out}, reward: {time_out_reward}")
            return final_cost_policy, reward_per_step
        else:
            print("Train mode: {}".format(self.train_mode))
            print("Status: {}".format(status))
            final_cost_policy = self.timeout_reward
            reward_per_step = [self.timeout_reward / join_order_trimmed.shape[0]
                               for _ in range(join_order_trimmed.shape[0])]
            return final_cost_policy, reward_per_step


    def _update_reward_cache(self, query, cost):
        if query.query not in self.min_reward_cache or self.min_reward_cache[query.query] > cost:
            self.min_reward_cache[query.query] = cost


    def _update_min_execution_time(self, query, exec_time):
        if query.query not in self.min_execution_time_cache or self.min_execution_time_cache[query.query] > exec_time:
            self.min_execution_time_cache[query.query] = exec_time

    def _determine_timeout(self, query):
        if query.query in self.min_execution_time_cache:
            return min(self.min_execution_time_cache[query.query] * 2, self._query_timeout)
        else:
            return self._query_timeout

    def _determine_timeout_reward(self, query):
        if query.query in self.min_reward_cache:
            return self.min_reward_cache[query.query] - 5
        else:
            return self.timeout_reward

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
