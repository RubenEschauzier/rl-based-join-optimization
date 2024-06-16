from SPARQLWrapper import JSON
import warnings
from src.datastructures.query import Query
from src.query_environments.blazegraph.blazegraph_execute_query_endpoint import BlazeGraphQueryRunner
from typing import Literal
import numpy as np
import pandas as pd
import time


class BlazeGraphQueryEnvironment:
    def __init__(self, endpoint_url):
        self.query_runner = BlazeGraphQueryRunner(endpoint_url)
        pass

    def run(self, query: Query, join_order, timeout: int, result_format, additional_params: dict):
        start = time.time()

        rewritten_query = self.set_join_order(query, join_order)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = self.query_runner.run_query(rewritten_query, timeout, result_format, additional_params)
        except TimeoutError:
            return "time-out", timeout

        execution_time = time.time() - start
        return result, execution_time

    def cardinality_triple_pattern(self, triple_pattern: str):
        query = r"SELECT (COUNT( * ) as ?triplecount) WHERE {{ {} }}".format(triple_pattern)
        result = self.query_runner.run_query(query, 60, JSON, {})
        count = result['results']['bindings'][0]['triplecount']['value']
        return count

    @staticmethod
    def reward(query_result, reward_type: Literal['intermediate-results', 'execution_time']):
        if reward_type == 'intermediate-results':
            # If the query timed out, return some very bad reward signal TBD
            if query_result == 'time-out':
                return 1000000
            try:
                explain_result = pd.read_html(query_result)
            except ValueError as e:
                return "FAIL"

            if len(explain_result) == 0:
                raise ValueError("Explain result of size 0")
            if len(explain_result) == 1:
                reward_information = explain_result[0][['unitsOut', 'bopSummary', 'joinRatio']]
            else:
                reward_information = explain_result[1][['unitsOut', 'bopSummary', 'joinRatio']]

            # The operators with join in them are likely pipeline joins and of interest
            join_ratio = np.array(reward_information['joinRatio'])
            is_join = np.array(["Join" in x for x in np.array(reward_information['bopSummary'])])

            # Blazegraph returns NaN for any join after a join ratio = 0, so we fill NaNs with zeros
            # To prevent taking log of zero we add one to join ratio
            penalty = np.log(np.nan_to_num(join_ratio[is_join]) + 1)

            return penalty
        elif reward_type == 'execution_time':
            raise NotImplementedError()

    """
    Set left-deep join order for blazegraph. Input is order in which triple patterns should be joined together.
    """

    @staticmethod
    def set_join_order(query: Query, join_order: list[int]):
        if len(join_order) != len(query.string_tp):
            raise ValueError("Join order wrong number of joins ({}) for query ({})".format(
                len(join_order), len(query.string_tp)))
        # Turn off join order optimizer virtuoso
        rewritten_query_string = query.query_string.split('{')[0] + ' { \n'
        rewritten_query_string += 'hint:Query hint:optimizer "None" . \n'

        for tp_index in join_order:
            rewritten_query_string += query.string_tp[tp_index] + '\n'

        rewritten_query_string += ' } '

        return rewritten_query_string


"""
Need to ensure environment is separate from agent. So agent merely communicates with environment through an api
What will the agent communicate?
 - Join order
 - Query (same query used within agent)

Steps within environment:
    - Receive query + join order from agent
    - Rewrite query according to environment rules
    - Execute query over endpoint
    - Process data to get reward signal
        - Allow different reward signals by using classes for query result processing (as param in run function)
"""
