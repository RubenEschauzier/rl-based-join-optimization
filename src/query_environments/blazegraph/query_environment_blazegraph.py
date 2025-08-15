import multiprocessing

from SPARQLWrapper import JSON
import warnings
from src.datastructures.query import Query
from src.query_environments.blazegraph.blazegraph_execute_query_endpoint import BlazeGraphQueryRunner
from typing import Literal
import numpy as np
import pandas as pd
import time
from multiprocessing import Queue, Pool


class BlazeGraphQueryEnvironment:
    def __init__(self, endpoint_url):
        self.query_runner = BlazeGraphQueryRunner(endpoint_url)
        pass

    def run(self, query: Query, join_order, timeout: int, result_format, additional_params: dict,
            additional_headers=None):
        rewritten_query = self.set_join_order(query, join_order)
        return self.run_raw(rewritten_query, timeout, result_format, additional_params, additional_headers)

    def run_raw(self, query: str, timeout: int, result_format, additional_params: dict, additional_headers=None):
        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = self.query_runner.run_query(query, timeout, result_format, additional_params, additional_headers)
        except TimeoutError:
            return "time-out", timeout

        execution_time = time.time() - start
        return result, execution_time

    def run_default_optimizer(self, query, timeout, result_format, additional_params):
        start = time.time()
        result = self.query_runner.run_query(query.query_string, timeout, result_format, additional_params)
        execution_time = time.time() - start

        return result, execution_time

    def cardinality_triple_pattern(self, triple_pattern: str):
        query = r"SELECT (COUNT( * ) as ?tripleCount) WHERE {{ {} }}".format(triple_pattern)
        result = self.query_runner.run_query(query, 60, JSON, {})
        count = result['results']['bindings'][0]['tripleCount']['value']
        return count

    def cardinality_term(self, term):
        query =  f"""
        SELECT ?term (COUNT(*) AS ?tripleCount)
        WHERE {{
          {{ {term} ?p ?o }} UNION   # Term in the subject position
          {{ ?s {term} ?o }} UNION   # Term in the predicate position
          {{ ?s ?p {term} }}         # Term in the object position
        }}
        GROUP BY ?term
        """
        result = self.query_runner.run_query(query, 60, JSON, {})
        if len(result['results']['bindings']) == 0:
            return 0
        count = int(result['results']['bindings'][0]['tripleCount']['value'])
        return count


    @staticmethod
    def reward(query_result, reward_type: Literal['intermediate-results', 'execution_time']):
        if reward_type == 'intermediate-results':
            # If the query timed out, return some very bad reward signal TBD
            if query_result == 'time-out':
                return 20
            try:
                explain_result = pd.read_html(query_result)
            except ValueError as e:
                print(query_result)
                print("FAIL")
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

    @staticmethod
    def process_output(query_result, reward_type: Literal['intermediate-results', 'execution_time']):
        if reward_type == 'intermediate-results':
            # If the query timed out, return some very bad reward signal TBD
            if query_result == 'time-out':
                return [], [], [], "TIMEOUT"
            try:
                explain_result = pd.read_html(query_result)
            except ValueError as e:
                if "Statistics are not available (query already terminated)" in str(query_result):
                    return [], [], [], "FAIL_FAST_QUERY_NO_STATS"
                else:
                    warnings.warn("FAIL NOT DUE TO NO AVAILABLE STATS")
                    return [], [], [], "FAIL"

            if len(explain_result) == 0:
                raise ValueError("Explain result of size 0")
            try:
                if len(explain_result) == 1:
                    reward_information = explain_result[0][['unitsOut', 'bopSummary', 'joinRatio', 'fastRangeCount']]
                else:
                    reward_information = explain_result[1][['unitsOut', 'bopSummary', 'joinRatio', 'fastRangeCount']]
            except KeyError:
                if "TimeoutException" in str(query_result):
                    return [], [], [], "TIME_OUT"
                else:
                    warnings.warn("FAIL IN PARSING EXPLAIN RESULT NOT CAUSED BY TIMEOUT")
                    return [], [], [], "FAIL"
            # The operators with join in them are likely pipeline joins and of interest
            is_join = np.array(["Join" in x for x in np.array(reward_information['bopSummary'])])
            join_ratio = np.array(reward_information['joinRatio'])[is_join]
            units_out = np.array(reward_information['unitsOut'])[is_join]
            counts = np.array(reward_information['fastRangeCount'])[is_join]

            return units_out, counts, join_ratio, "OK"
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
        # Turn off join order optimizer blazegraph
        rewritten_query_string = query.query_string.split('{')[0] + ' { \n'
        rewritten_query_string += 'hint:Query hint:optimizer "None" . \n'

        for tp_index in join_order:
            rewritten_query_string += query.string_tp[tp_index] + '\n'

        rewritten_query_string += ' } '

        return rewritten_query_string

    @staticmethod
    def set_join_order_json_query(query, join_order: list[int], triple_patterns):
        if len(join_order) != len(triple_patterns):
            raise ValueError("Join order wrong number of joins ({}) for query ({})".format(
                len(join_order), len(triple_patterns)))
        # Turn off join order optimizer blazegraph
        rewritten_query_string = query.split('{')[0] + ' { \n'
        rewritten_query_string += 'hint:Query hint:optimizer "None" . \n'

        for tp_index in join_order:
            rewritten_query_string += triple_patterns[tp_index] + '\n'

        rewritten_query_string += ' } '
        return rewritten_query_string