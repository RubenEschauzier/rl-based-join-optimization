from SPARQLWrapper import JSON
from src.datastructures.query import Query
from src.query_environments.blazegraph.blazegraph_execute_query_endpoint import BlazeGraphQueryRunner
from typing import Literal
import pandas as pd


class BlazeGraphQueryEnvironment:
    def __init__(self, endpoint_url):
        self.query_runner = BlazeGraphQueryRunner(endpoint_url)
        pass

    def run(self, query: Query, join_order, timeout: int, result_format, additional_params: dict):
        rewritten_query = self.set_join_order(query, join_order)
        result = self.query_runner.run_query(rewritten_query, timeout, result_format, additional_params)
        return result

    def cardinality_triple_pattern(self, triple_pattern: str):
        query = r"SELECT (COUNT( * ) as ?triplecount) WHERE {{ {} }}".format(triple_pattern)
        result = self.query_runner.run_query(query, 60, JSON, {})
        count = result['results']['bindings'][0]['triplecount']['value']
        return count

    @staticmethod
    def reward(query_result, reward_type: Literal['intermediate results', 'execution_time']):
        if reward_type == 'intermediate results':
            explain_result = pd.read_html(query_result)
            reward_sequence = explain_result[0]['unitsOut']
            return reward_sequence
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
