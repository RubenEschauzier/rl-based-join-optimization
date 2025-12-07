from lib.datastructures.query import Query
from lib.query_environment.virtuoso.virtuoso_execute_query_endpoint import VirtuosoQueryRunner


class VirtuosoQueryEnvironment:
    def __init__(self, endpoint_url, default_graph, result_type, ):
        self.query_runner = VirtuosoQueryRunner(endpoint_url, default_graph, result_type)
        pass

    def run(self, query: Query, join_order, timeout: int, additional_params):
        rewritten_query = self.set_join_order(query, join_order)
        result = self.query_runner.run_query(rewritten_query, timeout)

    """
    Set left-deep join order for virtuoso. Input is order in which triple patterns should be joined together.
    """
    @staticmethod
    def set_join_order(query: Query, join_order: list[int]):
        if len(join_order) != len(query.string_tp):
            raise ValueError("Join order wrong number of joins ({}) for query ({})".format(
                len(join_order), len(query.string_tp)))
        # Turn off join order optimizer virtuoso
        rewritten_query_string = 'DEFINE sql:select-option "order" \n'
        rewritten_query_string += query.query_string.split('{')[0] + ' { \n'

        for tp_index in join_order:
            rewritten_query_string += query.string_tp[tp_index] + '\n'

        rewritten_query_string += ' } '

        return rewritten_query_string



"""
Need to ensure environment is seperate from agent. So agent merely communicates with environment through an api
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
