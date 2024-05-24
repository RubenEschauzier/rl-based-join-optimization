from lib.datastructures.query import Query


class QueryJoinOrderHintVirtuoso:

    def __init__(self, query: Query):
        self.query = query
        pass

    """
    Set left-deep join order for virtuoso. Input is order in which triple patterns should be joined together.
    """
    def set_join_order(self, join_order: list[int]):
        # Turn off join order optimizer virtuoso
        rewritten_query_string = 'DEFINE sql:select-option "order" \n'
        rewritten_query_string += self.query.query_string.split('{')[0] + ' { \n'

        for tp_index in join_order:
            rewritten_query_string += self.query.string_tp[tp_index] + '\n'

        rewritten_query_string += ' } '

        return rewritten_query_string
