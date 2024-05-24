from lib.datastructures.query import Query


class QueryJoinOrderHintStardog:

    def __init__(self, query: Query):
        self.query = query
        pass

    def set_join_order(self, join_order: list[tuple[int, int]]):
        if len(join_order) != len(self.query.string_tp) - 1:
            raise ValueError("Join order insufficient number of joins ({}) for query ({})".format(
                len(join_order), len(self.query.string_tp) - 1))

        n_triple_patterns: int = len(self.query.string_tp)

        sub_query_build_dict = {i: "" for i in range(n_triple_patterns*2-1)}
        for i in range(n_triple_patterns):
            sub_query_build_dict[i] = self.query.string_tp[i]

        start_index = n_triple_patterns
        for join in join_order:
            inner_tp_string = "{}\n{}".format(sub_query_build_dict[join[0]], sub_query_build_dict[join[1]])
            formatted_inner_tp = "\t{}".format('\t'.join(inner_tp_string.splitlines(True)))
            join_hint_string = "#pragma group.joins \n { \n " + formatted_inner_tp + '\n }'
            sub_query_build_dict[start_index] = join_hint_string
            start_index += 1

        return sub_query_build_dict[start_index - 1]
