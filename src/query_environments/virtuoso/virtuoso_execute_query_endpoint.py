from SPARQLWrapper import SPARQLWrapper, JSON


class VirtuosoQueryRunner:
    def __init__(self, endpoint_url, default_graph, return_format):
        self.endpoint = SPARQLWrapper(endpoint_url)
        self.endpoint.setReturnFormat(return_format)
        self.endpoint.addDefaultGraph(default_graph)

    def run_query(self, query, timeout):
        self.endpoint.setTimeout(timeout)
        self.endpoint.setQuery(query)
        return self.endpoint.queryAndConvert()


# def wrapper(url, default_graph):
#     sparql = SPARQLWrapper(
#         url
#     )
#     sparql.setReturnFormat(JSON)
#     sparql.addDefaultGraph(default_graph)
#     return sparql
#
#
# def execute_query(query, wrapped_sparql_endpoint):
#     wrapped_sparql_endpoint.setTimeout(60)
#     wrapped_sparql_endpoint.setQuery(query)
#     return wrapped_sparql_endpoint.queryAndConvert()
#
#
# def count_results(result):
#     count = 0
#     bindings_found = []
#     for r in result["results"]["bindings"]:
#         count += 1
#     return count
