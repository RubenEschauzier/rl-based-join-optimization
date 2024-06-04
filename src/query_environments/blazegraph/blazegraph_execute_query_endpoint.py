from SPARQLWrapper import SPARQLWrapper


class BlazeGraphQueryRunner:
    def __init__(self, endpoint_url):
        self.endpoint = SPARQLWrapper(endpoint_url)

    def run_query(self, query, timeout, return_format, additional_params: dict):
        self.endpoint.resetQuery()
        self.endpoint.setReturnFormat(return_format)
        self.endpoint.setTimeout(timeout)
        self.endpoint.setQuery(query)
        for (key, value) in additional_params.items():
            self.endpoint.addParameter(key, value)
        return self.endpoint.queryAndConvert()
