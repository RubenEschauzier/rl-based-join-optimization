from src.datastructures.query import Query


class GraphEmbedding:
    def __init__(self, models):
        self.models = models
        pass

    def run(self, queries: [Query]):
        for query in queries:
            if not query.features or not query.query_graph_representations:
                raise ValueError("Passed query without features or graph representation to graph embedder.")
            if len(query.query_graph_representations) != len(self.models):
                raise ValueError("Passed more graph representation matrices than embedding models.")

        pass

    def embed_query_graph(self, query: Query):

        pass
