from src.datastructures.query import Query


class GraphEmbedder:
    def __init__(self, models):
        self.models = models
        pass

    def run(self, query: Query):
        if not query.features or not query.query_graph_representations:
            raise ValueError("Passed query without features or graph representation to graph embedder.")
        if len(query.query_graph_representations) != len(self.models):
            raise ValueError("Passed more graph representation matrices than embedding models.")

        embeddings = []
        for model in self.models:
            embedding = model.forward(query.features, query.query_graph_representations)
            embeddings.extend(embedding)

        return embeddings

    def embed_query_graph(self, query: Query):

        pass
