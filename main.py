from src.datastructures.query import Query
from src.models.graph_convolution_query_embedder import GCNConvQueryEmbeddingModel
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment

from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_graph_featurizers.quad_views import FeaturizeQueryGraphQuadViews
from src.training_procedure import run_training

if __name__ == "__main__":
    engine_endpoint_blazegraph = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/input/queries"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"
    n_epoch = 15
    batch_size = 4
    seed = 0
    run_training(engine_endpoint_blazegraph, query_location, rdf2vec_vector_location, n_epoch, batch_size, 1e-4, 3,
                 .99, seed)
    # test_query = """SELECT ?v0 ?v1 ?v2 ?v4 ?v5 ?v6 WHERE {
    #     ?v0 <http://schema.org/contentRating> ?v1 .
    #     ?v0 <http://schema.org/contentSize> ?v2 .
    #     ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/hasGenre> <http://db.uwaterloo.ca/~galuc/wsdbm/SubGenre41> .
    #     ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v5 .
    #     ?v5 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseDate> ?v6 .
    #     ?v5 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v0 .
    # }
    # """
    # test_query_2 = """
    # SELECT ?v0 ?v2 ?v3 WHERE {
    #     ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/subscribes> <http://db.uwaterloo.ca/~galuc/wsdbm/Website34> .
    #     ?v2 <http://schema.org/caption> ?v3 .
    #     ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/likes> ?v2 .
    # }
    # """
    # test_query_3 = """
    # SELECT ?v0 ?v2 ?v3 WHERE {
    #     ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/subscribes> <http://db.uwaterloo.ca/~galuc/wsdbm/Website34> .
    #     ?v3 <http://schema.org/caption> ?v2 .
    #     ?v2 <http://db.uwaterloo.ca/~galuc/wsdbm/likes> ?v0 .
    # }
    # """
    # engine_endpoint_virtuoso = "http://localhost:8890/sparql"
    # graph_uri_endpoint_virtuoso = "http://localhost:8890/watdiv-default-instantiation"
    # engine_endpoint_blazegraph = " http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    #
    # query = Query(test_query_2)
    # query2 = Query(test_query_3)
    # env = BlazeGraphQueryEnvironment(engine_endpoint_blazegraph)
    # vectors = FeaturizeQueriesRdf2Vec.load_vectors(
    #     'data/rdf2vec_vectors/vectors_depth_1_full_entities.json'
    # )
    # queries = [query]
    # rdf2vec_featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    # view_creator = FeaturizeQueryGraphQuadViews()
    # queries = rdf2vec_featurizer.run(queries)
    # queries = view_creator.run(queries, "edge_index")
    #
    # model = GCNConvQueryEmbeddingModel()
    # model.run(queries[0].features, queries[0].query_graph_representations[0])
    # model_factory = ModelFactory("experiments/configs/test_config.yaml")
    #
    # n_query_graph_representations = 4
    # models_graph_embedding = []
    # for i in range(n_query_graph_representations):
    #     models_graph_embedding.append(model_factory.build_model_from_config())
    #
    # result = models_graph_embedding[0].run(queries[0].features, queries[0].query_graph_representations[0])
    # print(result.shape)
    # print(result)
    # result = env.run(query, [1, 2, 0], 60, JSON, {"explain": "True"})
    # env.reward(result, reward_type='intermediate results')
# join_order_query = QueryJoinOrderHintStardog(query_object)
# rewritten = join_order_query.set_join_order([(0, 1), (2, 3), (6, 7), (8, 4), (9, 5)])
# join_order_query_virtuoso = QueryJoinOrderHintVirtuoso(query_object)
# join_order_query_virtuoso.set_join_order([1,2,3,0,4,5])
# print(rewritten)
# query_object.deconstruct_to_triple_pattern()
