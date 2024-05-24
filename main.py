from lib.datastructures.query import Query
from lib.query_environment.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from lib.query_environment.virtuoso.query_environment_virtuoso import VirtuosoQueryEnvironment
from SPARQLWrapper import JSON

from lib.query_featurizer.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from lib.query_graph_featurizer.quad_views import FeaturizeQueryGraphQuadViews
from lib.query_rewriter.stardog_join_order_bgp import QueryJoinOrderHintStardog
from lib.query_environment.virtuoso.virtuoso_left_deep_join_order_bgp import QueryJoinOrderHintVirtuoso

if __name__ == "__main__":
    test_query = """SELECT ?v0 ?v1 ?v2 ?v4 ?v5 ?v6 WHERE {
        ?v0 <http://schema.org/contentRating> ?v1 .
        ?v0 <http://schema.org/contentSize> ?v2 .
        ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/hasGenre> <http://db.uwaterloo.ca/~galuc/wsdbm/SubGenre41> .
        ?v4 <http://db.uwaterloo.ca/~galuc/wsdbm/makesPurchase> ?v5 .
        ?v5 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseDate> ?v6 .
        ?v5 <http://db.uwaterloo.ca/~galuc/wsdbm/purchaseFor> ?v0 .
    }
    """
    test_query_2 = """
    SELECT ?v0 ?v2 ?v3 WHERE {
        ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/subscribes> <http://db.uwaterloo.ca/~galuc/wsdbm/Website34> .
        ?v2 <http://schema.org/caption> ?v3 .
        ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/likes> ?v2 .
    }   
    """
    test_query_3 = """
    SELECT ?v0 ?v2 ?v3 WHERE {
        ?v0 <http://db.uwaterloo.ca/~galuc/wsdbm/subscribes> <http://db.uwaterloo.ca/~galuc/wsdbm/Website34> .
        ?v3 <http://schema.org/caption> ?v2 .
        ?v2 <http://db.uwaterloo.ca/~galuc/wsdbm/likes> ?v0 .
    }
    """
    engine_endpoint_virtuoso = "http://localhost:8890/sparql"
    graph_uri_endpoint_virtuoso = "http://localhost:8890/watdiv-default-instantiation"
    engine_endpoint_blazegraph = " http://172.17.35.20:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"

    query = Query(test_query_2)
    query2 = Query(test_query_3)
    env = BlazeGraphQueryEnvironment(engine_endpoint_blazegraph)
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(
        'data/rdf2vec_vectors/vectors_depth_1_full_entities.json'
    )
    queries = [query]
    featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    view_creator = FeaturizeQueryGraphQuadViews()
    s_s, o_o, s_o, o_s = FeaturizeQueryGraphQuadViews.create_query_graph_view(query)

    # featurizer.run(queries)
    # result = env.run(query, [1, 2, 0], 60, JSON, {"explain": "True"})
    # env.reward(result, reward_type='intermediate results')
# join_order_query = QueryJoinOrderHintStardog(query_object)
# rewritten = join_order_query.set_join_order([(0, 1), (2, 3), (6, 7), (8, 4), (9, 5)])
# join_order_query_virtuoso = QueryJoinOrderHintVirtuoso(query_object)
# join_order_query_virtuoso.set_join_order([1,2,3,0,4,5])
# print(rewritten)
# query_object.deconstruct_to_triple_pattern()
