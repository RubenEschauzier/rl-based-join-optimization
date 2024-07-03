from src.pretrain_procedure import run_training

if __name__ == "__main__":
    engine_endpoint_blazegraph_template = "http://localhost:{0:04n}/blazegraph/namespace/watdiv-default-instantiation" \
                                          "/sparql "
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/input/queries"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"
    n_endpoints = 4
    n_epoch = 50
    batch_size = 6
    seed = 0
    # endpoint_uris = [str(1000+i).format(i) for i in range(n_endpoints)]
    run_training(query_location, rdf2vec_vector_location, endpoint_location, n_epoch, batch_size, 1e-5, 4,
                 .99, seed)
