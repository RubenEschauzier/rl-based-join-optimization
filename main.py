from src.policy_gradient_rl_procedure import run_training_policy_gradient
from src.pretrain_procedure import run_pretraining


def main_policy_rl():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/input/queries"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"
    n_epoch = 50
    batch_size = 12
    seed = 0
    run_training_policy_gradient(query_location, rdf2vec_vector_location, endpoint_location, n_epoch, batch_size, 1e-6,
                                 4, .99, seed)


def main_pretraining():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/pretrain_data/queries.txt"
    cardinality_location = "data/pretrain_data/cardinalities.txt"
    prepared_queries_location = "data/pretrain_data/prepared_pretrain_queries/queries_prepared"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"

    n_epoch = 50
    batch_size = 6
    seed = 0
    lr = 1e-5

    run_pretraining(queries_location=query_location,
                    cardinalities_location=cardinality_location,
                    rdf2vec_vector_location=rdf2vec_vector_location,
                    prepared_queries_location=prepared_queries_location,
                    endpoint_uri=endpoint_location,
                    n_epoch=n_epoch,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed,
                    save_prepared_queries=True)


def main_q_learning_rl():
    pass


if __name__ == "__main__":
    main_pretraining()
