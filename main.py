from src.policy_gradient_rl_procedure import run_training_policy_gradient
from src.pretrain_procedure import run_pretraining, main_pretraining
import os

# Root dir global for file loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main_policy_rl():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/input/queries"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"
    n_epoch = 50
    batch_size = 12
    seed = 0
    run_training_policy_gradient(query_location, rdf2vec_vector_location, endpoint_location, n_epoch, batch_size, 1e-6,
                                 4, .99, seed)


def pretraining():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    train_query_location = "data/pretrain_data/generated_queries_sub_sampler/star_queries_2_3_5.json"
    train_cardinality = None
    # test_watdiv_query_location = "data/pretrain_data/test_queries/watdiv/template_queries"
    # test_watdiv_cardinality_location = "data/pretrain_data/test_queries/watdiv/template_cardinalities"

    # rdf2vec_vector_location = "data/output/entity_embeddings/rdf2vec_vectors_depth_2_quick.json"
    rdf2vec_vector_location = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"


    ckp_directory = "data/output/cardinality_estimation/cardinality_estimation_reproduction"
    # save_prepared_queries_location = ("data/pretrain_data/prepared_pretrain_queries/"
    #                                   "sub_sampler_queries_prepared_full_torch_dict")
    # load_prepared_queries_location = ("data/pretrain_data/prepared_pretrain_queries/"
    #                                   "sub_sampler_queries_prepared_full_torch_dict")

    n_epoch = 50
    batch_size = 32
    seed = 0
    lr = 1e-4

    main_pretraining(train_queries_location=train_query_location,
                    train_cardinalities_location=train_cardinality,
                    rdf2vec_vector_location=rdf2vec_vector_location,
                    test_query_location=None,
                    test_cardinalities_location=None,
                    endpoint_uri=endpoint_location,
                    n_epoch=n_epoch,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed,
                    ckp_dir=ckp_directory,
                    save_prepared_queries_location=None,
                    load_prepared_queries_location=None
                    )


def main_q_learning_rl():
    pass


if __name__ == "__main__":
    pretraining()
