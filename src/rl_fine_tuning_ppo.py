from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym import QueryExecutionGym
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset

if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_full_128"
    rdf2vec_vector_location = "data/input/rdf2vec_embeddings/rdf2vec_vectors_depth_2_quick.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/partial/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/partial/tp_cardinalities.json"
    model_config = r"experiments/model_configs/policy_networks/t_cv_repr_large.yaml"

    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.2, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location)
    model_factory_gine_conv= ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()

    gym_env = QueryExecutionGym(train_dataset, 128, gine_conv_model, query_env)
    gym_env.reset()