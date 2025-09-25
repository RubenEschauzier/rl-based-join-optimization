# First train the model on (query, plan, estimated_cost) tuples (like we already do), use those epistemic neural nets
#   - Create an entire dataset in advance and then normalize it to be between 0 and 1
#   - Use only optimal reward for a given sub plan after augmentation
# Then create candidate plans in batches for real query latency prediction based on quantiles of the epistemic
# neural nets and quantile beam search.
#   - Select k beams per quantile with z full plans per beam
#   - Select n quantiles to search, so for n=3 highest 25 quantile performance, highest 50 quantile,
#   and highest 75 quantile and highest average value.
# Execute these queries, record latency.
#   - Cache (query, plan, latency)
#   - Track execution times found for normalization between 0 and 1
#   - Augment data for sub plans, but ensure the best plan is used for reward of that sub plan
#   - Do adaptive timeouts by tracking execution times per query
from functools import partial

from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining


def prepare_cardinality_estimator(model_config, model_directory):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    load_weights_from_pretraining(gine_conv_model, model_directory,
                                  "embedding_model.pt",
                                  ["head_cardinality.pt"],
                                  float_weights=True)
    return gine_conv_model


def get_optimal_order(query, model):
    return enumerate_optimal_order(query, model)


def enumerate_optimal_order(query, model):
    bound_predict = partial(
        OrderDynamicProgramming.predict_cardinality,
        model,
        query
    )
    adjacency_list = build_adj_list(query)
    return JoinOrderEnumerator(adjacency_list, bound_predict, len(query.triple_patterns)).search_store()
