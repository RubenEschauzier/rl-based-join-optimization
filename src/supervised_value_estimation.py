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
from collections import defaultdict
from functools import partial

from torch_geometric.data import DataLoader
import numpy as np
from torch import nn

from main import find_best_epoch_directory
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining, prepare_queries
from src.utils.tree_conv_utils import prepare_trees


def prepare_data(endpoint_location,
                 queries_location_train, queries_location_val,
                 rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    print(rdf2vec_vector_location)
    train_dataset, val_dataset = prepare_queries(query_env,
                                                 queries_location_train, queries_location_val,
                                                 endpoint_location,
                                                 rdf2vec_vector_location, occurrences_location,
                                                 tp_cardinality_location)
    return train_dataset, val_dataset


def prepare_cardinality_estimator(model_config, model_directory):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    load_weights_from_pretraining(gine_conv_model, model_directory,
                                  "embedding_model.pt",
                                  ["head_cardinality.pt"],
                                  float_weights=True)
    return gine_conv_model


def get_optimal_order(query, model):
    return enumerate_orders(query, model)


def enumerate_orders(query, model):
    bound_predict_partial = partial(
        OrderDynamicProgramming.predict_cardinality,
        model,
        query
    )
    bound_predict = lambda join_order: bound_predict_partial(list(join_order), len(join_order))
    adjacency_list = build_adj_list(query)
    return JoinOrderEnumerator(adjacency_list, bound_predict, len(query.triple_patterns)).enumerate_left_deep_plans()


def prepare_simulated_dataset(train_dataset, model):
    data = {}
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for query in loader:
        plans = enumerate_orders(query[0], model)
        orders = [plan.get_order() for plan in plans]

        # Find best cost per sub plan, starting cost is infinity
        best_cost_per_sub_plan = defaultdict(lambda: float('inf'))
        for k, order in enumerate(orders):
            for i in range(1,len(order)):
                sub_order = order[:i]
                # If the parent cost is better than any before seen, we update the sub plan cost to that value
                if plans[k].cost < best_cost_per_sub_plan[tuple(sub_order)]:
                    best_cost_per_sub_plan[tuple(sub_order)] = plans[k].cost
        pass
        break
    pass


def main_simulated_training(train_dataset, val_dataset, model):
    prepare_simulated_dataset(train_dataset, model)
    pass


def main_supervised_value_estimation(endpoint_location,
                                     queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     occurrences_location, tp_cardinality_location,
                                     model_config, model_directory):
    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    cardinality_estimation_model = prepare_cardinality_estimator(model_config=model_config,
                                                                 model_directory=model_directory)
    main_simulated_training(train_dataset, val_dataset, cardinality_estimation_model)


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/yago/sparql"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"
    model_config = "experiments/model_configs/pretrain_model/t_cv_repr_exact_separate_head_own_embeddings.yaml"
    experiment_dir =  "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce-24-09-2025-11-41-58"
    model_dir = find_best_epoch_directory(experiment_dir, "val_q_error")

    main_supervised_value_estimation(endpoint_location, queries_location_train, queries_location_val,
                                     rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                     model_config, model_dir)


    # First tree:
    #               (0, 1)
    #       (1, 2)        (-3, 0)
    #   (0, 1) (-1, 0)  (2, 3) (1, 2)

    tree1 = (
        (0, 1),
        ((1, 2), ((0, 1),), ((-1, 0),)),
        ((-3, 0), ((2, 3),), ((1, 2),))
    )

    # Second tree:
    #               (16, 3)
    #       (0, 1)         (2, 9)
    #   (5, 3)  (2, 6)

    tree2 = (
        (16, 3),
        ((0, 1), ((5, 3),), ((2, 6),)),
        ((2, 9),)
    )


    trees = [tree1, tree2]

    # function to extract the left child of a node
    def left_child(x):
        assert isinstance(x, tuple)
        if len(x) == 1:
            # leaf.
            return None
        return x[1]

    # function to extract the right child of node
    def right_child(x):
        assert isinstance(x, tuple)
        if len(x) == 1:
            # leaf.
            return None
        return x[2]

    # function to transform a node into a (feature) vector,
    # should be a numpy array.
    def transformer(x):
        return np.array(x[0])


    # this call to `prepare_trees` will create the correct input for
    # a `tcnn.BinaryTreeConv` operator.
    prepared_trees = prepare_trees(trees, transformer, left_child, right_child)
    print(prepared_trees)
    test = 5
