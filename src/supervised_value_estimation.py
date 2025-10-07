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

# IDEAS:
# - Train a model using a large teacher model. Furthermore, make training be a binary classification model for which
#   plan is better and a value estimation task (two separate heads).
# - In finetuning make few separate plans, execute them then determine binary classification problem again. Use the
# paper as inspiration for fine-tuning approach.
# - Plan is made by comparing for each step different orders, can be efficient as if starting left to right you keep
# switching the current plan to the better predicted. However, how would you use the estimation uncertainty?
from collections import defaultdict
from functools import partial
import random

import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
import numpy as np

from main import find_best_epoch_directory
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.models.model_layers.tcnn import build_t_cnn_tree_from_order, transformer, left_child, right_child, \
    build_t_cnn_trees
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining, prepare_queries
from src.utils.tree_conv_utils import prepare_trees


def prepare_data(endpoint_location,
                 queries_location_train, queries_location_val,
                 rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
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


def prepare_simulated_dataset(train_dataset, oracle_model, max_plans_per_query=2500):
    # Dataset contains: (Query, order, plan representation, estimated_cost)
    # Use different model for estimating cost and estimating plan (plan estimation smaller, big model can act as oracle)
    data = []
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    j = 0
    for query in tqdm(loader):
        print(query)
        # encoded_query = oracle_model.forward(x=query.x,
        #                                      edge_index=query.edge_index,
        #                                      edge_attr=query.edge_attr,
        #                                      batch=query.batch)
        # # Get the embedding head from the model
        # embedded = next(head_output['output']
        #                 for head_output in encoded_query if head_output['output_type'] == 'triple_embedding')
        # # Freeze the layers only for pretraining
        # embedded = embedded.detach()

        #TODO: In case of split model for cost and training we should:
        # Return the sub query, as that will be stored in the dataset
        # Set all cardinality estimation functionality to zero, as this will be done at the end when iterating over
        # plans
        # In case of combinatorial explosion, we subsample the plans
        plans = enumerate_orders(query[0], oracle_model)
        random.shuffle(plans)
        plans = plans[:max_plans_per_query]
        orders = [plan.get_order() for plan in plans]

        best_plan_per_sub_plan = {}

        for k, order in enumerate(orders):
            cost = plans[k].cost
            # Build tuple incrementally (no slicing)
            sub = []
            for step in order[:-1]:
                sub.append(step)
                if len(sub) < 2:
                    continue
                sub_order = tuple(sub)
                prev = best_plan_per_sub_plan.get(sub_order)
                if prev is None or cost < prev[0]:
                    best_plan_per_sub_plan[sub_order] = (cost, k)

        # Group sub plans under their best full plan
        plan_to_sub_plans = [([plan.get_order()], plan.cost) for plan in plans]
        for sub_order, (_, k) in best_plan_per_sub_plan.items():
            plan_to_sub_plans[k][0].append(sub_order)

        # for plan_sub_plan in plan_to_sub_plans:
        #     trees = build_t_cnn_trees(plan_sub_plan[0], embedded)
        #     t_cnn_input = prepare_trees(trees, transformer, left_child, right_child)
        #     data.append((t_cnn_input, torch.full([t_cnn_input[0].shape[0],1], plan_sub_plan[1])))
    return data

def train(data, model):
    pass


def main_simulated_training(train_dataset, val_dataset, model,
                            save_simulated_dataset=None):
    data = prepare_simulated_dataset(train_dataset, model)
    if save_simulated_dataset:
        # TODO Serialize this data.
        pass

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
    model_config = "experiments/model_configs/pretrain_model/t_cv_repr_huge.yaml"
    experiment_dir = "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce_large_pretrain-05-10-2025-18-13-40"
    model_dir = find_best_epoch_directory(experiment_dir, "val_q_error")

    main_supervised_value_estimation(endpoint_location, queries_location_train, queries_location_val,
                                     rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                     model_config, model_dir)
