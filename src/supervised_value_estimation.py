# First train the model on (query, plan, estimated_cost) tuples (like we already do), use those epistemic neural nets
#   - Create an entire dataset in advance and then normalize it to be between 0 and 1
#   - Use only optimal reward for a given sub plan after augmentation
# Then create candidate plans in batches for real query latency prediction based on quantiles of the epistemic
# neural nets and quantile beam search.
#   - Select k beams per quantile with z full plans per beam
#   - Select n quantiles to search, so for n=3 highest 25 quantile performance, highest 50 quantile,
#   and highest 75 quantile and highest average value.
# We will use an adapted version of safe exploration from balsa:
#   - Prefer plans with highest 75 quantile performance
#   - Then investigate plans with 50 quantile
#   - Etc
# Execute these queries, record latency.
#   - Cache (query, plan, latency)
#   - Track execution times found for normalization between 0 and 1
#   - Augment data for sub plans, but ensure the best plan is used for reward of that sub plan
#   - Do adaptive timeouts by tracking execution times per query
import json
import math
import os
import time
# IDEAS:
# Two losses and estimation heads: Latency and Cost
# MoE in graph model
# Uncertainty aware MoE
from collections import defaultdict
from functools import partial

from torch_geometric.nn import GlobalAttention
from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader
from itertools import chain

from main import find_best_epoch_directory
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.models.model_layers.tcnn import build_t_cnn_tree_from_order, transformer, left_child, right_child, \
    build_t_cnn_trees, BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining, prepare_queries
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.tree_conv_utils import prepare_trees, build_batched_flat_trees_and_indexes, \
    build_batched_flat_trees_and_indexes_optimized, build_batch_trees_indexes_own, build_trees_and_indexes, \
    precompute_left_deep_tree_conv_index

import torch
import torch.nn as nn

class QueryPlansPredictionModel(nn.Module):
    def __init__(self, query_emb_model, query_plan_model, emb_dim, device, ):
        super().__init__()
        self.query_emb_model = query_emb_model
        self.query_plan_model = query_plan_model
        self.emb_dim = emb_dim
        self.device = device
        pass

    def embed_query_batched(self, queries):
        """
        First embed the query features using a query embedding model. This encodes the entire query and its shape
        :param queries: The DataBatch object (pytorch geometric) containing the queries
        :return: A list of embedding tensors with equal length to batch size
        """
        embedded = self.query_emb_model.forward(x=queries.x.to(self.device),
                                       edge_index=queries.edge_index.to(self.device),
                                       edge_attr=queries.edge_attr.to(self.device),
                                       batch=queries.batch.to(self.device))
        # Get the embedding head from the model
        embedded_combined, edge_batch = next(head_output['output']
                                             for head_output in embedded if
                                             head_output['output_type'] == 'triple_embedding')
        n_nodes_in_batch = nn.functional.one_hot(edge_batch).sum(dim=0)
        selection_index = list(torch.cumsum(n_nodes_in_batch, dim=0))[:-1]

        # List of tensors for each query with embedded triple patterns
        embedded = torch.vsplit(embedded_combined, selection_index)
        return embedded

    def prepare_plans(self, plans, embedded):
        pass

    def estimate_cost(self):
        pass





class PyGCardinalityHead(nn.Module):
    def __init__(self, device, feature_dim=256):
        super().__init__()
        self.device = device
        # 1. Tree/Graph Convolution
        # Assuming you have a GNN/TreeConv backbone defined elsewhere
        # (If using standard GCN/GAT, define them here)
        self.conv1 = BinaryTreeConv(200, 512)
        self.conv2 = BinaryTreeConv(512, feature_dim)

        # 2. Global Attention Pooling
        # gate_nn: Compares node features to scalar scores
        gate_nn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        self.plan_embedding_nn = nn.Sequential(
            BinaryTreeConv(200, 512),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(512, 512),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(512, feature_dim),
        )
        self.attn_pool = GlobalAttention(gate_nn=gate_nn, nn=None)

        # 3. Final Regressor
        # Input size is double feature_dim because we concat [Root, Attention_Pool]
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, trees, indexes):
        # 1. Encode Tree
        emb, idx = self.plan_embedding_nn((trees, indexes))

        # We reshape the (n_plans, dim, n_nodes) tensor to (n_plans, n_nodes, dim)
        emb_transposed = emb.transpose(1, 2)

        # Stack the node embeddings to a tensor (n_plans * n_nodes, dim) to use with batch variable
        emb_stacked = emb_transposed.reshape((-1, emb_transposed.shape[-1]))

        # Create batch vector to represent the fact that we merged plans
        n_plans = emb_transposed.shape[0]
        n_nodes = emb_transposed.shape[1]
        plan_indices = torch.arange(n_plans, device=self.device)

        # Repeat each plan index n_nodes times
        batch = plan_indices.repeat_interleave(n_nodes)

        # 3. Attention Pooling (Signal Extraction)
        # Automatically handles the batch-wise softmax!
        pool_vectors = self.attn_pool(emb_stacked, batch)

        # Get root representation
        root_vectors = emb_transposed[:, 1, :]

        # 4. Concatenate
        combined = torch.cat([root_vectors, pool_vectors], dim=1)

        # 5. Predict
        return self.regressor(combined), combined

class EpistemicNetwork(nn.Module):
    def __init__(self,
                 learnable_mlp_config,
                 epi_index_dim, prior_config,
                 device=torch.device('cpu')):
        # We use an ensemble of tiny gine_conv as priors
        model_factory_gine_conv = ModelFactory(prior_config)
        ensemble_gnn = [model_factory_gine_conv.load_gine_conv() for i in range(epi_index_dim)]

        model_factory_mlp = ModelFactory(learnable_mlp_config)

        # The learnable and fixed MLPs should have same config all three parts should have same output dimension
        self.learnable_epinet = model_factory_mlp.load_gine_conv()
        self.prior_epinet = model_factory_mlp.load_gine_conv()

        #TODO: Check if this works
        for param in self.prior_epinet.parameters():
            param.requires_grad = False
        pass

    def forward(self):
        #TODO:
        # This should be passed a batch of queries, then it should first make a prediction with the full model:
        # gnn + tcnn
        # Then it should sample a gaussian z for each predicted cost, with certain dimension
        # Then compute for the epinet part:
        # The MLP representation given z and last layer feature of the full model (combined variable).
        # This last layer feature should have its gradient removed (sg operator)
        # Then for prior:
        # Again the MLP but completely no gradients
        # Then also the full model but with no gradients, its fixed randomness
        # There should be two modes: precomputed trees with indexes and trees and on the fly.
        # We can do the calc of indexes again with cache when we train on actual query execution though if it turns out
        # very slow

        #TODO: For beam search we should investigate thompson sampling using epinet and just whatever that other paper
        # proposed.
        pass



def min_max_scale_plans(query_plans):
    min_cost = math.inf
    max_cost = -math.inf
    for query, plans in query_plans.items():
        for (order, cost) in plans:
            if min_cost > cost:
                min_cost = cost
            if max_cost < cost:
                max_cost = cost

    range_cost = max_cost - min_cost
    for query, plans in query_plans.items():
        for i in range(len(plans)):
            order, cost = plans[i]
            scaled_cost = (cost - min_cost) / range_cost
            plans[i] = (order, scaled_cost)

    return query_plans, min_cost, max_cost


def flatten_plans(plans):
    flattened_plans = {}
    for (query, query_plans) in plans.items():
        plan_cost_tuples = []

        for sub_plans in query_plans:
            cost = sub_plans[1]
            for sub_plan in sub_plans[0]:
                plan_cost_tuples.append((sub_plan, cost))

        flattened_plans[query] = plan_cost_tuples
    return flattened_plans

def write_plans_to_file(data_point, loc):
    data_file = loc + ".jsonl"

    with open(data_file, 'a', encoding="utf-8") as df:
        df.write(json.dumps(data_point) + "\n")

def prepare_data(endpoint_location,
                 queries_location_train, queries_location_val,
                 rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location_train, queries_location_val,
                                                           endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           shuffle_train=True, load_mappings=False
                                                           )
    return train_dataset, val_dataset


def prepare_cardinality_estimator(model_config, model_directory=None):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    if model_directory:
        load_weights_from_pretraining(gine_conv_model, model_directory,
                                      "embedding_model.pt",
                                      ["head_cardinality.pt"],
                                      float_weights=True)
    return gine_conv_model


def sample_orders(query, model, max_samples_per_relation, device):
    bound_predict_partial = partial(
        OrderDynamicProgramming.predict_cardinality,
        model,
        query,
        device = device
    )
    bound_predict = lambda join_order: bound_predict_partial(list(join_order), len(join_order))
    adjacency_list = build_adj_list(query)
    return (JoinOrderEnumerator(adjacency_list, bound_predict, len(query.triple_patterns))
            .sample_left_deep_plans(max_samples_per_relation))


def prepare_simulated_dataset(dataset_to_prepare, oracle_model, device, output_loc, max_plans_per_relation=50):
    # Dataset contains: [(Query, [(plans, cost for plans), ...]
    # Training is done over batches of Queries, to amortize the query embedding step. For each sub plan we create
    # tree-based embedding
    # Use different model for estimating cost and estimating plan (plan estimation smaller, big model can act as oracle)
    data = []
    k =  0
    if os.path.exists(output_loc + '.jsonl'):
        print("Using pre-made plans")
        with open(output_loc + '.jsonl', 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                data.append(json.loads(line))
                k += 1
        return data

    loader = DataLoader(dataset_to_prepare, batch_size=1, shuffle=False)
    for query in tqdm(loader):
        # Hack workaround due to the batch attribute disappearing when unbatching the data. Although it is
        # technically not needed, the GNN expects it, so we reattach it
        batch_query = query.batch

        un_batched_query = query[0]
        un_batched_query.batch = batch_query
        un_batched_query.to(device)

        # To prevent combinatorial explosion, we subsample the plans
        plans = sample_orders(un_batched_query, oracle_model, max_plans_per_relation, device)
        # random.shuffle(plans)
        # plans = plans[:max_plans_per_relation]
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

        write_plans_to_file({query[0].query: plan_to_sub_plans}, output_loc)
    return data

def prepare_plan_data_structures(queries, query_plans, precomputed_indexes):
    queries_loader = DataLoader(queries, batch_size=1, shuffle=False)
    #TODO Use dummy embeddings with their respective index in the plan, so a 0 in the order
    # becomes a zero vector in the plan tree. 1 becomes 1 vector etc.
    # Then use the first element of each vector in plan tree to access the actual embedding
    # Figure out how to handle padding, because that is a zero vector too. Prob use something like the number of
    # elements in a order
    # Also allow multiple sizes of embeddings, to facilitate ENNs as the small GNNs will output a MUCH smaller emb
    # dimension
    # Query plans: {query: list[tuple(order, cost)]
    # Should become query: list[tuple(order, cost, tree_rep, index)]
    # When iterating over this just use `for order, cost, tree_rep, index in list[tuple(...)]`
    query_plans_augmented = {}
    for query in queries_loader:
        plans_augmented = []
        plans = query_plans[query.query[0]]

        node_ids = torch.arange(query.x.shape[0], device=query.x.device).unsqueeze(1)
        embedded = node_ids.repeat(1, query.x.shape[1])

        prepared_trees, prepared_indexes = build_trees_and_indexes(
            [[plan[0] for plan in plans]], [embedded],
            precomputed_indexes
        )

        for i, plan in enumerate(plans):
            # Augment the original plans with their tree structures. Note that the trees are
            # shape: (n_plans, n_dim, max_nodes_plan), so will need some post-processing to fill in with embedding
            plans_augmented.append((plan[0], plan[1], prepared_trees[i], prepared_indexes[i]))
        query_plans_augmented[query.query[0]] = plans_augmented
    return query_plans_augmented


def validate(queries_val, query_plans_val,
             precomputed_indexes, min_cost, max_cost,
             train_loss,
             model_query, model_plan, device):
    query_to_val = {}
    mape = MeanAbsolutePercentageError()
    mape.to(device)
    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)
    for queries in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            embedded = model_query.forward(x=queries.x.to(device),
                                           edge_index=queries.edge_index.to(device),
                                           edge_attr=queries.edge_attr.to(device),
                                           batch=queries.batch.to(device))
            # Get the embedding head from the model
            embedded_combined, edge_batch = next(head_output['output']
                                                 for head_output in embedded if
                                                 head_output['output_type'] == 'triple_embedding')
            n_nodes_in_batch = nn.functional.one_hot(edge_batch).sum(dim=0)
            selection_index = list(torch.cumsum(n_nodes_in_batch, dim=0))[:-1]

            # List of tensors for each query with embedded triple patterns
            embedded = torch.vsplit(embedded_combined, selection_index)

            plans = query_plans_val[queries.query[0]]
            prepared_trees, prepared_indexes = build_trees_and_indexes(
                [[plan[0] for plan in plans]], [embedded[0]],
                precomputed_indexes
            )
            prepared_trees.to(device)
            [prepared_index.to(device) for prepared_index in prepared_indexes]

            target = torch.tensor([plan[1] for plan in plans], device=device)
            output, _ = model_plan(prepared_trees, prepared_indexes[0]).squeeze()
            original_cost = output * (max_cost - min_cost) + min_cost

            mape_val = mape(original_cost, target)
            query_loss = train_loss(original_cost, target)

            query_to_val[queries.query[0]] = {"loss": query_loss.cpu().item(), "mape": mape_val.cpu().item()}
    return query_to_val


def train_simulated(queries_train, query_plans_train,
                    queries_val, query_plans_val,
                    model_query, model_plan, device, query_batch_size, train_batch_size):
    model_plan.to(device)
    model_query.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(100, device)
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)

    query_plans_train = flatten_plans(query_plans_train)
    query_plans_train, min_val, max_val = min_max_scale_plans(query_plans_train)
    # query_plans_train_augmented = prepare_plan_data_structures(queries_train, query_plans_train, precomputed_indexes)

    query_plans_val = flatten_plans(query_plans_val)

    # --- 1. Define Hyperparameters ---
    lr = 1e-4
    n_epochs = 10

    # --- 2. Instantiate Model and Optimizer (AdamW) ---
    params = chain(model_query.parameters(), model_plan.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=0.01  # Standard L2 regularization for AdamW
    )

    loss = torch.nn.MSELoss(reduction='mean')
    #TODO: IDEA: PREMAKE THE PLAN STRUCTURES AND THEN HAVE A MAPPING THAT MAPS INDEX OF GNN TO INDEX IN THE TREE.
    # THIS WILL PREVENT DOUBLE WORK ESPECIALLY IN EPISTEMIC NN
    for epoch in range(n_epochs):
        query_loss_epoch = []
        for k, queries in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()
            embedded = model_query.forward(x=queries.x.to(device),
                                           edge_index=queries.edge_index.to(device),
                                           edge_attr=queries.edge_attr.to(device),
                                           batch=queries.batch.to(device))
            # Get the embedding head from the model
            embedded_combined, edge_batch = next(head_output['output']
                            for head_output in embedded if head_output['output_type'] == 'triple_embedding')
            n_nodes_in_batch = nn.functional.one_hot(edge_batch).sum(dim=0)
            selection_index = list(torch.cumsum(n_nodes_in_batch, dim=0))[:-1]

            # List of tensors for each query with embedded triple patterns
            embedded = torch.vsplit(embedded_combined, selection_index)

            total_loss_tensor = torch.tensor(0.0, device=device)
            for i in range(len(queries.query)):

                plans = query_plans_train[queries.query[i]]
                prepared_trees, prepared_indexes = build_trees_and_indexes(
                    [[plan[0] for plan in plans]], [embedded[i]],
                    precomputed_indexes
                )
                prepared_trees.to(device)
                [prepared_index.to(device) for prepared_index in prepared_indexes]

                target = torch.tensor([plan[1] for plan in plans], device=device)
                # output = model_plan((prepared_trees, prepared_indexes[0]))
                output, _ = model_plan.forward(prepared_trees, prepared_indexes[0])

                query_loss = loss(output.squeeze(), target)
                query_loss_epoch.append(query_loss.detach().cpu().item())
                total_loss_tensor += query_loss

            total_loss_tensor.backward()

            optimizer.step()

        query_to_val = validate(queries_val, query_plans_val,
                                 precomputed_indexes,
                                 min_val, max_val,
                                 loss,
                                 model_query, model_plan,
                                 device)
        val_losses = [val_output["loss"] for val_output in query_to_val.values()]
        val_mape = [val_output["mape"] for val_output in query_to_val.values()]

        print(f"Epoch {epoch + 1} finished ({sum(query_loss_epoch)/len(query_loss_epoch)})")
        print(f"Validation loss: {sum(val_losses)/len(val_losses)}, mape: {sum(val_mape)/len(val_mape)}")


def main_simulated_training(train_dataset, val_dataset,
                            oracle_model, embedding_model, cost_model,
                            device, query_batch_size, train_batch_size,
                            save_loc_simulated_dataset, save_loc_simulated_dataset_val):
    oracle_model = oracle_model.to(device)
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, save_loc_simulated_dataset)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, save_loc_simulated_dataset_val)
    query_plan_dict_val = {k: v for d in val_data for k, v in d.items()}
    # TODO: Separate oracle model (large training guide) and query embedding model (light weight embedder)
    train_simulated(queries_train=train_dataset, query_plans_train=query_plans_dict,
                    queries_val=val_dataset, query_plans_val=query_plan_dict_val,
                    model_query=embedding_model, model_plan=cost_model,
                    device=device,
                    query_batch_size=query_batch_size,
                    train_batch_size=train_batch_size
                    )

    pass


def main_supervised_value_estimation(endpoint_location,
                                     queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated_dataset, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_directory_oracle,
                                     model_config_embedder
                                     ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cost_net_attention_pooling = PyGCardinalityHead(256)

    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    oracle_model = prepare_cardinality_estimator(model_config=model_config_oracle,
                                                                        model_directory=model_directory_oracle)
    embedding_model = prepare_cardinality_estimator(model_config=model_config_embedder)
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, 200, device)
    main_simulated_training(train_dataset, val_dataset,
                            oracle_model, embedding_model,
                            cost_net_attention_pooling,
                            device,
                            6, 128,
                            save_loc_simulated_dataset, save_loc_simulated_val,)


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/yago/sparql"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"
    model_config_oracle = "experiments/model_configs/policy_networks/t_cv_repr_huge.yaml"
    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"
    experiment_dir = "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce_large_pretrain-05-10-2025-18-13-40"
    save_loc_simulated = "data/simulated_query_plan_data/star_yago_gnce/data"
    save_loc_simulated_val = "data/simulated_query_plan_data/star_yago_gnce/val_data"

    model_dir_oracle = find_best_epoch_directory(experiment_dir, "val_q_error")

    main_supervised_value_estimation(endpoint_location, queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_dir_oracle,
                                     model_config_emb
                                     )
