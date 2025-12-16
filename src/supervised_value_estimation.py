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
import gc
import json
import math
import os
import sys
from abc import abstractmethod

import time
# IDEAS:
# Two losses and estimation heads: Latency and Cost
# MoE in graph model
# Uncertainty aware MoE
from functools import partial
import psutil
from torch_geometric.nn import GlobalAttention, AttentionalAggregation
from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader
from itertools import chain
# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('..'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.models.model_layers.tcnn import build_t_cnn_tree_from_order, transformer, left_child, right_child, \
    build_t_cnn_trees, BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining, prepare_queries
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.tree_conv_utils import build_trees_and_indexes, \
    precompute_left_deep_tree_conv_index, precompute_left_deep_trees_placeholders, fill_placeholder_trees, \
    precompute_left_deep_tree_node_mask

import torch
import torch.nn as nn

class QueryPlansPredictionModel(nn.Module):
    def __init__(self, query_emb_model, query_plan_model, device):
        super().__init__()
        self.query_emb_model = query_emb_model
        self.query_plan_model = query_plan_model
        self.device = device

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

    def estimate_cost(self, plans, embedded_query, precomputed_indexes, precomputed_masks):
        prepared_trees, prepared_indexes = build_trees_and_indexes(
            [[plan[0] for plan in plans]], [embedded_query],
            precomputed_indexes
        )
        prepared_trees.to(self.device)
        [prepared_index.to(self.device) for prepared_index in prepared_indexes]
        mask = self.get_node_masks(plans,  prepared_trees, embedded_query.shape[0], precomputed_masks)

        return self.query_plan_model.forward(prepared_trees, prepared_indexes[0], mask)

    def get_node_masks(self, plans, trees, max_nodes, precomputed_masks):
        mask = torch.ones((trees.shape[0], max_nodes*2), dtype=torch.bool)
        for i, plan in enumerate(plans):
            n_nodes_in_plan = len(plan[0])
            mask[i][:n_nodes_in_plan*2] = precomputed_masks[n_nodes_in_plan]
        return mask


class BasePlanCostEstimator(nn.Module):
    def __init__(self, device, feature_dim=100):
        super().__init__()
        self.device = device
        self.plan_embedding_nn, self.attn_pool, self.regressor = self.init_model(feature_dim, device)

    def forward(self, trees, indexes, mask_padding):
        # Encode the query plan
        print(trees)
        test_trees = trees.detach().cpu().numpy()
        emb, idx = self.plan_embedding_nn((trees, indexes))

        # Consider hierarchical application of our trees. What if we take the output representation of the
        # query as input to the next embedding / plan phase like in TinyHierarchicalReason modelling?

        # We reshape the (n_plans, dim, n_nodes) tensor to (n_plans, n_nodes, dim)
        emb_transposed = emb.transpose(1, 2)
        test_emb_transposed = emb_transposed.detach().cpu().numpy()

        # Stack the node embeddings to a tensor (n_plans * n_nodes, dim) to use with batch variable
        emb_stacked = emb_transposed.reshape((-1, emb_transposed.shape[-1]))
        test_stacked = emb_stacked.detach().cpu().numpy()

        # Create batch vector to represent the fact that we merged plans
        n_plans = emb_transposed.shape[0]
        n_nodes = emb_transposed.shape[1]
        plan_indices = torch.arange(n_plans, device=self.device)

        # Repeat each plan index n_nodes times
        batch = plan_indices.repeat_interleave(n_nodes)

        valid = ~mask_padding.reshape(-1)
        emb_masked_padding = emb_stacked[valid]
        batch_masked_padding = batch[valid]

        test_mask = mask_padding.detach().cpu().numpy()
        test_emb_non_padding = emb_masked_padding.detach().cpu().numpy()
        test_batch_non_padding = batch_masked_padding.detach().cpu().numpy()

        pool_vectors = self.attn_pool(emb_masked_padding, batch_masked_padding)
        test_pool_vecs = pool_vectors.detach().cpu().numpy()

        # Get root representation.
        # First element is the zero vector (that is padded out in attention), so we take index 1
        root_vectors = emb_transposed[:, 1, :]
        test_root_vecs = root_vectors.detach().cpu().numpy()

        #TODO: Consider if we should also add a pooled version of the graph. Reasoning: Separate the graph structure and
        # the plan structure. Easy to figure out, just make a small experiment
        combined = torch.cat([root_vectors, pool_vectors], dim=1)
        test = self.regressor(combined)
        test_numpy = test.detach().cpu().numpy()
        five = 5
        return self.regressor(combined), combined

        # pool_vectors = self.attn_pool(emb_stacked, batch)
        # test_pool_vecs = pool_vectors.detach().cpu().numpy()
        # #TODO: It seems that the pool vectors entry is equivalent to the root vector, thus our concat is just a
        # # repetition
        #
        # # Get root representation
        # root_vectors = emb_transposed[:, 1, :]
        # test_root_vecs = root_vectors.detach().cpu().numpy()
        # combined = torch.cat([root_vectors, pool_vectors], dim=1)
        # test = self.regressor(combined)
        # test_numpy = test.detach().cpu().numpy()
        # five = 5
        # return self.regressor(combined), combined

    @abstractmethod
    def init_model(self, feature_dim, device):
        pass

class PlanCostEstimatorFull(BasePlanCostEstimator):
    def init_model(self, feature_dim, device):
        gate_nn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        ).to(device)
        plan_embedding_nn = nn.Sequential(
            BinaryTreeConv(200, 200),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(200, 100),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(100, feature_dim),
        ).to(device)
        attn_pool = AttentionalAggregation(gate_nn=gate_nn, nn=None).to(device)

        # Input size is double feature_dim because we concat [Root, Attention_Pool]
        regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        return plan_embedding_nn, attn_pool, regressor

class PlanCostEstimatorTiny(BasePlanCostEstimator):
    def init_model(self, feature_dim, device):
        gate_nn = nn.Sequential(
            nn.Linear(feature_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(device)
        plan_embedding_nn = nn.Sequential(
            BinaryTreeConv(10, 10),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(10, 10),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(10, feature_dim),
        ).to(device)
        attn_pool = GlobalAttention(gate_nn=gate_nn, nn=None).to(device)

        # Input size is double feature_dim because we concat [Root, Attention_Pool]
        regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        ).to(device)
        return plan_embedding_nn, attn_pool, regressor


class EpistemicNetwork(nn.Module):
    def __init__(self,
                 epi_index_dim, prior_config,
                 cost_estimation_model: QueryPlansPredictionModel,
                 device=torch.device('cpu')):
        super().__init__()
        self.epi_index_dim = epi_index_dim
        self.cost_estimation_model = cost_estimation_model
        self.device = device
        # We use an ensemble of tiny gine_conv as priors
        model_factory_gine_conv = ModelFactory(prior_config)
        ensemble_gnn = [model_factory_gine_conv.load_gine_conv().to(device) for _ in range(epi_index_dim)]
        ensemble_plan_cost = [PlanCostEstimatorTiny(device, 5).to(device) for _ in range(epi_index_dim)]
        self.ensemble_combined_prior_models = [
            QueryPlansPredictionModel(ensemble_gnn[i], ensemble_plan_cost[i], device) for i in range(epi_index_dim)
        ]

        # The learnable and fixed MLPs should have same config all three parts should have same output dimension
        self.learnable_epinet = nn.Sequential(
            nn.Linear(64+epi_index_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)

        self.prior_epinet = nn.Sequential(
            nn.Linear(64+epi_index_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)

        #TODO: Check if this works
        for param in self.prior_epinet.parameters():
            param.requires_grad = False

        for combined_prior in self.ensemble_combined_prior_models:
            for param in combined_prior.parameters():
                param.requires_grad = False

    # This model should separate query embedding from plan enumeration.
    # Predicting cost should take as input embedded query + join orders and output the number of required
    # predictions of the epistemic network
    def embed_query_batched(self, queries):
        return self.cost_estimation_model.embed_query_batched(queries)

    def estimate_cost_full(self, plans, embedded_query, precomputed_indexes, precomputed_masks):
        return self.cost_estimation_model.estimate_cost(plans, embedded_query, precomputed_indexes, precomputed_masks)

    def embed_query_batched_prior(self, queries):
        embedded_query_batches = []
        for prior_model in self.ensemble_combined_prior_models:
            embedded_query_batches.append(prior_model.embed_query_batched(queries))
        return embedded_query_batches

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

    def compute_mlp_prior(self, last_feature, epi_index):
        #TODO: Concat this properly
        pass

    def compute_learnable_mlp(self, last_feature, epi_index):
        pass


    def compute_ensemble_prior(self, plans, embedded_query,
                               precomputed_indexes, precomputed_masks,
                               epi_index, query_idx):
        with torch.no_grad():
            # (epi_index, n_plans)
            estimated_cost_priors = torch.zeros((self.epi_index_dim, len(plans)))
            for i in range(self.epi_index_dim):
                prepared_trees, prepared_indexes = build_trees_and_indexes(
                    [[plan[0] for plan in plans]], [embedded_query[i][query_idx]],
                    precomputed_indexes
                )
                prepared_trees.to(self.device)
                [prepared_index.to(self.device) for prepared_index in prepared_indexes]

                # TODO: Check how we get the predictions here, as they seem too similar
                # We should check the input to the estimate cost and see how values flow there.
                # The weird thing is different plans get the same prediction, this might also indicate
                # insufficient expressivity of the model

                # Est_cost: (n_plans, 1)
                est_cost, _ = self.ensemble_combined_prior_models[i].estimate_cost(
                    plans, embedded_query[i][query_idx], precomputed_indexes, precomputed_masks
                )
                # Est_cost: (1, n_plans)
                est_cost_t = est_cost.transpose(0,1)
                estimated_cost_priors[i] = est_cost_t
            #TODO: Make this a weighted sum for each plan in plans between estimated cost and epi_index value currently not yet correct!
            weighted_sum = torch.matmul(epi_index, estimated_cost_priors)
        return weighted_sum

    def sample_epistemic_indexes(self):
        return torch.normal(0, 1, size=(1, self.epi_index_dim))


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


def prepare_simulated_dataset(dataset_to_prepare, oracle_model, device,
                              output_loc_raw,
                              max_plans_per_relation=50):
    # Dataset contains: [(Query, [(plans, cost for plans), ...]
    # Training is done over batches of Queries, to amortize the query embedding step. For each sub plan we create
    # tree-based embedding
    # Use different model for estimating cost and estimating plan (plan estimation smaller, big model can act as oracle)
    def deep_size(obj, seen=None):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(deep_size(i, seen) for i in obj)

        return size

    def print_memory_usage():
        process = psutil.Process(os.getpid())
        print(f"Memory: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    from pympler import muppy, summary, tracker

    # Option A: Summary of all objects
    def show_memory_summary():
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

    data = []
    # TODO Figure out how to run this through WSL: https://pypi.org/project/memray/ 
    if os.path.exists(output_loc_raw + '.jsonl'):
        print("Using pre-made plans")
        k = 0
        print_memory_usage()
        with open(output_loc_raw + '.jsonl', 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                data.append(json.loads(line))
                k += 1
                if k % 5000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()  # If using GPU
                
                    print_memory_usage()
                #     size_data = deep_size(data) / 1024**2
                #     print(f"Loaded data size: {size_data:.2f} MB")
                if k == 20000:
                    break
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

        write_plans_to_file({query[0].query: plan_to_sub_plans}, output_loc_raw)

    return data

def validate(queries_val, query_plans_val,
             precomputed_indexes, min_cost, max_cost,
             train_loss,
             combined_model,
             device):
    query_to_val = {}
    mape = MeanAbsolutePercentageError()
    mape.to(device)
    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)
    for queries in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            embedded= combined_model.embed_query_batched(queries)

            plans = query_plans_val[queries.query[0]]
            target = torch.tensor([plan[1] for plan in plans], device=device)

            output, _ = combined_model.estimate_cost(plans, embedded[0], precomputed_indexes)
            output = output.squeeze()

            original_cost = output * (max_cost - min_cost) + min_cost

            mape_val = mape(original_cost, target)
            query_loss = train_loss(original_cost, target)

            query_to_val[queries.query[0]] = {"loss": query_loss.cpu().item(), "mape": mape_val.cpu().item()}
    return query_to_val


def train_simulated(queries_train, query_plans_train,
                    queries_val, query_plans_val,
                    combined_model, epinet_cost_estimation: EpistemicNetwork,
                    device, query_batch_size):
    combined_model.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20, device)
    precomputed_masks = precompute_left_deep_tree_node_mask(20, device)
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)

    query_plans_train = flatten_plans(query_plans_train)
    query_plans_train, min_val, max_val = min_max_scale_plans(query_plans_train)

    query_plans_val = flatten_plans(query_plans_val)

    # --- 1. Define Hyperparameters ---
    lr = 1e-4
    n_epochs = 10

    params = list(combined_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=0.01  # Standard L2 regularization for AdamW
    )
    total_params = 0
    for param in params:
        total_params += param.numel()

    print(f"Combined model has {total_params} parameters")

    loss = torch.nn.MSELoss(reduction='mean')

    for epoch in range(n_epochs):
        query_loss_epoch = []
        for k, queries in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()
            embedded = combined_model.embed_query_batched(queries)

            embedded_epinet = epinet_cost_estimation.embed_query_batched(queries)
            embedded_prior = epinet_cost_estimation.embed_query_batched_prior(queries)

            total_loss_tensor = torch.tensor(0.0, device=device)
            for i in range(len(queries.query)):
                test_embedded_epinet = embedded_epinet[i].cpu().detach().numpy()
                test_embedded_prior = embedded_prior[0][i].cpu().detach().numpy()
                plans = query_plans_train[queries.query[i]]

                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans, embedded_epinet[i], precomputed_indexes, precomputed_masks
                )
                # Apply stop gradient operator to last feature to serve as input to epinet
                last_feature = last_feature.detach()

                #TODO Here we just do one epistemic index for all plans, however that might not be the way to go?
                #TODO: Need to do this with multiple epinet indexes sampled so in a loop
                epinet_index = epinet_cost_estimation.sample_epistemic_indexes()
                epinet_cost_estimation.compute_ensemble_prior(
                    plans, embedded_prior, precomputed_indexes, precomputed_masks, epinet_index, i
                )
                epinet_cost_estimation.compute_mlp_prior(last_feature, epinet_index)

                target = torch.tensor([plan[1] for plan in plans], device=device)

                output, _ = combined_model.estimate_cost(plans, embedded[i], precomputed_indexes)

                query_loss = loss(output.squeeze(), target)
                query_loss_epoch.append(query_loss.detach().cpu().item())
                total_loss_tensor += query_loss

            total_loss_tensor.backward()

            optimizer.step()

        query_to_val = validate(queries_val, query_plans_val,
                                 precomputed_indexes,
                                 min_val, max_val,
                                 loss,
                                 combined_model,
                                 device)
        val_losses = [val_output["loss"] for val_output in query_to_val.values()]
        val_mape = [val_output["mape"] for val_output in query_to_val.values()]

        print(f"Epoch {epoch + 1} finished ({sum(query_loss_epoch)/len(query_loss_epoch)})")
        print(f"Validation loss: {sum(val_losses)/len(val_losses)}, mape: {sum(val_mape)/len(val_mape)}")


def main_simulated_training(train_dataset, val_dataset,
                            oracle_model,
                            combined_model, epinet_cost_estimation,
                            device, query_batch_size,
                            save_loc_simulated_dataset, save_loc_simulated_dataset_val):
    oracle_model = oracle_model.to(device)
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, save_loc_simulated_dataset)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, save_loc_simulated_dataset_val)
    query_plan_dict_val = {k: v for d in val_data for k, v in d.items()}
    train_simulated(queries_train=train_dataset, query_plans_train=query_plans_dict,
                    queries_val=val_dataset, query_plans_val=query_plan_dict_val,
                    combined_model=combined_model,
                    epinet_cost_estimation=epinet_cost_estimation,
                    device=device,
                    query_batch_size=query_batch_size,
                    )

def main_supervised_value_estimation(endpoint_location,
                                     queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated_dataset, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_directory_oracle,
                                     model_config_embedder, model_directory_embedder,
                                     model_config_epistemic_prior,
                                     ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    oracle_model = prepare_cardinality_estimator(
        model_config=model_config_oracle, model_directory=model_directory_oracle
    )

    cost_net_attention_pooling = PlanCostEstimatorFull(device, 100)
    embedding_model = prepare_cardinality_estimator(model_config=model_config_embedder,
                                                    model_directory=model_directory_embedder)
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, device)

    epinet_cost_estimation = EpistemicNetwork(8, model_config_epistemic_prior, combined_model, device=device)

    main_simulated_training(train_dataset, val_dataset,
                            oracle_model,
                            combined_model,
                            epinet_cost_estimation,
                            device,
                            2,
                            save_loc_simulated_dataset, save_loc_simulated_val,)


import gc
import sys
import torch
import psutil
import os
from pympler import asizeof


def comprehensive_memory_analysis():
    """Find ALL memory consumers including hidden ones"""

    process = psutil.Process(os.getpid())
    total_rss = process.memory_info().rss / 1024 ** 3
    print(f"\n{'=' * 70}")
    print(f"Total Process Memory (RSS): {total_rss:.2f} GB")
    print(f"{'=' * 70}\n")

    # 1. Check PyTorch GPU memory
    print("1. GPU Memory (PyTorch):")
    print("-" * 70)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"   GPU {i}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
    else:
        print("   No CUDA devices")
    print()

    # 2. Check PyTorch CPU tensors
    print("2. PyTorch CPU Tensors:")
    print("-" * 70)
    cpu_tensors = []
    total_tensor_memory = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cpu':
                size_mb = obj.numel() * obj.element_size() / 1024 ** 2
                total_tensor_memory += size_mb
                if size_mb > 10:  # Only show tensors > 10MB
                    cpu_tensors.append({
                        'shape': tuple(obj.shape),
                        'size_mb': size_mb,
                        'dtype': obj.dtype
                    })
        except:
            pass

    cpu_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    print(f"   Total CPU Tensor Memory: {total_tensor_memory / 1024:.2f} GB")
    print(f"   Number of CPU tensors: {len(cpu_tensors)}")
    print(f"\n   Top 10 largest CPU tensors:")
    for i, t in enumerate(cpu_tensors[:10], 1):
        print(f"   {i:2d}. Shape: {str(t['shape']):30s} "
              f"Size: {t['size_mb']:>10.2f} MB  dtype: {t['dtype']}")
    print()

    # 3. Check for large lists/dicts with deep size
    print("3. Large Container Objects (Deep Size):")
    print("-" * 70)
    large_objects = []

    for obj in gc.get_objects():
        try:
            obj_type = type(obj).__name__
            if obj_type in ['list', 'dict', 'tuple', 'set']:
                # Only check objects with some minimum shallow size
                shallow_size = sys.getsizeof(obj)
                if shallow_size > 1024 * 100:  # > 100 KB shallow
                    deep_size = asizeof.asizeof(obj) / 1024 ** 2
                    if deep_size > 50:  # > 50 MB deep
                        large_objects.append({
                            'type': obj_type,
                            'shallow_mb': shallow_size / 1024 ** 2,
                            'deep_mb': deep_size,
                            'len': len(obj) if hasattr(obj, '__len__') else 0,
                            'id': id(obj)
                        })
        except:
            pass

    large_objects.sort(key=lambda x: x['deep_mb'], reverse=True)
    total_container_memory = sum(obj['deep_mb'] for obj in large_objects) / 1024
    print(f"   Total Large Container Memory: {total_container_memory:.2f} GB")
    print(f"\n   Top 10 largest containers:")
    for i, obj in enumerate(large_objects[:10], 1):
        print(f"   {i:2d}. {obj['type']:10s} len={obj['len']:>8d}  "
              f"Deep Size: {obj['deep_mb']:>10.2f} MB  "
              f"(shallow: {obj['shallow_mb']:.2f} MB)")
    print()

    # 4. Check for C/C++ extensions memory (won't show in Python)
    print("4. Potential Hidden Memory:")
    print("-" * 70)
    accounted_memory = (total_tensor_memory / 1024 + total_container_memory)
    unaccounted = total_rss - accounted_memory
    print(f"   Accounted for: {accounted_memory:.2f} GB")
    print(f"   Unaccounted:   {unaccounted:.2f} GB")
    print()

    if unaccounted > 1.0:
        print("   Likely causes of unaccounted memory:")
        print("   - PyTorch C++ backend allocations")
        print("   - NumPy arrays")
        print("   - Memory fragmentation")
        print("   - Shared libraries")
        print("   - Python interpreter overhead")
    print()

    # 5. Memory fragmentation check
    print("5. Memory Fragmentation:")
    print("-" * 70)
    mem_info = process.memory_info()
    print(f"   RSS (actual memory):     {mem_info.rss / 1024 ** 3:.2f} GB")
    print(f"   VMS (virtual memory):    {mem_info.vms / 1024 ** 3:.2f} GB")
    if hasattr(mem_info, 'data'):
        print(f"   Data segment:            {mem_info.data / 1024 ** 3:.2f} GB")
    print()

    return {
        'total_rss_gb': total_rss,
        'tensor_memory_gb': total_tensor_memory / 1024,
        'container_memory_gb': total_container_memory,
        'unaccounted_gb': unaccounted,
        'large_objects': large_objects
    }


def find_specific_large_lists():
    """Specifically hunt for your JSON-loaded data"""
    print("\n6. Hunting for JSON/Data Lists:")
    print("=" * 70)

    for obj in gc.get_objects():
        try:
            if isinstance(obj, list):
                if len(obj) > 1000:  # Large lists
                    # Check if it contains dicts (likely JSON data)
                    if obj and isinstance(obj[0], dict):
                        size_mb = asizeof.asizeof(obj) / 1024 ** 2
                        if size_mb > 100:
                            print(f"Found large list: len={len(obj)}, "
                                  f"size={size_mb:.2f} MB")
                            # Try to identify it
                            if obj[0]:
                                print(f"  First element keys: {list(obj[0].keys())[:5]}")
                            print(f"  Object ID: {id(obj)}")
                            print()
        except:
            pass


def check_numpy_arrays():
    """Check for NumPy arrays which pympler might miss"""
    try:
        import numpy as np
        print("\n7. NumPy Arrays:")
        print("=" * 70)

        arrays = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray):
                    size_mb = obj.nbytes / 1024 ** 2
                    if size_mb > 10:
                        arrays.append({
                            'shape': obj.shape,
                            'dtype': obj.dtype,
                            'size_mb': size_mb
                        })
            except:
                pass

        arrays.sort(key=lambda x: x['size_mb'], reverse=True)
        total_np = sum(a['size_mb'] for a in arrays) / 1024
        print(f"Total NumPy memory: {total_np:.2f} GB")
        print(f"Number of arrays: {len(arrays)}")
        for i, arr in enumerate(arrays[:10], 1):
            print(f"{i:2d}. Shape: {str(arr['shape']):30s} "
                  f"Size: {arr['size_mb']:>10.2f} MB  dtype: {arr['dtype']}")
        print()
    except ImportError:
        print("NumPy not available")


def full_memory_debug():
    """Run all checks"""
    gc.collect()  # Clean up first
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    comprehensive_memory_analysis()
    find_specific_large_lists()
    check_numpy_arrays()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("If you still have unaccounted memory:")
    print("1. Run: torch.cuda.empty_cache() to clear GPU caches")
    print("2. Check if you're holding references to old DataLoader batches")
    print("3. Look for cached computation graphs (gradients not released)")
    print("4. Try: gc.collect() to force garbage collection")
    print("5. Use tracemalloc to see where allocations happened")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/yago/sparql"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_oracle = "experiments/model_configs/policy_networks/t_cv_repr_huge.yaml"
    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"
    model_config_emb_pair_norm = "experiments/model_configs/policy_networks/t_cv_repr_pair_norm_cardinality_head_own_embeddings.yaml"
    model_config_emb_graph_norm = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_cardinality_head_own_embeddings.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_tiny.yaml"


    oracle_experiment_dir = "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce_large_pretrain-05-10-2025-18-13-40"

    emb_experiment_dir = ("experiments/experiment_outputs/yago_gnce/pretrained_models/"
                      "pretrain_experiment_triple_conv-15-12-2025-11-10-45")
    emb_experiment_dir_pair_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_pair_norm-15-12-2025-10-00-26")

    emb_experiment_dir_graph_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_graph_norm-15-12-2025-09-12-57")

    save_loc_simulated = "data/simulated_query_plan_data/star_yago_gnce/data"
    save_loc_simulated_val = "data/simulated_query_plan_data/star_yago_gnce/val_data"


    model_dir_oracle = find_best_epoch_directory(oracle_experiment_dir, "val_q_error")
    model_dir_embedder = find_best_epoch_directory(emb_experiment_dir, "val_q_error")

    main_supervised_value_estimation(endpoint_location, queries_location_train, queries_location_val,
                                     rdf2vec_vector_location,
                                     save_loc_simulated, save_loc_simulated_val,
                                     occurrences_location, tp_cardinality_location,
                                     model_config_oracle, model_dir_oracle,
                                     model_config_emb, model_dir_embedder,
                                     model_config_prior
                                     )