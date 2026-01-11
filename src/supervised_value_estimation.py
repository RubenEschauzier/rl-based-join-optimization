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
import sys
import numpy as np
from abc import abstractmethod, ABC

import time
from functools import partial
from torch_geometric.nn import GlobalAttention, AttentionalAggregation
from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.models.model_instantiator import ModelFactory
from src.models.model_layers.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask, \
    get_shared_structure, apply_features_to_structure

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

    def estimate_cost(self, prepared_trees, prepared_indexes, prepared_masks):
        return self.query_plan_model.forward(prepared_trees, prepared_indexes, prepared_masks)

    # def build_t_cnn_input(self, join_orders_batched, features_batch,
    #                      precomputed_indexes, precomputed_masks, cache):
    #     """
    #     Optimized version of build_trees_and_indexes with caching and vectorization.
    #
    #     Args:
    #         join_orders_batched: List of lists of join orders (plans).
    #         features_batch: List of feature tensors (queries).
    #         precomputed_indexes: Dictionary of precomputed convolution indexes.
    #         precomputed_masks: Dictionary of precomputed node masks based on the number of join entries
    #         cache: A dictionary to store structural indices (pass self.cache here).
    #     """
    #     device = features_batch[0].device
    #     dtype = features_batch[0].dtype
    #
    #     # Lists to hold the final results for the whole batch
    #     batched_trees_results = []
    #     batched_indexes_results = []
    #     batched_mask_results = []
    #
    #     # Iterate over the batch (usually batch size is 1 for [plans] vs [query])
    #     for batch_idx, (join_orders, features) in enumerate(zip(join_orders_batched, features_batch)):
    #         n_nodes = features.shape[0]
    #
    #         # Create a unique key for the structure of this specific batch item
    #         # We need (tuple of join orders, number of nodes)
    #         # We convert lists to tuples to make them hashable
    #         structure_key = (tuple(tuple(jo) for jo in join_orders), n_nodes)
    #
    #         if structure_key in cache:
    #             # HIT: Retrieve pre-calculated structural tensors
    #             gather_indices, conv_indexes, mask = cache[structure_key]
    #         else:
    #             # MISS: Compute them efficiently and cache them
    #             gather_indices, conv_indexes, mask = self._compute_structural_indices(
    #                 join_orders, n_nodes, precomputed_indexes, precomputed_masks)
    #             cache[structure_key] = (gather_indices, conv_indexes, mask)
    #
    #         if gather_indices.device != device:
    #             gather_indices = gather_indices.to(device)
    #             conv_indexes = conv_indexes.to(device)
    #
    #         # Pre-compute zero vector and append to features
    #         # Shape: (n_nodes + 1, channels)
    #         channels = features.shape[1]
    #         zero_vec = torch.zeros((1, channels), dtype=dtype, device=device)
    #         features_padded = torch.cat([features, zero_vec], dim=0)
    #
    #         # Expand features: (1, n_nodes+1, channels) -> (num_plans, n_nodes+1, channels)
    #         num_plans = gather_indices.shape[0]
    #         features_expanded = features_padded.unsqueeze(0).expand(num_plans, -1, -1)
    #
    #         # Expand indices: (num_plans, tree_width) -> (num_plans, tree_width, channels)
    #         idx_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, channels)
    #
    #         # Gather: (num_plans, tree_width, channels)
    #         flattened_trees = torch.gather(features_expanded, 1, idx_expanded)
    #
    #         # Transpose to match original output: (num_plans, channels, tree_width)
    #         trees = flattened_trees.transpose(1, 2)
    #
    #         batched_trees_results.append(trees)
    #         batched_indexes_results.append(conv_indexes)
    #         batched_mask_results.append(mask)
    #
    #     final_trees = torch.cat(batched_trees_results, dim=0)
    #     final_indexes = torch.cat(batched_indexes_results, dim=0)
    #     final_masks = torch.cat(batched_mask_results, dim=0)
    #     return final_trees, final_indexes, final_masks
    #
    # def _compute_structural_indices(self, join_orders, n_nodes, precomputed_indexes, precomputed_masks):
    #     """
    #     Internal helper to build indices using fast NumPy vectorization and avoiding GPU communication costs
    #     """
    #     lengths = [len(jo) for jo in join_orders]
    #     max_len = max(lengths)
    #     max_nodes_in_batch = 2 * max_len
    #
    #     # Create the base array filled with 'n_nodes' (which points to the zero-vector)
    #     # Shape: (num_plans, max_nodes_in_batch)
    #     # Using int64 for indices
    #     batch_indices = np.full((len(join_orders), max_nodes_in_batch), n_nodes, dtype=np.int64)
    #
    #     # Fill in the actual join orders
    #     # Original logic: padded_order = [n_nodes]*len + join_order + [n_nodes]*remainder
    #     # This means the join_order data starts at index `len(jo)`
    #     for i, jo in enumerate(join_orders):
    #         l = len(jo)
    #         # We copy the join order into the middle of the array
    #         # The left side (0 to l) is already n_nodes
    #         # The right side (2*l to end) is already n_nodes
    #         batch_indices[i, l: l + l] = jo
    #
    #     gather_indices = torch.from_numpy(batch_indices).to(self.device)
    #
    #     # --- 2. Build Conv Indexes ---
    #     # Logic from original: max size based on precomputed_indexes shapes
    #     max_conv_size = max([precomputed_indexes[l].shape[0] for l in lengths])
    #
    #     conv_indexes = torch.zeros((len(join_orders), max_conv_size, 1), dtype=torch.long, device=self.device)
    #
    #     for i, l in enumerate(lengths):
    #         t = precomputed_indexes[l]
    #         actual_size = t.shape[0]
    #         conv_indexes[i, :actual_size] = t
    #
    #     target_width = n_nodes * 2
    #
    #     # Create batch mask on CPU using Numpy
    #     batch_mask_np = np.ones((len(join_orders), target_width), dtype=bool)
    #
    #     # We need the raw mask data available here.
    #     # Assumption: precomputed_masks is a dict of Tensors.
    #     # It is faster to have precomputed_masks_np (dict of numpy arrays) for this step.
    #
    #     for i, jo in enumerate(join_orders):
    #         l = len(jo)
    #         # Convert to numpy if it isn't already (ideally convert dict to numpy once in init)
    #         mask_data = precomputed_masks[l]
    #         batch_mask_np[i, :mask_data.shape[0]] = mask_data
    #
    #     masks = torch.from_numpy(batch_mask_np).to(self.device)
    #     return gather_indices, conv_indexes, masks
    #
    # def get_node_masks(self, plans, trees, max_nodes, precomputed_masks):
    #     mask = torch.ones((trees.shape[0], max_nodes*2), dtype=torch.bool, device=self.device)
    #     for i, plan in enumerate(plans):
    #         n_nodes_in_plan = len(plan[0])
    #         mask[i][:n_nodes_in_plan*2] = torch.from_numpy(precomputed_masks[n_nodes_in_plan]).to(self.device)
    #     return mask

class BasePlanCostEstimator(nn.Module, ABC):
    def __init__(self, device, feature_dim=100):
        super().__init__()
        self.device = device
        self.plan_embedding_nn, self.attn_pool, self.regressor = self.init_model(feature_dim, device)

    def forward(self, trees, indexes, mask_padding):
        # Encode the query plan
        emb, idx = self.plan_embedding_nn((trees, indexes))

        # Consider hierarchical application of our trees. What if we take the output representation of the
        # query as input to the next embedding / plan phase like in TinyHierarchicalReason modeling?

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

        valid = ~mask_padding.reshape(-1)
        emb_masked_padding = emb_stacked[valid]
        batch_masked_padding = batch[valid]

        pool_vectors = self.attn_pool(emb_masked_padding, batch_masked_padding)

        # Get root representation.
        # First element is the zero vector (that is padded out in attention), so we take index 1
        root_vectors = emb_transposed[:, 1, :]

        #TODO: Consider if we should also add a pooled version of the graph. Reasoning: Separate the graph structure and
        # the plan structure. Easy to figure out, just make a small experiment
        combined = torch.cat([root_vectors, pool_vectors], dim=1)
        return self.regressor(combined), combined

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

        self.tree_cache = {}
        # We use an ensemble of tiny gine_conv as priors
        model_factory_gine_conv = ModelFactory(prior_config)
        ensemble_gnn = [model_factory_gine_conv.load_gine_conv().to(device) for _ in range(epi_index_dim)]
        ensemble_plan_cost = [PlanCostEstimatorTiny(device, 5).to(device) for _ in range(epi_index_dim)]
        self.ensemble_combined_prior_models = [
            QueryPlansPredictionModel(ensemble_gnn[i], ensemble_plan_cost[i], device) for i in range(epi_index_dim)
        ]

        # The learnable and fixed MLPs should have same config all three parts should have same output dimension
        self.learnable_epinet = nn.Sequential(
            nn.Linear(200+epi_index_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)

        self.prior_epinet = nn.Sequential(
            nn.Linear(200+epi_index_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)

        for param in self.prior_epinet.parameters():
            param.requires_grad = False

        for combined_prior in self.ensemble_combined_prior_models:
            for param in combined_prior.parameters():
                param.requires_grad = False

    def embed_query_batched(self, queries):
        return self.cost_estimation_model.embed_query_batched(queries)

    def estimate_cost_full(self, plans, embedded_query, precomputed_indexes, precomputed_masks):
        join_orders = [plan[0] for plan in plans]
        n_nodes = embedded_query.shape[0]
        gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
            join_orders, n_nodes, precomputed_indexes,
            precomputed_masks, self.tree_cache, self.device
        )
        prepared_trees = apply_features_to_structure(embedded_query, gather_indices)
        return self.cost_estimation_model.estimate_cost(
            prepared_trees, prepared_indexes, prepared_masks
        )

    def embed_query_batched_prior(self, queries):
        embedded_query_batches = []
        for prior_model in self.ensemble_combined_prior_models:
            embedded_query_batches.append(prior_model.embed_query_batched(queries))
        return embedded_query_batches

    def forward(self):
        #TODO Move logic into forward pass?
        # Maybe hold off until we see what non-simulated requirements will be
        #TODO:
        # We can do the calc of indexes again with cache when we train on actual query execution though if it turns out
        # very slow

        #TODO: For beam search we should investigate thompson sampling using epinet and just whatever that other paper
        # proposed.

        #TODO: Ideas
        # Two losses and estimation heads: Latency and Cost?
        # MoE in graph model
        # Uncertainty aware MoE with router based on epinet uncertainty

        pass

    def compute_mlp_prior(self, last_feature, epi_index):
        concat_input = torch.cat([last_feature, epi_index.expand(last_feature.shape[0], -1)], dim=1)
        return self.prior_epinet(concat_input)

    def compute_learnable_mlp(self, last_feature, epi_index):
        concat_input = torch.cat([last_feature, epi_index.expand(last_feature.shape[0], -1)], dim=1)
        return self.learnable_epinet(concat_input)


    def compute_ensemble_prior(self, plans, embedded_query,
                               precomputed_indexes, precomputed_masks,
                               epi_index, query_idx):
        with torch.no_grad():
            num_plans = len(plans)
            num_ensembles = self.epi_index_dim
            join_orders = [plan[0] for plan in plans]

            # Determine query size from the first ensemble's embedding
            n_nodes = embedded_query[0][query_idx].shape[0]

            # Each query has same gather_indices, indexes and masks. So precompute
            gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
                join_orders, n_nodes, precomputed_indexes,
                precomputed_masks, self.tree_cache, self.device
            )
            # (epi_index, n_plans)
            estimated_cost_priors = torch.zeros((self.epi_index_dim, num_plans), device=self.device)

            # Precompute tree structure as this is fixed between epinet dimensions
            for i in range(num_ensembles):
                # i-th ensemble features: (n_nodes, channels)
                current_features = embedded_query[i][query_idx]

                # Fast feature mapping using the shared 'gather_indices' blueprint
                prepared_trees = apply_features_to_structure(current_features, gather_indices)

                # Est_cost: (n_plans, 1)
                est_cost, _ = self.ensemble_combined_prior_models[i].estimate_cost(
                    prepared_trees, prepared_indexes, prepared_masks
                )
                # Est_cost: (1, n_plans)
                est_cost_t = est_cost.transpose(0,1)
                estimated_cost_priors[i] = est_cost_t
            weighted_sum = torch.matmul(epi_index, estimated_cost_priors)
        return weighted_sum

    def sample_epistemic_indexes(self):
        return torch.normal(0, 1, size=(1, self.epi_index_dim), device=self.device)


def min_max_scale_plans(query_plans, min_cost=None, max_cost=None):
    if not min_cost or not max_cost:
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

    data = []
    if os.path.exists(output_loc_raw + '.jsonl'):
        k = 0
        with open(output_loc_raw + '.jsonl', 'r', encoding="utf-8") as f:
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

        write_plans_to_file({query[0].query: plan_to_sub_plans}, output_loc_raw)

    return data

def validate(queries_val, query_plans_val,
             precomputed_indexes, precomputed_masks,
             min_cost, max_cost,
             train_loss,
             epinet_cost_estimation,
             device,
             validate_epi_network, n_val_epi_indexes):
    query_to_val_metrics = {}
    mape = MeanAbsolutePercentageError()
    mape.to(device)
    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)
    for queries in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            embedded = epinet_cost_estimation.embed_query_batched(queries)

            embedded_prior = None
            if validate_epi_network:
                embedded_prior = epinet_cost_estimation.embed_query_batched_prior(queries)

            plans = query_plans_val[queries.query[0]]
            estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                plans, embedded[0], precomputed_indexes, precomputed_masks
            )

            query_metrics = {}

            if validate_epi_network:
                loss_epinet, repeated_target, epinet_cost_estimates = calculate_loss_epinet(
                    epinet_cost_estimation=epinet_cost_estimation,
                    loss=train_loss,
                    estimated_cost=estimated_cost, last_feature=last_feature,
                    embedded_prior=embedded_prior,
                    plans=plans, precomputed_indexes=precomputed_indexes, precomputed_masks=precomputed_masks, i=0,
                    n_epi_indexes=n_val_epi_indexes, device=device
                )
                validation_metrics = compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_val_epi_indexes,
                                                  min_cost, max_cost)
                query_metrics.update(validation_metrics)
            estimated_cost = estimated_cost.squeeze()

            #TODO Where is target being scaled? it should probably also be scaled (CHECK IT)
            target = torch.tensor([plan[1] for plan in plans], device=device)
            original_cost = estimated_cost * (max_cost - min_cost) + min_cost

            mape_val = mape(original_cost, target)
            query_loss = train_loss(original_cost, target)

            query_metrics["loss_cost"] = query_loss.cpu().item()
            query_metrics["mape_cost"] = mape_val.cpu().item()
            query_to_val_metrics[queries.query[0]] = query_metrics
    return query_to_val_metrics

def compute_validation_metrics_epinet(epinet_cost_estimates, repeated_target, n_epi_indexes, min_cost, max_cost):
    #TODO Investigate this code
    #TODO Add train loss of mean epinet prediction per plan in scaled values
    n_total = repeated_target.shape[0]
    n_plans = n_total // n_epi_indexes

    # 1. Move to CPU and Numpy
    # Reshape to (-1, 1) because scaler expects 2D array [samples, features]
    pred_flat = epinet_cost_estimates.detach().cpu().numpy().reshape(-1, 1)
    targets_flat = repeated_target.detach().cpu().numpy().reshape(-1, 1)

    # 2. Unscale Individually (Robust Method)
    # We unscale every single prediction point before averaging
    cost_range = max_cost - min_cost

    pred_unscaled = pred_flat * cost_range + min_cost
    y_scaled = targets_flat[:n_plans].flatten()
    y_true = y_scaled * cost_range + min_cost

    # Shape: (n_epi_indexes, n_plans)
    # Column j contains all 'n_epi_indexes' predictions for Plan j
    pred_matrix = pred_unscaled.reshape(n_epi_indexes, n_plans)
    pred_matrix_scaled = pred_flat.reshape(n_epi_indexes, n_plans)

    y_pred_mean = pred_matrix.mean(axis=0)
    y_pred_mean_scaled = pred_matrix_scaled.mean(axis=0)
    y_pred_std = pred_matrix.std(axis=0)


    mse = np.mean((y_pred_mean - y_true) ** 2)
    mse_scaled = np.mean((y_pred_mean_scaled - y_scaled) ** 2)

    abs_error = np.abs(y_pred_mean - y_true)
    uncertainty_corr, _ = pearsonr(abs_error, y_pred_std)

    # C. Calibration: 95% Interval Coverage
    # Does the truth fall within Mean ± 1.96 * Std?
    lower_bound = y_pred_mean - 1.96 * y_pred_std
    upper_bound = y_pred_mean + 1.96 * y_pred_std

    is_in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage_95 = np.mean(is_in_interval)

    return {
        "epi_mse": mse,
        "epi_mse_scaled": mse_scaled,
        "epi_uncertainty_corr": uncertainty_corr,
        "epi_coverage_95": coverage_95,
        "epi_avg_std": np.mean(y_pred_std)
    }

def summarize_epistemic_metrics(metrics_dict):
    keys = [
        "epi_mse",
        "epi_mse_scaled",
        "epi_uncertainty_corr",
        "epi_coverage_95",
        "epi_avg_std",
    ]

    summary = {
        k: np.mean([q_metrics[k] for q_metrics in metrics_dict.values()])
        for k in keys
    }

    print(
        "[Epistemic metrics — mean over queries]\n"
        f"  MSE            : {summary['epi_mse']:.4f}\n"
        f"  Scaled MSE     : {summary['epi_mse_scaled']:.4f}\n"
        f"  Unc. Corr      : {summary['epi_uncertainty_corr']:.3f}\n"
        f"  Coverage @95%  : {summary['epi_coverage_95']:.3f}\n"
        f"  Avg Std        : {summary['epi_avg_std']:.4f}"
    )

    return summary

def train_simulated_cost_model(queries_train, query_plans_train,
                                queries_val, query_plans_val,
                                epinet_cost_estimation: EpistemicNetwork,
                                device,
                                query_batch_size,
                                n_epi_indexes, train_epi_network: bool):
    epinet_cost_estimation.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20, device)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)

    query_plans_train = flatten_plans(query_plans_train)
    query_plans_train, min_val, max_val = min_max_scale_plans(query_plans_train)

    # Scale with min and max from validation set
    query_plans_val = flatten_plans(query_plans_val)
    query_plans_val, _, _ = min_max_scale_plans(query_plans_val, min_val, max_val)

    lr = 1e-4
    n_epochs = 10

    # Freeze base cost model when training the epistemic network
    if train_epi_network:
        for param in epinet_cost_estimation.cost_estimation_model.parameters():
            param.requires_grad = False
        epinet_cost_estimation.cost_estimation_model.eval()

    params = list(epinet_cost_estimation.parameters())
    params_cost_estimate = list(epinet_cost_estimation.cost_estimation_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=0.01
    )

    total_params_cost_estimation = 0
    for param in params_cost_estimate:
        total_params_cost_estimation += param.numel()
    print(f"Cost estimation model has {total_params_cost_estimation} parameters")

    if train_epi_network:
        total_params = 0
        for param in epinet_cost_estimation.parameters():
            total_params += param.numel()
        print(f"Epinet model has {total_params - total_params_cost_estimation} parameters")

    loss = torch.nn.MSELoss(reduction='mean')

    for epoch in range(n_epochs):
        query_loss_epoch = []
        for k, queries in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()

            embedded = epinet_cost_estimation.embed_query_batched(queries)

            embedded_prior = None
            if train_epi_network:
                embedded_prior = epinet_cost_estimation.embed_query_batched_prior(queries)

            total_loss_tensor = torch.tensor(0.0, device=device)
            for i in range(len(queries.query)):
                plans = query_plans_train[queries.query[i]]
                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans, embedded[i], precomputed_indexes, precomputed_masks
                )

                if train_epi_network:
                    loss_epinet, _, _ = calculate_loss_epinet(
                        epinet_cost_estimation=epinet_cost_estimation,
                        loss = loss,
                        estimated_cost=estimated_cost, last_feature=last_feature,
                        embedded_prior=embedded_prior,
                        plans=plans, precomputed_indexes=precomputed_indexes, precomputed_masks=precomputed_masks, i=i,
                        n_epi_indexes=n_epi_indexes, device=device
                    )
                    total_loss_tensor += loss_epinet
                else:
                    target = torch.tensor([plan[1] for plan in plans], device=device).squeeze()
                    total_loss_tensor += loss(estimated_cost.squeeze(), target)

                # # Apply stop gradient operator to last feature to serve as input to epinet
                # last_feature = last_feature.detach()
                #
                # n_plans = estimated_cost.shape[0]
                # epinet_estimated_cost = torch.zeros((n_plans*n_epi_indexes, 1), device=device)
                # for j in range(n_epi_indexes):
                #     epinet_index = epinet_cost_estimation.sample_epistemic_indexes()
                #     ensemble_prior = epinet_cost_estimation.compute_ensemble_prior(
                #         plans, embedded_prior, precomputed_indexes, precomputed_masks, epinet_index, i
                #     )
                #     ensemble_prior = ensemble_prior.view(-1,1)
                #     mlp_prior = epinet_cost_estimation.compute_mlp_prior(last_feature, epinet_index)
                #     learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp(last_feature, epinet_index)
                #     epinet_output = estimated_cost + (learnable_mlp_prior + mlp_prior + ensemble_prior)
                #
                #     start_idx = j * n_plans
                #     end_idx = (j + 1) * n_plans
                #
                #     epinet_estimated_cost[start_idx:end_idx] = epinet_output.view(n_plans, 1)
                #
                # raw_targets = torch.tensor([plan[1] for plan in plans], device=device)
                # target = raw_targets.repeat(n_epi_indexes)
                #
                # query_loss = loss(epinet_estimated_cost.squeeze(), target)
                # query_loss_epoch.append(query_loss.detach().cpu().item())
                # total_loss_tensor += query_loss

            total_loss_tensor.backward()

            optimizer.step()

        query_to_val_cost = validate(queries_val, query_plans_val,
                                 precomputed_indexes, precomputed_masks,
                                 min_val, max_val,
                                 loss,
                                 epinet_cost_estimation,
                                 device,
                                 train_epi_network, n_epi_indexes)
        val_losses = [val_output["loss"] for val_output in query_to_val_cost.values()]
        val_mape = [val_output["mape"] for val_output in query_to_val_cost.values()]
        print(f"Epoch {epoch + 1} finished ({sum(query_loss_epoch)/len(query_loss_epoch)})")
        print(f"Validation loss on cost: {sum(val_losses)/len(val_losses)}, mape: {sum(val_mape)/len(val_mape)}")
        if train_epi_network:
            mean_metrics = summarize_epistemic_metrics(query_to_val_cost)


def calculate_loss_epinet(epinet_cost_estimation,
                          loss,
                          estimated_cost, last_feature, embedded_prior,
                          plans, precomputed_indexes, precomputed_masks, i,
                          n_epi_indexes, device):
    # Apply stop gradient operator to last feature to serve as input to epinet
    last_feature = last_feature.detach()
    # Training the epinet is done on frozen model

    n_plans = estimated_cost.shape[0]
    epinet_estimated_cost = torch.zeros((n_plans*n_epi_indexes, 1), device=device)
    #TODO GNN PRIOR ONLY DEPENDS ON X, NOT THE EPI INDEX, SO CAN BE LIFTED OUTSIDE THE LOOP
    # Move weighted sum outside of compute_ensemble_prior and then move compute_ensemble_prior outside of the
    # epi indexes loop.
    for j in range(n_epi_indexes):
        epinet_index = epinet_cost_estimation.sample_epistemic_indexes()
        ensemble_prior = epinet_cost_estimation.compute_ensemble_prior(
            plans, embedded_prior, precomputed_indexes, precomputed_masks, epinet_index, i
        )
        ensemble_prior = ensemble_prior.view(-1,1)
        mlp_prior = epinet_cost_estimation.compute_mlp_prior(last_feature, epinet_index)
        learnable_mlp_prior = epinet_cost_estimation.compute_learnable_mlp(last_feature, epinet_index)
        epinet_output = estimated_cost + (learnable_mlp_prior + mlp_prior + ensemble_prior)

        start_idx = j * n_plans
        end_idx = (j + 1) * n_plans

        epinet_estimated_cost[start_idx:end_idx] = epinet_output.view(n_plans, 1)

    raw_targets = torch.tensor([plan[1] for plan in plans], device=device)
    target = raw_targets.repeat(n_epi_indexes)

    query_loss = loss(epinet_estimated_cost.squeeze(), target)
    return query_loss, target, epinet_estimated_cost


def main_simulated_training(train_dataset, val_dataset,
                            oracle_model,
                            epinet_cost_estimation,
                            device, query_batch_size,
                            save_loc_simulated_dataset, save_loc_simulated_dataset_val):
    oracle_model = oracle_model.to(device)
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, save_loc_simulated_dataset)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, save_loc_simulated_dataset_val)
    query_plan_dict_val = {k: v for d in val_data for k, v in d.items()}
    train_simulated_cost_model(queries_train=train_dataset, query_plans_train=query_plans_dict,
                                queries_val=val_dataset, query_plans_val=query_plan_dict_val,
                                epinet_cost_estimation=epinet_cost_estimation,
                                device=device,
                                query_batch_size=query_batch_size,
                                n_epi_indexes=2,
                                train_epi_network=False,
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
                            epinet_cost_estimation,
                            device,
                            8,
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