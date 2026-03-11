import os
import sys
from abc import abstractmethod, ABC
from torch_geometric.nn.aggr import AttentionalAggregation

from src.models.gine_conv_model import GINEConvModel

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_layers.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation
import torch
import torch.nn as nn


class BasePlanCostEstimator(nn.Module, ABC):
    def __init__(self, heads_config, device, feature_dim=100, mlp_output_dim=64):
        """

        :param heads_config: Configuration of estimation heads: {name, layer: nn.Linear(...)}
        :param device:
        :param feature_dim: The dimension of the embedding plans
        :param mlp_output_dim: The dimension after applying an MLP to the embedded plans
        """
        super().__init__()
        self.device = device

        self.heads_config = heads_config
        self.mlp_output_dim = mlp_output_dim

        self.plan_embedding_nn, self.attn_pool, self.mlp, self.heads = (
            self.init_model(feature_dim, device)
        )

    def forward(self, trees, indexes, mask_padding):
        # Encode the query plan
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

        valid = ~mask_padding.reshape(-1)
        emb_masked_padding = emb_stacked[valid]
        batch_masked_padding = batch[valid]

        pool_vectors = self.attn_pool(emb_masked_padding, batch_masked_padding)

        # Get root representation.
        # First element is the zero vector (that is padded out in attention), so we take index 1
        root_vectors = emb_transposed[:, 1, :]

        # Combine root and pooled representations
        combined = torch.cat([root_vectors, pool_vectors], dim=1)

        # Apply full MLP to plan representation
        mlp_out = self.mlp(combined)

        # Iterate through the dictionary of heads and collect predictions
        predictions = {
            head_name: head(mlp_out) for head_name, head in self.heads.items()
        }

        # Return a dictionary of predictions instead of a single tensor
        return predictions, mlp_out

    def serialize_model(self, model_dir, model_file_name: str = None):
        state_dict = self.state_dict()
        if model_file_name:
            torch.save(state_dict, os.path.join(model_dir, model_file_name))
        else:
            torch.save(state_dict, os.path.join(model_dir, "cost_estimator.pt"))

    @abstractmethod
    def init_model(self, feature_dim, device):
        """
        Must return: (plan_embedding_nn, attn_pool, regressor, estimation_heads)
        """
        pass

# class BasePlanCostEstimator(nn.Module, ABC):
#     def __init__(self, device, feature_dim=100):
#         super().__init__()
#         self.device = device
#         self.plan_embedding_nn, self.attn_pool, self.regressor = self.init_model(feature_dim, device)
#
#     def forward(self, trees, indexes, mask_padding):
#         # Encode the query plan
#         emb, idx = self.plan_embedding_nn((trees, indexes))
#
#         # Consider hierarchical application of our trees. What if we take the output representation of the
#         # query as input to the next embedding / plan phase like in TinyHierarchicalReason modeling?
#
#         # We reshape the (n_plans, dim, n_nodes) tensor to (n_plans, n_nodes, dim)
#         emb_transposed = emb.transpose(1, 2)
#
#         # Stack the node embeddings to a tensor (n_plans * n_nodes, dim) to use with batch variable
#         emb_stacked = emb_transposed.reshape((-1, emb_transposed.shape[-1]))
#
#         # Create batch vector to represent the fact that we merged plans
#         n_plans = emb_transposed.shape[0]
#         n_nodes = emb_transposed.shape[1]
#         plan_indices = torch.arange(n_plans, device=self.device)
#
#         # Repeat each plan index n_nodes times
#         batch = plan_indices.repeat_interleave(n_nodes)
#
#         valid = ~mask_padding.reshape(-1)
#         emb_masked_padding = emb_stacked[valid]
#         batch_masked_padding = batch[valid]
#
#         pool_vectors = self.attn_pool(emb_masked_padding, batch_masked_padding)
#
#         # Get root representation.
#         # First element is the zero vector (that is padded out in attention), so we take index 1
#         root_vectors = emb_transposed[:, 1, :]
#
#         #TODO: Consider if we should also add a pooled version of the graph. Reasoning: Separate the graph structure and
#         # the plan structure. Easy to figure out, just make a small experiment
#         combined = torch.cat([root_vectors, pool_vectors], dim=1)
#         return self.regressor(combined), combined
#
#     def serialize_model(self, model_dir, model_file_name: str=None):
#         state_dict = self.state_dict()
#         if model_file_name:
#             torch.save(state_dict, os.path.join(model_dir, model_file_name))
#         else:
#             torch.save(state_dict, os.path.join(model_dir, "cost_estimator.pt"))


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
        mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.mlp_output_dim),
            nn.ReLU(),
        ).to(device)

        heads = nn.ModuleDict()
        for head_name, config in self.heads_config.items():
            # You can easily customize the architecture per head type here
            heads[head_name] = config['layer'].to(device)

        return plan_embedding_nn, attn_pool, mlp, heads

    # # Alternative heavily regularized plan prediction model
    # def init_model(self, feature_dim, device):
    #     gate_nn = nn.Sequential(
    #         nn.Linear(feature_dim, feature_dim // 2),
    #         nn.LayerNorm(feature_dim // 2),  # Added normalization
    #         nn.ReLU(),
    #         nn.Linear(feature_dim // 2, 1)
    #     ).to(device)
    #
    #     plan_embedding_nn = nn.Sequential(
    #         BinaryTreeConv(200, 200),
    #         TreeLayerNorm(),
    #         TreeActivation(nn.ReLU()),
    #         BinaryTreeConv(200, 100),
    #         TreeLayerNorm(),
    #         TreeActivation(nn.ReLU()),
    #         BinaryTreeConv(100, feature_dim),
    #         TreeLayerNorm(),  # Added to normalize final encoder output before pooling
    #     ).to(device)
    #
    #     attn_pool = AttentionalAggregation(gate_nn=gate_nn, nn=None).to(device)
    #
    #     # Input size is double feature_dim because we concat [Root, Attention_Pool]
    #     mlp = nn.Sequential(
    #         nn.Linear(feature_dim * 2, 128),
    #         nn.LayerNorm(128),  # Added normalization
    #         nn.ReLU(),
    #         nn.Linear(128, self.mlp_output_dim),
    #         nn.LayerNorm(self.mlp_output_dim),  # Added normalization
    #         nn.ReLU(),  # Note: Ensure downstream heads are purely linear
    #     ).to(device)
    #
    #     heads = nn.ModuleDict()
    #     for head_name, config in self.heads_config.items():
    #         heads[head_name] = config['layer'].to(device)
    #
    #     return plan_embedding_nn, attn_pool, mlp, heads


class PlanCostEstimatorSmall(BasePlanCostEstimator):
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
        attn_pool = AttentionalAggregation(gate_nn=gate_nn, nn=None).to(device)

        # Input size is double feature_dim because we concat [Root, Attention_Pool]
        mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, self.mlp_output_dim),
            nn.ReLU(),
            # nn.Linear(5, 1),
        ).to(device)

        heads = nn.ModuleDict()
        for head_name, config in self.heads_config.items():
            # You can easily customize the architecture per head type here
            heads[head_name] = config['layer'].to(device)

        return plan_embedding_nn, attn_pool, mlp, heads


class PlanCostEstimatorTiny(BasePlanCostEstimator):
    def init_model(self, feature_dim, device):
        gate_nn = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        ).to(device)
        plan_embedding_nn = nn.Sequential(
            BinaryTreeConv(10, 10),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(10, 5),
            TreeLayerNorm(),
            TreeActivation(nn.ReLU()),
            BinaryTreeConv(5, feature_dim),
        ).to(device)

        attn_pool = AttentionalAggregation(gate_nn=gate_nn, nn=None).to(device)

        # Input size is double feature_dim because we concat [Root, Attention_Pool]
        mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, self.mlp_output_dim),
            nn.ReLU(),
            # nn.Linear(8, 1),
        ).to(device)

        heads = nn.ModuleDict()
        for head_name, config in self.heads_config.items():
            # You can easily customize the architecture per head type here
            heads[head_name] = config['layer'].to(device)

        return plan_embedding_nn, attn_pool, mlp, heads


class QueryPlansPredictionModel(nn.Module):
    def __init__(self, query_emb_model: GINEConvModel, query_plan_model: BasePlanCostEstimator, device):
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

    def estimate_cost(self, prepared_trees, prepared_indexes, prepared_masks, cost_name='plan_cost'):
        heads_output, feature = self.query_plan_model.forward(prepared_trees, prepared_indexes, prepared_masks)
        return heads_output[cost_name], feature

    def estimate_all_heads(self, prepared_trees, prepared_indexes, prepared_masks):
        return self.query_plan_model.forward(prepared_trees, prepared_indexes,  prepared_masks)

    def serialize_model(self, model_dir):
        self.query_emb_model.serialize_model(model_dir)
        self.query_plan_model.serialize_model(model_dir)
