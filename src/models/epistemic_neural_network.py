import concurrent
import os
import sys
import time

from src.utils.epinet_utils.epinet_model_utils import inspect_ensemble_params

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_instantiator import ModelFactory
from src.utils.tree_conv_utils import get_shared_structure, apply_features_to_structure, \
    apply_features_to_structure_batched
from src.models.query_plan_prediction_model import QueryPlansPredictionModel, PlanCostEstimatorSmall, \
    PlanCostEstimatorTiny

import torch
import torch.nn as nn

class EpistemicNetwork(nn.Module):
    def __init__(self,
                 epi_index_dim, prior_config,
                 cost_estimation_model: QueryPlansPredictionModel,
                 ensemble_prior_heads_config=None,
                 mlp_dimension=5,
                 device=torch.device('cpu'),
                 prior_device=torch.device('cpu')):
        super().__init__()
        self.epi_index_dim = epi_index_dim
        self.cost_estimation_model = cost_estimation_model
        self.device = device
        self.prior_device = device

        self.prior_config = prior_config
        self.mlp_output_dim_cost_model = cost_estimation_model.query_plan_model.mlp_output_dim

        # Glorot initialization as described in Epistemic Neural Networks paper
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # We use an ensemble of tiny gine_conv as priors
        model_factory_gine_conv = ModelFactory(prior_config)

        if not ensemble_prior_heads_config:
            ensemble_prior_heads_config = {
                'plan_cost': {
                    'layer': torch.nn.Linear(mlp_dimension, 1),
                }
            }

        ensemble_gnn = [model_factory_gine_conv.load_gine_conv().to(device) for _ in range(epi_index_dim)]
        for gnn in ensemble_gnn:
            gnn.freeze_model()

        ensemble_plan_cost = [PlanCostEstimatorTiny(
            ensemble_prior_heads_config, device, mlp_output_dim=mlp_dimension
        ).to(device) for _ in range(epi_index_dim)]

        for plan_cost in ensemble_plan_cost:
            plan_cost.eval()

        total_params_prior = inspect_ensemble_params(ensemble_gnn[0], ensemble_plan_cost[0])
        print(f"Total parameters in ensemble gnn prior: {total_params_prior*epi_index_dim}")

        self.learnable_epinet = nn.Sequential()
        self.ensemble_combined_prior_models = nn.ModuleList([
            QueryPlansPredictionModel(ensemble_gnn[i], ensemble_plan_cost[i], device)
            for i in range(epi_index_dim)
        ])
        self.ensemble_combined_prior_models.apply(init_weights)

        # The learnable and fixed MLPs should have same config all three parts should have same output dimension
        self.learnable_epinet = nn.Sequential(
            nn.Linear(self.mlp_output_dim_cost_model+epi_index_dim, self.mlp_output_dim_cost_model//2),
            nn.ReLU(),
        ).to(device)

        # Initialize the last layer to always output zero from learnable.
        last_layer = nn.Linear(self.mlp_output_dim_cost_model//2, epi_index_dim)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        self.learnable_epinet.apply(init_weights)
        self.learnable_epinet.add_module("last_layer", last_layer)

        # Prior MLP should just be Glorot initialized
        self.prior_epinet = nn.Sequential(
            nn.Linear(self.mlp_output_dim_cost_model+epi_index_dim, self.mlp_output_dim_cost_model//2),
            nn.ReLU(),
            nn.Linear(self.mlp_output_dim_cost_model//2, epi_index_dim),
        ).to(device)

        # self.prior_epinet = nn.Sequential(
        #     nn.Linear(200 + epi_index_dim, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 1)
        # ).to(device)
        self.prior_epinet.apply(init_weights)

        for param in self.prior_epinet.parameters():
            param.requires_grad = False

        for combined_prior in self.ensemble_combined_prior_models:
            for param in combined_prior.parameters():
                param.requires_grad = False

    def embed_query_batched(self, queries):
        return self.cost_estimation_model.embed_query_batched(queries)

    def embed_query_batched_prior(self, queries):
        embedded_query_batches = []
        for prior_model in self.ensemble_combined_prior_models:
            embedded_query_batches.append(prior_model.embed_query_batched(queries))
        return embedded_query_batches

    def prepare_cost_estimation_inputs(self, plans, embedded_query, precomputed_indexes, precomputed_masks,
                                       target_device=None):
        """Builds tree structures and indices. Meant to be run on CPU workers."""
        target_device = target_device or embedded_query.device
        join_orders = [plan[0] for plan in plans]
        n_nodes = embedded_query.shape[0]

        gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
            join_orders, n_nodes, precomputed_indexes, precomputed_masks, target_device
        )

        prepared_trees = apply_features_to_structure(embedded_query, gather_indices)
        return prepared_trees, prepared_indexes, prepared_masks

    def estimate_cost_from_prepared(self, prepared_trees, prepared_indexes, prepared_masks, cost_name='plan_cost'):
        """Pure PyTorch forward pass. Meant to be run on the GPU server."""
        # Ensure tensors are on the main device (GPU)
        prepared_trees = prepared_trees.to(self.device)
        prepared_indexes = prepared_indexes.to(self.device)
        prepared_masks = prepared_masks.to(self.device)

        return self.cost_estimation_model.estimate_cost(
            prepared_trees, prepared_indexes, prepared_masks,
            cost_name=cost_name
        )

    def estimate_cost_full(self, plans, embedded_query, precomputed_indexes, precomputed_masks, cost_name='plan_cost'):
        join_orders = [plan[0] for plan in plans]
        n_nodes = embedded_query.shape[0]

        gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
            join_orders, n_nodes, precomputed_indexes,
            precomputed_masks, self.device
        )

        prepared_trees = apply_features_to_structure(embedded_query, gather_indices)
        return self.cost_estimation_model.estimate_cost(
            prepared_trees, prepared_indexes, prepared_masks,
            cost_name=cost_name
        )

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
        mlp_output = self.prior_epinet(concat_input)
        epinet_output = mlp_output @ epi_index.T
        return epinet_output

    def compute_learnable_mlp(self, last_feature, epi_index):
        concat_input = torch.cat([last_feature, epi_index.expand(last_feature.shape[0], -1)], dim=1)
        mlp_output = self.learnable_epinet(concat_input)
        epinet_output = mlp_output @ epi_index.T
        return epinet_output

    # TODO: Validate this generated code.
    def compute_mlp_prior_batched(self, last_feature, epi_indexes):
        n = last_feature.shape[0]
        k = epi_indexes.shape[0]

        # 1. Repeat features K times: [N, F] -> [K*N, F]
        # Results in: [feat_0, ..., feat_N, feat_0, ..., feat_N, ...]
        last_feature_exp = last_feature.repeat(k, 1)

        # 2. Interleave indexes N times: [K, D] -> [K*N, D]
        # Results in: [z_0, ..., z_0, z_1, ..., z_1, ...]
        epi_indexes_exp = epi_indexes.repeat_interleave(n, dim=0)

        # 3. Concatenate and pass through MLP
        concat_input = torch.cat([last_feature_exp, epi_indexes_exp], dim=1)
        mlp_output = self.prior_epinet(concat_input)

        # 4. Batched dot product: element-wise multiply then sum along feature dimension
        epinet_output = (mlp_output * epi_indexes_exp).sum(dim=1, keepdim=True)
        return epinet_output

    def compute_learnable_mlp_batched(self, last_feature, epi_indexes):
        n = last_feature.shape[0]
        k = epi_indexes.shape[0]

        last_feature_exp = last_feature.repeat(k, 1)
        epi_indexes_exp = epi_indexes.repeat_interleave(n, dim=0)

        concat_input = torch.cat([last_feature_exp, epi_indexes_exp], dim=1)
        mlp_output = self.learnable_epinet(concat_input)

        epinet_output = (mlp_output * epi_indexes_exp).sum(dim=1, keepdim=True)
        return epinet_output

    def compute_ensemble_prior(self, plans, embedded_query,
                               precomputed_indexes, precomputed_masks,
                               query_idx, cost_name='plan_cost'):
        with torch.no_grad():
            num_plans = len(plans)
            num_ensembles = self.epi_index_dim
            join_orders = [plan[0] for plan in plans]

            # Determine query size from the first ensemble's embedding
            n_nodes = embedded_query[0][query_idx].shape[0]

            # Each query has same gather_indices, indexes and masks. So precompute
            gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
                join_orders, n_nodes, precomputed_indexes,
                precomputed_masks, self.device
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
                    prepared_trees, prepared_indexes, prepared_masks, cost_name=cost_name
                )
                # Est_cost: (1, n_plans)
                est_cost_t = est_cost.transpose(0,1)
                estimated_cost_priors[i] = est_cost_t

        return estimated_cost_priors

    def sample_epistemic_indexes(self):
        return torch.normal(0, 1, size=(1, self.epi_index_dim), device=self.device)

    def sample_epistemic_indexes_batched(self, n_epi_indexes):
        return torch.randn((n_epi_indexes, self.epi_index_dim), device=self.device)


    def serialize_model(self, model_dir, save_only_cost_model=False):
        """
        Saves either the full Epistemic Network or just the base cost model.
        """
        if save_only_cost_model:
            # Save only the 'big' cost estimation model (raw state dict)
            torch.save(
                self.cost_estimation_model.state_dict(),
                os.path.join(model_dir, "cost_estimation_model.pt")
            )
        else:
            # Save the full Epinet checkpoint (including priors and learnable MLPs)
            checkpoint = {
                'epi_index_dim': self.epi_index_dim,
                'prior_config': self.prior_config,
                'state_dict': self.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_dir, "epinet_model.pt"))

    def load_epinet(self, path, load_only_cost_model=False):
        """
        Loads the model weights. Can selectively load just the cost model
        even from a full Epinet checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if load_only_cost_model:
            # Scenario: We only want to load weights into self.cost_estimation_model

            # Case A: The file is a full Epinet checkpoint (has 'state_dict' key)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                full_state = checkpoint['state_dict']
                prefix = "cost_estimation_model."

                # Extract only keys belonging to the cost model and remove the prefix
                cost_model_state = {
                    k[len(prefix):]: v
                    for k, v in full_state.items()
                    if k.startswith(prefix)
                }
                self.cost_estimation_model.load_state_dict(cost_model_state, strict=True)

            # Case B: The file is a raw state dict for the cost model
            else:
                self.cost_estimation_model.load_state_dict(checkpoint, strict=True)

        else:
            # Scenario: Load the full Epinet
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                raise ValueError("Checkpoint does not contain 'state_dict'. It might be a raw cost model file.")

