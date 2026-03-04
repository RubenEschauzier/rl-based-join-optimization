import os
import sys

from torch_geometric.loader import DataLoader

from src.utils.epinet_utils.epinet_model_utils import inspect_ensemble_params
from src.utils.training_utils.query_loading_utils import prepare_data

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_instantiator import ModelFactory
from src.utils.tree_conv_utils import get_shared_structure, apply_features_to_structure
from src.models.query_plan_prediction_model import QueryPlansPredictionModel, PlanCostEstimatorTiny, \
    PlanCostEstimatorFull

import torch
import torch.nn as nn

class EpistemicNetwork(nn.Module):
    def __init__(self,
                 epi_index_dim, prior_config,
                 cost_estimation_model: QueryPlansPredictionModel,
                 ensemble_prior_heads_config=None,
                 mlp_dimension=5,
                 device=torch.device('cpu'),
                 prior_device=torch.device('cpu'),
                 verbose = 0):
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

        if verbose > 0:
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

    @staticmethod
    def prepare_cost_estimation_inputs(plans, embedded_query, precomputed_indexes, precomputed_masks,
                                       target_device=None):
        """Builds tree structures and indices. Meant to be run on CPU with possible parallelization"""
        target_device = target_device or embedded_query.device
        join_orders = [plan[0] for plan in plans]
        n_nodes = embedded_query.shape[0]

        gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
            join_orders, n_nodes, precomputed_indexes, precomputed_masks, target_device
        )

        prepared_trees = apply_features_to_structure(embedded_query, gather_indices)
        return prepared_trees, prepared_indexes, prepared_masks

    def estimate_cost_from_prepared(self, prepared_trees, prepared_indexes, prepared_masks, cost_name='plan_cost'):
        """Pure PyTorch forward pass. Meant to be run on the GPU"""
        # Ensure tensors are on the main device (GPU)
        prepared_trees = prepared_trees.to(self.device)
        prepared_indexes = prepared_indexes.to(self.device)
        prepared_masks = prepared_masks.to(self.device)

        return self.cost_estimation_model.estimate_cost(
            prepared_trees, prepared_indexes, prepared_masks,
            cost_name=cost_name
        )

    def estimate_cost_full(self, plans, embedded_query, precomputed_indexes, precomputed_masks, cost_name='plan_cost'):
        prepared_trees, prepared_indexes, prepared_masks = self.prepare_cost_estimation_inputs(
            plans, embedded_query, precomputed_indexes, precomputed_masks, target_device=self.device
        )
        return self.estimate_cost_from_prepared(prepared_trees, prepared_indexes, prepared_masks, cost_name)

    def prepare_ensemble_prior_inputs(self, plans, embedded_query, precomputed_indexes, precomputed_masks, query_idx):
        """Builds structures for the ensemble priors. Meant to be run on CPU."""
        join_orders = [plan[0] for plan in plans]
        n_nodes = embedded_query[0][query_idx].shape[0]

        gather_indices, prepared_indexes, prepared_masks = get_shared_structure(
            join_orders, n_nodes, precomputed_indexes, precomputed_masks, self.prior_device
        )

        prepared_trees_list = []
        for i in range(self.epi_index_dim):
            current_features = embedded_query[i][query_idx].to(self.prior_device)
            prepared_trees = apply_features_to_structure(current_features, gather_indices)
            prepared_trees_list.append(prepared_trees)

        return prepared_trees_list, prepared_indexes, prepared_masks

    def compute_ensemble_prior_from_prepared(self, prepared_trees_list, prepared_indexes, prepared_masks,
                                             cost_name='plan_cost'):
        """Pure PyTorch forward pass for priors. Runs on CPU, returns to main device."""
        with torch.no_grad():
            num_plans = prepared_trees_list[0].shape[0]
            num_ensembles = self.epi_index_dim

            # The output tensor should live on the main device (GPU) to combine with the MLP outputs later
            estimated_cost_priors = torch.zeros((num_ensembles, num_plans), device=self.device)

            for i in range(num_ensembles):
                est_cost, _ = self.ensemble_combined_prior_models[i].estimate_cost(
                    prepared_trees_list[i], prepared_indexes, prepared_masks, cost_name=cost_name
                )
                est_cost_t = est_cost.transpose(0, 1)
                estimated_cost_priors[i] = est_cost_t.to(self.device)  # Move to main device here

        return estimated_cost_priors

    def compute_ensemble_prior(self, plans, embedded_query, precomputed_indexes, precomputed_masks, query_idx,
                               cost_name='plan_cost'):
        prepared_trees_list, prepared_indexes, prepared_masks = self.prepare_ensemble_prior_inputs(
            plans, embedded_query, precomputed_indexes, precomputed_masks, query_idx
        )
        return self.compute_ensemble_prior_from_prepared(
            prepared_trees_list, prepared_indexes, prepared_masks, cost_name
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

            else:
                self.cost_estimation_model.load_state_dict(checkpoint, strict=True)

        else:
            # Scenario: Load the full Epinet
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                raise ValueError("Checkpoint does not contain 'state_dict'. It might be a raw cost model file.")


def prepare_epinet_model(full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension,
                         heads_config, heads_config_prior,
                         device,
                         model_weights=None, cost_only=False,
                         freeze_embedding=True):
    model_factory_gine_conv = ModelFactory(full_gnn_config)
    embedding_model_full = model_factory_gine_conv.load_gine_conv()

    if freeze_embedding:
        # Training on frozen backbone (we still train the plan estimator gnns)
        embedding_model_full.freeze_model()

    cost_net_full = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=mlp_dimension
    )
    combined_model_full = QueryPlansPredictionModel(embedding_model_full, cost_net_full, device)
    epinet_cost_estimation = EpistemicNetwork(epinet_index_dim, config_ensemble_prior, combined_model_full,
                                              ensemble_prior_heads_config=heads_config_prior, device=device)
    epinet_cost_estimation.to(device)
    if model_weights:
        epinet_cost_estimation.load_epinet(model_weights, load_only_cost_model=cost_only)
        if cost_only:
            print(f"Initialized cost model weights from {model_weights}")
        else:
            print(f"Initialized weights from {model_weights}")
    return epinet_cost_estimation

if __name__ == "__main__":
    queries_loc = "data/generated_queries/star_yago_gnce/dataset_train"
    endpoint_location = "http://localhost:8888"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"
    trained_cost_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-12-02-2026-17-17-13/epoch-25/model/epinet_model.pt"

    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    def find_query(query_loader, query_str):
        for query in loader:
            if query.query[0] == query_str:
                return query
        raise ValueError(f"Query {query_str} not found")

    find_str = "SELECT * WHERE {  ?s <http://example.com/13000080> <http://example.com/6957478> .  ?s <http://example.com/13000080> <http://example.com/7052642> .  ?s <http://example.com/13000080> <http://example.com/11351711> .  ?s <http://example.com/13000089> <http://example.com/1916054> .  ?s <http://example.com/13000080> ?o4 . ?s <http://example.com/13000080> ?o5 . ?s <http://example.com/13000080> ?o6 . ?s ?p7 ?o7 . }"
    query_to_investigate = find_query(loader, find_str)
    query_to_investigate_as_list = query_to_investigate[0]
    plans_to_investigate = [[4,7,5,6,0,2,1,3], [4,7,5,6,2,0,1,3]]

    epinet_cost_estimation_test = prepare_epinet_model(model_config_emb, model_config_prior, 32, 64, 'cpu')
    embedded = epinet_cost_estimation_test.embed_query_batched(query_to_investigate)
    embedded_numpy = embedded[0].detach().numpy()
    test = 5