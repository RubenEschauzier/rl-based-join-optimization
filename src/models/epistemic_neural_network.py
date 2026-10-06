import copy
import os
import re
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
from src.models.model_layers.batched_ensemble_plan_cost import BatchedEnsemblePlanCost
from src.models.query_plan_prediction_model import QueryPlansPredictionModel, PlanCostEstimatorTiny, \
    PlanCostEstimatorFull

import torch
import torch.nn as nn

from src.models.mlp_epinet import MLPEpinetMixin, glorot_init

def instantiate_heads_config(heads_config):
    """Return a heads config holding a *fresh* module per head.

    ``BasePlanCostEstimator.init_model`` registers ``config['layer']`` directly rather
    than a copy. Passing one config dict to several models therefore gives them a single
    shared output head. For the randomized-prior ensemble that is fatal: every prior
    collapses to a common projection of its own body, which destroys the ensemble
    diversity the epistemic index is supposed to resolve.

    ``layer`` may be either an ``nn.Module`` used as a shape template (deep-copied here)
    or a zero-argument callable invoked once per model. Weights carried by a template are
    not meaningful: callers re-initialize the ensemble afterwards (see the
    ``apply(init_weights)`` call in ``MultiHeadEpistemicNetwork.__init__``), which is what
    makes the copies differ from one another.
    """
    instantiated = {}
    for head_name, config in heads_config.items():
        layer = config['layer']
        instantiated[head_name] = {
            **config,
            'layer': layer() if callable(layer) and not isinstance(layer, nn.Module)
            else copy.deepcopy(layer),
        }
    return instantiated


class MultiHeadEpistemicNetwork(MLPEpinetMixin, nn.Module):
    def __init__(self,
                 epi_index_dim, prior_config,
                 cost_estimation_model: QueryPlansPredictionModel,
                 ensemble_prior_heads_config=None,
                 head_names=None,
                 mlp_dimension=5,
                 epinet_hidden_dim=None,
                 prior_epinet_hidden_dim=None,
                 device=torch.device('cpu'),
                 verbose = 0):
        super().__init__()
        self.epi_index_dim = epi_index_dim
        self.cost_estimation_model = cost_estimation_model
        self.device = device
        self.prior_device = device

        self.prior_config = prior_config
        self.mlp_output_dim_cost_model = cost_estimation_model.query_plan_model.mlp_output_dim
        # Width of sg[phi(x)] as actually returned by the cost model, which may be wider
        # than the MLP output when epinet_feature_mode == "mlp_plus_plan".
        self.epinet_feature_dim = cost_estimation_model.query_plan_model.epinet_feature_dim
        # Kept tied to the MLP width so the epinet's capacity does not silently grow with
        # the feature width; deepmind/enn uses a single hidden layer of 50 units here.
        self.epinet_hidden_dim = epinet_hidden_dim or max(self.mlp_output_dim_cost_model // 2, 1)
        # The learnable vs fixed prior gets its own width.
        self.prior_epinet_hidden_dim = prior_epinet_hidden_dim or self.epinet_hidden_dim

        # Glorot initialization as described in Epistemic Neural Networks paper
        init_weights = glorot_init

        # We use an ensemble of tiny gine_conv as priors
        model_factory_gine_conv = ModelFactory(prior_config)

        if not ensemble_prior_heads_config:
            ensemble_prior_heads_config = {
                'plan_cost': {
                    'layer': torch.nn.Linear(mlp_dimension, 1),
                }
            }
        self.head_names = list(ensemble_prior_heads_config.keys())

        ensemble_gnn = [model_factory_gine_conv.load_gine_conv().to(device) for _ in range(epi_index_dim)]
        for gnn in ensemble_gnn:
            gnn.freeze_model()

        ensemble_plan_cost = [PlanCostEstimatorTiny(
            instantiate_heads_config(ensemble_prior_heads_config), device,
            mlp_output_dim=mlp_dimension
        ).to(device) for _ in range(epi_index_dim)]

        for plan_cost in ensemble_plan_cost:
            plan_cost.eval()

        if verbose > 0:
            total_params_prior = inspect_ensemble_params(ensemble_gnn[0], ensemble_plan_cost[0])
            print(f"Total parameters in ensemble gnn prior: {total_params_prior*epi_index_dim}")

        self.ensemble_combined_prior_models = nn.ModuleList([
            QueryPlansPredictionModel(ensemble_gnn[i], ensemble_plan_cost[i], device)
            for i in range(epi_index_dim)
        ])
        self.ensemble_combined_prior_models.apply(init_weights)

        for combined_prior in self.ensemble_combined_prior_models:
            for param in combined_prior.parameters():
                param.requires_grad = False

        # Evaluating the ensemble as a Python loop over members is ~56% of all epinet time
        # and dominates planning latency; the members are tiny, so it is kernel-launch
        # bound rather than compute bound. This snapshot runs them as grouped ops instead,
        # to within float32 noise. Priors are frozen, so a snapshot stays valid -- except
        # after load_state_dict, which calls refresh_batched_ensemble().
        self.refresh_batched_ensemble()

        # Learnable epinet + frozen MLP prior (src/models/mlp_epinet.py), created here, after
        # the ensemble, so seeded constructions draw the same random numbers as before.
        self._build_mlp_epinet(self.epinet_feature_dim, epi_index_dim, self.head_names,
                               self.epinet_hidden_dim, self.prior_epinet_hidden_dim, device)

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

    def _move_to_device(self, trees, indexes, masks):
        return (
            trees.to(self.device),
            indexes.to(self.device),
            masks.to(self.device)
        )

    def estimate_cost_from_prepared(self, prepared_trees, prepared_indexes, prepared_masks):
        """Forward pass for a specific head."""
        trees, indexes, masks = self._move_to_device(prepared_trees, prepared_indexes, prepared_masks)

        return self.cost_estimation_model.estimate_cost(
            trees, indexes, masks
        )

    def estimate_cost_full(self, plans, embedded_query, precomputed_indexes, precomputed_masks):
        prepared_trees, prepared_indexes, prepared_masks = self.prepare_cost_estimation_inputs(
            plans, embedded_query, precomputed_indexes, precomputed_masks, target_device=self.device
        )
        return self.estimate_cost_from_prepared(prepared_trees, prepared_indexes, prepared_masks)

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

    def refresh_batched_ensemble(self):
        """(Re)snapshot the frozen ensemble weights into the grouped-op evaluator.

        Must be called after anything that changes the prior members' weights, i.e. after
        loading a checkpoint. The snapshot buffers are non-persistent, so they never enter
        a state_dict and cannot go stale on disk.
        """
        members = [model.query_plan_model for model in self.ensemble_combined_prior_models]
        self.batched_ensemble_prior = BatchedEnsemblePlanCost.from_members(
            members, head_name=self.head_names[0]
        ).to(self.device)

    def compute_ensemble_prior_from_prepared(self, prepared_trees_list, prepared_indexes, prepared_masks,
                                             use_batched=True):
        """Forward pass for the frozen priors, returning {head: (epi_index_dim, n_plans)}."""
        with torch.no_grad():
            if use_batched and len(self.head_names) == 1:
                stacked = torch.cat(prepared_trees_list, dim=1)
                values = self.batched_ensemble_prior(stacked, prepared_indexes, prepared_masks)
                return {self.head_names[0]: values.to(self.device)}

            num_plans = prepared_trees_list[0].shape[0]
            num_ensembles = self.epi_index_dim

            estimated_cost_priors = { key: torch.zeros((num_ensembles, num_plans), device=self.device)
                                     for key in self.head_names }

            for i in range(num_ensembles):
                est_cost, _ = self.ensemble_combined_prior_models[i].estimate_cost(
                    prepared_trees_list[i], prepared_indexes, prepared_masks
                )
                for key, output in est_cost.items():
                    output_t = output.transpose(0, 1)
                    estimated_cost_priors[key][i] = output_t.to(self.device)

        return estimated_cost_priors

    def compute_ensemble_prior(self, plans, embedded_query, precomputed_indexes, precomputed_masks, query_idx):
        prepared_trees_list, prepared_indexes, prepared_masks = self.prepare_ensemble_prior_inputs(
            plans, embedded_query, precomputed_indexes, precomputed_masks, query_idx
        )
        return self.compute_ensemble_prior_from_prepared(
            prepared_trees_list, prepared_indexes, prepared_masks
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

    # compute_mlp_prior[_batched], compute_learnable_mlp[_batched],
    # sample_epistemic_indexes[_batched] and get_learnable_epinet_params: see MLPEpinetMixin.

    def get_query_embedding_model_params(self):
        """Returns parameters for the base GNN query embedding model."""
        return self.cost_estimation_model.query_emb_model.parameters()

    def get_plan_cost_estimation_model_params(self):
        return self.cost_estimation_model.query_plan_model.parameters()

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

    def load_epinet(self, path, load_only_cost_model=False, strict=True, diff_filter=None):
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
                missing, unexpected = self.cost_estimation_model.load_state_dict(cost_model_state, strict=strict)

            else:
                missing, unexpected = self.cost_estimation_model.load_state_dict(checkpoint, strict=strict)

        else:
            # Scenario: Load the full Epinet
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                missing, unexpected = self.load_state_dict(checkpoint['state_dict'], strict=strict)
            else:
                raise ValueError("Checkpoint does not contain 'state_dict'. It might be a raw cost model file.")
            # Prior weights may have changed; the grouped snapshot must follow them.
            self.refresh_batched_ensemble()
        if not strict:
            if diff_filter:
                pattern = re.compile(diff_filter)
                # Filter out keys that match the regex pattern
                missing = [k for k in missing if not pattern.search(k)]
                unexpected = [k for k in unexpected if not pattern.search(k)]
            if len(unexpected) > 0:
                print(f"Unexpected keys: ${unexpected}")
            if len(missing) > 0:
                print(f"Missing keys: ${missing}")


def prepare_epinet_model(full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension,
                         heads_config, heads_config_prior,
                         device,
                         model_weights=None, cost_only=False, strict=True,
                         freeze_embedding=True, diff_filter=None,
                         epinet_feature_mode="mlp", epinet_hidden_dim=None,
                         prior_epinet_hidden_dim=None):
    model_factory_gine_conv = ModelFactory(full_gnn_config)
    embedding_model_full = model_factory_gine_conv.load_gine_conv()

    if freeze_embedding:
        # Training on frozen backbone (we still train the plan estimator gnns)
        embedding_model_full.freeze_model()

    cost_net_full = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=mlp_dimension,
        epinet_feature_mode=epinet_feature_mode
    )
    combined_model_full = QueryPlansPredictionModel(embedding_model_full, cost_net_full, device)
    epinet_cost_estimation = MultiHeadEpistemicNetwork(epinet_index_dim, config_ensemble_prior, combined_model_full,
                                                       ensemble_prior_heads_config=heads_config_prior,
                                                       epinet_hidden_dim=epinet_hidden_dim,
                                                       prior_epinet_hidden_dim=prior_epinet_hidden_dim,
                                                       device=device)
    epinet_cost_estimation.to(device)
    if model_weights:
        epinet_cost_estimation.load_epinet(model_weights,
                                           load_only_cost_model=cost_only,
                                           strict=strict,
                                           diff_filter=diff_filter)
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