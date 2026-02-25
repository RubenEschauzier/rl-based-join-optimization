from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from main import find_best_epoch_directory
from src.models.epistemic_neural_network import EpistemicNetwork
from src.models.model_instantiator import ModelFactory
from src.models.query_plan_prediction_model import PlanCostEstimatorFull, QueryPlansPredictionModel
from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.supervised_value_estimation import prepare_data
from src.utils.epinet_utils.disk_cache_frozen_representations import DiskCacheFrozenRepresentations
from src.utils.training_utils.training_tracking import ExperimentWriter
from torch import nn

from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask

class EpinetThompsonSampling:
    def __init__(self, epinet, alpha_mlp, alpha_ensemble, sigma, precomputed_indexes, precomputed_masks, device):
        self.epinet = epinet
        self.z = self.epinet.sample_epistemic_indexes()

        self.alpha_mlp = alpha_mlp
        self.alpha_ensemble = alpha_ensemble
        self.sigma = sigma

        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks
        self.device = device

    def embed_query(self, query, disk_cache):
        if query.query[0] not in disk_cache:
            embedded = self.epinet.embed_query_batched(query)[0]
            embedded_priors = self.epinet.embed_query_batched_prior(query)

            # embedded_priors_unbatched = [embedded_prior[0] for embedded_prior in embedded_priors]

            disk_cache.store_embeddings(query.query[0], (embedded, embedded_priors))

            return embedded, embedded_priors

        return disk_cache.get_embeddings(query.query[0], self.device)


    def step(self, state):
        embedded = state["embedded"]
        embedded_prior = state["embedded_prior"]
        current_plan = state["current_plan"]
        n_tp = state["n_tp_query"]

        possible_next = self._next_plan(current_plan, n_tp)

        #TODO: Use the 3 heads here, also use 3 different epi nets here?
        # OR first start with just an epinet on cost
        estimated_cost, last_feature = self.epinet.estimate_cost_full(
            possible_next, embedded, self.precomputed_indexes, self.precomputed_masks,
            cost_name='query_total_rows'
        )

        unweighted_ensemble_priors = self.epinet.compute_ensemble_prior(
            possible_next, embedded_prior, self.precomputed_indexes, self.precomputed_masks, 0,
            cost_name='query_total_rows'
        )

        # [n_epi_indexes, epi_index_dim] @ [epi_index_dim, n_plans] -> [n_epi_indexes, n_plans]
        ensemble_prior = torch.matmul(self.z, unweighted_ensemble_priors)
        # Flatten to match the target shape: [n_epi_indexes * n_plans, 1]
        ensemble_prior_flat = ensemble_prior.view(-1, 1)

        # Both return shape: [n_epi_indexes * n_plans, 1]
        mlp_prior = self.epinet.compute_mlp_prior(last_feature, self.z)
        learnable_mlp_prior = self.epinet.compute_learnable_mlp(last_feature, self.z)

        epinet_estimated_cost = estimated_cost + (
                learnable_mlp_prior + self.alpha_mlp * mlp_prior + self.alpha_ensemble * ensemble_prior_flat
        )

        # Reshape back to [n_epi_indexes, n_plans] to map back to possible_next easily
        n_epi_indexes = self.z.shape[0]
        epinet_estimated_cost = epinet_estimated_cost.view(n_epi_indexes, len(possible_next))

        # Attach the estimated epinet cost to the next_plans without sorting
        plans_with_cost = []
        for i, plan in enumerate(possible_next):
            # Extract the cost for this specific plan (shape: [n_epi_indexes])
            # If n_epi_indexes == 1, you can append .item() if you just want the float value
            plan_cost = epinet_estimated_cost[:, i].item()
            plans_with_cost.append((plan, plan_cost))

        return plans_with_cost

    def init_episode(self):
        self.z = self.epinet.sample_epistemic_indexes()

    @staticmethod
    def _next_plan(current, n_entries):
        # Create a set of actions already taken for O(1) lookup
        used_actions = set(current)

        # Only append actions not already in the plan
        return [(current + [a], 0) for a in range(n_entries) if a not in used_actions]

#TODO: CHECK CHATGPT CODE
class PlanPerformanceCache:
    def __init__(self, window_size=10, percentile=20):
        """
        Args:
            window_size: Max number of recent executions to remember per subplan.
                         Keeps memory low and allows the agent to "forget" very old
                         hardware states.
            percentile: The robust percentile to use as the target (e.g., 20).
        """
        self.window_size = window_size
        self.percentile = percentile

        # Maps plan_hash -> deque of recent latencies
        self.history = {}

        # Cache the computed percentile to avoid recomputing on every RL batch sample
        self._percentile_cache = {}

    def _get_hash(self, plan_identifier):
        """
        Converts a plan into a hashable key.
        Assuming your subplans can be represented as a list/set of table aliases
        or join nodes (e.g., ['A', 'B', 'C']).
        """
        if isinstance(plan_identifier, (list, set)):
            # frozenset ignores join order for the key, assuming (A join B) == (B join A)
            # If physical join order/type matters, use a tuple representing the exact tree!
            return frozenset(plan_identifier)
        return plan_identifier

    def add_execution(self, plan_identifier, latency: float):
        """Adds a single execution latency to a specific subplan."""
        plan_hash = self._get_hash(plan_identifier)

        if plan_hash not in self.history:
            self.history[plan_hash] = deque(maxlen=self.window_size)

        self.history[plan_hash].append(latency)

        # Invalidate the cached percentile for this plan
        if plan_hash in self._percentile_cache:
            del self._percentile_cache[plan_hash]

    def add_trajectory(self, subplans: list, final_latency: float):
        """
        The RL Data Augmentation step:
        Registers the final query latency to EVERY subplan in the tree.
        """
        for subplan in subplans:
            self.add_execution(subplan, final_latency)

    def get_robust_target(self, plan_identifier) -> float | None :
        """
        Retrieves the 20th percentile latency for the requested plan.
        """
        plan_hash = self._get_hash(plan_identifier)

        # If we have never seen this subplan finish, return a penalty/timeout value
        if plan_hash not in self.history or not self.history[plan_hash]:
            return None

            # Return cached computation if available (O(1) lookup during RL training)
        if plan_hash in self._percentile_cache:
            return self._percentile_cache[plan_hash]

        # Compute the robust percentile
        latencies = list(self.history[plan_hash])
        robust_target = float(np.percentile(latencies, self.percentile))

        # Cache and return
        self._percentile_cache[plan_hash] = robust_target
        return robust_target

    def get_buffer_stats(self):
        return {
            "unique_subplans_tracked": len(self.history),
            "memory_usage_approx_mb": (len(self.history) * self.window_size * 8) / (1024 * 1024)
        }

def prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                       rdf2vec_vector_location, occurrences_location, tp_cardinality_location):
    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    client = QLeverOptimizerClient("http://localhost:8888")
    writer = ExperimentWriter("experiments/experiment_outputs/yago_gnce/online_epinet_training",
                              "actual_cost_epinet_training",
                              {}, {})
    return train_dataset, val_dataset, client, writer


def prepare_model(full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension, device):
    heads_config = {
        'query_total_rows': {
            'layer': nn.Linear(mlp_dimension, 1),
        },
        'join_rows': {
            'layer': nn.Linear(mlp_dimension, 1),
        },
        'latency': {
            'layer': nn.Linear(mlp_dimension, 1),
        }
    }
    heads_config_prior = {
        'query_total_rows': {
            'layer': nn.Linear(5, 1),
        },
        'join_rows': {
            'layer': nn.Linear(5, 1),
        },
        'latency': {
            'layer': nn.Linear(5, 1),
        }
    }

    model_factory_gine_conv = ModelFactory(full_gnn_config)
    embedding_model_full = model_factory_gine_conv.load_gine_conv()
    # Training on frozen backbone (we still train the plan estimator gnns)
    embedding_model_full.freeze_model()

    cost_net_full = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=mlp_dimension
    )
    combined_model_full = QueryPlansPredictionModel(embedding_model_full, cost_net_full, device)
    epinet_cost_estimation = EpistemicNetwork(epinet_index_dim, config_ensemble_prior, combined_model_full,
                                              ensemble_prior_heads_config=heads_config_prior, device=device)
    epinet_cost_estimation.to(device)

    return epinet_cost_estimation


def beam_search(query, embedded, embedded_prior, agent, beam_width, n_triple_patterns):
    current_plans = [[i] for i in range(n_triple_patterns)]
    top_candidates = []
    for i in range(n_triple_patterns):
        plans_to_rank = []
        for current_plan in current_plans:
            state = {
                "embedded": embedded,
                "embedded_prior": embedded_prior,
                "current_plan": current_plan,
                "n_tp_query": n_triple_patterns,

            }
            possible_next_with_cost = agent.step(state)
            plans_to_rank.extend(possible_next_with_cost)

        plans_to_rank.sort(key=lambda x: x[1])
        top_candidates = plans_to_rank[:beam_width]
        current_plans = [plan for plan, cost in top_candidates]

    return top_candidates

def main_train(queries_train, queries_val, disk_cache_embedded_query,
               agent,
               beam_width, n_epochs):
    loader = DataLoader(queries_train, batch_size=1, shuffle=True)
    for epoch in range(n_epochs):
        for query in tqdm(loader):
            n_tp_query = len(query.triple_patterns[0])
            embedded, embedded_prior = agent.embed_query(query, disk_cache_embedded_query)
            plans = beam_search(query, embedded, embedded_prior, agent, beam_width, n_tp_query)
            print(plans)
            #TODO: Execute plans concurrently
            #TODO: Store in a buffer (three targets + embedded query plan + perturbation)
            #TODO: Actually one target is stored, the others are computed using a cache that stores the best plan
            # from a given plan. So a tree-like structure that points smaller plans to the best plan from that smaller
            # plan and its cost and latency.
            # TODO: Every x queries do training run
            # TODO: Training should be all three heads at same time with epinet training.
            # TODO: This requires a separate perturbed target per query episode, so needs to be stored in buffer
            # TODO: Annealing method to slowly switch from exploration with epinet on cost to epinet on latency
            # TODO: Initial phase is just executing x queries with plans to get a sense of distribution of rewards
            # TODO: Then normalize based on these rewards
            # TODO: For robust query planning use: Conditional Value at Risk (CVaR / Expected Shortfall)



def main_online_estimation_experiment(endpoint_location, queries_location_train, queries_location_val,
                                      rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                      full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension,
                                      alpha_mlp, alpha_ensemble, sigma,
                                      beam_width, n_epochs,
                                      device):
    queries_train, queries_val, client, writer =\
        prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                           rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    embedded_query_cache = DiskCacheFrozenRepresentations('frozen_query_embeddings.h5')
    query_plan_cache = DiskCacheFrozenRepresentations('frozen_query_plans.h5')

    model = prepare_model(full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension, device)
    agent = EpinetThompsonSampling(model,
                                   alpha_mlp, alpha_ensemble, sigma,
                                   precomputed_indexes, precomputed_masks,
                                   device)
    main_train(queries_train, queries_val, embedded_query_cache, agent, beam_width, n_epochs)

def parameter_train_wrapper():
    n_queries_per_train_batch = 32
    beam_width = 4
    max_plans = 10
    mlp_dimension_full = 64
    n_epochs = 25

    # Hyperparameter results from tuning runs on partial data
    n_epi_indexes_train = 16
    n_epi_indexes_val = 1000
    epinet_index_dim = 32
    sigma = 0.60
    alpha_mlp = 0.08
    alpha_ensemble = 0.30
    lr = .0001
    weight_decay = 0.04
    n_epochs = 25

    endpoint_location = "http://localhost:8888"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_oracle = "experiments/model_configs/policy_networks/t_cv_repr_huge.yaml"
    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"
    model_config_emb_pair_norm = "experiments/model_configs/policy_networks/t_cv_repr_pair_norm_cardinality_head_own_embeddings.yaml"
    model_config_emb_graph_norm = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_cardinality_head_own_embeddings.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"
    trained_cost_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-12-02-2026-17-17-13/epoch-25/model/epinet_model.pt"

    emb_experiment_dir = ("experiments/experiment_outputs/yago_gnce/pretrained_models/"
                      "pretrain_experiment_triple_conv-15-12-2025-11-10-45")
    emb_experiment_dir_pair_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_pair_norm-15-12-2025-10-00-26")

    emb_experiment_dir_graph_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_graph_norm-15-12-2025-09-12-57")

    save_loc_simulated = "data/simulated_query_plan_data/star_yago_gnce/data"
    save_loc_simulated_val = "data/simulated_query_plan_data/star_yago_gnce/val_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    main_online_estimation_experiment(endpoint_location,queries_location_train, queries_location_val,
                                      rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                      full_gnn_config=model_config_emb, config_ensemble_prior=model_config_prior,
                                      epinet_index_dim=epinet_index_dim, mlp_dimension=mlp_dimension_full,
                                      alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble, sigma=sigma,
                                      beam_width=beam_width, n_epochs=n_epochs,
                                      device=device,
                                      )

if __name__ == '__main__':
    parameter_train_wrapper()
