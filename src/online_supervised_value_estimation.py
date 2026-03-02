import asyncio
import re
from collections import deque

from typing import List, Tuple, Any, TypedDict

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from src.models.epistemic_neural_network import EpistemicNetwork, prepare_model
from src.models.model_instantiator import ModelFactory
from src.models.query_plan_prediction_model import PlanCostEstimatorFull, QueryPlansPredictionModel
from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient
from src.utils.epinet_utils.disk_cache_frozen_representations import DiskCacheFrozenRepresentations
from src.utils.training_utils.query_loading_utils import prepare_data
from src.utils.training_utils.training_tracking import ExperimentWriter
from torch import nn
import torch.multiprocessing as mp
from tqdm import tqdm

from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask

Plan = List[int]
EnvState = TypedDict('EnvState', {
    'unweighted_ensemble_prior': np.ndarray,
    "prepared_trees": np.ndarray,
    "prepared_idx": np.ndarray,
    "prepared_masks": np.ndarray,
    "z": np.ndarray
})
Cost = float
HistoryStep = Tuple[Plan, EnvState]
History = List[HistoryStep]
Candidate = Tuple[Plan, Cost, History, set[str]]

class AbstractCostAgent:
    def setup_episode(self, query):
        raise NotImplementedError

    def estimate_costs(self, possible_next_plans, query_state):
        raise NotImplementedError


class EpinetMultiprocessAgent(AbstractCostAgent):
    def __init__(self, model,
                 inference_queue, result_queue, worker_id,
                 disk_cache,
                 precomputed_indexes, precomputed_masks,
                 alpha_mlp, alpha_ensemble):
        self.model = model
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.disk_cache = disk_cache
        self.precomputed_indexes = precomputed_indexes
        self.precomputed_masks = precomputed_masks
        self.alpha_mlp = alpha_mlp
        self.alpha_ensemble = alpha_ensemble

    def setup_episode(self, query):
        """
        Starts a search episode. Samples a single z and embeds the query based on this z
        :param query:
        :return:
        """
        self.model.eval()
        z = torch.randn((1, self.model.epi_index_dim), device=torch.device('cpu'))

        # TODO: This + any other cache on disk should use a one write many read architecture
        # if query.query[0] not in self.disk_cache:
        embedded, embedded_prior = self.model.embed_query_batched(query)[0], self.model.embed_query_batched_prior(query)
            # self.disk_cache.store_embeddings(query.query[0], (embedded, embedded_prior))
        # else:
        #     embedded, embedded_prior = self.disk_cache.get_embeddings(query.query[0], torch.device('cpu'))

        return {"embedded": embedded, "embedded_prior": embedded_prior, "z": z, "query_idx": 0}

    def estimate_costs(self, possible_next, query_state):
        # Format the plans
        formatted_plans = [(p,) for p in possible_next]

        # Prepare query plan structures on CPU
        prep_trees, prep_idx, prep_masks = self.model.prepare_cost_estimation_inputs(
            formatted_plans, query_state["embedded"], self.precomputed_indexes,
            self.precomputed_masks, target_device=torch.device('cpu')
        )

        # Delegate forward pass to GPU server
        payload = {
            "prep_trees": prep_trees, "prep_idx": prep_idx,
            "prep_masks": prep_masks, "z": query_state["z"]
        }
        self.inference_queue.put((self.worker_id, payload))

        # Compute (tiny) ensemble models on CPU while we wait for GPU
        prior_trees, prior_idx, prior_masks = self.model.prepare_ensemble_prior_inputs(
            formatted_plans, query_state["embedded_prior"], self.precomputed_indexes,
            self.precomputed_masks, query_idx=query_state["query_idx"]
        )
        # TODO: These priors should return all heads or at least latency and query_total_rows
        unweighted_priors = self.model.compute_ensemble_prior_from_prepared(
            prior_trees, prior_idx, prior_masks, cost_name='query_total_rows'
        )
        ensemble_prior = torch.matmul(query_state["z"], unweighted_priors).view(-1, 1)

        # Get GPU cost estimates
        gpu_results = self.result_queue.get()
        est_cost = gpu_results["est_cost"]
        mlp_prior = gpu_results["mlp_prior"]
        learnable_mlp = gpu_results["learnable_mlp"]

        # Combine for total output
        epinet_cost = est_cost + learnable_mlp + (self.alpha_mlp * mlp_prior) + (self.alpha_ensemble * ensemble_prior)

        # Combine relevant (frozen) environment states and return. This will be stored in the buffer
        environment_state: List[EnvState] = []
        for i, plan in enumerate(formatted_plans):
            environment_state.append({
                "unweighted_ensemble_prior": unweighted_priors.cpu().numpy()[i],
                "prepared_trees": prep_trees.cpu().numpy()[i],
                "prepared_idx": prep_idx.cpu().numpy()[i],
                "prepared_masks": prep_masks.cpu().numpy()[i],
                "z": query_state["z"].cpu().numpy()
            })

        return epinet_cost.view(-1).tolist(), environment_state


def generic_gpu_server(inference_queue, result_queues, handler_class, handler_kwargs):
    model_handler = handler_class(**handler_kwargs)
    print("GPU Server: Handler initialized successfully.")

    while True:
        request = inference_queue.get()
        if request is None: break

        worker_id, payload = request
        result = model_handler.process(payload)
        result_queues[worker_id].put(result)


class EpinetServerHandler:
    def __init__(self, model_builder_fn, model_kwargs, state_dict, device):
        self.device = device

        # Build model using passed build function and arguments
        self.model = model_builder_fn(**model_kwargs, device=self.device)

        # Load any weights passed
        self.model.load_state_dict(state_dict)

        # Set to eval as this is an inference 'server'. Training is done
        # on separate worker
        self.model.eval()

    def process(self, payload):
        with torch.no_grad():
            z_gpu = payload["z"].to(self.device)

            # TODO: This should return all heads output
            est_cost, last_feature = self.model.estimate_cost_from_prepared(
                payload["prep_trees"], payload["prep_idx"], payload["prep_masks"], cost_name='query_total_rows'
            )

            mlp_prior = self.model.compute_mlp_prior(last_feature, z_gpu)
            learnable_mlp = self.model.compute_learnable_mlp(last_feature, z_gpu)

            return {
                "est_cost": est_cost.cpu(),
                "mlp_prior": mlp_prior.cpu(),
                "learnable_mlp": learnable_mlp.cpu()
            }

class PlanBestPerformanceCache:
    def __init__(self):
        self.plan_performance_cache = {}

    def add_execution(self, plan: List[int], query: str, latency: float, total_rows: int):
        plan_identifier = self.create_plan_identifier(plan, query)
        self.add_execution_raw(plan_identifier, latency, total_rows)

    def add_execution_raw(self, plan_identifier, latency: float, total_rows: int):
        """
        Function that registers an execution, if it is not the best execution in some way it does nothing
        :param plan_identifier:
        :param latency:
        :param total_rows:
        :return:
        """
        if plan_identifier in self.plan_performance_cache:
            performance = self.plan_performance_cache[plan_identifier]
            # Our target is latency, so best plan is determined by this
            if performance["latency"] > latency:
                performance["latency"] = latency
                performance["total_rows"] = total_rows
        else:
            self.plan_performance_cache[plan_identifier] = { "latency": latency, "total_rows": total_rows }

    def get_target(self, plan: List[int], query: str) -> tuple[float, float]:
        plan_identifier = self.create_plan_identifier(plan, query)
        return self.get_target_raw(plan_identifier)

    def get_target_raw(self, plan_identifier) -> tuple[float, float]:
        # The cache content must always be set as you can only try to construct latencies from plans
        # that have been executed
        if plan_identifier not in self.plan_performance_cache:
            raise KeyError(f"No plan: {plan_identifier}")
        return self.plan_performance_cache[plan_identifier]

    @staticmethod
    def create_plan_identifier(plan: List[int], query: str) -> tuple[tuple[int, ...], str]:
        return tuple(plan), query


class PlanExperienceBuffer:
    def __init__(self, max_size):
        self.plan_execution_buffer = []
        # Not yet implemented
        self.plan_execution_sample_weights = []
        self.max_size = max_size
        pass

    def sample_experiences(self):
        pass

    def add_experience(self, experience):
        self.plan_execution_data_points.append(experience)

        pass

    def __size__(self):
        return len(self.plan_execution_data_points)


def prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                       rdf2vec_vector_location, occurrences_location, tp_cardinality_location):
    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    client = QLeverOptimizerClient("http://localhost:8888")
    writer = ExperimentWriter("experiments/experiment_outputs/yago_gnce/online_epinet_training",
                              "actual_cost_epinet_training",
                              {}, {})
    return train_dataset, val_dataset, client, writer


def beam_search(query: Any, agent: AbstractCostAgent, beam_width: int) -> List[Candidate]:
    test = query.query[0]
    # Get all variables per triple pattern to filter out cross-products
    tp_query = query.triple_patterns[0]
    variables_query = []
    for tp in tp_query:
        variable_pattern = r"[\?\$]\w+"
        variables_tp = set(re.findall(variable_pattern, tp))
        variables_query.append(variables_tp)


    n_tp_query = len(query.triple_patterns[0])
    query_state = agent.setup_episode(query)

    # current_plans stores tuples of (plan, history)
    current_plans: List[Tuple[Plan, History, set[str]]] = [([i], [], variables_query[i]) for i in range(n_tp_query)]

    top_candidates: List[Candidate] = []
    for depth in range(n_tp_query - 1):
        possible_next: List[Plan] = []
        histories: List[History] = []
        variables: List[set[str]] = []

        for plan, history, variables_plan in current_plans:
            used_actions = set(plan)
            for a in range(n_tp_query):
                if a not in used_actions:
                    # The order of the first two entries does not matter in query optimization
                    if depth == 0 and a < plan[0]:
                        continue
                    # Prevent cartesian joins
                    if len(variables_plan.intersection(variables_query[a])) == 0:
                        continue
                    possible_next.append(plan + [a])
                    histories.append(history)
                    variables.append(variables_plan.union(variables_query[a]))

        costs, environment_states = agent.estimate_costs(possible_next, query_state)

        # Rank, construct history lineage, and prune
        plans_with_data: List[Candidate] = []
        for i in range(len(possible_next)):
            new_plan = possible_next[i]
            cost = float(costs[i])
            state = environment_states[i]

            # Append the current step's plan and state to its specific history path
            new_history = histories[i] + [(new_plan, state)]
            plans_with_data.append((new_plan, cost, new_history, variables[i]))

        plans_with_data.sort(key=lambda x: x[1])
        top_candidates = plans_with_data[:beam_width]

        # Prepare the next iteration
        current_plans = [(plan, history, variables_candidate) for plan, cost, history, variables_candidate in top_candidates]

    return top_candidates


def cpu_search_worker_epinet(model_builder_fn, model_kwargs, state_dict,
                             worker_id, query_queue, plan_queue, inference_queue, result_queue,
                             disk_cache, precomputed_indexes, precomputed_masks,
                             alpha_mlp, alpha_ensemble, beam_width):
    import torch

    # Restrict this specific worker process to 1 PyTorch thread
    torch.set_num_threads(1)

    # Build model using passed build function and arguments
    model = model_builder_fn(**model_kwargs, device=torch.device('cpu'))
    # Load any weights passed
    model.load_state_dict(state_dict)
    model.eval()

    agent = EpinetMultiprocessAgent(
        model, inference_queue, result_queue, worker_id,
        disk_cache, precomputed_indexes, precomputed_masks,
        alpha_mlp, alpha_ensemble
    )

    while True:
        query = query_queue.get()
        if query is None:  # Poison pill to shut down
            print("SHUTDOWN?")
            break

        top_k_plans = beam_search(query, agent, beam_width)

        plan_queue.put({
            "query": query,
            "top_k_plans": top_k_plans
        })


async def evaluate_batch_async(qlever_client, training_buffer, max_concurrent=4):
    """
    Executes a batch of query plans asynchronously but limits
    how many queries hit the database at the exact same time.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded_execute(item):
        query = item["query"].to_data_list()[0]
        best_plan = item["plan"][0]

        async with semaphore:
            raw_result = await qlever_client.execute_plan(query, join_order=best_plan)
            metrics = qlever_client.extract_signal(raw_result)
            item["rl_metrics"] = metrics
            return item

    # Run all tasks in the batch concurrently (up to the semaphore limit)
    tasks = [_bounded_execute(item) for item in training_buffer]
    return await asyncio.gather(*tasks)

def execute_plans(client, plans, max_concurrent=4):
    # Batch queries with asyncio and max_concurrent
    # return results
    result = asyncio.run(
        evaluate_batch_async(client, plans, max_concurrent=max_concurrent)
    )
    return result

def update_best_execution_cache_and_buffer(execution_result, result_cache: PlanBestPerformanceCache):
    # Get execution results, decompose into partial plans
    for execution in execution_result:
        # Convert pytorch geometric DataBatch to list, take first (and only) element and the string representation
        # of the query
        query = execution["query"].to_data_list()[0].query
        latency = execution["rl_metrics"]["latency"]
        total_cost = execution["rl_metrics"]["total_cost"]
        # History contains all partial plans and their state (including the full plan)
        history = execution["plan"][2]
        for plan in history:
            # Each subplan gets updated by this execution. Will only update if latency is better than previous
            # recorded latency for that subplan
            result_cache.add_execution(plan[0], query, latency, total_cost)

            # TODO: Add this execution to buffer. Buffer requires the query, state, intermediate join size, and
            # two perturbation vectors

            # Add the plan to the buffer with attached partial cost and two perturbation vectors as perturbation
            # is on a per-observation basis
    pass

def retrieve_buffer_samples():
    # Retrieve buffer samples with size n
    # Determine the sample's target
    # return
    pass

def determine_targets_of_sample():
    # Retrieve the best latency and cost of a given (sub)plan
    # Perturb the latency and cost with the stored observation
    # return
    pass

def train_step(model, optimizer, losses):
    pass

def main_train(queries_train,
               client,
               model_kwargs,
               disk_cache,
               precomputed_indexes, precomputed_masks,
               alpha_mlp, alpha_ensemble, beam_width, n_epochs,
               gpu_device):
    execution_result_cache = PlanBestPerformanceCache()
    mp.set_start_method('spawn', force=True)
    num_workers = 1

    # Create the model here once to obtain the state_dict to pass to the workers
    initial_model = prepare_model(**model_kwargs, device=gpu_device)
    shared_state_dict = initial_model.state_dict()

    # create the queue passing query to cpu workers
    query_queue = mp.Queue()
    # create the queue passing complete query plans to main worker
    plan_queue = mp.Queue()
    # create queue passing computed plan representations to GPU for forward pass
    inference_queue = mp.Queue()
    # create queues passing forward pass result to workers
    result_queues = [mp.Queue() for _ in range(num_workers)]
    # create queues passing updated weights to the CPU and GPU workers
    weights_queue = [mp.Queue() for _ in range(num_workers)]

    handler_kwargs = {
        # Function that builds the model
        "model_builder_fn": prepare_model,
        "model_kwargs": model_kwargs,
        "state_dict": shared_state_dict,
        # GPU to run on
        "device": gpu_device  # The target GPU
    }

    # gpu_handler = EpinetServerHandler(model, device)
    gpu_process = mp.Process(
        target=generic_gpu_server,
        args=(inference_queue, result_queues, EpinetServerHandler, handler_kwargs)
    )
    gpu_process.start()

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=cpu_search_worker_epinet,
            args=(prepare_model, model_kwargs, shared_state_dict,
                  i, query_queue, plan_queue, inference_queue, result_queues[i],
                  disk_cache, precomputed_indexes, precomputed_masks,
                  alpha_mlp, alpha_ensemble, beam_width)
        )
        p.start()
        workers.append(p)

    loader = DataLoader(queries_train, batch_size=1, shuffle=True)

    for epoch in range(n_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{n_epochs} ---")

        for query in loader:
            query_queue.put(query)

        completed_queries = 0
        queries_since_last_train = 0
        train_buffer = []

        with tqdm(total=len(loader)) as pbar:
            while completed_queries < len(loader):
                result = plan_queue.get()
                for plan in result['top_k_plans']:
                    train_buffer.append({ "query": result["query"], "plan": plan })

                completed_queries += 1
                queries_since_last_train += 1

                pbar.update(1)

                # TODO: Every x queries do training run
                if queries_since_last_train >= 32:
                    print("\n[Triggering Model Training...]")
                    execution_results = execute_plans(client, train_buffer, 1)
                    update_best_execution_cache_and_buffer(execution_results, execution_result_cache)
                    # execute_training_step(training_buffer, model, optimizer)
                    queries_since_last_train = 0
                    train_buffer.clear()

    for _ in range(num_workers):
        query_queue.put(None)
    for p in workers:
        p.join()

    inference_queue.put(None)
    gpu_process.join()

    print("Training Complete.")

    #TODO: Execute plans concurrently
    #TODO: Store in a buffer (three targets + embedded query plan + perturbation)
    #TODO: Actually one target is stored, the others are computed using a cache that stores the best plan
    # from a given plan. So a tree-like structure that points smaller plans to the best plan from that smaller
    # plan and its cost and latency.
    # TODO: Every x queries do training run
    # TODO: Training should be all two heads at same time with epinet training and just the join cost
    # TODO: This requires a separate perturbed target per query episode, so needs to be stored in buffer
    # TODO: Annealing method to slowly switch from exploration with epinet on cost to epinet on latency
    # TODO: Initial phase is just executing x queries with plans to get a sense of distribution of rewards
    # TODO: Then normalize based on these rewards
    # TODO: For robust query planning use: Conditional Value at Risk (CVaR / Expected Shortfall)
    # TODO: This should be when the agent is set to eval mode
    # TODO: For optimization of eval mode we can look into incremental tree building. Where you just append
    # TODO: some tensors to current representation

    #TODO: For generalization we can look into two things
    # 1. MoE in GNN based on graph structural embedding. Or other MoE mechanism. Maybe use epinet to determine
    #  how many experts are needed for a query; when uncertainty is high route to more experts?
    # 2. Use epinet uncertainty estimate to generate unseen query templates. We train epinet on the default shapes
    #  then sample using random walks (with restrictions on valid walks and 5% being corrupted walks with no results)
    #  rank the queries based on uncertainty -> top k get entered into the data training pipeline.
    #  prevent forgetting in some smart way
    #  Or we could even do generation through an endpoint and ranking possible next walks by epistemic uncertainty.
    #  so not only guide plan generation but also training data generation through uncertainty estimates
    # 3. Train an epinet on the standard optimizer??? What if we could make that bad boy robust?



def main_online_estimation_experiment(endpoint_location, queries_location_train, queries_location_val,
                                      rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                      full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension,
                                      trained_epinet_location,
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

    model_kwargs = {
        "full_gnn_config": full_gnn_config,
        "config_ensemble_prior": config_ensemble_prior,
        "epinet_index_dim": epinet_index_dim,
        "mlp_dimension": mlp_dimension,
        "model_weights": trained_epinet_location
    }
    main_train(queries_train,
               client,
               model_kwargs,
               embedded_query_cache,
               precomputed_indexes,
               precomputed_masks,
               alpha_mlp,
               alpha_ensemble,
               beam_width,
               n_epochs,
               device)

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

    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"
    trained_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-12-02-2026-17-17-13/epoch-25/model/epinet_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_online_estimation_experiment(endpoint_location,queries_location_train, queries_location_val,
                                      rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                      full_gnn_config=model_config_emb, config_ensemble_prior=model_config_prior,
                                      epinet_index_dim=epinet_index_dim, mlp_dimension=mlp_dimension_full,
                                      trained_epinet_location=trained_model_file,
                                      alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble, sigma=sigma,
                                      beam_width=beam_width, n_epochs=n_epochs,
                                      device=device,
                                      )

if __name__ == '__main__':
    parameter_train_wrapper()
