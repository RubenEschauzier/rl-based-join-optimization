import os
import queue
import sys
from dataclasses import asdict
import asyncio
import numpy as np
from typing import List

import ray
from ray.util import ActorPool
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch, Data
from tqdm import tqdm

import torch
from torch import nn
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader

from main import find_best_epoch_directory

from src.models.epistemic_neural_network import prepare_epinet_model

from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient

from src.supervised_value_estimation.agents.EpinetMultiprocessAgent import EpinetMultiprocessAgent
from src.supervised_value_estimation.loss_functions.hinge_mse_loss import right_censored_hinge_loss
from src.supervised_value_estimation.normalizers.normalizer_exponential_moving_average import \
    NormalizerExponentialMovingAverage
from src.supervised_value_estimation.storage.PlanBestPerformanceCache import PlanBestPerformanceCache
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search
from src.supervised_value_estimation.storage.ExecutionReplayBuffer import ExecutionReplayBuffer, ExecutionBufferSamples, \
    ExecutionBufferSamplesWithTargets

from src.utils.training_utils.query_loading_utils import prepare_data
from src.utils.training_utils.training_tracking import ExperimentWriter
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask


class AsyncExecutionStrategy:
    """Handles standard single-endpoint execution via asyncio."""

    def __init__(self, endpoint_location: str, max_concurrent: int = 4):
        self.client = QLeverOptimizerClient(endpoint_location)
        self.max_concurrent = max_concurrent
        self.async_loop = None

    def setup(self):
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_until_complete(self.client.create_session())

    def execute(self, plans: list):
        return self.async_loop.run_until_complete(
            self._evaluate_batch_async(plans)
        )

    def teardown(self):
        if self.async_loop:
            self.async_loop.run_until_complete(self.client.close())
            self.async_loop.close()

    @property
    def default_timeout_s(self) -> float:
        return self.client.default_timeout_s

    async def _evaluate_batch_async(self, training_buffer):
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _bounded_execute(item):
            query = item["query"].to_data_list()[0]
            best_plan = item["plan"]["plan"]

            async with semaphore:
                raw_result = await self.client.execute_plan(query, join_order=best_plan)
                metrics = self.client.extract_signal(raw_result)
                item["rl_metrics"] = metrics
                return item

        tasks = [_bounded_execute(item) for item in training_buffer]
        return await asyncio.gather(*tasks)


class RayExecutionStrategy:
    """Handles distributed execution across multiple baremetal servers via Ray."""

    def __init__(self, ray_endpoints: list[str]):
        if not ray.is_initialized():
            context = ray.init()
            print(context.dashboard_url)

        from src.query_environments.qlever.qlever_execute_query_ray import QLeverOptimizerClientRay as RayClient

        # 1. Instantiate the actors
        self.actors = [RayClient.remote(url) for url in ray_endpoints]

        # 2. Wrap them in a Ray ActorPool for dynamic load balancing
        self.pool = ActorPool(self.actors)

        # 3. Local dummy parser to avoid remote execution overhead for simple dictionary extraction
        self.local_parser = QLeverOptimizerClient("dummy_endpoint")

    def setup(self):
        pass

    def execute(self, plans: list):
        """Executes plans using load-aware scheduling."""

        # Define the execution instruction for the pool
        def submit_query(actor, item):
            query_data = item["query"].to_data_list()[0]

            query_payload = {
                "query": query_data.query,
                "triple_patterns": query_data.triple_patterns
            }

            best_plan = item["plan"]["plan"]

            return actor.execute_plan.remote(query_payload, join_order=best_plan)

        # pool.map automatically routes tasks to idle actors.
        # It guarantees the output list matches the order of the input 'plans' list,
        # ensuring zip() aligns the metrics perfectly with the original items.
        raw_results = list(self.pool.map(submit_query, plans))

        # Parse results locally
        for item, raw_result in zip(plans, raw_results):
            item["rl_metrics"] = self.local_parser.extract_signal(raw_result)

        return plans

    def teardown(self):
        pass

    @property
    def default_timeout_s(self) -> float:
        return self.local_parser.default_timeout_s

def tensors_to_numpy(obj):
    """
    Recursively converts all PyTorch tensors in a dictionary, list, or tuple
    to NumPy arrays to prevent file descriptor leaks in multiprocessing queues.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: tensors_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensors_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensors_to_numpy(v) for v in obj)
    else:
        return obj

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor,
                    valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    mse_loss = torch.nn.functional.mse_loss(predictions, targets, reduction='none')
    return (mse_loss * valid_mask).sum() / valid_mask.sum()


def generic_gpu_server(inference_queue, result_queues, weights_queue, handler_class, handler_kwargs):
    model_handler = handler_class(**handler_kwargs)
    print("GPU Server: Handler initialized successfully.")

    while True:
        try:
            while True:
                new_state_dict_numpy = weights_queue.get_nowait()
                new_state_dict = {k: torch.from_numpy(v) for k, v in new_state_dict_numpy.items()}
                model_handler.model.load_state_dict(new_state_dict)
        except queue.Empty:
            pass

        request = inference_queue.get()
        if request is None: break

        worker_id, payload = request

        result = model_handler.process(payload)
        result = tensors_to_numpy(result)
        result_queues[worker_id].put(result)


# This can be extended to have a base abstract class
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

            est_cost, last_feature = self.model.estimate_cost_from_prepared(
                payload["prep_trees"], payload["prep_idx"], payload["prep_masks"]
            )

            mlp_prior = self.model.compute_mlp_prior(last_feature, z_gpu)
            learnable_mlp = self.model.compute_learnable_mlp(last_feature, z_gpu)

            return {
                "est_cost": {key: output.cpu() for key, output in est_cost.items()},
                "mlp_prior": {key: output.cpu() for key, output in mlp_prior.items()},
                "learnable_mlp": {key: output.cpu() for key, output in learnable_mlp.items()}
            }


def prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                       rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                       use_ray=False, ray_endpoints=None):
    train_dataset, val_dataset = prepare_data(endpoint_location, queries_location_train, queries_location_val,
                                              rdf2vec_vector_location, occurrences_location, tp_cardinality_location)
    # client = QLeverOptimizerClient("http://localhost:8888")
    # Initialize the selected execution strategy
    if use_ray:
        if not ray_endpoints:
            raise ValueError("ray_endpoints must be provided when use_ray is True")
        execution_strategy = RayExecutionStrategy(ray_endpoints)
    else:
        execution_strategy = AsyncExecutionStrategy(endpoint_location)

    writer = ExperimentWriter("experiments/experiment_outputs/yago_gnce/online_epinet_training",
                              "actual_cost_epinet_training",
                              {}, {})
    return train_dataset, val_dataset, execution_strategy, writer


def cpu_search_worker_epinet(model_builder_fn, model_kwargs, state_dict,
                             worker_id,
                             query_queue, plan_queue, inference_queue, result_queue, uncertainty_queue,
                             precomputed_indexes, precomputed_masks,
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
        model, inference_queue, result_queue, uncertainty_queue, worker_id,
        precomputed_indexes, precomputed_masks,
        alpha_mlp, alpha_ensemble
    )

    while True:
        safe_query = query_queue.get()
        if safe_query is None:  # Poison pill to shut down
            print("SHUTDOWN?")
            break

        # Reconstruct PyTorch tensors
        query_tensors = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in safe_query.items()
        }

        # Rebuild PyG Batch
        query_data = Data.from_dict(query_tensors)
        query = Batch.from_data_list([query_data])

        top_k_plans = beam_search(query, agent, beam_width)

        plan_queue.put({
            "query": safe_query,
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
        best_plan = item["plan"]["plan"]

        async with semaphore:
            raw_result = await qlever_client.execute_plan(query, join_order=best_plan)
            metrics = qlever_client.extract_signal(raw_result)
            item["rl_metrics"] = metrics
            return item

    # Run all tasks in the batch concurrently (up to the semaphore limit)
    tasks = [_bounded_execute(item) for item in training_buffer]
    return await asyncio.gather(*tasks)


def execute_plans(client, plans, async_loop, max_concurrent=4):
    # Batch queries with asyncio and max_concurrent
    # return results
    result = async_loop.run_until_complete(
        evaluate_batch_async(client, plans, max_concurrent=max_concurrent)
    )
    return result


def process_execution_results(
        execution_results: list,
        client_default_timeout: float,
        result_cache: PlanBestPerformanceCache,
        executions_buffer: ExecutionReplayBuffer,
        normalizers: dict[str, NormalizerExponentialMovingAverage],
        epi_index_dim: int,
        device
):
    latencies = []
    total_costs = []
    valid_costs_mask = []
    join_rows = []
    valid_joins_mask = []

    for execution in execution_results:
        query = execution["query"].to_data_list()[0].query

        metrics = execution["rl_metrics"]
        is_error = metrics["is_error"]

        # Apply timeout penalty and mask total_cost if query failed
        latency = np.log1p(client_default_timeout if is_error else metrics["latency"])
        latencies.append(latency)

        # Total execution cost (# of rows) is invalid when query fails so we discard it
        total_cost = -1 if is_error else np.log1p(metrics["total_cost"])
        is_censored = bool(is_error)
        total_costs.append(total_cost)
        valid_costs_mask.append(not is_censored)

        for i, row_cnt in enumerate(metrics["per_join_rows"]):
            join_rows.append(np.log1p(row_cnt))
            valid_joins_mask.append(metrics["is_valid_join_row"][i])

        history = execution["plan"]["history"]

        for i, (join_plan, plan_features) in enumerate(history):
            # Update cache with censorship awareness
            result_cache.add_execution(join_plan, query, latency, total_cost, is_censored)

            # Generate and normalize target perturbation vectors
            c_vectors = torch.randn((2, 1, epi_index_dim), device='cpu')
            c_vectors = torch.nn.functional.normalize(c_vectors, dim=-1).numpy()

            # Record state in execution buffer
            executions_buffer.add(
                query_string=query,
                join_plan=join_plan,
                prepared_trees=plan_features["prepared_trees"],
                prepared_idx=plan_features["prepared_idx"],
                prepared_masks=plan_features["prepared_masks"],
                unweighted_ensemble_prior=plan_features["unweighted_ensemble_prior"],
                episode_z=plan_features["z"],
                intermediate_join_size=join_rows[i],
                is_valid_size=valid_joins_mask[i],
                is_valid_episode=not is_error,
                c_vectors_observation=c_vectors
            )

    normalizers["latency"].update(
        torch.tensor(latencies, dtype=torch.float32, device=device)
    )
    normalizers["plan_cost"].update(
        torch.tensor(total_costs, dtype=torch.float32, device=device),
        valid_mask=torch.tensor(valid_costs_mask, dtype=torch.bool, device=device)
    )
    normalizers["join_rows"].update(
        torch.tensor(join_rows, dtype=torch.float32, device=device),
        valid_mask=torch.tensor(valid_joins_mask, dtype=torch.bool, device=device)
    )

    return np.mean(latencies), np.ma.array(data=total_costs, mask=~np.array(valid_costs_mask)).mean()


def retrieve_buffer_samples(batch_size,
                            execution_buffer: ExecutionReplayBuffer,
                            reward_cache: PlanBestPerformanceCache):
    batched_samples: ExecutionBufferSamples = execution_buffer.sample(batch_size)
    latencies, total_cost, is_censored = determine_targets_of_sample(batched_samples, reward_cache)
    batched_samples_with_targets = ExecutionBufferSamplesWithTargets(
        **asdict(batched_samples),
        latencies=latencies,
        total_cost=total_cost,
        is_censored=is_censored
    )
    return batched_samples_with_targets


def determine_targets_of_sample(batched_sample: ExecutionBufferSamples,
                                reward_cache: PlanBestPerformanceCache):
    latencies, total_cost, is_censored = [], [], []
    for i in range(batched_sample.queries.shape[0]):
        reward_info = reward_cache.get_target(batched_sample.join_plans[i], batched_sample.queries[i])
        latencies.append(reward_info["latency"])
        total_cost.append(reward_info["total_cost"])
        is_censored.append(reward_info["is_censored"])
    return (torch.tensor(latencies, dtype=torch.float32),
            torch.tensor(total_cost, dtype=torch.float32),
            torch.tensor(is_censored, dtype=torch.bool))


def perturb_reward_signals(sample_with_targets: ExecutionBufferSamplesWithTargets,
                           normalizers,
                           epinet_indexes, sigma, device):
    latency = sample_with_targets.latencies.to(device)
    latency = normalizers["latency"].normalize(latency)
    # (batch_size, n_target_types, 1, epinet_index_dim) -> (batch_size, epi_index_dim)
    latency_c_vector = sample_with_targets.c_vectors[:, 0, :, :].squeeze()
    perturbed_latency = perturb_vector(latency, latency_c_vector, epinet_indexes, sigma)

    total_cost = sample_with_targets.total_cost.to(device)
    total_cost = normalizers["plan_cost"].normalize(total_cost)
    total_cost_c_vector = sample_with_targets.c_vectors[:, 1, :, :].squeeze()
    perturbed_total_cost = perturb_vector(total_cost, total_cost_c_vector, epinet_indexes, sigma)

    is_censored_expanded = sample_with_targets.is_censored.squeeze().repeat(epinet_indexes.shape[0])
    return is_censored_expanded, {
        "perturbed_latency": perturbed_latency,
        "perturbed_total_cost": perturbed_total_cost
    },


def get_intermediate_join_targets(sample_with_targets: ExecutionBufferSamples, epi_indexes):
    intermediate_join_targets = sample_with_targets.intermediate_join_sizes.squeeze()
    valid_join_size = sample_with_targets.is_valid_size.squeeze()
    valid_join_size_mask = (valid_join_size == True)
    return intermediate_join_targets, valid_join_size_mask


def perturb_vector(raw_targets, c_vectors, epinet_indexes, sigma):
    # (n_epinet_indexes, batch_size)
    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    # (n_epinet_indexes * batch_size) in n_epinet_indexes blocks of batch_size
    anchor_term_flat = anchor_matrix.view(-1)
    raw_targets_expanded = raw_targets.repeat(epinet_indexes.shape[0])
    # (n_epinet_indexes * batch_size) in n_epinet_indexes blocks of batch_size
    return raw_targets_expanded + sigma * anchor_term_flat


def get_variance_estimates(estimates, heads_to_extract_variance=('latency', 'plan_cost')):
    variances = {}
    for head_name in heads_to_extract_variance:
        estimated_variance = torch.var(estimates[head_name], dim=0)
        variances[head_name] = estimated_variance.mean().item()
    return variances


def estimate_cost(epinet_latency_estimation, sample_with_targets: ExecutionBufferSamples, epinet_indexes,
                  alpha_mlp, alpha_ensemble, device,
                  epistemic_nn_heads=('plan_cost', 'latency')):
    n_epi_indexes = epinet_indexes.shape[0]
    head_outputs = {}
    for batch_idx in range(sample_with_targets.prepared_trees.shape[0]):
        # Unsqueeze to add back batch dimension
        prepared_tree = torch.from_numpy(sample_with_targets.prepared_trees[batch_idx]).unsqueeze(dim=0).to(device)
        prepared_idx = torch.from_numpy(sample_with_targets.prepared_idx[batch_idx]).unsqueeze(dim=0).to(device)
        prepared_mask = torch.from_numpy(sample_with_targets.prepared_masks[batch_idx]).unsqueeze(dim=0).to(device)

        heads_output, last_feature = epinet_latency_estimation.estimate_cost_from_prepared(
            prepared_tree, prepared_idx, prepared_mask)

        for key, ensemble_prior in sample_with_targets.unweighted_ensemble_priors[batch_idx].items():
            if not key in head_outputs:
                head_outputs[key] = []

            if key in epistemic_nn_heads:
                ensemble_prior = torch.matmul(epinet_indexes,
                                              torch.tensor(ensemble_prior, device=device).unsqueeze(dim=-1))
                ensemble_prior_flat = ensemble_prior.view(-1, 1)
                mlp_prior = epinet_latency_estimation.compute_mlp_prior_batched(last_feature, epinet_indexes)
                learnable_mlp_prior = epinet_latency_estimation.compute_learnable_mlp_batched(last_feature,
                                                                                              epinet_indexes)

                estimated_cost_expanded = heads_output[key].repeat(n_epi_indexes, 1)
                epinet_estimated_cost = estimated_cost_expanded + (
                        learnable_mlp_prior[key] + alpha_mlp * mlp_prior[key] + alpha_ensemble * ensemble_prior_flat
                )
                head_outputs[key].append(epinet_estimated_cost)
            else:
                # If we don't train an epistemic neural network on this head we can just use base model outputs.
                head_outputs[key].append(heads_output[key])

    return {key: torch.stack(value, dim=1).squeeze() for key, value in head_outputs.items()}


def train_step(model, optimizer, normalizers,
               client_default_timeout,
               executions_buffer, execution_result_cache,
               train_step_batch_size, n_epi_indexes_train,
               lambda_aux_task,
               alpha_mlp, alpha_ensemble, sigma, device):
    #TODO:
    # - Investigate the popart normalization approach
    # - Validate annealing works by logging it
    # - Validate PER
    samples_with_targets = retrieve_buffer_samples(batch_size=train_step_batch_size,
                                                   execution_buffer=executions_buffer,
                                                   reward_cache=execution_result_cache)
    epinet_indexes = model.sample_epistemic_indexes_batched(n_epi_indexes_train)

    # Intermediate join is invalid when QLever output says it did not finish that particular
    # node in the tree
    intermediate_join_target, valid_intermediate_join_mask = get_intermediate_join_targets(
        samples_with_targets, epinet_indexes
    )
    intermediate_join_target = normalizers["join_rows"].normalize(intermediate_join_target)

    is_censored, perturbed_targets = perturb_reward_signals(samples_with_targets,
                                                            normalizers,
                                                            epinet_indexes,
                                                            sigma, device)

    estimated_costs = estimate_cost(model, samples_with_targets, epinet_indexes,
                                    alpha_mlp, alpha_ensemble, device=device)
    estimated_variances = get_variance_estimates(estimated_costs)

    loss_intermediate_size = masked_mse_loss(estimated_costs["join_rows"],
                                             intermediate_join_target,
                                             valid_intermediate_join_mask)

    # # Right censored loss to reflect that timeouts mean the plan would take at least timeout seconds
    normalized_threshold = normalizers["latency"].normalize(torch.tensor(np.log1p(client_default_timeout))).item()

    # All perturbed targets have shape (n_epinet_indexes * batch_size) in n_epinet_indexes blocks of batch_size
    # Estimated cost is shape (n_epinet_indexes, batch_size), which flattens into n_epinet_indexes blocks of batch_size
    loss_latency = right_censored_hinge_loss(estimated_costs["latency"].flatten(),
                                             perturbed_targets["perturbed_latency"],
                                             is_censored,
                                             normalized_threshold)
    loss_total_cost = masked_mse_loss(estimated_costs["plan_cost"].flatten(),
                                      perturbed_targets["perturbed_total_cost"],
                                      ~is_censored
                                      )
    total_loss = ((1 - lambda_aux_task * 2) * loss_latency + lambda_aux_task * loss_total_cost
                  + lambda_aux_task * loss_intermediate_size)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # per_abs_error = torch.abs(estimated_costs["latency"].flatten() - perturbed_targets["perturbed_latency"])
    # mean_per_abs_error = torch.mean(per_abs_error.view(n_epi_indexes_train, -1).T, dim = 1)
    # executions_buffer.update_priorities(samples_with_targets.indices, mean_per_abs_error)

    return total_loss.detach().cpu().item(), estimated_variances


def main_train(queries_train,
               execution_strategy,
               model_kwargs,
               precomputed_indexes, precomputed_masks,
               alpha_mlp, alpha_ensemble, sigma, lambda_aux_task,
               n_epi_indexes_train, n_epi_indexes_val,
               beam_width, n_epochs,
               samples_per_train, train_step_batch_size, n_batches_per_train_step, n_steps_before_train,
               gpu_device):

    # async_loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(async_loop)
    # async_loop.run_until_complete(client.create_session())

    execution_strategy.setup()
    client_default_timeout = execution_strategy.default_timeout_s

    execution_result_cache = PlanBestPerformanceCache()
    executions_buffer = ExecutionReplayBuffer(buffer_size=50000,
                                              epi_index_dim=model_kwargs["epinet_index_dim"],
                                              device=gpu_device)

    mp.set_start_method('spawn', force=True)
    num_workers = 4

    # Create the model here once and obtain the state_dict to pass to the workers
    epinet_latency_estimation = prepare_epinet_model(**model_kwargs, device=gpu_device)
    shared_state_dict = epinet_latency_estimation.state_dict()

    # Freeze query embedding model
    for param in epinet_latency_estimation.get_query_embedding_model_params():
        param.requires_grad = False
    epinet_latency_estimation.cost_estimation_model.query_emb_model.eval()

    # In online cost estimation only the plan model and epinet are trained
    params = list(epinet_latency_estimation.get_plan_cost_estimation_model_params())
    params.extend(list(epinet_latency_estimation.get_learnable_epinet_params()))

    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.04)

    # Apply gradient clipping for stability as this is in essence off-policy RL
    torch.nn.utils.clip_grad_norm_(params, max_norm=.5)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=1e-2)
    previous_lr = scheduler.get_last_lr()

    normalizers = {
        "latency": NormalizerExponentialMovingAverage(gpu_device),
        "plan_cost": NormalizerExponentialMovingAverage(gpu_device),
        "join_rows": NormalizerExponentialMovingAverage(gpu_device)
    }

    # Initialize CUDA before any asyncio happens
    if gpu_device.type == 'cuda':
        _ = torch.zeros(1).to(gpu_device)

    # create the queue passing query to cpu workers
    query_queue = mp.Queue()
    # create the queue passing complete query plans to main worker
    plan_queue = mp.Queue()
    # create queue passing computed plan representations to GPU for forward pass
    inference_queue = mp.Queue()
    # create queues passing forward pass result to workers
    result_queues = [mp.Queue() for _ in range(num_workers)]
    # create queue passing updated weights to the GPU workers. Cpu workers don't need them as they
    # have frozen weights only
    weights_queue = mp.Queue()
    # Queues passing predicted uncertainties to the worker to anneal cost and latency-based exploration
    uncertainty_queues = [mp.Queue() for _ in range(num_workers)]

    handler_kwargs = {
        # Function that builds the model
        "model_builder_fn": prepare_epinet_model,
        "model_kwargs": model_kwargs,
        "state_dict": shared_state_dict,
        # GPU to run on
        "device": gpu_device
    }

    # gpu_handler = EpinetServerHandler(model, device)
    gpu_process = mp.Process(
        target=generic_gpu_server,
        args=(inference_queue, result_queues, weights_queue, EpinetServerHandler, handler_kwargs)
    )
    gpu_process.start()

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=cpu_search_worker_epinet,
            args=(prepare_epinet_model, model_kwargs, shared_state_dict,
                  i, query_queue, plan_queue, inference_queue, result_queues[i], uncertainty_queues[i],
                  precomputed_indexes, precomputed_masks,
                  alpha_mlp, alpha_ensemble, beam_width)
        )
        p.start()
        workers.append(p)

    loader = DataLoader(queries_train, batch_size=1, shuffle=True)

    for epoch in range(n_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{n_epochs} ---")

        loader_iter = iter(loader)

        # Query progress
        completed_queries = 0
        queries_since_last_train = 0
        train_buffer = []

        # Backpressure
        queries_in_flight = 0
        max_in_flight = (samples_per_train * 2) + (num_workers * beam_width)

        # Metric tracking
        epoch_train_losses = []
        epoch_latencies = []

        with tqdm(total=len(loader)) as pbar:
            while completed_queries < len(loader):

                # Maintain backpressure
                while queries_in_flight < max_in_flight:
                    try:
                        query = next(loader_iter)

                        # Deconstruct PyG Batch to a dictionary, then convert tensors to NumPy
                        safe_query = tensors_to_numpy(query.to_data_list()[0].to_dict())
                        query_queue.put(safe_query)

                        queries_in_flight += 1
                    except StopIteration:
                        break

                result = plan_queue.get()
                queries_in_flight -= 1

                query_tensors = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in result["query"].items()
                }
                # Build standard Data object, then collate properly
                query_data = Data.from_dict(query_tensors)
                reconstructed_query = Batch.from_data_list([query_data])

                for plan in result['top_k_plans']:
                    train_buffer.append({"query": reconstructed_query, "plan": plan})

                completed_queries += 1
                queries_since_last_train += 1

                pbar.update(1)
                if queries_since_last_train >= samples_per_train and completed_queries > n_steps_before_train:
                    execution_results = execution_strategy.execute(train_buffer)
                    # execution_results = execute_plans(client, train_buffer, async_loop, 1)
                    avg_latency, avg_total_cost = process_execution_results(
                        execution_results=execution_results,
                        client_default_timeout=client_default_timeout,
                        result_cache=execution_result_cache,
                        executions_buffer=executions_buffer,
                        normalizers=normalizers,
                        epi_index_dim=epinet_latency_estimation.epi_index_dim,
                        device=gpu_device
                    )

                    for i in range(n_batches_per_train_step):
                        train_loss, estimates_variance = train_step(
                            model=epinet_latency_estimation,
                            optimizer=optimizer,
                            normalizers=normalizers,
                            client_default_timeout=client_default_timeout,
                            executions_buffer=executions_buffer,
                            execution_result_cache=execution_result_cache,
                            train_step_batch_size=train_step_batch_size,
                            n_epi_indexes_train=n_epi_indexes_train,
                            lambda_aux_task=lambda_aux_task,
                            alpha_mlp=alpha_mlp,
                            alpha_ensemble=alpha_ensemble,
                            sigma=sigma,
                            device=gpu_device,
                        )
                        epoch_train_losses.append(train_loss)
                        epoch_latencies.append(avg_latency)

                        for uncertainty_queue in uncertainty_queues:
                            # Move updated weights to the forward pass executor process
                            uncertainty_queue.put(estimates_variance)

                    # Update running average
                    running_avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
                    running_avg_latency = sum(epoch_latencies) / len(epoch_latencies)
                    pbar.set_postfix({
                        "Epoch Avg Loss": f"{running_avg_loss:.4f}",
                        "Epoch Avg Latency": f"{running_avg_latency:.4f}",
                        "Recent Loss": f"{train_loss:.4f}",
                        "Recent Latency": f"{avg_latency:.4f}"
                    })

                    # Move updated weights to the forward pass executor process
                    updated_state_dict = {k: v.cpu().numpy() for k, v in epinet_latency_estimation.state_dict().items()}
                    weights_queue.put(updated_state_dict)

                    queries_since_last_train = 0
                    train_buffer.clear()

    for _ in range(num_workers):
        query_queue.put(None)
    for p in workers:
        p.join()

    inference_queue.put(None)
    gpu_process.join()

    execution_strategy.teardown()
    # async_loop.run_until_complete(client.close())
    # async_loop.close()
    print("Training Complete.")

    #TODO:
    # - Investigate the difference between batch mean and total mean as annealing value for both latency and cost.
    #   So basically a measure of how stable the uncertainty is, where if latency is equally stable it will switch
    #   entirely to latency based exploration

    #TODO: Experiments default
    # - Experiment on same queries

    #TODO:
    # Once that works we test our new generated queries on pretrained on other generated queries to see performance deg
    # then we argue
    # 1. Not sufficient to generate queries
    # 2. Generalization gap
    # 3. We need better query generation and better generalization
    # Proceed to next steps

    #TODO: For generalization we can look into two things
    # 1. MoE in GNN based on graph structural embedding. Or other MoE mechanism. Maybe use epinet to determine
    #  how many experts are needed for a query; when uncertainty is high route to more experts?
    # Actually really cool: Using this MoE: https://gemini.google.com/share/2c059d54fa7e
    # 2. Use epinet uncertainty estimate to generate unseen query templates. We train epinet on the default shapes
    #  then sample using random walks (with restrictions on valid walks and 5% being corrupted walks with no results)
    #  rank the queries based on uncertainty -> top k get entered into the data training pipeline.
    #  Another option is to dynamically merge queries of different shapes. Then pass those through epinet
    #  prevent forgetting in some smart way.
    #  Or we could even do generation through an endpoint and ranking possible next walks by epistemic uncertainty.
    #  so not only guide plan generation but also training data generation through uncertainty estimates


def main_online_estimation_experiment(endpoint_location, queries_location_train, queries_location_val,
                                      rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                                      full_gnn_config, config_ensemble_prior, epinet_index_dim, mlp_dimension,
                                      trained_epinet_location,
                                      alpha_mlp, alpha_ensemble, sigma,
                                      lambda_aux_task,
                                      n_epi_index_train, n_epi_index_val,
                                      beam_width, n_epochs,
                                      samples_per_train,
                                      train_step_batch_size,
                                      n_batches_per_train_step,
                                      n_steps_before_train,
                                      device,
                                      use_ray=False, ray_endpoints=None):
    queries_train, queries_val, execution_strategy, writer = \
        prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                           rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                           use_ray=use_ray, ray_endpoints=ray_endpoints)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    heads_config = {
        'plan_cost': {
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
        'plan_cost': {
            'layer': nn.Linear(5, 1),
        },
        'join_rows': {
            'layer': nn.Linear(5, 1),
        },
        'latency': {
            'layer': nn.Linear(5, 1),
        }
    }

    model_kwargs = {
        "full_gnn_config": full_gnn_config,
        "config_ensemble_prior": config_ensemble_prior,
        "heads_config": heads_config,
        "heads_config_prior": heads_config_prior,
        "epinet_index_dim": epinet_index_dim,
        "mlp_dimension": mlp_dimension,
        "model_weights": trained_epinet_location,
        "strict": False,
        #TODO: Change to false when we have trained epinet
        "cost_only": True,
    }

    main_train(queries_train,
               execution_strategy,
               model_kwargs,
               precomputed_indexes,
               precomputed_masks,
               alpha_mlp=alpha_mlp,
               alpha_ensemble=alpha_ensemble,
               sigma=sigma,
               lambda_aux_task=lambda_aux_task,
               n_epi_indexes_train=n_epi_index_train,
               n_epi_indexes_val=n_epi_index_val,
               beam_width=beam_width,
               n_epochs=n_epochs,
               samples_per_train=samples_per_train,
               train_step_batch_size=train_step_batch_size,
               n_batches_per_train_step=n_batches_per_train_step,
               n_steps_before_train=n_steps_before_train,
               gpu_device=device)


def parameter_train_wrapper():
    n_queries_per_train_batch = 32
    beam_width = 4
    max_plans = 10
    mlp_dimension_full = 64
    n_epochs = 25

    # This should be sufficiently high, as we will use this for reward normalization
    n_steps_before_train = 0
    # Number of samples to obtain before another train step
    samples_per_train = 32
    train_step_batch_size = 128
    n_batches_per_train_step = 2
    # Randomly picked hyperparameters
    lambda_aux_task = .2
    # Hyperparameter results from tuning runs on partial data
    n_epi_indexes_train = 16
    n_epi_indexes_val = 100
    epinet_index_dim = 32
    sigma = 0.60
    alpha_mlp = 0.08
    alpha_ensemble = 0.30
    lr = .0001
    weight_decay = 0.04
    n_epochs = 25

    use_ray = True
    ray_endpoints = [f"http://127.0.0.2:{8888 + i}" for i in range(1)]

    endpoint_location = "http://localhost:8888"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_separate_head.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"
    #TODO: This should be a trained epinet
    trained_cost_model_dir = ("experiments/experiment_outputs/yago_gnce/supervised_epinet_training/"
                              "simulated_cost-24-03-2026-16-36-28")
    best_epinet_dir = find_best_epoch_directory(trained_cost_model_dir, "val_loss_cost_unscaled")
    trained_model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    main_online_estimation_experiment(endpoint_location=endpoint_location,
                                      queries_location_train=queries_location_train,
                                      queries_location_val=queries_location_val,
                                      rdf2vec_vector_location=rdf2vec_vector_location,
                                      occurrences_location=occurrences_location,
                                      tp_cardinality_location=tp_cardinality_location,
                                      full_gnn_config=model_config_emb,
                                      config_ensemble_prior=model_config_prior,
                                      epinet_index_dim=epinet_index_dim,
                                      mlp_dimension=mlp_dimension_full,
                                      trained_epinet_location=trained_model_file,
                                      alpha_mlp=alpha_mlp, alpha_ensemble=alpha_ensemble, sigma=sigma,
                                      lambda_aux_task=lambda_aux_task,
                                      n_epi_index_train=n_epi_indexes_train,
                                      n_epi_index_val=n_epi_indexes_val,
                                      beam_width=beam_width,
                                      n_epochs=n_epochs,
                                      device=device,
                                      samples_per_train=samples_per_train,
                                      train_step_batch_size=train_step_batch_size,
                                      n_batches_per_train_step=n_batches_per_train_step,
                                      n_steps_before_train=n_steps_before_train,
                                      use_ray=use_ray,
                                      ray_endpoints=ray_endpoints,
                                      )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parameter_train_wrapper()
