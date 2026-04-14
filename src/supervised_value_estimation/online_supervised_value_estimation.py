import logging
import os
import queue
from collections import defaultdict
from dataclasses import asdict
import asyncio

import diskcache
import hydra
import numpy as np

import ray
from omegaconf import OmegaConf, DictConfig
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
from src.supervised_value_estimation.agents.EpinetCostEstimatorAgent import EpinetCostEstimatorAgent

from src.supervised_value_estimation.agents.EpinetMultiprocessAgent import EpinetMultiprocessAgent
from src.supervised_value_estimation.loss_functions.hinge_mse_loss import right_censored_hinge_loss
from src.supervised_value_estimation.normalizers.normalizer_exponential_moving_average import \
    NormalizerExponentialMovingAverage
from src.supervised_value_estimation.schedulers.UncertaintyAnnealingScheduler import UncertaintyAnnealingScheduler
from src.supervised_value_estimation.storage.PlanBestPerformanceCache import PlanBestPerformanceCache
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search
from src.supervised_value_estimation.storage.ExecutionReplayBuffer import ExecutionReplayBuffer, ExecutionBufferSamples, \
    ExecutionBufferSamplesWithTargets
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import validate_cached
from src.supervised_value_estimation.utils.utils import tensors_to_numpy
from src.supervised_value_estimation.validation.validation_runner import multiprocess_validate_agent

from src.utils.training_utils.query_loading_utils import prepare_data
from src.utils.training_utils.training_tracking import ExperimentWriter, TrainSummary
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

    def __init__(self,
                 ray_endpoints: list[str],
                 num_workers: int = 4,
                 logging_location=os.path.join("logs","ray_execution_error.log"),
                 debug_location=os.path.join("logs", "ray_execution_debug.log")
                 ):
        if not ray.is_initialized():
            context = ray.init()
            print(context.dashboard_url)

        from src.query_environments.qlever.qlever_multi_execute_ray import MultiEndpointWorker as RayClient
        num_workers = min(num_workers, len(ray_endpoints))

        chunks = [ray_endpoints[i::num_workers] for i in range(num_workers)]
        self.actors = [RayClient.remote(chunk) for chunk in chunks]
        # self.actors = [RayClient.remote(url) for url in ray_endpoints]
        self.pool = ActorPool(self.actors)

        self.local_parser = QLeverOptimizerClient("dummy_endpoint")

        self.query_timeouts = {}
        self.default_timeout = self.local_parser.default_timeout

        self.logger = logging.getLogger('RayExecutionLogger')
        # Set the master logger to the lowest level you intend to capture
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Create a single formatter to reuse
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # Configure the Debug Handler (captures DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug_handler = logging.FileHandler(debug_location)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        self.logger.addHandler(debug_handler)

        # Configure the Crash Handler (captures ONLY ERROR and CRITICAL)
        crash_handler = logging.FileHandler(logging_location)
        crash_handler.setLevel(logging.ERROR)
        crash_handler.setFormatter(formatter)
        self.logger.addHandler(crash_handler)
        if num_workers > len(ray_endpoints):
            self.logger.debug(f"Num_workers: {num_workers} "
                              f"> number of endpoints: {len(ray_endpoints)}"
                              f"clamping number of workers to number of endpoints")

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

            timeout = self.default_timeout
            if query_data.query in self.query_timeouts:
                timeout = self.local_parser.format_latency(self.query_timeouts[query_data.query])

            mem = ray.available_resources().get("memory", None)
            self.logger.debug(
                f"Submitting query | available_ray_memory={mem} | "
                f"timeout={timeout} | query={query_data.query}"
            )

            execution_result = actor.execute_plan.remote(query_payload, join_order=best_plan, parse_local=True,
                                                         timeout=timeout)
            return execution_result

        raw_results = [None] * len(plans)

        plan_iterator = self.pool.map(submit_query, plans)

        for i in range(len(plans)):
            query_data = plans[i]["query"].to_data_list()[0]
            query_string = query_data.query

            try:
                result = next(plan_iterator)

                self.logger.debug(
                    f"[{i}] Raw result received | query={query_string[:120]} | result={result}"
                )

                time_query = result["time_total"]
                if time_query != "0ms":
                    time_in_seconds = self.local_parser.decode_to_seconds(time_query)
                    new_timeout = max(min((time_in_seconds * 2), self.default_timeout_s), 1)
                    self.logger.debug(
                        f"[{i}] Tightening timeout | query={query_string[:120]} | "
                        f"time={time_query} | new_timeout={new_timeout:.2f}s"
                    )
                    self.query_timeouts[query_string] = new_timeout

                raw_results[i] = result

            except ray.exceptions.OutOfMemoryError as e:
                self.logger.error(
                    f"[{i}] OOM on actor | query={query_string[:120]} | error={e}",
                    exc_info=True
                )
                self.logger.debug(
                    f"[{i}] OOM — actor may have been killed | query={query_string[:120]}"
                )

            except ray.exceptions.RayActorError as e:
                # Actor died (OOM kill, segfault, etc.) — Ray will have already restarted it
                self.logger.error(
                    f"[{i}] Actor crashed | query={query_string[:120]} | error={e}",
                    exc_info=True
                )
                self.logger.debug(
                    f"[{i}] RayActorError — worker process died, likely OOM | "
                    f"query={query_string[:120]} | cause={e.__cause__}"
                )

            except ray.exceptions.WorkerCrashedError as e:
                # Worker was forcibly killed (e.g. by the OS OOM killer)
                self.logger.error(
                    f"[{i}] Worker killed by OS | query={query_string[:120]} | error={e}",
                    exc_info=True
                )

            except StopIteration:
                self.logger.error(
                    f"[{i}] Pool exhausted early — likely a prior actor crash killed the iterator"
                )
                break

            except Exception as e:
                self.logger.error(
                    f"[{i}] Unexpected error | query={query_string[:120]} | "
                    f"error={type(e).__name__}: {e}",
                    exc_info=True
                )
        # # pool.map automatically routes tasks to idle actors.
        # # It guarantees the output list matches the order of the input 'plans' list,
        # # ensuring zip() aligns the metrics perfectly with the original items.
        # raw_results = []
        # for i, result in enumerate(tqdm(self.pool.map(submit_query, plans), total=len(plans), desc="Executing Plans")):
        #     # Reconstruct query_data from the original plans list using the index
        #     query_data = plans[i]["query"].to_data_list()[0]
        #     query_string = query_data.query
        #
        #     # Tighten bounds on successful execution
        #     time_query = result["time_total"]
        #     if time_query != "0ms":
        #         time_in_seconds = self.local_parser.decode_to_seconds(time_query)
        #         self.query_timeouts[query_string] = max(min((time_in_seconds * 2), self.default_timeout_s),1)
        #
        #     raw_results.append(result)

        for item, raw_result in zip(plans, raw_results):
            item["rl_metrics"] = raw_result

        return plans

    def teardown(self):
        pass

    @property
    def default_timeout_s(self) -> float:
        return self.local_parser.default_timeout_s


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

    return train_dataset, val_dataset, execution_strategy


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
                            reward_cache: PlanBestPerformanceCache,
                            device):
    batched_samples: ExecutionBufferSamples = execution_buffer.sample(batch_size)
    latencies, total_cost, is_censored = determine_targets_of_sample(batched_samples, reward_cache, device)
    batched_samples_with_targets = ExecutionBufferSamplesWithTargets(
        **asdict(batched_samples),
        latencies=latencies,
        total_cost=total_cost,
        is_censored=is_censored
    )
    return batched_samples_with_targets


def determine_targets_of_sample(batched_sample: ExecutionBufferSamples,
                                reward_cache: PlanBestPerformanceCache,
                                device):
    latencies, total_cost, is_censored = [], [], []
    for i in range(batched_sample.queries.shape[0]):
        reward_info = reward_cache.get_target(batched_sample.join_plans[i], batched_sample.queries[i])
        latencies.append(reward_info["latency"])
        total_cost.append(reward_info["total_cost"])
        is_censored.append(reward_info["is_censored"])
    return (torch.tensor(latencies, dtype=torch.float32, device=device),
            torch.tensor(total_cost, dtype=torch.float32, device=device),
            torch.tensor(is_censored, dtype=torch.bool, device=device))


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
    samples_with_targets = retrieve_buffer_samples(batch_size=train_step_batch_size,
                                                   execution_buffer=executions_buffer,
                                                   reward_cache=execution_result_cache,
                                                   device=device)
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
                                      ~is_censored)

    total_loss = ((1 - lambda_aux_task * 2) * loss_latency + lambda_aux_task * loss_total_cost
                  + lambda_aux_task * loss_intermediate_size)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    per_abs_error = torch.abs(estimated_costs["latency"].flatten() - perturbed_targets["perturbed_latency"])
    mean_per_abs_error = torch.mean(per_abs_error.view(n_epi_indexes_train, -1).T, dim = 1)
    executions_buffer.update_priorities(samples_with_targets.indices, mean_per_abs_error)

    return total_loss.detach().cpu().item(), estimated_variances, {
        "latency": loss_latency,
        "total_cost": loss_total_cost,
        "join_rows": loss_intermediate_size,
    }


def build_validation_agent(model_kwargs, agent_kwargs, state_dict, precomputed_indexes, precomputed_masks, device):
    """Wraps the existing validation agent and syncs the latest training weights."""
    # Remove model weights as this is given by the passed state_dict
    model_kwargs.pop("model_weights")

    epinet = prepare_epinet_model(**model_kwargs, device=device)
    epinet.load_state_dict(state_dict)
    epinet.eval()

    # Reuse your existing agent logic
    agent = EpinetCostEstimatorAgent(
        epinet, precomputed_indexes, precomputed_masks,
        **agent_kwargs,
        head_name="latency"
    )
    return agent


def main_validate(val_loader, val_cache,
                  epinet_latency_estimation,
                  model_kwargs, agent_kwargs,
                  normalizers,
                  loss_fns,
                  execution_strategy,
                  sigma,
                  beam_width, num_workers,
                  precomputed_indexes, precomputed_masks,
                  epoch_total_losses, epoch_latency_losses, epoch_plan_cost_losses,
                  epoch_join_rows_losses, epoch_latencies, blending_weights,
                  train_summary, writer):
    val_cache.clear()
    current_state_dict = tensors_to_numpy(epinet_latency_estimation.state_dict())
    current_state_dict = {k: torch.from_numpy(v) for k, v in current_state_dict.items()}

    epoch_summary = {
        "train_loss_total": np.mean(epoch_total_losses)/len(epoch_total_losses),
        "train_loss_latency": np.mean(epoch_latency_losses)/len(epoch_latency_losses),
        "train_loss_plan_cost": np.mean(epoch_plan_cost_losses)/len(epoch_plan_cost_losses),
        "train_loss_join_rows": np.mean(epoch_join_rows_losses)/len(epoch_join_rows_losses),
        "train_loss_total_during_epoch": epoch_total_losses,
        "train_loss_latency_during_epoch": epoch_latency_losses,
        "train_loss_plan_cost_during_epoch": epoch_plan_cost_losses,
        "train_loss_join_rows_during_epoch": epoch_join_rows_losses,
        "train_latency_during_epoch": epoch_latencies,
        "blending_weights": blending_weights,
    }

    agent_kwargs = {
        "model_kwargs": model_kwargs,
        "agent_kwargs": agent_kwargs,
        "state_dict": current_state_dict,
        "precomputed_indexes": precomputed_indexes,
        "precomputed_masks": precomputed_masks,
        "device": torch.device('cpu')
    }

    search_metrics, execution_results = multiprocess_validate_agent(
        val_loader=val_loader,
        execution_strategy=execution_strategy,
        agent_builder_fn=build_validation_agent,
        agent_kwargs=agent_kwargs,
        beam_width=beam_width,
        num_workers=num_workers,
        samples_per_execution_batch=4
    )
    search_metrics = {k: v for k, v in search_metrics.items() if 'planning' not in str(k)}
    search_metrics.pop("timeout_error_rate")
    search_metrics.pop("total_queries")
    epoch_summary.update(search_metrics)

    # query_plans_val = {}
    # targets = defaultdict(dict)
    # for res in execution_results:
    #     q_id = res["query"].query[0]
    #     best_plan = res["plan"]["plan"] if isinstance(res["plan"], dict) else res["plan"]
    #
    #     metrics = res["rl_metrics"]
    #     is_err = metrics["is_error"]
    #     lat = np.log1p(client_default_timeout if is_err else metrics["latency"])
    #     cost = -1 if is_err else np.log1p(metrics["total_cost"])
    #     cost_join = [-1 if is_err else np.log1p(join_rows) for join_rows in metrics["per_join_rows"]]
    #
    #     query_plans_val[q_id] = [(best_plan, {"latency": lat, "plan_cost": cost}, hash(q_id) % 10000)]
    #     #TODO: These are probably wrong, due to multiple join cost and one plan_cost. How to align?
    #     #TODO: Probably by aligning it with the plans and saying each plan has same cost
    #     targets[q_id]["latency"] = torch.tensor([lat], dtype=torch.float32, device=torch.device('cpu'))
    #     targets[q_id]["plan_cost"] = torch.tensor([cost], dtype=torch.float32, device=torch.device('cpu'))
    #     targets[q_id]["per_join_rows"] = torch.tensor(cost_join, dtype=torch.float32, device=torch.device('cpu'))
    #
    # mean_vals, std_vals, head_tracker = {}, {}, {}
    #
    # for head in ["latency", "plan_cost", "per_join_rows"]:
    #     try:
    #         mean_vals[head] = normalizers[head].mean.item() if normalizers[head].mean is not None else 0.0
    #         std_vals[head] = torch.sqrt(normalizers[head].var).item() if normalizers[head].var is not None else 1.0
    #     except AttributeError:
    #         mean_vals[head], std_vals[head] = 0.0, 1.0
    #
    #     tracker = validate_cached(
    #         val_loader=val_loader, query_plans_val=query_plans_val, targets=targets,
    #         epinet_cost_estimation=epinet_latency_estimation, val_cache=val_cache,
    #         mean_vals=mean_vals, std_vals=std_vals, train_loss=loss_fns[head], device=torch.device('cpu'),
    #         n_val_epi_indexes=agent_kwargs["n_epinet_samples"], sigma=sigma, alpha_mlp=0.0,
    #         alpha_ensemble=0.0, precomputed_indexes=precomputed_indexes,
    #         precomputed_masks=precomputed_masks, head_names_to_val=(head,)
    #     )
    #     # TODO: Add total loss (using mixing values)
    #
    #     epoch_summary.update(tracker.summarize())

    return epoch_summary


def main_train(queries_train,
               queries_val,
               execution_strategy,
               model_kwargs,
               precomputed_indexes, precomputed_masks,
               alpha_mlp, alpha_ensemble, sigma, lambda_aux_task,
               n_epi_indexes_train, n_epi_indexes_val,
               use_per,
               num_workers, beam_width, n_epochs,
               lr, weight_decay,
               samples_per_train, train_step_batch_size, n_batches_per_train_step, n_steps_before_train,
               writer,
               val_cache,
               gpu_device):

    train_summary = TrainSummary([
                                   # ("val_epi_mse_latency", "min"), ("val_epi_mse_latency_scaled", "min"),
                                   # ("val_epi_mse_plan_cost", "min"), ("val_epi_mse_plan_cost_scaled", "min"),
                                   # ("val_epi_avg_std", "min"),
                                   # ("val_joint_gaussian_nll_latency", "min"),
                                   # ("val_joint_gaussian_nll_plan_cost", "min"),
                                   # ("val_joint_nll_latency_no_epinet", "min"),
                                   # ("val_joint_nll_cost_no_epinet", "min"),
                                   # ("val_loss_total_epinet", "min"),
                                   # ("val_loss_latency_epinet", "min"),
                                   # ("val_loss_plan_cost_epinet", "min"),
                                   ("train_loss_total", "min"),
                                   ("train_loss_latency", "min"),
                                   ("train_loss_plan_cost", "min"),
                                   ("train_loss_join_rows", "min"),
                                   ("execution_latency_median_s", "min"),
                                   ("execution_latency_mean_s", "min"),
                                   ("execution_latency_p90_s", "min"),
                                   ("execution_latency_p99_s", "min"),
                                   ("execution_latency_max_s", "min"),
                                   # ("val_calibration_error_latency", "min"), ("val_sharpness_latency", "min"),
                                   # ("val_calibration_error_plan_cost", "min"), ("val_sharpness_plan_cost", "min"),
                                   ("train_latency_during_epoch", "list"),
                                   ("train_loss_total_during_epoch", "list"),
                                   ("train_loss_latency_during_epoch", "list"),
                                   ("train_loss_plan_cost_during_epoch", "list"),
                                   ("train_loss_join_rows_during_epoch", "list"),
                                   ("blending_weights", "list")
                                   ])
    writer.create_experiment_directory()
    execution_strategy.setup()
    client_default_timeout = execution_strategy.default_timeout_s

    execution_result_cache = PlanBestPerformanceCache()
    executions_buffer = ExecutionReplayBuffer(buffer_size=50000,
                                              epi_index_dim=model_kwargs["epinet_index_dim"],
                                              device=gpu_device,
                                              use_per=use_per)

    mp.set_start_method('spawn', force=True)

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

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Apply gradient clipping for stability as this is in essence off-policy RL
    torch.nn.utils.clip_grad_norm_(params, max_norm=.5)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=1e-2)
    previous_lr = scheduler.get_last_lr()

    annealing_scheduler = UncertaintyAnnealingScheduler(ema_alpha=0.05)

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
    blending_weight_queues = [mp.Queue() for _ in range(num_workers)]

    handler_kwargs = {
        # Function that builds the model
        "model_builder_fn": prepare_epinet_model,
        "model_kwargs": model_kwargs,
        "state_dict": shared_state_dict,
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
                  i, query_queue, plan_queue, inference_queue, result_queues[i], blending_weight_queues[i],
                  precomputed_indexes, precomputed_masks,
                  alpha_mlp, alpha_ensemble, beam_width)
        )
        p.start()
        workers.append(p)

    loader = DataLoader(queries_train, batch_size=1, shuffle=True)
    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)

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
        epoch_total_train_losses = []
        epoch_latency_train_losses = []
        epoch_plan_cost_train_losses = []
        epoch_join_rows_train_losses = []

        epoch_latencies = []
        epoch_blending_weights = []

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
                        train_loss, estimates_variance, head_losses = train_step(
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
                        epoch_total_train_losses.append(train_loss)
                        epoch_latency_train_losses.append(head_losses["latency"].cpu().item())
                        epoch_plan_cost_train_losses.append(head_losses["total_cost"].cpu().item())
                        epoch_join_rows_train_losses.append(head_losses["join_rows"].cpu().item())

                        epoch_latencies.append(avg_latency)

                        current_blending_weight = annealing_scheduler.update_and_get_weight(estimates_variance)
                        epoch_blending_weights.append(current_blending_weight)

                        for uncertainty_queue in blending_weight_queues:
                            # Move updated weights to the forward pass executor process
                            uncertainty_queue.put(current_blending_weight)

                    # Update running average
                    running_avg_loss = sum(epoch_total_train_losses) / len(epoch_total_train_losses)
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
                    break

        def dynamic_masked_mse(predictions: torch.Tensor, targets: torch.Tensor):
            valid_mask: torch.Tensor = targets.where(targets != -1, torch.ones_like(targets))
            return masked_mse_loss(predictions, targets, valid_mask)

        # Start validation of query heads
        epoch_summary = main_validate(
            val_loader=val_loader,
            val_cache=val_cache,
            epinet_latency_estimation=epinet_latency_estimation,
            model_kwargs=model_kwargs,
            agent_kwargs={
                "n_epinet_samples": n_epi_indexes_val ,
                "alpha_mlp": alpha_mlp,
                "alpha_ensemble": alpha_ensemble,
            },
            normalizers=normalizers,
            loss_fns={
              "latency": right_censored_hinge_loss,
              "plan_cost": dynamic_masked_mse,
              "per_join_rows": dynamic_masked_mse
            },
            execution_strategy=execution_strategy,
            sigma=sigma,
            beam_width=beam_width,
            num_workers=4,
            precomputed_indexes=precomputed_indexes,
            precomputed_masks=precomputed_masks,
            epoch_total_losses=epoch_total_train_losses,
            epoch_latency_losses=epoch_latency_train_losses,
            epoch_plan_cost_losses=epoch_plan_cost_train_losses,
            epoch_join_rows_losses=epoch_join_rows_train_losses,
            epoch_latencies=epoch_latencies,
            blending_weights=epoch_blending_weights,
            train_summary=train_summary,
            writer=writer,
        )

        train_summary.update(epoch_summary, epoch)
        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file([], best, per_epoch, epinet_latency_estimation, epoch)

    for _ in range(num_workers):
        query_queue.put(None)
    for p in workers:
        p.join()

    inference_queue.put(None)
    gpu_process.join()

    execution_strategy.teardown()
    print("Training Complete.")

    #TODO:
    # - Validate GNCE performance
    # - Figure out why epinet is not training (womp womp)
    # - Run experiment on virtual wall

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
                                      use_per,
                                      num_workers, beam_width, n_epochs,
                                      lr, weight_decay,
                                      samples_per_train,
                                      train_step_batch_size,
                                      n_batches_per_train_step,
                                      n_steps_before_train,
                                      writer,
                                      cache_directory,
                                      device,
                                      use_ray=False, ray_endpoints=None):
    queries_train, queries_val, execution_strategy = \
        prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                           rdf2vec_vector_location, occurrences_location, tp_cardinality_location,
                           use_ray=use_ray, ray_endpoints=ray_endpoints)
    val_cache = diskcache.Cache(os.path.join(cache_directory, "val_cache"), size_limit=50 * 1024 ** 3)

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
        "cost_only": False,
        # Filter any diffs for join_rows and latency as these are new heads
        "diff_filter": r"join_rows|latency"
    }

    main_train(queries_train,
               queries_val,
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
               use_per=use_per,
               num_workers = num_workers,
               beam_width=beam_width,
               n_epochs=n_epochs,
               lr=lr,
               weight_decay=weight_decay,
               samples_per_train=samples_per_train,
               train_step_batch_size=train_step_batch_size,
               n_batches_per_train_step=n_batches_per_train_step,
               n_steps_before_train=n_steps_before_train,
               writer=writer,
               val_cache=val_cache,
               gpu_device=device,
               )


def run_online_estimation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = ExperimentWriter(cfg.models.trained.experiment_base_dir, "online_latency_epinet_training",
                              OmegaConf.to_container(cfg, resolve=True),
                              {})
    cache_directory = os.path.join(cfg.models.trained.experiment_base_dir,
                                   'online_latency_epinet_training',
                                   'prior_cache_dir')


    main_online_estimation_experiment(
        endpoint_location=cfg.dataset.endpoint_location,
        queries_location_train=cfg.dataset.queries_train,
        queries_location_val=cfg.dataset.queries_val,
        rdf2vec_vector_location=cfg.dataset.rdf2vec_vector_location,
        occurrences_location=cfg.dataset.occurrences_location,
        tp_cardinality_location=cfg.dataset.tp_cardinality_location,
        full_gnn_config=cfg.models.embedder.config,
        config_ensemble_prior=cfg.models.epinet.prior_config,
        epinet_index_dim=cfg.hyperparameters.epinet_index_dim,
        mlp_dimension=cfg.hyperparameters.mlp_dimension_full,
        trained_epinet_location=cfg.models.epinet.model_file,
        alpha_mlp=cfg.hyperparameters.alpha_mlp,
        alpha_ensemble=cfg.hyperparameters.alpha_ensemble,
        sigma=cfg.hyperparameters.sigma,
        lambda_aux_task=cfg.hyperparameters.lambda_aux_task,
        n_epi_index_train=cfg.hyperparameters.n_epi_indexes_train,
        n_epi_index_val=cfg.hyperparameters.n_epi_indexes_val,
        use_per=cfg.hyperparameters.use_per,
        num_workers=cfg.hyperparameters.num_workers,
        beam_width=cfg.hyperparameters.beam_width,
        n_epochs=cfg.hyperparameters.n_epochs,
        lr=cfg.hyperparameters.lr,
        weight_decay=cfg.hyperparameters.weight_decay,
        device=device,
        samples_per_train=cfg.hyperparameters.samples_per_train,
        train_step_batch_size=cfg.hyperparameters.train_step_batch_size,
        n_batches_per_train_step=cfg.hyperparameters.n_batches_per_train_step,
        n_steps_before_train=cfg.hyperparameters.n_steps_before_train,
        writer=writer,
        cache_directory = cache_directory,
        use_ray=cfg.execution.use_ray,
        ray_endpoints=list(cfg.execution.ray_endpoints),
    )


@hydra.main(version_base=None,
            config_path="../../experiments/experiment_configs/online_cost_latency_estimation",
            config_name="online_supervised_latency_cost_estimation_epinet_distributed_virtual_wall.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    best_epinet_dir = find_best_epoch_directory(cfg.models.epinet.experiment_dir, "val_loss_cost_unscaled")

    cfg.models.epinet.dir = str(best_epinet_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    OmegaConf.set_struct(cfg, True)

    run_online_estimation(cfg)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()