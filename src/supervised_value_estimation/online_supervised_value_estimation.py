import asyncio
import os

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader

from main import find_best_epoch_directory
from src.models.epistemic_neural_network import prepare_epinet_model

from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient

from src.supervised_value_estimation.agents.EpinetMultiprocessAgent import EpinetMultiprocessAgent
from src.supervised_value_estimation.storage.PlanBestPerformanceCache import PlanBestPerformanceCache
from src.supervised_value_estimation.search_algorithms.beam_search import beam_search
from src.supervised_value_estimation.storage.ExecutionReplayBuffer import ExecutionReplayBuffer, ExecutionBufferSamples, \
    ExecutionBufferSamplesWithTargets

from src.utils.epinet_utils.disk_cache_frozen_representations import DiskCacheFrozenRepresentations
from src.utils.training_utils.query_loading_utils import prepare_data
from src.utils.training_utils.training_tracking import ExperimentWriter

from tqdm import tqdm

from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask

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
                payload["prep_trees"], payload["prep_idx"], payload["prep_masks"], cost_name='plan_cost'
            )

            mlp_prior = self.model.compute_mlp_prior(last_feature, z_gpu)
            learnable_mlp = self.model.compute_learnable_mlp(last_feature, z_gpu)

            return {
                "est_cost": est_cost.cpu(),
                "mlp_prior": mlp_prior.cpu(),
                "learnable_mlp": learnable_mlp.cpu()
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
        best_plan = item["plan"]["plan"]

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

def update_best_execution_cache_and_buffer(execution_result, result_cache: PlanBestPerformanceCache,
                                           executions_buffer: ExecutionReplayBuffer, epi_index_dim: int):
    # Get execution results, decompose into partial plans
    for execution in execution_result:
        # Convert pytorch geometric DataBatch to list, take first (and only) element and the string representation
        # of the query
        query = execution["query"].to_data_list()[0].query
        latency = execution["rl_metrics"]["latency"]
        total_cost = execution["rl_metrics"]["total_cost"]
        # History contains all partial plans and their state (including the full plan)
        history = execution["plan"]["history"]
        for i, plan in enumerate(history):

            join_plan = plan[0]
            prepared_trees = plan[1]["prepared_trees"]
            prepared_idx = plan[1]["prepared_idx"]
            prepared_masks = plan[1]["prepared_masks"]
            unweighted_ensemble_prior = plan[1]["unweighted_ensemble_prior"]
            episode_z = plan[1]["z"]
            intermediate_join_size = execution["rl_metrics"]["per_join_rows"][i]
            is_valid_size = execution["rl_metrics"]["is_valid_join_row"][i]

            # one vector for both latency and total cost estimation.
            # Perturbs single target, so shape (2,1,epi_index_dim)
            c_vectors_observation = torch.randn((2, 1, epi_index_dim), device=torch.device('cpu'))
            c_vectors_observation = torch.nn.functional.normalize(c_vectors_observation, dim=1).numpy()

            # Each subplan gets updated by this execution. Will only update if latency is better than previous
            # recorded latency for that subplan
            result_cache.add_execution(plan[0], query, latency, total_cost)

            # Record state in execution buffer. Only store the intermediate join size as target, as the other
            # targets are dynamically obtained from result cache
            executions_buffer.add(
                query_string=query,
                join_plan=join_plan, prepared_trees=prepared_trees, prepared_idx=prepared_idx,
                prepared_masks=prepared_masks, unweighted_ensemble_prior=unweighted_ensemble_prior,
                episode_z=episode_z, intermediate_join_size=intermediate_join_size,
                is_valid_size=is_valid_size, c_vectors_observation=c_vectors_observation)
    return

def retrieve_buffer_samples(batch_size,
                            execution_buffer: ExecutionReplayBuffer,
                            reward_cache: PlanBestPerformanceCache):
    batched_samples: ExecutionBufferSamples = execution_buffer.sample(batch_size)
    latencies, total_cost = determine_targets_of_sample(batched_samples, reward_cache)
    batched_samples_with_targets = ExecutionBufferSamplesWithTargets(
        **batched_samples._asdict(),
        latencies = latencies,
        total_cost = total_cost
    )
    return batched_samples_with_targets

def determine_targets_of_sample(batched_sample: ExecutionBufferSamples,
                                reward_cache: PlanBestPerformanceCache):
    latencies, total_cost = [], []
    for i in range(batched_sample.queries.shape[0]):
        reward_info = reward_cache.get_target(batched_sample.join_plans[i], batched_sample.queries[i])
        latencies.append(reward_info["latency"])
        total_cost.append(reward_info["total_cost"])
    latencies = torch.tensor(latencies)
    total_cost = torch.tensor(total_cost)
    return latencies, total_cost

def perturb_reward_signals(sample_with_targets: ExecutionBufferSamplesWithTargets, epinet_indexes, sigma, device):
    latency = sample_with_targets.latencies
    latency_c_vector = sample_with_targets.c_vectors[0]
    perturbed_latency = perturb_vector(latency, latency_c_vector, epinet_indexes, sigma)

    total_cost = sample_with_targets.latencies
    total_cost_c_vector = sample_with_targets.c_vectors[0]
    perturbed_total_cost = perturb_vector(total_cost, total_cost_c_vector, epinet_indexes, sigma)
    return {
        "perturbed_latency": perturbed_latency,
        "perturbed_total_cost": perturbed_total_cost,
    }

def perturb_vector(raw_targets, c_vectors, epinet_indexes, sigma):
    anchor_matrix = torch.matmul(epinet_indexes, c_vectors.T)
    anchor_term_flat = anchor_matrix.view(-1)

    raw_targets_exp = raw_targets.repeat(epinet_indexes.shape[0])
    return raw_targets_exp + sigma * anchor_term_flat

def estimate_cost(epinet_latency_estimation, sample_with_targets: ExecutionBufferSamples, epinet_indexes,
                  alpha_mlp, alpha_ensemble):
    heads_output, last_feature = \
        (epinet_latency_estimation.
         estimate_cost_from_prepared_all_heads(prepared_trees=sample_with_targets.prepared_trees,
                                               prepared_indexes=sample_with_targets.prepared_idx,
                                               prepared_masks=sample_with_targets.prepared_masks))

    ensemble_prior = torch.matmul(epinet_indexes, sample_with_targets.unweighted_ensemble_priors)

    ensemble_prior_flat = ensemble_prior.view(-1, 1)

    mlp_prior = epinet_latency_estimation.compute_mlp_prior_batched(last_feature, epinet_indexes)
    learnable_mlp_prior = epinet_latency_estimation.compute_learnable_mlp_batched(last_feature, epinet_indexes)

    estimated_cost_exp = estimated_cost.repeat(n_epi_indexes, 1)
    epinet_estimated_cost = estimated_cost_exp + (
            learnable_mlp_prior + alpha_mlp * mlp_prior + alpha_ensemble * ensemble_prior_flat
    )
    pass


def train_step(model, optimizer, losses):
    pass

def main_train(queries_train,
               client,
               model_kwargs,
               disk_cache,
               precomputed_indexes, precomputed_masks,
               alpha_mlp, alpha_ensemble, sigma,
               n_epi_indexes_train, n_epi_indexes_val,
               beam_width, n_epochs,
               train_step_batch_size, n_batches_per_train_step, n_steps_before_train,
               gpu_device):
    execution_result_cache = PlanBestPerformanceCache()
    executions_buffer = ExecutionReplayBuffer(buffer_size = 50000,
                                              epi_index_dim=model_kwargs["epinet_index_dim"],
                                              device = gpu_device)
    mp.set_start_method('spawn', force=True)
    num_workers = 1

    # Create the model here once and obtain the state_dict to pass to the workers
    epinet_latency_estimation = prepare_epinet_model(**model_kwargs, device=gpu_device)
    shared_state_dict = epinet_latency_estimation.state_dict()

    # create the queue passing query to cpu workers
    query_queue = mp.Queue()
    # create the queue passing complete query plans to main worker
    plan_queue = mp.Queue()
    # create queue passing computed plan representations to GPU for forward pass
    inference_queue = mp.Queue()
    # create queues passing forward pass result to workers
    result_queues = [mp.Queue() for _ in range(num_workers)]
    #TODO Pass weights back to the gpu worker as this is the only trainable worker that does forward passes.
    # cpu workers mainly do ensemble calculation and preparing trainable model inputs

    # create queue passing updated weights to the CPU and GPU workers
    weights_queue = mp.Queue()

    handler_kwargs = {
        # Function that builds the model
        "model_builder_fn": prepare_epinet_model,
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
            args=(prepare_epinet_model, model_kwargs, shared_state_dict,
                  i, query_queue, plan_queue, inference_queue, result_queues[i],
                  disk_cache, precomputed_indexes, precomputed_masks,
                  alpha_mlp, alpha_ensemble, beam_width)
        )
        p.start()
        workers.append(p)

    loader = DataLoader(queries_train, batch_size=1, shuffle=True)

    for epoch in range(n_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{n_epochs} ---")

        #TODO: This should only load enough queries such that a next train can be initialized.
        # this is to prevent out of date weights
        temp = 0
        for query in loader:
            if temp == 100:
                break
            query_queue.put(query)
            temp += 1
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

                if queries_since_last_train >= 32 and completed_queries > n_steps_before_train:
                    print("\n[Triggering Model Training...]")
                    execution_results = execute_plans(client, train_buffer, 1)
                    update_best_execution_cache_and_buffer(execution_results,
                                                           result_cache=execution_result_cache,
                                                           executions_buffer=executions_buffer,
                                                           epi_index_dim=epinet_latency_estimation.epi_index_dim)
                    for i in range(n_batches_per_train_step):
                        samples_with_targets = retrieve_buffer_samples(batch_size=train_step_batch_size,
                                                                       execution_buffer=executions_buffer,
                                                                       reward_cache=execution_result_cache)
                        epinet_indexes = epinet_latency_estimation.sample_epistemic_indexes_batched(n_epi_indexes_train)
                        perturbed_targets = perturb_reward_signals(samples_with_targets, epinet_indexes,
                                                                   sigma, gpu_device)
                        estimated_costs = estimate_cost(epinet_latency_estimation, samples_with_targets, epinet_indexes)

                        latency_loss_value = latency_loss(estimated_costs, perturbed_targets)
                        total_cost_value = total_cost_loss(estimated_costs, perturbed_targets)

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
                                      n_epi_index_train, n_epi_index_val,
                                      beam_width, n_epochs,
                                      train_step_batch_size,
                                      n_batches_per_train_step,
                                      n_steps_before_train,
                                      device):
    queries_train, queries_val, client, writer =\
        prepare_experiment(endpoint_location, queries_location_train, queries_location_val,
                           rdf2vec_vector_location, occurrences_location, tp_cardinality_location)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)

    embedded_query_cache = DiskCacheFrozenRepresentations('frozen_query_embeddings.h5')
    query_plan_cache = DiskCacheFrozenRepresentations('frozen_query_plans.h5')

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
    }

    main_train(queries_train,
               client,
               model_kwargs,
               embedded_query_cache,
               precomputed_indexes,
               precomputed_masks,
               alpha_mlp=alpha_mlp,
               alpha_ensemble=alpha_ensemble,
               sigma=sigma,
               n_epi_indexes_train=n_epi_index_train,
               n_epi_indexes_val=n_epi_index_val,
               beam_width=beam_width,
               n_epochs=n_epochs,
               train_step_batch_size = train_step_batch_size,
               n_batches_per_train_step = n_batches_per_train_step,
               n_steps_before_train = n_steps_before_train,
               gpu_device=device)

def parameter_train_wrapper():
    n_queries_per_train_batch = 32
    beam_width = 4
    max_plans = 10
    mlp_dimension_full = 64
    n_epochs = 25

    # This should be sufficiently high, as we will use this for reward normalization
    n_steps_before_train = 2500
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

    endpoint_location = "http://localhost:8888"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_separate_head.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_smallest.yaml"
    trained_cost_model_dir = ("experiments/experiment_outputs/yago_gnce/supervised_epinet_training/"
                          "simulated_cost-03-03-2026-08-56-11")
    best_epinet_dir = find_best_epoch_directory(trained_cost_model_dir, "val_loss_cost_unscaled")
    trained_model_file = str(os.path.join(best_epinet_dir, "epinet_model.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                      n_epi_index_train=n_epi_indexes_train,
                                      n_epi_index_val=n_epi_indexes_val,
                                      beam_width=beam_width,
                                      n_epochs=n_epochs,
                                      device=device,
                                      train_step_batch_size=train_step_batch_size,
                                      n_batches_per_train_step=n_batches_per_train_step,
                                      n_steps_before_train=n_steps_before_train,
                                      )

if __name__ == '__main__':
    parameter_train_wrapper()
