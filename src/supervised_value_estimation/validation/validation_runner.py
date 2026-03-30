import queue
import time
from collections import defaultdict

import torch
import torch.multiprocessing as mp

from torch_geometric.data import Data, Batch
from tqdm import tqdm
import numpy as np

from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search
from src.supervised_value_estimation.utils.utils import tensors_to_numpy


def cpu_search_worker(query_queue, plan_queue,
                      agent_builder_fn, agent_kwargs, beam_width):
    """
    Generic worker that instantiates an agent and runs beam search.
    """
    # Restrict worker to 1 PyTorch thread to prevent CPU contention
    torch.set_num_threads(1)

    # Initialize the specific agent using the provided factory function.
    # We pass the multiprocessing queues and worker ID explicitly here.
    agent = agent_builder_fn(
        **agent_kwargs
    )

    while True:
        safe_query = query_queue.get()
        # Poison pill to shut down
        if safe_query is None:
            break

        # Reconstruct PyTorch tensors
        query_tensors = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in safe_query.items()
        }

        # Rebuild PyG Batch
        query_data = Data.from_dict(query_tensors)
        query = Batch.from_data_list([query_data])

        start_time = time.perf_counter()
        top_k_plans = beam_search(query, agent, beam_width)
        planning_time = time.perf_counter() - start_time

        plan_queue.put({
            "query": safe_query,
            "top_k_plans": top_k_plans,
            "planning_time": planning_time
        })


def multiprocess_validate_agent(
        val_loader,
        execution_strategy,
        agent_builder_fn,
        agent_kwargs,
        beam_width=4,
        num_workers=4,
        samples_per_execution_batch=8,
        display_metrics = False,
):
    """
    Executes a multiprocess validation pipeline for a generic search agent.
    """
    mp.set_start_method('spawn', force=True)

    query_queue = mp.Queue()
    plan_queue = mp.Queue()

    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=cpu_search_worker,
            args=(query_queue, plan_queue, agent_builder_fn, agent_kwargs, beam_width)
        )
        p.start()
        workers.append(p)

    loader_iter = iter(val_loader)
    completed_queries = 0
    queries_in_flight = 0
    max_in_flight = samples_per_execution_batch + (num_workers * beam_width)

    execution_plans = []
    execution_results = []
    planning_times = []

    print(f"\n--- Starting Multiprocess Validation ({num_workers} workers) ---")

    with tqdm(total=len(val_loader), desc="Validating") as pbar:
        while completed_queries < len(val_loader):
            while queries_in_flight < max_in_flight:
                try:
                    query = next(loader_iter)
                    # Convert to numpy for safe multiprocessing transport
                    safe_query = tensors_to_numpy(query.to_data_list()[0].to_dict())
                    query_queue.put(safe_query)
                    queries_in_flight += 1
                except StopIteration:
                    break

            result = plan_queue.get()
            queries_in_flight -= 1
            planning_times.append(result["planning_time"])

            if result['top_k_plans']:
                best_plan = result['top_k_plans'][0]

                # Reconstruct PyG batch
                query_tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in
                                 result["query"].items()}

                query_data = Data.from_dict(query_tensors)
                reconstructed_query = Batch.from_data_list([query_data])
                execution_plans.append({"query": reconstructed_query, "plan": best_plan})

            completed_queries += 1
            pbar.update(1)

            if len(execution_plans) >= samples_per_execution_batch or completed_queries == len(val_loader):
                # Abstracted execution handles Ray vs Async automatically
                batch_results = execution_strategy.execute(execution_plans)
                execution_results.extend(batch_results)
                execution_plans.clear()
                break
    for _ in range(num_workers):
        query_queue.put(None)

    # Drain the results queue to unblock workers waiting to .put()
    while queries_in_flight > 0:
        try:
            # Timeout prevents infinite hangs if a worker crashed
            plan_queue.get(timeout=5.0)
            queries_in_flight -= 1
        except queue.Empty:
            print("\nWarning: Queue empty before all in-flight queries were drained. A worker may have crashed.")
            break

    for p in workers:
        p.join()

    # Pass the strategy's timeout dynamically
    metrics = compile_validation_metrics(execution_results, planning_times, execution_strategy.default_timeout_s)
    if display_metrics:
        display_validation_metrics(metrics)

    # Return BOTH metrics and raw results for Epinet calibration
    return metrics, execution_results

def compile_validation_metrics(execution_results, planning_times, timeout_val):
    latencies = []
    total_costs = []
    errors = 0

    for res in execution_results:
        metrics = res["rl_metrics"]
        if metrics["is_error"]:
            errors += 1
            latencies.append(timeout_val)
        else:
            latencies.append(metrics["latency"])
            total_costs.append(metrics["total_cost"])

    latencies = np.array(latencies)
    planning_times = np.array(planning_times)
    total_queries = len(execution_results)

    return {
        "planning_time_mean_ms": np.mean(planning_times) * 1000 if len(planning_times) else 0,
        "planning_time_median_ms": np.median(planning_times) * 1000 if len(planning_times) else 0,
        "planning_time_p90_ms": np.percentile(planning_times, 90) * 1000 if len(planning_times) else 0,
        "planning_time_p99_ms": np.percentile(planning_times, 99) * 1000 if len(planning_times) else 0,
        "planning_time_max_ms": np.max(planning_times) * 1000 if len(planning_times) else 0,
        "execution_latency_mean_s": np.mean(latencies) if len(latencies) else 0,
        "execution_latency_median_s": np.median(latencies) if len(latencies) else 0,
        "execution_latency_p90_s": np.percentile(latencies, 90) if len(latencies) else 0,
        "execution_latency_p99_s": np.percentile(latencies, 99) if len(latencies) else 0,
        "execution_latency_max_s": np.max(latencies) if len(latencies) else 0,
        "timeout_error_rate": errors / total_queries if total_queries > 0 else 0.0,
        "total_queries": total_queries
    }


def run_multiple_validations(
        n_runs,
        val_loader,
        execution_strategy,
        agent_builder_fn,
        agent_kwargs,
        beam_width=4,
        num_workers=4,
        samples_per_execution_batch=32,
):
    """Executes the validation pipeline multiple times and computes aggregated metrics."""
    all_metrics = defaultdict(list)

    for run in range(n_runs):
        print(f"\n" + "="*50)
        print(f"STARTING VALIDATION RUN {run + 1}/{n_runs}")
        print("="*50)

        metrics, _ = multiprocess_validate_agent(
            val_loader=val_loader,
            execution_strategy=execution_strategy,
            agent_builder_fn=agent_builder_fn,
            agent_kwargs=agent_kwargs,
            beam_width=beam_width,
            num_workers=num_workers,
            samples_per_execution_batch=samples_per_execution_batch,
        )

        for key, value in metrics.items():
            all_metrics[key].append(value)

    aggregated_results = {}
    for key, values in all_metrics.items():
        if key == "total_queries":
            aggregated_results[key] = values[0]
        else:
            aggregated_results[key] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }

    display_aggregated_metrics(aggregated_results, n_runs)
    return aggregated_results


def display_aggregated_metrics(aggregated_results, n_runs):
    """Prints the mean and standard deviation for all continuous metrics."""
    print("\n" + "="*55)
    print(f"AGGREGATED VALIDATION METRICS ({n_runs} RUNS)")
    print("="*55)
    for key, stats in aggregated_results.items():
        if isinstance(stats, dict):
            print(f"{key:<30}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            print(f"{key:<30}: {stats}")
    print("="*55)


def display_validation_metrics(metrics):
    print("\n" + "="*45)
    print("VALIDATION METRICS")
    print("="*45)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<30}: {value:.4f}")
        else:
            print(f"{key:<30}: {value}")
    print("="*45)