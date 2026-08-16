import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import concurrent.futures
from threading import Lock

from functools import partial

import sklearn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming


def min_max_scale_plans(query_plans, min_cost=None, max_cost=None):
    if not min_cost or not max_cost:
        min_cost = math.inf
        max_cost = -math.inf
        for query, plans in query_plans.items():
            for (order, cost) in plans:
                if min_cost > cost:
                    min_cost = cost
                if max_cost < cost:
                    max_cost = cost

    range_cost = max_cost - min_cost
    for query, plans in query_plans.items():
        for i in range(len(plans)):
            order, cost = plans[i]
            scaled_cost = (cost - min_cost) / range_cost
            plans[i] = (order, scaled_cost)

    return query_plans, min_cost, max_cost


def normalize_plans(query_plans, mean_cost=None, std_cost=None):
    if not mean_cost or not std_cost:
        costs = [plan[1] for plans in query_plans.values() for plan in plans]
        mean_cost = np.mean(costs, keepdims=False).item()
        std_cost = np.std(costs, keepdims=False).item()

    for query, plans in query_plans.items():
        for i in range(len(plans)):
            order, cost = plans[i]
            scaled_cost = (cost - mean_cost) / std_cost
            plans[i] = (order, scaled_cost)

    return query_plans, mean_cost, std_cost

def flatten_plans(plans):
    flattened_plans = {}
    for (query, query_plans) in plans.items():
        plan_cost_tuples = []

        for sub_plans in query_plans:
            cost = sub_plans[1]
            for sub_plan in sub_plans[0]:
                plan_cost_tuples.append((sub_plan, cost))

        flattened_plans[query] = plan_cost_tuples
    return flattened_plans


def preprocess_plans(plans, mean_cost=None, std_cost=None):
    plans_flattened = flatten_plans(plans)
    plans_standardized, mean_cost, std_cost = normalize_plans(plans_flattened, mean_cost, std_cost)
    plans_deduped = dedupe_plans(plans_standardized)
    return attach_unique_id_to_plan(plans_deduped), mean_cost, std_cost

def dedupe_plans(plans):
    deduped = {}

    for (query, query_plans) in plans.items():
        unique_plans = []
        seen = set()

        for plan in query_plans:
            order = tuple(plan[0])
            if order not in seen:
                seen.add(order)
                unique_plans.append(plan)

        deduped[query] = unique_plans
        
    return deduped

def attach_unique_id_to_plan(simulated_query_plans):
    data_index = 0
    for query in sorted(simulated_query_plans.keys()):
        plans = simulated_query_plans[query]
        new_plans = []
        for i, plan in enumerate(plans):
            plan_with_anchor = (*plan, data_index)
            new_plans.append(plan_with_anchor)
            data_index += 1
        simulated_query_plans[query] = new_plans
    return simulated_query_plans


def _process_single_query(query, oracle_model, max_plans_per_relation, estimate_is_log, device, output_loc_raw,
                          write_lock):
    """Processes a single query and writes the generated plans safely to disk."""
    batch_query = query.batch

    un_batched_query = query[0]
    un_batched_query.batch = batch_query
    un_batched_query.to(device)

    # Subsample plans to prevent combinatorial explosion
    plans = sample_orders(
        un_batched_query,
        oracle_model,
        max_plans_per_relation,
        estimate_is_log=estimate_is_log,
        device=device
    )
    orders = [plan.get_order() for plan in plans]

    best_plan_per_sub_plan = {}

    for k, order in enumerate(orders):
        cost = plans[k].cost
        sub = []
        for step in order[:-1]:
            sub.append(step)
            if len(sub) < 2:
                continue
            sub_order = tuple(sub)
            prev = best_plan_per_sub_plan.get(sub_order)
            if prev is None or cost < prev[0]:
                best_plan_per_sub_plan[sub_order] = (cost, k)

    # Group sub plans under their best full plan, applying logarithmic scaling to the cost
    plan_to_sub_plans = [([plan.get_order()], math.log(plan.cost)) for plan in plans]
    for sub_order, (_, k) in best_plan_per_sub_plan.items():
        plan_to_sub_plans[k][0].append(sub_order)

    # Ensure thread-safe file I/O
    with write_lock:
        if len(plan_to_sub_plans) == 0:
            print(f"WARNING: No plans for for query: {un_batched_query.query}")
        write_plans_to_file({un_batched_query.query: plan_to_sub_plans}, output_loc_raw)

#
# def prepare_simulated_dataset(dataset_to_prepare, oracle_model, device,
#                               output_loc_raw,
#                               estimate_is_log=True,
#                               max_plans_per_relation=50,
#                               max_workers=4,
#                               debug_single_batch=False,
#                               query_batch_size=None):
#
#     if debug_single_batch and query_batch_size is None:
#         raise ValueError("query_batch_size must be provided when debug_single_batch is True.")
#     oracle_model.to(device)
#
#     data = []
#     queries_loaded = set()
#     if os.path.exists(output_loc_raw + '.jsonl'):
#         k = 0
#         with open(output_loc_raw + '.jsonl', 'r', encoding="utf-8") as f:
#             for line in f:
#                 query_to_plans = json.loads(line)
#                 query_string = next(iter(query_to_plans.keys()))
#
#                 if query_string not in queries_loaded:
#                     data.append(query_to_plans)
#                     queries_loaded.add(query_string)
#                     k += 1
#
#                 if debug_single_batch and len(data) >= query_batch_size:
#                     break
#     print(f"Loaded {len(data)}/{len(dataset_to_prepare)} plan entries ({len(queries_loaded)} unique queries)")
#
#     if len(dataset_to_prepare) == len(data) or debug_single_batch:
#         return data
#
#     loader = DataLoader(dataset_to_prepare, batch_size=1, shuffle=False)
#     write_lock = Lock()
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for query in tqdm(loader):
#             # Get plans for queries not in (partially) loaded dataset
#             if query.query[0] not in queries_loaded:
#                 futures.append(
#                     executor.submit(
#                         _process_single_query,
#                         query,
#                         oracle_model,
#                         max_plans_per_relation,
#                         estimate_is_log,
#                         device,
#                         output_loc_raw,
#                         write_lock
#                     )
#                 )
#
#         # Iterate over futures as they complete to maintain the progress bar and catch exceptions
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             future.result()
#
#     return data
#

def prepare_simulated_dataset(
    dataset_to_prepare,
    oracle_model,
    device,
    output_loc_raw: str,
    estimate_is_log: bool = True,
    max_plans_per_relation: int = 50,
    debug_single_batch: bool = False,
    query_batch_size: int | None = None,
) -> list[dict]:
    """Build (or resume) a simulated dataset of join-order plans, one query at a time.

    Dataset contains: [(Query, [(plans, cost for plans), ...])]
    Training is done over batches of Queries, to amortize the query embedding step.
    For each sub plan we create a tree-based embedding. A different model can be used
    for estimating cost vs. estimating the plan (plan estimation smaller, big model
    can act as oracle).
    """
    if debug_single_batch and query_batch_size is None:
        raise ValueError("query_batch_size must be provided when debug_single_batch is True.")

    oracle_model.to(device)

    data: list[dict] = []
    queries_loaded: set[str] = set()

    output_path = Path(f"{output_loc_raw}.jsonl")
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                query_to_plans = json.loads(line)
                query_string = next(iter(query_to_plans))

                if query_string not in queries_loaded:
                    data.append(query_to_plans)
                    queries_loaded.add(query_string)

                if debug_single_batch and len(data) >= query_batch_size:
                    break

    print(f"Loaded {len(data)}/{len(dataset_to_prepare)} plan entries ({len(queries_loaded)} unique queries)")

    if len(dataset_to_prepare) == len(data) or debug_single_batch:
        return data

    loader = DataLoader(dataset_to_prepare, batch_size=1, shuffle=False)

    for query in tqdm(loader):
        if query.query[0] in queries_loaded:
            continue

        # Hack workaround due to the batch attribute disappearing when unbatching the
        # data. Although it is technically not needed, the GNN expects it, so we
        # reattach it.
        batch_query = query.batch
        un_batched_query = query[0]
        un_batched_query.batch = batch_query
        un_batched_query.to(device)

        # To prevent combinatorial explosion, we subsample the plans
        plans = sample_orders(
            un_batched_query,
            oracle_model,
            max_plans_per_relation,
            estimate_is_log=estimate_is_log,
            device=device,
        )
        orders = [plan.get_order() for plan in plans]

        best_plan_per_sub_plan: dict[tuple, tuple[float, int]] = {}

        for k, order in enumerate(orders):
            cost = plans[k].cost
            # Build tuple incrementally (no slicing)
            sub = []
            for step in order[:-1]:
                sub.append(step)
                if len(sub) < 2:
                    continue
                sub_order = tuple(sub)
                prev = best_plan_per_sub_plan.get(sub_order)
                if prev is None or cost < prev[0]:
                    best_plan_per_sub_plan[sub_order] = (cost, k)

        # Group sub plans under their best full plan
        plan_to_sub_plans = [([plan.get_order()], math.log(plan.cost)) for plan in plans]
        for sub_order, (_, k) in best_plan_per_sub_plan.items():
            plan_to_sub_plans[k][0].append(sub_order)

        entry = {un_batched_query.query: plan_to_sub_plans}

        if len(plan_to_sub_plans) == 0:
            print(f"WARNING: No plans for for query: {un_batched_query.query}")

        write_plans_to_file(entry, output_loc_raw)

        # Keep the in-memory result consistent with what's on disk
        queries_loaded.add(un_batched_query.query)

    return data


def sample_orders(query, model, max_samples_per_relation, estimate_is_log, device):
    bound_predict_partial = partial(
        OrderDynamicProgramming.predict_cardinality,
        model,
        query,
        device = device
    )
    bound_predict = lambda join_order: bound_predict_partial(list(join_order), len(join_order))
    adjacency_list = build_adj_list(query)

    plans = (JoinOrderEnumerator(adjacency_list, bound_predict, len(query.triple_patterns), estimate_is_log)
            .sample_left_deep_plans(max_samples_per_relation))
    if len(plans) == 0:
        query_text = query.query
        test = 5
    return plans


def write_plans_to_file(data_point, loc):
    data_file = loc + ".jsonl"

    with open(data_file, 'a', encoding="utf-8") as df:
        df.write(json.dumps(data_point) + "\n")
