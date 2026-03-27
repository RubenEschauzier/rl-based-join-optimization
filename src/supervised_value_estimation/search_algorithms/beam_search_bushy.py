import re
from typing import List, Tuple, Any

from src.supervised_value_estimation.typings.typings import Plan, Candidate, History, EnvState
from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent


def beam_search(query: Any, agent: AbstractCostAgent, beam_width: int) -> list[dict[str, Any]]:
    # Get all variables per triple pattern to filter out cross-products
    tp_query = query.triple_patterns[0]
    variables_query = []
    for tp in tp_query:
        variable_pattern = r"[\?\$]\w+"
        variables_tp = set(re.findall(variable_pattern, tp))
        variables_query.append(variables_tp)

    n_tp_query = len(query.triple_patterns[0])
    query_state = agent.setup_episode(query)

    # Edge case: no joins needed
    if n_tp_query == 1:
        return [dict(plan=0, cost=0.0, history=[])]

    # State representation for Bushy Search:
    # A forest is a list of independent trees.
    # Each element is a tuple: (tree_structure, provided_variables)
    # Initially, the forest is just the isolated base relations.
    initial_forest = [(i, variables_query[i]) for i in range(n_tp_query)]

    # current_plans stores tuples of (forest, history, total_accumulated_cost)
    current_plans = [(initial_forest, [], 0.0)]

    top_candidates = []

    # A full plan requires N - 1 joins. Each iteration reduces the forest size by 1.
    for depth in range(n_tp_query - 1):
        possible_next = []
        candidate_metadata = []

        for forest, history, total_cost in current_plans:
            n_trees = len(forest)

            # Try all valid pairs of trees in the current forest
            for i in range(n_trees):
                for j in range(n_trees):
                    if i == j:
                        continue

                    u_tree, u_vars = forest[i]
                    v_tree, v_vars = forest[j]

                    # Prevent cartesian joins (cross products)
                    if len(u_vars.intersection(v_vars)) == 0:
                        continue

                    # The order of the first two entries does not matter in query optimization.
                    # Break symmetry for base relations to avoid duplicate topologies like (0, 1) and (1, 0).
                    if isinstance(u_tree, int) and isinstance(v_tree, int) and u_tree > v_tree:
                        continue

                    new_tree = (u_tree, v_tree)
                    new_vars = u_vars.union(v_vars)

                    # Build the new forest by removing the joined trees and adding the combined tree
                    new_forest_base = [forest[k] for k in range(n_trees) if k != i and k != j]

                    possible_next.append(new_tree)

                    # Store metadata to reconstruct the full state after agent evaluation
                    candidate_metadata.append((new_forest_base, new_tree, new_vars, history, total_cost))

        # Handle queries that strictly require cross products (if query graph is disconnected)
        if not possible_next:
            break

        # Pass the newly formed subtrees to the agent to estimate step cost
        costs, environment_states = agent.estimate_costs(possible_next, query_state)

        plans_with_data = []
        for idx in range(len(possible_next)):
            new_tree = possible_next[idx]
            step_cost = float(costs[idx])
            state = environment_states[idx]

            new_forest_base, _, new_vars, old_history, old_total_cost = candidate_metadata[idx]

            new_forest = new_forest_base + [(new_tree, new_vars)]
            new_history = old_history + [(new_tree, state)]

            # Accumulate cost to rank the overall quality of the bushy topology (C_out metric)
            new_total_cost = old_total_cost + step_cost

            plans_with_data.append((new_forest, new_total_cost, new_history, new_tree))

        # Rank by the total accumulated cost
        plans_with_data.sort(key=lambda x: x[1])

        # Deduplicate structurally identical forests.
        # (Different execution paths can yield the exact same forest topologies in a bushy search)
        seen_forests = set()
        unique_candidates = []
        for forest, cost, history, final_tree in plans_with_data:
            # Freeze the forest structure into a unique string signature for safe hashing
            forest_signature = frozenset([str(t[0]) for t in forest])
            if forest_signature not in seen_forests:
                seen_forests.add(forest_signature)
                unique_candidates.append((forest, cost, history, final_tree))
                if len(unique_candidates) == beam_width:
                    break

        # Prepare the next iteration
        current_plans = [(forest, history, cost) for forest, cost, history, _ in unique_candidates]
        top_candidates = unique_candidates

    # Map the top final single-tree forests to the expected return format
    result = [
        dict(plan=final_tree, cost=cost, history=history)
        for forest, cost, history, final_tree in top_candidates
    ]
    return result