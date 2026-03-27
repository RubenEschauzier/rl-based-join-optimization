import re
from typing import List, Tuple, Any

from src.supervised_value_estimation.typings.typings import Plan, Candidate, History, EnvState

from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent


def beam_search(query: Any, agent: AbstractCostAgent, beam_width: int) -> list[
    dict[str, list[int] | float | list[tuple[list[int], EnvState]]]]:
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
    result = [
        dict(plan=plan, cost=cost, history=history) for plan, cost, history, variables_candidate in top_candidates
    ]
    return result
