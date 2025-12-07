import gymnasium as gym
import torch
import numpy as np
from src.baselines.enumeration import JoinOrderEnumerator, build_adj_list, JoinPlan
from src.query_environments.gym.query_gym_base import QueryGymBase
from src.query_environments.gym.query_gym_estimated_cost import QueryGymEstimatedCost


class OrderDynamicProgramming(gym.Wrapper):
    """
    A wrapper that adds optimal join order computation and caching functionality
    to a custom gymnasium environment.
    """

    def __init__(self, env: QueryGymBase, device, cache_optimal_cost=True):
        """
        Initialize the wrapper.

        Args:
            env: The base gymnasium environment to wrap
        """
        super().__init__(env)
        self.env: QueryGymBase = env
        self.cache_optimal_cost = cache_optimal_cost

        # Cache for optimal plans (left-deep)
        self.optimal_plans_left_deep = {}
        self.optimal_rewards_left_deep = {}

        # Track the last optimal reward for info reporting
        self.last_optimal_cost = None
        self.last_optimal_reward = None
        self.last_optimal_plan = None
        self.device = device

    def reset(self, seed=None, options=None):
        """
        Reset the environment and compute optimal order if needed.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Tuple of (observation, info)
        """
        # Call the base environment's reset
        obs, info = self.env.reset(seed=seed, options=options)

        # Access the query from the wrapped environment
        current_query = self.env.query

        # Compute and cache optimal join order if not seen before
        if current_query.query not in self.optimal_plans_left_deep or not self.cache_optimal_cost:
            optimal_plan_bushy, optimal_plan_ld = self.get_optimal_order(current_query)
            reward_plan = self.get_reward_left_deep_plan(current_query, optimal_plan_ld)
            self.optimal_plans_left_deep[current_query.query] = optimal_plan_ld
            self.optimal_rewards_left_deep[current_query.query] = reward_plan

        # Update the last optimal reward
        self.last_optimal_cost = self.optimal_plans_left_deep[current_query.query].cost
        self.last_optimal_reward = self.optimal_rewards_left_deep[current_query.query]
        self.last_optimal_plan = self.optimal_plans_left_deep[current_query.query]
        return obs, info

    def step(self, action):
        """
        Execute a step in the environment.

        Args:
            action: The action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Call the base environment's step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add optimal reward information to infos
        if terminated:
            info["optimal_reward_left_deep"] = self.last_optimal_reward

        return obs, reward, terminated, truncated, info

    def get_reward_left_deep_plan(self, query, optimal_plan):
        def traverse_plan(plan: JoinPlan):
            if plan.left.is_leaf and plan.right.is_leaf:
                return [list(plan.left.entries)[-1], list(plan.right.entries)[-1]]
            order = traverse_plan(plan.left)
            order.append(list(plan.right.entries).pop())
            return order
        join_order = np.array(traverse_plan(optimal_plan))
        plan_reward = self.get_intermediate_cost_estimates(query, join_order)
        return plan_reward

    def get_intermediate_cost_estimates(self, query, join_order):
        total = 0
        for i in range(1, len(join_order)+1):
            reward, _ = self.env.get_reward(query, join_order[0:i], len(join_order[0:i]))
            if isinstance(reward, (int, float)):
                total += reward
            else:
                total += np.array(reward).squeeze().item()
        return total

    def get_optimal_order(self, query):
        return self.enumerate_optimal_order(query)

    def enumerate_optimal_order(self, query):
        adjacency_list = build_adj_list(query)
        return JoinOrderEnumerator(adjacency_list, self.bound_predict, len(query.triple_patterns)).search()

    def bound_predict(self, join_order):
        return self.predict_cardinality(self.env.query_embedder, self.env.query, list(join_order), len(join_order),
                                        self.device)

    def action_masks(self):
        return self.env.action_masks()

    def action_masks_ppo(self):
        return self.env.action_masks_ppo()

    @staticmethod
    def predict_cardinality(model, query, join_order, join_count, device):
        query_to_estimate = QueryGymEstimatedCost.reduced_form_query(query, join_order, join_count, device)
        with torch.no_grad():
            output = model.forward(x=query_to_estimate.x,
                                   edge_index=query_to_estimate.edge_index,
                                   edge_attr=query_to_estimate.edge_attr,
                                   batch=query_to_estimate.batch)
            card = float(next(head_output['output'] for head_output in output
                              if head_output['output_type'] == 'cardinality'))
            return card

