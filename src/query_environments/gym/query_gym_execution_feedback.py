from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from SPARQLWrapper import JSON
from charset_normalizer.cli import query_yes_no
from torch_geometric.loader import DataLoader
import random
from math import factorial

from tqdm import tqdm

from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment


#https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
class QueryExecutionGymExecutionFeedback(gym.Env):
    def __init__(self, query_dataset, query_embedder, env,
                 reward_type: Literal['intermediate_results', 'execution_time', "cost_ratio"], max_triples=20,
                 alpha = .3, gamma = .99, train_mode=True, feature_dim = None):
        super().__init__()
        # The environment used to execute queries and obtain rewards
        self.env = env
        self.reward_type = reward_type
        self.query_timeout = 200

        # Our frozen pretrained GNN embedding the query.
        self.query_embedder = query_embedder

        # If train_mode is off, the DataLoader will not shuffle the queries
        self.train_mode = train_mode
        # Output feature size (lets infer this from the embedder instead)
        if not feature_dim:
            self.feature_dim = self.query_embedder.embedding_model[-1].nn[-1].out_features
        else:
            self.feature_dim = feature_dim

        # Max # of triples in query
        self.max_triples = max_triples
        # Dataset / loader of generated queries used to train RL algorithm on
        self.query_dataset = query_dataset
        self.query_loader = iter(DataLoader(query_dataset, batch_size=1, shuffle=self.train_mode)) # type: ignore

        self.observation_space = gym.spaces.Dict({
            "result_embeddings": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_triples, self.feature_dim)),
            "joined": gym.spaces.MultiBinary(self.max_triples),
            "join_order": gym.spaces.MultiDiscrete([self.max_triples + 1] * (self.max_triples - 1)),
        })

        self._result_embeddings = None
        self._join_embedding = None
        self._joined = None
        self._query = None

        self.join_count = 0
        # Action is choosing an index to join.
        self.action_space = gym.spaces.Discrete(max_triples)

        # Initialize the query (might need to be from the data loader)
        self.join_order: np.array = None
        self.n_triples_query = 0

        # Mixing functions for intermediate result reward
        self.alpha = alpha
        self.gamma = gamma


    def step(self, action):
        # Here is what we'll do: One step is ONE query. This function will output a list of actions taken by the model,
        # which are the actions and combine it with the representations. We store transitions in the replay buffer in
        # such away that we can rebuild the computational graph.
        # So we need not only base representations, but also the preceding joins.
        # This way we can iteratively apply the tree-lstm to get back the representation and the gradient.

        # Then for reward function we will follow the paper I found and try to look up if I can get blazegraph to output
        # a cost. Furthermore, we will use curriculum learning (test it by using small queries first and see if it will
        # train). Finally, we will include a latency tuning phase but this will be on virtual wall.

        # For actual model usage. I suggest combining cardinality estimation and QR-DQN to form a robust cardinality
        # estimation model.

        # Use this to implement Tree-LSTM:
        # https://github.com/pyg-team/pytorch_geometric/issues/121?utm_source=chatgpt.com?utm_source=chatgpt.com

        # In this step function we have to define how the representations are updated by adding a join.
        # One idea is to use a simple MLP over the two joined representations to output a new representation. Other is
        # things like tree-lstm. Then I need to figure out a way to include the output of a model in the environment
        # Input of the model will probably be
        if action >= self.n_triples_query or self._joined[action] == 1:
            raise ValueError("Invalid action")

        self._joined[action] = 1
        # Join order is defined from 0 to max_triples + 1 for processing purposes. 0 denotes not made joins
        self.join_order[self.join_count] = action + 1
        self.join_count += 1

        next_obs = self._build_obs()
        if self.join_count >= self.n_triples_query:
            # Set join order
            join_order_true = self.retrieve_true_join_order(self.join_order)
            join_order_trimmed = join_order_true[join_order_true != -1]
            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(self._query.query,
                                                                             join_order_trimmed,
                                                                             self._query.triple_patterns)
            # Execute query to obtain selectivity
            env_result, exec_time = self.env.run_raw(rewritten, self.query_timeout, JSON, {"explain": "True"})
            final_reward, reward_per_step = self._get_reward(self._query, env_result, exec_time, join_order_trimmed,
                                                             self.reward_type)
            next_obs = self._build_obs()
            return next_obs, final_reward, True, False, {"reward_per_step": reward_per_step}

        return next_obs, 0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            query = next(self.query_loader)[0]
        except StopIteration:
            self.query_loader = iter(DataLoader(self.query_dataset, batch_size=1, shuffle=self.train_mode)) # type: ignore
            query = next(self.query_loader)[0]
        embedded = self.query_embedder.forward(x=query.x,
                                       edge_index=query.edge_index,
                                       edge_attr=query.edge_attr,
                                       batch=query.batch)
        # Get the embedding head from the model
        embedded = next(head_output['output']
                        for head_output in embedded if head_output['output_type'] == 'triple_embedding')

        # Query graphs are with two edges to make undirected. These are simply duplicate embeddings so remove them
        # (n_triple_patterns, emb_size)
        embedded = embedded[::2]
        # Set joined to 1 for padding (it will function as a mask for policy network)
        joined = torch.cat((torch.zeros(embedded.shape[0],dtype=torch.int8),
                            torch.ones(self.max_triples - embedded.shape[0], dtype=torch.int8)))

        self.join_order = np.array([0] * (self.max_triples - 1))
        self.join_count = 0
        self.n_triples_query = embedded.shape[0]
        self._query = query
        self._result_embeddings = torch.nn.functional.pad(
                input=embedded, pad=(0 ,0, 0, self.max_triples - embedded.shape[0]), mode="constant", value=0 )
        self._joined = joined.numpy()

        return self._build_obs(), {}

    def _build_obs(self):
        return {
            "result_embeddings": self._result_embeddings,
            "join_order": self.join_order,
            "joined": self._joined
        }
    def _get_reward(self, query, env_result_policy, exec_time_policy, join_order_trimmed, reward_type):
        if reward_type == 'execution_time':
            reward_per_step_policy = [0] * join_order_trimmed.shape[0]
            reward_per_step_policy[-1] = np.log(exec_time_policy)
            final_reward_policy = - np.log(exec_time_policy)
            return final_reward_policy, reward_per_step_policy

        units_out, counts, join_ratio, status = self.env.process_output(env_result_policy, "intermediate-results")
        if reward_type == 'cost_ratio':
            #TODO TEST THIS BADBOY
            # Run original query to get cost from the default optimizer
            env_result_base, exec_time_base = self.env.run_raw(
                query.query, self.query_timeout, JSON, {"explain": "True"}
            )
            units_out_base, counts_base, join_ratio_base, status_base = (
                self.env.process_output(env_result_base, "intermediate-results"))
            if status == "OK":
                if status_base != "OK":
                    # For some reason the default optimizer cant execute it. Return high reward
                    reward_per_step_policy = 1 * join_order_trimmed.shape[0]
                    final_reward_policy = 1
                    return final_reward_policy, reward_per_step_policy

                reward_per_step_policy = QueryExecutionGymExecutionFeedback.query_plan_cost(units_out, counts)
                reward_per_step_base = QueryExecutionGymExecutionFeedback.query_plan_cost(units_out_base, counts_base)

                final_cost_policy = np.sum(reward_per_step_policy)
                final_cost_base = np.sum(reward_per_step_base)
                reward_per_step = []
                for i, (step_reward_policy, step_reward_base) in enumerate(zip(reward_per_step_policy, reward_per_step_base)):
                    ratio = np.log((((self.alpha * step_reward_base) +
                              (1 - self.alpha) * (self.gamma**i) * final_cost_base) /
                             ((self.alpha * step_reward_policy) +
                              (1 - self.alpha) * (self.gamma ** i) * final_cost_policy)))
                    reward_per_step.append(ratio)


                final_reward = np.log(final_cost_base/final_cost_policy)
            else:
                # Very large negative reward when query fails.
                reward_per_step = [-3] * join_order_trimmed.shape[0]
                final_reward = -3

            return final_reward, reward_per_step

        if reward_type == 'intermediate_results':
            if status == "OK":
                reward_per_step_policy = QueryExecutionGymExecutionFeedback.query_plan_cost(units_out, counts)
                final_reward_policy = - np.log(np.sum(reward_per_step_policy) + 1)
                reward_per_step_policy = [
                    self.alpha * - np.log(step_reward) + (1 - self.alpha) * self.gamma ** i * final_reward_policy
                    for i, step_reward in enumerate(reward_per_step_policy)]
            else:
                # Very large negative reward when query fails.
                reward_per_step_policy = [-20] * join_order_trimmed.shape[0]
                final_reward_policy = -20
            return final_reward_policy, reward_per_step_policy

        raise ValueError("Invalid reward_type in environment: {}".format(reward_type))


    def action_masks(self):
        return self._joined

    # TODO: Also validate how execution times differ between the join orders for a single query
    #  (this is doable for 3 size queries not otherwise)
    def validate_cost_function(self, queries, n_to_validate, orders_per_query,
                               reward_type: Literal['intermediate_results', 'execution_time', "cost_ratio"]):
        loader = iter(DataLoader(queries, batch_size=1, shuffle=self.train_mode)) # type: ignore
        data_execution_time = []
        data_reward = []
        for i in tqdm(range(n_to_validate)):
            query = next(loader)[0]
            join_orders = QueryExecutionGymExecutionFeedback.generate_random_join_order(query, orders_per_query)
            for order in join_orders:
                rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(query.query,
                                                                                 order,
                                                                                 query.triple_patterns)
                # Execute query to obtain selectivity and execution time
                env_result, exec_time = self.env.run_raw(rewritten, self.query_timeout, JSON, {"explain": "True"})
                reward, step_reward = self._get_reward(query, env_result, exec_time, order, reward_type)
                data_execution_time.append(exec_time)
                data_reward.append(reward)
        return data_execution_time, data_reward

    @staticmethod
    def generate_random_join_order(query, k):
        n = len(query.triple_patterns)
        if k > factorial(n):
            raise ValueError(f"Cannot generate {k} unique permutations of {n} elements (max is {factorial(n)}).")

        seen = set()
        while len(seen) < k:
            perm = tuple(random.sample(range(0, n), n))
            seen.add(perm)

        return [np.array(p) for p in seen]

    @staticmethod
    def retrieve_true_join_order(join_order):
        return join_order - 1

    @staticmethod
    def query_plan_cost(units_out, counts):
        # We add first count to reward query plans with small initial scans
        cost = [counts[0]]
        # cost = counts[0]
        for i in range(units_out.shape[0] - 1):
            # Join work assuming index-based nested loop join (should include a cost for hash join)
            cost.append(units_out[i]*counts[i+1])
            # cost.append((units_out[i] * np.log(counts[i + 1]+1)))
            # cost += (units_out[i] * np.log(counts[i + 1]+1))
        return cost



