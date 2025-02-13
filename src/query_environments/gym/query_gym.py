import gymnasium as gym
import numpy as np
import torch
from SPARQLWrapper import JSON
from torch_geometric.loader import DataLoader

from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment


#https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html
class QueryExecutionGym(gym.Env):
    def __init__(self, query_dataset, feature_dim, query_embedder, join_embedder, env, max_triples=20, ):
        super().__init__()
        # The environment used to execute queries and obtain rewards
        self.env = env
        self.query_timeout = 20
        # Our frozen pretrained GNN embedding the query.
        self.query_embedder = query_embedder
        for param in self.query_embedder.parameters():
            param.requires_grad = False

        # This is a learned embedder and requires gradients
        self.join_embedder = join_embedder

        # Output feature size (lets infer this from the embedder instead
        self.feature_dim = feature_dim
        # Max # of triples in query
        self.max_triples = max_triples
        # Dataset / loader of generated queries used to train RL algorithm on
        self.query_dataset = query_dataset
        self.query_loader = iter(DataLoader(query_dataset, batch_size=1, shuffle=True)) # type: ignore

        self.observation_space = gym.spaces.Dict({
           "result_embeddings": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_triples, self.feature_dim)),
           "joined": gym.spaces.MultiBinary(self.max_triples),
           "last_action": gym.spaces.Box(low=-1, high=self.max_triples, shape=(1,) ),
        })
        self._result_embeddings = None
        self._join_embedding = None
        self._joined = None
        self._query = None

        self.join_count = 0
        # Action is choosing an index to join.
        self.action_space = gym.spaces.Discrete(max_triples)

        # Initialize the query (might need to be from the data loader)
        self.join_order = []
        self.n_triples_query = 0


    def step(self, action):
        # In this step function we have to define how the representations are updated by adding a join.
        # One idea is to use a simple MLP over the two joined representations to output a new representation. Other is
        # things like tree-lstm. Then I need to figure out a way to include the output of a model in the environment
        # Input of the model will probably be
        if action >= self.n_triples_query or self._joined[action] == 1:
            raise ValueError("Invalid action")
            # Invalid action
            next_obs = {
                "result_embeddings": self._result_embeddings,
                "joined": self._joined,
                "last_action": action,
            }
            return next_obs, -1, True, False, {}
        self._joined[action] = 1
        self.join_order.append(action)
        self.join_count += 1

        if self.join_count == 1:
            # First run we just set the join embedding of the chosen value
            next_obs = {
                "result_embeddings": self._result_embeddings,
                "joined": self._joined,
                "last_action": action,
            }
            return next_obs, 0, False, False, {}

        if len(self.join_order) >= self.n_triples_query - 1:
            # Set join order
            rewritten = BlazeGraphQueryEnvironment.set_join_order_json_query(self._query.query,
                                                                             self.join_order,
                                                                             self._query.triple_patterns)
            # Execute query to obtain selectivity
            env_result, exec_time = self.env.run_raw(rewritten, self.query_timeout, JSON, {"explain": "True"})
            # (Old penalty implementation, should look into how to shape this reward properly)
            units_out, counts, join_ratio, status = self.env.process_output(env_result, "intermediate-results")
            if status == "OK":
                reward = - np.log(QueryExecutionGym.query_plan_cost(units_out, counts)+1)
            else:
                # Very large negative reward when query fails.
                reward = -20

            next_obs = {
                "result_embeddings": self._result_embeddings,
                "joined": self._joined,
                "last_action": action,
            }
            return next_obs, reward, True, False, {}

        # Use join embedder to get representation of new result set
        # join_representation = torch.cat(
        #     (self._join_embedding, self._result_embeddings[action])
        # )
        # join_emb = self.join_embedder.forward(join_representation)
        next_obs = {
            "result_embeddings": self._result_embeddings,
            "joined": self._joined,
            "last_action": action,
        }
        return next_obs, 0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            query = next(self.query_loader)[0]
        except StopIteration:

            self.query_loader = iter(DataLoader(self.dataset, batch_size=1, shuffle=True)) # type: ignore
            query = next(self.query_loader)[0]

        embedded = self.query_embedder.forward(x=query.x,
                                       edge_index=query.edge_index,
                                       edge_attr=query.edge_attr,
                                       batch=query.batch)
        # Query graphs are with two edges to make undirected. These are simply duplicate embeddings so remove them
        # (n_triple_patterns, emb_size)
        embedded = embedded[::2]
        # Set joined to 1 for padding (it will function as a mask for policy network)
        joined = torch.cat((torch.zeros(embedded.shape[0],dtype=torch.int8),
                            torch.ones(self.max_triples - embedded.shape[0], dtype=torch.int8)))

        self.join_order = []
        self.join_count = 0
        self.n_triples_query = embedded.shape[0]
        self._query = query
        self._result_embeddings = torch.nn.functional.pad(
                input=embedded, pad=(0 ,0, 0, self.max_triples - embedded.shape[0]), mode="constant", value=0 )
        # self._join_embedding = torch.zeros((embedded.shape[1],))
        self._joined = joined.numpy()

        return {
            "result_embeddings": self._result_embeddings,
            "joined": self._joined,
            "last_action": -1,
        }, {}

    def action_masks(self):
        return self._joined

    @staticmethod
    def query_plan_cost(units_out, counts):
        # We add first count to reward query plans with small initial scans
        cost = counts[0]
        for i in range(units_out.shape[0] - 1):
            # Join work assuming index-based nested loop join (should include a cost for hash join)\
            cost += (units_out[i] * np.log(counts[i + 1]+1))
        return cost

