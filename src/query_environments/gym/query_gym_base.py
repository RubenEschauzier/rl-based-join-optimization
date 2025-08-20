from abc import abstractmethod
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import gymnasium as gym

class QueryGymBase(gym.Env):
    def __init__(self, query_dataset, query_embedder, env, max_triples=20,
                 alpha = .3, gamma = .99, train_mode=True, feature_dim=None):
        super().__init__()
        # The environment used to execute queries and obtain rewards
        self.env = env

        # Our frozen pretrained GNN embedding the query.
        self._query_embedder = query_embedder

        # If train_mode is off, the DataLoader will not shuffle the queries
        self.train_mode = train_mode

        # Output feature size (lets infer this from the embedder instead)
        if not feature_dim:
            self.feature_dim = self._query_embedder.embedding_model[-1].nn[-1].out_features
        else:
            self.feature_dim = feature_dim

        # Max # of triples in query
        self.max_triples = max_triples
        # Dataset / loader of generated queries used to train RL algorithm on
        self.query_dataset = query_dataset
        self.query_loader = iter(DataLoader(query_dataset, batch_size=1, shuffle=self.train_mode))  # type: ignore

        self.observation_space = gym.spaces.Dict({
            # Default observation space
            "result_embeddings": gym.spaces.Box(low=-np.inf, high=np.inf, shape=((self.max_triples*2)-1, self.feature_dim)),
            "joined": gym.spaces.MultiBinary(self.max_triples),
            "join_order": gym.spaces.MultiDiscrete([self.max_triples + 1] * (self.max_triples - 1)),
            # Join tree edge_index. This needs no masking as the lstm_order already functions as a mask.
            "join_graph": gym.spaces.Box(low=0, high=(self.max_triples-1)*2, shape=(2, (self.max_triples-1)*2),
                                         dtype=np.int64),
            # The order in which tree-lstm operations should be executed, denoting 1 if a parent node representation
            # should be computed in that pass
            "lstm_order": gym.spaces.Box(low=0, high=1, shape=(self.max_triples-1, (self.max_triples*2)-1),
                                         dtype=np.int64),
            # What orders should be masked out as these are padding orders, 1 means it is NOT masked
            "lstm_order_mask": gym.spaces.Box(low=0, high=1, shape=(self.max_triples-1,),
                                              dtype=np.int64),
            # How many joins have already been made
            "joins_made": gym.spaces.Box(low=0, high=self.max_triples-1, shape=(1,), dtype=np.int64),
            # Number of triple patterns in the query
            "n_triples": gym.spaces.Box(low=0, high=self.max_triples-1, shape=(1,), dtype=np.int64)
        })

        self._result_embeddings = None
        self._join_embedding = None
        self._joined = None
        self._query = None
        self._joins_made = 0
        self._join_order: np.array = None
        self._n_triples_query = 0
        # Initialize lstm application datastructures
        self._join_graph = None
        self._lstm_order = None
        self._lstm_order_mask = None

        # Action is choosing an index to join in left-deep plan.
        self.action_space = gym.spaces.Discrete(max_triples)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            query = next(self.query_loader)[0]
        except StopIteration:
            self.query_loader = iter(DataLoader(self.query_dataset, batch_size=1, shuffle=self.train_mode)) # type: ignore
            query = next(self.query_loader)[0]
        embedded = self._query_embedder.forward(x=query.x,
                                                edge_index=query.edge_index,
                                                edge_attr=query.edge_attr,
                                                batch=query.batch)
        # Get the embedding head from the model
        embedded = next(head_output['output']
                        for head_output in embedded if head_output['output_type'] == 'triple_embedding')

        # Query graphs are with two edges to make undirected. These are simply duplicate embeddings so remove them
        # (n_triple_patterns, emb_size)
        embedded = embedded[::2]
        # Pad until max_triples + possible joins (# of actual triples patterns - 1)
        self._result_embeddings = torch.nn.functional.pad(
            embedded,
            pad=(0, 0, 0, 2 * self.max_triples - embedded.shape[0] - 1),
            mode="constant",
            value=0
        )

        # Set joined to 1 for padding (it will function as a mask for policy network)
        joined = torch.cat((torch.zeros(embedded.shape[0],dtype=torch.int8),
                            torch.ones(self.max_triples - embedded.shape[0], dtype=torch.int8)))

        # Initialize internal variables for observation space
        self._joined = joined.numpy()
        self._join_order = np.array([-1] * (self.max_triples - 1))
        # Join graph is edges: source -> target denoting join graph. So has given (max_triples - 1) joins needed and
        # each join 2 edges, we get given shape.
        self._join_graph = np.zeros((2, (self.max_triples-1)*2), dtype=np.int64)
        # Order (hierarchy) in which Tree-LSTM should be applied equal to n_joins (self.max_triples-1) and the number
        # of nodes max_triples (triple patter nodes) + max_triples - 1 (join nodes)
        # Order basically is a mask that says which _parent_ nodes (or target) should be computed
        self._lstm_order = np.zeros((self.max_triples-1,(self.max_triples*2)-1), dtype=np.int64)
        # Mask to remove padding from lstm_order (probably can be done without)
        self._lstm_order_mask = np.zeros((self.max_triples-1,), dtype=np.int64)
        self._joins_made = 0
        self._n_triples_query = embedded.shape[0]

        self._query = query

        return self._build_obs(), {}


    def step(self, action):
        if action >= self._n_triples_query or self._joined[action] == 1:
            raise ValueError("Invalid action")

        self._joined[action] = 1
        # Join order is defined from 0 to max_triples + 1 for processing purposes. 0 denotes not made joins
        self._join_order[self._joins_made] = action
        self._joins_made += 1

        next_obs = self._build_obs()

        reward, _ = self.get_reward(self._query, self._join_order, self._joins_made)

        done = False
        if self._joins_made >= self._n_triples_query:
            done = True

        infos = self._build_infos(done)

        return next_obs, reward, done, False, infos

    def action_masks(self):
        return self._joined

    def action_masks_ppo(self):
        return (1 - self._joined).astype(bool)

    def _build_obs(self):
        self._build_tree_input()

        return {
            "result_embeddings": self._result_embeddings,
            "join_order": self._join_order,
            "joined": self._joined,
            "join_graph": self._join_graph,
            "lstm_order": self._lstm_order,
            "lstm_order_mask": self._lstm_order_mask,
            "joins_made": self._joins_made,
            "n_triples": self._n_triples_query,
        }

    def _build_tree_input(self):
        # Add selected join to a single child tree node
        if self._joins_made == 1:
            self._join_graph[0][0] = self._join_order[0]
            self._join_graph[1][0] = self._n_triples_query
            self._lstm_order[0][self._n_triples_query] = 1
            self._lstm_order_mask[0] = 1
        # Fill the other side of join, don't need to change the order, because it still is the same 'parent' join node
        elif self._joins_made == 2:
            self._join_graph[0][1] = self._join_order[1]
            self._join_graph[1][1] = self._n_triples_query
        elif self._joins_made > 2:
            n_triple_patterns = self._n_triples_query

            # First two edges are added when join count = 2, then subsequent joins add two edges.
            index_to_add_edge = 2 + (self._joins_made - 3)*2
            self._join_graph[0][index_to_add_edge] = self._join_order[self._joins_made-1]
            # The index representing the join increments by one for each join, starting from n_tps - 1
            # The first two join counts increment it by 1, so subtract 1.
            self._join_graph[1][index_to_add_edge] = n_triple_patterns - 1 + self._joins_made - 1
            # The previous join is always included in the next (left-deep)
            self._join_graph[0][index_to_add_edge+1] = n_triple_patterns - 1 + self._joins_made - 1 - 1
            self._join_graph[1][index_to_add_edge+1] = n_triple_patterns - 1 + self._joins_made - 1

            # Set the order array for the new join node
            self._lstm_order[self._joins_made - 2][self._n_triples_query+(self._joins_made-2)] = 1

            # Unmask this order
            self._lstm_order_mask[self._joins_made - 2] = 1

    # First join
        # if self._joins_made == 2:
        #     # First join is a special case
        #     self._join_graph[0][0] = self._join_order[0]
        #     self._join_graph[1][0] = self._n_triples_query
        #     self._join_graph[0][1] = self._join_order[1]
        #     self._join_graph[1][1] = self._n_triples_query
        #     # First order mask set join node to 1 to represent it should get its hidden state computed
        #     self._lstm_order[0][self._n_triples_query] = 1
        #     # Unmask the order we just made to show model that this is a valid order entry
        #     self._lstm_order_mask[0] = 1
        #
        # # Subsequent joins
        # elif self._joins_made > 2:
        #     n_triple_patterns = self._n_triples_query
        #
        #     # First two edges are added when join count = 2, then subsequent joins add two edges.
        #     index_to_add_edge = 2 + (self._joins_made - 3)*2
        #     self._join_graph[0][index_to_add_edge] = self._join_order[self._joins_made-1]
        #     # The index representing the join increments by one for each join, starting from n_tps - 1
        #     # The first two join counts increment it by 1, so subtract 1.
        #     self._join_graph[1][index_to_add_edge] = n_triple_patterns - 1 + self._joins_made - 1
        #     # The previous join is always included in the next (left-deep)
        #     self._join_graph[0][index_to_add_edge+1] = n_triple_patterns - 1 + self._joins_made - 1 - 1
        #     self._join_graph[1][index_to_add_edge+1] = n_triple_patterns - 1 + self._joins_made - 1
        #
        #     # Set the order array for the new join node
        #     self._lstm_order[self._joins_made - 2][self._n_triples_query+(self._joins_made-2)] = 1
        #
        #     # Unmask this order
        #     self._lstm_order_mask[self._joins_made - 2] = 1

    def _build_infos(self, done: bool):
        return {}

    @abstractmethod
    def  get_reward(self, query, join_order, joins_made):
        pass

    @property
    def query_embedder(self):
        return self._query_embedder

    @property
    def query(self):
        return self._query

    @property
    def join_order(self):
        return self._join_order

    @property
    def joins_made(self):
        return self._joins_made

