import gymnasium as gym
import numpy as np
from torch_geometric.data import DataLoader


class QueryExecutionGym(gym.Env):
    def __init__(self, query_dataset, feature_dim, query_embedder, env, max_triples=20, ):
        super().__init__()
        # The environment used to execute queries and obtain rewards
        self.env = env
        # Our frozen pretrained GNN embedding the query.
        self.query_embedder = query_embedder
        for param in self.query_embedder.parameters():
            param.requires_grad = False
        # Output feature size (lets infer this from the embedder instead
        self.feature_dim = feature_dim
        # Max # of triples in query
        self.max_triples = max_triples
        # Dataset / loader of generated queries used to train RL algorithm on
        self.query_dataset = query_dataset
        self.query_loader = iter(DataLoader(query_dataset, batch_size=2, shuffle=True)) # type: ignore
        # Observation space is processed query graphs, which result in tp embeddings
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_triples, self.feature_dim))
        # Action is choosing an index to join.
        self.action_space = gym.spaces.Discrete(max_triples - 1)
        self.metadata = {'torch': True}

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        try:
            query = next(self.query_loader) # Assuming dataset returns tensors
        except StopIteration:

            self.query_loader = iter(DataLoader(self.dataset, batch_size=1, shuffle=True)) # type: ignore
            query = next(self.query_loader)
        embedded = self.query_embedder.forward(x=query.x,
                                       edge_index=query.edge_index,
                                       edge_attr=query.edge_attr,
                                       batch=query.batch)
        # Embedded (n_nodes, emb_size)
        print(query.query)
        print(embedded.shape)
        return embedded

    """
    In this function we will load a query, pass it through our GNN layers and create embeddings of triple patterns in 
    the query. We will _freeze_ these layers, as to retain the pretraining we have done and speed up our training.
    As such they are not part of the policy and value networks being trained and are just a preprocessing step that is
    part of the environment
    """
    def build_initial_observation(self):
        pass

