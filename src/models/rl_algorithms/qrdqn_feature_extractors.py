import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric import EdgeIndex
from torch_geometric.utils import is_undirected

from src.models.model_layers.tree_lstm import NAryTreeLSTM, ChildSumTreeLSTM


class QRDQNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim=512):
        super().__init__(observation_space, feature_dim)

        self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
        self.feature_dim = feature_dim

        # Preprocess the entire query graph
        # AdaptiveMaxPool doesn't work
        self.result_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Embed the mask to allow model to learn what actions are invalid
        self.mask_embedder = nn.Sequential(
            nn.Linear(self.max_triples, self.max_triples),
            nn.ReLU(),
            nn.Linear(self.max_triples, self.max_triples),
            nn.ReLU()
        )

        # Hierarchical embedding of join order
        self.join_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Final feature output size
        self._features_dim = feature_dim + self.max_triples

    def forward(self, observations):
        """
        observations:
            - result_embeddings: (batch_size, max_triples, feature_dim)
            - join_embedding: (batch_size, feature_dim)
            - joined: (batch_size, max_triples) - Binary mask
        """
        # (B, max_triples, feature_dim)
        result_embeddings = observations["result_embeddings"]
        joined = observations["joined"]

        # Here we do -1 to obtain the actual join order as the environment increments all join orders by 1 for
        # sb3 preprocessing purposes
        join_order = observations["join_order"] - 1
        # TODO Hierarchical MLP application based on join order

        # Pool features through sum pooling, as the masked out features will be 0 biasing the representation
        pooled = torch.sum(result_embeddings, dim=1)
        # (B, feature_dim)
        result_features = self.result_mlp(pooled)

        mask_emb = self.mask_embedder(joined)
        return torch.concat((result_features, mask_emb), dim=1)

class QRDQNFeatureExtractorTreeLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim=512):
        super().__init__(observation_space, feature_dim)

        self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
        self.feature_dim = feature_dim

        # Recursively builds the join state. In RTOS this is separate model for each
        # TODO Test separate model for each recursion depth
        self.n_ary_tree_lstm = NAryTreeLSTM(self.feature_dim, self.feature_dim, 2)

        # Takes all other triple patterns not joined and the join representation from the n-ary tree lstm
        # and outputs a vector representing the current state
        self.child_sum_tree_lstm = ChildSumTreeLSTM(self.feature_dim, self.feature_dim)
        # Preprocess the entire query graph
        # AdaptiveMaxPool doesn't work

        # We apply this over the embedding of the triple pattern used in a state with a single join. Then this value
        # is used as 'h' input, while the embedding from self.intermediate_join_embedding is used as x. This signifies
        # to the model that this triple pattern has already been selected as a join.
        self.join_embedding_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
        )

        # Intermediate join_embedding, this will give a learnable dummy 'x' value for intermediate joins
        self.intermediate_join_embedding = nn.Embedding(1, feature_dim)


        self.result_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Embed the mask to allow model to learn what actions are invalid
        self.mask_embedder = nn.Sequential(
            nn.Linear(self.max_triples, self.max_triples),
            nn.ReLU(),
            nn.Linear(self.max_triples, self.max_triples),
            nn.ReLU()
        )

        # Hierarchical embedding of join order
        self.join_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Final feature output size
        self._features_dim = feature_dim + self.max_triples

    def forward(self, observations):
        """
        observations:
            - result_embeddings: (batch_size, max_triples, feature_dim)
            - join_embedding: (batch_size, feature_dim)
            - joined: (batch_size, max_triples) - Binary mask
        """
        # (B, max_triples, feature_dim)
        result_embeddings = observations["result_embeddings"]
        join_graphs = observations["join_graph"]
        lstm_order = observations["lstm_order"]
        lstm_order_mask = observations["lstm_order_mask"]

        # Observations are batched, so we need to do forward passes on a per-batch basis
        # Furthermore, observations also requires a 'graph' to be constructed of edge_indexes and an 'order'
        # which is how we execute our n_ary_tree_lstm recursively. On the other hand, h, c can be zero tensors with the
        # right shape
        state_embeddings = []
        for i in range(result_embeddings.shape[0]):
            # TODO: Building the new graph + mask can be shared, just how x, h, c are created is different.
            if observations["joins_made"][i] == 0:
                # Just pass them all through the child-sum tree-lstm
                pass
            elif observations["joins_made"][i] == 1:
                # Encode the one triple pattern by setting its 'x' to 'h' and setting 'x' to the learned dummy
                pass
            else:
                edge_index = EdgeIndex(join_graphs[i], is_undirected=False)
                x = result_embeddings[i]
                h = torch.zeros_like(x)
                c = torch.zeros_like(x)
                order_mask = lstm_order[i][lstm_order_mask[i].bool()]
                h_join, c_join = self.n_ary_tree_lstm.forward(x, edge_index, h, c, order_mask)
                # TODO: Feed this with the left-over triple pattern result-embeddings into child-sum
                #  Create x by concatenating left-over triple pattern representations with
                #  the learnable dummy representation. Then create h, c with all zeros and only the intermediate set
                #  Finally, the edge_index is all pointing towards n_triples_unjoined + 1
                #  Order mask is zeros except for the last one.

            break

        # for i, order in enumerate(join_order):
        #     x = result_embeddings[i].squeeze()
        #     h = torch.zeros_like(x)
        #     c = torch.zeros_like(x)
            #TODO Construct edge_index and order mask here


        # forward(self, x, edge_index, h, c, orders):
        # TODO Hierarchical MLP application based on join order

        # Pool features through sum pooling, as the masked out features will be 0 biasing the representation
        pooled = torch.sum(result_embeddings, dim=1)
        # (B, feature_dim)
        result_features = self.result_mlp(pooled)
        raise ValueError("Fuck you")
        mask_emb = self.mask_embedder(joined)
        return torch.concat((result_features, mask_emb), dim=1)
