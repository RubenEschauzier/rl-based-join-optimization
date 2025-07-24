import gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric import EdgeIndex
from torch_geometric.utils import is_undirected

from src.models.model_layers.tree_lstm import NAryTreeLSTM, ChildSumTreeLSTM


class QRDQNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim=512):
        super().__init__(observation_space, feature_dim)

        self.max_join_masks = observation_space["joined"].shape[0]  # max_triples
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
            nn.Linear(self.max_join_masks, self.max_join_masks),
            nn.ReLU(),
            nn.Linear(self.max_join_masks, self.max_join_masks),
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
        self._features_dim = feature_dim + self.max_join_masks

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
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim):
        super().__init__(observation_space, features_dim=feature_dim)

        self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
        self.feature_dim = feature_dim

        # Recursively builds the join state
        self.n_ary_tree_lstm = NAryTreeLSTM(self.feature_dim, self.feature_dim, 2)

        # Takes all other triple patterns not joined and the join representation from the n-ary tree lstm
        # and outputs a vector representing the current state
        self.child_sum_tree_lstm = ChildSumTreeLSTM(self.feature_dim, self.feature_dim)

        # We apply this over the embedding of the triple pattern used in a state with a single join. Then this value
        # is used as 'h' input, while the embedding from self.intermediate_join_embedding is used as x. This signifies
        # to the model that this triple pattern has already been selected as a join
        self.join_embedding_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
        )

        # Intermediate join_embedding, this will give a learnable dummy 'x' value for intermediate joins
        self.int_join_emb = nn.Embedding(1, feature_dim)

    def forward(self, observations):
        """
        Something is wrong: when using this extractor, the model learns to always do the same action / prediction.
        Possible avenues of fixing: Maybe the model is unaware of what is joined and isn't?
        observations:
            - result_embeddings: (batch_size, max_triples, feature_dim)
            - join_embedding: (batch_size, feature_dim)
            - joined: (batch_size, max_triples) - Binary mask
            - join_graph (batch_size, 2, (max_triples-1)*2) - EdgeIndex representing join tree
            - lstm_order (batch_size, max_triples-1, (max_triples-1)*2)
            - lstm_order_mask (batch_size, max_triples-1)
        """
        # (B, max_triples, feature_dim)
        result_embeddings = observations["result_embeddings"]
        join_graphs = observations["join_graph"]
        lstm_order = observations["lstm_order"]
        lstm_order_mask = observations["lstm_order_mask"]
        joined = observations["joined"]
        joins_made = observations["joins_made"]
        n_triples_obs = observations["n_triples"]
        # Observations are batched, so we need to do forward passes on a per-batch basis
        # Furthermore, observations also requires a 'graph' to be constructed of edge_indexes and an 'order'
        # which is how we execute our n_ary_tree_lstm recursively. On the other hand, h, c can be zero tensors with the
        # right shape
        device = result_embeddings.device
        state_embeddings = []
        for i in range(result_embeddings.shape[0]):
            n_joins = joins_made[i][0]
            n_triples = n_triples_obs[i][0]

            x = result_embeddings[i]
            result_embeddings_un_joined = self.get_un_joined_embeddings(joined[i], x)
            if n_joins == 0:
                empty_join_representation = torch.empty(result_embeddings_un_joined[0].shape).unsqueeze(dim=0)
                h_c, c_c, x_c = self.get_state_child_sum_input(result_embeddings_un_joined,
                                                               empty_join_representation,
                                                               empty_join_representation,
                                                               empty_join_representation,
                                                               x)
            elif n_joins == 1:
                join_order = observations["join_order"][i]
                last_join = join_order[0]

                # The triple pattern embedding is transformed by a single layer mlp and then fed as hidden state
                h_last_join = self.join_embedding_mlp(x[last_join])
                # The x is the learned dummy representation of intermediate joins
                x_last_join = self.int_join_emb(torch.tensor([0], device=device))
                empty_join_representation = torch.empty((0,result_embeddings_un_joined[0].shape[0]))
                h_c, c_c, x_c = self.get_state_child_sum_input(result_embeddings_un_joined,
                                                               empty_join_representation,
                                                               empty_join_representation,
                                                               empty_join_representation,
                                                               x)
                h_c[last_join] = h_last_join
                x_c[last_join] = x_last_join

            else:
                edge_index = EdgeIndex(join_graphs[i], is_undirected=False)
                h = torch.zeros_like(x)
                c = torch.zeros_like(x)
                order_mask = lstm_order[i][lstm_order_mask[i].bool()].bool()
                h_join, c_join = self.n_ary_tree_lstm.forward(x, edge_index, h, c, order_mask)


                # TODO: Add zero embeddings to these three to represent the to be added parent embedding that will
                # Input to the child-sum tree-lstm. Consists of:
                # - The left-over triple patterns not joined yet
                # - The intermediate join representation obtained from hierarchical 2-ary tree-lstm. The x value for
                # the intermediate join is obtained from a learnable dummy
                # - A zeros tensor for the aggregating parent node, this parent node obtains its representation from
                # a forward pass from child-sum tree-lstm.
                h_c, c_c, x_c = self.get_state_child_sum_input(result_embeddings_un_joined,
                                                               h_join[n_triples + n_joins - 2].unsqueeze(dim=0),
                                                               c_join[n_triples + n_joins - 2].unsqueeze(dim=0),
                                                               self.int_join_emb(torch.tensor([0], device=device)),
                                                               x)
                # edge_index_state, order_state = self.get_join_state_graph(x_c, device)
                # h_state, c_state = self.child_sum_tree_lstm.forward(
                #     x_c, edge_index_state, h_c, c_c, order_state)
                # state_embeddings.append(h_state[-1])
            edge_index_state, order_state = self.get_join_state_graph(x_c, device)
            h_state, c_state = self.child_sum_tree_lstm.forward(
                x_c, edge_index_state, h_c, c_c, order_state)
            state_embeddings.append(h_state[-1])

        return torch.stack(state_embeddings)

    @staticmethod
    def get_un_joined_embeddings(joined, x):
        # Add padding to joined tensor that masks out the x values corresponding to intermediate join features
        joined_padded = torch.nn.functional.pad(joined.bool(),
                                                (0, x.shape[0] - joined.shape[0]),
                                                value=True)
        return x[~joined_padded]

    @staticmethod
    def get_state_child_sum_input(result_emb_un_joined,
                                  h_join_emb, c_join_emb, x_join_emb,
                                  x):
        h_child_sum = torch.concat(
            (torch.zeros_like(result_emb_un_joined),
             h_join_emb,
             torch.zeros_like(x[0]).unsqueeze(dim=0)),
            dim=0)
        c_child_sum = torch.concat(
            (torch.zeros_like(result_emb_un_joined),
             c_join_emb,
             torch.zeros_like(x[0]).unsqueeze(dim=0)),
            dim=0)
        # The intermediate join is represented by the dummy embedding
        x_child_sum = torch.concat(
            (result_emb_un_joined,
             x_join_emb,
             torch.zeros_like(x[0]).unsqueeze(dim=0)),
            dim=0
        )
        return h_child_sum, c_child_sum, x_child_sum

    @staticmethod
    def get_join_state_graph(x, device):
        edge_index_state = EdgeIndex([
            [i for i in range(x.shape[0] - 1)],
            [x.shape[0] - 1 for _ in range(x.shape[0] - 1)],
        ], is_undirected=False, device=device)
        order_tensor = torch.zeros((x.shape[0]), dtype=torch.bool, device=device)
        order_tensor[-1] = 1
        return edge_index_state, [order_tensor]