import math

import gymnasium as gym
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

import treelstm

class QRDQNFeatureExtractorTreeLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim):
        super().__init__(observation_space, features_dim=feature_dim)

        self.max_join_masks = observation_space["joined"].shape[0]  # max_triples
        self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
        self.feature_dim = feature_dim

        # Recursively builds the join state
        # self.n_ary_tree_lstm = NAryTreeLSTM(self.feature_dim, self.feature_dim, 2)
        self.child_sum_tree_lstm_intermediate = ChildSumTreeLSTM(self.feature_dim, self.feature_dim)
        # Takes all other triple patterns not joined and the join representation from the n-ary tree lstm
        # and outputs a vector representing the current state
        self.child_sum_tree_lstm = ChildSumTreeLSTM(self.feature_dim, self.feature_dim)

        self.tree_lstm_intermediate = treelstm.TreeLSTM(self.feature_dim, self.feature_dim)
        self.tree_lstm_final = treelstm.TreeLSTM(self.feature_dim, self.feature_dim)

        # We apply this over the embedding of the triple pattern used in a state with a single join. Then this value
        # is used as 'h' input, while the embedding from self.intermediate_join_embedding is used as x. This signifies
        # to the model that this triple pattern has already been selected as a join
        self.join_embedding_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
        )

        # Intermediate join_embedding, this will give a learnable dummy 'x' value for intermediate joins
        self.int_join_emb = nn.Embedding(1, feature_dim)

        # Embed the mask to allow model to learn what actions are invalid
        self.mask_embedder = nn.Sequential(
            nn.Linear(self.max_join_masks, self.max_join_masks),
            nn.ReLU(),
            nn.Linear(self.max_join_masks, self.max_join_masks),
            nn.ReLU()
        )

        self.representation_contractor = nn.Linear(feature_dim+self.max_join_masks, self.feature_dim)

    def forward(self, observations):
        """
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
        joined = observations["joined"]
        n_triples_obs = observations["n_triples"]
        # Observations are batched, so we need to do forward passes on a per-batch basis
        # Furthermore, observations also requires a 'graph' to be constructed of edge_indexes and an 'order'
        # which is how we execute our n_ary_tree_lstm recursively. On the other hand, h, c can be zero tensors with the
        # right shape
        device = result_embeddings.device
        state_embeddings = []
        for i in range(result_embeddings.shape[0]):
            n_triples = n_triples_obs[i][0]
            x = result_embeddings[i]

            # Construct edge index in format of package with all padding edges removed (This can be done faster)
            edge_index_tree_lstm = join_graphs[i][[1, 0], :].T
            mask = ~(edge_index_tree_lstm == 0).all(dim=1)
            edge_index_join_tree = edge_index_tree_lstm[mask]

            # First join makes intermediate join with one edge, rest all 2 edges per join
            n_intermediate_join_nodes = math.ceil(mask.sum()/2)
            n_nodes_join_tree = n_triples+n_intermediate_join_nodes

            # Create final 'super' node representing the current join state by adding all un-joined triple patterns and
            # the intermediate result representation as children.
            # result_embeddings_un_joined = self.get_un_joined_embeddings(joined[i], x)
            index_un_joined = torch.where(~joined[i].bool())[0]
            index_forest_representation = torch.concat(
                (index_un_joined, torch.tensor([n_nodes_join_tree-1])),
                dim=0)
            edges_forest = torch.stack(
                (torch.full(index_forest_representation.shape, n_nodes_join_tree), index_forest_representation),
                dim=0
            ).T
            edge_index_join_forest = torch.concat((edge_index_join_tree, edges_forest), dim=0)
            node_order, edge_order = treelstm.calculate_evaluation_orders(edge_index_join_forest,
                                                                          n_intermediate_join_nodes+n_triples+1)

            x_input = x[:n_nodes_join_tree+1]

            h_final_lstm, c_final_lstm = self.tree_lstm_final(x_input, node_order, edge_index_join_forest, edge_order)
            state_embeddings.append(h_final_lstm[-1])

        mask_emb = self.mask_embedder(joined)
        return self.representation_contractor(torch.concat((torch.stack(state_embeddings), mask_emb), dim=1))

    @staticmethod
    def get_un_joined_embeddings(joined, x):
        # Add padding to joined tensor that masks out the x values corresponding to intermediate join features
        joined_padded = torch.nn.functional.pad(joined.bool(),
                                                (0, x.shape[0] - joined.shape[0]),
                                                value=True)
        return x[~joined_padded]

    @staticmethod
    def get_final_input_state_mask(joined, state):
        un_joined_indexes = torch.cat([
            (joined == 0).nonzero(as_tuple=True)[0],
            torch.tensor([state.shape[0] - 1], device=joined.device)
        ])
        return un_joined_indexes

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