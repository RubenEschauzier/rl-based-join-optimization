import gymnasium as gym
import numpy as np
import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.data import Data

from src.query_environments.gym.query_gym_execution_feedback import QueryExecutionGymExecutionFeedback


class QueryGymCardinalityEstimationFeedback(QueryExecutionGymExecutionFeedback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            "lstm_order": gym.spaces.Box(low=0, high=1, shape=(self.max_triples-1,(self.max_triples-1)*2),
                                         dtype=np.int64),
            # What orders should be masked out as these are padding orders, 1 means it is NOT masked
            "lstm_order_mask": gym.spaces.Box(low=0, high=1, shape=(self.max_triples-1,),
                                              dtype=np.int64),

            "joins_made": gym.spaces.Discrete(self.max_triples),
        })
        self.join_graph = None
        self.lstm_order = None
        self.lstm_order_mask = None

    def reset(self, seed=None, options=None):
        self.join_graph = np.zeros((2, (self.max_triples-1)*2), dtype=np.int64)
        self.lstm_order = np.zeros((self.max_triples-1,(self.max_triples-1)*2), dtype=np.int64)
        self.lstm_order_mask = np.zeros((self.max_triples-1,), dtype=np.int64)
        obs, info = super().reset(seed=seed, options=options)

        # Add padding in result embedding for the intermediate join nodes that will be created
        intermediate_join_padding = torch.zeros(
            (self.max_triples-1, self._result_embeddings.shape[1]),
            device=self._result_embeddings.device, dtype=self._result_embeddings.dtype)
        self._result_embeddings = torch.cat([self._result_embeddings, intermediate_join_padding], dim=0)
        obs['result_embeddings'] = self._result_embeddings
        return obs, info

    def step(self, action):
        if action >= self.n_triples_query or self._joined[action] == 1:
            raise ValueError("Invalid action")

        self._joined[action] = 1
        # Join order is defined from 0 to max_triples + 1 for processing purposes. 0 denotes not made joins
        self.join_order[self.join_count] = action
        self.join_count += 1

        next_obs = self._build_obs()
        reward = self._get_reward(self._query,
                                  join_order=self.join_order,
                                  join_count=self.join_count)
        done = False
        if self.join_count >= self.n_triples_query:
            done = True

        return next_obs, reward, done, False, {}

    def _get_reward(self, query, join_order, join_count):
        query_to_estimate = self.reduced_form_query(query, join_order, join_count)
        output = self.query_embedder.forward(x=query_to_estimate.x,
                                       edge_index=query_to_estimate.edge_index,
                                       edge_attr=query_to_estimate.edge_attr,
                                       batch=query_to_estimate.batch)
        card =  next(head_output['output'] for head_output in output if head_output['output_type'] == 'cardinality')
        return -card

    def _build_obs(self):
        self._build_tree_input()

        return {
            "result_embeddings": self._result_embeddings,
            "join_order": self.join_order,
            "joined": self._joined,
            "join_graph": self.join_graph,
            "lstm_order": self.lstm_order,
            "lstm_order_mask": self.lstm_order_mask,
            "joins_made": self.join_count
        }

    def _build_tree_input(self):
        # First join
        if self.join_count == 2:
            # First join is a special case
            self.join_graph[0][0] = self.join_order[0]
            self.join_graph[1][0] = self.n_triples_query
            self.join_graph[0][1] = self.join_order[1]
            self.join_graph[1][1] = self.n_triples_query
            # First order mask set join node to 1 to represent it should get its hidden state computed
            self.lstm_order[0][self.n_triples_query] = 1
            # Unmask the order we just made to show model that this is a valid order entry
            self.lstm_order_mask[0] = 1
            test = 5
        # Subsequent joins
        elif self.join_count > 2:
            n_triple_patterns = self.n_triples_query
            n_join_nodes = self.join_count - 1

            # new_join = self.join_order[-1]
            # First two edges are added when join count = 2, then subsequent joins add two edges.
            index_to_add_edge = 2 + (self.join_count - 3)*2
            self.join_graph[0][index_to_add_edge] = self.join_order[self.join_count-1]
            # The index representing the join increments by one for each join, starting from n_tps - 1
            # The first two join counts increment it by 1, so subtract 1.
            self.join_graph[1][index_to_add_edge] = n_triple_patterns - 1 + self.join_count - 1
            # The previous join is always included in the next (left-deep)
            self.join_graph[0][index_to_add_edge+1] = n_triple_patterns - 1 + self.join_count - 1 - 1
            self.join_graph[1][index_to_add_edge+1] = n_triple_patterns - 1 + self.join_count - 1

            # Set the order array for the new join node
            self.lstm_order[self.join_count - 2][self.n_triples_query+(self.join_count-2)] = 1

            # Unmask this order
            self.lstm_order_mask[self.join_count - 2] = 1
        else:
            # No join yet, what to doooo
            pass


    def _build_order(self):
        pass

    def reduced_form_query(self, query, join_order, join_count):
        # Only get set join_order, slice out all unset 0 padding entries of join order array
        triple_indexes_to_include = join_order[:join_count]
        reduced_triple_patterns = np.array(query.triple_patterns)[triple_indexes_to_include]

        sub_sampled_term_attributes, old_to_new_id = self.sub_sample_term_features(query.x,
                                                                    query.triple_patterns,
                                                                    query.term_to_id,
                                                                    triple_indexes_to_include)
        sub_sampled_edge_attr, sub_sampled_edge_index = self.sub_sample_edges(query.edge_attr,
                                                                              query.edge_index,
                                                                              triple_indexes_to_include,
                                                                              old_to_new_id)

        reduced_query_string = self.triple_patterns_to_query(reduced_triple_patterns)
        # Don't need y as this query will only be used for prediction target in RL training
        reduced_query_data = Data(
            x=sub_sampled_term_attributes,
            edge_index=sub_sampled_edge_index,
            edge_attr=sub_sampled_edge_attr,
            term_to_id=query.term_to_id,
            query = reduced_query_string,
            triple_patterns=reduced_triple_patterns,
            type=query.type,
            batch=query.batch)
        return reduced_query_data


    @staticmethod
    def triple_patterns_to_query(triple_patterns):
        query = "SELECT * WHERE { \n"
        for tp in triple_patterns:
            query = query + "\t" + tp + "\n"
        query += "}"
        return query

    @staticmethod
    def sub_sample_edges(edge_attr, edge_index, indexes_to_include, old_to_new_indexes):
        indexes_to_include = torch.tensor(indexes_to_include)
        # Edges are assumed directed, so we have repeat entries to make undirected edges, so we need to repeat
        # the indexes to sample to get both directions of edge
        edge_indexes_to_sub_sample = indexes_to_include.repeat_interleave(2) * 2 + torch.tensor(
            [0, 1] * len(indexes_to_include))

        sub_sampled_edge_indexes = edge_index[:, edge_indexes_to_sub_sample]
        sub_sampled_edge_attr = edge_attr[edge_indexes_to_sub_sample]

        # Pytorch geometric requires the indexes of the edges to be matched with features so we map the old indexes
        # to a 'compact' index
        sub_sampled_edge_indexes_mapped = []
        for i in range(2):
            sub_sampled_edge_indexes_mapped.append([
                old_to_new_indexes[int(node_id)]
                for node_id in sub_sampled_edge_indexes[i]
            ])
        sub_sampled_edge_indexes_mapped = torch_geometric.EdgeIndex(sub_sampled_edge_indexes_mapped)
        return sub_sampled_edge_attr, sub_sampled_edge_indexes_mapped

    @staticmethod
    def sub_sample_term_features(term_features, triple_patterns, term_to_id, indexes_to_include):
        x_index_to_include = []
        for index in indexes_to_include:
            tp = triple_patterns[index].split(' ')
            if tp[0] not in term_to_id or tp[2] not in term_to_id:
                raise ValueError("Term {} or {} not in {}".format(tp[0], tp[2], term_to_id))
            x_index_to_include.append(int(term_to_id[tp[0]]))
            x_index_to_include.append(int(term_to_id[tp[2]]))
        old_to_new_id = {x: i for i, x in enumerate(x_index_to_include)}
        return term_features[x_index_to_include, :], old_to_new_id
