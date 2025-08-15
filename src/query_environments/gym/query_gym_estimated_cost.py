import gymnasium as gym
import numpy as np
import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.data import Data

from src.baselines.enumeration import build_adj_list, JoinOrderEnumerator
from src.query_environments.gym.query_gym_base import QueryGymBase


class QueryGymEstimatedCost(QueryGymBase):

    def get_reward(self, query, join_order, joins_made):
        query_to_estimate = self.reduced_form_query(query,
                                                    join_order,
                                                    joins_made)
        output = self._query_embedder.forward(x=query_to_estimate.x,
                                              edge_index=query_to_estimate.edge_index,
                                              edge_attr=query_to_estimate.edge_attr,
                                              batch=query_to_estimate.batch)
        card =  next(head_output['output'] for head_output in output if head_output['output_type'] == 'cardinality')
        return -card, None


    @staticmethod
    def reduced_form_query(query, join_order, join_count):
        # Only get set join_order, slice out all unset 0 padding entries of join order array
        triple_indexes_to_include = join_order[:join_count]
        reduced_triple_patterns = np.array(query.triple_patterns)[triple_indexes_to_include]

        (sub_sampled_term_attributes,
         old_to_new_id) = QueryGymEstimatedCost.sub_sample_term_features(query.x,
                                                                    query.triple_patterns,
                                                                    query.term_to_id,
                                                                    triple_indexes_to_include)
        (sub_sampled_edge_attr,
         sub_sampled_edge_index) = QueryGymEstimatedCost.sub_sample_edges(query.edge_attr,
                                                                              query.edge_index,
                                                                              triple_indexes_to_include,
                                                                              old_to_new_id)

        reduced_query_string = QueryGymEstimatedCost.triple_patterns_to_query(reduced_triple_patterns)
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
