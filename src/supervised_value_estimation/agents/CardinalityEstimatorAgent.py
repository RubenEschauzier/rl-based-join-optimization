#TODO Use base cardinality estimation model, trained exactly like GNCE to estimate cardinality to guide beam search
# Also use G-Care framework
import numpy as np
import torch

import torch_geometric
from torch_geometric.data import Data, Batch

from src.supervised_value_estimation.agents.AbstractAgent import AbstractCostAgent


class CardinalityEstimatorValidationAgent(AbstractCostAgent):
    def __init__(self, model,
                 inference_queue, result_queue, worker_id,
                 estimator_fn,
                 estimator_requires_features = True,
                 ):
        self.model = model
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_id = worker_id

        self.current_query = None
        self.estimator_requires_features = estimator_requires_features
        self.estimator_fn = estimator_fn

    def setup_episode(self, query):
        """
        Starts a search episode. Samples a single z and embeds the query based on this z
        :param query:
        :return:
        """
        self.current_query = query
        return

    def estimate_costs(self, possible_next, query_state):
        # Format the plans
        formatted_plans = [(p,) for p in possible_next]

        # Construct subqueries from plans
        sub_queries = [self.reduced_form_query(self.current_query, plan) for plan in formatted_plans]

        # For ML-based methods (like GNCE) we pass the features as batch
        if self.estimator_requires_features:
            batch = Batch.from_data_list(sub_queries)
            estimates = self.estimator_fn(batch)
            return estimates, []

        # For cardinality estimators in G-Care we pass the subqueries to the estimator
        return self.estimator_fn(sub_queries)

    def reduced_form_query(self, query, join_order, device=torch.device('cpu')):
        # Only get set join_order, slice out all unset 0 padding entries of join order array
        reduced_triple_patterns = np.array(query.triple_patterns)[join_order]
        reduced_query_string = self.triple_patterns_to_query(reduced_triple_patterns)

        if not self.estimator_requires_features:
            return reduced_query_string

        sub_sampled_term_attributes, old_to_new_id = self.sub_sample_term_features(
            query.x,query.triple_patterns, query.term_to_id, join_order
        )

        sub_sampled_edge_attr, sub_sampled_edge_index = self.sub_sample_edges(
            query.edge_attr, query.edge_index, join_order, old_to_new_id
        )

        # Create a new batch tensor of zeros matching the new node count
        # We use the device of the subsampled attributes to ensure compatibility before the final move
        num_new_nodes = sub_sampled_term_attributes.shape[0]
        new_batch = torch.zeros(num_new_nodes, dtype=torch.long, device=sub_sampled_term_attributes.device)

        reduced_query_data = Data(
            x=sub_sampled_term_attributes,
            edge_index=sub_sampled_edge_index,
            edge_attr=sub_sampled_edge_attr,
            term_to_id=query.term_to_id,
            query = reduced_query_string,
            triple_patterns=reduced_triple_patterns,
            type=query.type,
            batch=new_batch)
        return reduced_query_data

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
