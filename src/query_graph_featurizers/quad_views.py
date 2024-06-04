import json
from typing import Literal
import torch
import numpy as np
from rdflib.term import Variable
from src.datastructures.query import Query


class FeaturizeQueryGraphQuadViews:
    def __init__(self):
        pass

    def run(self, queries: [Query], format_graph: Literal["adj_matrix", "edge_index"]):
        for query in queries:
            s_s, o_o, s_o, o_s = self.featurize_query(query)
            if format_graph == "edge_index":
                s_s = self.convert_to_edge_index_format(s_s)
                o_o = self.convert_to_edge_index_format(o_o)
                s_o = self.convert_to_edge_index_format(s_o)
                o_s = self.convert_to_edge_index_format(o_s)
            query.query_graph_representations = [torch.tensor(s_s), torch.tensor(o_o),
                                                 torch.tensor(s_o), torch.tensor(o_s)]
        return queries

    @staticmethod
    def featurize_query(query: Query):
        return FeaturizeQueryGraphQuadViews.create_query_graph_views(query)

    @staticmethod
    def create_query_graph_views(query: Query):
        n_patterns = len(query.rdflib_tp)
        subj_subj_view = np.zeros((n_patterns, n_patterns))
        obj_obj_view = np.zeros((n_patterns, n_patterns))
        subj_obj_view = np.zeros((n_patterns, n_patterns))
        obj_subj_view = np.zeros((n_patterns, n_patterns))

        for i in range(len(query.rdflib_tp)):
            subj_subj_view[i][i] = 1
            obj_obj_view[i][i] = 1
            subj_obj_view[i][i] = 1
            obj_subj_view[i][i] = 1
            outer_pattern = query.rdflib_tp[i]
            for j in range(i, len(query.rdflib_tp)):
                inner_pattern = query.rdflib_tp[j]
                if type(inner_pattern[0]) == Variable and type(outer_pattern[0]) == Variable \
                        and inner_pattern[0] == outer_pattern[0]:
                    subj_subj_view[i][j] = 1
                    subj_subj_view[j][i] = 1
                if type(inner_pattern[2]) == Variable and type(outer_pattern[2]) == Variable \
                        and inner_pattern[2] == outer_pattern[2]:
                    obj_obj_view[i][j] = 1
                    obj_obj_view[j][i] = 1
                if type(inner_pattern[2]) == Variable and type(outer_pattern[2]) == Variable \
                        and inner_pattern[0] == outer_pattern[2]:
                    subj_obj_view[i][j] = 1
                    subj_obj_view[j][i] = 1
                if type(inner_pattern[2]) == Variable and type(outer_pattern[2]) == Variable \
                        and inner_pattern[2] == outer_pattern[0]:
                    obj_subj_view[i][j] = 1
                    obj_subj_view[j][i] = 1

        return subj_subj_view, obj_obj_view, subj_obj_view, obj_subj_view

    @staticmethod
    def convert_to_edge_index_format(adj_matrix):
        start_of_edge = []
        end_of_edge = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] == 1:
                    start_of_edge.append(i)
                    end_of_edge.append(j)
        edge_index = [start_of_edge, end_of_edge]
        return edge_index
