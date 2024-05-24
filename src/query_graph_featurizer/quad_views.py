import json

import numpy as np
import rdflib.term
from rdflib.term import Variable
from lib.datastructures.query import Query


class FeaturizeQueryGraphQuadViews:
    def __init__(self):
        pass

    def run(self, queries: [Query]):
        
        return queries

    def featurize_query(self, query: Query):
        return FeaturizeQueryGraphQuadViews.create_query_graph_views(query)
        pass

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
