import json

from src.baselines.enumeration import JoinOrderEnumerator, build_adj_list
from src.datastructures.filter_duplicate_predicate_queries import filter_duplicate_subject_predicate_combinations
from src.datastructures.query_pytorch_dataset import QueryCardinalityDataset
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
import os

from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.utils.training_utils.query_loading_utils import load_featurizer

dataset_to_g_care = {
    'yago': '<http://example.com/13000179>',
    'yago_inductive': '<http://example.com/13000179>',
    'wikidata': '<http://www.wikidata.org/prop/direct/P31>',
    'swdf': '<http://ex.org/03>',
    'lubm': '<http://example.org/1>',
    'watdiv': ""
}


def query_to_g_care(query, id_to_id_mapping, id_to_id_mapping_predicate, dataset_name):
    """
    Original author: https://github.com/DE-TUM/GNCE/blob/master/GCARE/transform_query.py (function `query_to_gcare`)
    Modified by: Ruben Eschauzier
    """
    vertices = set()
    vertex_labels = {}
    # Get unique vertices
    for tp in query:
        vertices.add(tp[0])
        vertices.add(tp[2])

        rdf_type_uri = dataset_to_g_care[dataset_name]

        if not rdf_type_uri:
            raise AssertionError("rdf type uri missing !")

        if (tp[1] == rdf_type_uri) and ('?' not in tp[2]):

            if tp[0] in vertex_labels:
                vertex_labels[tp[0]] += [tp[2]]
            else:
                vertex_labels[tp[0]] = [tp[2]]

    # Creating Vertex Dict
    vertex_dict = {}
    vid = 0
    for vertex in vertices:
        # dvid = vertex.split("/")[-1].replace(">", "") if not "?" in vertex else -1
        try:
            dvid = id_to_id_mapping[vertex] if not "?" in vertex else -1
        except KeyError:
            dvid = 76711
        if vertex in vertex_labels:
            # print('In vertex')
            try:
                labels = [id_to_id_mapping[v] for v in vertex_labels[vertex]]
            except KeyError:
                labels = [-1]

        else:
            labels = [-1]
        vertex_dict[vertex] = [vid] + labels + [dvid]
        vid += 1

    # Creating Edge List
    edge_list = []
    for tp in query:
        edge_label = id_to_id_mapping_predicate[tp[1]] if not "?" in tp[1] else -1

        edge_list.append([vertex_dict[tp[0]][0], vertex_dict[tp[2]][0], edge_label])

    return vertex_dict, edge_list


def write_g_care_to_file(filename, vertex_dict, edge_list, query_idx):
    # Writing the Query File
    with open(filename, "w") as f:
        f.write("t # s " + str(query_idx))
        f.write("\n")
        for v in vertex_dict:
            label_str = ''
            for l in vertex_dict[v][1:-1]:
                label_str += str(l)
                label_str += ' '
            f.write("v " + str(vertex_dict[v][0]) + " " + label_str + str(vertex_dict[v][2]))
            f.write("\n")
        for e in edge_list:
            f.write("e " + str(e[0]) + " " + str(e[1]) + " " + str(e[2]))
            f.write("\n")


def map_dataset_to_g_care_query_files(torch_query_dataset, dataset_name, id_to_id_mapping, id_to_id_mapping_predicate,
                                      base_output_location):
    os.makedirs(os.path.join(base_output_location, dataset_name), exist_ok=True)
    file_name_to_query = {}
    query_dir_template = "query_{}"
    query_name_template = "sub_query_{}.txt"
    for i, query in enumerate(torch_query_dataset):
        query_dir = query_dir_template.format(i)
        file_name_to_query[query_dir] = query
        with open(os.path.join(base_output_location, dataset_name, "file_name_to_query.json"), "w") as f:
            json.dump(file_name_to_query, f, indent=2)
        vertex_dict, edge_list = query_to_g_care(query, id_to_id_mapping, id_to_id_mapping_predicate, dataset_name)
        sub_queries = map_query_to_sub_queries(query)
        for j, sub_query in enumerate(sub_queries):
            file_name_sub_query = os.path.join(base_output_location,
                                               dataset_name, query_dir,
                                               query_name_template.format(i))
            # TODO: Convert subquery to a triple pattern list
        # Map file name to query in dictionary and save. Do it every iteration to ensure we can come back when fails
        # Then name all files based on keys used in enumeration, and also name the output of g-care based on the
        # keys used in enumeration. We can then import these as keys to be used in enumeration in the code, because
        # that also uses these keys to get cardinalities
        pass

    pass


def map_query_string_to_g_care(query_string):
    pass


def map_query_to_sub_queries(query):
    n_entries = len(query.triple_patterns)
    adj_list = build_adj_list(query)
    enumerator = JoinOrderEnumerator(adj_list, lambda x: 1, n_entries)
    sub_queries = find_sub_queries(n_entries, enumerator)
    sub_query_strings = []
    for sub_query in sub_queries:
        sub_query_string = BlazeGraphQueryEnvironment.set_join_order_json_query(query.query,
                                                                                list(sub_query),
                                                                                query.triple_patterns)
        sub_query_string.replace('hint:Query hint:optimizer "None" . \n', '')
        sub_query_strings.append(sub_query_string)
    return sub_query_strings


def find_sub_queries(n_entries, enumerator):
    sub_queries = set()
    # Initialize singleton plans
    for i in range(n_entries):
        singleton_key = (i,)
        sub_queries.add(singleton_key)

    # Enumerate all connected subgraph-complement pairs
    csg_cmp_pairs = enumerator.enumerate_csg_cmp_pairs(n_entries)
    for csg_cmp_pair in csg_cmp_pairs:
        csg, cmp = csg_cmp_pair[0], csg_cmp_pair[1]
        tree1_key = set(enumerator.sort_array_asc(list(csg)))
        tree2_key = set(enumerator.sort_array_asc(list(cmp)))

        new_entries = tree1_key | tree2_key

        estimate_key = tuple(enumerator.sort_array_asc(list(new_entries)))
        sub_queries.add(estimate_key)

    return sub_queries

if __name__ == "__main__":
    # Helper script that converts a query to all sub_queries and converts them to the format used by G-CARE so G-CARE
    # Can estimate cardinality for each sub-query.
    # TEMP This should point to the processed validation queries for each dataset
    dataset_location = r"C:\Users\ruben\projects\rl-based-join-optimization\data/pretrain_data/datasets/p_e_size_3_5_101"
    rdf2vec_vectors_location = r"C:\Users\ruben\projects\rl-based-join-optimization\data\input\rdf2vec_embeddings\rdf2vec_vectors_depth_2_quick.json"
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"
    raw_data_dir = r"C:\Users\ruben\projects\rl-based-join-optimization\data/pretrain_data/datasets/p_e_size_3_5_101/raw"
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vectors_location)
    env = BlazeGraphQueryEnvironment(endpoint_location)

    # query_location_dict:
    #   queries: "data/pretrain_data/datasets/p_e_size_3_5_101"
    #   rdf2vec_vectors: "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"
    #   occurrences: "data/pretrain_data/pattern_term_cardinalities/full/occurrences.json"
    #   tp_cardinalities: "data/pretrain_data/pattern_term_cardinalities/full/tp_cardinalities.json"
    featurizer_edge_labeled_graph = load_featurizer("predicate_edge",
                                                    vectors, env,
                                                    rdf2vec_vectors_location, endpoint_location)
    post_processor = filter_duplicate_subject_predicate_combinations

    dataset = QueryCardinalityDataset(root=dataset_location,
                                      featurizer=featurizer_edge_labeled_graph,
                                      post_processor=post_processor,
                                      load_mappings=True,
                                      raw_data_dir=raw_data_dir,)
    for data in dataset:
        print(data)
        break
