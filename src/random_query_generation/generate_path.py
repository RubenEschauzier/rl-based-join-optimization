from collections import defaultdict

import numpy as np
import rdflib
import random
from typing import Literal
from scipy.special import softmax

from tqdm import tqdm
from numpy.random import choice

from src.random_query_generation.generation_utils import convert_binding_to_rdflib, query_exhaustively, \
    query_all_terms, sample_start_triples, create_variable_dictionary, filter_isomorphic_queries, cardinality_query, \
    save_queries_to_file, track_predicate_counts


def main_sample_paths(endpoint_url, default_graph_uri, n_paths, max_size, proportion_unique_predicates,
                      path_start_type, output_dir, file_name_non_literal, file_name_literal):
    path_non_literal, path_literal = generate_path_queries(endpoint_url, default_graph_uri, n_paths, max_size,
                                                           proportion_unique_predicates, path_start_type, output_dir)
    save_queries_to_file(output_dir, file_name_non_literal, path_non_literal)
    save_queries_to_file(output_dir, file_name_literal, path_literal)


def generate_path_queries(endpoint_url, default_graph_uri, n_paths, max_size, proportion_unique_predicates,
                          path_start_type, output_dir):
    sampled_paths, predicate_counts = sample_all_paths(endpoint_url,
                                                       default_graph_uri,
                                                       n_paths,
                                                       max_size,
                                                       proportion_unique_predicates,
                                                       path_start_type)
    save_queries_to_file(output_dir, "predicate_counts_path.json", predicate_counts)
    queries_no_literal = filter_isomorphic_queries(triples_to_query(sampled_paths, False))
    queries_literal = filter_isomorphic_queries(triples_to_query(sampled_paths, True))
    data_path_no_literal = [cardinality_query(endpoint_url, query, 30, default_graph_uri)
                            for query in tqdm(queries_no_literal)]
    data_path_literal = [cardinality_query(endpoint_url, query, 30, default_graph_uri)
                         for query in tqdm(queries_literal)]
    return data_path_no_literal, data_path_literal


def triples_to_query(path_walks, literal_in_path):
    queries = []
    for path in path_walks:
        # Create dictionary mapping terms to variable names
        variable_dict = create_variable_dictionary(path)
        # Randomly choose a triple to add literal / uri in query (if literal_in_path = True)
        possible_literal_locations = [(key, value) for (key, value) in variable_dict.items()
                                      if isinstance(key, rdflib.term.URIRef)]
        literal_variable = random.choice(possible_literal_locations)

        query = "SELECT * WHERE { \n "
        for i, triple in enumerate(path):
            triple_string = "{} {} {} . \n "
            triple_string = triple_string.format(variable_dict[triple[0]],
                                                 triple[1].n3(),
                                                 variable_dict[triple[2]])
            query += triple_string

        query += "}"
        if literal_in_path:
            query = query.replace(literal_variable[1], literal_variable[0].n3())
        queries.append(query)
    return queries


def sample_all_paths(endpoint_url, default_graph_uri,
                     n_paths, max_size, proportion_unique_predicates, path_start_type: Literal["?s", "?p"]):
    generated_paths = []
    start_terms = query_all_terms(endpoint_url=endpoint_url,
                                  term_type=path_start_type,
                                  default_graph_uri=default_graph_uri)
    predicate_counts = defaultdict(int)
    for term in tqdm(start_terms):
        start_triples = []
        # Query start triples according to the star seed type in the function
        if path_start_type == "?s":
            start_triples = sample_start_triples(endpoint_url=endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 limit=10000,
                                                 samples=n_paths,
                                                 s=term)
        if path_start_type == "?p":
            start_triples = sample_start_triples(endpoint_url=endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 limit=10000,
                                                 samples=n_paths,
                                                 p=term)
        for triple in start_triples:
            path_size = random.randint(2, max_size)
            # Randomly decide whether path has unique predicates in walk or not. Most paths should probably have unique
            # predicates, or else most paths will include most common predicates. For example, in watdiv, friendOf and
            # likes relations are very common and will 1. occur very often, 2. generate extremely large result sets.
            # This flag only works with 'predicate' sampling
            unique_predicates = True
            if random.random() > proportion_unique_predicates:
                unique_predicates = False

            path_walk = sample_path(endpoint_url, default_graph_uri,
                                    seed_triple=triple,
                                    size=path_size,
                                    sample_strategy='predicate',
                                    unique_predicates=unique_predicates,
                                    predicate_counts=predicate_counts)

            # If random generation of path failed ignore it
            if len(path_walk) < 2:
                continue
            predicate_counts = track_predicate_counts(predicate_counts, path_walk)
            generated_paths.append(path_walk)
    return generated_paths, predicate_counts


def sample_path(endpoint_url, default_graph_uri, seed_triple, size,
                sample_strategy: Literal['uniform', 'predicate'], unique_predicates,
                predicate_counts=None):
    # Start subject is what acts as subject to get all predicates/objects to extend with
    start_subject = seed_triple[2]
    start_object = seed_triple[0]

    predicates_in_path = {seed_triple[1]}
    path = [seed_triple]

    for i in range(size):
        subj_extensions, obj_extensions = query_path_extension(endpoint_url=endpoint_url,
                                                               default_graph_uri=default_graph_uri,
                                                               limit=10000,
                                                               subj=start_subject,
                                                               obj=start_object
                                                               )
        # Filter out already traversed triples
        for triple in path:
            subj_extensions.discard(triple)
            obj_extensions.discard(triple)

        # Uniform strategy simply chooses an extending triple randomly
        if sample_strategy == 'uniform':
            all_extensions = list(subj_extensions) + list(obj_extensions)
            # sample according to index to retrieve if we have an object or subject extension
            raise NotImplementedError
        # Sample each predicate with equal probability. Then choose a triple with that predicate randomly
        elif sample_strategy == 'predicate':
            predicate_dict = {}
            for subj_ext in subj_extensions:
                # Ensure no duplicate predicates
                if unique_predicates and subj_ext[1] in predicates_in_path:
                    continue
                # Construct predicate - extension value dictionary
                if subj_ext[1] in predicate_dict:
                    predicate_dict[subj_ext[1]].append({'type': 'subject', 'value': subj_ext})
                else:
                    predicate_dict[subj_ext[1]] = [{'type': 'subject', 'value': subj_ext}]
            for obj_ext in obj_extensions:
                if unique_predicates and obj_ext[1] in predicates_in_path:
                    continue
                if obj_ext[1] in predicate_dict:
                    predicate_dict[obj_ext[1]].append({'type': 'object', 'value': obj_ext})
                else:
                    predicate_dict[obj_ext[1]] = [{'type': 'object', 'value': obj_ext}]

            # If no valid triples, stop generation
            if len(predicate_dict.keys()) == 0:
                break

            if predicate_counts is not None:
                predicates_possible = list(predicate_dict.keys())
                index_array = np.arange(len(predicates_possible))
                counts = np.array([predicate_counts[predicates_possible[i]] if predicates_possible[i]
                                                                               in predicate_counts else 1 for i in
                                   index_array])
                probability = softmax(-counts)

                sample_index = np.random.choice(index_array, p=probability)
                predicate_sampled = predicates_possible[sample_index]
            else:
                predicate_sampled = random.choice(list(predicate_dict.keys()))

            sampled = random.choice(predicate_dict[predicate_sampled])

            path.append(sampled['value'])
            predicates_in_path.add(sampled['value'][1])

            # Update the 'heads' of the path
            if sampled['type'] == 'subject':
                start_object = sampled['value'][0]
            if sampled['type'] == 'object':
                start_subject = sampled['value'][2]
        else:
            raise NotImplementedError

    return path


def query_path_extension(endpoint_url, default_graph_uri, limit, subj, obj):
    # If subj and obj are part of one triple, this will include the triple too. This must be filtered by
    # any code calling the function
    query_string_obj_side_extension = "SELECT ?p ?o WHERE { { SELECT DISTINCT ?p ?o WHERE " + \
                                      "{ " + subj.n3() + " ?p ?o . } " + \
                                      "ORDER BY ASC(?o) } } LIMIT " + str(limit)
    query_string_subj_side_extension = "SELECT * WHERE { { SELECT DISTINCT ?s ?p WHERE { " + \
                                       " ?s ?p " + obj.n3() + ". } " + \
                                       "ORDER BY ASC (?s) } } LIMIT " + str(limit)
    r_subj_extensions = query_exhaustively(endpoint_url, default_graph_uri, query_string_subj_side_extension, limit)
    r_obj_extensions = query_exhaustively(endpoint_url, default_graph_uri, query_string_obj_side_extension, limit)

    res_subj, res_obj = [], []
    for r_subj_ext in r_subj_extensions:
        res_subj.extend(r_subj_ext.json()["results"]["bindings"])
    for r_obj_ext in r_obj_extensions:
        res_obj.extend(r_obj_ext.json()["results"]["bindings"])

    obj_extensions = set([(subj, convert_binding_to_rdflib(binding['p']), convert_binding_to_rdflib(binding['o']))
                          for binding in res_obj])
    subj_extensions = set([(convert_binding_to_rdflib(binding['s']), convert_binding_to_rdflib(binding['p']), obj)
                           for binding in res_subj])
    return subj_extensions, obj_extensions


if __name__ == "__main__":
    main_sample_paths(endpoint_url="http://localhost:8890/sparql",
                      default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
                      n_paths=1,
                      max_size=5,
                      proportion_unique_predicates=.95,
                      path_start_type="?p",
                      output_dir=r"C:\Users\ruben\projects\rl-based-join-optimization\data\pretrain_data"
                                 r"\generated_queries",
                      file_name_literal="path_with_literal.json",
                      file_name_non_literal="path_without_literal.json")
