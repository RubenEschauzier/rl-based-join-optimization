import rdflib
import random
import requests
from typing import Literal

from src.random_query_generation.generation_utils import convert_binding_to_rdflib, query_exhaustively, query_all_terms


def main_sample_paths():
    pass


def sample_all_paths(endpoint_url, default_graph_uri,
                     n_paths, max_size, path_start_type: Literal["?s", "?p"]):
    start_terms = query_all_terms(endpoint_url=endpoint_url,
                                  term_type=path_start_type,
                                  default_graph_uri=default_graph_uri)
    print(start_terms)


def sample_path(endpoint_url, default_graph_uri, seed_triple, size,
                sample_strategy: Literal['uniform', 'predicate'], unique_predicates):
    # Start subject is what acts as subject to get all predicates/objects to extend with
    start_subject = seed_triple[2]
    start_object = seed_triple[0]

    predicates_in_path = set(seed_triple[1])
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

        if sample_strategy == 'uniform':
            all_extensions = list(subj_extensions) + list(obj_extensions)
            # sample according to index to retrieve if we have an object or subject extension
            raise NotImplementedError
        elif sample_strategy == 'predicate':
            predicate_dict = {}
            for subj_ext in subj_extensions:
                # Ensure no duplicate predicates
                if unique_predicates and subj_ext[1] in predicates_in_path:
                    continue
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
            # Keep sampling until we sample a triple not yet in path

            predicate_sampled = random.choice(list(predicate_dict.keys()))
            sampled = random.choice(predicate_dict[predicate_sampled])

            path.append(sampled['value'])
            predicates_in_path.add(sampled['value'][1])

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
    sample_all_paths(endpoint_url="http://localhost:8890/sparql",
                     default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
                     n_paths=100,
                     max_size=5,
                     path_start_type="?p")
    # sample_path(endpoint_url="http://localhost:8890/sparql",
    #             default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
    #             seed_triple=(rdflib.term.URIRef('http://db.uwaterloo.ca/~galuc/wsdbm/Country7'),
    #                          rdflib.term.URIRef('http://www.geonames.org/ontology#parentCountry'),
    #                          rdflib.term.URIRef('http://db.uwaterloo.ca/~galuc/wsdbm/City118')),
    #             size=5,
    #             sample_strategy='predicate',
    #             unique_predicates=True
    #             )
    # subjs, objs = query_path_extension(endpoint_url="http://localhost:8890/sparql",
    #                                  default_graph_uri=['http://localhost:8890/watdiv-default-instantiation'],
    #                                  subj=rdflib.term.URIRef('http://db.uwaterloo.ca/~galuc/wsdbm/City118'),
    #                                  obj=rdflib.term.URIRef('http://db.uwaterloo.ca/~galuc/wsdbm/Country7'),
    #                                  limit=10000
    #                                  )
    test = 5
