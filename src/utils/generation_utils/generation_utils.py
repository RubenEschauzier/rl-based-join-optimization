import json
import os.path
import re
from time import sleep
from typing import Literal
import random

import urllib3.exceptions
from rdflib.compare import to_isomorphic
from rdflib.graph import Graph
import rdflib
import requests
from rdflib.plugins.sparql.processor import prepareQuery

ENDPOINT_LIMIT = 10000


class WrapperIsomorphicGraphFixedHashing:
    def __init__(self, graph):
        self.iso_graph = graph

    def __hash__(self):
        return self.iso_graph.internal_hash()

    def __eq__(self, other):
        return True


def filter_isomorphic_queries(queries):
    accepted_iso_set = set()
    accepted_queries = []
    for query in queries:
        iso = to_isomorphic(convert_query_to_graph(query))
        if WrapperIsomorphicGraphFixedHashing(iso) in accepted_iso_set:
            continue
        accepted_iso_set.add(WrapperIsomorphicGraphFixedHashing(iso))
        accepted_queries.append(query)
    print("Filtered queries, before: {}, after: {}".format(len(queries), len(accepted_queries)))
    return accepted_queries


def convert_query_to_graph(query):
    g = Graph()
    bgp = re.search(r'(?<={)[^}]*(?=})', query).group().replace("}", "").replace("{", "")
    tps = [tp.strip() for tp in bgp.split(" . ")[:-1]]
    for tp in tps:
        spo = tp.strip().split(' ')
        triple_to_add = []
        # If subject is variable add BNode to represent variable
        if spo[0][0] == "?":
            triple_to_add.append(rdflib.term.BNode(spo[0]))
        # Add element as a literal, should not influence isomorphism determination
        else:
            triple_to_add.append(rdflib.term.Literal(spo[0]))

        triple_to_add.append(rdflib.term.URIRef(spo[1][1:-1]))

        if spo[2][0] == "?":
            triple_to_add.append(rdflib.term.BNode(spo[2]))
        else:
            triple_to_add.append(rdflib.term.Literal(spo[2]))
        g.add((triple_to_add[0], triple_to_add[1], triple_to_add[2]))
    return g


def create_variable_dictionary(triples):
    terms = set()
    for triple in triples:
        terms.add(triple[0])
        terms.add(triple[2])
    variable_dict = {}
    i = 0
    for term in terms:
        variable_dict[term] = "?v{}".format(i)
        i += 1
    return variable_dict


# Adapted from https://github.com/DE-TUM/rdf-subgraph-sampler/blob/main/samplers/star_query_generator.py
def cardinality_query(endpoint_url, query, query_timeout, default_graph_uri=None):
    sleep(0.0001)
    try:
        rn = requests.get(endpoint_url,
                          params={'query': "SELECT COUNT(*) as ?res WHERE { " + query + " }",
                                  'format': 'json'},
                          timeout=query_timeout,
                          )
    except:
        print("Timeout in cardinality query")
        return {}
    if rn.status_code == 200:
        res = rn.json()["results"]["bindings"]
        y = int(res[0]["res"]["value"])
        query_triples = deconstruct_to_triple_patterns(query)
        datapoint = {"y": y,
                     "query": query,
                     "triple-patterns": query_triples,
                     'default-graph-uri': default_graph_uri
                     }
        return datapoint
    else:
        print(rn)
        print("TEST!")


def deconstruct_to_triple_patterns(query):
    rdflib_query = prepareQuery(query)
    rdflib_triple_patterns: list[rdflib.term] = rdflib_query.algebra.get('p').get('p').get('triples')
    return rdflib_triple_patterns


"""
Helper function to convert bindings from virtuoso endpoint results to rdflib terms
Allows users to define what type of terms should not be passed as binding and should throw an error
"""


def convert_binding_to_rdflib(binding, disallowed_terms: set = None):
    if disallowed_terms and binding['type'] in disallowed_terms:
        raise ValueError('Received binding type: {}, which is in set of disallowed terms: {}'.format(
            binding['type'], disallowed_terms
        ))
    if binding['type'] == 'uri':
        result = rdflib.term.URIRef(binding['value'])
    elif binding['type'] == 'literal':
        result = rdflib.term.Literal(binding['value'])
    elif binding['type'] == 'variable':
        result = rdflib.term.Variable(binding['value'])
    elif binding['type'] == 'bnode':
        result = rdflib.term.Literal(binding['value'])
    else:
        raise ValueError("Unknown binding found: {}".format(binding))
    return result


def query_exhaustively(endpoint_url, default_graph_uri, query_string, limit):
    offset = 0
    responses = []
    while True:
        sleep(0.0001)
        # Make request per endpoint limit rows
        query = query_string + " OFFSET {}".format(offset)
        r = requests.get(endpoint_url,
                         params={'query': query,
                                 'format': 'json',
                                 'default-graph-uri': default_graph_uri}
                         )
        if r.status_code == 503:
            print(r)
            print("WARNING: 503 status code found")
        try:
            if len(r.json()['results']['bindings']) == 0:
                break
        except:
            print("Error decoding response:")
            print(r.text)

        responses.append(r)
        offset += limit
    return responses


# This function is partly redundant due to the query exhaustively function. Should refactor at some point :)
def query_all_terms(endpoint_url, term_type: Literal["?s", "?p", "?o"], default_graph_uri=None):
    # Create query string for querying the specified term type
    query_string = \
        "SELECT " + term_type + " WHERE { " \
                                "{ " "SELECT  DISTINCT  " + term_type + " WHERE " \
                                                                        "{ ?s  ?p  ?o } " \
                                                                        "ORDER BY ASC(" + term_type + ") " \
                                                                                                      "}" \
                                                                                                      "} LIMIT " \
        + str(ENDPOINT_LIMIT)
    offset = 0
    terms = []
    while True:
        sleep(0.0001)
        # Make request per endpoint limit rows
        query = query_string + " OFFSET {}".format(offset)
        r = requests.get(endpoint_url,
                         params={'query': query,
                                 'format': 'json',
                                 'default-graph-uri': default_graph_uri}
                         )
        # Try to decode the query response, catch a fail occurring due to the offset being larger than the number
        # of rows.
        try:
            res = r.json()["results"]["bindings"]
            # Empty results also means all predicates are found
            if len(res) == 0:
                break
            # Process found terms of type
            terms.extend(
                [rdflib.term.URIRef(binding[term_type[1]]['value']) if binding[term_type[1]]['type'] == 'uri'
                 else rdflib.term.Literal(binding[term_type[1]]['value']) for binding in res])

        except ValueError as err:
            break
        offset += ENDPOINT_LIMIT
    return terms


def sample_start_triples(endpoint_url, default_graph_uri, limit, samples,
                         s=rdflib.term.Variable("?s"), p=rdflib.term.Variable("?p")):
    if samples > limit:
        raise ValueError("Sampling more triples than available through limit")
    # Randomly sample triples with predicate
    subjects, predicates, objects = query_triple(endpoint_url,
                                                 default_graph_uri=default_graph_uri,
                                                 s=s, p=p,
                                                 limit=limit)
    # Form triples from the result by merging the lists and predicates into tuples
    triples_from_pred = list(map(lambda e: (e[0], e[1], e[2]), zip(subjects, predicates, objects)))
    # If we find less triples than we want to sample we just return all found triples
    if len(triples_from_pred) <= samples:
        return triples_from_pred

    # Return sample of triples
    return random.sample(triples_from_pred, k=samples)


# Adapted from https://github.com/DE-TUM/rdf-subgraph-sampler/
def query_triple(endpoint_url,
                 limit,
                 s=rdflib.term.Variable("?s"), p=rdflib.term.Variable("?p"), o=rdflib.term.Variable("?o"),
                 default_graph_uri=None):
    sleep(.001)
    # Get the query and select statement based on supplied values for s p o
    spo = "{} {} {}".format(s.n3(), p.n3(), o.n3())
    select = ""
    select += (" ?s" if s.n3() == "?s" else "")
    select += (" ?p" if p.n3() == "?p" else "")
    select += (" ?o" if o.n3() == "?o" else "")

    # Adapt query string to use SPARQL 1.1 RAND()
    query_string = "SELECT DISTINCT " + select + " WHERE {" + spo + " } " + \
                   "ORDER BY RAND() LIMIT " + str(limit)
    # SELECT DISTINCT  ?p ?o WHERE {<http://db.uwaterloo.ca/~galuc/wsdbm/Offer555> ?p ?o } ORDER BY RAND() LIMIT 10000
    # Triple: 5298
    # SELECT DISTINCT  ?p ?o WHERE {<http://db.uwaterloo.ca/~galuc/wsdbm/Offer5538> ?p ?o } ORDER BY RAND() LIMIT 10000
    # Triple: 5284

    r = requests.get(endpoint_url,
                     params={'query': query_string,
                             'format': 'json',
                             'default-graph-uri': default_graph_uri}
                     )
    # Sometimes gives error (service unavailable etc) if that is the case we have to retry the query
    if r.status_code == 503:
        print(r)
        print("503 found!!!!")
    res = r.json()["results"]["bindings"]

    # Get bindings from results, if any of the queried triple terms is not a variable we repeat the term in the
    # triple pattern
    subjects = [rdflib.term.URIRef(binding['s']['value']) if s.n3() == "?s" else s for binding in res]
    predicates = [rdflib.term.URIRef(binding['p']['value']) if p.n3() == "?p" else p for binding in res]

    objects = []
    for binding in res:
        if o.n3() == "?o":
            if binding['o']['type'] == "literal":
                objects.append(rdflib.term.Literal(binding['o']['value']))
            else:
                objects.append(rdflib.term.URIRef(binding['o']['value']))
        else:
            objects.append(o)

    return subjects, predicates, objects


def track_predicate_counts(predicate_counts, walk):
    for triple in walk:
        predicate_counts[triple[1]] += 1
    return predicate_counts


def save_queries_to_file(directory, file_name, queries):
    with open(os.path.join(directory, file_name), 'w') as f:
        json.dump(queries, f, indent=4)
