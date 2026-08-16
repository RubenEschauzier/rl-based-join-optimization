import json
import math
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
from tqdm import tqdm

ENDPOINT_LIMIT = 10000
# Configuration
SEED_SUBJECTS = 50000
SUBJECTS_BATCH = 50
QUERIES_PER_SEED = 2
FINAL_QUERY_TIMEOUT = 5

# Sampling Probabilities
P_BOUND_SUBJECT = 0.1
P_BOUND_PREDICATE = 0.9
P_BOUND_OBJECT = 0.2
P_ZERO_CARDINALITY = 0.15  # 15% of queries will be forced to 0

MAX_COMBINATIONS_PER_STAR = 50

# QLever requires standard Accept headers for JSON SPARQL results
QLEVER_HEADERS = {'Accept': 'application/sparql-results+json'}


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
        iso = to_isomorphic(convert_query_to_graph(query["query"]))
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
        if len(spo) < 3:
            test = 5
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


def query_sparql(endpoint_url, query_string, limit, offset=0):
    """Execute a single SPARQL query against the endpoint."""
    params = {
        'query': f"{query_string} LIMIT {limit} OFFSET {offset}",
        'action': 'sparql-query',  # Required by some QLever configurations
    }
    headers = {
        'Accept': 'application/sparql-results+json'
    }

    response = requests.get(endpoint_url, params=params, headers=headers)

    if response.status_code == 503:
        print("WARNING: 503 status code found")

    return response


def query_exhaustively(endpoint_url, query_string, limit):
    """Fetch all results for a query using pagination."""
    offset = 0
    all_bindings = []

    while True:
        sleep(0.0001)
        response = query_sparql(endpoint_url, query_string, limit, offset)

        try:
            data = response.json()
            bindings = data.get('results', {}).get('bindings', [])

            if not bindings:
                break

            all_bindings.extend(bindings)
            offset += limit
        except (ValueError, KeyError) as e:
            print(f"Error decoding response at offset {offset}: {e}")
            break

    return all_bindings


def query_all_terms(endpoint_url, term_type: Literal["?s", "?p", "?o"], limit):
    """Retrieve all distinct terms of a specific type."""
    var_name = term_type[1:]  # Remove '?' for dict keys
    query_string = (
        f"SELECT DISTINCT {term_type} WHERE {{ ?s ?p ?o }} "
        f"ORDER BY ASC({term_type})"
    )

    bindings = query_exhaustively(endpoint_url, query_string, limit)

    terms = []
    for binding in bindings:
        val = binding[var_name]['value']
        if binding[var_name]['type'] == 'uri':
            terms.append(rdflib.term.URIRef(val))
        else:
            terms.append(rdflib.term.Literal(val))

    return terms

def sample_start_triples(endpoint_url, limit, samples,
                         s=rdflib.term.Variable("?s"), p=rdflib.term.Variable("?p")):
    if samples > limit:
        raise ValueError("Sampling more triples than available through limit")
    # Randomly sample triples with predicate
    subjects, predicates, objects = query_triple(endpoint_url,
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

def get_total_subject_count(endpoint_url: str) -> int:
    query = "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE { ?s ?p ?o . }"
    r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
    r.raise_for_status()
    return int(r.json()["results"]["bindings"][0]["count"]["value"])


def get_global_predicates(endpoint_url: str) -> list:
    query = "SELECT DISTINCT ?p WHERE { ?s ?p ?o . }"
    r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
    return [d['p']['value'] for d in r.json()["results"]["bindings"]]

def get_random_seed_subjects(endpoint_url: str, total_subjects: int) -> list:
    subjects = []
    num_requests = math.ceil(SEED_SUBJECTS / ENDPOINT_LIMIT)
    offsets = [random.randint(0, max(0, total_subjects - ENDPOINT_LIMIT)) for _ in range(num_requests)]

    for offset in tqdm(offsets, desc="Fetching Random Subject Batches"):
        query = f"SELECT DISTINCT ?s WHERE {{ ?s ?p ?o . }} LIMIT {ENDPOINT_LIMIT} OFFSET {offset}"
        try:
            r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
            bindings = r.json()["results"]["bindings"]
            subjects.extend([f"<{d['s']['value']}>" for d in bindings if d['s']['type'] == 'uri'])
        except Exception:
            continue

    return list(set(subjects))

def save_queries_to_file(directory, file_name, queries):
    with open(os.path.join(directory, file_name), 'w') as f:
        json.dump(queries, f, indent=4)


