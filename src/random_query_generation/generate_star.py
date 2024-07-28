# TODO: Implement query generation with an endpoint

import numpy as np
import requests
import random

ENDPOINT_LIMIT = 10000


def binding_to_triples():
    pass


# Adapted from https://github.com/DE-TUM/rdf-subgraph-sampler/
def query_triple(endpoint_url,
                 limit,
                 s="?s", p="?p", o="?o",
                 default_graph_uri=None):
    # Get the query and select statement based on supplied values for s p o
    spo = "{} {} {}".format(s, p, o)
    select = ""
    select += (" ?s" if s == "?s" else "")
    select += (" ?p" if p == "?p" else "")
    select += (" ?o" if o == "?o" else "")
    query_string = "SELECT DISTINCT" + select + " WHERE {" + spo + ". } " + \
                   "ORDER BY ASC(bif:rnd(2000000000)) LIMIT " + str(limit)
    print(query_string)
    r = requests.get(endpoint_url,
                     params={'query': query_string,
                             'format': 'json',
                             'default-graph-uri': default_graph_uri}
                     )
    print(r.text)
    res = r.json()["results"]["bindings"]

    subjects, predicates, objects = [], [], []

    for binding in res:
        if s == "?s":
            subjects.append('<' + binding['s']['value'] + '>')
        if p == "?p":
            predicates.append('<' + binding['p']['value'] + '>')
        if o == "?o":
            if binding['o']['type'] == "literal":
                objects.append(binding['o']['value'])
            else:
                objects.append('<' + binding['o']['value'] + '>')

    return subjects, predicates, objects


def sample_start_triples(endpoint_url, default_graph, limit, samples, p):
    if samples > limit:
        raise ValueError("Sampling more triples than available through limit")
    # Randomly sample triples with predicate
    subjects, _, objects = query_triple(endpoint_url,
                                        default_graph_uri=default_graph,
                                        p=p,
                                        limit=limit)
    # Form triples from the result by merging the lists and predicates into tuples
    triples_from_pred = list(map(lambda e: (e[0], p, e[1]), zip(subjects, objects)))
    sampled = random.sample(triples_from_pred, k=samples)
    return sampled


def sample_star(endpoint_url, default_graph, seed_triple, subject_star, size):
    if subject_star:
        # Sample triples with the given subject
        _, pred, obj = query_triple(endpoint_url, ENDPOINT_LIMIT, s=seed_triple[0], default_graph_uri=default_graph)
        star_triples = list(map(lambda e: (seed_triple[0], e[0], e[1]), zip(pred, obj)))
        print(star_triples)
        # Filter out seed_triple
    pass


if __name__ == "__main__":
    # sub, pred, obj = query_triple(endpoint_url="http://localhost:8890/sparql",
    #                               limit=10000,
    #                               p="<http://purl.org/stuff/rev#text>",
    #                               default_graph_uri=['http://localhost:8890/watdiv-default-instantiation']
    #                               )
    triples = sample_start_triples(endpoint_url="http://localhost:8890/sparql",
                                   default_graph=['http://localhost:8890/watdiv-default-instantiation'],
                                   limit=10,
                                   samples=5,
                                   p="<http://purl.org/stuff/rev#text>"
                                   )
    print("Seed triple: {}".format(triples[0]))
    sample_star(endpoint_url="http://localhost:8890/sparql",
                default_graph=['http://localhost:8890/watdiv-default-instantiation'],
                seed_triple=triples[0],
                subject_star=True,
                size=5)
