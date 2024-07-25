# TODO: Implement query generation with an endpoint

import requests


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

    r = requests.get(endpoint_url,
                     params={'query': query_string,
                             'format': 'json',
                             'default-graph-uri': default_graph_uri}
                     )
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
    return list(map(lambda e: (e[0], p, e[1]), zip(subjects, objects)))


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
    print(triples)
