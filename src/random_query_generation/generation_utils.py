import re
from rdflib.compare import to_isomorphic
from rdflib.graph import Graph
import rdflib
import requests
from rdflib.plugins.sparql.processor import prepareQuery


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
    subj, obj = set(), set()
    for triple in triples:
        subj.add(triple[0])
        obj.add(triple[2])
    variable_dict = {}
    i = 0
    for s in subj:
        variable_dict[s] = "?v{}".format(i)
        i += 1
    for o in obj:
        variable_dict[o] = "?v{}".format(i)
        i += 1
    return variable_dict


# Adapted from https://github.com/DE-TUM/rdf-subgraph-sampler/blob/main/samplers/star_query_generator.py
def cardinality_query(endpoint_url, query, query_timeout, default_graph_uri=None):
    rn = requests.get(endpoint_url,
                      params={'query': "SELECT COUNT(*) as ?res WHERE { " + query + " }",
                              'format': 'json'},
                      timeout=query_timeout)
    if rn.status_code == 200:
        res = rn.json()["results"]["bindings"]
        y = int(res[0]["res"]["value"])
        query_triples = deconstruct_to_triple_patterns(query)
        datapoint = {"y": y,
                     "query": query,
                     }


def deconstruct_to_triple_patterns(query):
    rdflib_query = prepareQuery(query)
    rdflib_triple_patterns: list[rdflib.term] = rdflib_query.algebra.get('p').get('p').get('triples')
    return 5
    print(query)
    print(rdflib_triple_patterns)
