import json
import os
from typing import Tuple, Dict, Optional

import rdflib
import requests
from tqdm import tqdm

from src.datastructures.query import ProcessQuery


def read_queries(query_dir):
    dir_files = os.listdir(query_dir)
    queries_total = []
    for file in dir_files:
        if os.path.isdir(os.path.join(query_dir, file)):
            continue
        queries = []
        path = os.path.join(query_dir, file)
        with open(path, 'r') as f:
            raw_data = json.load(f)
        for i, data in tqdm(enumerate(raw_data)):
            tp_str, tp_rdflib = ProcessQuery.deconstruct_to_triple_pattern(data['query'])
            queries.append({
                "query": data['query'],
                "cardinality": data['y'],
                "triple_patterns": tp_str,
                "rdflib_patterns": tp_rdflib
            })
        queries_total.extend(queries)
    return queries_total


def calculate_occurrences(queries, env):
    occurrences = {}
    for query in tqdm(queries):
        for tp_rdflib in query['rdflib_patterns']:
            for entity in tp_rdflib:
                if type(entity) != rdflib.term.Variable and entity.n3() not in occurrences:
                    occurrence_entity = env.cardinality_term(entity.n3())
                    occurrences[entity.n3()] = occurrence_entity
    return occurrences


# Encodes dict {tp_str: cardinality}
def calculate_query_triple_pattern_cardinalities(queries, env):
    triple_pattern_cardinalities = {}
    for query in tqdm(queries):
        for tp_str in query['triple_patterns']:
            if tp_str not in triple_pattern_cardinalities:
                tp_cardinality = env.cardinality_triple_pattern(tp_str)
                triple_pattern_cardinalities[tp_str] = tp_cardinality
    return triple_pattern_cardinalities


def calculate_multiplicities_queries(queries, endpoint_url):
    print(endpoint_url)
    predicate_multiplicities = {}
    for query in tqdm(queries):
        for tp_str in query['triple_patterns']:
            predicate = tp_str.split()[1]
            if predicate not in predicate_multiplicities:
                multiplicity_predicate = extract_multiplicities(endpoint_url, predicate)
                predicate_multiplicities[predicate] = multiplicity_predicate[predicate]
    return predicate_multiplicities

def extract_multiplicities(
        endpoint_url: str,
        predicate: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Extracts Subject-to-Object (M_so) and Object-to-Subject (M_os) multiplicities.

    Args:
        endpoint_url: The URL of the SPARQL endpoint.
        predicate: Optional specific predicate IRI (e.g., '<http://example.org/pred>').
                   If None, extracts metrics for all predicates in the database.

    Returns:
        Dictionary mapping predicate IRIs to (M_so, M_os) tuples.
    """

    if predicate:
        query = f"""
        SELECT (COUNT(?s) AS ?triples) (COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects)
        WHERE {{
            ?s {predicate} ?o .
        }}
        """
    else:
        query = """
        SELECT ?p (COUNT(?s) AS ?triples) (COUNT(DISTINCT ?s) AS ?subjects) (COUNT(DISTINCT ?o) AS ?objects)
        WHERE {
            ?s ?p ?o .
        }
        GROUP BY ?p
        """

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(endpoint_url, data={"query": query}, headers=headers)
    response.raise_for_status()

    results = response.json()
    multiplicities = {}

    for binding in results["results"]["bindings"]:
        # If a specific predicate was passed, use it as the key; otherwise extract from binding
        p_iri = predicate if predicate else binding["p"]["value"]

        triples = float(binding["triples"]["value"])
        subjects = float(binding["subjects"]["value"])
        objects = float(binding["objects"]["value"])

        m_so = triples / subjects if subjects > 0 else 0.0
        m_os = triples / objects if objects > 0 else 0.0

        multiplicities[p_iri] = (m_so, m_os)

    return multiplicities