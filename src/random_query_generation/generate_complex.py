import json
import requests
import random
import math
from datetime import datetime
import os
from tqdm import tqdm

from src.random_query_generation.generation_validation import analyze_query_dataset_statistics
from src.utils.generation_utils.generation_utils import filter_isomorphic_queries

# Configuration
SEED_SUBJECTS = 50000
SUBJECTS_BATCH = 50
ENDPOINT_LIMIT = 500
QUERIES_PER_SEED = 2
FINAL_QUERY_TIMEOUT = 5

# Sampling Probabilities
P_BOUND_SUBJECT = 0.1
P_BOUND_PREDICATE = 0.95
P_BOUND_OBJECT = 0.1
P_ZERO_CARDINALITY = 0.15

QLEVER_HEADERS = {'Accept': 'application/sparql-results+json'}


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
        query = f"SELECT DISTINCT ?s WHERE {{ ?s ?p ?o . FILTER(isIRI(?s)) }} LIMIT {ENDPOINT_LIMIT} OFFSET {offset}"
        try:
            r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
            bindings = r.json()["results"]["bindings"]
            subjects.extend([d['s']['value'] for d in bindings])
        except Exception:
            continue

    return list(set(subjects))


def extract_random_subgraph(endpoint_url: str, seed_node: str, n_triples: int) -> list:
    """Performs a random walk to extract a valid, connected subgraph of arbitrary shape."""
    triples = set()
    active_nodes = {seed_node}
    explored_nodes = set()

    while len(triples) < n_triples and active_nodes:
        current_node = random.choice(list(active_nodes))

        # Fetch both outgoing and incoming edges to allow bidirectional traversal
        query = f"""
        SELECT ?s ?p ?o WHERE {{
            {{ <{current_node}> ?p ?o . BIND(<{current_node}> AS ?s) FILTER(isIRI(?o)) }}
            UNION
            {{ ?s ?p <{current_node}> . BIND(<{current_node}> AS ?o) FILTER(isIRI(?s)) }}
        }} LIMIT 200
        """
        try:
            r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query}, timeout=FINAL_QUERY_TIMEOUT)
            bindings = r.json().get("results", {}).get("bindings", [])

            candidates = []
            for b in bindings:
                t = (b['s']['value'], b['p']['value'], b['o']['value'])
                if t not in triples:
                    candidates.append(t)

            if not candidates:
                active_nodes.remove(current_node)
                explored_nodes.add(current_node)
                continue

            chosen_triple = random.choice(candidates)
            triples.add(chosen_triple)

            s, _, o = chosen_triple
            if s not in explored_nodes: active_nodes.add(s)
            if o not in explored_nodes: active_nodes.add(o)

        except Exception:
            active_nodes.remove(current_node)
            explored_nodes.add(current_node)

    return list(triples) if len(triples) == n_triples else []


def build_arbitrary_query(concrete_triples: list, seed_node: str, force_zero: bool, global_predicates: list) -> tuple:
    nodes = {}
    preds = {}
    node_counter = 0
    pred_counter = 0

    # Map URIs to variables
    for s, p, o in concrete_triples:
        if s not in nodes:
            nodes[s] = f"?v{node_counter}"
            node_counter += 1
        if o not in nodes:
            nodes[o] = f"?v{node_counter}"
            node_counter += 1
        if p not in preds:
            preds[p] = f"?p{pred_counter}"
            pred_counter += 1

    # Determine binding probability based on node proximity to the seed
    bound_nodes = {}
    for n in nodes:
        prob = P_BOUND_SUBJECT if n == seed_node else P_BOUND_OBJECT
        bound_nodes[n] = random.random() < prob

    bound_preds = {p: (random.random() < P_BOUND_PREDICATE) for p in preds}

    if force_zero and global_predicates:
        mutate_p = random.choice(list(preds.keys()))
        invalid_p = random.choice([x for x in global_predicates if x not in preds])

        # Replace the predicate in the triples and force it to be bound
        concrete_triples = [(s, invalid_p if p == mutate_p else p, o) for s, p, o in concrete_triples]
        bound_preds[invalid_p] = True
        preds[invalid_p] = preds[mutate_p]

    where_clauses = []
    entities = []

    for s, p, o in concrete_triples:
        bind_s = bound_nodes.get(s, False)
        bind_p = bound_preds.get(p, False)
        bind_o = bound_nodes.get(o, False)

        # Enforce at least one bound element per triple to prevent execution timeouts
        if not bind_s and not bind_p and not bind_o:
            bind_p = True
            bound_preds[p] = True

        s_str = f"<{s}>" if bind_s else nodes[s]
        p_str = f"<{p}>" if bind_p else preds[p]
        o_str = f"<{o}>" if bind_o else nodes[o]

        where_clauses.append(f"{s_str} {p_str} {o_str} .")

        if bind_s and s not in entities: entities.append(s)
        if bind_p and p not in entities: entities.append(p)
        if bind_o and o not in entities: entities.append(o)

    count_query = f"SELECT (COUNT(*) AS ?res) WHERE {{ {' '.join(where_clauses)} }}"
    raw_query = f"SELECT * WHERE {{ {' '.join(where_clauses)} }}"

    return count_query, raw_query, entities, where_clauses


def get_queries(
        dataset_dir: str,
        dataset_name: str,
        n_triples: int = 1,
        n_queries: int = 1000,
        endpoint_url: str = None):
    print("Initializing global graph statistics...")
    total_subjects = get_total_subject_count(endpoint_url)
    global_predicates = get_global_predicates(endpoint_url)
    subjects = get_random_seed_subjects(endpoint_url, total_subjects)

    if not subjects:
        print("No valid subjects found.")
        return []

    testdata = []
    seen_queries = set()

    pbar = tqdm(total=n_queries, desc="Generating Queries")
    while len(testdata) < n_queries:
        seed = random.choice(subjects)

        concrete_triples = extract_random_subgraph(endpoint_url, seed, n_triples)
        if not concrete_triples:
            continue

        force_zero = random.random() < P_ZERO_CARDINALITY
        count_query, raw_query, entities, triples = build_arbitrary_query(
            concrete_triples, seed, force_zero, global_predicates
        )

        if raw_query in seen_queries:
            continue

        try:
            rn = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': count_query},
                              timeout=FINAL_QUERY_TIMEOUT)
            if rn.status_code == 200:
                bindings = rn.json()["results"]["bindings"]
                if not bindings:
                    continue

                cardinality = int(bindings[0]["res"]["value"])

                seen_queries.add(raw_query)
                testdata.append({
                    "x": entities,
                    "y": cardinality,
                    "query": raw_query,
                    "triples": [t.strip().split() for t in triples]
                })
                pbar.update(1)

        except requests.exceptions.RequestException:
            continue

    pbar.close()

    testdata = filter_isomorphic_queries(testdata)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{dataset_name}_arbitrary_{timestamp}_{n_triples}.json"
    os.makedirs(dataset_dir, exist_ok=True)
    loc = os.path.join(dataset_dir, filename)

    with open(loc, "w") as fp:
        json.dump(testdata, fp, indent=2)

    analyze_query_dataset_statistics(loc)
    return testdata


if __name__ == "__main__":
    n_triples_to_sample = [2, 3, 5, 8]
    for n in n_triples_to_sample:
        get_queries("data/generated_queries/arbitrary_yago_empty", "arbitrary-yago-gnce-empty",
                    n_triples=n, n_queries=5000, endpoint_url="http://localhost:8888")