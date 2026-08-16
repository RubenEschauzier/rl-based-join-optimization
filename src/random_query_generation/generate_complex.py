import json
import requests
import random
import math
import collections
from datetime import datetime
import os
from tqdm import tqdm

from src.random_query_generation.generation_validation import analyze_query_dataset_statistics
from src.utils.generation_utils.generation_utils import filter_isomorphic_queries

# Configuration
SEED_SUBJECTS = 50000
ENDPOINT_LIMIT = 500
FINAL_QUERY_TIMEOUT = 5

# Sampling Probabilities
P_BOUND_SUBJECT = 0.1
P_BOUND_PREDICATE = 1
P_BOUND_OBJECT = 0.1
P_ZERO_CARDINALITY_PREDICATE = 0.10
P_ZERO_CARDINALITY_LITERAL = 0.10

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


def get_global_literals(endpoint_url: str, limit: int = 100000) -> list:
    """Fetches a pool of real database literals for zero-cardinality mutations."""
    query = f"""
    SELECT DISTINCT ?o WHERE {{
        ?s ?p ?o .
        FILTER(isLiteral(?o))
    }} LIMIT {limit}
    """
    try:
        r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
        r.raise_for_status()
        return [f'"{d["o"]["value"]}"' for d in r.json().get("results", {}).get("bindings", [])]
    except Exception as e:
        print(f"Failed to fetch global literals: {e}")
        return []

def _fetch_subjects_batch(endpoint_url: str, offset: int, limit: int, headers: dict) -> list:
    """Executes a single paginated SPARQL query and returns the bindings."""
    query = f"""
        SELECT DISTINCT ?s 
        WHERE {{ 
            ?s ?p ?o . 
            FILTER(isIRI(?s)) 
        }} 
        ORDER BY ?s 
        LIMIT {limit} 
        OFFSET {offset}
    """
    r = requests.get(endpoint_url, headers=headers, params={'query': query})
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])


def get_seed_subjects(endpoint_url: str, total_subjects: int, endpoint_limit: int, headers: dict) -> list:
    subjects = set()

    if total_subjects == -1:
        offset = 0
        with tqdm(desc="Fetching all subjects exhaustively") as pbar:
            while True:
                try:
                    bindings = _fetch_subjects_batch(endpoint_url, offset, endpoint_limit, headers)
                    if not bindings:
                        break

                    subjects.update(d['s']['value'] for d in bindings)
                    offset += endpoint_limit
                    pbar.update(1)

                    if len(bindings) < endpoint_limit:
                        break
                except requests.RequestException as e:
                    print(f"Fetch failed at offset {offset}: {e}")
                    break
    else:
        num_requests = math.ceil(total_subjects / endpoint_limit)
        max_offset = max(0, total_subjects - endpoint_limit)
        offsets = [random.randint(0, max_offset) for _ in range(num_requests)]

        for offset in tqdm(offsets, desc="Fetching random subject batches"):
            try:
                bindings = _fetch_subjects_batch(endpoint_url, offset, endpoint_limit, headers)
                subjects.update(d['s']['value'] for d in bindings)
            except requests.RequestException:
                continue

    return list(subjects)


def fetch_local_neighborhood(endpoint_url: str, seed_node: str) -> list:
    """Fetches a 2-hop neighborhood around the seed node in a single SPARQL query."""
    query = f"""
    SELECT DISTINCT ?s ?p ?o WHERE {{
        {{ <{seed_node}> ?p ?o . BIND(<{seed_node}> AS ?s) }}
        UNION
        {{ ?s ?p <{seed_node}> . BIND(<{seed_node}> AS ?o) }}
        UNION
        {{ <{seed_node}> ?p1 ?mid . ?mid ?p ?o . BIND(?mid AS ?s) }}
        UNION
        {{ ?mid ?p1 <{seed_node}> . ?s ?p ?mid . BIND(?mid AS ?o) }}
    }} LIMIT 5000
    """
    try:
        r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query}, timeout=FINAL_QUERY_TIMEOUT)
        bindings = r.json().get("results", {}).get("bindings", [])
        return [(b['s']['value'], b['p']['value'], b['o']['value']) for b in bindings]
    except Exception:
        return []


def memory_random_walk(edges: list, seed_node: str, n_triples: int, global_pred_counts: collections.Counter) -> list:
    adj = collections.defaultdict(list)
    for s, p, o in edges:
        adj[s].append((s, p, o))
        adj[o].append((s, p, o))

    triples = set()
    active_nodes = {seed_node}
    explored_nodes = set()

    used_sp = set()
    used_po = set()

    used_p_subjects = collections.defaultdict(set)
    used_p_objects = collections.defaultdict(set)

    # Track node degrees locally to penalize clustering on a single hub
    node_edge_counts = collections.defaultdict(int)

    while len(triples) < n_triples and active_nodes:

        # DIVERSIFICATION - Apply inverse frequency weighting to active nodes
        # Nodes that already have many edges get lower selection probabilities
        active_list = list(active_nodes)
        node_weights = [1.0 / (node_edge_counts[n] + 1) for n in active_list]
        current_node = random.choices(active_list, weights=node_weights, k=1)[0]

        candidates = []
        for t in adj[current_node]:
            s, p, o = t

            if t in triples or (s, p) in used_sp or (p, o) in used_po:
                continue

            if p in used_p_subjects:
                if s not in used_p_subjects[p] and o not in used_p_objects[p]:
                    continue

            candidates.append(t)

        if not candidates:
            active_nodes.remove(current_node)
            explored_nodes.add(current_node)
            continue

        pred_groups = collections.defaultdict(list)
        for t in candidates:
            pred_groups[t[1]].append(t)

        candidate_preds = list(pred_groups.keys())

        weights = [1.0 / (global_pred_counts[p] + 1) for p in candidate_preds]
        chosen_pred = random.choices(candidate_preds, weights=weights, k=1)[0]
        chosen_triple = random.choice(pred_groups[chosen_pred])

        triples.add(chosen_triple)
        s, p, o = chosen_triple

        used_sp.add((s, p))
        used_po.add((p, o))

        used_p_subjects[p].add(s)
        used_p_objects[p].add(o)

        # Update degree tracking for diversification penalty
        node_edge_counts[s] += 1
        node_edge_counts[o] += 1

        if s not in explored_nodes: active_nodes.add(s)
        if o not in explored_nodes: active_nodes.add(o)

    return list(triples) if len(triples) == n_triples else []


def build_arbitrary_query(concrete_triples: list, seed_node: str, force_zero_pred: bool, force_zero_literal: bool,
                          global_predicates: list, global_literals: list) -> tuple:
    # Calculate structural degrees and map leaves to their parent hubs
    node_degrees = collections.defaultdict(int)
    adj_list = collections.defaultdict(list)

    for s, p, o in concrete_triples:
        node_degrees[s] += 1
        node_degrees[o] += 1
        adj_list[s].append(o)
        adj_list[o].append(s)

    nodes = {}
    preds = {}
    node_counter = 0
    pred_counter = 0

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

    # Standard probabilistic binding
    bound_nodes = {n: (random.random() < P_BOUND_SUBJECT) for n in nodes}
    bound_preds = {p: (random.random() < P_BOUND_PREDICATE) for p in preds}

    # HUB-BASED LEAF BOUNDING
    HUB_SIZE = 3
    MAX_UNBOUND_LEAVES_PER_HUB = 5
    hub_leaves = collections.defaultdict(list)

    # 1. Map all leaves to their direct parent hub
    for n in nodes:
        if node_degrees[n] == 1:
            parent = adj_list[n][0]
            hub_leaves[parent].append(n)

    # 2. Enforce constraints on hubs
    for hub, leaves in hub_leaves.items():
        if node_degrees[hub] >= HUB_SIZE:

            # Constraint 1: Must have AT LEAST 1 bound leaf node
            if leaves and not any(bound_nodes[l] for l in leaves):
                bound_nodes[random.choice(leaves)] = True

            # Constraint 2: Cap the remaining unbound leaves to prevent cartesian explosion
            unbound_leaves = [l for l in leaves if not bound_nodes[l]]
            if len(unbound_leaves) > MAX_UNBOUND_LEAVES_PER_HUB:
                # Shuffle so we randomly pick which excess leaves to bind
                random.shuffle(unbound_leaves)
                for l in unbound_leaves[MAX_UNBOUND_LEAVES_PER_HUB:]:
                    bound_nodes[l] = True

    if force_zero_pred and global_predicates:
        mutate_p = random.choice(list(preds.keys()))
        invalid_p = random.choice([x for x in global_predicates if x not in preds])
        concrete_triples = [(s, invalid_p if p == mutate_p else p, o) for s, p, o in concrete_triples]
        bound_preds[invalid_p] = True
        preds[invalid_p] = preds[mutate_p]

    if force_zero_literal and global_literals:
        bound_objects = [n for n in nodes if bound_nodes.get(n) and n != seed_node]
        if bound_objects:
            target = random.choice(bound_objects)

            # Sample a real database literal
            invalid_literal = random.choice(global_literals)

            # Ensure it does not accidentally exist in the current pattern
            while invalid_literal in nodes.values():
                invalid_literal = random.choice(global_literals)

            nodes[target] = invalid_literal

    where_clauses = []
    unique_patterns = set()
    entities = []

    for s, p, o in concrete_triples:
        bind_s = bound_nodes.get(s, False)
        bind_p = bound_preds.get(p, False)
        bind_o = bound_nodes.get(o, False)

        if not bind_s and not bind_p and not bind_o:
            bind_p = True
            bound_preds[p] = True

        s_str = f"<{s}>" if bind_s else nodes[s]
        p_str = f"<{p}>" if bind_p else preds[p]
        o_str = nodes[o] if nodes[o].startswith('"') else (f"<{o}>" if bind_o else nodes[o])

        pattern = f"{s_str} {p_str} {o_str} ."

        if pattern in unique_patterns:
            continue

        unique_patterns.add(pattern)
        where_clauses.append(pattern)

        if bind_s and s not in entities: entities.append(s)
        if bind_p and p not in entities: entities.append(p)
        if bind_o and o not in entities: entities.append(o)

    count_query = f"SELECT (COUNT(*) AS ?res) WHERE {{ {' '.join(where_clauses)} }}"
    raw_query = f"SELECT * WHERE {{ {' '.join(where_clauses)} }}"

    return count_query, raw_query, entities, where_clauses


def get_queries(dataset_dir: str, dataset_name: str, n_triples: int = 1, n_queries: int = 1000,
                endpoint_url: str = None):
    print("Initializing global graph statistics...")
    total_subjects = get_total_subject_count(endpoint_url)
    global_predicates = get_global_predicates(endpoint_url)
    global_literals = get_global_literals(endpoint_url)
    subjects = get_seed_subjects(endpoint_url, total_subjects, endpoint_limit=1000000, headers={})

    if not subjects:
        return []

    testdata = []
    seen_queries = set()
    global_pred_counts = collections.Counter()

    pbar = tqdm(total=n_queries, desc=f"Generating {n_triples}-triple Queries")
    while len(testdata) < n_queries:
        seed = random.choice(subjects)

        local_edges = fetch_local_neighborhood(endpoint_url, seed)
        if not local_edges:
            continue

        concrete_triples = memory_random_walk(local_edges, seed, n_triples, global_pred_counts)
        if not concrete_triples:
            continue

        force_zero_pred = random.random() < P_ZERO_CARDINALITY_PREDICATE
        force_zero_lit = random.random() < P_ZERO_CARDINALITY_LITERAL

        count_query, raw_query, entities, triples = build_arbitrary_query(
            concrete_triples, seed, force_zero_pred, force_zero_lit, global_predicates,
            global_literals=global_literals
        )

        # Reject query if duplicate triples reduced the desired size
        if len(triples) < n_triples or raw_query in seen_queries:
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

                # Update global counts with accepted query predicates
                for s, p, o in concrete_triples:
                    global_pred_counts[p] += 1

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
    n_triples_to_sample = [4, 6, 8, 10]
    for n in n_triples_to_sample:
        get_queries("data/generated_queries/complex_yago", "complex_yago",
                    n_triples=n, n_queries=2500, endpoint_url="http://localhost:9000")