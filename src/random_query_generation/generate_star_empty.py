import json
import requests
import random
import math
import itertools
from datetime import datetime
import os
import hashlib
from tqdm import tqdm

from src.random_query_generation.generation_validation import analyze_query_dataset_statistics, \
    analyze_all_term_coverage
from src.utils.generation_utils.generation_utils import filter_isomorphic_queries

# Configuration
SEED_SUBJECTS = 50000
SUBJECTS_BATCH = 50
ENDPOINT_LIMIT = 500  # Smaller limit for offset sampling
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


def get_seed_stars(endpoint_url: str, subjects: list, n_triples: int) -> dict:
    stars = {}

    for i in tqdm(range(0, len(subjects), SUBJECTS_BATCH), desc="Extracting Stars"):
        batch = subjects[i:i + SUBJECTS_BATCH]
        values = " ".join([f"({s})" for s in batch])
        query = f"SELECT ?s ?p WHERE {{ ?s ?p ?o . VALUES (?s) {{ {values} }} }}"

        try:
            r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': query})
            bindings = r.json()["results"]["bindings"]
        except Exception:
            continue

        cand_stars = {}
        for elem in bindings:
            subj, pred = elem['s']['value'], elem['p']['value']
            cand_stars.setdefault(subj, []).append(pred)

        # Indented to process candidates immediately after each batch
        for subj, preds in cand_stars.items():
            unique_preds = list(set(preds))
            if len(unique_preds) >= n_triples:
                if math.comb(len(unique_preds), n_triples) > MAX_COMBINATIONS_PER_STAR:
                    combos = [tuple(random.sample(unique_preds, n_triples)) for _ in range(MAX_COMBINATIONS_PER_STAR)]
                    combos = list(set(combos))
                else:
                    combos = list(itertools.combinations(unique_preds, n_triples))

                for p_tuple in combos:
                    stars.setdefault(p_tuple, []).append(f"<{subj}>")

    return stars

def build_query_string(subject: str, predicates: tuple, objects: list, bind_subject: bool, force_zero: bool,
                       global_predicates: list) -> tuple:
    s_var = subject if bind_subject else "?s"
    where_clauses = []
    entities = []

    if bind_subject:
        entities.append(subject.strip("<>"))

    current_predicates = list(predicates)

    if force_zero and global_predicates:
        mutate_idx = random.randint(0, len(current_predicates) - 1)
        invalid_predicate = random.choice([p for p in global_predicates if p not in current_predicates])
        current_predicates[mutate_idx] = invalid_predicate

    paired = list(zip(current_predicates, objects))
    paired.sort(key=lambda x: x[0])
    sorted_predicates, sorted_objects = zip(*paired)

    for i, p in enumerate(sorted_predicates):
        # Calculate binding flags first
        bind_p = random.random() < P_BOUND_PREDICATE or force_zero
        bind_o = bool(sorted_objects[i]) and random.random() < P_BOUND_OBJECT

        # Enforce at least one bound element (predicate or object) per pattern
        if not bind_p and not bind_o:
            if sorted_objects[i] and random.random() < 0.5:
                bind_o = True
            else:
                bind_p = True

        p_var = f"<{p}>" if bind_p else f"?p{i}"
        o_var = f"<{sorted_objects[i]}>" if bind_o else f"?o{i}"

        where_clauses.append(f"{s_var} {p_var} {o_var} .")

        if bind_p: entities.append(p)
        if bind_o: entities.append(sorted_objects[i])

    query = f"SELECT (COUNT(*) AS ?res) WHERE {{ {' '.join(where_clauses)} }}"
    return query, entities, where_clauses

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

    stars = list(get_seed_stars(endpoint_url, subjects, n_triples).items())
    if not stars:
        print("No valid stars found.")
        return []

    testdata = []
    seen_queries = set()

    pbar = tqdm(total=n_queries, desc="Generating Queries")
    while len(testdata) < n_queries:
        try:
            predicates, candidate_subjects = random.choice(stars)
            target_subject = random.choice(candidate_subjects)
            bind_subject = random.random() < P_BOUND_SUBJECT
            force_zero = random.random() < P_ZERO_CARDINALITY

            # Construct a query to fetch one valid object for EVERY predicate in the star
            obj_where_clauses = [f"{target_subject} <{p}> ?o{i} ." for i, p in enumerate(predicates)]
            obj_query = f"SELECT * WHERE {{ {' '.join(obj_where_clauses)} }} LIMIT 1"

            r = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': obj_query},
                             timeout=FINAL_QUERY_TIMEOUT)

            objects = [None] * n_triples
            if r.status_code == 200 and r.json()["results"]["bindings"]:
                result_binding = r.json()["results"]["bindings"][0]
                for i in range(n_triples):
                    var_name = f"o{i}"
                    if var_name in result_binding:
                        objects[i] = result_binding[var_name]["value"]

            final_query, entities, triples = build_query_string(
                target_subject, predicates, objects, bind_subject, force_zero, global_predicates
            )

            # Use the QLever-compatible replacement string we established earlier
            raw_query_string = final_query.replace("(COUNT(*) AS ?res)", "*")

            if raw_query_string in seen_queries:
                continue

            rn = requests.get(endpoint_url, headers=QLEVER_HEADERS, params={'query': final_query},
                              timeout=FINAL_QUERY_TIMEOUT)
            if rn.status_code == 200:
                cardinality = int(rn.json()["results"]["bindings"][0]["res"]["value"])

                seen_queries.add(raw_query_string)
                testdata.append({
                    "x": entities,
                    "y": cardinality,
                    "query": raw_query_string,
                    "triples": [t.strip().split() for t in triples]
                })
                pbar.update(1)

        except requests.exceptions.RequestException:
            continue

    pbar.close()

    testdata = filter_isomorphic_queries(testdata)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{dataset_name}_stars_{timestamp}_{n_triples}.json"
    loc = os.path.join(dataset_dir, filename)
    with open(loc, "w") as fp:
        json.dump(testdata, fp, indent=2)

    analyze_query_dataset_statistics(loc)
    # analyze_all_term_coverage(loc, endpoint_url, 15)
    return testdata


if __name__ == "__main__":
    n_triples_to_sample = [2, 3, 5, 8]
    for n in n_triples_to_sample:
        get_queries("data/generated_queries/star_yago_empty", "star-yago-gnce-empty",
                    n_triples=n, n_queries=5000, endpoint_url="http://localhost:8888")
