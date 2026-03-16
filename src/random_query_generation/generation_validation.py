import csv
import statistics
import json
from io import StringIO

import requests
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

QLEVER_HEADERS = {'Accept': 'application/sparql-results+json'}


def analyze_query_dataset_statistics(filepath: str) -> None:
    """Reads a generated query dataset and prints summary statistics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' contains invalid JSON.")
        return

    if not data:
        print("Dataset is empty.")
        return

    total_queries = len(data)
    cardinalities = [q['y'] for q in data]
    bound_entities = [len(q['x']) for q in data]

    zero_card_count = sum(1 for c in cardinalities if c == 0)
    zero_card_pct = (zero_card_count / total_queries) * 100

    non_zero_cards = [c for c in cardinalities if c > 0]

    print(f"\n=== Dataset Statistics: {filepath} ===")
    print(f"Total Queries: {total_queries}")
    print(f"Zero-Cardinality Queries: {zero_card_count} ({zero_card_pct:.2f}%)")

    if non_zero_cards:
        print("\n--- Non-Zero Cardinality Distribution ---")
        print(f"Min: {min(non_zero_cards)}")
        print(f"Max: {max(non_zero_cards)}")
        print(f"Mean: {statistics.mean(non_zero_cards):.2f}")
        print(f"Median: {statistics.median(non_zero_cards)}")

    print("\n--- Structural Statistics ---")
    print(f"Mean Bound Entities per Query: {statistics.mean(bound_entities):.2f}")
    print(f"Min Bound Entities: {min(bound_entities)}")
    print(f"Max Bound Entities: {max(bound_entities)}")
    print("=======================================\n")


def analyze_all_term_coverage(filepath: str, endpoint_url: str, top_n: int = 15) -> None:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error loading {filepath}")
        return

    all_terms = [term for q in data for term in q['x']]
    query_term_counts = Counter(all_terms)

    if not query_term_counts:
        print("No bound terms found in the dataset.")
        return

    db_uris, db_literals = fetch_global_vocabulary(endpoint_url)

    # Cross-reference query terms with the global database vocabulary
    query_uris = {t: c for t, c in query_term_counts.items() if t in db_uris or str(t).startswith('http')}
    query_literals = {t: c for t, c in query_term_counts.items() if t not in query_uris}

    def compute_stats(query_counts, db_counts):
        stats = []
        for term, q_freq in query_counts.items():
            db_freq = db_counts.get(term, 0)
            coverage = (q_freq / db_freq * 100) if db_freq > 0 else 0.0
            stats.append({'term': term, 'q_freq': q_freq, 'db_freq': db_freq, 'coverage': coverage})
        return stats

    uri_stats = compute_stats(query_uris, db_uris)
    literal_stats = compute_stats(query_literals, db_literals)

    # Output Global Coverage Metrics
    print(f"\n=== Global Vocabulary Coverage ===")
    uri_pct = (len(query_uris) / len(db_uris) * 100) if db_uris else 0
    lit_pct = (len(query_literals) / len(db_literals) * 100) if db_literals else 0
    print(f"URIs: {len(query_uris)} sampled / {len(db_uris)} total ({uri_pct:.4f}%)")
    print(f"Literals: {len(query_literals)} sampled / {len(db_literals)} total ({lit_pct:.4f}%)")

    # Display URI Extremes
    print("\n" + "=" * 85 + "\nURI ANALYSIS\n" + "=" * 85)
    by_coverage_uri = sorted(uri_stats, key=lambda x: x['coverage'])
    by_q_freq_uri = sorted(uri_stats, key=lambda x: x['q_freq'])

    print_coverage_table("Highest Coverage URIs", by_coverage_uri[::-1], top_n)
    print_coverage_table("Lowest Coverage URIs", by_coverage_uri, top_n)
    print_coverage_table("Most Frequent Query URIs", by_q_freq_uri[::-1], top_n)
    print_coverage_table("Least Frequent Query URIs", by_q_freq_uri, top_n)

    # Display Literal Extremes
    if literal_stats:
        print("\n" + "=" * 85 + "\nLITERAL ANALYSIS\n" + "=" * 85)
        by_coverage_lit = sorted(literal_stats, key=lambda x: x['coverage'])
        by_q_freq_lit = sorted(literal_stats, key=lambda x: x['q_freq'])

        print_coverage_table("Highest Coverage Literals", by_coverage_lit[::-1], top_n)
        print_coverage_table("Lowest Coverage Literals", by_coverage_lit, top_n)
        print_coverage_table("Most Frequent Query Literals", by_q_freq_lit[::-1], top_n)
        print_coverage_table("Least Frequent Query Literals", by_q_freq_lit, top_n)

    # Plot Independent ECDFs
    plt.figure(figsize=(10, 6))

    if uri_stats:
        uri_cov = np.sort([s['coverage'] for s in uri_stats])
        plt.plot(uri_cov, np.arange(1, len(uri_cov) + 1) / len(uri_cov), marker='.', linestyle='none', color='blue',
                 label='URIs')

    if literal_stats:
        lit_cov = np.sort([s['coverage'] for s in literal_stats])
        plt.plot(lit_cov, np.arange(1, len(lit_cov) + 1) / len(lit_cov), marker='.', linestyle='none', color='orange',
                 label='Literals')

    plt.xscale('log')
    plt.xlabel('Coverage (%) - Log Scale')
    plt.ylabel('Cumulative Probability')
    plt.title('Empirical Cumulative Distribution of Term Coverage (URIs vs Literals)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plot_filename = filepath.replace(".json", "_coverage_ecdf.png")
    plt.savefig(plot_filename)
    print(f"\nSaved independent coverage distribution plot to {plot_filename}")
    plt.show()

def print_coverage_table(title: str, term_data: list, top_n: int) -> None:
    if not term_data:
        return
    print(f"\n=== {title} ===")
    print(f"{'Term':<45} | {'Q-Freq':<8} | {'DB-Freq':<8} | {'Coverage %'}")
    print("-" * 85)
    for data in term_data[:top_n]:
        term = str(data['term'])
        display_term = term if len(term) <= 43 else term[:20] + "..." + term[-20:]
        bar_length = min(int(data['coverage'] * 10), 20)
        bar = "█" * bar_length
        print(f"{display_term:<45} | {data['q_freq']:<8} | {data['db_freq']:<8} | {data['coverage']:>6.4f}% {bar}")


def fetch_global_vocabulary(endpoint_url: str) -> tuple:
    """Retrieves all terms and their absolute frequencies from the database."""
    db_uris = Counter()
    db_literals = Counter()

    queries = [
        "SELECT ?x (COUNT(*) AS ?count) WHERE { ?x ?p ?o } GROUP BY ?x",
        "SELECT ?x (COUNT(*) AS ?count) WHERE { ?s ?x ?o } GROUP BY ?x",
        "SELECT ?x (COUNT(*) AS ?count) WHERE { ?s ?p ?x } GROUP BY ?x"
    ]

    headers = {'Accept': 'application/sparql-results+json'}

    print("\nExtracting global database vocabulary. This requires processing the entire graph...")
    for q in queries:
        r = requests.get(endpoint_url, headers=headers, params={'query': q}, timeout=120)
        if r.status_code != 200:
            print(f"Failed to execute vocabulary query: {r.status_code}")
            continue

        try:
            data = r.json()
            bindings = data.get("results", {}).get("bindings", [])
        except ValueError:
            print("Failed to parse JSON response.")
            continue

        for row in bindings:
            if "x" not in row or "count" not in row:
                continue

            term_value = row["x"]["value"]
            term_type = row["x"]["type"]
            count = int(row["count"]["value"])

            if term_type == "uri":
                db_uris[term_value] += count
            else:
                db_literals[term_value] += count

    return db_uris, db_literals