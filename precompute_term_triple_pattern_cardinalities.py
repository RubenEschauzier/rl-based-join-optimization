import argparse
import os
import json
import glob

from src.datastructures.prepare_feature_data import read_queries, get_occurrences, \
    get_query_triple_pattern_cardinalities
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment


def main():
    parser = argparse.ArgumentParser(description="Process SPARQL queries for a dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., yago, dbpedia)."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="List of query directory paths or glob patterns."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for occurrences and cardinalities."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:9999/blazegraph/namespace/{}/sparql",
        help="SPARQL endpoint template. Must contain {} for dataset name."
    )
    args = parser.parse_args()

    # Setup
    output_location = args.output
    os.makedirs(output_location, exist_ok=True)

    query_env = BlazeGraphQueryEnvironment(args.endpoint.format(args.dataset))

    # Expand glob patterns and load queries
    loaded_queries = []
    for qpattern in args.queries:
        for qpath in glob.glob(qpattern):
            loaded_queries.extend(read_queries(qpath))
    print(f"Loaded {len(loaded_queries)} queries")
    # Extract occurrences and cardinalities
    loaded_occurrences = get_occurrences(loaded_queries, query_env)
    loaded_tp_cardinalities = get_query_triple_pattern_cardinalities(loaded_queries, query_env)

    # Save results
    with open(os.path.join(output_location, 'occurrences.json'), 'w') as f0:
        json.dump(loaded_occurrences, f0)

    with open(os.path.join(output_location, 'tp_cardinalities.json'), 'w') as f1:
        json.dump(loaded_tp_cardinalities, f1)


if __name__ == "__main__":
    main()