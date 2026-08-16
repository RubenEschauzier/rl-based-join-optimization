import argparse
import os
import json
import glob

from src.datastructures.prepare_feature_data import (
    read_queries,
    calculate_occurrences,
    calculate_query_triple_pattern_cardinalities,
    calculate_multiplicities_queries, calculate_hll_sketches
)
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.qlever.qlever_execute_query_default import QLeverOptimizerClient


def main():
    parser = argparse.ArgumentParser(description="Process SPARQL queries and calculate specified metrics.")
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
        help="Output directory for calculated metrics."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:9999/blazegraph/namespace/{}/sparql",
        help="SPARQL endpoint template. Must contain {} for dataset name."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "occurrences", "cardinalities", "multiplicities", "hll"],
        default="all",
        help="Specify which metric to calculate. Defaults to 'all'."
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    query_env = QLeverOptimizerClient(args.endpoint.format(args.dataset))

    # Expand glob patterns and load queries
    loaded_queries = []
    for qpattern in args.queries:
        for qpath in glob.glob(qpattern):
            loaded_queries.extend(read_queries(qpath))
    print(f"Loaded {len(loaded_queries)} queries.")

    # Execute requested calculations
    if args.mode in ["all", "occurrences"]:
        occurrences = calculate_occurrences(loaded_queries, query_env)
        with open(os.path.join(args.output, 'occurrences.json'), 'w') as f:
            json.dump(occurrences, f)
        print("Saved occurrences.")

    if args.mode in ["all", "cardinalities"]:
        tp_cardinalities = calculate_query_triple_pattern_cardinalities(loaded_queries, query_env)
        with open(os.path.join(args.output, 'tp_cardinalities.json'), 'w') as f:
            json.dump(tp_cardinalities, f)
        print("Saved cardinalities.")

    if args.mode in ["all", "multiplicities"]:
        predicate_multiplicities = calculate_multiplicities_queries(loaded_queries, args.endpoint)
        with open(os.path.join(args.output, 'multiplicities.json'), 'w') as f:
            json.dump(predicate_multiplicities, f)
        print("Saved multiplicities.")
    if args.mode in ["all", "hll"]:
        hll_sketches = calculate_hll_sketches(loaded_queries, args.endpoint.format(args.dataset))
        with open(os.path.join(args.output, 'hll_sketches.json'), 'w') as f:
            json.dump(hll_sketches, f)
        print("Saved HyperLogLog sketches.")

if __name__ == "__main__":
    main()