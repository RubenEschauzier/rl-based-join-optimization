import json
import argparse
import glob
import os

from src.random_query_generation.generation_validation import analyze_query_dataset_statistics


def process_queries(directory, pattern, max_y):
    # Construct search path
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"No files found matching: {search_path}")
        return

    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {file_path}")
                continue

        # Handle both single objects and lists of objects
        if isinstance(data, dict):
            data = [data]
        print("Before filter")
        analyze_query_dataset_statistics(file_path)

        # Apply filter on 'y'
        filtered_data = [q for q in data if q.get('y', 0) <= max_y]

        # Save results
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_filtered{ext}"

        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        print(f"Processed {file_path} -> {output_path} ({len(filtered_data)} queries remain)")
        print("After filter")
        analyze_query_dataset_statistics(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter query JSON files by max y value.")
    parser.add_argument("directory", help="Target directory")
    parser.add_argument("pattern", help="Glob pattern (e.g., '*.json')")
    parser.add_argument("--max-y", type=int, required=True, help="Maximum allowed value for y")

    args = parser.parse_args()
    process_queries(args.directory, args.pattern, args.max_y)