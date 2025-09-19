import os
import argparse

from src.utils.training_utils.query_loading_utils import split_raw_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw queries into train/val sets.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., star_yago_gnce)."
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.1,
        help="Proportion of queries to use for validation (default: 0.1)."
    )
    args = parser.parse_args()

    project_root = os.getcwd()
    raw_queries_loc = os.path.join(project_root, f"data/generated_queries/{args.dataset_name}")
    train_queries_loc = os.path.join(raw_queries_loc, "dataset_train", "raw")
    val_queries_loc = os.path.join(raw_queries_loc, "dataset_val", "raw")

    split_raw_queries(raw_queries_loc, args.split, train_queries_loc, val_queries_loc)