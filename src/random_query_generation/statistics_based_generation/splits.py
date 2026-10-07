"""Split a generated workload into train / val / test, plus a test set of held-out templates.

    python -m src.random_query_generation.statistics_based_generation.splits <generation output dir> \\
        [--out <dataset dir>] [--heldout-templates 0.1] [--val 0.1] [--test 0.1] [--seed 0]

A template is shape | size | choke point | zero-or-not | strata of the constants (the
"template" field generate.py writes). A random `heldout_templates` share of the templates goes
ENTIRELY to dataset_test_heldout (CEB-style generalisation: those query templates are never
seen in training); the remaining queries are split at random into dataset_train / dataset_val /
dataset_test. Each split is a directory with raw/<shape>.json, the layout the dataset loader
(QueryCardinalityDataset) reads.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def split(input_directory, output_directory, heldout_templates=0.1, val=0.1, test=0.1, seed=0):
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(input_directory, "statistics_based_*.json")))
    files = [f for f in files if not f.endswith("_summary.json")]
    if not files:
        raise FileNotFoundError(f"no statistics_based_<shape>.json in {input_directory}")
    by_shape = {os.path.basename(f)[len("statistics_based_"):-len(".json")]: json.load(open(f)) for f in files}
    templates = sorted({q["generation"]["template"] for queries in by_shape.values() for q in queries})
    held = set(rng.choice(templates, int(round(heldout_templates * len(templates))), replace=False).tolist())
    counts = Counter()
    for shape, queries in by_shape.items():
        parts = {"train": [], "val": [], "test": [], "test_heldout": []}
        for query in queries:
            if query["generation"]["template"] in held:
                parts["test_heldout"].append(query)
                continue
            r = rng.random()
            parts["val" if r < val else "test" if r < val + test else "train"].append(query)
        for name, items in parts.items():
            directory = os.path.join(output_directory, f"dataset_{name}", "raw")
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f"statistics_based_{shape}.json"), "w") as f:
                json.dump(items, f)
            counts[(name, shape)] = len(items)
    summary = {"heldout_templates": sorted(held), "n_templates": len(templates),
               "counts": {f"{name}/{shape}": n for (name, shape), n in sorted(counts.items())}}
    with open(os.path.join(output_directory, "splits_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{len(templates)} templates, {len(held)} held out -> {output_directory}")
    for name in ("train", "val", "test", "test_heldout"):
        print(f"  {name:13s} {sum(n for (s, _), n in counts.items() if s == name):>8,}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument("--out", default=None)
    parser.add_argument("--heldout-templates", type=float, default=0.1)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    split(args.input_directory, args.out or args.input_directory, args.heldout_templates, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
