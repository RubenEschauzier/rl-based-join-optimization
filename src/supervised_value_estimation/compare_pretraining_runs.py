"""Compare cardinality pretraining runs on the same validation set, broken down by the
generation metadata of statistics-based workloads (zero / non-zero, shape, choke point, size).

    python -m src.supervised_value_estimation.compare_pretraining_runs \\
        --runs baseline=<run dir> emptiness=<run dir> ... \\
        --val-raw data/generated_queries/mixed_yago_v2/dataset_val/raw --out <directory>

Each run is scored at its best validation epoch (lowest val_loss in train_progress.json; for a
model with an emptiness head val_loss is computed on its combined prediction, so runs compare).
Writes comparison.json and comparison.md to --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def _metadata(val_raw):
    meta = {}
    for path in glob.glob(os.path.join(val_raw, "*.json")):
        for query in json.load(open(path)):
            meta[" ".join(query["query"].split())] = query.get("generation", {})
    return meta


def _stats(q_errors):
    q = np.asarray(q_errors, dtype=float)
    return {"n": int(len(q)), "median": float(np.median(q)), "p90": float(np.quantile(q, 0.9)),
            "p95": float(np.quantile(q, 0.95)), "p99": float(np.quantile(q, 0.99)),
            "mean": float(np.mean(q)), "share_over_100": float(np.mean(q > 100))}


def _size_group(n):
    return "4-7" if n <= 7 else "8-11" if n <= 11 else "12-16"


def score_run(run_dir, meta):
    progress = json.load(open(os.path.join(run_dir, "train_progress.json")))
    val_loss = progress["val_loss"]
    epoch = int(np.argmin(val_loss))
    predictions = json.load(open(os.path.join(run_dir, f"epoch-{epoch}", "val_predictions.json")))
    groups = defaultdict(list)
    for p in predictions:
        q, actual = float(p["q_error"]), float(p["actual"])
        g = meta.get(" ".join(p["query"].split()), {})
        groups["all"].append(q)
        if actual == 0:
            groups["zero"].append(q)
            continue
        groups["non-zero"].append(q)
        groups[f"shape:{g.get('shape', p.get('type'))}"].append(q)
        groups[f"choke:{g.get('choke_point')}"].append(q)
        groups[f"size:{_size_group(g.get('n_patterns', 0))}"].append(q)
    result = {"run": run_dir, "best_epoch": epoch, "epochs": len(val_loss), "val_loss": float(val_loss[epoch]),
              "groups": {k: _stats(v) for k, v in sorted(groups.items())}}
    empty_keys = [k for k in progress if k.startswith("val_empty_")]
    if empty_keys:
        result["emptiness"] = {k[len("val_empty_"):]: float(progress[k][epoch]) for k in empty_keys}
    return result


def _markdown(results):
    names = list(results)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines.append("| best epoch / val loss | " + " | ".join(
        f"{r['best_epoch']} / {r['val_loss']:.3f}" for r in results.values()) + " |")
    groups = sorted({g for r in results.values() for g in r["groups"]},
                    key=lambda g: (g not in ("all", "non-zero", "zero"), g))
    for group in groups:
        for stat in (("median", "p90", "p99") if group in ("all", "non-zero", "zero") else ("median", "p99")):
            cells = []
            for r in results.values():
                s = r["groups"].get(group)
                cells.append("" if s is None else f"{s[stat]:,.2f}" if stat == "median" else f"{s[stat]:,.0f}")
            lines.append(f"| {group} {stat} | " + " | ".join(cells) + " |")
    for key in ("accuracy", "precision", "recall", "auroc"):
        if any("emptiness" in r for r in results.values()):
            lines.append(f"| emptiness {key} | " + " | ".join(
                f"{r['emptiness'][key]:.3f}" if "emptiness" in r else "" for r in results.values()) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=run_directory")
    parser.add_argument("--val-raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = _metadata(args.val_raw)
    results = {}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        results[name] = score_run(run_dir, meta)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    table = _markdown(results)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
