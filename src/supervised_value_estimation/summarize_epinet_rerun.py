"""Aggregate a multi-seed epinet rerun into tables and figures.

    python -m src.supervised_value_estimation.summarize_epinet_rerun <rerun output dir>

Writes <dir>/summary/:
    per_epoch.csv         every metric, every seed, every evaluated epoch (long format)
    final_per_seed.csv    each seed at its best *validation* epoch
    final_table.md/.csv   mean, std, 95% CI and seed range per metric, grouped by section
    test_report.png       joint NLL vs tau, selective risk, calibration, regret (mean ± 95% CI)
    training_curves.png   key metrics per epoch across seeds (mean ± 95% CI, val and test)

Paper numbers are the TEST columns of final_table: test metrics read at the epoch chosen
by validation. Gains ("*_gain") are within-seed differences, so their CI is a paired one.
"""
import argparse
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.utils.epinet_utils.epinet_report import INK, MODEL_STYLES, REFERENCE_COLOR, _style_axis  # noqa: E402

SECTIONS = [
    ("Point accuracy (does the uncertainty cost accuracy?)", r"_(mse_scaled|qerror)_"),
    ("Joint NLL (excess per target, 0 = optimal)", r"_jnll_tau\d+_\w+_excess_per_target$"),
    ("Joint NLL gains (positive = epinet better)", r"_jnll_tau\d+_(dependence_gain|gain_vs_base_fitted|gain_vs_base_fixed)$"),
    ("Joint NLL coverage (queries with >= tau plans)", r"_jnll_tau\d+_n_queries$"),
    ("Selective prediction", r"_(aurc|selective_skill|mse_at_coverage)"),
    ("Calibration", r"_(calibration_error|coverage|sharpness|epistemic_std_mean|base_noise_std)"),
    ("Plan selection (log-cost regret; exp = cost ratio)", r"_(regret|optimal_rate)_"),
    ("Cost", r"(_ms_per_query_|_eval_seconds$|^train_epoch_seconds$)"),
]
TRAINING_CURVE_METRICS = [
    ("jnll_tau8_epinet_excess_per_target", "Joint NLL τ=8, epinet (excess)"),
    ("jnll_tau8_dependence_gain", "Dependence gain τ=8 (vs shuffled)"),
    ("jnll_tau8_gain_vs_base_fitted", "Gain τ=8 vs base, fitted noise"),
    ("selective_skill", "Selective-prediction skill"),
    ("calibration_error_epinet", "Calibration error, epinet"),
    ("mse_scaled_epinet", "MSE of epinet mean (standardized)"),
]


def _find_runs(rerun_directory):
    """seed -> run directory. Prefers a finished run; falls back to the newest partial one."""
    candidates = {}
    for directory in glob.glob(os.path.join(rerun_directory, "seed-*")):
        match = re.match(r"seed-(\d+)-", os.path.basename(directory))
        if not match or not os.path.exists(os.path.join(directory, "metrics.jsonl")):
            continue
        finished = os.path.exists(os.path.join(directory, "final_summary.json"))
        # Directory timestamps are dd-mm-yyyy and do not sort; use modification time.
        candidates.setdefault(int(match.group(1)), []).append(
            (finished, os.path.getmtime(directory), directory)
        )
    return {seed: (max(options)[2], max(options)[0]) for seed, options in candidates.items()}


def _read_metrics(directory):
    with open(os.path.join(directory, "metrics.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _best_epoch(rows, objective_metric):
    valid = [row for row in rows if row.get(objective_metric) is not None]
    return min(valid, key=lambda row: row[objective_metric])["epoch"] if valid else rows[-1]["epoch"]


def _confidence_interval(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return math.nan, math.nan, math.nan, 0
    mean = values.mean()
    if len(values) < 2:
        return mean, math.nan, math.nan, len(values)
    std = values.std(ddof=1)
    half_width = stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))
    return mean, std, half_width, len(values)


def _section_of(metric):
    for title, pattern in SECTIONS:
        if re.search(pattern, metric):
            return title
    return "Other"


def _table(final):
    rows = []
    metric_columns = [c for c in final.columns if c not in ("seed", "best_epoch", "finished", "epoch")]
    for metric in metric_columns:
        mean, std, half_width, n = _confidence_interval(final[metric].tolist())
        values = final[metric].dropna()
        rows.append({
            "section": _section_of(metric), "metric": metric, "mean": mean, "std": std,
            "ci95_half_width": half_width, "min": values.min() if len(values) else math.nan,
            "max": values.max() if len(values) else math.nan, "n_seeds": n,
            "n_seeds_positive": int((values > 0).sum()) if "gain" in metric else None,
        })
    return pd.DataFrame(rows)


def _format(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.4g}"


def _write_markdown(table, path, n_seeds, n_finished):
    lines = [f"# Epinet rerun summary — {n_seeds} seeds ({n_finished} finished)", "",
             "Each seed is read at its best **validation** epoch. Gains are within-seed "
             "differences (paired). CI = 95% t-interval over seeds.", ""]
    for split in ("test", "val"):
        lines += [f"## {split}", ""]
        split_table = table[table.metric.str.startswith(f"{split}_")]
        for section, _ in SECTIONS:
            section_rows = split_table[split_table.section == section]
            if section_rows.empty:
                continue
            lines += [f"### {section}", "", "| metric | mean ± 95% CI | std | range | seeds > 0 |",
                      "|---|---|---|---|---|"]
            for _, row in section_rows.iterrows():
                positive = "" if row.n_seeds_positive is None or pd.isna(row.n_seeds_positive) \
                    else f"{int(row.n_seeds_positive)}/{row.n_seeds}"
                lines.append(
                    f"| `{row.metric[len(split) + 1:]}` | {_format(row['mean'])} ± {_format(row.ci95_half_width)} "
                    f"| {_format(row['std'])} | {_format(row['min'])} … {_format(row['max'])} | {positive} |"
                )
            lines.append("")
    training = table[~table.metric.str.startswith(("test_", "val_"))]
    if not training.empty:
        lines += ["## training", "", "| metric | mean ± 95% CI |", "|---|---|"]
        lines += [f"| `{row.metric}` | {_format(row['mean'])} ± {_format(row.ci95_half_width)} |"
                  for _, row in training.iterrows()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _band(axis, x, matrix, color, label, linestyle="-", marker=None):
    matrix = np.asarray(matrix, dtype=float)
    mean = np.nanmean(matrix, axis=0)
    n = np.sum(np.isfinite(matrix), axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
    half = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * std / np.sqrt(np.maximum(n, 1)), 0.0)
    axis.plot(x, mean, color=color, linewidth=2, linestyle=linestyle, marker=marker, markersize=6, label=label)
    axis.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)


def _plot_test_report(curves_by_seed, path, split="test"):
    curves = [c[split] for c in curves_by_seed if split in c]
    if not curves:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axis = axes[0, 0]
    taus = curves[0]["taus"]
    for model, style in MODEL_STYLES.items():
        matrix = [[math.nan if v is None else v for v in c["jnll_excess_per_target"][model]] for c in curves]
        _band(axis, taus, matrix, style["color"], style["label"], style["linestyle"], style["marker"])
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_xticks(taus)
    axis.set_xticklabels([str(t) for t in taus])
    _style_axis(axis, "Joint NLL vs group size (lower is better)", "τ (plans per group)",
                "Excess NLL per target (0 = optimal)")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[0, 1]
    coverage = curves[0]["selective"]["coverage"]
    _band(axis, coverage, [c["selective"]["epinet"] for c in curves], MODEL_STYLES["epinet"]["color"],
          "Epinet (drop most uncertain)")
    _band(axis, coverage, [c["selective"]["oracle"] for c in curves], REFERENCE_COLOR,
          "Oracle (drop largest errors)", "--")
    axis.axhline(np.mean([c["selective"]["random"] for c in curves]), color=REFERENCE_COLOR,
                 linestyle=":", linewidth=1.5, label="Random (no ranking)")
    _style_axis(axis, "Selective prediction (lower is better)", "Fraction of plans kept",
                "MSE of kept plans (standardized)")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 0]
    expected = curves[0]["calibration"]["expected"]
    axis.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linewidth=1, linestyle="--", label="Perfect calibration")
    for model in ("epinet", "base_fitted"):
        style = MODEL_STYLES[model]
        _band(axis, expected, [c["calibration"][model] for c in curves], style["color"], style["label"],
              style["linestyle"])
    _style_axis(axis, "Calibration", "Predicted quantile", "Observed fraction below")
    axis.legend(frameon=False, fontsize=9, loc="upper left")

    axis = axes[1, 1]
    labels = {"base": "Base mean", "epinet_mean": "Epinet mean", "epinet_thompson": "Epinet Thompson"}
    colors = [MODEL_STYLES["base_fixed"]["color"], MODEL_STYLES["epinet"]["color"],
              MODEL_STYLES["independent"]["color"]]
    means, halves = [], []
    for name in labels:
        mean, _, half, _ = _confidence_interval([c["regret"][name] for c in curves])
        means.append(mean)
        halves.append(0.0 if not np.isfinite(half) else half)
    bars = axis.bar(list(labels.values()), means, yerr=halves, color=colors, width=0.6,
                    edgecolor="white", linewidth=2, capsize=4, ecolor=INK)
    for bar, mean, half in zip(bars, means, halves):
        axis.annotate(f"{mean:.4f}", (bar.get_x() + bar.get_width() / 2, mean + half), ha="center",
                      va="bottom", fontsize=9, color=INK, xytext=(0, 3), textcoords="offset points")
    _style_axis(axis, "Plan-selection regret (lower is better)", "", "Mean regret, log-cost (exp = cost ratio)")

    figure.suptitle(f"{split} at best validation epoch · mean ± 95% CI over {len(curves)} seeds",
                    fontsize=13, color=INK)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_training_curves(per_epoch, path):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    split_styles = {"val": ("#2a78d6", "-"), "test": ("#eb6834", "--")}
    for axis, (metric, title) in zip(axes.flat, TRAINING_CURVE_METRICS):
        for split, (color, linestyle) in split_styles.items():
            column = f"{split}_{metric}"
            if column not in per_epoch:
                continue
            pivot = per_epoch.pivot_table(index="seed", columns="epoch", values=column)
            _band(axis, pivot.columns.to_numpy(), pivot.to_numpy(), color, split, linestyle)
        _style_axis(axis, title, "Epoch", "")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Per-epoch metrics · mean ± 95% CI over seeds", fontsize=13, color=INK)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def summarize_rerun(rerun_directory, objective_metric="val_jnll_tau8_epinet_excess_per_target"):
    runs = _find_runs(str(rerun_directory))
    if not runs:
        print(f"No runs with metrics.jsonl under {rerun_directory}")
        return None
    summary_directory = os.path.join(str(rerun_directory), "summary")
    os.makedirs(summary_directory, exist_ok=True)

    per_epoch_rows, final_rows, best_curves = [], [], []
    for seed, (directory, finished) in sorted(runs.items()):
        rows = _read_metrics(directory)
        per_epoch_rows += [{"seed": seed, "finished": finished, **row} for row in rows]
        best_epoch = _best_epoch(rows, objective_metric)
        best_row = next(row for row in rows if row["epoch"] == best_epoch)
        final_rows.append({"seed": seed, "best_epoch": best_epoch, "finished": finished,
                           **{k: v for k, v in best_row.items() if k != "epoch"}})
        curves_path = os.path.join(directory, f"epoch-{best_epoch}", "curves.json")
        if os.path.exists(curves_path):
            with open(curves_path, encoding="utf-8") as f:
                best_curves.append(json.load(f))

    per_epoch = pd.DataFrame(per_epoch_rows)
    final = pd.DataFrame(final_rows)
    per_epoch.to_csv(os.path.join(summary_directory, "per_epoch.csv"), index=False)
    final.to_csv(os.path.join(summary_directory, "final_per_seed.csv"), index=False)

    table = _table(final.apply(pd.to_numeric, errors="coerce").assign(seed=final.seed))
    table.to_csv(os.path.join(summary_directory, "final_table.csv"), index=False)
    n_finished = int(final.finished.sum())
    _write_markdown(table, os.path.join(summary_directory, "final_table.md"), len(final), n_finished)

    _plot_test_report(best_curves, os.path.join(summary_directory, "test_report.png"), "test")
    _plot_test_report(best_curves, os.path.join(summary_directory, "val_report.png"), "val")
    _plot_training_curves(per_epoch, os.path.join(summary_directory, "training_curves.png"))

    print(f"Summary of {len(final)} seeds ({n_finished} finished) written to {summary_directory}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rerun_directory")
    parser.add_argument("--objective", default="val_jnll_tau8_epinet_excess_per_target")
    arguments = parser.parse_args()
    summarize_rerun(arguments.rerun_directory, arguments.objective)
