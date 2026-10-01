"""Per-epoch artifacts for an epinet run: a metrics log, curve data, figures, TensorBoard.

Layout inside a run directory:
    metrics.jsonl             one JSON object per evaluated epoch, every scalar metric
    epoch-N/curves.json       curve data per split (joint NLL vs tau, selective risk, ...)
    epoch-N/report_<split>.png  four-panel figure of those curves
    tensorboard/              scalars + figures; `tensorboard --logdir <sweep dir>` overlays seeds
    final_summary.json        written once, at the end: best validation epoch and its metrics
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated for CVD separation). Two of them sit below
# 3:1 contrast on white, so every series also gets a marker, a line style and a legend.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MODEL_STYLES = {
    "epinet_fitted": {"color": BLUE, "marker": "o", "linestyle": "-", "label": "Epinet, fitted noise"},
    "independent_fitted": {"color": ORANGE, "marker": "s", "linestyle": "--",
                           "label": "Epinet, fitted noise, samples shuffled across plans"},
    "base_fixed": {"color": AQUA, "marker": "^", "linestyle": "-.", "label": "Base, fixed noise"},
    "base_fitted": {"color": YELLOW, "marker": "D", "linestyle": ":", "label": "Base, fitted noise"},
    "epinet": {"color": MAGENTA, "marker": "v", "linestyle": (0, (6, 2)), "label": "Epinet, fixed noise"},
    # Only drawn for curves written before the fitted-noise evaluation existed.
    "independent": {"color": ORANGE, "marker": "s", "linestyle": "--",
                    "label": "Epinet, fixed noise, samples shuffled across plans"},
}


def joint_models_to_plot(curves):
    """Fitted-noise models first; the fixed-noise shuffle control only as a fallback."""
    available = curves["jnll_excess_per_target"]
    models = [m for m in ("epinet_fitted", "independent_fitted", "base_fixed", "base_fitted", "epinet")
              if m in available]
    if "independent_fitted" not in available and "independent" in available:
        models.insert(1, "independent")
    return models


def calibration_models_to_plot(curves):
    return [m for m in ("epinet_fitted", "epinet", "base_fitted") if m in curves["calibration"]]
REFERENCE_COLOR = "#52514e"
INK = "#0b0b0b"
GRID = "#e4e3df"


def _clean(value):
    """NaN/inf are not valid JSON; write them as null so any JSON reader can load the file."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2, sort_keys=True)


def append_metrics_line(run_directory, epoch, metrics):
    with open(os.path.join(run_directory, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(_clean({"epoch": epoch, **metrics}), sort_keys=True) + "\n")


def _style_axis(axis, title, xlabel, ylabel):
    axis.set_title(title, loc="left", fontsize=11, color=INK)
    axis.set_xlabel(xlabel, color=INK)
    axis.set_ylabel(ylabel, color=INK)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(REFERENCE_COLOR)
    axis.tick_params(colors=REFERENCE_COLOR)


def plot_split_report(curves, title, save_path=None):
    """Four panels: joint NLL vs tau, selective risk, calibration, plan-selection regret."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    figure.patch.set_facecolor("white")

    axis = axes[0, 0]
    taus = curves["taus"]
    for model in joint_models_to_plot(curves):
        values = [math.nan if v is None else v for v in curves["jnll_excess_per_target"][model]]
        axis.plot(taus, values, linewidth=2, markersize=7, **MODEL_STYLES[model])
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_xticks(taus)
    axis.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(taus, curves["jnll_n_queries"])])
    _style_axis(axis, "Joint NLL vs group size (lower is better)", "τ (plans per group)",
                "Excess NLL per target (0 = optimal)")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[0, 1]
    selective = curves["selective"]
    axis.plot(selective["coverage"], selective["epinet"], linewidth=2,
              color=BLUE, label="Epinet (drop most uncertain)")
    axis.plot(selective["coverage"], selective["oracle"], linewidth=1.5, linestyle="--",
              color=REFERENCE_COLOR, label="Oracle (drop largest errors)")
    axis.axhline(selective["random"], linewidth=1.5, linestyle=":", color=REFERENCE_COLOR,
                 label="Random (no ranking)")
    _style_axis(axis, "Selective prediction (lower is better)", "Fraction of plans kept",
                "MSE of kept plans (standardized)")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 0]
    calibration = curves["calibration"]
    axis.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linewidth=1, linestyle="--", label="Perfect calibration")
    for model in calibration_models_to_plot(curves):
        style = MODEL_STYLES[model]
        axis.plot(calibration["expected"], calibration[model], linewidth=2,
                  color=style["color"], linestyle=style["linestyle"], label=style["label"])
    coverage = curves["coverage"]
    fitted = coverage.get("epinet_fitted")
    coverage_text = "\n".join(
        f"{int(round(level * 100))}% interval: "
        + (f"epinet fitted {fitted[i]:.1%} · " if fitted else "")
        + f"epinet fixed {coverage['epinet'][i]:.1%} · base {coverage['base_fitted'][i]:.1%}"
        for i, level in enumerate(coverage["levels"])
    )
    noise = curves.get("noise")
    if noise:
        coverage_text += (f"\nnoise std: epinet fitted {noise['epinet_fitted']:.3g} · "
                          f"fixed {noise['epinet_fixed']:.3g} · base fitted {noise['base_fitted']:.3g}")
    axis.text(0.98, 0.04, coverage_text, transform=axis.transAxes, ha="right", va="bottom",
              fontsize=8.5, color=INK)
    _style_axis(axis, "Calibration", "Predicted quantile", "Observed fraction below")
    axis.legend(frameon=False, fontsize=9, loc="upper left")

    axis = axes[1, 1]
    regret = curves["regret"]
    labels = {"base": "Base mean", "epinet_mean": "Epinet mean", "epinet_thompson": "Epinet Thompson"}
    colors = {"base": AQUA, "epinet_mean": BLUE, "epinet_thompson": ORANGE}
    names = list(labels)
    bars = axis.bar([labels[n] for n in names], [regret[n] for n in names],
                    color=[colors[n] for n in names], width=0.6, edgecolor="white", linewidth=2)
    for bar, name in zip(bars, names):
        axis.annotate(f"{regret[name]:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 3),
                      textcoords="offset points")
    _style_axis(axis, "Plan-selection regret (lower is better)", "",
                "Mean regret, log-cost (exp = cost ratio)")

    figure.suptitle(title, fontsize=13, color=INK)
    if save_path:
        figure.savefig(save_path, dpi=110)
    return figure


class RunReporter:
    """Writes every artifact for one run. TensorBoard is optional (import is lazy)."""

    def __init__(self, run_directory, use_tensorboard=True, write_figures=True):
        self.run_directory = run_directory
        self.write_figures = write_figures
        self.tensorboard = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(os.path.join(run_directory, "tensorboard"))

    def log_epoch(self, epoch, epoch_dir, metrics, curves_per_split):
        append_metrics_line(self.run_directory, epoch, metrics)
        write_json(os.path.join(epoch_dir, "curves.json"), curves_per_split)

        figures = {}
        if self.write_figures:
            for split, curves in curves_per_split.items():
                figures[split] = plot_split_report(
                    curves, f"{split} · epoch {epoch}", os.path.join(epoch_dir, f"report_{split}.png")
                )

        if self.tensorboard is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    # "val_jnll_tau8_..." -> "val/jnll_tau8_...": one section per split,
                    # and every seed overlaid on the same chart.
                    split, _, rest = name.partition("_")
                    tag = f"{split}/{rest}" if split in curves_per_split or split == "train" else name
                    self.tensorboard.add_scalar(tag, value, epoch)
            for split, figure in figures.items():
                self.tensorboard.add_figure(f"report/{split}", figure, epoch, close=False)
            self.tensorboard.flush()

        for figure in figures.values():
            plt.close(figure)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
