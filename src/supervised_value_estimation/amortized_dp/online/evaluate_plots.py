"""Plots of evaluate_planners.py's report: colour = the estimator (GNCE, est, value, QLever), line / marker = the search
(greedy, beam, DP, DP + bound)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HUE = {"GNCE": "#2a78d6", "est": "#eb6834", "value": "#1baf7a", "QLever": "#52514e"}
STYLE = {"greedy": ("-", "o"), "beam": ("--", "s"), "DP": (":", "^"), "DP + bound": ("-.", "D"), "QLever": ("-", "x")}


def look(name):
    family = name.split(" ")[0]
    search = "QLever" if name == "QLever" else ("DP + bound" if name.endswith("DP + bound") else
                                                 "DP" if name.endswith("DP") else "beam" if "beam" in name else "greedy")
    return HUE[family], *STYLE[search]


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)


def cout_error(table, key, name):
    known = [max(r["cout"], 1) for r in table[key].values() if r["cout"] is not None]
    c = table[key][name]["cout"]
    return None if c is None or not known else max(c, 1) / min(known)


def plot_all(plans, table, names, out):
    keys = list(plans)
    log10 = lambda x: math.log10(max(x, 1e-3))

    # C_out against execution time, one panel per planner
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes.flat, names):
        colour, _, marker = look(name)
        pts = [(math.log10(max(table[k][name]["cout"], 1)), log10(max(table[k][name]["exec"], 0.5)), table[k][name]["censored"])
               for k in keys if table[k][name]["cout"] is not None and table[k][name]["exec"] is not None]
        if pts:
            x, y, cens = map(np.array, zip(*pts))
            ax.scatter(x[~cens], y[~cens], s=6, color=colour, alpha=0.35, lw=0)
            ax.scatter(x[cens], y[cens], s=14, color=colour, marker="^", lw=0, label="timed out")
            rho = spearmanr(x, y)[0] if len(x) > 2 else float("nan")
            ax.text(0.03, 0.95, f"Spearman {rho:.2f}", transform=ax.transAxes, va="top", fontsize=8, color=INK)
        style(ax, name)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("C_out (log10 rows)", color=INK2, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("execution (log10 ms)", color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "cout_vs_latency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # quantile curves
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    q = np.linspace(0, 100, 201)
    for name in names:
        colour, ls, _ = look(name)
        exec_ms = [max(table[k][name]["exec"], 0.5) for k in keys if table[k][name]["exec"] is not None]
        a.plot(q, np.log10(np.percentile(exec_ms, q)), color=colour, ls=ls, lw=1.6, label=name)
        err = [e for e in (cout_error(table, k, name) for k in keys) if e is not None]
        if err:
            b.plot(q, np.log10(np.percentile(err, q)), color=colour, ls=ls, lw=1.6, label=name)
    style(a, "Execution time per query, quantiles", "quantile (%)", "execution (log10 ms)")
    style(b, "C_out error per query, quantiles", "quantile (%)", "C_out / best of any planner (log10)")
    b.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "quantiles.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # violins per query shape
    shapes = sorted({plans[k]["shape"] for k in keys})
    metrics = {"cout_error": ("C_out / best of any planner (log10)", lambda k, n: cout_error(table, k, n)),
               "exec_vs_qlever": ("execution / QLever's (log10; < 0 is faster)",
                                  lambda k, n: None if table[k][n]["exec"] is None or table[k]["QLever"]["exec"] is None
                                  else max(table[k][n]["exec"], 1) / max(table[k]["QLever"]["exec"], 1))}
    for metric, (label, value) in metrics.items():
        shown = names if metric == "cout_error" else [n for n in names if n != "QLever"]
        cols = 4
        rows = math.ceil(len(shapes) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.3 * rows), sharey=True, facecolor=SURFACE)
        for ax, shape in zip(np.atleast_1d(axes).flat, shapes):
            ks = [k for k in keys if plans[k]["shape"] == shape]
            data = [[math.log10(v) for v in (value(k, n) for k in ks) if v is not None and v > 0] or [0.0] for n in shown]
            parts = ax.violinplot(data, showmedians=True, widths=0.85)
            for body, n in zip(parts["bodies"], shown):
                body.set_facecolor(look(n)[0])
                body.set_alpha(0.55)
                body.set_edgecolor(SURFACE)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                parts[key].set_color(INK2)
                parts[key].set_linewidth(0.8)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_xticks(range(1, len(shown) + 1))
            ax.set_xticklabels(shown, rotation=60, ha="right", fontsize=6.5)
            style(ax, f"{shape} ({len(ks)} queries)")
        for ax in np.atleast_1d(axes)[:, 0] if rows > 1 else [np.atleast_1d(axes)[0]]:
            ax.set_ylabel(label, color=INK2, fontsize=8.5)
        for ax in np.atleast_1d(axes).flat[len(shapes):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"violins_{metric}_by_shape.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # inference cost against query size
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    sizes = sorted({plans[k]["n"] for k in keys})
    for name in [n for n in names if n != "QLever"] + ["QLever"]:
        colour, ls, marker = look(name)
        field = "qlever_planning" if name == "QLever" else "inference"
        med = [np.nanmedian([table[k][name][field] for k in keys if plans[k]["n"] == s] or [np.nan]) for s in sizes]
        ax.plot(sizes, np.log10(np.maximum(med, 1e-2)), color=colour, ls=ls, marker=marker, ms=4, lw=1.4,
                label=name + (" (its own planning)" if name == "QLever" else ""))
        if name != "QLever" and any(table[k][name].get("inference_cpu") is not None for k in keys):
            cpu = [np.nanmedian([table[k][name].get("inference_cpu") or np.nan for k in keys if plans[k]["n"] == s]) for s in sizes]
            ax.plot(sizes, np.log10(np.maximum(cpu, 1e-2)), color=colour, ls=ls, marker=marker, ms=4, lw=0.8, alpha=0.45)
    style(ax, "Planning cost: inference only (GPU; faint: CPU) vs QLever's planning", "triple patterns", "median ms (log10)")
    ax.legend(frameon=False, fontsize=7, ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "planning_cost.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
