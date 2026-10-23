"""Figures of evaluate_planners.py's report (PNG and PDF), also on their own from an evaluation directory:

    python -m src.supervised_value_estimation.amortized_dp.online.evaluate_plots RUN_DIR/evaluation-<R>

Colour carries the estimator only (GNCE, est, value; QLever neutral), the search (greedy / beam-8 / DP / DP + bound)
is the position within a group or the line style. Ratios are per query against QLever's own plan, on a log scale.
    fig_violins           execution time and C_out relative to QLever, every planner (the aggregate)
    fig_violins_by_shape  the same per query shape, two figures, fixed y-ranges
    fig_shape_heatmap     geomean speedup and C_out ratio vs QLever per shape and planner
    fig_tails             survival curves: share of queries with C_out error / execution time above x
    fig_cout_vs_latency   execution time against C_out (all plans), and the per-query ratios to QLever
    fig_planning_cost     CPU inference ms against query size, QLever's own planning for reference
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
HUE = {"GNCE": "#2a78d6", "est": "#eb6834", "value": "#1baf7a", "QLever": "#52514e"}
FAMILY_LABEL = {"GNCE": "GNCE estimates", "est": "our estimates", "value": "value planner (ours)"}
LINE = {"greedy": "-", "beam-8": "--", "DP": ":", "DP + bound": "-."}
SHORT = {"greedy": "G", "beam-8": "B8", "DP": "DP", "DP + bound": "DP+b"}
SHORT_FAMILY = {"GNCE": "GNCE", "est": "est", "value": "value"}
DIVERGING = LinearSegmentedColormap.from_list("better_worse", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c2302f"])
EXEC_RANGE, COUT_RANGE = (-1.0, math.log10(3)), (-5.0, 2.0)   # log10 ratio to QLever shown; beyond: clipped, counted
FINE_TICKS = (0.1, 0.2, 1 / 3, 0.5, 1, 2, 3)                   # ratio ticks for ranges within a decade or two
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})


def family(name):
    return name.split(" ")[0]


def search(name):
    return "QLever" if name == "QLever" else name.split(" ", 1)[1]


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", color=INK, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def ratio_ticks(ax, lo, hi, axis="y"):
    if hi - lo < 2.5:
        ticks = [math.log10(r) for r in FINE_TICKS if lo - 1e-9 <= math.log10(r) <= hi + 1e-9]
        labels = [f"{10 ** t:.0f}×" if t >= 0 else f"1/{10 ** -t:.0f}" for t in ticks]
    else:
        ticks = list(range(math.ceil(lo), math.floor(hi) + 1))
        labels = [("1×" if t == 0 else f"{10 ** t:g}×" if 0 < t < 3 else f"1/{10 ** -t:g}" if -3 < t < 0
                   else f"$10^{{{t}}}$×") for t in ticks]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(labels)


def log_ratio(table, key, name, field):
    a, b = table[key][name][field], table[key]["QLever"][field]
    if a is None or b is None:
        return None
    return math.log10(max(a, 1) / max(b, 1))


def cout_error(table, key, name):
    known = [max(r["cout"], 1) for r in table[key].values() if r["cout"] is not None]
    c = table[key][name]["cout"]
    return None if c is None or not known else max(c, 1) / min(known)


def positions(names):
    """x position per planner: groups by estimator with a gap between them; group centres."""
    x, pos, centres, current = 0.0, {}, {}, None
    for name in names:
        f = family(name)
        if current is not None and f != current:
            x += 0.7
        pos[name] = x
        centres.setdefault(f, []).append(x)
        current = f
        x += 1.0
    return pos, {f: float(np.mean(v)) for f, v in centres.items()}


def violin_panel(ax, data, names, rng, ylabel, annotate=None, title=None, compact=False):
    pos, centres = positions(names)
    lo, hi = rng
    for name in names:
        v = np.asarray(data[name], dtype=float)
        if not len(v):
            continue
        clipped = np.clip(v, lo, hi)
        body = ax.violinplot([clipped], positions=[pos[name]], widths=0.82, showextrema=False, bw_method=0.25)["bodies"][0]
        body.set_facecolor(HUE[family(name)])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
        p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.vlines(pos[name], max(p5, lo), min(p95, hi), color=INK2, lw=0.8)
        ax.vlines(pos[name], max(p25, lo), min(p75, hi), color=INK, lw=3.2)
        ax.scatter([pos[name]], [min(max(p50, lo), hi)], s=16, color=SURFACE, edgecolor=INK, lw=0.8, zorder=3)
        above, below = int((v > hi).sum()), int((v < lo).sum())
        if above:
            ax.text(pos[name], hi, f"+{above}", ha="center", va="bottom", fontsize=6, color=INK2)
        if below:
            ax.text(pos[name], lo, f"{below}", ha="center", va="top", fontsize=6, color=INK2)
        if annotate:
            ax.text(pos[name], hi + 0.12 * (hi - lo), annotate(name), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + (0.3 if annotate else 0.06) * (hi - lo))
    ratio_ticks(ax, lo, hi)
    ax.set_xticks([pos[n] for n in names])
    ax.set_xticklabels([SHORT[search(n)] if compact else search(n) for n in names])
    for f, c in centres.items():
        ax.text(c, -0.15 if compact else -0.13, (SHORT_FAMILY if compact else FAMILY_LABEL)[f],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7 if compact else 7.5,
                color=HUE[f], fontweight="bold")
    style(ax, title, None, ylabel)
    ax.grid(False, axis="x")


def ratios(table, keys, names, field):
    return {n: [r for r in (log_ratio(table, k, n, field) for k in keys) if r is not None] for n in names}


def geo_text(values, better_below=True):
    v = np.asarray(values)
    factor = 10 ** -v.mean() if better_below else 10 ** v.mean()
    return f"{factor:.2f}×\n{(v < 0).mean():.0%} / {(v > 0).mean():.0%}"


def fig_violins(plans, table, names, out):
    keys = list(plans)
    ours = [n for n in names if n != "QLever"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4), facecolor=SURFACE)
    ex = ratios(table, keys, ours, "exec")
    violin_panel(a, ex, ours, EXEC_RANGE, "execution time / QLever's (log)", annotate=lambda n: geo_text(ex[n]),
                 title=f"(a) Execution time relative to QLever's plan, per query ({len(keys):,} queries)")
    co = ratios(table, keys, ours, "cout")
    violin_panel(b, co, ours, COUT_RANGE, "C_out / QLever's (log)", annotate=lambda n: geo_text(co[n]),
                 title="(b) Intermediate results (C_out) relative to QLever's plan, per query")
    fig.text(0.01, -0.01, "Below the dashed line: better than QLever. Violin: all queries; bar: interquartile range, thin line: "
             "5-95%, dot: median; numbers at the edges: queries beyond the shown range. Above each violin: geomean factor "
             "better than QLever, and the share of queries better / worse.", fontsize=6.5, color=INK2, wrap=True)
    fig.tight_layout(h_pad=3.2)
    save(fig, out, "fig_violins")


def fig_violins_by_shape(plans, table, names, out):
    keys = list(plans)
    ours = [n for n in names if n != "QLever"]
    shapes = sorted({plans[k]["shape"] for k in keys})
    for field, rng, label, fname in (("exec", EXEC_RANGE, "execution time / QLever's (log)", "fig_violins_by_shape_exec"),
                                     ("cout", COUT_RANGE, "C_out / QLever's (log)", "fig_violins_by_shape_cout")):
        fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.6), sharey=True, facecolor=SURFACE)
        for ax, shape in zip(axes.flat, shapes):
            ks = [k for k in keys if plans[k]["shape"] == shape]
            violin_panel(ax, ratios(table, ks, ours, field), ours, rng, label if ax in axes[:, 0] else None,
                         title=f"{shape.replace('_', '-')} ({len(ks)} queries)", compact=True)
        for ax in axes.flat[len(shapes):]:
            ax.set_visible(False)
        fig.tight_layout(h_pad=3.0, rect=(0, 0, 1, 0.975))
        fig.text(0.01, 0.995, "G greedy · B8 beam-8 · DP exact DP · DP+b DP with the degree bound. Below the dashed line: "
                 "better than QLever;\nbar: interquartile range, thin line: 5-95%, dot: median; numbers at the edges: queries "
                 "beyond the shown range.", fontsize=6.5, color=INK2, va="top")
        save(fig, out, fname)


def fig_shape_heatmap(plans, table, names, out):
    keys = list(plans)
    ours = [n for n in names if n != "QLever"]
    shapes = sorted({plans[k]["shape"] for k in keys})
    rows = [("all queries", keys)] + [(s.replace("_", "-"), [k for k in keys if plans[k]["shape"] == s]) for s in shapes]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), facecolor=SURFACE)
    for ax, field, title in ((axes[0], "exec", "(a) Execution time: geomean speedup over QLever"),
                             (axes[1], "cout", "(b) C_out: geomean factor fewer rows than QLever")):
        grid = np.array([[np.mean(ratios(table, ks, [n], field)[n]) for n in ours] for _, ks in rows])
        lim = {"exec": 0.3, "cout": 1.5}[field]
        ax.imshow(grid, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{10 ** -grid[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color=SURFACE if abs(grid[i, j]) > 0.7 * lim else INK)
        ax.set_xticks(range(len(ours)))
        ax.set_xticklabels([f"{family(n)} {search(n)}" for n in ours], rotation=55, ha="right")
        for tick, n in zip(ax.get_xticklabels(), ours):
            tick.set_color(HUE[family(n)])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r} ({len(ks)})" for r, ks in rows])
        ax.axhline(0.5, color=SURFACE, lw=2)
        style(ax, title)
        ax.grid(False)
    fig.text(0.01, -0.02, "Cells: geomean over the queries of QLever's / the planner's value; > 1 (blue) is better than QLever, "
             "< 1 (red) worse.", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig_shape_heatmap")


def survival(ax, values, colour, ls, label, lw=1.5):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if not n:
        return
    frac = 1.0 - np.arange(n) / n
    ax.step(v, frac, where="post", color=colour, ls=ls, lw=lw, label=label)


def fig_tails(plans, table, names, out, timeout_ms):
    keys = list(plans)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), facecolor=SURFACE)
    for name in names:
        colour = HUE[family(name)]
        ls = "-" if name == "QLever" else LINE[search(name)]
        lw = 1.8 if name == "QLever" else 1.2
        err = [e for e in (cout_error(table, k, name) for k in keys) if e is not None]
        survival(a, err, colour, ls, name, lw)
        survival(b, [max(table[k][name]["exec"], 1) for k in keys if table[k][name]["exec"] is not None], colour, ls, name, lw)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1 / len(keys) * 0.8, 1.05)
    a.set_xlim(1, None)
    b.axvline(timeout_ms, color=INK3, lw=0.8, ls=(0, (2, 2)))
    b.text(timeout_ms, 0.5, " timeout", color=INK2, fontsize=6.5, rotation=90, va="center", ha="right")
    style(a, "(a) C_out error tail", "C_out / cheapest plan of any planner", "share of queries above x")
    style(b, "(b) Execution time tail", "execution time (ms)", None)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1), labelcolor=INK)
    fig.tight_layout()
    save(fig, out, "fig_tails")


def binned_median(ax, x, y, colour, label, bins=12, band=True):
    x, y = np.asarray(x), np.asarray(y)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    mid, med, lo, hi = [], [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        s = (x >= e0) & (x <= e1)
        if s.sum() >= 15:
            mid.append(np.median(x[s])), med.append(np.median(y[s]))
            lo.append(np.percentile(y[s], 25)), hi.append(np.percentile(y[s], 75))
    if band:
        ax.fill_between(mid, lo, hi, color=colour, alpha=0.16, lw=0)
    ax.plot(mid, med, color=colour, lw=1.8, label=label)


def fig_cout_vs_latency(plans, table, names, out):
    keys = list(plans)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), facecolor=SURFACE)
    xs, ys = [], []
    for k in keys:
        for name in names:
            r = table[k][name]
            if r["cout"] is not None and r["exec"] is not None and not r["censored"]:
                xs.append(math.log10(max(r["cout"], 1))), ys.append(math.log10(max(r["exec"], 1)))
    a.hexbin(xs, ys, gridsize=40, bins="log", mincnt=1, cmap=LinearSegmentedColormap.from_list(
        "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]), linewidths=0.15, edgecolors=SURFACE)
    binned_median(a, xs, ys, INK, "median, interquartile band")
    a.text(0.03, 0.96, f"{len(xs):,} plans of all planners\nSpearman {spearmanr(xs, ys)[0]:.2f}", transform=a.transAxes,
           va="top", fontsize=7, color=INK)
    style(a, "(a) Execution time against C_out", "C_out (log10 rows)", "execution time (log10 ms)")
    a.legend(loc="lower right", frameon=False, labelcolor=INK)

    shown = [n for n in names if n != "QLever" and (search(n) == "DP" or n == "value beam-8")]
    for name in shown:
        pts = [(log_ratio(table, k, name, "cout"), log_ratio(table, k, name, "exec")) for k in keys]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if name == "value beam-8":
            x, y = zip(*pts)
            b.scatter(np.clip(x, -4, 1), np.clip(y, *EXEC_RANGE), s=3, color=HUE["value"], alpha=0.18, lw=0)
        binned_median(b, *zip(*pts), HUE[family(name)], name, band=name == "value beam-8")
    b.axhline(0, color=INK2, lw=0.7, ls=(0, (4, 3)))
    b.axvline(0, color=INK2, lw=0.7, ls=(0, (4, 3)))
    b.set_xlim(-4, 1)
    b.set_ylim(*EXEC_RANGE)
    ratio_ticks(b, -4, 1, axis="x")
    ratio_ticks(b, *EXEC_RANGE)
    b.text(0.02, 0.03, "fewer rows and faster", transform=b.transAxes, fontsize=6.5, color=INK2)
    b.text(0.98, 0.97, "more rows and slower", transform=b.transAxes, fontsize=6.5, color=INK2, ha="right", va="top")
    style(b, "(b) Relative to QLever, per query", "C_out / QLever's", "execution time / QLever's")
    b.legend(loc="upper left", frameon=False, labelcolor=INK)
    fig.tight_layout()
    save(fig, out, "fig_cout_vs_latency")


def fig_planning_cost(plans, table, names, out):
    keys = list(plans)
    sizes = sorted({plans[k]["n"] for k in keys})
    if all(table[k][name].get("inference") is None for k in keys for name in names if name != "QLever"):
        return                                           # no CPU timing (inference-cpu.json) yet
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=SURFACE)
    for name in names:
        f, colour = ("qlever_planning", HUE["QLever"]) if name == "QLever" else ("inference", HUE[family(name)])
        per = [np.array([table[k][name].get(f) if table[k][name].get(f) is not None else np.nan
                         for k in keys if plans[k]["n"] == s], dtype=float) for s in sizes]
        med = [np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in per]
        if name == "QLever":                     # bimodal from ~12 patterns: DP, or greedy past its budget
            lo = [np.nanpercentile(v, 25) if np.isfinite(v).any() else np.nan for v in per]
            hi = [np.nanpercentile(v, 75) if np.isfinite(v).any() else np.nan for v in per]
            ax.fill_between(sizes, np.maximum(lo, 1e-2), np.maximum(hi, 1e-2), color=colour, alpha=0.15, lw=0)
        ax.plot(sizes, np.maximum(med, 1e-2), color=colour, ls="-" if name == "QLever" else LINE[search(name)],
                lw=2.0 if name == "QLever" else 1.3,
                label="QLever's own planning (median, interquartile band)" if name == "QLever" else name)
    ax.set_yscale("log")
    style(ax, "Planning cost on the CPU (1 thread)", "triple patterns", "median ms per query")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2), labelcolor=INK)
    fig.text(0.01, -0.3, "Ours: the time inside the network calls only (cardinality model, bound, value net), one query at a "
             "time, on one CPU thread. QLever: meta.time_query_planning of its unforced execution; it plans by DP up to 1,500 connected "
             "subgraphs, greedily beyond (query-planning-budget), hence the drop for the largest queries.", fontsize=6.5,
             color=INK2, wrap=True)
    fig.tight_layout()
    save(fig, out, "fig_planning_cost")


def plot_all(plans, table, names, out, timeout_ms=30000.0):
    out = Path(out)
    fig_violins(plans, table, names, out)
    fig_violins_by_shape(plans, table, names, out)
    fig_shape_heatmap(plans, table, names, out)
    fig_tails(plans, table, names, out, timeout_ms)
    fig_cout_vs_latency(plans, table, names, out)
    fig_planning_cost(plans, table, names, out)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    plans = json.load(open(directory / "plans.json"))
    table = json.load(open(directory / "report.json"))["per_query"]
    names = list(next(iter(plans.values()))["planners"]) + ["QLever"]
    plot_all(plans, table, names, directory)
    print(f"figures -> {directory}")
