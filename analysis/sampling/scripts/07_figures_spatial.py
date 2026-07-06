#!/usr/bin/env python
"""
07_figures_spatial.py - Generate spatial error figures from per-node sampled caches.

Reads per-node method caches ({network}_sampled_{dist}m_adaptive.pkl) produced by
validate_adaptive.py. Pass --baseline to read the canonical-schedule caches
({network}_sampled_{dist}m.pkl) instead.

Outputs:
  - fig7_rank_shift.png:             Spatial map of per-node rank shift (all networks, 20km)
  - fig8_error_vs_reach.pdf:         Per-node error vs reach, binned by decile (20km)
  - fig11_decile_transition.pdf:     Decile transition heatmap (metro networks both
                                     metrics, suburb betweenness; 20km)
  - tables/spatial_macros.tex:       Paired baseline-vs-method rank-shift statistics
                                     (held-out network, 20km; default mode only), plus
                                     cross-network summary macros (worst sampled median
                                     rank shift and worst top-decile retention)

Usage:
    python 07_figures_spatial.py
    python 07_figures_spatial.py --distance 10000  # Use 10km instead of 20km
    python 07_figures_spatial.py --baseline        # canonical-schedule caches
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import figstyle
from utilities import CACHE_DIR, FIGURES_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

figstyle.apply()

# Metric colours and the rank-shift/error hexbin ramp come from the shared design
# system, so this script shares one palette, one type scale, and one sequential map
# with the rest of the figure set.
COLOUR_CLOSENESS = figstyle.COLOR_CLOSENESS
COLOUR_BETWEENNESS = figstyle.COLOR_BETWEENNESS

# Network display names come from the shared design system, so every figure in the
# set labels this network "London" (figstyle.NETWORK_LABELS), rather than the local
# "GLA" that diverged from the rest of the figures.
NETWORK_NAMES = figstyle.NETWORK_LABELS

# Per-network line styles for fig8, matching the assignment used by fig6 in
# 05_figures_validation.py, so a network keeps its dash pattern across figures and
# the four same-hue series separate in greyscale.
NETWORK_LINESTYLES = {"gla": "-", "madrid": "--", "cary": ":", "woodlands": "-."}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_sampled_cache(network: str, dist: int, suffix: str = "_adaptive") -> dict | None:
    """Load per-node sampled cache for a network and distance.

    Default reads the per-node method caches ({network}_sampled_{dist}m_adaptive.pkl,
    written by validate_adaptive.py); pass suffix="" for the canonical-schedule caches.
    """
    cache_path = CACHE_DIR / f"{network}_sampled_{dist}m{suffix}.pkl"
    if not cache_path.exists():
        print(f"  Cache not found: {cache_path}")
        return None
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    # Verify required fields
    required = {"node_x", "node_y", "true_harmonic", "est_harmonic", "node_reach"}
    missing = required - set(data.keys())
    if missing:
        print(f"  Cache {cache_path} missing fields: {missing}")
        return None
    print(f"  Loaded {cache_path}: {len(data['node_x'])} nodes")
    return data


# =============================================================================
# FIG 7: SPATIAL ERROR MAP (GLA, 20km)
# =============================================================================


def generate_fig7_spatial_error(gla_data, madrid_data, dist, cary_data=None, woodlands_data=None):
    """Spatial map of per-node RANK SHIFT under sampling.

    For each node: |percentile-rank(true) - percentile-rank(sampled)| in percentile
    points --- the quantity the schedule aims to preserve. Hexbin (median per cell) on a
    FIXED 0-10 scale common to all panels with a sequential colourmap, so a pale map means
    ranks barely move and panels are directly comparable. Binning (rather than a dense
    scatter) avoids overplotting small values into apparent saturation.
    Exact-routed cells (suburb closeness) are drawn as the same hexbin at zero shift,
    with an in-panel annotation, so the network footprint stays visible and the panel
    sits on the shared colour scale. Per-panel saturation diagnostics (max cell median,
    share of cells above the cap) are printed for the caption.
    Rows: closeness, betweenness; columns: one per network. The wide 2x4 layout fills the
    printed text width and matches the page economy of the rest of the figure set.
    """
    print(f"\nGenerating Figure 7: spatial rank-shift map ({dist // 1000}km)...")

    nets = [(NETWORK_NAMES["gla"], gla_data, "gla")]
    if madrid_data is not None:
        nets.append((NETWORK_NAMES["madrid"], madrid_data, "madrid"))
    if cary_data is not None:
        nets.append((NETWORK_NAMES["cary"], cary_data, "cary"))
    if woodlands_data is not None:
        nets.append((NETWORK_NAMES["woodlands"], woodlands_data, "woodlands"))

    metrics = [
        ("Closeness", "true_harmonic", "est_harmonic"),
        ("Betweenness", "true_betweenness", "est_betweenness"),
    ]
    ncols = len(nets)
    nrows = 2
    # Author at the printed text width so the shared type scale renders true size and no
    # half the page is left blank; 300 dpi keeps the hexbins crisp. Network names sit as
    # column titles on the top row and the metric is a left row label, so the maps read as
    # a small-multiples matrix with no duplicated per-panel titles.
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 4.0), constrained_layout=True)
    axes = np.atleast_2d(axes)

    vmax = 10.0  # percentile points; fixed scale common to all panels
    cmap = figstyle.CMAP_SEQUENTIAL
    letters = "ABCDEFGH"
    hb = None
    for col, (label, data, key) in enumerate(nets):
        # Centre suburb windows on the coordinate bounding box (symmetric margin) and
        # metros on the robust median (which keeps the dense core in view as the network
        # extends past the frame).
        use_bbox = key in ("cary", "woodlands")
        x = np.asarray(data["node_x"], float)
        y = np.asarray(data["node_y"], float)
        cx = 0.5 * (x.min() + x.max()) if use_bbox else np.median(x)
        cy = 0.5 * (y.min() + y.max()) if use_bbox else np.median(y)
        # Bin every panel on the same 20 km extent at gridsize 50 (400 m hexes), so the
        # per-cell max and saturation statistics stay comparable across panels and match
        # the caption. Every panel also DISPLAYS the same 20 km window at a common
        # scale, so the single scale bar on the top-left panel serves all of them (the
        # caption states this). The Woodlands footprint (~16 km across) simply sits
        # inside its window with a margin.
        crop = 10000.0
        view = crop
        for row, (metric, true_key, est_key) in enumerate(metrics):
            letter = letters[row * ncols + col]
            ax = axes[row, col]
            if row == 0:
                ax.set_title(label)
            if col == 0:
                # metric once, as a left row label (small-multiples matrix)
                ax.text(
                    -0.10, 0.5, metric, transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=figstyle.SIZE_TITLE, color=figstyle.COLOR_INK,
                )
            t, e = data.get(true_key), data.get(est_key)
            if t is None or e is None:
                figstyle.panel_label(ax, letter, inside=True, halo=True)
                ax.set_axis_off()
                continue
            t, e = np.asarray(t, float), np.asarray(e, float)
            exact = bool(np.allclose(t, e))
            valid = (t != 0) | (e != 0)
            n = max(int(valid.sum()), 1)
            shift = np.full(len(t), np.nan)
            shift[valid] = (
                np.abs(rankdata(t[valid], method="average") - rankdata(e[valid], method="average")) / n * 100.0
            )  # percentile-point shift
            m = valid & (x >= cx - crop) & (x <= cx + crop) & (y >= cy - crop) & (y <= cy + crop)
            # The exact-routed panels are drawn with a visible grey hex edge so the empty
            # (zero-shift) footprint reads as a honeycomb rather than a blank box; the
            # dense sampled panels drop the edge so the low end of the ramp stays clean
            # pale pink instead of a grey wash.
            hex_edge = "#9e9e9e" if exact else "none"
            hex_lw = 0.3 if exact else 0.0
            hb = ax.hexbin(
                x[m],
                y[m],
                C=shift[m],
                reduce_C_function=np.median,
                gridsize=50,
                mincnt=1,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                linewidths=hex_lw,
                edgecolors=hex_edge,
                extent=(cx - crop, cx + crop, cy - crop, cy + crop),
            )
            cells = np.asarray(hb.get_array())
            over_pct = float((cells > vmax).mean() * 100.0)
            print(
                f"    {letter}) {label} {metric}: max cell median "
                f"{float(cells.max()):.1f} pts, {over_pct:.1f}% of cells above {vmax:.0f}"
            )
            if exact:
                # compact tag: it disambiguates the pale honeycomb from a small nonzero
                # shift, and the caption carries the fuller explanation. A soft white box
                # lifts the words off the hex-edge honeycomb behind them.
                ax.text(
                    0.5,
                    0.5,
                    "exact:\nshift = 0",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=figstyle.SIZE_ANNOT,
                    color=figstyle.COLOR_INK,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                    path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
                )
            ax.set_aspect("equal")
            ax.set_xlim(cx - view, cx + view)
            ax.set_ylim(cy - view, cy + view)
            # All panels share one 20 km window at a common scale, so a single scale
            # bar on the top-left panel serves the whole figure.
            if row == 0 and col == 0:
                figstyle.scale_bar(ax, 5000)
            figstyle.panel_label(ax, letter, inside=True, halo=True)
            ax.set_axis_off()

    # One shared colourbar under the grid: the 0-10 scale is common to all panels.
    if hb is not None:
        cbar = fig.colorbar(
            hb,
            ax=axes,
            location="bottom",
            shrink=0.5,
            aspect=40,
            extend="max",
            pad=0.02,
        )
        cbar.set_label("Median rank shift (percentile pts)")
        # integer-preferring ticks (0, 2.5, 5, 7.5, 10) match the caption's "0-10"
        cbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    out = FIGURES_DIR / "fig7_rank_shift.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 8: PER-NODE ERROR VS REACH SCATTER
# =============================================================================


def _binned_error_panel(ax, datasets, true_key, est_key, colour, metric_name, n_bins=10):
    """Plot dodged median markers of relative error by reach decile.

    Bins each network's nodes by its own reach distribution into deciles, computes
    the median of per-node relative error (|est - true| / true, zero-true nodes
    excluded) per decile, and plots a dodged median marker per network. Position
    encodes the median on a log y-axis. The x axis is ordinal (decile 1-10): reach at
    a given decile differs across networks, so no single reach value labels a decile.
    Relative error makes the panels comparable across networks whose absolute values
    differ by orders of magnitude. The median trend is the message and is the only
    plotted mark, so it fills the panel rather than being squeezed by tall spread
    whiskers. Networks are distinguished by marker shape, per-network line style
    (NETWORK_LINESTYLES), and lightness steps of the panel hue, and each series
    carries a direct label at its right-hand end, so the panel decodes in greyscale
    without a trip to the legend. The y-limits are driven by the plotted medians,
    so the trend occupies the frame.
    """
    dodge = 0.8 / max(len(datasets), 1)
    tints = figstyle.NETWORK_TINT_STEPS
    all_medians: list[float] = []
    end_labels: list[tuple[str, float, tuple[float, float, float]]] = []
    # faint wash on the top decile, where the highest-value nodes carry the lowest
    # relative error and the series converge low: it directs the eye to the message
    # ("lowest in the top decile") through the mid-decile crossings.
    ax.axvspan(9.5, 10.5, color=figstyle.COLOR_FAINT, zorder=0)
    for d_idx, (label, data, marker, linestyle) in enumerate(datasets):
        true_vals = data.get(true_key)
        est_vals = data.get(est_key)
        if true_vals is None or est_vals is None:
            continue
        reach = data["node_reach"]
        # relative error: absolute errors scale with network size and value magnitude,
        # so cross-network panels must normalise per node (zero-true nodes excluded)
        mask = np.asarray(true_vals, float) > 0
        reach = np.asarray(reach, float)[mask]
        rel_err = np.abs(true_vals[mask] - est_vals[mask]) / true_vals[mask]
        # Bin by reach decile (per network)
        bin_edges = np.percentile(reach, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(reach, bin_edges, right=True)
        bin_indices = np.clip(bin_indices, 1, n_bins)
        medians = np.full(n_bins, np.nan)
        for b in range(1, n_bins + 1):
            in_bin = bin_indices == b
            if in_bin.sum() == 0:
                continue
            medians[b - 1] = np.median(rel_err[in_bin])
        all_medians.extend(medians[np.isfinite(medians)].tolist())
        offset = (d_idx - (len(datasets) - 1) / 2) * dodge
        x = np.arange(1, n_bins + 1, dtype=float) + offset
        step = tints[d_idx % len(tints)]
        tint = figstyle.tint(colour, step)
        # the median connector runs a shade darker than its marker fill while preserving
        # the inter-network lightness order (0.0, 0.08, 0.15, 0.20), so each series' line
        # stays trackable through the crossings in the betweenness panel. A fixed darkening
        # offset instead clamped the two darkest networks onto one identical line colour.
        line_tint = figstyle.tint(colour, step * 0.5)
        # draw the darkest networks on top (darker = higher zorder): the reference
        # series that carries the roughly-constant message is not buried under paler noise
        zbias = len(datasets) - d_idx
        ax.plot(x, medians, linestyle=linestyle, color=line_tint, linewidth=1.8, alpha=1.0, zorder=3 + zbias)
        ax.plot(
            x,
            medians,
            linestyle="none",
            marker=marker,
            markersize=6,
            markerfacecolor=tint,
            markeredgecolor=colour,
            markeredgewidth=0.6,
            label=label,
            zorder=3 + zbias + 0.5,
        )
        finite = np.isfinite(medians)
        if finite.any():
            end_labels.append((label, float(medians[finite][-1]), line_tint))
    ax.set_xticks(np.arange(1, n_bins + 1))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    # label the 2/3/4/5 sub-decade ticks (0.02-0.05, 0.2-0.5) a point below the tick
    # size so mid-decade magnitudes read within the narrow, median-driven range; the 4
    # tick puts a reference beside the Madrid closeness peak (~0.04) and the Woodlands
    # betweenness peak (~0.5) that would otherwise sit between labelled lines.
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=(2, 3, 4, 5)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.tick_params(axis="y", which="minor", labelsize=figstyle.SIZE_TICK - 2)
    # frame the panel on the medians (not on any spread), so the roughly-constant trend
    # and the top-decile dip fill the axes instead of sitting in a thin band
    if all_medians:
        ax.set_ylim(min(all_medians) / 1.12, max(all_medians) * 1.25)
    # Direct labels at the right-hand line ends, in each series' own line colour. The
    # decile-10 medians converge (that is the message), so label positions are dodged
    # apart in log space by a minimum gap sized to the text height; the x-axis is
    # extended past decile 10 to hold the words.
    if end_labels and all_medians:
        lo, hi = min(all_medians) / 1.12, max(all_medians) * 1.25
        min_gap = 0.055 * np.log10(hi / lo)
        order = sorted(range(len(end_labels)), key=lambda i: end_labels[i][1], reverse=True)
        pos = [np.log10(end_labels[i][1]) for i in order]
        for j in range(1, len(pos)):
            pos[j] = min(pos[j], pos[j - 1] - min_gap)
        floor = np.log10(lo) + 0.5 * min_gap
        for j in range(len(pos) - 1, -1, -1):
            need = floor if j == len(pos) - 1 else pos[j + 1] + min_gap
            pos[j] = max(pos[j], need)
        for i, y_log in zip(order, pos, strict=True):
            label, _y, line_tint = end_labels[i]
            ax.text(
                10.75, 10.0**y_log, label, fontsize=figstyle.SIZE_ANNOT,
                color=line_tint, ha="left", va="center",
            )
    ax.set_xlim(0.35, 13.4)
    ax.set_xlabel("Reach decile (per network)")
    ax.set_ylabel("Median relative error")
    ax.set_title(metric_name)
    # a few recessive references at the labelled decade and sub-decade ticks
    ax.grid(True, axis="y", which="both")


def generate_fig8_error_vs_reach(
    gla_data: dict,
    madrid_data: dict | None,
    dist: int,
    cary_data: dict | None = None,
    woodlands_data: dict | None = None,
):
    """Median relative error by reach decile: dodged markers with IQR line-ranges.

    Shows that median relative error is roughly constant across reach deciles and
    lowest in the top decile: the highest-value nodes are estimated with the best
    relative precision. Networks whose closeness cell was routed to exact
    computation (zero error, undrawable on a log scale) are excluded from panel A.
    """
    print(f"\nGenerating Figure 8: per-node error vs reach ({dist // 1000}km)...")

    # Two panels at ~textwidth, authored near the printed size so the shared type
    # scale lands at its point values (a larger canvas would print below the floor).
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)

    datasets = [
        (NETWORK_NAMES[key], data, figstyle.NETWORK_MARKERS[key], NETWORK_LINESTYLES[key])
        for key, data in [
            ("gla", gla_data), ("madrid", madrid_data), ("cary", cary_data), ("woodlands", woodlands_data)
        ]
        if data is not None
    ]

    # suburbs route closeness to exact at this distance: zero error, nothing to draw
    # on a log axis, so keep them out of panel A's datasets
    exact_closeness = [
        name for name, data, _marker, _ls in datasets
        if data is not None and np.allclose(
            np.asarray(data["true_harmonic"], float), np.asarray(data["est_harmonic"], float)
        )
    ]
    sampled_closeness = [d for d in datasets if d[0] not in exact_closeness]
    _binned_error_panel(
        axes[0],
        sampled_closeness,
        "true_harmonic",
        "est_harmonic",
        COLOUR_CLOSENESS,
        "Closeness",
    )
    if exact_closeness:
        axes[0].annotate(
            ", ".join(exact_closeness) + ": exact (zero error)",
            xy=(0.02, 0.965), xycoords="axes fraction",
            fontsize=figstyle.SIZE_ANNOT, color=figstyle.COLOR_INK, va="top",
        )
    _binned_error_panel(
        axes[1],
        datasets,
        "true_betweenness",
        "est_betweenness",
        COLOUR_BETWEENNESS,
        "Betweenness",
    )
    figstyle.panel_label(axes[0], "A")
    figstyle.panel_label(axes[1], "B")

    # One figure-level legend keyed on marker shape and line style (the stable
    # network identifiers; hue carries the metric, not the network), so the two
    # panels need no duplicate per-panel legends. Neutral ink lines keep the legend
    # metric-agnostic and legible in greyscale.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [], [], linestyle=ls, color=figstyle.COLOR_INK, linewidth=1.4,
            marker=marker, markersize=6,
            markerfacecolor=figstyle.COLOR_MUTED, markeredgecolor=figstyle.COLOR_INK,
            markeredgewidth=0.6, label=label,
        )
        for label, _data, marker, ls in datasets
    ]
    # "outside" placement lets constrained_layout reserve a strip below the panels,
    # so the shared legend never collides with the x-axis labels
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))

    out = FIGURES_DIR / "fig8_error_vs_reach.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# PAIRED RANK-SHIFT STATISTICS (HELD-OUT NETWORK, BASELINE VS METHOD)
# =============================================================================


def _rank_shift_stats(data: dict, true_key: str, est_key: str, k_neighbours: int = 8) -> tuple[float, float] | None:
    """Mean rank displacement and neighbour-error correlation for one cache.

    Rank displacement is |rank(true) - rank(est)| / n, the absolute rank shift as a
    fraction of the rank range. The neighbour-error correlation is the Pearson
    correlation between a node's rank displacement and the mean displacement of its
    k nearest neighbours; it measures whether errors cluster in space.
    """
    from scipy.spatial import cKDTree

    t, e = data.get(true_key), data.get(est_key)
    if t is None or e is None:
        return None
    t, e = np.asarray(t, float), np.asarray(e, float)
    x, y = np.asarray(data["node_x"], float), np.asarray(data["node_y"], float)
    valid = (t != 0) | (e != 0)
    t, e, x, y = t[valid], e[valid], x[valid], y[valid]
    n = len(t)
    shift = np.abs(rankdata(t, method="average") - rankdata(e, method="average")) / n
    tree = cKDTree(np.c_[x, y])
    _, idx = tree.query(np.c_[x, y], k=k_neighbours + 1)
    nb_mean = shift[idx[:, 1:]].mean(axis=1)
    # exact computation gives all-zero shifts: no clustering to measure
    degenerate = shift.std() == 0.0 or nb_mean.std() == 0.0
    corr = 0.0 if degenerate else float(np.corrcoef(shift, nb_mean)[0, 1])
    return float(shift.mean()), corr


def _sampled_cell_summary(data: dict, true_key: str, est_key: str) -> tuple[float, float] | None:
    """Median rank shift (percentile points) and top-decile retention (%) for one cell.

    Returns None when the cell was computed exactly (every shift is zero), so exact-mode
    cells do not enter the cross-network summary macros.
    """
    import pandas as pd

    t, e = data.get(true_key), data.get(est_key)
    if t is None or e is None:
        return None
    t, e = np.asarray(t, float), np.asarray(e, float)
    valid = (t != 0) | (e != 0)
    t, e = t[valid], e[valid]
    n = len(t)
    if n == 0:
        return None
    shift = np.abs(rankdata(t, method="average") - rankdata(e, method="average")) / n * 100.0
    if float(shift.max()) == 0.0:
        return None  # exact mode: est equals true
    median_shift = float(np.median(shift))
    mask = (t > 0) & np.isfinite(t) & np.isfinite(e)
    true_dec = pd.qcut(t[mask], 10, labels=False, duplicates="drop")
    est_dec = pd.qcut(e[mask], 10, labels=False, duplicates="drop")
    top_true, top_est = true_dec.max(), est_dec.max()
    retention = float(np.mean(est_dec[true_dec == top_true] == top_est) * 100.0)
    return median_shift, retention


def _quartile_median_shifts(data: dict, true_key: str, est_key: str) -> list[float] | None:
    """Median rank shift (percentile points) per reach quartile for one sampled cell.

    Returns None for exact-mode cells (every shift is zero). The rank-shift statistic
    does not depend on value separation, so it complements the within-quartile rank
    correlations, which range restriction depresses.
    """
    t, e = data.get(true_key), data.get(est_key)
    reach = data.get("node_reach")
    if t is None or e is None or reach is None:
        return None
    t, e, reach = np.asarray(t, float), np.asarray(e, float), np.asarray(reach, float)
    valid = (t != 0) | (e != 0)
    t, e, reach = t[valid], e[valid], reach[valid]
    n = len(t)
    if n == 0:
        return None
    shift = np.abs(rankdata(t, method="average") - rankdata(e, method="average")) / n * 100.0
    if float(shift.max()) == 0.0:
        return None
    edges = np.percentile(reach, [25, 50, 75])
    bins = np.digitize(reach, edges)
    return [float(np.median(shift[bins == b])) for b in range(4)]


def generate_spatial_macros(dist: int, network_caches: list[tuple[str, dict]]) -> None:
    """Paired baseline-vs-method rank-shift statistics on the held-out network.

    Reads the canonical-schedule and per-node method caches for The Woodlands and
    writes LaTeX macros used by the validation section's error-structure prose.
    Also emits cross-network summary macros over the sampled cells at this distance
    (worst median rank shift, worst top-decile retention), cited by the accuracy-metric
    paragraph of the Preliminaries.
    """
    base = load_sampled_cache("woodlands", dist, suffix="")
    adap = load_sampled_cache("woodlands", dist, suffix="_adaptive")
    if base is None or adap is None:
        print("\nSkipping spatial macros: missing Woodlands baseline or method cache.")
        return
    lines = [
        "% AUTO-GENERATED by 07_figures_spatial.py - paired rank-shift statistics",
        f"% Held-out network (The Woodlands), {dist // 1000}km, canonical schedule vs per-node method.",
        "% Rank displacement: mean |rank(true)-rank(est)|/n. Neighbour-error correlation:",
        "% Pearson correlation of a node's rank displacement with the mean displacement of",
        "% its 8 nearest neighbours.",
    ]
    specs = [
        ("C", "true_harmonic", "est_harmonic"),
        ("B", "true_betweenness", "est_betweenness"),
    ]
    for suffix_label, cache, cache_label in [("Baseline", base, "baseline"), ("Adaptive", adap, "method")]:
        for metric_label, true_key, est_key in specs:
            stats = _rank_shift_stats(cache, true_key, est_key)
            if stats is None:
                continue
            disp, corr = stats
            lines.append(f"\\newcommand{{\\woodlands{suffix_label}RankDisp{metric_label}}}{{{disp:.3f}}}")
            lines.append(f"\\newcommand{{\\woodlands{suffix_label}NbrCorr{metric_label}}}{{{corr:.2f}}}")
            print(f"  {cache_label} {metric_label}: rank displacement {disp:.4f}, neighbour corr {corr:.3f}")

    # Cross-network summary over the sampled cells at this distance. Exact-mode cells
    # are excluded (their shifts are zero by construction).
    median_shifts: list[float] = []
    retentions: list[float] = []
    for name, data in network_caches:
        for metric_label, true_key, est_key in [("closeness", *specs[0][1:]), ("betweenness", *specs[1][1:])]:
            summary = _sampled_cell_summary(data, true_key, est_key)
            if summary is None:
                print(f"  {name} {metric_label}: exact mode or missing, excluded from summary")
                continue
            median_shift, retention = summary
            median_shifts.append(median_shift)
            retentions.append(retention)
            print(f"  {name} {metric_label}: median shift {median_shift:.2f} pctile pts, top-decile {retention:.1f}%")
    if median_shifts:
        lines.append("% Cross-network summary, sampled cells only, same distance:")
        lines.append("% worst (largest) median rank shift in percentile points, and worst")
        lines.append("% (smallest) top-decile retention in percent.")
        lines.append(f"\\newcommand{{\\sampledMedianShiftMax}}{{{max(median_shifts):.1f}}}")
        lines.append(f"\\newcommand{{\\sampledTopDecileRetentionMin}}{{{min(retentions):.0f}}}")

    # Worst per-reach-quartile median rank shift over the sampled closeness cells at this
    # distance: the range-restriction-robust companion to the within-quartile rho macros.
    quartile_shift_max = None
    for _name, data in network_caches:
        shifts = _quartile_median_shifts(data, "true_harmonic", "est_harmonic")
        if shifts is None:
            continue
        cell_max = max(shifts)
        quartile_shift_max = cell_max if quartile_shift_max is None else max(quartile_shift_max, cell_max)
    if quartile_shift_max is not None:
        lines.append("% Worst per-reach-quartile median rank shift (percentile points) across the")
        lines.append("% sampled closeness cells at this distance.")
        lines.append(f"\\newcommand{{\\sampledClosenessQuartileShiftMax}}{{{quartile_shift_max:.1f}}}")
    out = TABLES_DIR / "spatial_macros.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out}")


# =============================================================================
# FIG 11: DECILE TRANSITION MATRIX (HEATMAP)
# =============================================================================


def _decile_panel(ax, true_vals, est_vals, title, cmap, n_groups=10, show_xlabel=True, show_ylabel=True):
    """Plot a single decile transition heatmap panel.

    Rows = true decile, columns = sampled decile.
    Row-normalised so each row sums to 1.0 (or 100%).
    ``cmap`` carries the metric's own hue (closeness blue, betweenness red), so the
    colour identifies the metric consistently with the rest of the set. The axis labels
    are drawn only on the left column (``show_ylabel``) and bottom row (``show_xlabel``),
    since the axes mean the same thing in every panel of the small-multiples matrix.
    Returns the image (for the shared colourbars) and the top-decile retention rate.
    """
    import pandas as pd

    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    true_m, est_m = true_vals[mask], est_vals[mask]

    # Assign decile labels (1 = lowest, 10 = highest centrality)
    true_decile = pd.qcut(true_m, n_groups, labels=False, duplicates="drop") + 1
    est_decile = pd.qcut(est_m, n_groups, labels=False, duplicates="drop") + 1

    n_actual = max(true_decile.max(), est_decile.max())

    # Cross-tabulate, row-normalised
    ct = pd.crosstab(
        pd.Series(true_decile, name="True decile"),
        pd.Series(est_decile, name="Sampled decile"),
        normalize="index",
    )
    # Reindex to ensure full n_actual x n_actual grid
    full_idx = range(1, n_actual + 1)
    ct = ct.reindex(index=full_idx, columns=full_idx, fill_value=0.0)

    # Plot heatmap
    data = ct.values * 100  # convert to percentages
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="equal", origin="lower")

    # Thin white separators between cells, so each value sits in its own bounded box
    # and adjacent two-digit numbers in the tighter betweenness panels cannot merge
    # into one run of digits. Drawn above the image (axisbelow off) but below the text.
    ax.set_xticks(np.arange(-0.5, n_actual, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_actual, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.set_axisbelow(False)
    ax.tick_params(which="minor", length=0)

    # Annotate cells at or above 5%: the diagonal and the immediate off-diagonal moves
    # that define the diagonal band. Dropping the 1-4% noise lets the numbers breathe at
    # print size. The digits sit at the 7 pt legibility floor (SIZE_ANNOT-2): the colour
    # field carries the pattern and the numbers are secondary, so they stay quiet; a
    # thin contrasting halo keeps every digit legible regardless of cell tone.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val >= 5.0:
                dark_cell = val > 55
                text_colour = "white" if dark_cell else figstyle.COLOR_INK
                halo = figstyle.COLOR_INK if dark_cell else "white"
                ax.text(
                    j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=figstyle.SIZE_ANNOT - 2, color=text_colour,
                    path_effects=[patheffects.withStroke(linewidth=1.2, foreground=halo)],
                )

    # Bold outline on the top-decile retention cell (true 10 -> sampled 10), the
    # single number the caption leans on, so the eye lands on it first in every panel.
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (n_actual - 1.5, n_actual - 1.5), 1.0, 1.0, fill=False,
            edgecolor=figstyle.COLOR_INK, linewidth=1.4, zorder=4,
        )
    )

    ax.set_xticks(range(n_actual))
    ax.set_xticklabels(range(1, n_actual + 1))
    ax.set_yticks(range(n_actual))
    ax.set_yticklabels(range(1, n_actual + 1))
    ax.tick_params(labelsize=figstyle.SIZE_ANNOT - 1)
    if show_xlabel:
        ax.set_xlabel("Sampled decile")
    if show_ylabel:
        ax.set_ylabel("True decile")
    ax.set_title(title, fontsize=figstyle.SIZE_LEGEND)
    # the cell grid already bounds the matrix, so drop the axes frame
    ax.set_frame_on(False)
    ax.tick_params(length=0)

    # Top-decile retention rate
    top_retention = data[n_actual - 1, n_actual - 1]

    # Print summary
    diag_retention = np.trace(data) / n_actual
    print(f"    {title}: diagonal retention = {diag_retention:.1f}%, top-decile retention = {top_retention:.1f}%")

    return im, top_retention


def generate_fig11_decile_transition(
    gla_data: dict,
    madrid_data: dict | None,
    dist: int,
    cary_data: dict | None = None,
    woodlands_data: dict | None = None,
):
    """Decile transition matrix heatmap: metro networks (both metrics) plus suburb betweenness.

    For each (network, metric) combination, cross-tabulates true vs sampled
    decile membership. A strong diagonal means nodes stay in the same decile
    after sampling. The top-decile retention rate is annotated.

    The suburb closeness panels are omitted: the work test computes those cells
    exactly, so their matrices are identity and carry no information. Dropping
    them lets the remaining six panels print at legible size.
    """
    print(f"\nGenerating Figure 11: decile transition matrix ({dist // 1000}km)...")

    # metric hue rides the sequential ramp, so colour identifies the metric (blue
    # closeness, red betweenness) as it does elsewhere in the set
    closeness_spec = ("Closeness", "true_harmonic", "est_harmonic", figstyle.CMAP_CLOSENESS)
    betweenness_spec = ("Betweenness", "true_betweenness", "est_betweenness", figstyle.CMAP_BETWEENNESS)

    # Order the panels metric-contiguous: the closeness (blue) panels first, then every
    # betweenness (red) panel. A colourbar can then sit beside only the panels it
    # describes, so no red heatmap ever lands next to the blue closeness scale.
    metro = [(net, data) for net, data in
             [(NETWORK_NAMES["gla"], gla_data), (NETWORK_NAMES["madrid"], madrid_data)] if data is not None]
    suburb = [(net, data) for net, data in
              [(NETWORK_NAMES["cary"], cary_data), (NETWORK_NAMES["woodlands"], woodlands_data)] if data is not None]
    closeness_panels = [(net, data, closeness_spec) for net, data in metro]
    # suburb closeness is exact (identity matrix): betweenness only
    betweenness_panels = [(net, data, betweenness_spec) for net, data in metro + suburb]
    panels = closeness_panels + betweenness_panels
    n_close = len(closeness_panels)

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    # Author at the exact printed width (0.85\textwidth = 5.35 in on this A4/2.5 cm-margin
    # layout) so LaTeX applies no downscale and the shared type scale, including the cell
    # numbers, renders at its true point size; the per-row height is trimmed so the rows
    # pack close under the metric colourbars.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.35, 2.1 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    im_by_metric: dict[str, object] = {}

    for p_idx, (net_label, data, (metric, true_key, est_key, cmap)) in enumerate(panels):
        row, col = p_idx // ncols, p_idx % ncols
        ax = axes[row, col]
        letter = chr(ord("A") + p_idx)
        # two-line title (network, then metric) stays narrower than the square panel,
        # so the centred title does not overflow into the corner panel letter
        im, top_ret = _decile_panel(
            ax,
            data[true_key],
            data[est_key],
            f"{net_label}\n{metric}",
            cmap,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )
        im_by_metric[metric] = im
        # raise the letter clear of the two-line centred title
        figstyle.panel_label(ax, letter, y=1.14)
    for p_idx in range(len(panels), nrows * ncols):
        axes[p_idx // ncols, p_idx % ncols].set_axis_off()

    # One vertical colourbar per metric block, each beside only the panels it scales: a
    # blue bar to the right of the closeness row and a red bar to the right of the
    # betweenness rows, so a metric's scale never sits under a panel of the other metric.
    flat = axes.flatten()
    if "Closeness" in im_by_metric:
        cbar_c = fig.colorbar(
            im_by_metric["Closeness"], ax=list(flat[:n_close]), location="right",
            shrink=0.9, aspect=15, pad=0.02,
        )
        cbar_c.set_label("Closeness: % of true-decile row")
    if "Betweenness" in im_by_metric:
        # The betweenness bar is ~2.4x the closeness bar's length (two rows plus fixed
        # title/label overhead vs one), so its aspect is scaled by the same factor
        # (15 -> 36) to render the same thickness and share a right edge with the blue bar.
        cbar_b = fig.colorbar(
            im_by_metric["Betweenness"], ax=list(flat[n_close:len(panels)]), location="right",
            shrink=0.9, aspect=36, pad=0.02,
        )
        cbar_b.set_label("Betweenness: % of true-decile row")
    out = FIGURES_DIR / "fig11_decile_transition.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 15: PAIRED CLOSENESS RANK-SHIFT MAPS, FIXED RATE VS METHOD (HELD-OUT NETWORK)
# =============================================================================


def generate_fig15_rank_shift_paired(network: str, dist: int):
    """Paired spatial rank-shift maps: canonical schedule beside the method.

    One network (the held-out suburb), one distance, closeness only: the canonical
    schedule samples this cell and clusters its errors; the method routes it to exact. Same
    hexbin styling and fixed 0-10 percentile-point scale as fig7, so the pair reads
    directly. Betweenness improves in clustering and gradient, not magnitude, which a
    median hexbin cannot show; those claims live in the spatial_macros statistics.
    The spatial companion to fig12 panel B.
    """
    print(f"\nGenerating Figure 15: paired rank-shift maps ({network}, {dist // 1000}km)...")
    base = load_sampled_cache(network, dist, suffix="")
    adap = load_sampled_cache(network, dist, suffix="_adaptive")
    if base is None or adap is None:
        print("  Skipped: need both baseline and _adaptive caches for", network)
        return

    variants = [
        ("Canonical schedule", base, figstyle.COLOR_CANONICAL),
        ("Per-node method", adap, figstyle.COLOR_METHOD),
    ]
    # Two maps side by side; the short height keeps them width-limited so they fill the
    # columns and no dead gutter opens between them.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), constrained_layout=True)
    vmax = 10.0
    cmap = figstyle.CMAP_SEQUENTIAL
    # Both panels share the same network, so centre on the coordinate bounding box
    # (identical across the two caches) for a symmetric margin. Bin on the same 20 km
    # extent as fig7 (gridsize 50), so the per-cell statistics match the caption; only the
    # DISPLAY window is fitted to the footprint's bounding box on each axis, which fills the
    # panels with the ~16 km footprint and removes the white bands around the network. This
    # shows the same footprint as fig7's Woodlands column, so the pair stays registered.
    xc = np.asarray(base["node_x"], float)
    yc = np.asarray(base["node_y"], float)
    cx = 0.5 * (xc.min() + xc.max())
    cy = 0.5 * (yc.min() + yc.max())
    crop = 10000
    viewx = 0.5 * (xc.max() - xc.min()) + 300.0
    viewy = 0.5 * (yc.max() - yc.min()) + 300.0
    # The exact-routed panel B keeps a mid-grey hex edge so its empty (zero-shift)
    # footprint reads as a honeycomb rather than a blank box; the sampled panel A drops
    # the edge so its cell tones read cleanly off the shared ramp.
    hb = None
    for col, (variant, data, title_colour) in enumerate(variants):
        ax = axes[col]
        # the panel letter and the caption carry the metric/network/distance; the
        # title keeps only the discriminator between the two panels
        title = variant
        t = np.asarray(data["true_harmonic"], float)
        e = np.asarray(data["est_harmonic"], float)
        exact = bool(np.allclose(t, e))
        x, y = np.asarray(data["node_x"], float), np.asarray(data["node_y"], float)
        valid = (t != 0) | (e != 0)
        n = max(int(valid.sum()), 1)
        shift = np.full(len(t), np.nan)
        shift[valid] = (
            np.abs(rankdata(t[valid], method="average") - rankdata(e[valid], method="average")) / n * 100.0
        )
        hex_edge = "#9e9e9e" if exact else "none"
        hex_lw = 0.3 if exact else 0.0
        m = valid & (x >= cx - crop) & (x <= cx + crop) & (y >= cy - crop) & (y <= cy + crop)
        # the all-zero panel B fills at the zero colour of the shared white-to-red
        # scale, so the footprint stays visible and reads off the shared colourbar
        hb = ax.hexbin(
            x[m],
            y[m],
            C=shift[m],
            reduce_C_function=np.median,
            gridsize=50,
            mincnt=1,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            linewidths=hex_lw,
            edgecolors=hex_edge,
            extent=(cx - crop, cx + crop, cy - crop, cy + crop),
        )
        cell_max = float(np.nanmax(hb.get_array()))
        print(f"  Panel {'AB'[col]} ({variant}): max hexbin median shift = {cell_max:.1f} percentile pts")
        if exact:
            # same compact tag as fig7's exact panels, on a soft white box so it lifts
            # off the hex-edge honeycomb; the caption carries the fuller explanation
            ax.text(
                0.5,
                0.5,
                "exact:\nshift = 0",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=figstyle.SIZE_ANNOT,
                color=figstyle.COLOR_INK,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
            )
        # tint the title with the schedule/method colour used across the set (grey for
        # the canonical schedule, orange for the per-node method), matching fig12; a
        # small neutral subtitle names the shared subject under each title, so the
        # panels state what is mapped without a trip to the caption
        ax.set_title(title, color=title_colour, pad=18)
        ax.text(
            0.5, 1.01, "The Woodlands, closeness, 20 km", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=figstyle.SIZE_ANNOT, color=figstyle.COLOR_INK,
        )
        ax.set_aspect("equal")
        ax.set_xlim(cx - viewx, cx + viewx)
        ax.set_ylim(cy - viewy, cy + viewy)
        # both panels share one network at one scale, so a single scale bar (panel A)
        # suffices; the second would only repeat it
        if col == 0:
            figstyle.scale_bar(ax, 5000)
        figstyle.panel_label(ax, "AB"[col], inside=True, halo=True)
        ax.set_axis_off()

    # Horizontal bar under the maps, matching fig7 (same location, shrink, aspect and
    # label), so the two rank-shift figures present the identical 0-10 scale the same way.
    if hb is not None:
        cbar = fig.colorbar(
            hb,
            ax=axes,
            location="bottom",
            shrink=0.6,
            aspect=40,
            extend="max",
            pad=0.02,
        )
        cbar.set_label("Median rank shift (percentile pts)")
        cbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    out = FIGURES_DIR / "fig15_rank_shift_paired.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate spatial error figures")
    parser.add_argument(
        "--distance",
        type=int,
        default=20000,
        help="Analysis distance to use for spatial figures (default: 20000)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Read the canonical-schedule caches instead of the per-node method caches",
    )
    args = parser.parse_args()
    dist = args.distance
    suffix = "" if args.baseline else "_adaptive"

    print("=" * 70)
    print(f"07_figures_spatial.py - Spatial Error Figures ({dist // 1000}km, suffix='{suffix}')")
    print("=" * 70)

    # Load per-node caches
    print("\nLoading per-node sampled caches...")
    gla_data = load_sampled_cache("gla", dist, suffix)
    madrid_data = load_sampled_cache("madrid", dist, suffix)
    cary_data = load_sampled_cache("cary", dist, suffix)
    woodlands_data = load_sampled_cache("woodlands", dist, suffix)

    if gla_data is None and madrid_data is None:
        print("\nERROR: No per-node caches found. Run validate_adaptive.py first")
        print("(or the per-network validation scripts with --force for --baseline).")
        return 1

    # Generate figures
    if gla_data is not None:
        generate_fig7_spatial_error(gla_data, madrid_data, dist, cary_data, woodlands_data)

    if gla_data is not None:
        generate_fig8_error_vs_reach(gla_data, madrid_data, dist, cary_data, woodlands_data)

    if gla_data is not None or madrid_data is not None:
        generate_fig11_decile_transition(gla_data, madrid_data, dist, cary_data, woodlands_data)

    if not args.baseline:
        network_caches = [
            (name, data)
            for name, data in [
                ("gla", gla_data),
                ("madrid", madrid_data),
                ("cary", cary_data),
                ("woodlands", woodlands_data),
            ]
            if data is not None
        ]
        generate_spatial_macros(dist, network_caches)
        generate_fig15_rank_shift_paired("woodlands", dist)

    print("\nDone. Figures saved to:", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
