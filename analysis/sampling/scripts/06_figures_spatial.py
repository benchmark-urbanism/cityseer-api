#!/usr/bin/env python
"""
06_figures_spatial.py - Generate spatial error figures from per-node sampled caches.

Reads per-node sampled caches ({network}_sampled_{dist}m.pkl) produced by
01_validate_gla.py and 02_validate_madrid.py, plus optional sensitivity CSVs.

Figures:
  - fig7_rank_shift.png:             Spatial map of per-node rank shift (all networks, 20km)
  - fig8_error_vs_reach.pdf:         Per-node error vs reach scatter (20km)
  - fig9_residual_histogram.pdf:     Distribution of normalised residuals (20km)
  - fig9b_rank_displacement.png:     Spatial rank displacement (20km)
  - fig11_decile_transition.pdf:     Decile transition heatmap (all networks, 20km)
  - fig10_sensitivity.pdf:           ρ vs grid spacing s (if sensitivity CSVs exist)

Usage:
    python 06_figures_spatial.py
    python 06_figures_spatial.py --distance 10000  # Use 10km instead of 20km
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from utilities import CACHE_DIR, FIGURES_DIR, OUTPUT_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

COLOUR_CLOSENESS = "#2166AC"
COLOUR_BETWEENNESS = "#B2182B"

# Sequential colourmap: blue → red (for magnitude-only data)
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_to_red",
    [COLOUR_CLOSENESS, COLOUR_BETWEENNESS],
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_sampled_cache(network: str, dist: int) -> dict | None:
    """Load per-node sampled cache for a network and distance."""
    cache_path = CACHE_DIR / f"{network}_sampled_{dist}m.pkl"
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


def generate_fig7_spatial_error(gla_data: dict, madrid_data: dict | None, dist: int, cary_data: dict | None = None):
    """Spatial map of per-node RANK SHIFT under sampling.

    For each node: |percentile-rank(true) - percentile-rank(sampled)| in percentile
    points --- the quantity the schedule aims to preserve. Hexbin (mean per cell) on a
    FIXED 0-10 scale common to all panels with a sequential colourmap, so a pale map means
    ranks barely move and panels are directly comparable. Binning (rather than a dense
    scatter) avoids overplotting small values into apparent saturation.
    Rows: GLA, Madrid, Cary; columns: closeness, betweenness.
    """
    print(f"\nGenerating Figure 7: spatial rank-shift map ({dist // 1000}km)...")

    nets = [("GLA", gla_data)]
    if madrid_data is not None:
        nets.append(("Madrid", madrid_data))
    if cary_data is not None:
        nets.append(("Cary", cary_data))

    nrows = len(nets)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    dist_km = dist // 1000
    crop = 10000
    vmax = 10.0  # percentile points; fixed scale common to all panels
    letters = "ABCDEFGH"
    li = 0
    for row, (label, data) in enumerate(nets):
        for col, (metric, true_key, est_key) in enumerate(
            [("Closeness", "true_harmonic", "est_harmonic"), ("Betweenness", "true_betweenness", "est_betweenness")]
        ):
            title = f"{letters[li]}) {label} {metric} ({dist_km}km)"
            li += 1
            ax = axes[row, col]
            t, e = data.get(true_key), data.get(est_key)
            if t is None or e is None:
                ax.set_title(f"{letters[li - 1]}) {label} {metric} (no data)")
                continue
            t, e = np.asarray(t, float), np.asarray(e, float)
            x, y = np.asarray(data["node_x"], float), np.asarray(data["node_y"], float)
            valid = (t != 0) | (e != 0)
            n = max(int(valid.sum()), 1)
            shift = np.full(len(t), np.nan)
            shift[valid] = (
                np.abs(rankdata(t[valid], method="average") - rankdata(e[valid], method="average")) / n * 100.0
            )  # percentile-point shift
            cx, cy = np.median(x), np.median(y)
            m = valid & (x >= cx - crop) & (x <= cx + crop) & (y >= cy - crop) & (y <= cy + crop)
            hb = ax.hexbin(
                x[m],
                y[m],
                C=shift[m],
                reduce_C_function=np.median,
                gridsize=50,
                mincnt=1,
                cmap="Reds",
                vmin=0,
                vmax=vmax,
                linewidths=0.0,
                extent=(cx - crop, cx + crop, cy - crop, cy + crop),
            )
            plt.colorbar(hb, ax=ax, label="Median rank shift (percentile pts)", shrink=0.4, extend="max")
            ax.set_title(title, pad=4)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            ax.set_xlim(cx - crop, cx + crop)
            ax.set_ylim(cy - crop, cy + crop)

    fig.suptitle(
        f"Spatial Distribution of Rank Shift under Sampling ({dist_km}km)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(h_pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig7_rank_shift.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 8: PER-NODE ERROR VS REACH SCATTER
# =============================================================================


def _binned_error_panel(ax, datasets, true_key, est_key, colour, title, n_bins=10):
    """Plot binned bar chart of median absolute error by reach decile.

    Bins nodes by reach into quantiles, computes median absolute error per bin,
    and plots grouped bars (one group per network) with IQR whiskers.
    """
    bar_width = 0.8 / len(datasets)
    for d_idx, (label, data, hatch) in enumerate(datasets):
        true_vals = data.get(true_key)
        est_vals = data.get(est_key)
        if true_vals is None or est_vals is None:
            continue
        reach = data["node_reach"]
        abs_err = np.abs(true_vals - est_vals)
        # For betweenness, exclude nodes with zero true value
        if true_key == "true_betweenness":
            mask = true_vals > 0
            reach, abs_err = reach[mask], abs_err[mask]
        # Bin by reach decile
        bin_edges = np.percentile(reach, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(reach, bin_edges, right=True)
        bin_indices = np.clip(bin_indices, 1, n_bins)
        medians = []
        q25s = []
        q75s = []
        bin_labels = []
        for b in range(1, n_bins + 1):
            in_bin = bin_indices == b
            if in_bin.sum() == 0:
                continue
            err_bin = abs_err[in_bin]
            reach_bin = reach[in_bin]
            medians.append(np.median(err_bin))
            q25s.append(np.percentile(err_bin, 25))
            q75s.append(np.percentile(err_bin, 75))
            median_reach = int(np.median(reach_bin))
            if median_reach >= 1000:
                bin_labels.append(f"{median_reach / 1000:.1f}k")
            else:
                bin_labels.append(str(median_reach))
        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        x = np.arange(len(medians))
        offset = (d_idx - (len(datasets) - 1) / 2) * bar_width
        yerr_lo = np.maximum(medians - q25s, 1e-10)
        yerr_hi = q75s - medians
        ax.bar(
            x + offset,
            medians,
            bar_width * 0.9,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            color=colour,
            alpha=0.7 if d_idx == 0 else 0.5,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            error_kw={"linewidth": 0.8},
        )
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Median Reach (per decile)")
    ax.set_ylabel("Median Absolute Error")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", which="both")


def generate_fig8_error_vs_reach(gla_data: dict, madrid_data: dict | None, dist: int):
    """Binned bar chart: median absolute error by reach decile.

    Shows that absolute error scales with reach (high-reach nodes have higher
    absolute error but lower normalised error — precision scales with importance).
    """
    print(f"\nGenerating Figure 8: per-node error vs reach ({dist // 1000}km)...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    datasets = [("GLA", gla_data, None)]
    if madrid_data is not None:
        datasets.append(("Madrid", madrid_data, "//"))

    _binned_error_panel(
        axes[0],
        datasets,
        "true_harmonic",
        "est_harmonic",
        COLOUR_CLOSENESS,
        "A) Closeness: Error vs Reach",
    )
    _binned_error_panel(
        axes[1],
        datasets,
        "true_betweenness",
        "est_betweenness",
        COLOUR_BETWEENNESS,
        "B) Betweenness: Error vs Reach",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "fig8_error_vs_reach.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 9: RESIDUAL HISTOGRAM
# =============================================================================


def generate_fig9_residual_histogram(gla_data: dict, madrid_data: dict | None, dist: int):
    """2x2 histogram of residuals: value residuals (top) and rank residuals (bottom).

    Top row: normalised value residuals (est - true) / true for closeness and betweenness.
    Bottom row: normalised rank displacement (rank_est - rank_true) / n for closeness and betweenness.
    """
    print(f"\nGenerating Figure 9: residual histogram ({dist // 1000}km)...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    dist_km = dist // 1000

    datasets = [("GLA", gla_data), ("Madrid", madrid_data)]

    # ---- Row 0: Value residuals ----

    # Panel A: Closeness value residuals
    ax = axes[0, 0]
    for label, data in datasets:
        if data is None or data["est_harmonic"] is None:
            continue
        true_h = data["true_harmonic"]
        est_h = data["est_harmonic"]
        mask = true_h > 0
        res = (est_h[mask] - true_h[mask]) / true_h[mask]
        ax.hist(
            res,
            bins=100,
            alpha=0.6,
            density=True,
            range=(-0.5, 0.5),
            color=COLOUR_CLOSENESS,
            edgecolor="none",
            label=f"{label} (n={mask.sum():,})",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("(est − true) / true")
    ax.set_ylabel("Density")
    ax.set_title(f"A) Closeness Value Residuals ({dist_km}km)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    # Panel B: Betweenness value residuals
    ax = axes[0, 1]
    has_betw_val = False
    for label, data in datasets:
        if data is None or data.get("est_betweenness") is None:
            continue
        true_b = data["true_betweenness"]
        est_b = data["est_betweenness"]
        nonzero = true_b > 0
        threshold = np.percentile(true_b[nonzero], 10)
        mask = true_b >= threshold
        res = (est_b[mask] - true_b[mask]) / true_b[mask]
        ax.hist(
            res,
            bins=200,
            alpha=0.6,
            density=True,
            range=(-2, 2),
            color=COLOUR_BETWEENNESS,
            edgecolor="none",
            label=f"{label} (n={mask.sum():,})",
        )
        has_betw_val = True
    if has_betw_val:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("(est − true) / true")
        ax.set_ylabel("Density")
        ax.set_title(f"B) Betweenness Value Residuals ({dist_km}km)")
        ax.legend(fontsize=9)
        ax.set_xlim(-2, 2)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("B) Betweenness (no data)")

    # ---- Row 1: Rank residuals ----

    for col_idx, (metric, true_key, est_key, colour, panel_label) in enumerate(
        [
            ("Closeness", "true_harmonic", "est_harmonic", COLOUR_CLOSENESS, "C"),
            ("Betweenness", "true_betweenness", "est_betweenness", COLOUR_BETWEENNESS, "D"),
        ]
    ):
        ax = axes[1, col_idx]
        has_data = False
        for label, data in datasets:
            if data is None:
                continue
            true_vals = data.get(true_key)
            est_vals = data.get(est_key)
            if true_vals is None or est_vals is None:
                continue

            # Rank among nodes where at least one of true/est is nonzero
            valid = (true_vals != 0) | (est_vals != 0)
            n_valid = valid.sum()

            rank_true = rankdata(true_vals[valid], method="average")
            rank_est = rankdata(est_vals[valid], method="average")

            # Signed rank displacement normalised to fraction of ranked nodes
            rank_res = (rank_est - rank_true) / n_valid

            ax.hist(
                rank_res,
                bins=200,
                alpha=0.6,
                density=True,
                range=(-0.15, 0.15),
                color=colour,
                edgecolor="none",
                label=f"{label} (n={n_valid:,})",
            )
            has_data = True

        if has_data:
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("(rank_est − rank_true) / n")
            ax.set_ylabel("Density")
            ax.set_title(f"{panel_label}) {metric} Rank Residuals ({dist_km}km)")
            ax.legend(fontsize=9)
            ax.set_xlim(-0.15, 0.15)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{panel_label}) {metric} (no data)")

    plt.tight_layout()
    out = FIGURES_DIR / "fig9_residual_histogram.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 9b: SPATIAL RANK DISPLACEMENT MAP
# =============================================================================


def _spatial_rank_panel(ax, node_x, node_y, rank_disp, n_nodes, title, crop_half=10000):
    """Plot a single spatial rank displacement panel.

    Uses a sequential colourmap: light = small displacement, dark = large.
    rank_disp is normalised to percentage of total nodes.
    """
    cx, cy = np.median(node_x), np.median(node_y)
    mask = (
        (node_x >= cx - crop_half)
        & (node_x <= cx + crop_half)
        & (node_y >= cy - crop_half)
        & (node_y <= cy + crop_half)
    )
    x, y, disp = node_x[mask], node_y[mask], rank_disp[mask]

    # Normalise to percentage of total ranked nodes
    disp_pct = disp / n_nodes * 100

    # Use 95th percentile as vmax to avoid outlier domination
    vmax = np.percentile(disp_pct, 95)
    vmax = max(vmax, 0.1)  # floor to avoid degenerate colourmap

    scatter = ax.scatter(
        x,
        y,
        c=disp_pct,
        s=0.3,
        alpha=0.7,
        cmap=CMAP_SEQUENTIAL,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )
    plt.colorbar(scatter, ax=ax, label="Rank displacement (%)", shrink=0.4)
    ax.set_title(title, pad=4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.set_xlim(cx - crop_half, cx + crop_half)
    ax.set_ylim(cy - crop_half, cy + crop_half)


def generate_fig9b_rank_displacement(gla_data: dict, madrid_data: dict | None, dist: int):
    """Spatial map of per-node rank displacement: 2x2 grid.

    For each node, computes |rank_true[i] - rank_sampled[i]|. Nodes where
    true and estimated values are both zero are excluded (no meaningful rank).

    Top row: GLA closeness, GLA betweenness.
    Bottom row: Madrid closeness, Madrid betweenness.
    """
    print(f"\nGenerating Figure 9b: spatial rank displacement ({dist // 1000}km)...")

    nrows = 2 if madrid_data is not None else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    dist_km = dist // 1000
    panel_labels = iter("ABCDEFGH")

    for row_idx, (net_label, data) in enumerate([("GLA", gla_data), ("Madrid", madrid_data)]):
        if data is None:
            for col_idx in range(2):
                label = next(panel_labels)
                axes[row_idx, col_idx].set_title(f"{label}) {net_label} (no data)")
            continue

        for col_idx, (metric, true_key, est_key, _colour_label) in enumerate(
            [
                ("Closeness", "true_harmonic", "est_harmonic", "closeness"),
                ("Betweenness", "true_betweenness", "est_betweenness", "betweenness"),
            ]
        ):
            label = next(panel_labels)
            true_vals = data.get(true_key)
            est_vals = data.get(est_key)

            if true_vals is None or est_vals is None:
                axes[row_idx, col_idx].set_title(f"{label}) {net_label} {metric} (no data)")
                continue

            # Rank among nodes where at least one of true/est is nonzero
            valid = (true_vals != 0) | (est_vals != 0)
            n_valid = valid.sum()

            # Compute ranks on valid subset (average ties)
            rank_true = np.full(len(true_vals), np.nan)
            rank_est = np.full(len(est_vals), np.nan)
            rank_true[valid] = rankdata(true_vals[valid], method="average")
            rank_est[valid] = rankdata(est_vals[valid], method="average")

            # Absolute rank displacement (NaN for excluded nodes)
            rank_disp = np.abs(rank_true - rank_est)
            # Set excluded nodes to 0 so they appear as background
            rank_disp[~valid] = 0

            _spatial_rank_panel(
                axes[row_idx, col_idx],
                data["node_x"],
                data["node_y"],
                rank_disp,
                n_valid,
                f"{label}) {net_label} {metric} ({dist_km}km)",
            )

    fig.suptitle(
        f"Spatial Distribution of Rank Displacement ({dist_km}km)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(h_pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig9b_rank_displacement.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 10: SENSITIVITY TO GRID SPACING s
# =============================================================================


def generate_fig10_sensitivity():
    """ρ vs grid spacing s, showing accuracy is stable around s=175m.

    Reads sensitivity CSVs if they exist (from --sensitivity runs).
    """
    import pandas as pd

    gla_path = OUTPUT_DIR / "gla_sensitivity.csv"
    madrid_path = OUTPUT_DIR / "madrid_sensitivity.csv"

    has_gla = gla_path.exists()
    has_madrid = madrid_path.exists()

    if not has_gla and not has_madrid:
        print("\nSkipping Figure 10: no sensitivity data found.")
        print("  Run 01_validate_gla.py --sensitivity and/or 02_validate_madrid.py --sensitivity")
        return

    print("\nGenerating Figure 10: grid spacing sensitivity...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panels = [
        ("rho_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("rho_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    for ax, (col, title, colour) in zip(axes, panels, strict=True):
        if has_gla:
            gla_df = pd.read_csv(gla_path)
            for dist, grp in gla_df.groupby("distance"):
                grp = grp.sort_values("grid_spacing")
                valid = grp.dropna(subset=[col])
                if not valid.empty:
                    ax.plot(
                        valid["grid_spacing"],
                        valid[col],
                        "o-",
                        color=colour,
                        linewidth=1.5,
                        markersize=6,
                        alpha=0.85,
                        label=f"GLA {int(dist) // 1000}km",
                    )

        if has_madrid:
            madrid_df = pd.read_csv(madrid_path)
            for dist, grp in madrid_df.groupby("distance"):
                grp = grp.sort_values("grid_spacing")
                valid = grp.dropna(subset=[col])
                if not valid.empty:
                    ax.plot(
                        valid["grid_spacing"],
                        valid[col],
                        "s--",
                        color=colour,
                        linewidth=1.5,
                        markersize=6,
                        alpha=0.7,
                        label=f"Madrid {int(dist) // 1000}km",
                    )

        # Mark the default s=175m
        ax.axvline(175, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            175,
            ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.9,
            "s=175m",
            fontsize=8,
            color="grey",
            ha="center",
            va="bottom",
            rotation=90,
        )

        # Target line
        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_xlabel("Grid Spacing s (m)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Spearman $\rho$")
    axes[0].set_ylim(0.88, 1.01)

    plt.tight_layout()
    out = FIGURES_DIR / "fig10_sensitivity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 11: DECILE TRANSITION MATRIX (HEATMAP)
# =============================================================================


def _decile_panel(ax, true_vals, est_vals, title, colour, n_groups=10):
    """Plot a single decile transition heatmap panel.

    Rows = true decile, columns = sampled decile.
    Row-normalised so each row sums to 1.0 (or 100%).
    Returns the top-decile retention rate.
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
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="equal", origin="lower")

    # Annotate cells with percentages
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val >= 1.0:
                text_colour = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=text_colour)

    ax.set_xticks(range(n_actual))
    ax.set_xticklabels(range(1, n_actual + 1), fontsize=8)
    ax.set_yticks(range(n_actual))
    ax.set_yticklabels(range(1, n_actual + 1), fontsize=8)
    ax.set_xlabel("Sampled decile")
    ax.set_ylabel("True decile")
    ax.set_title(title, fontsize=11)

    # Top-decile retention rate
    top_retention = data[n_actual - 1, n_actual - 1]

    # Print summary
    diag_retention = np.trace(data) / n_actual
    print(f"    {title}: diagonal retention = {diag_retention:.1f}%, top-decile retention = {top_retention:.1f}%")

    return im, top_retention


def generate_fig11_decile_transition(
    gla_data: dict, madrid_data: dict | None, dist: int, cary_data: dict | None = None
):
    """Decile transition matrix heatmap: 2x2 grid.

    For each (network, metric) combination, cross-tabulates true vs sampled
    decile membership. A strong diagonal means nodes stay in the same decile
    after sampling. The top-decile retention rate is annotated.

    Top row: GLA closeness, GLA betweenness.
    Bottom row: Madrid closeness, Madrid betweenness.
    """
    print(f"\nGenerating Figure 11: decile transition matrix ({dist // 1000}km)...")

    configs = []
    if gla_data is not None:
        configs.append(("GLA", gla_data))
    if madrid_data is not None:
        configs.append(("Madrid", madrid_data))
    if cary_data is not None:
        configs.append(("Cary", cary_data))

    nrows = len(configs)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 6 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    dist_km = dist // 1000
    im = None

    for row_idx, (net_label, data) in enumerate(configs):
        for col_idx, (metric, true_key, est_key, colour) in enumerate(
            [
                ("Closeness", "true_harmonic", "est_harmonic", COLOUR_CLOSENESS),
                ("Betweenness", "true_betweenness", "est_betweenness", COLOUR_BETWEENNESS),
            ]
        ):
            ax = axes[row_idx, col_idx]
            true_vals = data.get(true_key)
            est_vals = data.get(est_key)

            if true_vals is None or est_vals is None:
                ax.set_title(f"{net_label} {metric} (no data)")
                continue

            panel_label = chr(ord("A") + row_idx * 2 + col_idx)
            im, top_ret = _decile_panel(
                ax,
                true_vals,
                est_vals,
                f"{panel_label}) {net_label} {metric} ({dist_km}km)",
                colour,
            )

    plt.tight_layout()

    # Add a shared colourbar outside the plot area
    if im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("% of nodes in true decile", fontsize=10)
    out = FIGURES_DIR / "fig11_decile_transition.pdf"
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
    args = parser.parse_args()
    dist = args.distance

    print("=" * 70)
    print(f"06_figures_spatial.py - Spatial Error Figures ({dist // 1000}km)")
    print("=" * 70)

    # Load per-node caches
    print("\nLoading per-node sampled caches...")
    gla_data = load_sampled_cache("gla", dist)
    madrid_data = load_sampled_cache("madrid", dist)
    cary_data = load_sampled_cache("cary", dist)

    if gla_data is None and madrid_data is None:
        print("\nERROR: No per-node caches found. Run validation scripts with --force first:")
        print("  python 01_validate_gla.py --force")
        print("  python 02_validate_madrid.py --force")
        return 1

    # Generate figures
    if gla_data is not None:
        generate_fig7_spatial_error(gla_data, madrid_data, dist, cary_data)

    if gla_data is not None:
        generate_fig8_error_vs_reach(gla_data, madrid_data, dist)

    if gla_data is not None:
        generate_fig9_residual_histogram(gla_data, madrid_data, dist)

    if gla_data is not None:
        generate_fig9b_rank_displacement(gla_data, madrid_data, dist)

    if gla_data is not None or madrid_data is not None:
        generate_fig11_decile_transition(gla_data, madrid_data, dist, cary_data)

    generate_fig10_sensitivity()

    print("\nDone. Figures saved to:", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
