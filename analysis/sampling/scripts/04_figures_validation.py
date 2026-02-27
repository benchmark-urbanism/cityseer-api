#!/usr/bin/env python
"""
04_figures_validation.py - Generate validation figures for GLA and Madrid networks.

Reads cached validation CSVs and produces publication figures:
  - fig2_error_vs_reach.pdf:       Error vs per-node reach quartiles (GLA + Madrid)
  - fig4_validation_accuracy.pdf:  Spearman rho vs distance (closeness + betweenness)
  - fig5_validation_speedup.pdf:   Speedup vs distance (closeness + betweenness)
  - fig6_reach_comparison.pdf:     Canonical vs actual network reach

Usage:
    python 04_figures_validation.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer.config import GRID_SPACING

sys.path.insert(0, str(Path(__file__).parent))
from utilities import CACHE_DIR, FIGURES_DIR, OUTPUT_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

COLOUR_CLOSENESS = "#2166AC"
COLOUR_BETWEENNESS = "#B2182B"

DISTANCES_KM = [1, 2, 5, 10, 20]
DISTANCES_M = [d * 1000 for d in DISTANCES_KM]

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GLA and Madrid validation CSVs."""
    gla_path = OUTPUT_DIR / "gla_validation_summary.csv"
    madrid_path = OUTPUT_DIR / "madrid_validation.csv"

    if not gla_path.exists():
        raise FileNotFoundError(f"GLA validation summary not found: {gla_path}\n  Run 02_validate_gla.py first.")
    if not madrid_path.exists():
        raise FileNotFoundError(f"Madrid validation not found: {madrid_path}\n  Run 03_validate_madrid.py first.")

    gla = pd.read_csv(gla_path)
    madrid = pd.read_csv(madrid_path)

    gla["distance_km"] = gla["distance"] / 1000
    madrid["distance_km"] = madrid["distance"] / 1000

    print(f"GLA:    {len(gla)} distance rows")
    print(f"Madrid: {len(madrid)} distance rows")
    return gla, madrid


# =============================================================================
# FIG 4: ACCURACY (RHO VS DISTANCE)
# =============================================================================


def generate_fig4_accuracy(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 4: Spearman rho vs distance for GLA and Madrid, closeness and betweenness."""
    print("\nGenerating Figure 4: validation accuracy...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    panels = [
        ("rho_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("rho_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    for ax, (col, title, colour) in zip(axes, panels):
        # GLA
        gla_valid = gla.dropna(subset=[col])
        ax.plot(
            gla_valid["distance_km"],
            gla_valid[col],
            "o-",
            color=colour,
            linewidth=1.8,
            markersize=7,
            label="Greater London",
        )

        # Madrid
        madrid_valid = madrid.dropna(subset=[col])
        ax.plot(
            madrid_valid["distance_km"],
            madrid_valid[col],
            "s--",
            color=colour,
            linewidth=1.8,
            markersize=7,
            alpha=0.75,
            label="Madrid",
        )

        # Target line
        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(1.1, 0.951, r"$\rho$=0.95", fontsize=8, color="green", va="bottom")

        ax.set_xlabel("Distance (km)")
        ax.set_title(title)
        ax.set_xticks(DISTANCES_KM)
        ax.set_xlim(0, 22)
        ax.set_ylim(0.88, 1.01)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Spearman $\rho$")

    fig.suptitle("Ranking Accuracy on Real Networks", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "fig4_validation_accuracy.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 5: SPEEDUP VS DISTANCE
# =============================================================================


def generate_fig5_speedup(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 5: Speedup vs distance for GLA and Madrid, closeness and betweenness."""
    print("\nGenerating Figure 5: validation speedup...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        ("speedup_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("speedup_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    for ax, (col, title, colour) in zip(axes, panels):
        gla_valid = gla.dropna(subset=[col])
        madrid_valid = madrid.dropna(subset=[col])

        # Clip to ≥1 — sub-1 values are timing noise at distances where p=1
        gla_plot = gla_valid.copy()
        madrid_plot = madrid_valid.copy()
        gla_plot[col] = gla_plot[col].clip(lower=1.0)
        madrid_plot[col] = madrid_plot[col].clip(lower=1.0)

        ax.plot(
            gla_plot["distance_km"],
            gla_plot[col],
            "o-",
            color=colour,
            linewidth=1.8,
            markersize=7,
            label="Greater London",
        )
        ax.plot(
            madrid_plot["distance_km"],
            madrid_plot[col],
            "s--",
            color=colour,
            linewidth=1.8,
            markersize=7,
            alpha=0.75,
            label="Madrid",
        )

        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(0.5, 1.05, "1× (no speedup)", fontsize=8, color="grey", va="bottom",
                transform=ax.get_yaxis_transform())

        ax.set_yscale("log")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Speedup (×)")
        ax.set_title(title)
        ax.set_xticks(DISTANCES_KM)
        ax.set_xlim(0, 22)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

        # Clean y-axis ticks for log scale
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))

    fig.suptitle("Sampling Speedup on Real Networks", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "fig5_validation_speedup.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 6: CANONICAL REACH VS ACTUAL REACH
# =============================================================================


def load_reach_data() -> list[dict]:
    """Load mean reach per network and distance from ground truth caches."""
    rows = []

    # GLA
    for dist in [1000, 2000, 5000, 10000, 20000]:
        p = CACHE_DIR / f"gla_ground_truth_{dist}m.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            rows.append({"network": "Greater London", "distance": dist, "mean_reach": d["mean_reach"]})

    # Madrid
    for dist in [1000, 2000, 5000, 10000, 20000]:
        p = CACHE_DIR / f"madrid_ground_truth_{dist}m.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            rows.append({"network": "Madrid", "distance": dist, "mean_reach": d["mean_reach"]})

    # Synthetic topologies — topology reach values don't change with epsilon, use most recent cache
    for cache_name in sorted(CACHE_DIR.glob("sampling_analysis_v*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(cache_name, "rb") as f:
            synthetic = pd.DataFrame(pickle.load(f))
        if "topology" not in synthetic.columns:
            continue
        topo_reach = (
            synthetic[synthetic["sweep_type"] == "distance_based"]
            [["topology", "distance", "mean_reach"]]
            .drop_duplicates()
        )
        topo_labels = {"trellis": "Synthetic (trellis)", "tree": "Synthetic (tree)", "linear": "Synthetic (linear)"}
        for _, row in topo_reach.iterrows():
            rows.append({
                "network": topo_labels.get(row["topology"], row["topology"]),
                "distance": row["distance"],
                "mean_reach": row["mean_reach"],
            })
        break  # use only the most recent synthetic cache (by mtime)

    return rows


def generate_fig6_reach_comparison():
    """Figure 6: Canonical grid reach vs actual network reach.

    The canonical model r = π*d²/s² underpins the distance-based p schedule.
    Networks above the canonical curve are denser than assumed — the schedule
    is conservative (over-samples) for them. Networks below are sparser, so
    the deterministic schedule under-samples relative to reach-based Hoeffding.
    """
    print("\nGenerating Figure 6: canonical vs actual reach...")

    reach_rows = load_reach_data()
    if not reach_rows:
        print("  No reach data found — skipping.")
        return

    df = pd.DataFrame(reach_rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Canonical curve
    d_fine = np.linspace(300, 22000, 300)
    r_canonical = np.pi * d_fine**2 / GRID_SPACING**2
    ax.plot(d_fine / 1000, r_canonical, "-", color="black", linewidth=2.0,
            label=f"Canonical grid model ($s$={GRID_SPACING:.0f}m)", zorder=5)

    # Real networks
    network_styles = {
        "Greater London":    ("o", "#2166AC", 8, "-"),
        "Madrid":            ("s", "#B2182B", 8, "--"),
        "Synthetic (trellis)": ("^", "#4DAC26", 7, ":"),
        "Synthetic (tree)":    ("v", "#D01C8B", 7, ":"),
        "Synthetic (linear)":  ("D", "#F1A340", 7, ":"),
    }

    for network, style_args in network_styles.items():
        subset = df[df["network"] == network].sort_values("distance")
        if subset.empty:
            continue
        marker, colour, ms, ls = style_args
        ax.plot(subset["distance"] / 1000, subset["mean_reach"],
                marker=marker, linestyle=ls, color=colour, markersize=ms,
                linewidth=1.4, label=network, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Mean reachability (nodes)")
    ax.set_title("Canonical vs Actual Network Reach")
    ax.set_xticks([1, 2, 4, 5, 10, 20])
    ax.set_xlim(0, 23)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Shade the region above the canonical curve to indicate where the schedule is conservative
    ax.fill_between(d_fine / 1000, r_canonical, r_canonical * 20, color="#2166AC", alpha=0.04,
                    label="_nolegend_")
    ax.text(18, 200000, "Denser than canonical\n(schedule conservative)", fontsize=8, color="#2166AC",
            ha="center", va="top", style="italic")

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_reach_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 2: ERROR VS REACH (GLA + MADRID QUARTILE DATA)
# =============================================================================


def generate_fig2_error_vs_reach(gla_full: pd.DataFrame, madrid_full: pd.DataFrame):
    """Figure 2: Absolute and normalised error vs per-node reach quartiles.

    Uses GLA and Madrid validation quartile data (reach_q1-q4, mae_q1-q4)
    across distances where sampling occurs (p < 1). Shows that absolute error
    grows with reach while normalised error decreases — precision scales with
    importance.
    """
    print("\nGenerating Figure 2: error vs reach (GLA + Madrid)...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    from matplotlib.lines import Line2D

    # Build per-quartile rows — normalise by per-quartile reach (not mean_reach)
    records_abs = []
    records_norm = []

    # --- GLA ---
    for _, row in gla_full.iterrows():
        p_row = row.get("budget_param", np.nan)
        if not np.isfinite(p_row) or p_row >= 1.0:
            continue
        metric = row["metric"]
        colour = COLOUR_CLOSENESS if metric == "harmonic" else COLOUR_BETWEENNESS
        for q in [1, 2, 3, 4]:
            reach = row[f"reach_q{q}"]
            mae = row[f"mae_q{q}"]
            if reach > 0 and mae > 0:
                norm_denom = reach if metric == "harmonic" else max(reach * (reach - 1), 1)
                records_abs.append({"reach": reach, "error": mae, "colour": colour, "marker": "o"})
                records_norm.append({"reach": reach, "error": mae / norm_denom, "colour": colour, "marker": "o"})

    # --- Madrid ---
    for _, row in madrid_full.iterrows():
        for prefix, colour, is_harm in [("h", COLOUR_CLOSENESS, True), ("b", COLOUR_BETWEENNESS, False)]:
            p_col = "hoeffding_p_close" if is_harm else "hoeffding_p_betw"
            p_val = row.get(p_col, np.nan)
            if not np.isfinite(p_val) or p_val >= 1.0:
                continue
            for q in [1, 2, 3, 4]:
                reach = row.get(f"{prefix}_reach_q{q}", None)
                mae = row.get(f"{prefix}_mae_q{q}", None)
                if reach is not None and mae is not None and reach > 0 and mae > 0:
                    norm_denom = reach if is_harm else max(reach * (reach - 1), 1)
                    records_abs.append({"reach": reach, "error": mae, "colour": colour, "marker": "s"})
                    records_norm.append({"reach": reach, "error": mae / norm_denom, "colour": colour, "marker": "s"})

    df_abs = pd.DataFrame(records_abs)
    df_norm = pd.DataFrame(records_norm)

    # Shared legend handles
    legend_handles = [
        Line2D([0], [0], color=COLOUR_CLOSENESS, marker="o", linestyle="-", markersize=6, label="Closeness"),
        Line2D([0], [0], color=COLOUR_BETWEENNESS, marker="o", linestyle="-", markersize=6, label="Betweenness"),
        Line2D([0], [0], color="grey", marker="o", linestyle="none", markersize=6, label="GLA"),
        Line2D([0], [0], color="grey", marker="s", linestyle="none", markersize=6, label="Madrid"),
    ]

    for ax, df, ylabel, title, is_norm in [
        (axes[0], df_abs,  "Median Absolute Error",   "A) Absolute Error",    False),
        (axes[1], df_norm, "Median Normalised Error",  "B) Normalised Error",  True),
    ]:
        if df.empty:
            ax.set_title(title + " (no data)")
            continue

        for (colour, marker), grp in df.groupby(["colour", "marker"]):
            ax.scatter(grp["reach"], grp["error"], color=colour,
                       marker=marker, s=35, alpha=0.85, zorder=4)

        ax.legend(handles=legend_handles, fontsize=8, loc="upper right" if is_norm else "upper left")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-Node Reach")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Error vs Reach: Precision Scales with Importance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "fig2_error_vs_reach.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("04_figures_validation.py - Validation Figures")
    print("=" * 70)

    gla, madrid = load_data()
    gla_full = pd.read_csv(OUTPUT_DIR / "gla_validation.csv")

    generate_fig2_error_vs_reach(gla_full, madrid)
    generate_fig4_accuracy(gla, madrid)
    generate_fig5_speedup(gla, madrid)
    generate_fig6_reach_comparison()

    print("\nDone. Figures saved to:", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
