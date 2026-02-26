#!/usr/bin/env python
"""
04_figures_validation.py - Generate validation figures for GLA and Madrid networks.

Reads cached validation CSVs and produces three publication figures:
  - fig7_validation_accuracy.pdf:  Spearman rho vs distance (closeness + betweenness)
  - fig8_validation_speedup.pdf:   Speedup vs distance (closeness + betweenness)
  - fig9_validation_sampling.pdf:  Sampling probability vs distance vs theoretical curve

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
from cityseer.config import GRID_SPACING, HOEFFDING_EPSILON, compute_distance_p

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

    # Normalise Madrid column names to match GLA summary
    if "mean_reach" in madrid.columns and "reach" not in madrid.columns:
        madrid = madrid.rename(columns={"mean_reach": "reach"})

    gla["distance_km"] = gla["distance"] / 1000
    madrid["distance_km"] = madrid["distance"] / 1000

    print(f"GLA:    {len(gla)} distance rows")
    print(f"Madrid: {len(madrid)} distance rows")
    return gla, madrid


# =============================================================================
# FIG 7: ACCURACY (RHO VS DISTANCE)
# =============================================================================


def generate_fig7_accuracy(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 7: Spearman rho vs distance for GLA and Madrid, closeness and betweenness."""
    print("\nGenerating Figure 7: validation accuracy...")

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

    out = FIGURES_DIR / "fig7_validation_accuracy.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 8: SPEEDUP VS DISTANCE
# =============================================================================


def generate_fig8_speedup(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 8: Speedup vs distance for GLA and Madrid, closeness and betweenness."""
    print("\nGenerating Figure 8: validation speedup...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        ("speedup_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("speedup_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    for ax, (col, title, colour) in zip(axes, panels):
        gla_valid = gla.dropna(subset=[col])
        madrid_valid = madrid.dropna(subset=[col])

        # Filter out sub-1 speedups (full computation at short distances) for log scale
        gla_plot = gla_valid[gla_valid[col] > 0]
        madrid_plot = madrid_valid[madrid_valid[col] > 0]

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

    out = FIGURES_DIR / "fig8_validation_speedup.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 9: SAMPLING PROBABILITY VS DISTANCE
# =============================================================================


def generate_fig9_sampling(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 9: Empirical vs theoretical sampling probability vs distance."""
    print("\nGenerating Figure 9: sampling probability...")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Theoretical curves
    d_fine = np.linspace(500, 20000, 200)
    p_close_theory = [compute_distance_p(d, epsilon=gla["epsilon_closeness"].iloc[0]) * 100 for d in d_fine]
    p_betw_theory = [compute_distance_p(d, epsilon=gla["epsilon_betweenness"].iloc[0]) * 100 for d in d_fine]

    ax.plot(d_fine / 1000, p_close_theory, "-", color=COLOUR_CLOSENESS, linewidth=1.5,
            alpha=0.5, label="Closeness (theoretical)")
    ax.plot(d_fine / 1000, p_betw_theory, "-", color=COLOUR_BETWEENNESS, linewidth=1.5,
            alpha=0.5, label="Betweenness (theoretical)")

    # GLA empirical
    ax.plot(
        gla["distance_km"], gla["hoeffding_p_close"] * 100,
        "o", color=COLOUR_CLOSENESS, markersize=8, label="GLA closeness",
    )
    ax.plot(
        gla["distance_km"], gla["hoeffding_p_betw"] * 100,
        "o", color=COLOUR_BETWEENNESS, markersize=8, label="GLA betweenness",
    )

    # Madrid empirical
    ax.plot(
        madrid["distance_km"], madrid["hoeffding_p_close"] * 100,
        "s", color=COLOUR_CLOSENESS, markersize=8, alpha=0.75, label="Madrid closeness",
        markerfacecolor="white", markeredgewidth=2,
    )
    ax.plot(
        madrid["distance_km"], madrid["hoeffding_p_betw"] * 100,
        "s", color=COLOUR_BETWEENNESS, markersize=8, alpha=0.75, label="Madrid betweenness",
        markerfacecolor="white", markeredgewidth=2,
    )

    ax.axhline(100, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(0.5, 101, "100% (full computation)", fontsize=8, color="grey",
            transform=ax.get_yaxis_transform())

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Sampling probability (%)")
    ax.set_title("Sampling Probability vs Distance")
    ax.set_xticks(DISTANCES_KM)
    ax.set_xlim(0, 22)
    ax.set_ylim(-5, 115)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = FIGURES_DIR / "fig9_validation_sampling.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 10: CANONICAL REACH VS ACTUAL REACH
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
    for cache_name in sorted(CACHE_DIR.glob("sampling_analysis_v*.pkl")):
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
        break  # use only the most recent synthetic cache

    return rows


def generate_fig10_reach_comparison():
    """Figure 10: Canonical grid reach vs actual network reach.

    The canonical model r = π*d²/s² underpins the distance-based p schedule.
    Actual network reaches plotted above the canonical curve confirm the model
    is conservative — real networks always meet or exceed the assumed reach,
    so the method never under-samples.
    """
    print("\nGenerating Figure 10: canonical vs actual reach...")

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

    # Shade the region below the canonical curve to indicate "under-sampled" zone
    ax.fill_between(d_fine / 1000, 0, r_canonical, color="grey", alpha=0.08,
                    label="_nolegend_")
    ax.text(18, 800, "Conservative\nregion", fontsize=8, color="grey",
            ha="center", va="center", style="italic")

    plt.tight_layout()
    out = FIGURES_DIR / "fig10_reach_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 11: SAMPLING PROBABILITY + ACCURACY (DUAL AXIS)
# =============================================================================


def generate_fig11_p_and_rho(gla: pd.DataFrame, madrid: pd.DataFrame):
    """Figure 11: Sampling probability and achieved accuracy vs distance on dual y-axes.

    Left axis: p (%) from compute_distance_p — how much we sample.
    Right axis: Spearman rho — what accuracy we achieve.
    Both plotted against distance on the same x-axis.
    Two panels: closeness (left) and betweenness (right).
    """
    print("\nGenerating Figure 11: p + rho dual axis...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        ("hoeffding_p_close", "rho_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("hoeffding_p_betw",  "rho_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    d_fine = np.linspace(500, 20000, 300)
    p_theory = [compute_distance_p(d) * 100 for d in d_fine]

    for ax, (p_col, rho_col, title, colour) in zip(axes, panels):
        ax2 = ax.twinx()

        # Theoretical p curve on left axis
        ax.plot(d_fine / 1000, p_theory, "-", color="grey", linewidth=1.5,
                alpha=0.6, label="Theoretical $p$")
        ax.axhline(100, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

        # GLA empirical p
        gla_v = gla.dropna(subset=[p_col, rho_col])
        ax.plot(gla_v["distance_km"], gla_v[p_col] * 100,
                "o", color=colour, markersize=7, alpha=0.6, markerfacecolor="white",
                markeredgewidth=2)

        # Madrid empirical p
        madrid_v = madrid.dropna(subset=[p_col, rho_col])
        ax.plot(madrid_v["distance_km"], madrid_v[p_col] * 100,
                "s", color=colour, markersize=7, alpha=0.6, markerfacecolor="white",
                markeredgewidth=2)

        # GLA rho on right axis
        ax2.plot(gla_v["distance_km"], gla_v[rho_col],
                 "o-", color=colour, linewidth=1.8, markersize=7, label="Greater London $\\rho$")

        # Madrid rho on right axis
        ax2.plot(madrid_v["distance_km"], madrid_v[rho_col],
                 "s--", color=colour, linewidth=1.8, markersize=7, alpha=0.75, label="Madrid $\\rho$")

        # rho target
        ax2.axhline(0.95, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
        ax2.text(0.98, 0.955, r"$\rho$=0.95", fontsize=8, color="green",
                 ha="right", va="bottom", transform=ax2.transAxes)

        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Sampling probability (%)", color="grey")
        ax.tick_params(axis="y", labelcolor="grey")
        ax.set_xticks(DISTANCES_KM)
        ax.set_xlim(0, 22)
        ax.set_ylim(-5, 115)

        ax2.set_ylabel(r"Spearman $\rho$", color=colour)
        ax2.tick_params(axis="y", labelcolor=colour)
        ax2.set_ylim(0.88, 1.01)
        ax2.spines["right"].set_visible(True)

        ax.set_title(title)
        ax.grid(True, alpha=0.2)

        # Combined legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        theory_line = Line2D([0], [0], color="grey", linewidth=1.5, alpha=0.6, label="Theoretical $p$")
        empirical_dot = Line2D([0], [0], marker="o", color=colour, linestyle="None",
                               markersize=7, markerfacecolor="white", markeredgewidth=2,
                               label="Empirical $p$ (GLA / Madrid)")
        ax.legend(handles=[theory_line, empirical_dot] + lines2,
                  labels=["Theoretical $p$", "Empirical $p$"] + labels2,
                  loc="center right", fontsize=8)

    fig.suptitle("Sampling Probability and Achieved Accuracy vs Distance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "fig11_p_and_rho.pdf"
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

    generate_fig7_accuracy(gla, madrid)
    generate_fig8_speedup(gla, madrid)
    generate_fig9_sampling(gla, madrid)
    generate_fig10_reach_comparison()
    generate_fig11_p_and_rho(gla, madrid)

    print("\nDone. Figures saved to:", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
