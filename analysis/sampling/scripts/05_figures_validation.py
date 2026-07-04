#!/usr/bin/env python
"""
05_figures_validation.py - Generate validation figures for the three real networks.

Reads cached validation CSVs and produces publication figures (Greater London,
Madrid, and Cary, NC):
  - fig2_error_vs_reach.pdf:       Error vs per-node reach quartiles
  - fig4_validation_accuracy.pdf:  Spearman rho vs distance (closeness + betweenness)
  - fig5_validation_speedup.pdf:   Speedup vs distance (closeness + betweenness)
  - fig6_reach_comparison.pdf:     Canonical grid reach vs actual network reach

Usage:
    python 05_figures_validation.py
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
from cityseer.sampling import GRID_SPACING

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load GLA, Madrid, Cary and Woodlands validation CSVs (suburbs optional)."""
    gla_path = OUTPUT_DIR / "gla_validation_summary.csv"
    madrid_path = OUTPUT_DIR / "madrid_validation.csv"
    cary_path = OUTPUT_DIR / "cary_validation.csv"
    woodlands_path = OUTPUT_DIR / "woodlands_validation.csv"

    if not gla_path.exists():
        raise FileNotFoundError(f"GLA validation summary not found: {gla_path}\n  Run 01_validate_gla.py first.")
    if not madrid_path.exists():
        raise FileNotFoundError(f"Madrid validation not found: {madrid_path}\n  Run 02_validate_madrid.py first.")

    gla = pd.read_csv(gla_path)
    madrid = pd.read_csv(madrid_path)
    cary = pd.read_csv(cary_path) if cary_path.exists() else None
    woodlands = pd.read_csv(woodlands_path) if woodlands_path.exists() else None

    gla["distance_km"] = gla["distance"] / 1000
    madrid["distance_km"] = madrid["distance"] / 1000
    if cary is not None:
        cary["distance_km"] = cary["distance"] / 1000
    if woodlands is not None:
        woodlands["distance_km"] = woodlands["distance"] / 1000

    print(f"GLA:    {len(gla)} distance rows")
    print(f"Madrid: {len(madrid)} distance rows")
    print(f"Cary:   {len(cary) if cary is not None else 0} distance rows")
    print(f"Woodlands: {len(woodlands) if woodlands is not None else 0} distance rows")
    return gla, madrid, cary, woodlands


# =============================================================================
# FIG 4: ACCURACY (RHO VS DISTANCE)
# =============================================================================


def generate_fig4_accuracy(gla, madrid, cary=None, woodlands=None):
    """Figure 4: Spearman rho vs distance for the real networks, closeness and betweenness."""
    print("\nGenerating Figure 4: validation accuracy...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    panels = [
        ("rho_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("rho_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    # (dataframe, marker, linestyle, alpha, label)
    series = [(gla, "o", "-", 1.0, "Greater London"), (madrid, "s", "--", 0.75, "Madrid")]
    if cary is not None:
        series.append((cary, "^", ":", 0.7, "Cary (suburban)"))
    if woodlands is not None:
        series.append((woodlands, "D", "-.", 0.65, "The Woodlands (held-out)"))

    for ax, (col, title, colour) in zip(axes, panels, strict=True):
        for df, marker, ls, alpha, label in series:
            valid = df.dropna(subset=[col])
            ax.plot(
                valid["distance_km"],
                valid[col],
                marker=marker,
                linestyle=ls,
                color=colour,
                linewidth=1.8,
                markersize=7,
                alpha=alpha,
                label=label,
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


def _load_live_fraction(network: str) -> float:
    """Load live fraction from cached node info JSON."""
    import json

    path = CACHE_DIR / f"{network}_n_nodes.json"
    if path.exists():
        with open(path) as f:
            info = json.load(f)
            return info.get("live_fraction", 1.0)
    return 1.0


def generate_fig5_speedup(gla, madrid, cary=None, woodlands=None):
    """Figure 5: Speedup vs distance for the real networks, closeness and betweenness.

    Only distances where sampling was actually used (p < live_fraction) are shown.
    """
    print("\nGenerating Figure 5: validation speedup...")

    # (dataframe, network_key, marker, linestyle, alpha, label)
    series = [(gla, "gla", "o", "-", 1.0, "Greater London"), (madrid, "madrid", "s", "--", 0.75, "Madrid")]
    if cary is not None:
        series.append((cary, "cary", "^", ":", 0.7, "Cary (suburban)"))
    if woodlands is not None:
        series.append((woodlands, "woodlands", "D", "-.", 0.65, "The Woodlands (held-out)"))

    # Filter each network to the distances where sampling actually engaged (p < phi)
    prepared = []
    for df, key, marker, ls, alpha, label in series:
        phi = _load_live_fraction(key)
        sampled = df[df["hoeffding_p_close"] < phi].copy()
        prepared.append((sampled, marker, ls, alpha, f"{label} ($\\varphi$={phi:.2f})"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        ("speedup_closeness", "A) Closeness", COLOUR_CLOSENESS),
        ("speedup_betweenness", "B) Betweenness", COLOUR_BETWEENNESS),
    ]

    for ax, (col, title, colour) in zip(axes, panels, strict=True):
        for sampled, marker, ls, alpha, label in prepared:
            valid = sampled.dropna(subset=[col])
            if valid.empty:
                continue
            ax.plot(
                valid["distance_km"],
                valid[col],
                marker=marker,
                linestyle=ls,
                color=colour,
                linewidth=1.8,
                markersize=7,
                alpha=alpha,
                label=label,
            )

        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)

        ax.set_yscale("log")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Speedup (×)")
        ax.set_title(title)
        ax.set_xlim(0, 22)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

        # Clean y-axis ticks: no scientific notation
        fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}×")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)

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

    # Cary (suburban)
    for dist in [1000, 2000, 5000, 10000, 20000]:
        p = CACHE_DIR / f"cary_ground_truth_{dist}m.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            rows.append({"network": "Cary (suburban)", "distance": dist, "mean_reach": d["mean_reach"]})

    # The Woodlands (held-out suburban)
    for dist in [1000, 2000, 5000, 10000, 20000]:
        p = CACHE_DIR / f"woodlands_ground_truth_{dist}m.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            rows.append({"network": "The Woodlands (held-out)", "distance": dist, "mean_reach": d["mean_reach"]})

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
    ax.plot(
        d_fine / 1000,
        r_canonical,
        "-",
        color="black",
        linewidth=2.0,
        label=f"Canonical grid model ($s$={GRID_SPACING:.0f}m)",
        zorder=5,
    )

    # Real networks (dense metros above the canonical curve, sparse suburb below)
    network_styles = {
        "Greater London": ("o", "#333333", 8, "-"),
        "Madrid": ("s", "#888888", 8, "--"),
        "Cary (suburban)": ("^", "#B2182B", 8, ":"),
        "The Woodlands (held-out)": ("D", "#E08214", 8, "-."),
    }

    for network, style_args in network_styles.items():
        subset = df[df["network"] == network].sort_values("distance")
        if subset.empty:
            continue
        marker, colour, ms, ls = style_args
        ax.plot(
            subset["distance"] / 1000,
            subset["mean_reach"],
            marker=marker,
            linestyle=ls,
            color=colour,
            markersize=ms,
            linewidth=1.4,
            label=network,
            alpha=0.85,
        )

    # annotate the assumed-vs-actual gap that drives baseline under-sampling (held-out network, 20km)
    wood = df[df["network"] == "The Woodlands (held-out)"]
    w20 = wood[wood["distance"] == 20000]
    if not w20.empty:
        actual = float(w20.iloc[0]["mean_reach"])
        assumed = float(np.pi * 20000**2 / GRID_SPACING**2)
        ax.annotate(
            "",
            xy=(20, actual),
            xytext=(20, assumed),
            arrowprops={"arrowstyle": "<->", "color": "#B2182B", "lw": 1.2},
        )
        ax.text(
            20.4,
            np.sqrt(actual * assumed),
            f"assumed vs actual:\n{assumed / actual:.1f}x under-sampled",
            fontsize=7.5,
            color="#B2182B",
            va="center",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Mean reachability (nodes)")
    ax.set_title("Canonical vs Actual Network Reach")
    ax.set_xticks([1, 2, 4, 5, 10, 20])
    ax.set_xlim(0, 26)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Shade the region above the canonical curve to indicate where the schedule is conservative
    ax.fill_between(d_fine / 1000, r_canonical, r_canonical * 20, color="#999999", alpha=0.06, label="_nolegend_")
    ax.text(
        18,
        200000,
        "Denser than canonical\n(schedule conservative)",
        fontsize=8,
        color="#666666",
        ha="center",
        va="top",
        style="italic",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_reach_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# FIG 2: ERROR VS REACH (GLA + MADRID QUARTILE DATA)
# =============================================================================


def generate_fig2_error_vs_reach(
    gla_full: pd.DataFrame,
    madrid_full: pd.DataFrame,
    cary_full: pd.DataFrame | None = None,
    woodlands_full: pd.DataFrame | None = None,
):
    """Figure 2: Absolute and relative error vs per-node reach quartiles.

    Uses GLA and Madrid validation quartile data (reach_q1-q4, mae_q1-q4,
    median_true_q1-q4) across distances where sampling occurs (p < 1). Shows
    that absolute error grows with reach while relative error (mae/true_value)
    decreases — precision scales with importance.
    """
    print("\nGenerating Figure 2: error vs reach (GLA + Madrid)...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    from matplotlib.lines import Line2D

    records_abs = []
    records_rel = []

    gla_phi = _load_live_fraction("gla")
    madrid_phi = _load_live_fraction("madrid")

    # --- GLA ---
    for _, row in gla_full.iterrows():
        p_row = row.get("budget_param", np.nan)
        if not np.isfinite(p_row) or p_row >= gla_phi:
            continue
        metric = row["metric"]
        colour = COLOUR_CLOSENESS if metric == "harmonic" else COLOUR_BETWEENNESS
        for q in [1, 2, 3, 4]:
            reach = row[f"reach_q{q}"]
            mae = row[f"mae_q{q}"]
            median_true = row.get(f"median_true_q{q}", np.nan)
            if reach > 0 and mae > 0:
                records_abs.append({"reach": reach, "error": mae, "colour": colour, "marker": "o"})
                if np.isfinite(median_true) and median_true > 0:
                    records_rel.append({"reach": reach, "error": mae / median_true, "colour": colour, "marker": "o"})

    # --- Wide-schema networks (Madrid, Cary) ---
    def add_wide_network(df, phi, marker):
        if df is None:
            return
        for _, row in df.iterrows():
            for prefix, colour in [("h", COLOUR_CLOSENESS), ("b", COLOUR_BETWEENNESS)]:
                p_col = "hoeffding_p_close" if prefix == "h" else "hoeffding_p_betw"
                p_val = row.get(p_col, np.nan)
                if not np.isfinite(p_val) or p_val >= phi:
                    continue
                for q in [1, 2, 3, 4]:
                    reach = row.get(f"{prefix}_reach_q{q}", None)
                    mae = row.get(f"{prefix}_mae_q{q}", None)
                    median_true = row.get(f"{prefix}_median_true_q{q}", np.nan)
                    if reach is not None and mae is not None and reach > 0 and mae > 0:
                        records_abs.append({"reach": reach, "error": mae, "colour": colour, "marker": marker})
                        if np.isfinite(median_true) and median_true > 0:
                            records_rel.append(
                                {"reach": reach, "error": mae / median_true, "colour": colour, "marker": marker}
                            )

    add_wide_network(madrid_full, madrid_phi, "s")
    add_wide_network(cary_full, _load_live_fraction("cary"), "^")
    add_wide_network(woodlands_full, _load_live_fraction("woodlands"), "D")

    df_abs = pd.DataFrame(records_abs)
    df_rel = pd.DataFrame(records_rel)

    # Shared legend handles
    legend_handles = [
        Line2D([0], [0], color=COLOUR_CLOSENESS, marker="o", linestyle="-", markersize=6, label="Closeness"),
        Line2D([0], [0], color=COLOUR_BETWEENNESS, marker="o", linestyle="-", markersize=6, label="Betweenness"),
        Line2D([0], [0], color="grey", marker="o", linestyle="none", markersize=6, label="GLA"),
        Line2D([0], [0], color="grey", marker="s", linestyle="none", markersize=6, label="Madrid"),
        Line2D([0], [0], color="grey", marker="^", linestyle="none", markersize=6, label="Cary"),
        Line2D([0], [0], color="grey", marker="D", linestyle="none", markersize=6, label="Woodlands"),
    ]

    for ax, df, ylabel, title, is_rel in [
        (axes[0], df_abs, "Median Absolute Error", "A) Absolute Error", False),
        (axes[1], df_rel, "Median Relative Error", "B) Relative Error", True),
    ]:
        if df.empty:
            ax.set_title(title + " (no data)")
            continue

        for (colour, marker), grp in df.groupby(["colour", "marker"]):
            ax.scatter(grp["reach"], grp["error"], color=colour, marker=marker, s=35, alpha=0.85, zorder=4)

        ax.legend(handles=legend_handles, fontsize=8, loc="upper right" if is_rel else "upper left")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-Node Reach")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Error vs Reach: Precision Scales with Importance", fontsize=13, fontweight="bold", y=1.02)
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
    print("05_figures_validation.py - Validation Figures")
    print("=" * 70)

    gla, madrid, cary, woodlands = load_data()
    gla_full = pd.read_csv(OUTPUT_DIR / "gla_validation.csv")

    generate_fig2_error_vs_reach(gla_full, madrid, cary, woodlands)
    generate_fig4_accuracy(gla, madrid, cary, woodlands)
    generate_fig5_speedup(gla, madrid, cary, woodlands)
    generate_fig6_reach_comparison()

    print("\nDone. Figures saved to:", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
