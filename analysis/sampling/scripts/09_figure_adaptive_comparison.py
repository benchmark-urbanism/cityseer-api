#!/usr/bin/env python
"""
09_figure_adaptive_comparison.py - Baseline vs per-node adaptive sampling, side by side.

Panel A: Spearman rho at 20 km for each network and metric, baseline schedule vs the
per-node method. Adaptive closeness entries computed exactly (work-test fallback) are
drawn hatched and annotated.

Panel B: per-reach-quartile rho at 20 km betweenness on the held-out network, baseline
vs adaptive: the uniformity-of-precision evidence (the baseline degrades toward the
lowest-reach quartile; the per-node method does not).

Reads output/{network}_validation[ _adaptive].csv; includes whichever networks have
adaptive results, so it can be regenerated as runs complete.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utilities import FIGURES_DIR, OUTPUT_DIR

COLOUR_BASELINE = "#999999"
COLOUR_ADAPTIVE = "#2166AC"
TARGET = 0.95

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

NETWORKS = [("gla", "London"), ("madrid", "Madrid"), ("cary", "Cary"), ("woodlands", "Woodlands")]


def canonical_20km(network: str) -> dict | None:
    """Baseline rho at 20km for both metrics, handling the two CSV schemas."""
    if network == "gla":
        path = OUTPUT_DIR / "gla_validation.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        rows = df[df["distance"] == 20000]
        rho_c = rows[rows["metric"] == "harmonic"]["spearman"]
        rho_b = rows[rows["metric"] == "betweenness"]["spearman"]
        if rho_c.empty or rho_b.empty:
            return None
        return {"rho_c": float(rho_c.iloc[0]), "rho_b": float(rho_b.iloc[0])}
    path = OUTPUT_DIR / f"{network}_validation.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["distance"] == 20000]
    if row.empty:
        return None
    return {"rho_c": float(row.iloc[0]["rho_closeness"]), "rho_b": float(row.iloc[0]["rho_betweenness"])}


def adaptive_20km(network: str) -> dict | None:
    path = OUTPUT_DIR / f"{network}_validation_adaptive.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["distance"] == 20000]
    if row.empty:
        return None
    return {
        "rho_c": float(row.iloc[0]["rho_closeness"]),
        "rho_b": float(row.iloc[0]["rho_betweenness"]),
        "exact_c": bool(row.iloc[0]["rho_closeness"] >= 0.99999),
    }


def quartile_series(path, prefix: str) -> list[float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["distance"] == 20000]
    if row.empty:
        return None
    cols = [f"{prefix}spearman_q{i}" for i in (1, 2, 3, 4)]
    if not all(c in row.columns for c in cols):
        return None
    vals = [float(row.iloc[0][c]) for c in cols]
    return vals if all(np.isfinite(vals)) else None


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Panel A: 20km rho, baseline vs adaptive ---
    ax = axes[0]
    labels, base_vals, adap_vals, exact_flags = [], [], [], []
    for key, label in NETWORKS:
        base = canonical_20km(key)
        adap = adaptive_20km(key)
        if base is None or adap is None:
            continue
        for metric, blabel in [("rho_c", "closeness"), ("rho_b", "betweenness")]:
            labels.append(f"{label}\n{blabel}")
            base_vals.append(base[metric])
            adap_vals.append(adap[metric])
            exact_flags.append(metric == "rho_c" and adap.get("exact_c", False))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, base_vals, width, color=COLOUR_BASELINE, label="baseline schedule")
    bars = ax.bar(x + width / 2, adap_vals, width, color=COLOUR_ADAPTIVE, label="per-node method")
    for bar, is_exact in zip(bars, exact_flags, strict=True):
        if is_exact:
            bar.set_hatch("//")
            ax.annotate(
                "exact",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=7.5,
                color=COLOUR_ADAPTIVE,
            )
    ax.axhline(TARGET, color="green", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(len(labels) - 0.4, TARGET + 0.002, f"$\\rho$={TARGET}", fontsize=8, color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel("Spearman $\\rho$ (20 km)")
    ax.set_title("A) Baseline vs per-node method at 20 km")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel B: quartile uniformity on the held-out network (betweenness) ---
    ax = axes[1]
    base_q = quartile_series(OUTPUT_DIR / "woodlands_validation.csv", "b_")
    adap_q = quartile_series(OUTPUT_DIR / "woodlands_validation_adaptive.csv", "b_")
    if base_q and adap_q:
        qx = [1, 2, 3, 4]
        ax.plot(qx, base_q, "o--", color=COLOUR_BASELINE, linewidth=1.8, markersize=7, label="baseline schedule")
        ax.plot(qx, adap_q, "o-", color=COLOUR_ADAPTIVE, linewidth=1.8, markersize=7, label="per-node method")
        ax.set_xticks(qx)
        ax.set_xticklabels(["q1\n(lowest reach)", "q2", "q3", "q4\n(highest reach)"], fontsize=8)
        ax.set_ylabel("Spearman $\\rho$ within quartile")
        ax.set_title("B) Precision by reach quartile\n(held-out network, betweenness, 20 km)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("B) Quartile data unavailable")

    plt.tight_layout()
    for ext in ("pdf", "svg"):
        out = FIGURES_DIR / f"fig12_baseline_vs_adaptive.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
