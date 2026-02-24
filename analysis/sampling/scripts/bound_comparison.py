#!/usr/bin/env python
"""
bound_comparison.py - Compare sampling bound formulas for betweenness.

Compares different approaches to determining the sampling budget for
betweenness centrality, focusing on which formulas naturally require
100% sampling (exact computation) at small reach values.

Formulas compared:
  1. Hoeffding source sampling (v2): k = log(2r/δ)/(2ε²), p = min(1, k/r)
     - Budget = p × r sources, each covering all r-1 reachable pairs
     - Naturally gives p=1 at small r

  2. R-K path sampling (v3): VD = ceil(√r), m = ⌈(⌊log₂(VD-2)⌋+1+ln(1/δ))/(2ε²)⌉
     - Fixed budget of m random paths regardless of graph size
     - Never requires full computation (m is always modest)

  3. Hoeffding pair sampling: treat each (s,t) pair as a trial
     - Total pairs T = r(r-1)/2
     - k_pairs = log(2T/δ)/(2ε²), p_pairs = min(1, k_pairs/T)
     - m = p_pairs × T paths
     - Naturally gives m=T (full) at small r

  4. Source-sampling scaled: same as (1) but budget expressed as
     equivalent path count = p × r × (r-1)/2
     - Shows how many pair observations source sampling actually generates

  5. R-K with full-computation floor: use R-K formula but enforce
     m = min(m_rk, r(r-1)/2), flag full when m_rk ≥ T

  6. Hoeffding source sampling (betweenness-adjusted): same Hoeffding
     source probability p, but the effective number of independent
     betweenness observations per source is ~√r (not r), giving:
     k = log(2r/δ)/(2ε²), p = min(1, k/(r × √r_correction))
     This is more conservative—requires more sources.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utilities import FIGURES_DIR, HOEFFDING_DELTA

DELTA = HOEFFDING_DELTA  # 0.1

# Reach values spanning small (local streets) to large (metropolitan)
REACHES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]
EPSILONS = [0.05, 0.1, 0.2]


def hoeffding_source_p(reach, eps, delta=DELTA):
    """V2 approach: Hoeffding bound → source sampling probability."""
    if reach <= 0 or eps <= 0:
        return 1.0
    k = math.log(2 * reach / delta) / (2 * eps**2)
    return min(1.0, k / reach)


def rk_path_budget(reach, eps, delta=DELTA):
    """V3 approach: R-K path sampling budget."""
    if reach <= 0:
        return 0
    vd = max(3, int(math.ceil(math.sqrt(reach))))
    vd_log = int(math.floor(math.log2(max(1, vd - 2)))) + 1
    return int(math.ceil((1 / (2 * eps**2)) * (vd_log + math.log(1 / delta))))


def hoeffding_pair_budget(reach, eps, delta=DELTA):
    """Hoeffding bound treating each (s,t) pair as a trial."""
    if reach <= 1:
        return 0
    total_pairs = reach * (reach - 1) / 2
    k = math.log(2 * total_pairs / delta) / (2 * eps**2)
    p = min(1.0, k / total_pairs)
    return p, int(math.ceil(p * total_pairs))


def source_equiv_pairs(reach, eps, delta=DELTA):
    """V2 source sampling expressed as equivalent pair observations."""
    p = hoeffding_source_p(reach, eps, delta)
    n_sources = p * reach
    # Each source generates (reach - 1) pair observations
    return p, int(math.ceil(n_sources * (reach - 1) / 2))


def print_comparison():
    """Print comparison table for all formulas."""
    for eps in EPSILONS:
        print(f"\n{'='*120}")
        print(f"  ε = {eps}, δ = {DELTA}")
        print(f"{'='*120}")
        print(
            f"  {'Reach':>7} │ {'Pairs':>9} │"
            f" {'(1) Hoeff src':>14} {'p':>6} {'full?':>5} │"
            f" {'(2) R-K m':>9} {'m/T':>7} {'full?':>5} │"
            f" {'(3) Hoeff pair':>14} {'p':>7} {'full?':>5} │"
            f" {'(4) Src→pairs':>14} {'ratio':>7}"
        )
        print("  " + "─" * 116)

        for r in REACHES:
            total_pairs = r * (r - 1) // 2 if r > 1 else 0

            # (1) Hoeffding source sampling
            p1 = hoeffding_source_p(r, eps)
            full1 = p1 >= 1.0

            # (2) R-K path sampling
            m2 = rk_path_budget(r, eps)
            ratio2 = m2 / total_pairs if total_pairs > 0 else float("inf")
            full2 = m2 >= total_pairs if total_pairs > 0 else True

            # (3) Hoeffding pair sampling
            p3, m3 = hoeffding_pair_budget(r, eps)
            full3 = p3 >= 1.0

            # (4) Source sampling → equivalent pairs
            p4, equiv4 = source_equiv_pairs(r, eps)
            ratio4 = equiv4 / total_pairs if total_pairs > 0 else float("inf")

            full1_s = "YES" if full1 else ""
            full2_s = "YES" if full2 else ""
            full3_s = "YES" if full3 else ""

            print(
                f"  {r:>7,} │ {total_pairs:>9,} │"
                f" k={math.log(2*r/DELTA)/(2*eps**2):>9,.0f} {p1:>6.3f} {full1_s:>5} │"
                f" {m2:>9,} {ratio2:>7.2%} {full2_s:>5} │"
                f" m={m3:>10,} {p3:>7.4f} {full3_s:>5} │"
                f" {equiv4:>14,} {ratio4:>7.2%}"
            )


def find_crossover_reaches():
    """Find the reach at which each formula transitions from full → sampled."""
    print(f"\n\n{'='*80}")
    print("  CROSSOVER REACHES (full → sampled transition)")
    print(f"{'='*80}")
    print(
        f"  {'ε':>6} │ {'(1) Hoeff src':>14} │ {'(2) R-K':>14} │ {'(3) Hoeff pair':>14}"
    )
    print("  " + "─" * 55)

    for eps in [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]:
        # (1) Find reach where p drops below 1
        cross1 = None
        for r in range(2, 200_000):
            p = hoeffding_source_p(r, eps)
            if p < 1.0:
                cross1 = r
                break

        # (2) Find reach where m < T
        cross2 = None
        for r in range(3, 200_000):
            total_pairs = r * (r - 1) // 2
            m = rk_path_budget(r, eps)
            if m < total_pairs:
                cross2 = r
                break

        # (3) Find reach where pair-sampling p < 1
        cross3 = None
        for r in range(2, 200_000):
            p, _ = hoeffding_pair_budget(r, eps)
            if p < 1.0:
                cross3 = r
                break

        c1 = f"{cross1:>12,}" if cross1 else "   never full"
        c2 = f"{cross2:>12,}" if cross2 else "   never full"
        c3 = f"{cross3:>12,}" if cross3 else "   never full"
        print(f"  {eps:>6.3f} │ {c1:>14} │ {c2:>14} │ {c3:>14}")


def generate_figure():
    """Generate comparison figure."""
    print("\nGenerating bound comparison figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    reach_range = np.logspace(0.7, 5, 1000)  # 5 to 100,000

    colours = {
        "(1) Hoeffding source p": "#2166AC",
        "(2) R-K m / T": "#B2182B",
        "(3) Hoeffding pair p": "#4DAF4A",
    }

    for col, eps in enumerate(EPSILONS):
        # ---- Top row: Sampling fraction (p or m/T) ----
        ax = axes[0, col]

        p_source = [hoeffding_source_p(r, eps) for r in reach_range]
        m_rk = [rk_path_budget(r, eps) for r in reach_range]
        t_pairs = [r * (r - 1) / 2 for r in reach_range]
        frac_rk = [m / t if t > 0 else 1.0 for m, t in zip(m_rk, t_pairs)]
        # Clamp to [0, 1] for display
        frac_rk = [min(1.0, f) for f in frac_rk]

        p_pair = []
        for r in reach_range:
            pp, _ = hoeffding_pair_budget(r, eps)
            p_pair.append(pp)

        ax.plot(reach_range, p_source, "-", color=colours["(1) Hoeffding source p"],
                linewidth=2, label="(1) Hoeffding source p")
        ax.plot(reach_range, frac_rk, "--", color=colours["(2) R-K m / T"],
                linewidth=2, label="(2) R-K m/T")
        ax.plot(reach_range, p_pair, "-.", color=colours["(3) Hoeffding pair p"],
                linewidth=2, label="(3) Hoeffding pair p")

        ax.axhline(1.0, color="grey", linewidth=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Reach (r)")
        ax.set_ylabel("Sampling Fraction")
        ax.set_title(f"ε = {eps}")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="right", fontsize=8)

        # ---- Bottom row: Sample budget (absolute) ----
        ax = axes[1, col]

        k_source = [math.log(2 * r / DELTA) / (2 * eps**2) for r in reach_range]
        m_rk_abs = [rk_path_budget(r, eps) for r in reach_range]
        m_pair = [hoeffding_pair_budget(r, eps)[1] for r in reach_range]

        ax.plot(reach_range, k_source, "-", color=colours["(1) Hoeffding source p"],
                linewidth=2, label="(1) Hoeffding k (sources)")
        ax.plot(reach_range, m_rk_abs, "--", color=colours["(2) R-K m / T"],
                linewidth=2, label="(2) R-K m (paths)")
        ax.plot(reach_range, m_pair, "-.", color=colours["(3) Hoeffding pair p"],
                linewidth=2, label="(3) Hoeffding k (pairs)")
        ax.plot(reach_range, t_pairs, ":", color="black", linewidth=1, alpha=0.5,
                label="T = r(r-1)/2 (all pairs)")
        ax.plot(reach_range, reach_range, ":", color="grey", linewidth=1, alpha=0.5,
                label="r (all sources)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reach (r)")
        ax.set_ylabel("Sample Budget")
        ax.set_title(f"ε = {eps}")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        f"Betweenness Sampling Bound Comparison (δ = {DELTA})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "bound_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("  BETWEENNESS SAMPLING BOUND COMPARISON")
    print("=" * 80)
    print(f"  δ = {DELTA}")
    print()
    print("  Formulas:")
    print("    (1) Hoeffding source: k = log(2r/δ)/(2ε²), p = min(1, k/r)")
    print("    (2) R-K path:         VD = ⌈√r⌉, m = ⌈(⌊log₂(VD-2)⌋+1+ln(1/δ))/(2ε²)⌉")
    print("    (3) Hoeffding pair:   T = r(r-1)/2, k = log(2T/δ)/(2ε²), p = min(1, k/T)")
    print("    (4) Source → pairs:   equivalent pair count from (1)")

    print_comparison()
    find_crossover_reaches()
    generate_figure()

    return 0


if __name__ == "__main__":
    exit(main())
