#!/usr/bin/env python
"""
12_figure_work_test.py - Work-test decision schematic (illustrative constants, no data).

Two mini-panels of two bars each illustrate the work test of the methods section:
predicted exact work (grey) against predicted sampled work (orange), with the decision
threshold gamma x exact drawn as a dashed line. On a dense network at 20 km the
predicted sampled cost falls well below the threshold, so the test engages sampling.
On a low-live-fraction suburb, exact closeness runs only the live streets and is
already cheap, so the predicted sampled cost misses the threshold and the test
computes exactly.

Every bar height is an illustrative constant on a normalised scale (exact = 100);
nothing is read from the validation outputs. Grey and orange are the design's
canonical/method colours and separate by lightness in greyscale.

Outputs: paper/figures/fig16_work_test.pdf
"""

import matplotlib

matplotlib.use("Agg")
import figstyle
import matplotlib.pyplot as plt
from utilities import FIGURES_DIR

GAMMA = 0.75
EXACT_COST = 100.0


def main() -> int:
    figstyle.apply()
    # Half-text-width proportions: authored near the printed size (0.7 textwidth on
    # the A4 layout), so the shared type scale renders at its true point values.
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.7), sharey=True)
    threshold = GAMMA * EXACT_COST
    panels = [
        # (title, sampled cost, decision annotation, note inside the exact bar)
        ("Dense network, 20 km", 55.0, "samples", None),
        ("Suburb, closeness", 90.0, "computes exactly", "live streets only"),
    ]
    for ax, (title, sampled_cost, decision, exact_note) in zip(axes, panels, strict=True):
        bars_x = [0, 1]
        heights = [EXACT_COST, sampled_cost]
        ax.bar(
            bars_x, heights, width=0.55,
            color=[figstyle.COLOR_CANONICAL, figstyle.COLOR_METHOD],
        )
        # Decision threshold: sampling engages only below the dashed line.
        ax.axhline(threshold, color=figstyle.COLOR_INK, linestyle="--", linewidth=1.1, zorder=3)
        # The decision, stated over the sampled bar in the bar's own colour.
        ax.text(
            1, sampled_cost + 4, decision, ha="center", va="bottom",
            fontsize=figstyle.SIZE_ANNOT, style="italic", color=figstyle.COLOR_METHOD,
        )
        if exact_note:
            # Why suburban exact closeness is already cheap: it runs only the live
            # streets. Set vertically inside the grey bar in white, where it cannot
            # collide with the bars, the threshold line, or the titles.
            ax.text(
                0, EXACT_COST / 2, exact_note, ha="center", va="center", rotation=90,
                fontsize=figstyle.SIZE_ANNOT, color="white",
            )
        ax.set_title(title, fontsize=figstyle.SIZE_LEGEND)
        ax.set_xticks(bars_x)
        ax.set_xticklabels(["exact", "sampled"])
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
    # Label the threshold once, on the left panel (shared scale), in the free space
    # above the low sampled bar and to the right of the exact bar.
    axes[0].text(
        1.66, threshold + 2.5, "$\\gamma \\times$ exact", ha="right", va="bottom",
        fontsize=figstyle.SIZE_ANNOT, color=figstyle.COLOR_INK,
    )
    axes[0].set_ylabel("Predicted work (exact = 100)")
    plt.tight_layout()
    out = FIGURES_DIR / "fig16_work_test.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
