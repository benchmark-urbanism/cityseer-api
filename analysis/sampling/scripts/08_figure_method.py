#!/usr/bin/env python
"""
08_figure_method.py - Method schematic: per-node reach-based sampling on a worked example.

Four equal panels on a hypothetical network with a dense core and a sparse fringe:

  A) Pilot: measure each node's local reach (catchment sizes differ by area).
  B) Assign: per-node inclusion probability q = min(1, k(r)/r); dense low, sparse high.
  C) Sample with per-node rates: every catchment receives approximately k effective
     samples; inverse-probability weighting (1/q per source) keeps estimates unbiased.
  D) Contrast: a single fixed rate starves the sparse catchment.

Outputs both PDF (paper) and SVG (docs site).
"""

import matplotlib

matplotlib.use("Agg")
import figstyle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from utilities import FIGURES_DIR

# Exemplar hues sit outside the four semantic metric colours (closeness blue,
# betweenness red, canonical grey, method orange) so the schematic never reads as
# a closeness/betweenness figure. Teal and purple are the reserved suburb hues.
COLOUR_DENSE = "#1B9E77"  # teal   - dense-core exemplar (identity only, not a metric)
COLOUR_SPARSE = "#7B3FA0"  # purple - sparse-fringe exemplar
COLOUR_POINT = figstyle.COLOR_MUTED  # network node
COLOUR_SAMPLED = figstyle.COLOR_INK  # sampled source
COLOUR_UNSAMPLED = figstyle.COLOR_FAINT  # not sampled

RADIUS = 0.75
K_TARGET = 18.0
XLIM = (-1.95, 4.05)
YLIM = (-1.6, 1.65)


def make_network(rng: np.random.Generator) -> np.ndarray:
    """Hypothetical network: a dense urban core plus a sparse dendritic fringe."""
    core = rng.normal(loc=[0.0, 0.0], scale=[0.55, 0.55], size=(240, 2))
    arm_x = np.linspace(0.9, 3.4, 26)
    arm = np.column_stack([arm_x, rng.normal(0, 0.06, arm_x.size)])
    branches = []
    for bx in [1.5, 2.2, 2.9]:
        by = np.linspace(0.12, 0.85, 6) * rng.choice([-1.0, 1.0])
        branches.append(np.column_stack([np.full(6, bx) + rng.normal(0, 0.04, 6), by]))
    return np.vstack([core, arm, *branches])


def counts_within(pts: np.ndarray, radius: float) -> np.ndarray:
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    return (d2 <= radius**2).sum(axis=1).astype(float)


def style_panel(ax, letter: str, title: str) -> None:
    """Coordinate-free panel: bold letter stamp, concise step title, one light frame."""
    figstyle.panel_label(ax, letter)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=figstyle.COLOR_FAINT,
            linewidth=0.8,
            zorder=0,
            clip_on=False,
        )
    )


def mark_exemplar(ax, xy, colour: str, linestyle="--") -> None:
    """Crosshair-style marker: dashed catchment circle plus a centre dot.

    Each exemplar carries its own dash pattern (dense dashed, sparse dash-dot), so
    the two circles separate in greyscale as well as by hue.
    """
    ax.add_patch(Circle(xy, RADIUS, fill=False, edgecolor=colour, linewidth=1.5, linestyle=linestyle))
    ax.scatter(*xy, s=46, c=colour, zorder=6, edgecolors="white", linewidths=0.8)


def label_below(
    ax,
    dense_text: str,
    sparse_text: str,
    xfracs: tuple[float, float],
    note: str | None = None,
    sparse_weight: str = "normal",
) -> None:
    """Exemplar values beneath the panel, each centred under its own exemplar.

    ``xfracs`` are the two exemplar x-positions in axes fractions, so the labels
    track the markers when the coordinate limits change rather than sitting at
    fixed fractions. ``sparse_weight="bold"`` flags a failing sparse count (panel D).
    """
    xf_dense, xf_sparse = xfracs
    ax.text(
        xf_dense,
        -0.075,
        dense_text,
        transform=ax.transAxes,
        fontsize=figstyle.SIZE_LABEL,
        color=COLOUR_DENSE,
        ha="center",
    )
    ax.text(
        xf_sparse,
        -0.075,
        sparse_text,
        transform=ax.transAxes,
        fontsize=figstyle.SIZE_LABEL,
        color=COLOUR_SPARSE,
        ha="center",
        fontweight=sparse_weight,
    )
    if note:
        ax.text(
            0.5,
            -0.165,
            note,
            transform=ax.transAxes,
            fontsize=figstyle.SIZE_ANNOT,
            style="italic",
            color=figstyle.COLOR_INK,
            ha="center",
        )


def catchment_neff(pts: np.ndarray, sampled: np.ndarray, u: int) -> int:
    in_catch = ((pts - pts[u]) ** 2).sum(axis=1) <= RADIUS**2
    return int((sampled & in_catch).sum())


def draw_sampling_panel(ax, pts, sampled, exemplars, xfracs, targets, warn_sparse: bool = False) -> None:
    ax.scatter(pts[~sampled, 0], pts[~sampled, 1], s=7, c=COLOUR_UNSAMPLED, linewidths=0)
    ax.scatter(pts[sampled, 0], pts[sampled, 1], s=13, c=COLOUR_SAMPLED, linewidths=0)
    for u, colour, linestyle in exemplars:
        mark_exemplar(ax, pts[u], colour, linestyle)
    neffs = [catchment_neff(pts, sampled, u) for u, _, _ in exemplars]
    # Each n_eff reads "reached / target" so the draw is judged on the figure, not
    # only in the caption. The target is k for a catchment larger than k, and the
    # whole catchment (a census) when k exceeds its population.
    # warn_sparse (panel D): the sparse catchment's failing count is set in bold and
    # tagged "under-sampled" beside its circle, so the fixed rate's failure is
    # typographically salient rather than a number to be compared against panel C.
    label_below(
        ax,
        f"$n_{{\\mathrm{{eff}}}}$ = {neffs[0]} / {targets[0]}",
        f"$n_{{\\mathrm{{eff}}}}$ = {neffs[1]} / {targets[1]}",
        xfracs,
        note="reached / target",
        sparse_weight="bold" if warn_sparse else "normal",
    )
    if warn_sparse:
        u_sparse = exemplars[1][0]
        ax.text(
            pts[u_sparse][0],
            pts[u_sparse][1] + RADIUS + 0.14,
            "under-sampled",
            ha="center",
            va="bottom",
            fontsize=figstyle.SIZE_ANNOT,
            style="italic",
            fontweight="bold",
            color=COLOUR_SPARSE,
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )


def main() -> int:
    figstyle.apply()

    rng = np.random.default_rng(7)
    pts = make_network(rng)
    reach = counts_within(pts, RADIUS)
    q = np.minimum(1.0, K_TARGET / np.maximum(reach, 1.0))

    u_dense = int(np.argmax(reach))
    u_sparse = int(np.argmin(np.abs(pts[:, 0] - 3.2) + np.abs(pts[:, 1])))
    # Distinct dash patterns (dense dashed, sparse dash-dot) keep the two catchment
    # circles apart in greyscale, where teal and purple converge.
    exemplars = [(u_dense, COLOUR_DENSE, "--"), (u_sparse, COLOUR_SPARSE, (0, (5, 2, 1, 2)))]

    # Per-exemplar target: k for a catchment larger than k, else a census of the
    # whole catchment (min(k, reach)). These anchor the n_eff labels in C and D.
    targets = tuple(int(min(K_TARGET, reach[u])) for u, _, _ in exemplars)

    # Below-panel labels sit under the exemplar they annotate: convert each
    # exemplar's x-coordinate to an axes fraction so the labels stay aligned when
    # the coordinate limits change.
    span = XLIM[1] - XLIM[0]
    xfracs = (
        (pts[u_dense, 0] - XLIM[0]) / span,
        (pts[u_sparse, 0] - XLIM[0]) / span,
    )

    fig = plt.figure(figsize=(8.0, 5.3))
    grid = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # --- A: pilot ---
    style_panel(ax_a, "A", "Pilot: measure local reach")
    ax_a.scatter(pts[:, 0], pts[:, 1], s=9, c=COLOUR_POINT, alpha=0.85, linewidths=0)
    for u, colour, linestyle in exemplars:
        mark_exemplar(ax_a, pts[u], colour, linestyle)
    label_below(ax_a, f"$\\hat{{r}}$ = {int(reach[u_dense])}", f"$\\hat{{r}}$ = {int(reach[u_sparse])}", xfracs)

    # --- B: per-node q ---
    style_panel(ax_b, "B", "Assign: $q = \\min(1,\\, k/\\hat{r}\\,)$")
    ax_b.scatter(pts[:, 0], pts[:, 1], s=4 + 40 * q, c=COLOUR_POINT, alpha=0.85, linewidths=0)
    for u, colour, linestyle in exemplars:
        mark_exemplar(ax_b, pts[u], colour, linestyle)
    label_below(ax_b, f"$q$ = {q[u_dense]:.2f}", f"$q$ = {q[u_sparse]:.2f}", xfracs)
    # In-panel size key in the empty upper-left column (x < -1.4 holds almost no
    # core nodes): a header, then the small and large reference dots, each anchored
    # with a "low"/"high" q cue so the ramp direction reads without cross-referencing
    # the numeric q labels below. Those numeric values appear once below, so no
    # number is repeated here.
    ax_b.text(
        -1.80,
        1.50,
        "marker size $\\propto q$",
        fontsize=figstyle.SIZE_ANNOT,
        ha="left",
        va="center",
        color=figstyle.COLOR_INK,
    )
    ax_b.scatter([-1.62], [1.16], s=4 + 40 * 0.11, c=COLOUR_POINT, linewidths=0)
    ax_b.text(-1.46, 1.16, "low", fontsize=figstyle.SIZE_ANNOT, ha="left", va="center", color=figstyle.COLOR_INK)
    ax_b.scatter([-1.62], [0.78], s=4 + 40 * 1.0, c=COLOUR_POINT, linewidths=0)
    ax_b.text(-1.42, 0.78, "high", fontsize=figstyle.SIZE_ANNOT, ha="left", va="center", color=figstyle.COLOR_INK)

    # --- C: per-node draw (the method's outcome) ---
    style_panel(ax_c, "C", "Sample with per-node rates")
    node_sampled = np.random.default_rng(21).random(len(pts)) < q
    draw_sampling_panel(ax_c, pts, node_sampled, exemplars, xfracs, targets)

    # --- D: fixed-rate counterfactual, for illustration only ---
    p_fixed = float(K_TARGET / reach.mean())
    style_panel(ax_d, "D", f"Contrast: one fixed rate ($p$ = {p_fixed:.2f})")
    fixed_sampled = np.random.default_rng(3).random(len(pts)) < p_fixed
    draw_sampling_panel(ax_d, pts, fixed_sampled, exemplars, xfracs, targets, warn_sparse=True)

    # single shared legend
    # Three entries that match the marks in the panels: a filled grey network node
    # (A, B), a near-black filled sampled source (C, D), and a pale filled node with
    # a thin muted edge for "not sampled" (C, D). The pale fill matches the in-panel
    # unsampled dots; the thin edge only lets that pale mark hold its shape at legend
    # size and in greyscale. The teal/purple exemplar circles need no key here: the
    # caption names them and the coloured below-panel labels sit under their markers.
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=COLOUR_POINT, label="network node (A, B)"),
        Line2D(
            [], [], marker="o", linestyle="none", markersize=6.5, color=COLOUR_SAMPLED, label="sampled source (C, D)"
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=COLOUR_UNSAMPLED,
            markeredgecolor=figstyle.COLOR_MUTED,
            markeredgewidth=0.5,
            label="not sampled (C, D)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))

    for ext in ("pdf", "svg"):
        out = FIGURES_DIR / f"fig1_method_schematic.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
