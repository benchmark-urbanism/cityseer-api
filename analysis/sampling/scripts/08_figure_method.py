#!/usr/bin/env python
"""
08_figure_method.py - Method schematic: per-node reach-based sampling on a worked example.

Four equal panels on a hypothetical network with a dense core and a sparse fringe:

  A) Pilot: count nodes within the radius of each node (local reach differs by area).
  B) Assign: per-node inclusion probability q = min(1, k(r)/r); dense low, sparse high.
  C) Contrast: a single fixed rate starves the sparse catchment.
  D) Per-node rates: every catchment receives approximately k effective samples;
     inverse-probability weighting (1/q per source) keeps estimates unbiased.

Outputs both PDF (paper) and SVG (docs site).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from utilities import FIGURES_DIR

COLOUR_DENSE = "#2166AC"
COLOUR_SPARSE = "#B2182B"
COLOUR_POINT = "#9a9a9a"
COLOUR_SAMPLED = "#2b2b2b"
COLOUR_UNSAMPLED = "#d4d4d4"

RADIUS = 0.75
K_TARGET = 18.0
XLIM = (-2.1, 4.15)
YLIM = (-2.1, 2.1)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


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


def style_panel(ax, title: str) -> None:
    ax.set_title(title, pad=7)
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)


def annotate_exemplar(ax, xy, text: str, colour: str, corner: str) -> None:
    """Place an exemplar label in a fixed corner with a thin leader line."""
    corners = {"tl": (-1.95, 1.62), "tr": (3.05, 1.62), "bl": (-1.95, -1.75), "br": (3.05, -1.75)}
    ax.annotate(
        text,
        xy=xy,
        xytext=corners[corner],
        fontsize=9.5,
        color=colour,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": colour, "alpha": 0.55, "lw": 0.9},
    )


def catchment_neff(pts: np.ndarray, sampled: np.ndarray, u: int) -> int:
    in_catch = ((pts - pts[u]) ** 2).sum(axis=1) <= RADIUS**2
    return int((sampled & in_catch).sum())


def draw_sampling_panel(ax, pts, sampled, exemplars, note: str) -> None:
    ax.scatter(pts[~sampled, 0], pts[~sampled, 1], s=7, c=COLOUR_UNSAMPLED, linewidths=0)
    ax.scatter(pts[sampled, 0], pts[sampled, 1], s=13, c=COLOUR_SAMPLED, linewidths=0)
    for u, colour, corner in exemplars:
        ax.add_patch(Circle(pts[u], RADIUS, fill=False, edgecolor=colour, linewidth=1.5, linestyle="--"))
        annotate_exemplar(ax, pts[u], f"$n_{{eff}}$ = {catchment_neff(pts, sampled, u)}", colour, corner)
    ax.text(0.03, 0.035, note, transform=ax.transAxes, fontsize=8.3, style="italic", color="#444444")


def main() -> int:
    rng = np.random.default_rng(7)
    pts = make_network(rng)
    reach = counts_within(pts, RADIUS)
    q = np.minimum(1.0, K_TARGET / np.maximum(reach, 1.0))

    u_dense = int(np.argmax(reach))
    u_sparse = int(np.argmin(np.abs(pts[:, 0] - 3.2) + np.abs(pts[:, 1])))
    exemplars = [(u_dense, COLOUR_DENSE, "tl"), (u_sparse, COLOUR_SPARSE, "tr")]

    fig = plt.figure(figsize=(16.4, 4.5))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 1, 1], wspace=0.08)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[0, 3])
    ax_d = fig.add_subplot(grid[0, 4])

    # --- A: pilot ---
    style_panel(ax_a, "A) Pilot: measure local reach")
    ax_a.scatter(pts[:, 0], pts[:, 1], s=9, c=COLOUR_POINT, alpha=0.75, linewidths=0)
    for u, colour, corner in exemplars:
        ax_a.add_patch(Circle(pts[u], RADIUS, fill=False, edgecolor=colour, linewidth=1.5, linestyle="--"))
        ax_a.scatter(*pts[u], s=38, c=colour, zorder=5)
        annotate_exemplar(ax_a, pts[u], f"$\\hat{{r}}$ = {int(reach[u])}", colour, corner)
    ax_a.text(
        0.03,
        0.035,
        "same radius, very different catchments",
        transform=ax_a.transAxes,
        fontsize=8.3,
        style="italic",
        color="#444444",
    )

    # --- B: per-node q, dedicated colorbar axis so the panel keeps its size ---
    style_panel(ax_b, "B) Assign: $q = \\min(1,\\, k/\\hat{r}\\,)$ per node")
    sc = ax_b.scatter(pts[:, 0], pts[:, 1], s=13, c=q, cmap="viridis", vmin=0, vmax=1, linewidths=0)
    for u, colour, corner in exemplars:
        annotate_exemplar(ax_b, pts[u], f"$q$ = {q[u]:.2f}", colour, corner)
    ax_b.text(
        0.03,
        0.035,
        "dense: low $q$;  sparse: high $q$",
        transform=ax_b.transAxes,
        fontsize=8.3,
        style="italic",
        color="#444444",
    )
    cbar = fig.colorbar(sc, cax=ax_cbar)
    cbar.set_label("inclusion probability $q$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    # equal-aspect panels shrink vertically; match the colorbar to panel B's drawn height
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    pos_c = ax_cbar.get_position()
    ax_cbar.set_position([pos_c.x0, pos_b.y0, pos_c.width, pos_b.height])

    # --- C: fixed-rate contrast ---
    p_fixed = float(K_TARGET / reach.mean())
    style_panel(ax_c, f"C) Fixed rate ($p$ = {p_fixed:.2f}): fringe starved")
    fixed_sampled = np.random.default_rng(3).random(len(pts)) < p_fixed
    draw_sampling_panel(ax_c, pts, fixed_sampled, exemplars, "one rate for all: dense fine, sparse starved")

    # --- D: per-node draw ---
    style_panel(ax_d, "D) Per-node rates: every catchment $\\approx k$")
    node_sampled = np.random.default_rng(21).random(len(pts)) < q
    draw_sampling_panel(ax_d, pts, node_sampled, exemplars, "contributions weighted $1/q$: unbiased")

    # single shared legend
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=COLOUR_SAMPLED, label="sampled source"),
        Line2D([], [], marker="o", linestyle="none", markersize=4, color=COLOUR_UNSAMPLED, label="not sampled"),
        Line2D([], [], linestyle="--", color="#666666", label="exemplar catchment (radius $d$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Per-node reach-based sampling (worked example, $k = {int(K_TARGET)}$)",
        fontsize=12,
        fontweight="bold",
        y=1.0,
    )
    for ext in ("pdf", "svg"):
        out = FIGURES_DIR / f"fig1_method_schematic.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())
