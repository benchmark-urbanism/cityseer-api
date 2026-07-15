"""Dark-theme sampling figures for the docs site.

Regenerates the two illustrations on the /metrics/sampling page in the site's
matplotlib style (dark background, light ink). Pure numpy + matplotlib; run with
the project venv from this directory or the repo root:

    python docs/plots/sampling_plots.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PLOT_RC_PATH = pathlib.Path(__file__).parent / "matplotlibrc"
plt.style.use(PLOT_RC_PATH)

IMAGES_PATH = pathlib.Path(__file__).parent.parent / "public/images"

# site palette
COLOUR_BG = "#19181b"
COLOUR_INK = "#f1f1f1"
COLOUR_NODE = "#8f8f8f"  # network node
COLOUR_FAINT = "#4a4a4a"  # not sampled
COLOUR_SAMPLED = "#f1f1f1"  # sampled source
COLOUR_DENSE = "#64c1ff"  # dense-core exemplar
COLOUR_SPARSE = "#d32f2f"  # sparse-fringe exemplar

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


def style_panel(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, color=COLOUR_INK, pad=6)
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOUR_FAINT)
        spine.set_linewidth(0.8)


def mark_exemplar(ax, xy, colour: str) -> None:
    """Dashed catchment circle plus a centre dot for the two exemplar nodes."""
    ax.add_patch(Circle(xy, RADIUS, fill=False, edgecolor=colour, linewidth=1.4, linestyle="--"))
    ax.scatter(*xy, s=42, c=colour, zorder=6, edgecolors=COLOUR_BG, linewidths=0.8)


def label_below(ax, dense_text: str, sparse_text: str, xfracs: tuple[float, float]) -> None:
    """One value beneath each exemplar, set in the exemplar's own colour."""
    for xf, text, colour in zip(xfracs, (dense_text, sparse_text), (COLOUR_DENSE, COLOUR_SPARSE), strict=True):
        ax.text(xf, -0.09, text, transform=ax.transAxes, fontsize=8, color=colour, ha="center")


def catchment_count(pts: np.ndarray, sampled: np.ndarray, u: int) -> int:
    in_catch = ((pts - pts[u]) ** 2).sum(axis=1) <= RADIUS**2
    return int((sampled & in_catch).sum())


def draw_sampling_panel(ax, pts, sampled, exemplars, xfracs, targets, warn_sparse: bool = False) -> None:
    ax.scatter(pts[~sampled, 0], pts[~sampled, 1], s=7, c=COLOUR_FAINT, linewidths=0)
    ax.scatter(pts[sampled, 0], pts[sampled, 1], s=13, c=COLOUR_SAMPLED, linewidths=0)
    for u, colour in exemplars:
        mark_exemplar(ax, pts[u], colour)
    counts = [catchment_count(pts, sampled, u) for u, _ in exemplars]
    label_below(
        ax,
        f"{counts[0]} sources (needs {targets[0]})",
        f"{counts[1]} sources (needs {targets[1]})",
        xfracs,
    )
    if warn_sparse:
        u_sparse = exemplars[1][0]
        ax.text(
            pts[u_sparse][0],
            pts[u_sparse][1] + RADIUS + 0.14,
            "under-sampled",
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
            color=COLOUR_SPARSE,
        )


def method_schematic() -> None:
    rng = np.random.default_rng(7)
    pts = make_network(rng)
    reach = counts_within(pts, RADIUS)
    q = np.minimum(1.0, K_TARGET / np.maximum(reach, 1.0))

    u_dense = int(np.argmax(reach))
    u_sparse = int(np.argmin(np.abs(pts[:, 0] - 3.2) + np.abs(pts[:, 1])))
    exemplars = [(u_dense, COLOUR_DENSE), (u_sparse, COLOUR_SPARSE)]
    targets = tuple(int(min(K_TARGET, reach[u])) for u, _ in exemplars)

    span = XLIM[1] - XLIM[0]
    xfracs = (
        (pts[u_dense, 0] - XLIM[0]) / span,
        (pts[u_sparse, 0] - XLIM[0]) / span,
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.08, hspace=0.42)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # A: pilot measures reach
    style_panel(ax_a, "A. Measure each node's reach")
    ax_a.scatter(pts[:, 0], pts[:, 1], s=9, c=COLOUR_NODE, linewidths=0)
    for u, colour in exemplars:
        mark_exemplar(ax_a, pts[u], colour)
    label_below(ax_a, f"reach = {int(reach[u_dense])}", f"reach = {int(reach[u_sparse])}", xfracs)

    # B: per-node probability
    style_panel(ax_b, "B. Assign probabilities (dot size = probability)")
    ax_b.scatter(pts[:, 0], pts[:, 1], s=4 + 40 * q, c=COLOUR_NODE, linewidths=0)
    for u, colour in exemplars:
        mark_exemplar(ax_b, pts[u], colour)
    label_below(ax_b, f"p = {q[u_dense]:.2f}", f"p = {q[u_sparse]:.2f}", xfracs)

    # C: draw under per-node probabilities
    style_panel(ax_c, "C. Sample with per-node probabilities")
    node_sampled = np.random.default_rng(21).random(len(pts)) < q
    draw_sampling_panel(ax_c, pts, node_sampled, exemplars, xfracs, targets)

    # D: fixed-rate counterfactual
    p_fixed = float(K_TARGET / reach.mean())
    style_panel(ax_d, f"D. A single fixed rate (p = {p_fixed:.2f}) for comparison")
    fixed_sampled = np.random.default_rng(3).random(len(pts)) < p_fixed
    draw_sampling_panel(ax_d, pts, fixed_sampled, exemplars, xfracs, targets, warn_sparse=True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6, color=COLOUR_NODE, label="network node"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6, color=COLOUR_SAMPLED, label="sampled source"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6, color=COLOUR_FAINT, label="not sampled"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8,
    )

    out = IMAGES_PATH / "sampling_method_schematic.svg"
    fig.savefig(out, bbox_inches="tight", facecolor=COLOUR_BG)
    print(f"Saved: {out}")
    plt.close(fig)


def work_test() -> None:
    gamma = 0.75
    exact_cost = 100.0
    threshold = gamma * exact_cost
    panels = [
        # (title, predicted sampled cost, decision)
        ("Dense network, 20 km threshold", 55.0, "cheaper: this distance is sampled"),
        ("Sparse suburb, short threshold", 90.0, "not cheaper: runs exactly"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.12, wspace=0.08)
    for ax, (title, sampled_cost, decision) in zip(axes, panels, strict=True):
        ax.bar([0, 1], [exact_cost, sampled_cost], width=0.55, color=[COLOUR_NODE, COLOUR_DENSE])
        ax.axhline(threshold, color=COLOUR_INK, linestyle="--", linewidth=1.0, zorder=3)
        ax.text(
            0.5,
            sampled_cost + 6 if sampled_cost < threshold else 106,
            decision,
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
            color=COLOUR_DENSE,
            transform=ax.transData,
        )
        ax.set_title(title, fontsize=9, color=COLOUR_INK, pad=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["exact\ncomputation", "predicted cost\nof sampling"], fontsize=8)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].text(
        -0.62,
        threshold + 3,
        "sampling must beat this line",
        ha="left",
        va="bottom",
        fontsize=8,
        color=COLOUR_INK,
    )
    axes[0].set_ylabel("Predicted work (exact = 100)")

    out = IMAGES_PATH / "sampling_work_test.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=COLOUR_BG)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    method_schematic()
    work_test()
