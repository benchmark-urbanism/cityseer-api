#!/usr/bin/env python
"""
09_figure_adaptive_comparison.py - Per-node method figures from the adaptive CSVs.

fig12_baseline_vs_adaptive: Panels A (closeness) and B (betweenness) compare Spearman
rho at 20 km per network, canonical schedule vs the per-node method, as paired points
on a truncated axis (bars would exaggerate differences against an arbitrary baseline).
Closeness entries the work test routed to exact computation are drawn as open markers.
Panel C shows per-reach-quartile rho at 20 km betweenness on the held-out network.

fig13_adaptive_accuracy: rho vs distance under the per-node method; calibration
networks in panels A/B, the held-out network in panel C (extended to 50 km via the
frontier rebuild). Open markers denote distances computed exactly (work test);
filled markers are sampled. The rule holds in every panel.

Colours follow the shared design system (figstyle): closeness = blue, betweenness =
red, canonical schedule = grey, per-node method = orange. The canonical schedule and
the per-node method also differ by marker shape (square vs circle), so the pairing
survives greyscale. Panel C of both figures denotes metrics (closeness blue,
betweenness red), not networks; fig13's panel C also separates the metrics by line
style and labels each line directly. Every panel carries a descriptive in-artwork
title, and the open-marker rule (open = computed exactly) appears in the legends, so
the figures read without the LaTeX caption.

Reads output/{network}_validation[_adaptive].csv; includes whichever networks have
adaptive results, so it can be regenerated as runs complete.
"""

import matplotlib

matplotlib.use("Agg")
import figstyle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from utilities import FIGURES_DIR, OUTPUT_DIR

TARGET = 0.95

# White halo matching figstyle's idiom, so an open "exact" marker reads cleanly over
# a gridline rather than with the line running through its ring.
HALO_WHITE = [patheffects.withStroke(linewidth=2.5, foreground="white")]

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
        "exact_c": _closeness_ran_exact(row.iloc[0]),
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


def _closeness_ran_exact(row: pd.Series) -> bool:
    """True when the closeness work test selected exact computation for this row.

    Newer CSVs record the decision in ``closeness_mode``; older CSVs lack it, and the
    decision is recovered from the recorded errors (zero error in every reach quartile).
    """
    if "closeness_mode" in row.index and isinstance(row["closeness_mode"], str):
        return row["closeness_mode"] == "exact"
    cols = [f"h_mae_q{i}" for i in (1, 2, 3, 4)]
    if not all(c in row.index for c in cols):
        return False
    vals = [row[c] for c in cols]
    return all(np.isfinite(v) and v == 0.0 for v in vals)


def _betweenness_ran_exact(row: pd.Series) -> bool:
    """True when the betweenness work test selected exact computation for this row."""
    if "betweenness_mode" in row.index and isinstance(row["betweenness_mode"], str):
        return row["betweenness_mode"] == "exact"
    return row["mode"] == "exact"


def _target_line(ax, x_label: float, ha: str = "left", show_label: bool = True) -> None:
    """Neutral dashed reference at the rho = 0.95 target, with an optional matching label.

    The dashed line is the same reference in every panel, so a shared-scale row labels
    it once (``show_label=True`` on the leftmost panel only) rather than repeating the
    tag at a different x-position in each panel.
    """
    ax.axhline(TARGET, color=figstyle.COLOR_INK, linestyle="--", linewidth=1.0, alpha=0.55, zorder=1)
    if show_label:
        ax.text(
            x_label,
            TARGET + 0.002,
            f"$\\rho$={TARGET}",
            fontsize=figstyle.SIZE_ANNOT,
            color=figstyle.COLOR_INK,
            ha=ha,
        )


def generate_fig13_adaptive_accuracy() -> None:
    """rho vs distance: calibration networks (A, B); held-out network to 50 km (C).

    One marker rule figure-wide: open = exact (work test), filled = sampled; panel B's
    legend states it. Panel C shows the held-out network's main 20 km-buffered build
    (circles/squares, to 20 km) and the frontier rebuild with a 50 km buffer (diamonds
    at 30/40/50 km; a different graph, so points are not joined to the main lines).
    Panel C colours denote metrics (closeness blue, betweenness red), not networks;
    the metrics are further separated by line style (solid vs dashed) and labelled
    directly at the line ends, so the panel reads without the A/B network key.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    # (axis, column, panel letter, metric key for the exact test)
    metric_specs = [
        (axes[0], "rho_closeness", "A", "c"),
        (axes[1], "rho_betweenness", "B", "b"),
    ]
    plotted = False
    for key in ("gla", "madrid", "cary"):  # calibration networks; held-out goes to panel C
        path = OUTPUT_DIR / f"{key}_validation_adaptive.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("distance")
        plotted = True
        colour = figstyle.NETWORK_COLORS[key]
        marker = figstyle.NETWORK_MARKERS[key]
        label = figstyle.NETWORK_LABELS[key]
        for ax, col, _letter, metric in metric_specs:
            x = df["distance"].to_numpy(float) / 1000
            y = df[col].to_numpy(float)
            if metric == "c":
                exact = df.apply(lambda r: r["mode"] == "exact" or _closeness_ran_exact(r), axis=1).to_numpy(bool)
            else:
                exact = df.apply(_betweenness_ran_exact, axis=1).to_numpy(bool)
            ax.plot(x, y, "-", color=colour, linewidth=1.6, alpha=0.85, zorder=2)
            ax.plot(x[~exact], y[~exact], marker, color=colour, markersize=6, linestyle="none", label=label, zorder=3)
            ax.plot(
                x[exact],
                y[exact],
                marker,
                markerfacecolor="white",
                markeredgecolor=colour,
                markersize=6,
                linestyle="none",
                zorder=3,
            )
    if not plotted:
        plt.close()
        print("  No adaptive validation CSVs found; skipping fig13.")
        return
    # Descriptive two-line titles carry the panel identity in the artwork (the bold
    # letter stamp keeps the cross-reference); two lines keep the centred titles
    # narrower than the panels.
    panel_titles = {
        "A": "Closeness,\ncalibration networks",
        "B": "Betweenness,\ncalibration networks",
    }
    for ax, _col, letter, _metric in metric_specs:
        # Shared scale (sharey): label the target and the y-axis once, on the leftmost
        # panel; B repeats neither.
        _target_line(ax, 8.0, show_label=ax is axes[0])
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel("Analysis distance (km)")
        ax.set_ylabel("Spearman $\\rho$" if ax is axes[0] else "")
        ax.set_ylim(0.945, 1.0065)
        ax.set_title(panel_titles[letter], fontsize=figstyle.SIZE_LEGEND)
        ax.grid(True)
        figstyle.panel_label(ax, letter)
    axes[0].legend(loc="lower left")
    # The open-marker rule, stated once in the artwork. Panel B's mid-left region is
    # the one space clear of every line, marker, legend, and the target line; the
    # rule applies to all three panels (C repeats the open markers on its diamonds).
    axes[1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markeredgecolor=figstyle.COLOR_INK,
                markerfacecolor="white",
                markersize=7,
                label="computed exactly\n($\\rho$ = 1)",
            ),
        ],
        loc="center left",
        fontsize=figstyle.SIZE_ANNOT,
    )

    # --- Panel C: held-out network on the 50 km buffer, full 1-50 km range ---
    # One build (woodlands50, 11_frontier_woodlands.py --distances 1000..50000), so the
    # panel is a single continuous series per metric. Marker rule matches A/B: open =
    # exact, filled = sampled. Colours denote metrics (blue closeness, red betweenness).
    ax = axes[2]
    frontier = OUTPUT_DIR / "woodlands_frontier.csv"
    if frontier.exists():
        dff = pd.read_csv(frontier).sort_values("distance")
        xf = dff["distance"].to_numpy(float) / 1000
        # The frontier CSV has no mode column; an exact cell reproduces the ground
        # truth on every seed, so exact mode is identified as rho = 1 with zero spread.
        exact_c = (dff["rho_closeness"].to_numpy(float) >= 1.0 - 1e-12) & (
            dff["rho_closeness_std"].to_numpy(float) == 0.0
        )
        exact_b = (dff["rho_betweenness"].to_numpy(float) >= 1.0 - 1e-12) & (
            dff["rho_betweenness_std"].to_numpy(float) == 0.0
        )
        # Closeness circles are slightly larger and sit beneath the betweenness squares
        # so both markers stay visible where the two metrics coincide at rho = 1. A/B
        # use colour for networks, so C separates its metric lines by line style as
        # well (closeness solid, betweenness dashed) and labels each directly, keeping
        # the panel legible in greyscale and without the A/B key.
        for col, colour, marker, ls, exact, msize, zord in [
            ("rho_closeness", figstyle.COLOR_CLOSENESS, "o", "-", exact_c, 7, 3),
            ("rho_betweenness", figstyle.COLOR_BETWEENNESS, "s", "--", exact_b, 5, 4),
        ]:
            y = dff[col].to_numpy(float)
            ax.plot(xf, y, linestyle=ls, color=colour, linewidth=1.6, alpha=0.85, zorder=2)
            ax.plot(xf[~exact], y[~exact], marker, color=colour, markersize=msize, linestyle="none", zorder=zord)
            ax.plot(
                xf[exact],
                y[exact],
                marker,
                markerfacecolor="white",
                markeredgecolor=colour,
                markersize=msize,
                linestyle="none",
                zorder=zord,
            )
        # Direct labels on the two lines: closeness sits just above its flat rho = 1
        # line (the ylim headroom above 1.004 exists for it); betweenness sits above
        # its 30-50 km plateau, clear of the target line below.
        ax.text(
            6.8,
            1.002,
            "closeness",
            fontsize=figstyle.SIZE_ANNOT,
            color=figstyle.COLOR_CLOSENESS,
            ha="center",
            va="bottom",
        )
        ax.text(
            38,
            0.9575,
            "betweenness",
            fontsize=figstyle.SIZE_ANNOT,
            color=figstyle.COLOR_BETWEENNESS,
            ha="center",
            va="bottom",
        )
        # The dashed target is labelled once on panel A (shared scale); C draws the
        # line without repeating the tag.
        _target_line(ax, 3.0, show_label=False)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 30, 40, 50])
        # Blank the 30 and 40 labels: on the log axis they collide with 20/50 and read
        # as "2030"/"4050". The tick marks stay so each point has a tick beneath it.
        ax.set_xticklabels(["1", "2", "5", "10", "20", "", "", "50"])
        ax.set_xlabel("Analysis distance (km)")
        ax.set_ylim(0.945, 1.0065)
        ax.set_title("Held-out (The Woodlands),\nto 50 km", fontsize=figstyle.SIZE_LEGEND)
        ax.grid(True)
        figstyle.panel_label(ax, "C")
    else:
        figstyle.panel_label(ax, "C")
    plt.tight_layout()
    out = FIGURES_DIR / "fig13_adaptive_accuracy.pdf"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()


def main() -> int:
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    # --- Panels A/B: 20km rho, baseline vs adaptive, one metric per panel ---
    any_exact = False
    for panel, (metric, exact_key, letter) in enumerate([("rho_c", "exact_c", "A"), ("rho_b", None, "B")]):
        ax = axes[panel]
        labels, base_vals, adap_vals, exact_flags = [], [], [], []
        for key, label in NETWORKS:
            base = canonical_20km(key)
            adap = adaptive_20km(key)
            if base is None or adap is None:
                continue
            labels.append(label)
            base_vals.append(base[metric])
            adap_vals.append(adap[metric])
            exact_flags.append(bool(exact_key) and adap.get(exact_key, False))
        any_exact = any_exact or any(exact_flags)
        # Paired points on the truncated axis: position encodes rho honestly where
        # bar length against an arbitrary axis floor would not. Canonical = grey,
        # per-node method = orange; the connector is the muted neutral. Marker shape
        # doubles the colour coding (canonical = square, per-node = circle) so the
        # pairing survives greyscale print.
        x = np.arange(len(labels))
        off = 0.14
        for xi, bv, av, is_exact in zip(x, base_vals, adap_vals, exact_flags, strict=True):
            ax.plot([xi - off, xi + off], [bv, av], "-", color=figstyle.COLOR_MUTED, linewidth=1.0, alpha=0.9, zorder=2)
            ax.plot(xi - off, bv, "s", color=figstyle.COLOR_CANONICAL, markersize=6.5, zorder=3)
            if is_exact:
                # White halo so the rho=1.00 gridline reads behind the ring, not across it.
                ax.plot(
                    xi + off,
                    av,
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=figstyle.COLOR_METHOD,
                    markersize=7,
                    zorder=3,
                    path_effects=HALO_WHITE,
                )
            else:
                ax.plot(xi + off, av, "o", color=figstyle.COLOR_METHOD, markersize=7, zorder=3)
        # Shared A/B scale: label the target once, on panel A only.
        _target_line(ax, 1.0, ha="center", show_label=(panel == 0))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0.92, 1.012)
        # A carries the shared A/B axis title; B drops the duplicate title and its
        # duplicate y-tick labels (same quantity, same scale). Panel C keeps its
        # distinct within-quartile label and its own scale below.
        ax.set_ylabel("Spearman $\\rho$ (20 km)" if panel == 0 else "")
        if panel != 0:
            ax.tick_params(labelleft=False)
        # Descriptive title alongside the bold letter stamp, matching fig13. Panel B
        # has no y-tick labels, so its centred title starts further left; a small
        # rightward nudge keeps it clear of the corner letter.
        ax.set_title(
            "Closeness, by network" if panel == 0 else "Betweenness, by network",
            fontsize=figstyle.SIZE_LEGEND,
            x=0.5 if panel == 0 else 0.56,
        )
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)  # gridlines behind the markers, not across their rings
        figstyle.panel_label(ax, letter)

    # --- Panel C: quartile uniformity on the held-out network (betweenness) ---
    ax = axes[2]
    base_q = quartile_series(OUTPUT_DIR / "woodlands_validation.csv", "b_")
    adap_q = quartile_series(OUTPUT_DIR / "woodlands_validation_adaptive.csv", "b_")
    if base_q and adap_q:
        # Pair each quartile with a short vertical connector between the canonical and
        # per-node markers, reusing the A/B pairing idiom so all three panels read the
        # same way. The x-axis is ordinal, so no line traces the sequence across
        # quartiles. No target line here: the rho = 0.95 target applies to the
        # network-wide ranking, not to within-quartile correlations.
        qx = [1, 2, 3, 4]
        for xi, bv, av in zip(qx, base_q, adap_q, strict=True):
            ax.plot([xi, xi], [bv, av], "-", color=figstyle.COLOR_MUTED, linewidth=1.0, alpha=0.9, zorder=2)
        ax.plot(qx, base_q, "s", color=figstyle.COLOR_CANONICAL, markersize=6.5, linestyle="none", zorder=3)
        ax.plot(qx, adap_q, "o", color=figstyle.COLOR_METHOD, markersize=7, linestyle="none", zorder=3)
        ax.set_xticks(qx)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlim(0.5, 4.5)
        ax.set_xlabel("Reach quartile (low $\\rightarrow$ high)")
        # C's own scale: within-quartile rho spans ~0.930--0.985, a different quantity
        # from A/B's network-wide rho, so its y-limits are decoupled to enlarge the
        # paired gap without exaggerating it.
        ax.set_ylim(0.925, 0.99)
        # Plain-words axis label: within-quartile rho compares streets whose reach is
        # similar, which is what the quantity means for a lay reader.
        ax.set_ylabel("$\\rho$ among streets\nof similar reach")
        # Short title: the y- and x-labels carry "reach", so the title only names the
        # metric and network; longer titles collide with the corner letter stamp.
        ax.set_title("Betweenness, held-out", fontsize=figstyle.SIZE_LEGEND)
        # Mechanism note in the empty upper-left region, clear of the Q3/Q4 markers.
        ax.text(
            0.04,
            0.97,
            "both designs dip\nwhere near-zero ties\nreorder easily",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=figstyle.SIZE_ANNOT,
            color=figstyle.COLOR_INK,
        )
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        figstyle.panel_label(ax, "C")
    else:
        figstyle.panel_label(ax, "C")

    # One shared legend for the whole figure (all three panels use the same
    # canonical/method encoding), placed as a horizontal row below the panels.
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=figstyle.COLOR_CANONICAL,
            markersize=6.5,
            label="canonical schedule",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=figstyle.COLOR_METHOD,
            markersize=7,
            label="per-node method",
        ),
    ]
    if any_exact:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=figstyle.COLOR_METHOD,
                markersize=7,
                label="per-node method (exact)",
            )
        )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.02))
    for ext in ("pdf", "svg"):
        out = FIGURES_DIR / f"fig12_baseline_vs_adaptive.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close()

    generate_fig13_adaptive_accuracy()
    return 0


if __name__ == "__main__":
    exit(main())
