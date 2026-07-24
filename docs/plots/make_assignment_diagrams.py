#!/usr/bin/env python
"""Schematics for "How distances to data points are measured".

Rendered in the cityseer site's own visual language (dark #19181b ground, Raleway,
red primal junction nodes, white streets, blue dual segment node, amber for the
data feature and its measured legs) so they sit natively beside the existing
network figures. Street rendering is identical across panels; node markers carry
the distinction (red dot = junction, blue diamond = segment midpoint).

Two figures:
  data_distance_schematic  - point: primal (two junctions) vs dual (one midpoint).
  data_polygon_schematic   - binding count: a point takes its nearest street; a
                             line or polygon takes every street within range.

Each writes SVG (docs) and PNG (preview) next to this script.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "images"))

BG = "#19181b"
STREET = "#e9e9e9"
PRIMAL = "#d32333"
DUAL = "#0091ea"
DATA = "#f5a623"
INK = "#f1f1f1"
MUTE = "#a8a8a8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Raleway", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 12,
    "text.color": INK,
    "svg.fonttype": "none",
    "savefig.dpi": 200,
})
MINUS = "−"


def base(ax, title):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=13, fontweight="medium", pad=10)


def street(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=STREET, lw=2.4, solid_capstyle="round", zorder=2)


def dot(ax, xy, color, s=120, marker="o", z=6):
    ax.scatter(*xy, s=s, marker=marker, c=color, zorder=z, edgecolors=BG, linewidths=1.4)


def ring(ax, xy):
    ax.scatter(*xy, s=300, facecolors="none", edgecolors=DATA, linewidths=2.2, zorder=5)


def datamark(ax, xy, label="P"):
    dot(ax, xy, DATA, s=150, marker="s", z=7)
    if label:
        ax.text(xy[0], xy[1] + 0.5, label, ha="center", va="bottom", color=DATA, fontweight="bold", fontsize=12)


def leg(ax, p_from, p_to, label=None):
    ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], color=DATA, lw=2.0, ls=(0, (2, 2)), zorder=5)
    ax.scatter(*p_to, s=26, c=STREET, zorder=6)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx + 0.28, my, label, color=DATA, fontsize=12, va="center", style="italic")


def dim(ax, x0, x1, y, label):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=DATA, lw=1.5, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y - 0.42, label, ha="center", va="center", color=DATA, fontsize=11.5, style="italic")


def figure_legend(fig, handles):
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                     bbox_to_anchor=(0.5, -0.02), fontsize=11)
    for t in leg.get_texts():
        t.set_color(MUTE)


# =================================================================== figure 1
def panel_primal(ax):
    base(ax, "Primal · point")
    y = 3.0
    n0, n1, F, P = (1.0, y), (7.0, y), (5.0, y), (5.0, 4.9)
    street(ax, n0, n1)
    dot(ax, n0, PRIMAL)
    dot(ax, n1, PRIMAL)
    leg(ax, P, F, "s")
    datamark(ax, P)
    dim(ax, n0[0], F[0], 1.7, "a")
    dim(ax, F[0], n1[0], 1.7, f"L {MINUS} a")


def panel_dual(ax):
    base(ax, "Dual · point")
    y = 3.0
    n0, n1, M, F, P = (1.0, y), (7.0, y), (4.0, y), (5.0, y), (5.0, 4.9)
    street(ax, n0, n1)
    dot(ax, n0, PRIMAL)
    dot(ax, n1, PRIMAL)
    dot(ax, M, DUAL, s=170, marker="D")
    ax.text(M[0], M[1] - 0.55, "M", ha="center", va="top", color=DUAL, fontsize=12, style="italic")
    leg(ax, P, F, "s")
    datamark(ax, P)
    dim(ax, M[0], F[0], 1.7, f"|L/2 {MINUS} a|")
    ax.text((M[0] + F[0]) / 2, 1.7 + 0.42, "±", ha="center", va="center", color=DATA, fontsize=15)


def make_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    panel_primal(axes[0])
    panel_dual(axes[1])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, wspace=0.06)
    figure_legend(fig, [
        Line2D([], [], color=STREET, lw=2.4, label="street"),
        Line2D([], [], marker="o", ls="none", ms=9, c=PRIMAL, mec=BG, label="junction node"),
        Line2D([], [], marker="D", ls="none", ms=9, c=DUAL, mec=BG, label="segment node (dual)"),
        Line2D([], [], marker="s", ls="none", ms=9, c=DATA, mec=BG, label="data point"),
        Line2D([], [], color=DATA, lw=2.0, ls=(0, (2, 2)), label="measured"),
    ])
    for ext in ("svg",):
        fig.savefig(os.path.join(OUT, f"data_distance_schematic.{ext}"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# =================================================================== figure 2
def u_streets(ax):
    """Three streets framing the feature; returns the four junction coordinates."""
    j = [(1.3, 1.3), (6.7, 1.3), (1.3, 4.7), (6.7, 4.7)]
    street(ax, j[0], j[1])   # bottom
    street(ax, j[0], j[2])   # left
    street(ax, j[1], j[3])   # right
    return j


def panel_point_count(ax):
    base(ax, "Point · nearest street only")
    j = u_streets(ax)
    P, F = (4.0, 2.4), (4.0, 1.3)
    leg(ax, P, F)                       # binds to the single nearest street
    datamark(ax, P)
    for c in (j[0], j[1]):              # that street's two junctions are reached
        ring(ax, c)
    for c in j:
        dot(ax, c, PRIMAL)


def panel_polygon_count(ax):
    base(ax, "Polygon · every frontage in range")
    j = u_streets(ax)
    park = [(2.7, 2.2), (5.3, 2.2), (5.3, 3.8), (2.7, 3.8)]
    ax.add_patch(Polygon(park, closed=True, facecolor=DATA, alpha=0.20, edgecolor=DATA, lw=2.0, zorder=3))
    ax.text(4.0, 3.6, "polygon", ha="center", va="center", color=DATA, fontweight="bold", fontsize=12)
    leg(ax, (4.0, 2.2), (4.0, 1.3))     # a setback leg to each street it faces
    leg(ax, (2.7, 2.9), (1.3, 2.9))
    leg(ax, (5.3, 2.9), (6.7, 2.9))
    for c in j:                         # every reached street's junctions
        ring(ax, c)
        dot(ax, c, PRIMAL)


def make_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    panel_point_count(axes[0])
    panel_polygon_count(axes[1])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, wspace=0.06)
    figure_legend(fig, [
        Line2D([], [], color=STREET, lw=2.4, label="street"),
        Line2D([], [], marker="o", ls="none", ms=9, c=PRIMAL, mec=BG, label="junction node"),
        Line2D([], [], marker="o", ls="none", ms=13, mfc="none", mec=DATA, mew=2.0, label="node reached"),
        Line2D([], [], marker="s", ls="none", ms=9, c=DATA, mec=BG, label="data feature"),
        Line2D([], [], color=DATA, lw=2.0, ls=(0, (2, 2)), label="setback to a street"),
    ])
    for ext in ("svg",):
        fig.savefig(os.path.join(OUT, f"data_polygon_schematic.{ext}"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    print("saved data_distance_schematic and data_polygon_schematic (svg)")
