"""figstyle.py - Shared visual design system for the sampling paper figures.

One import gives every figure script the same fonts, type scale, spine and grid
treatment, colour semantics, panel-letter stamps, and scale bars, so the figure
set reads as one system. Import it, apply the rcParams once, and use the exported
constants and helpers in place of per-script literals.

Usage
-----
    import figstyle
    figstyle.apply()                    # or: plt.rcParams.update(figstyle.RCPARAMS)
    ax.plot(x, y, color=figstyle.COLOR_METHOD, marker="o")
    figstyle.panel_label(ax, "A")
    figstyle.scale_bar(ax, 5000, loc="lower left")

Colour semantics (a quantity keeps its colour in every figure)
    COLOR_CLOSENESS    blue    closeness / harmonic centrality
    COLOR_BETWEENNESS  red     betweenness centrality
    COLOR_CANONICAL    grey    canonical distance-only schedule (the baseline)
    COLOR_METHOD       orange  per-node reach-based method (the accent)

The four semantic colours are colourblind-safe under deuteranopia and protanopia
(worst adjacent CVD separation well above the perceptual floor) and separate by
lightness in greyscale. Where the same panel shows several networks, marker shape
is the primary identifier and colour is secondary. Sequential maps start near
white and darken monotonically, so they survive both colour-vision deficiency and
greyscale printing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# =============================================================================
# COLOUR SEMANTICS  (fixed mapping; never repaint the same quantity)
# =============================================================================

COLOR_CLOSENESS = "#2166AC"   # blue   - closeness / harmonic centrality
COLOR_BETWEENNESS = "#B2182B"  # red    - betweenness centrality
COLOR_CANONICAL = "#737373"   # grey   - canonical distance-only schedule (baseline)
COLOR_METHOD = "#D9730D"      # orange - per-node reach-based method (the accent)

# De-emphasised neutrals for supporting marks (never carry a semantic quantity).
COLOR_INK = "#2B2B2B"    # near-black: contours, scale bars, key marks, panel letters
COLOR_MUTED = "#BBBBBB"  # paired-point connectors, faint reference lines
COLOR_FAINT = "#E0E0E0"  # background context points (nodes outside the region of interest)
COLOR_WARN = "#C9A227"   # desaturated tan: semantic-free "attention" wash, tied to no metric

# British-spelling aliases so scripts that currently use COLOUR_* adopt the module
# by import alone. COLOUR_ADAPTIVE now resolves to the accent (previously blue),
# which is the intended change: the method is no longer the closeness colour.
COLOUR_CLOSENESS = COLOR_CLOSENESS
COLOUR_BETWEENNESS = COLOR_BETWEENNESS
COLOUR_CANONICAL = COLOR_CANONICAL
COLOUR_METHOD = COLOR_METHOD
COLOUR_ADAPTIVE = COLOR_METHOD

# -----------------------------------------------------------------------------
# Per-network categorical scale (identity, not magnitude).
# Marker SHAPE is the primary channel and is safe on its own; the hues are a
# colourblind-safe secondary channel. London and Madrid keep the entrenched
# blue/red; the suburbs use a teal-green and a purple distinct from every
# semantic colour above.
# -----------------------------------------------------------------------------
NETWORK_COLORS = {
    "gla": "#2166AC",       # London
    "madrid": "#B2182B",    # Madrid
    "cary": "#1B9E77",      # Cary
    "woodlands": "#7B3FA0",  # The Woodlands (held out)
}
NETWORK_MARKERS = {"gla": "o", "madrid": "s", "cary": "^", "woodlands": "D"}
NETWORK_LABELS = {
    "gla": "London",
    "madrid": "Madrid",
    "cary": "Cary",
    "woodlands": "Woodlands",
}
# Lightness steps for encoding several networks within one metric hue (fig8),
# ordered gla, madrid, cary, woodlands. 0.0 = full hue, 1.0 = white. The light end
# is compressed so the lightest step (the held-out network) keeps enough saturation
# to hold contrast against white and in greyscale; marker shape carries the primary
# network identity, so the hues need not spread all the way to pale.
NETWORK_TINT_STEPS = (0.0, 0.16, 0.30, 0.40)

# =============================================================================
# TYPE SCALE  (points; reused everywhere so sizes never drift between figures)
# =============================================================================

SIZE_PANEL = 13   # panel letters (bold)
SIZE_TITLE = 12   # axis / panel title
SIZE_LABEL = 11   # axis label, base font
SIZE_TICK = 10    # tick labels
SIZE_LEGEND = 10  # legend entries
SIZE_ANNOT = 9    # in-figure annotations, scale-bar labels

# =============================================================================
# SEQUENTIAL COLOURMAPS  (near-white -> dark; monotonic lightness)
# =============================================================================


def sequential_cmap(
    dark_hex: str,
    name: str = "figstyle_seq",
    light_hex: str = "#FFF7F0",
) -> LinearSegmentedColormap:
    """Return a near-white to ``dark_hex`` single-hue ramp for hexbins or heatmaps.

    A single-hue light-to-dark ramp is legible under colour-vision deficiency and
    in greyscale because it varies monotonically in lightness.
    """
    return LinearSegmentedColormap.from_list(name, [light_hex, dark_hex])


# Rank-shift / error hexbin map: white to deep red, matching the paper captions
# ("white-to-red scale"). Small values read pale; capped values read darkest.
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "figstyle_error",
    ["#FFF5F0", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"],
)

# Optional metric-tinted ramps for magnitude-only panels (e.g. decile heatmaps),
# so a sequential map can carry the metric's own hue and stay within the system.
CMAP_CLOSENESS = sequential_cmap(COLOR_CLOSENESS, "figstyle_closeness", "#F7FBFF")
CMAP_BETWEENNESS = sequential_cmap(COLOR_BETWEENNESS, "figstyle_betweenness", "#FFF5F0")

# =============================================================================
# RCPARAMS  (apply once per script)
# =============================================================================

RCPARAMS: dict = {
    # rendering
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    # fonts: prefer Helvetica/Arial for the Elsevier look, fall back to the
    # matplotlib-bundled DejaVu Sans so the scripts render on any machine.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": SIZE_LABEL,
    "axes.titlesize": SIZE_TITLE,
    "axes.labelsize": SIZE_LABEL,
    "xtick.labelsize": SIZE_TICK,
    "ytick.labelsize": SIZE_TICK,
    "legend.fontsize": SIZE_LEGEND,
    "figure.titlesize": SIZE_PANEL,
    # ink: near-black rather than pure black, dark-grey axis furniture
    "text.color": COLOR_INK,
    "axes.labelcolor": COLOR_INK,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    # spines: drop the top and right rules; keep left and bottom light
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.titlepad": 6.0,
    "axes.titleweight": "normal",
    # ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    # grid: recessive and standardised, but off by default. Figures opt in per
    # axis with ax.grid(True); the style below then applies uniformly.
    "axes.grid": False,
    "grid.color": "#B0B0B0",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.30,
    "grid.linestyle": "-",
    # legend: no frame, tight spacing
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.4,
    # lines and markers
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "lines.markeredgewidth": 0.6,
    # embed real fonts (Type42/TrueType) rather than Type3 outlines, which
    # Elsevier preflight rejects; keep SVG text as text for the docs site.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply() -> None:
    """Apply the shared rcParams. Call once near the top of a figure script."""
    plt.rcParams.update(RCPARAMS)


# =============================================================================
# HELPERS
# =============================================================================


def panel_label(
    ax,
    letter: str,
    x: float | None = None,
    y: float | None = None,
    *,
    inside: bool = False,
    halo: bool = False,
    size: float = SIZE_PANEL,
    weight: str = "bold",
    color: str = COLOR_INK,
):
    """Stamp a panel letter (``"A"``, ``"B"``, ...) in one consistent style.

    Default (``inside=False``) places a bold letter just above the top-left corner
    of the axes, clear of a centred title. For coordinate maps whose axes are
    turned off, pass ``inside=True`` to place the letter inside the top-left
    corner, and ``halo=True`` to keep it legible over dense marks. Coordinates are
    in axes fractions; override ``x``/``y`` for a one-off adjustment.

    Returns the created Text so callers can tweak it further.
    """
    if inside:
        if x is None:
            x = 0.02
        if y is None:
            y = 0.98
        ha, va = "left", "top"
    else:
        if x is None:
            x = -0.05
        if y is None:
            y = 1.02
        ha, va = "left", "bottom"
    effects = [patheffects.withStroke(linewidth=3.0, foreground="white")] if halo else None
    return ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        color=color,
        path_effects=effects,
    )


def scale_bar(
    ax,
    length_m: float,
    *,
    loc: str = "lower left",
    label: str | None = None,
    fontsize: float = SIZE_ANNOT,
    color: str = "black",
    halo: bool = True,
    pad_frac: float = 0.04,
    lw: float = 1.8,
):
    """Draw a scale bar on a coordinate map (used with the axes turned off).

    ``length_m`` is the bar length in metres. ``label`` defaults to ``"<n> km"``
    for whole-kilometre lengths, otherwise ``"<n> m"``. ``loc`` is one of the four
    corners ("lower left", "lower right", "upper left", "upper right"). The white
    halo (on by default) keeps the bar and its label legible over dark hexbins.
    Call after the axis limits are set.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    if "right" in loc:
        bx1 = x1 - dx * pad_frac
        bx0 = bx1 - length_m
    else:
        bx0 = x0 + dx * pad_frac
        bx1 = bx0 + length_m
    by = (y1 - dy * pad_frac) if "upper" in loc else (y0 + dy * pad_frac)
    effects = [patheffects.withStroke(linewidth=3.0, foreground="white")] if halo else None
    ax.plot(
        [bx0, bx1],
        [by, by],
        color=color,
        linewidth=lw,
        solid_capstyle="butt",
        path_effects=effects,
        zorder=5,
    )
    if label is None:
        label = f"{length_m / 1000:g} km" if length_m % 1000 == 0 else f"{length_m:g} m"
    ax.text(
        (bx0 + bx1) / 2,
        by + dy * 0.015,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        path_effects=effects,
        zorder=5,
    )


def tint(color: str, factor: float) -> tuple[float, float, float]:
    """Blend ``color`` toward white by ``factor`` in [0, 1] (0 = full colour, 1 = white).

    Used to build within-hue lightness steps, e.g. one metric hue shared across
    networks that are told apart by marker shape (see ``NETWORK_TINT_STEPS``).
    """
    r, g, b = to_rgb(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def network_color(key: str) -> str:
    """Categorical colour for a network key (gla, madrid, cary, woodlands)."""
    return NETWORK_COLORS[key]


def network_marker(key: str) -> str:
    """Marker shape for a network key (the primary, colourblind-safe identifier)."""
    return NETWORK_MARKERS[key]
