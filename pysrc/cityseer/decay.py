"""Functions for generating decay expression strings.

Decay expressions control how feature importance decreases with distance from an analysis point. They use a
single variable `p` that ranges from 0 at the analysis point (source) to 1 at the distance threshold (cutoff),
where `p = network_distance / max_distance`. The functions below produce expression strings that can be passed
directly to the `decay_fn` parameter of [`betweenness_demand`](/metrics/networks#betweenness_demand),
[`compute_stats`](/metrics/layers#compute_stats),
[`compute_accessibilities`](/metrics/layers#compute_accessibilities), and other analysis functions.

```python
from cityseer import decay
from cityseer.metrics import layers

nodes_gdf, data_gdf = layers.compute_stats(
    ...,
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)
```

Available presets:

```text
exponential (default)       gaussian(peak, cutoff)        linear
╷                           ╷       ╭─╮                   ╷╲
│╲                          │      ╱   ╲                  │ ╲
│ ╲                         │     ╱     ╲                 │  ╲
│  ╲                        │    ╱       ╲                │   ╲
│   ╲                       │   ╱         ╲               │    ╲
│    ╰─╮                    │  ╱           ╲              │     ╲
│      ╰───╮                │ ╱             ╰╮            │      ╲
│          ╰────────        │╱               ╰───         │       ╲
╵───────────────────        ╵───────────────────          ╵────────╳
p=0                p=1      p=0                p=1        p=0      p=1

logistic(midpoint, cutoff)        flat (no decay)
╷──────╮                          ╷───────────────
│      ╲                          │
│       ╲                         │
│        │                        │
│        ╲                        │
│         ╲                       │
│          ╰╮                     │
│           ╰────────             │
╵───────────────────              ╵───────────────
p=0                p=1            p=0            p=1
```
"""

from __future__ import annotations


def exponential(steepness: float = 4.0) -> str:
    """Exponential decay: weight = exp(-steepness * p).

    At p=1 (the cutoff), weight = exp(-steepness). The default steepness of 4
    gives ~1.8% weight at the cutoff boundary, matching cityseer's historical
    default.

    Parameters
    ----------
    steepness: float
        Controls how quickly weight decays. Higher values produce steeper decay.
        Common values: 2 (~13.5% at cutoff), 4 (~1.8%), 6 (~0.25%).
    """
    return f"exp(-{steepness} * p)"


def linear() -> str:
    """Linear decay from 1 at the source to 0 at the cutoff (output is clamped to [0, 1])."""
    return "1 - p"


def flat() -> str:
    """No decay: constant weight of 1 everywhere within the cutoff."""
    return "1"


def gaussian(
    peak: float,
    cutoff: float,
    std: float | None = None,
) -> str:
    """Gaussian bell curve peaking at a specified location.

    Parameters
    ----------
    peak: float
        The point at which the weight is highest, in the same units as
        ``cutoff`` (e.g. metres or minutes).
    cutoff: float
        The cutoff threshold in the same units as ``peak`` (must match the
        ``distances`` or ``minutes`` parameter).
    std: float
        Standard deviation controlling the width of the bell curve, in the same
        units as ``peak``/``cutoff``. If not provided, defaults to ``peak / 2``.
    """
    if std is None:
        std = peak / 2
    peak_p = peak / cutoff
    std_p = std / cutoff
    # parenthesise the squared term so the leading minus negates it (avoids unary-minus/^ ambiguity)
    return f"exp(-((p - {peak_p:.6g})^2) / (2 * {std_p:.6g}^2))"


def logistic(
    midpoint: float,
    cutoff: float,
    rate: float = 0.05,
) -> str:
    """Logistic (sigmoid) decay centred at a specified location.

    Weight transitions from ~1 (near source) to ~0 (beyond midpoint). The
    transition is centred at ``midpoint``.

    Parameters
    ----------
    midpoint: float
        The point at which weight = 0.5, in the same units as ``cutoff``.
    cutoff: float
        The cutoff threshold in the same units as ``midpoint`` (must match the
        ``distances`` or ``minutes`` parameter).
    rate: float
        Steepness of the transition. Higher values produce a sharper step.
        The rate is specified in per-unit terms; internally it is scaled to
        normalised ``p`` coordinates.
    """
    midpoint_p = midpoint / cutoff
    rate_p = rate * cutoff
    return f"1 / (1 + exp({rate_p:.6g} * (p - {midpoint_p:.6g})))"
