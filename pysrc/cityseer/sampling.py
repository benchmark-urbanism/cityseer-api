"""Lightweight sampling probability functions.

This module has minimal dependencies (math, numpy) so it can be imported in
environments like QGIS without pulling in tqdm, rustalgos, etc.
"""

from __future__ import annotations

import math

import numpy as np

# === SAMPLING MODEL: Distance-based Hoeffding / Eppstein-Wang Bound ===
# Sampling probability derived from distance alone using a canonical grid network model.
# Reachability is estimated as r = π * d² / s² for grid spacing s (metres).
# The Hoeffding bound then gives:
#   k = log(2r / δ) / (2ε²)
#   p = min(1, k / r)
#
# Using a fixed grid spacing produces deterministic p values for any distance,
# enabling reach-agnostic comparison across networks.
#
# Default parameters:
#   ε = 0.06  (normalised additive error tolerance; unified for closeness and betweenness)
#   δ = 0.1   (failure probability → 90% confidence)
#   s = 175m  (canonical sparse street network inter-node spacing)
HOEFFDING_EPSILON: float = 0.06
HOEFFDING_DELTA: float = 0.1
GRID_SPACING: float = 175.0  # metres — canonical sparse street network inter-node spacing


def compute_hoeffding_p(
    mean_reachability: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute sampling probability from the Hoeffding/Eppstein-Wang bound.

    k = log(2r / δ) / (2ε²)
    p = min(1, k / r)

    Parameters
    ----------
    mean_reachability : float
        Average number of nodes reachable within distance threshold.
    epsilon : float
        Normalised additive error tolerance. Default 0.06 (via HOEFFDING_EPSILON).
    delta : float
        Failure probability (1 - confidence). Default 0.1 (via HOEFFDING_DELTA).

    Returns
    -------
    float
        Required sampling probability in [0, 1]. Returns 1.0 if reach is invalid.
    """
    if (
        not np.isfinite(mean_reachability)
        or not np.isfinite(epsilon)
        or not np.isfinite(delta)
        or mean_reachability <= 0
        or epsilon <= 0
        or delta <= 0
        or delta >= 1
    ):
        return 1.0

    k = math.log(2 * mean_reachability / delta) / (2 * epsilon**2)
    return min(1.0, k / mean_reachability)


def compute_distance_p(
    distance: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
    grid_spacing: float = GRID_SPACING,
) -> float:
    """
    Compute sampling probability from distance using a canonical grid network model.

    Estimates reachability as r = π * d² / s² for grid spacing s, then applies
    the Hoeffding/Eppstein-Wang bound. This produces deterministic p values for
    any distance, independent of the actual network, enabling cross-network comparison.

    Parameters
    ----------
    distance : float
        Distance threshold in metres.
    epsilon : float
        Normalised additive error tolerance. Default 0.06 (unified for closeness and betweenness).
    delta : float
        Failure probability (1 - confidence). Default 0.1.
    grid_spacing : float
        Canonical inter-node spacing in metres. Default 175m (sparse street network).

    Returns
    -------
    float
        Required sampling probability in [0, 1].
    """
    if distance <= 0 or grid_spacing <= 0:
        return 1.0
    r = math.pi * distance**2 / grid_spacing**2
    return compute_hoeffding_p(r, epsilon=epsilon, delta=delta)
