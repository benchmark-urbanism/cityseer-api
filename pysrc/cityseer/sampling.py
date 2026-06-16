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
#   ε = 0.05  (normalised additive error tolerance; the single calibrated parameter)
#   δ = 0.1   (failure probability → 90% confidence)
#   s = 175m  (canonical sparse grid — a fixed reference, not fitted)
#
# The grid spacing s=175m is a fixed, network-agnostic reach model (r = π·d²/s²) motivated by
# observed street block lengths; it is not tuned per network. The accuracy tolerance ε is the
# single calibrated parameter: it is set empirically on real networks so that the sparsest
# validated network (a low-density US suburb, Cary, NC) preserves rank (ρ≥0.95) at all
# distances. ε=0.05 clears that bound with a small margin; the denser metropolitan networks
# (London, Madrid) oversample and clear it comfortably.
HOEFFDING_EPSILON: float = 0.05
HOEFFDING_DELTA: float = 0.1
GRID_SPACING: float = 175.0  # metres — canonical sparse grid (fixed reference, not fitted)


def compute_hoeffding_p(
    mean_reachability: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute sampling probability from the Hoeffding/Eppstein-Wang bound.

    Given a reachability estimate (how many nodes each source can reach), this function
    determines the minimum sampling probability needed to guarantee that results stay
    within ``epsilon`` of the exact computation with ``1 - delta`` confidence.

    k = log(2r / δ) / (2ε²)
    p = min(1, k / r)

    Parameters
    ----------
    mean_reachability : float
        Average number of nodes reachable within distance threshold.
    epsilon : float
        Normalised additive error tolerance. Default 0.05 (via HOEFFDING_EPSILON).
    delta : float
        Failure probability (1 - confidence). Default 0.1 (via HOEFFDING_DELTA), meaning
        90% confidence that the error stays within epsilon.

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

    Rather than requiring knowledge of the actual network, this function estimates
    reachability from the distance threshold alone using a regular grid model with
    spacing ``grid_spacing``. The estimated reachability is r = π * d² / s², which is
    then passed to ``compute_hoeffding_p`` to determine the sampling probability.

    This produces deterministic, network-independent sampling probabilities: the same
    distance always yields the same probability, enabling consistent comparison across
    different networks.

    Parameters
    ----------
    distance : float
        Distance threshold in metres.
    epsilon : float
        Normalised additive error tolerance. Default 0.05.
    delta : float
        Failure probability (1 - confidence). Default 0.1.
    grid_spacing : float
        Canonical inter-node spacing in metres. The default 175m is a fixed reference grid
        (not fitted), motivated by observed street block lengths. Denser networks have more
        reachable nodes than the model predicts, so the computed sampling probability is
        conservative (oversamples) — safe but slightly slower. The accuracy tolerance epsilon
        is the calibrated knob (see HOEFFDING_EPSILON).

    Returns
    -------
    float
        Required sampling probability in [0, 1].
    """
    if distance <= 0 or grid_spacing <= 0:
        return 1.0
    r = math.pi * distance**2 / grid_spacing**2
    return compute_hoeffding_p(r, epsilon=epsilon, delta=delta)
