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


# === PER-NODE (ADAPTIVE) SAMPLING ===
# The runtime `sample=True` path measures each node's reach and assigns a per-node source
# inclusion probability q = min(1, k(r)/r), so every catchment receives approximately the
# Hoeffding-required number of effective samples regardless of local density. Horvitz-Thompson
# weighting (1/q per source) keeps estimates unbiased. The distance-only canonical schedule
# above is retained as a reference model; it is no longer used by the runtime.

# Euclidean discs contain more nodes than network catchments of the same radius (network
# distance exceeds straight-line distance). Reach estimated from a Euclidean count is therefore
# deflated before use. Measured on four validation networks (London, Madrid, Cary NC, The
# Woodlands TX) at 10-20km, the per-node ratio of Euclidean count to network reach has median
# 1.3-1.8 and 99th percentile <= 3.4; a deflation of 2.5 covers all but the extreme tail, and
# over-deflation only oversamples (a cost, never a bias or accuracy risk).
EUCLIDEAN_REACH_DEFLATION: float = 2.5

# Floor on per-node inclusion probabilities: bounds the Horvitz-Thompson weights (1/q <= 100)
# so that a badly underestimated reach cannot produce high-variance contributions.
MIN_NODE_PROBABILITY: float = 0.01


def estimate_euclidean_reach(
    node_xs: np.ndarray | list[float],
    node_ys: np.ndarray | list[float],
    distance: float,
    deflation: float = EUCLIDEAN_REACH_DEFLATION,
) -> np.ndarray:
    """
    Estimate per-node network reach from Euclidean neighbour counts.

    Counts the nodes within straight-line ``distance`` of each node (KDTree, no pairwise
    materialisation) and divides by ``deflation`` to convert the Euclidean disc count into a
    conservative estimate of the network catchment count. Used as the pilot stage for
    per-node sampling probabilities: underestimating reach leads to oversampling, which is
    safe; overestimating leads to undersampling, which the deflation guards against.

    Parameters
    ----------
    node_xs : array-like
        Node x coordinates in a projected CRS (metres).
    node_ys : array-like
        Node y coordinates in a projected CRS (metres).
    distance : float
        Distance threshold in metres.
    deflation : float
        Divisor converting Euclidean counts to conservative network-reach estimates.

    Returns
    -------
    np.ndarray
        Per-node estimated network reach (float).
    """
    from scipy.spatial import KDTree  # lazy: keeps module importable without scipy

    points = np.column_stack([np.asarray(node_xs, dtype=float), np.asarray(node_ys, dtype=float)])
    tree = KDTree(points)
    counts = tree.query_ball_point(points, r=float(distance), return_length=True, workers=-1)
    return np.asarray(counts, dtype=float) / float(deflation)


def compute_node_p(
    reach: np.ndarray | list[float],
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
    min_probability: float = MIN_NODE_PROBABILITY,
) -> np.ndarray:
    """
    Compute per-node source inclusion probabilities from per-node reach estimates.

    For each node, ``k(r) = log(2r/delta) / (2 * epsilon**2)`` and ``q = min(1, k/r)``, the
    per-node analogue of ``compute_hoeffding_p``. Sparse areas (small ``r``) receive high
    probabilities and dense areas low ones, so every catchment accumulates approximately
    ``k`` effective samples and precision is uniform across the network. Probabilities are
    floored at ``min_probability`` to bound the inverse-probability weights.

    Parameters
    ----------
    reach : array-like
        Per-node reach estimates (e.g. from ``estimate_euclidean_reach``).
    epsilon : float
        Normalised additive error tolerance. Default 0.05.
    delta : float
        Failure probability (1 - confidence). Default 0.1.
    min_probability : float
        Lower bound on the returned probabilities.

    Returns
    -------
    np.ndarray
        Per-node inclusion probabilities in ``[min_probability, 1]``.
    """
    r = np.asarray(reach, dtype=float)
    q = np.ones_like(r)
    if not (np.isfinite(epsilon) and np.isfinite(delta)) or epsilon <= 0 or delta <= 0 or delta >= 1:
        return q
    valid = np.isfinite(r) & (r > 0)
    k = np.log(2.0 * r[valid] / delta) / (2.0 * epsilon**2)
    q[valid] = np.minimum(1.0, k / r[valid])
    return np.maximum(q, float(min_probability))
