"""
# Adaptive Sampling for Network Centrality

> **Experimental.** Adaptive sampling is under active development and its API or behaviour may change in future
releases.

Computing network centrality at long distance thresholds can be slow because each source node reaches a large portion
of the network. Adaptive sampling reduces this cost by using only a subset of nodes as sources, then correcting the
results using inverse-probability weighting (IPW) to produce unbiased estimates. The required sampling rate is
determined by the Hoeffding inequality, a concentration bound that guarantees the approximation error stays within a
specified tolerance `epsilon`.

## How It Works

Sampling is per-node and adaptive: probabilities derive from each node's measured local reach rather than from the
distance threshold alone, and are applied per distance threshold:

1. **Measure reach.** A pilot polls the network with one bounded shortest-path traversal from each of a small sample
   of sources; each node's hit rate across the pilot sources estimates its own reach at every distance threshold.
   Because the pilot traverses the network itself, it respects barriers, dead ends, and disconnected fringes that a
   straight-line count would miss.
2. **Assign per-node probabilities.** The Hoeffding bound converts each node's reach estimate into its own inclusion
   probability: sparse areas receive high probabilities and dense areas low ones, so every catchment accumulates
   approximately the required number of effective samples and precision is uniform across the network. Probabilities
   derive from a conservative lower bound on the estimated reach, so pilot estimation error lands on the oversampling
   (safe) side.
3. **Decide whether sampling pays.** For each distance, the estimated sampled work is compared against exact work; if
   properly powered sampling would not be cheaper, that distance runs exactly.
4. **Correct results.** Each sampled source's contribution is reweighted by the reciprocal of its own inclusion
   probability (inverse-probability weighting), producing an unbiased estimate of the full computation regardless of
   how rough the pilot estimate is.

Because probabilities depend on measured reach, they vary from node to node and from network to network. An earlier
schedule that derived a single probability per distance from a canonical grid model is retained in this module
(`compute_distance_p`) as a reference model, but it is no longer used by the runtime.

## Accuracy

A single set of sampled sources serves both closeness and betweenness: each sampled traversal computes both metrics at
once, so sharing the sources halves the work relative to sampling each metric separately.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is calibrated empirically on real street
networks spanning the urban density range such that
node *rankings* are preserved: Spearman ρ ≥ 0.95 against exact computation at 1–20 km. Because probabilities are set
from measured per-node reach, precision holds in sparse districts and on sparse networks as well as in dense cores; a
fourth network held out from calibration (The Woodlands, TX, a very sparse dendritic suburb) meets the target at all
distances under this method. Loosen `epsilon` (for example 0.08 to 0.1) for exploratory work where approximate
rankings suffice; halving `epsilon` roughly quadruples the number of samples. Speedups are largest on dense networks
at long distances; on sparse networks with small live areas the work test will often select exact computation, because
exact closeness is already cheap there.

The full methodology and validation are documented in the sampling study under
[`analysis/sampling/`](https://github.com/benchmark-urbanism/cityseer-api/tree/master/analysis/sampling) in the
repository.

## Usage

With the high-level API:

```python
cn.centrality_shortest(
    distances=[500, 2000, 5000, 20000],
    sample=True,  # epsilon=0.05 default; pass epsilon=... to override
)
```

Or with the lower-level functional API:

```python
from cityseer.metrics import networks

nodes_gdf = networks.centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[500, 2000, 5000, 20000],
    sample=True,
)
```

## API Reference

- [`centrality_shortest`](/metrics/networks#centrality_shortest)
- [`centrality_simplest`](/metrics/networks#centrality_simplest)

"""

# This module keeps minimal hard dependencies (math, numpy; scipy imported lazily) so it can be
# imported in environments like QGIS without pulling in tqdm, rustalgos, etc.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cityseer import rustalgos

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
# oversample and clear it comfortably.
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
# deflated before use. Measured on four real-world validation networks spanning the urban
# density range at 10-20km (analysis/sampling/scripts/measure_disc_reach_ratio.py), the
# per-node ratio of Euclidean count to network reach has median 1.3-1.8 and 99th percentile
# <= 3.4; a deflation of 2.5 covers all but the extreme tail (at most ~5% of live nodes in the
# worst network/distance cell), and over-deflation only oversamples (a cost, never a bias or
# accuracy risk).
EUCLIDEAN_REACH_DEFLATION: float = 2.5

# The runtime pilot measures reach on the network itself rather than counting Euclidean
# neighbours (which is blind to barriers: rivers, highways, disconnected fringes). It polls
# m uniformly sampled sources with one bounded Dijkstra each; on an undirected network the
# probability that a random source lies within distance d of node u is reach_u(d) / n, so
# hits/m * n estimates reach at every distance threshold from a single traversal set. The m
# sources are drawn without replacement, so the per-node hit count is hypergeometric; the
# binomial Clopper-Pearson bounds used below remain valid and are conservative for it.
# Estimation error is asymmetric in cost: overestimating
# reach undersamples (an accuracy risk), underestimating only oversamples (a time cost).
# Inclusion probabilities therefore derive from a one-sided lower confidence bound on the hit
# rate; rarely hit nodes (behind barriers, sparse fringes) fall toward q = 1, a census.
# m defaults to 2.5% of nodes, floored at 400. Because q derives from the lower bound, m does
# not affect correctness: a smaller m widens the bounds, pushing q toward 1 (oversampling) and
# the work test toward exact computation. m only trades pilot cost against realised speedup;
# with the parallel Rust counter the pilot is cheap, and the larger m tightens the bounds,
# recovering most of the confidence premium (measured work within a few percent of the
# true-reach optimum at 20 km on the largest validation network).
POLL_SOURCE_FRACTION: float = 0.025
POLL_MIN_SOURCES: int = 400
POLL_LCB_ALPHA: float = 0.1

# The work test compares predicted sampled work against predicted exact work in traversal-node
# counts, which omit constant overheads (the pilot poll, per-source random draws, IPW
# arithmetic). Cells predicted as narrow wins (ratios of 0.8-1.0) deliver realised slowdowns on
# the validation networks, so sampling engages only when the predicted work falls below this
# fraction of exact work; marginal cells route to exact computation.
WORK_TEST_MARGIN: float = 0.75

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


def estimate_polled_reach(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    n_sources: int | None = None,
    alpha: float = POLL_LCB_ALPHA,
    random_seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Estimate per-node network reach by polling sampled sources with bounded Dijkstra.

    Runs one shortest-path traversal (to the largest requested distance) from each of
    ``n_sources`` uniformly sampled nodes. Each source lies within distance ``d`` of node
    ``u`` with probability ``reach_u(d) / n``, so the hit count over the sources (drawn
    without replacement, hence hypergeometric) gives an unbiased reach estimate at every
    distance threshold. Uncertainty is priced in both directions with one-sided binomial
    Clopper-Pearson bounds, which are valid and conservative for the hypergeometric
    count: inclusion probabilities derive from the lower bound (an
    overestimated reach undersamples, an accuracy risk), and work predictions from the
    upper bound (an unhit node is not free; it will be censused, and its traversal cost
    must be counted). Unlike a Euclidean neighbour count, the traversal respects
    barriers, dead ends, and disconnected fringes.

    Parameters
    ----------
    network_structure : rustalgos.graph.NetworkStructure
        The network to poll.
    distances : list[int]
        Distance thresholds in metres.
    n_sources : int | None
        Number of pilot sources. Defaults to ``POLL_SOURCE_FRACTION`` of nodes, floored
        at ``POLL_MIN_SOURCES`` and capped at the node count.
    alpha : float
        One-sided significance level for each bound. Default 0.1 (90% bounds).
    random_seed : int | None
        Seed for source selection.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]
        ``(reach_lcb, reach_point, reach_ucb)``: per-distance arrays of lower-bound,
        point, and upper-bound reach estimates, in compact node order (aligned with
        ``node_xs``).
    """
    from scipy.stats import beta  # lazy: keeps module importable without scipy

    node_idxs = np.asarray(network_structure.node_indices(), dtype=np.int64)
    n = len(node_idxs)
    dists = sorted(int(d) for d in distances)
    max_dist = dists[-1]
    m = n_sources if n_sources is not None else max(POLL_MIN_SOURCES, math.ceil(POLL_SOURCE_FRACTION * n))
    m = min(m, n)
    rng = np.random.default_rng(random_seed)
    sources = rng.choice(node_idxs, size=m, replace=False)
    if hasattr(network_structure, "poll_reach_hits"):
        # Rust counter: parallel traversals, hit counts returned per raw node index
        rows = network_structure.poll_reach_hits([int(s) for s in sources], dists, 1.0)
        hits = {d: np.asarray(rows[i], dtype=np.int64)[node_idxs] for i, d in enumerate(dists)}
    else:
        # fallback for older extensions: per-source traversals counted in Python.
        # visits are indexed by raw graph index; results are compact (node_xs) order
        compact = np.full(int(node_idxs.max()) + 1, -1, dtype=np.int64)
        compact[node_idxs] = np.arange(n, dtype=np.int64)
        hits = {d: np.zeros(n, dtype=np.int64) for d in dists}
        for src in sources:
            reachable, visits = network_structure.dijkstra_tree_shortest(int(src), max_dist, 1.0)
            r_idx = np.fromiter(reachable, dtype=np.int64)
            r_dist = np.fromiter((visits[j].short_dist for j in reachable), dtype=np.float64, count=len(r_idx))
            for d in dists:
                hits[d][compact[r_idx[r_dist <= d]]] += 1
    reach_lcb: dict[int, np.ndarray] = {}
    reach_point: dict[int, np.ndarray] = {}
    reach_ucb: dict[int, np.ndarray] = {}
    for d in dists:
        h = hits[d]
        hit = h > 0
        lcb = np.zeros(n, dtype=float)
        lcb[hit] = beta.ppf(alpha, h[hit], m - h[hit] + 1) * n
        below = h < m
        ucb = np.full(n, float(n))
        ucb[below] = beta.ppf(1.0 - alpha, h[below] + 1, m - h[below]) * n
        reach_lcb[d] = lcb
        reach_point[d] = h / m * n
        reach_ucb[d] = ucb
    return reach_lcb, reach_point, reach_ucb


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
