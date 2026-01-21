from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from queue import Queue

import numpy as np
from tqdm import tqdm

from . import rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.seterr(invalid="ignore")


def prep_gdf_key(key: str, dist: int, angular: bool = False, weighted: bool | None = None) -> str:
    """Format a column label for GeoPandas."""
    key = key.replace(".0", "")
    key = key.replace(".0_", "_")
    key = f"cc_{key}_{dist}"
    if angular is True:
        key += "_ang"
    if weighted is True:
        key += "_wt"
    elif weighted is False:
        key += "_nw"
    return key


def check_quiet() -> bool:
    """Check whether to enable quiet mode."""
    if "GCP_PROJECT" in os.environ:
        return True
    return "CITYSEER_QUIET_MODE" in os.environ and os.environ["CITYSEER_QUIET_MODE"].lower() in [
        "true",
        "1",
    ]


QUIET_MODE = check_quiet()


def check_debug() -> bool:
    """Check whether to enable debug mode."""
    return "CITYSEER_DEBUG_MODE" in os.environ and os.environ["CITYSEER_DEBUG_MODE"].lower() in [
        "true",
        "1",
    ]


DEBUG_MODE: bool = check_debug()
# for turning off validation
SKIP_VALIDATION: bool = False
# for calculating default betas vs. distances
MIN_THRESH_WT: float = 0.01831563888873418
SPEED_M_S = 1.33333
# for all_close equality checks
ATOL: float = 0.01
RTOL: float = 0.0001

# Empirical model parameters for sampling accuracy prediction
# Fitted from analysis across network topologies (trellis, tree, linear)
# effective_n = mean_reachability × sample_probability
# See analysis/output/README.md for methodology
#
# Models fitted to 10th percentile for conservative estimates.
# Betweenness has higher variance than closeness, so separate models are provided.
#
# Model for expected Spearman ρ: rho = 1 - A / (B + eff_n)
# Model for standard deviation: std = C / sqrt(D + eff_n)
# Model for expected scale (magnitude bias): scale = 1 - E / (F + eff_n)

# Harmonic (closeness) model - lower variance, more permissive
SAMPLING_MODEL_HARMONIC_A: float = 32.4
SAMPLING_MODEL_HARMONIC_B: float = 31.54

# Betweenness model - higher variance, more conservative (used as default)
SAMPLING_MODEL_BETWEENNESS_A: float = 48.31
SAMPLING_MODEL_BETWEENNESS_B: float = 49.12

# Additional models for std deviation and bias estimation
SAMPLING_MODEL_STD_C: float = 1.166  # Numerator coefficient for std model
SAMPLING_MODEL_STD_D: float = 14.01  # Denominator offset for std model
SAMPLING_MODEL_BIAS_E: float = 0.46  # Numerator coefficient for bias model
SAMPLING_MODEL_BIAS_F: float = -0.13  # Denominator offset for bias model


def get_expected_spearman(
    effective_n: float,
    metric: str = "betweenness",
) -> tuple[float, float]:
    """
    Get expected Spearman ρ and standard deviation for given effective sample size.

    Uses fitted empirical models:
    - Expected ρ = 1 - A / (B + eff_n)
    - Std dev = C / sqrt(D + eff_n)

    Parameters
    ----------
    effective_n : float
        The effective sample size (reach × p)
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.

    Returns
    -------
    tuple[float, float]
        (expected_spearman, std_dev)
    """
    # Ensure minimum effective_n of 1 to avoid edge cases
    eff_n = max(1.0, effective_n)

    # Select model parameters based on metric
    if metric == "harmonic":
        a, b = SAMPLING_MODEL_HARMONIC_A, SAMPLING_MODEL_HARMONIC_B
    else:
        # "betweenness" or "both" - use conservative model
        a, b = SAMPLING_MODEL_BETWEENNESS_A, SAMPLING_MODEL_BETWEENNESS_B

    # Expected Spearman ρ
    expected_rho = 1.0 - a / (b + eff_n)
    expected_rho = max(0.0, min(1.0, expected_rho))  # Clamp to [0, 1]

    # Standard deviation
    std_dev = SAMPLING_MODEL_STD_C / (SAMPLING_MODEL_STD_D + eff_n) ** 0.5
    std_dev = max(0.001, std_dev)  # Minimum std

    return expected_rho, std_dev


def get_expected_bias(effective_n: float) -> float:
    """
    Get expected magnitude bias for given effective sample size.

    Uses fitted empirical model: scale = 1 - E / (F + eff_n)
    Bias is reported as (1 - scale), i.e., the fraction by which
    magnitudes are expected to be underestimated.

    Parameters
    ----------
    effective_n : float
        The effective sample size (reach × p)

    Returns
    -------
    float
        Expected bias as a fraction (e.g., 0.05 means ~5% underestimate)
    """
    eff_n = max(1.0, effective_n)
    expected_scale = 1.0 - SAMPLING_MODEL_BIAS_E / (SAMPLING_MODEL_BIAS_F + eff_n)
    expected_scale = max(0.0, min(1.0, expected_scale))  # Clamp to [0, 1]
    return 1.0 - expected_scale


def get_required_effective_n(
    target_spearman: float,
    metric: str = "betweenness",
) -> float | None:
    """
    Get the minimum effective_n required to achieve a target Spearman ρ.

    Inverts the model: eff_n = A / (1 - target) - B

    Parameters
    ----------
    target_spearman : float
        Target Spearman ρ (e.g., 0.95)
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.

    Returns
    -------
    float | None
        Required effective_n, or None if target is impossible (≥1.0)
    """
    if target_spearman >= 1.0:
        return None
    if target_spearman <= 0.0:
        return 0.0

    # Select model parameters based on metric
    if metric == "harmonic":
        a, b = SAMPLING_MODEL_HARMONIC_A, SAMPLING_MODEL_HARMONIC_B
    else:
        # "betweenness" or "both" - use conservative model
        a, b = SAMPLING_MODEL_BETWEENNESS_A, SAMPLING_MODEL_BETWEENNESS_B

    # Invert: rho = 1 - A / (B + n) => n = A / (1 - rho) - B
    required_n = a / (1.0 - target_spearman) - b
    return max(1.0, required_n)


def compute_required_p(mean_reachability: float, target_spearman: float = 0.95) -> float | None:
    """
    Compute the sampling probability required to achieve target accuracy.

    Parameters
    ----------
    mean_reachability : float
        Average number of nodes reachable within distance threshold
    target_spearman : float
        Target Spearman ρ (default 0.95)

    Returns
    -------
    float | None
        Required sampling probability, or None if impossible
    """
    required_n = get_required_effective_n(target_spearman)
    if required_n is None or mean_reachability <= 0:
        return None
    required_p = required_n / mean_reachability
    return min(1.0, required_p)


def log_thresholds(
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
):
    # pair distances, betas, and time for logging - DO AFTER PARTIAL FUNC
    distances, betas, seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    # log distances, betas, minutes
    logger.info("Metrics computed for:")
    for distance, beta, walking_time in zip(distances, betas, seconds, strict=True):
        logger.info(f"Distance: {distance}m, Beta: {round(beta, 5)}, Walking Time: {walking_time / 60} minutes.")
    return distances


def log_sampling(
    sample_probability: float | None = None,
    distances: list[int] | None = None,
    reachability_totals: list[int] | None = None,
    sampled_source_count: int | None = None,
) -> None:
    """
    Log sampling statistics when sampling is enabled.

    Provides users with:
    - Expected ranking accuracy (Spearman ρ) based on effective sample size
    - Recommendations for achieving target accuracy levels where applicable
    - Warnings when effective sample size is very low

    The effective sample size (eff_n = reachability × p) determines accuracy:
    - eff_n ≥ 260: Expected ρ ≥ 0.95
    - eff_n ≥ 120: Expected ρ ≥ 0.90
    - eff_n < 50:  High variance expected
    """
    if sample_probability is None:
        return

    # Log sampling overview
    logger.info("")  # Visual separator
    speedup = 1 / sample_probability
    logger.info(f"Sampling enabled: p={sample_probability:.0%}, theoretical speedup ~{speedup:.1f}x")
    logger.info("  Spearman ρ measures ranking preservation (0=uncorrelated, 1=identical)")
    logger.info("  Bias shows expected magnitude underestimate (0%=unbiased)")

    # If we have actual reachability data, log effective_n and expected accuracy
    if reachability_totals and sampled_source_count and sampled_source_count > 0 and distances:
        logger.info("  Per-distance accuracy estimates:")

        for dist, total_reach in zip(distances, reachability_totals, strict=True):
            mean_reach = total_reach / sampled_source_count
            if mean_reach <= 0:
                continue

            effective_n = mean_reach * sample_probability
            expected_rho, std_rho = get_expected_spearman(effective_n)

            # Compute required p for different accuracy targets
            p_for_90 = compute_required_p(mean_reach, 0.90)
            p_for_95 = compute_required_p(mean_reach, 0.95)

            # Build recommendation based on achievable targets
            # Check if targets are achievable (p < 1.0 means sampling still provides speedup)
            can_reach_95 = p_for_95 is not None and p_for_95 < 1.0
            can_reach_90 = p_for_90 is not None and p_for_90 < 1.0

            if expected_rho >= 0.95:
                # Already at ρ≥0.95
                recommendation = ""
            elif can_reach_95:
                # Can achieve ρ≥0.95 with higher p
                recommendation = f" → p≥{p_for_95:.0%} for ρ≥0.95"
            elif can_reach_90:
                # Can achieve ρ≥0.90 with higher p, but ρ≥0.95 requires full computation
                recommendation = f" → p≥{p_for_90:.0%} for ρ≥0.90"
            else:
                # Neither target achievable with sampling - reach is too low
                recommendation = " (reach too low for ρ≥0.90 with sampling)"

            # Get expected bias
            expected_bias = get_expected_bias(effective_n)
            bias_str = f", bias={expected_bias:.0%}" if expected_bias >= 0.01 else ""

            # Log the main info line
            logger.info(
                f"    {dist}m: reach={mean_reach:.0f}, eff_n={effective_n:.0f}, "
                f"expected ρ={expected_rho:.2f}±{std_rho:.2f}{bias_str}{recommendation}"
            )


# =============================================================================
# Adaptive Sampling Helpers
# =============================================================================
# These functions support per-distance adaptive sampling, where sampling
# probability is calibrated separately for each distance threshold based on
# reachability. This allows aggressive sampling at large distances (high reach)
# while maintaining accuracy at short distances (low reach).


def probe_reachability(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    n_probes: int = 50,
    speed_m_s: float = SPEED_M_S,
) -> dict[int, float]:
    """
    Estimate mean reachability per distance by probing a sample of nodes.

    This is a lightweight pre-computation step that runs Dijkstra from a small
    sample of nodes to estimate how many nodes are typically reachable at each
    distance threshold. The estimates are used to compute appropriate sampling
    probabilities for adaptive sampling.

    Parameters
    ----------
    network_structure
        The network to probe.
    distances
        Distance thresholds in metres.
    n_probes
        Number of random nodes to sample for probing. Default 50.
        More probes = more accurate estimates but longer probe time.
    speed_m_s
        Walking speed for converting distance to seconds.

    Returns
    -------
    dict[int, float]
        Mean reachability (node count) for each distance threshold.
    """
    import random

    # Get live nodes for sampling
    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    if not live_indices:
        return {d: 0.0 for d in distances}

    # Sample probe nodes
    n_probes = min(n_probes, len(live_indices))
    probe_indices = random.sample(live_indices, n_probes)

    # Accumulate reach counts per distance
    reach_counts: dict[int, list[int]] = {d: [] for d in distances}
    max_seconds = int(max(distances) / speed_m_s)

    for src_idx in probe_indices:
        # Run single Dijkstra to max distance
        visited, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None
        )

        # Count nodes reachable at each distance threshold
        for d in distances:
            count = sum(1 for v_idx in visited if tree_map[v_idx].short_dist <= d and v_idx != src_idx)
            reach_counts[d].append(count)

    # Return mean reachability per distance
    return {d: float(np.mean(counts)) if counts else 0.0 for d, counts in reach_counts.items()}


def compute_sample_probs_for_target_rho(
    reach_estimates: dict[int, float],
    target_rho: float = 0.95,
    metric: str = "both",
    safety_margin: float = 0.02,
) -> dict[int, float | None]:
    """
    Compute the sampling probability required at each distance to achieve target accuracy.

    Uses the empirical model: ρ = 1 - A / (B + effective_n)
    where effective_n = reach × p.

    Solving for p: p = required_eff_n / reach
    where required_eff_n = A / (1 - target_rho) - B

    A safety margin is added to the target ρ to account for model variance.
    The model is fitted to the 10th percentile, so ~10% of observations may
    fall below the predicted value. The safety margin compensates for this.

    Parameters
    ----------
    reach_estimates
        Mean reachability per distance (from probe_reachability).
    target_rho
        Target Spearman ρ correlation. Default 0.95.
    metric
        Which metric model to use: "harmonic", "betweenness", or "both".
        - "harmonic": Use closeness model (less conservative, more speedup)
        - "betweenness": Use betweenness model (more conservative)
        - "both": Use betweenness model to ensure both metrics meet target
    safety_margin
        Additional margin added to target_rho for robustness. Default 0.02.
        E.g., target_rho=0.95 with margin=0.02 internally targets 0.97.

    Returns
    -------
    dict[int, float | None]
        Sampling probability for each distance.
        Returns None for distances where reach is too low to achieve target with any p ≤ 1.0.
    """
    # Apply safety margin to target - this ensures we overshoot slightly
    # to account for model variance (fitted to 10th percentile)
    effective_target = min(0.99, target_rho + safety_margin)
    required_eff_n = get_required_effective_n(effective_target, metric=metric)
    if required_eff_n is None:
        # Target ρ ≥ 1.0 is impossible
        return {d: None for d in reach_estimates}

    result = {}
    for d, reach in reach_estimates.items():
        if reach <= 0:
            result[d] = None
        else:
            p = required_eff_n / reach
            # Cap at 1.0 - if p > 1.0, we need full computation
            result[d] = min(1.0, p)

    return result


def log_adaptive_sampling_plan(
    distances: list[int],
    reach_estimates: dict[int, float],
    sample_probs: dict[int, float | None],
    target_rho: float,
    metric: str = "both",
    safety_margin: float = 0.02,
) -> None:
    """
    Log the adaptive sampling plan before execution.

    Parameters
    ----------
    distances
        Distance thresholds.
    reach_estimates
        Estimated reachability per distance.
    sample_probs
        Computed sampling probabilities per distance.
    target_rho
        Target accuracy level.
    metric
        Which metric model is being used: "harmonic", "betweenness", or "both".
    safety_margin
        Safety margin applied to target_rho internally.
    """
    logger.info("")  # Visual separator
    metric_label = {"harmonic": "closeness", "betweenness": "betweenness", "both": "both metrics"}.get(
        metric, "both metrics"
    )
    effective_target = min(0.99, target_rho + safety_margin)
    logger.info(
        f"Adaptive sampling plan (target ρ ≥ {target_rho:.2f}, "
        f"internal target {effective_target:.2f} for {metric_label}):"
    )
    logger.info("  Distance │  Reach │ Sample p │ Expected ρ")
    logger.info("  ─────────┼────────┼──────────┼───────────")

    for d in sorted(distances):
        reach = reach_estimates.get(d, 0)
        p = sample_probs.get(d)

        # Full computation (p >= 1.0 or None) means exact results
        if p is None or p >= 1.0:
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │     full │ 1.00 (exact)")
        else:
            eff_n = reach * p
            exp_rho, _ = get_expected_spearman(eff_n, metric=metric)
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │ {p:>7.0%} │ {exp_rho:.2f} (eff_n={eff_n:.0f})")


RustResults = (
    rustalgos.centrality.CentralityShortestResult
    | rustalgos.centrality.CentralitySimplestResult
    | rustalgos.centrality.CentralitySegmentResult
    | rustalgos.data.AccessibilityResult
    | rustalgos.data.MixedUsesResult
    | rustalgos.data.StatsResult
)


def wrap_progress(
    total: int,
    rust_struct: rustalgos.graph.NetworkStructure | rustalgos.data.DataMap | rustalgos.viewshed.Viewshed,
    partial_func: Callable,
) -> RustResults:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[RustResults | Exception]):
        try:
            result: RustResults = partial_func()
            queue.put(result)
        except Exception as e:
            queue.put(e)

    result_queue: Queue[RustResults | Exception] = Queue()
    thread = threading.Thread(target=wrapper, args=(result_queue,))
    pbar = tqdm(
        total=total,
        disable=QUIET_MODE,
    )
    thread.start()
    while thread.is_alive():
        time.sleep(0.1)
        pbar.update(rust_struct.progress() - pbar.n)  # type: ignore
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()

    if isinstance(result, Exception):
        raise result

    return result
