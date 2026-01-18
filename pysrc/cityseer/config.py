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
# Model for expected Spearman ρ: rho = 1 - A / (B + eff_n)
# Model for standard deviation: std = C / sqrt(D + eff_n)
# Model for expected scale (magnitude bias): scale = 1 - E / (F + eff_n)
SAMPLING_MODEL_RHO_A: float = 14.27  # Numerator coefficient for rho model
SAMPLING_MODEL_RHO_B: float = 20.1  # Denominator offset for rho model
SAMPLING_MODEL_STD_C: float = 0.806  # Numerator coefficient for std model
SAMPLING_MODEL_STD_D: float = 8.31  # Denominator offset for std model
SAMPLING_MODEL_BIAS_E: float = 0.46  # Numerator coefficient for bias model
SAMPLING_MODEL_BIAS_F: float = -0.13  # Denominator offset for bias model


def get_expected_spearman(effective_n: float) -> tuple[float, float]:
    """
    Get expected Spearman ρ and standard deviation for given effective sample size.

    Uses fitted empirical models:
    - Expected ρ = 1 - A / (B + eff_n)
    - Std dev = C / sqrt(D + eff_n)

    Parameters
    ----------
    effective_n : float
        The effective sample size (reach × p)

    Returns
    -------
    tuple[float, float]
        (expected_spearman, std_dev)
    """
    # Ensure minimum effective_n of 1 to avoid edge cases
    eff_n = max(1.0, effective_n)

    # Expected Spearman ρ
    expected_rho = 1.0 - SAMPLING_MODEL_RHO_A / (SAMPLING_MODEL_RHO_B + eff_n)
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


def get_required_effective_n(target_spearman: float) -> float | None:
    """
    Get the minimum effective_n required to achieve a target Spearman ρ.

    Inverts the model: eff_n = A / (1 - target) - B

    Parameters
    ----------
    target_spearman : float
        Target Spearman ρ (e.g., 0.95)

    Returns
    -------
    float | None
        Required effective_n, or None if target is impossible (≥1.0)
    """
    if target_spearman >= 1.0:
        return None
    if target_spearman <= 0.0:
        return 0.0

    # Invert: rho = 1 - A / (B + n) => n = A / (1 - rho) - B
    required_n = SAMPLING_MODEL_RHO_A / (1.0 - target_spearman) - SAMPLING_MODEL_RHO_B
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
    partial_func: Callable,  # type: ignore
) -> RustResults:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[RustResults | Exception]):
        try:
            result: RustResults = partial_func()  # type: ignore
            queue.put(result)  # type: ignore
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
        time.sleep(1)
        pbar.update(rust_struct.progress() - pbar.n)  # type: ignore
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()

    if isinstance(result, Exception):
        raise result

    return result
