#!/usr/bin/env python
"""
06_generate_macros.py - Generate LaTeX macros and paper tables.

Reads the JSON/CSV output files from the analysis pipeline and generates
a LaTeX macros file that can be included in the paper. This ensures
all values in the paper are derived from actual data, not hardcoded.

Both closeness and betweenness use the Hoeffding/EW bound:
    k = log(2r/delta) / (2*epsilon^2), p = min(1, k/r)

Outputs:
    - paper/tables/model_macros.tex: LaTeX macro definitions
    - paper/tables/tab2_validation.tex ... tab6_woodlands_validation.tex: per-network
      canonical-schedule tables (appendix ablation record)
    - paper/tables/tab7_adaptive_validation.tex: consolidated per-node method table
    - paper/tables/tab8_ablation.tex: condensed canonical-schedule ablation table
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from cityseer.sampling import GRID_SPACING, compute_distance_p
from utilities import (
    CACHE_DIR,
    HOEFFDING_DELTA,
    OUTPUT_DIR,
    TABLES_DIR,
)

# Paper default epsilons — unified at 0.05 for both metrics (single calibrated parameter)
PAPER_EPSILON_CLOSENESS = 0.05
PAPER_EPSILON_BETWEENNESS = 0.05

# =============================================================================
# DATA LOADING
# =============================================================================


def load_gla_summary() -> pd.DataFrame:
    """Load GLA validation summary results."""
    path = OUTPUT_DIR / "gla_validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"GLA validation summary not found: {path}. Run 01_validate_gla.py first.")
    return pd.read_csv(path)


def load_madrid_validation() -> pd.DataFrame | None:
    """Load Madrid validation results (optional)."""
    path = OUTPUT_DIR / "madrid_validation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_cary_validation() -> pd.DataFrame | None:
    """Load Cary validation results (optional)."""
    path = OUTPUT_DIR / "cary_validation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_woodlands_validation() -> pd.DataFrame | None:
    """Load The Woodlands (held-out) validation results (optional)."""
    path = OUTPUT_DIR / "woodlands_validation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# MACRO GENERATION
# =============================================================================


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number with thousands separators for LaTeX."""
    if n >= 1000:
        return f"{n:,.0f}".replace(",", "{,}")
    elif decimals == 0:
        return f"{n:.0f}"
    else:
        return f"{n:.{decimals}f}"


def generate_macros() -> str:
    """Generate all LaTeX macros from data files."""

    gla_df = load_gla_summary()

    # -------------------------------------------------------------------------
    # Compute deterministic distance-based sampling values at standard distances
    # -------------------------------------------------------------------------
    distance_scenarios = {}
    for dist in [1000, 2000, 5000, 10000, 20000]:
        p = compute_distance_p(dist, epsilon=PAPER_EPSILON_CLOSENESS, delta=HOEFFDING_DELTA)
        r = math.pi * dist**2 / GRID_SPACING**2
        k = math.ceil(math.log(2 * r / HOEFFDING_DELTA) / (2 * PAPER_EPSILON_CLOSENESS**2))
        speedup = 1.0 / p if p < 1.0 else 1.0
        distance_scenarios[dist] = {"p": p, "k": k, "canonical_reach": r, "speedup": speedup}

    # Min rho across all distances (closeness)
    gla_min_rho_c = gla_df["rho_closeness"].min()
    gla_min_rho_c_conservative = int(gla_min_rho_c * 100) / 100

    # Min rho betweenness (may have NaN)
    gla_betw_rhos = gla_df["rho_betweenness"].dropna()
    gla_min_rho_b = gla_betw_rhos.min() if len(gla_betw_rhos) > 0 else float("nan")

    # Load GLA node counts from cache (written by 01_validate_gla.py)
    gla_n_nodes_path = CACHE_DIR / "gla_n_nodes.json"
    gla_n_nodes = None
    gla_n_total = None
    gla_live_fraction = None
    if gla_n_nodes_path.exists():
        with open(gla_n_nodes_path) as f:
            gla_node_info = json.load(f)
            gla_n_nodes = gla_node_info["n_nodes"]
            gla_n_total = gla_node_info.get("n_total")
            gla_live_fraction = gla_node_info.get("live_fraction")

    # -------------------------------------------------------------------------
    # Generate LaTeX content
    # -------------------------------------------------------------------------
    macros = f"""% =============================================================================
% Model Macros - AUTO-GENERATED from analysis pipeline
% Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
% Source files:
%   - output/{{gla,madrid,cary,woodlands}}_validation_adaptive.csv (per-node method)
%   - output/gla_validation_summary.csv and output/{{madrid,cary,woodlands}}_validation.csv
%     (canonical-schedule ablation record)
%   - .cache/{{network}}_n_nodes.json (node counts and live fractions)
%
% DO NOT EDIT THIS FILE MANUALLY - regenerate with 06_generate_macros.py
% =============================================================================

% -----------------------------------------------------------------------------
% DETERMINISTIC DISTANCE-BASED SAMPLING MODEL (both metrics)
% Canonical grid: r = pi * d^2 / s^2, s = {GRID_SPACING:.0f}m
% Hoeffding bound: k = log(2r/delta) / (2*epsilon^2), p = min(1, k/r)
% Zero fitted parameters: epsilon and delta are user-chosen conventions.
% -----------------------------------------------------------------------------

% Default parameters
\\newcommand{{\\hoeffdingEpsilon}}{{{PAPER_EPSILON_CLOSENESS}}}
\\newcommand{{\\hoeffdingEpsilonCloseness}}{{{PAPER_EPSILON_CLOSENESS}}}
\\newcommand{{\\hoeffdingEpsilonBetweenness}}{{{PAPER_EPSILON_BETWEENNESS}}}
\\newcommand{{\\hoeffdingDelta}}{{{HOEFFDING_DELTA}}}
\\newcommand{{\\gridSpacing}}{{{GRID_SPACING:.0f}}}
\\newcommand{{\\targetRho}}{{0.95}}

% Deterministic sampling (eps={PAPER_EPSILON_CLOSENESS}, delta={HOEFFDING_DELTA}, s={GRID_SPACING:.0f}m)
\\newcommand{{\\hoeffdingKFiveKm}}{{{distance_scenarios[5000]["k"]:.0f}}}
\\newcommand{{\\hoeffdingKTenKm}}{{{distance_scenarios[10000]["k"]:.0f}}}
\\newcommand{{\\hoeffdingKTwentyKm}}{{{distance_scenarios[20000]["k"]:.0f}}}

% Deterministic sampling probability at standard distances
\\newcommand{{\\hoeffdingPOneKm}}{{{distance_scenarios[1000]["p"] * 100:.1f}}}
\\newcommand{{\\hoeffdingPTwoKm}}{{{distance_scenarios[2000]["p"] * 100:.1f}}}
\\newcommand{{\\hoeffdingPFiveKm}}{{{distance_scenarios[5000]["p"] * 100:.1f}}}
\\newcommand{{\\hoeffdingPTenKm}}{{{distance_scenarios[10000]["p"] * 100:.1f}}}
\\newcommand{{\\hoeffdingPTwentyKm}}{{{distance_scenarios[20000]["p"] * 100:.1f}}}

% Deterministic speedup at standard distances
\\newcommand{{\\hoeffdingSpeedupFiveKm}}{{{distance_scenarios[5000]["speedup"]:.1f}}}
\\newcommand{{\\hoeffdingSpeedupTenKm}}{{{distance_scenarios[10000]["speedup"]:.1f}}}
\\newcommand{{\\hoeffdingSpeedupTwentyKm}}{{{distance_scenarios[20000]["speedup"]:.1f}}}

% -----------------------------------------------------------------------------
% GLA VALIDATION RESULTS
% -----------------------------------------------------------------------------

% GLA network size (live nodes within boundary)

% Minimum observed rho across all GLA distances
\\newcommand{{\\glaMinRho}}{{{gla_min_rho_c_conservative}}}
\\newcommand{{\\glaMinRhoCloseness}}{{{gla_min_rho_c:.4f}}}
"""

    if gla_n_nodes is not None:
        macros += f"\\newcommand{{\\glaNnodes}}{{{format_number(gla_n_nodes, 0)}}}\n"
    if gla_n_total is not None:
        macros += f"\\newcommand{{\\glaNtotal}}{{{format_number(gla_n_total, 0)}}}\n"
    if gla_live_fraction is not None:
        macros += f"\\newcommand{{\\glaLiveFraction}}{{{gla_live_fraction:.2f}}}\n"

    if not np.isnan(gla_min_rho_b):
        macros += f"\\newcommand{{\\glaMinRhoBetweenness}}{{{gla_min_rho_b:.4f}}}\n"

    # Per-distance GLA macros — all p values from deterministic distance-based formula
    for dist, label in [(5000, "FiveKm"), (10000, "TenKm"), (20000, "TwentyKm")]:
        dist_row = gla_df[gla_df["distance"] == dist]
        if dist_row.empty:
            continue
        r = dist_row.iloc[0]
        det_p = compute_distance_p(dist, epsilon=PAPER_EPSILON_CLOSENESS, delta=HOEFFDING_DELTA)
        det_speedup = 1.0 / det_p if det_p < 1.0 else 1.0

        macros += f"\n% {dist // 1000}km validation\n"
        macros += f"\\newcommand{{\\gla{label}RhoH}}{{{r['rho_closeness']:.3f}}}\n"
        macros += f"\\newcommand{{\\gla{label}Rho}}{{{r['rho_closeness']:.3f}}}\n"
        macros += f"\\newcommand{{\\gla{label}P}}{{{det_p * 100:.1f}}}\n"
        macros += f"\\newcommand{{\\gla{label}TheoreticalSpeedup}}{{{det_speedup:.1f}}}\n"

        # Measured wall-clock speedup
        spd_c = r.get("speedup_closeness", float("nan"))
        if np.isfinite(spd_c):
            macros += f"\\newcommand{{\\gla{label}Speedup}}{{{spd_c:.1f}}}\n"
            macros += f"\\newcommand{{\\gla{label}SpeedupCloseness}}{{{spd_c:.1f}}}\n"

        # Betweenness rho and speedup
        rho_b = r.get("rho_betweenness", float("nan"))
        if np.isfinite(rho_b):
            macros += f"\\newcommand{{\\gla{label}RhoB}}{{{rho_b:.3f}}}\n"
        spd_b = r.get("speedup_betweenness", float("nan"))
        if np.isfinite(spd_b):
            macros += f"\\newcommand{{\\gla{label}SpeedupBetweenness}}{{{spd_b:.1f}}}\n"

    macros += "\n% Live node buffer (km)\n\\newcommand{\\glaBuffer}{20}\n"

    # -------------------------------------------------------------------------
    # Madrid validation macros (optional)
    # -------------------------------------------------------------------------
    madrid_df = load_madrid_validation()
    madrid_min_rho_c = None
    if madrid_df is not None:
        madrid_min_rho_c = madrid_df["rho_closeness"].min()
        madrid_betw_rhos = (
            madrid_df["rho_betweenness"].dropna() if "rho_betweenness" in madrid_df.columns else pd.Series(dtype=float)
        )
        madrid_min_rho_b = madrid_betw_rhos.min() if len(madrid_betw_rhos) > 0 else float("nan")

        madrid_n_nodes_path = CACHE_DIR / "madrid_n_nodes.json"
        madrid_n_nodes = None
        madrid_n_total = None
        madrid_live_fraction = None
        if madrid_n_nodes_path.exists():
            with open(madrid_n_nodes_path) as f:
                madrid_node_info = json.load(f)
                madrid_n_nodes = madrid_node_info["n_nodes"]
                madrid_n_total = madrid_node_info.get("n_total")
                madrid_live_fraction = madrid_node_info.get("live_fraction")

        macros += """
% -----------------------------------------------------------------------------
% MADRID VALIDATION RESULTS
% -----------------------------------------------------------------------------

"""
        if madrid_n_nodes is not None:
            macros += f"\\newcommand{{\\madridNnodes}}{{{format_number(madrid_n_nodes, 0)}}}\n"
        if madrid_n_total is not None:
            macros += f"\\newcommand{{\\madridNtotal}}{{{format_number(madrid_n_total, 0)}}}\n"
        if madrid_live_fraction is not None:
            macros += f"\\newcommand{{\\madridLiveFraction}}{{{madrid_live_fraction:.2f}}}\n"
        macros += f"\\newcommand{{\\madridMinRho}}{{{madrid_min_rho_c:.4f}}}\n"
        if not np.isnan(madrid_min_rho_b):
            macros += f"\\newcommand{{\\madridMinRhoBetweenness}}{{{madrid_min_rho_b:.4f}}}\n"
        macros += "\n"

        for dist, label in [(5000, "FiveKm"), (10000, "TenKm"), (20000, "TwentyKm")]:
            dist_row = madrid_df[madrid_df["distance"] == dist]
            if dist_row.empty:
                continue
            r = dist_row.iloc[0]
            macros += f"% {dist // 1000}km validation\n"
            macros += f"\\newcommand{{\\madrid{label}RhoH}}{{{r['rho_closeness']:.4f}}}\n"

            spd_c = r.get("speedup_closeness", float("nan"))
            if np.isfinite(spd_c):
                macros += f"\\newcommand{{\\madrid{label}Speedup}}{{{spd_c:.1f}}}\n"

            rho_b = r.get("rho_betweenness", float("nan"))
            if np.isfinite(rho_b):
                macros += f"\\newcommand{{\\madrid{label}RhoB}}{{{rho_b:.4f}}}\n"
            spd_b = r.get("speedup_betweenness", float("nan"))
            if np.isfinite(spd_b):
                macros += f"\\newcommand{{\\madrid{label}SpeedupBetweenness}}{{{spd_b:.1f}}}\n"
            macros += "\n"

    # -------------------------------------------------------------------------
    # Cary (suburban) validation macros (optional)
    # -------------------------------------------------------------------------
    cary_df = load_cary_validation()
    cary_min_rho_c = None
    if cary_df is not None:
        cary_min_rho_c = cary_df["rho_closeness"].min()
        cary_betw_rhos = (
            cary_df["rho_betweenness"].dropna() if "rho_betweenness" in cary_df.columns else pd.Series(dtype=float)
        )
        cary_min_rho_b = cary_betw_rhos.min() if len(cary_betw_rhos) > 0 else float("nan")

        cary_n_nodes_path = CACHE_DIR / "cary_n_nodes.json"
        cary_n_nodes = None
        cary_n_total = None
        cary_live_fraction = None
        if cary_n_nodes_path.exists():
            with open(cary_n_nodes_path) as f:
                cary_node_info = json.load(f)
                cary_n_nodes = cary_node_info["n_nodes"]
                cary_n_total = cary_node_info.get("n_total")
                cary_live_fraction = cary_node_info.get("live_fraction")

        macros += """
% -----------------------------------------------------------------------------
% CARY (SUBURBAN) VALIDATION RESULTS
% -----------------------------------------------------------------------------

"""
        if cary_n_nodes is not None:
            macros += f"\\newcommand{{\\caryNnodes}}{{{format_number(cary_n_nodes, 0)}}}\n"
        if cary_n_total is not None:
            macros += f"\\newcommand{{\\caryNtotal}}{{{format_number(cary_n_total, 0)}}}\n"
        if cary_live_fraction is not None:
            macros += f"\\newcommand{{\\caryLiveFraction}}{{{cary_live_fraction:.2f}}}\n"
        macros += f"\\newcommand{{\\caryMinRho}}{{{cary_min_rho_c:.4f}}}\n"
        if not np.isnan(cary_min_rho_b):
            macros += f"\\newcommand{{\\caryMinRhoBetweenness}}{{{cary_min_rho_b:.4f}}}\n"
        macros += "\n"

        for dist, label in [(5000, "FiveKm"), (10000, "TenKm"), (20000, "TwentyKm")]:
            dist_row = cary_df[cary_df["distance"] == dist]
            if dist_row.empty:
                continue
            r = dist_row.iloc[0]
            macros += f"% {dist // 1000}km validation\n"
            macros += f"\\newcommand{{\\cary{label}RhoH}}{{{r['rho_closeness']:.4f}}}\n"
            spd_c = r.get("speedup_closeness", float("nan"))
            if np.isfinite(spd_c):
                macros += f"\\newcommand{{\\cary{label}Speedup}}{{{spd_c:.1f}}}\n"
            rho_b = r.get("rho_betweenness", float("nan"))
            if np.isfinite(rho_b):
                macros += f"\\newcommand{{\\cary{label}RhoB}}{{{rho_b:.4f}}}\n"
            spd_b = r.get("speedup_betweenness", float("nan"))
            if np.isfinite(spd_b):
                macros += f"\\newcommand{{\\cary{label}SpeedupBetweenness}}{{{spd_b:.1f}}}\n"
            macros += "\n"

    # -------------------------------------------------------------------------
    # The Woodlands (held-out suburban) validation macros (optional)
    # -------------------------------------------------------------------------
    woodlands_df = load_woodlands_validation()
    woodlands_min_rho_c = None
    if woodlands_df is not None:
        woodlands_min_rho_c = woodlands_df["rho_closeness"].min()
        woodlands_betw_rhos = (
            woodlands_df["rho_betweenness"].dropna()
            if "rho_betweenness" in woodlands_df.columns
            else pd.Series(dtype=float)
        )
        woodlands_min_rho_b = woodlands_betw_rhos.min() if len(woodlands_betw_rhos) > 0 else float("nan")

        woodlands_n_nodes_path = CACHE_DIR / "woodlands_n_nodes.json"
        woodlands_n_nodes = None
        woodlands_n_total = None
        woodlands_live_fraction = None
        if woodlands_n_nodes_path.exists():
            with open(woodlands_n_nodes_path) as f:
                woodlands_node_info = json.load(f)
                woodlands_n_nodes = woodlands_node_info["n_nodes"]
                woodlands_n_total = woodlands_node_info.get("n_total")
                woodlands_live_fraction = woodlands_node_info.get("live_fraction")

        macros += """
% -----------------------------------------------------------------------------
% THE WOODLANDS (HELD-OUT) VALIDATION RESULTS
% -----------------------------------------------------------------------------

"""
        if woodlands_n_nodes is not None:
            macros += f"\\newcommand{{\\woodlandsNnodes}}{{{format_number(woodlands_n_nodes, 0)}}}\n"
        if woodlands_n_total is not None:
            macros += f"\\newcommand{{\\woodlandsNtotal}}{{{format_number(woodlands_n_total, 0)}}}\n"
        if woodlands_live_fraction is not None:
            macros += f"\\newcommand{{\\woodlandsLiveFraction}}{{{woodlands_live_fraction:.2f}}}\n"
        macros += f"\\newcommand{{\\woodlandsMinRho}}{{{woodlands_min_rho_c:.4f}}}\n"
        if not np.isnan(woodlands_min_rho_b):
            macros += f"\\newcommand{{\\woodlandsMinRhoBetweenness}}{{{woodlands_min_rho_b:.4f}}}\n"
        macros += "\n"

        for dist, label in [(5000, "FiveKm"), (10000, "TenKm"), (20000, "TwentyKm")]:
            dist_row = woodlands_df[woodlands_df["distance"] == dist]
            if dist_row.empty:
                continue
            r = dist_row.iloc[0]
            macros += f"% {dist // 1000}km validation\n"
            macros += f"\\newcommand{{\\woodlands{label}RhoH}}{{{r['rho_closeness']:.4f}}}\n"
            spd_c = r.get("speedup_closeness", float("nan"))
            if np.isfinite(spd_c):
                macros += f"\\newcommand{{\\woodlands{label}Speedup}}{{{spd_c:.1f}}}\n"
            rho_b = r.get("rho_betweenness", float("nan"))
            if np.isfinite(rho_b):
                macros += f"\\newcommand{{\\woodlands{label}RhoB}}{{{rho_b:.4f}}}\n"
            spd_b = r.get("speedup_betweenness", float("nan"))
            if np.isfinite(spd_b):
                macros += f"\\newcommand{{\\woodlands{label}SpeedupBetweenness}}{{{spd_b:.1f}}}\n"
            macros += "\n"

    # -------------------------------------------------------------------------
    # Overall minimum rho across all validated networks
    # -------------------------------------------------------------------------
    overall_min_rho_values = [gla_min_rho_c]
    if not np.isnan(gla_min_rho_b):
        overall_min_rho_values.append(gla_min_rho_b)
    if madrid_min_rho_c is not None:
        overall_min_rho_values.append(madrid_min_rho_c)
    if madrid_df is not None and "rho_betweenness" in madrid_df.columns:
        madrid_b = madrid_df["rho_betweenness"].dropna()
        if len(madrid_b) > 0:
            overall_min_rho_values.append(float(madrid_b.min()))
    if cary_min_rho_c is not None:
        overall_min_rho_values.append(cary_min_rho_c)
    if cary_df is not None and "rho_betweenness" in cary_df.columns:
        cary_b = cary_df["rho_betweenness"].dropna()
        if len(cary_b) > 0:
            overall_min_rho_values.append(float(cary_b.min()))
    # The held-out network (Woodlands) is reported separately and is not folded into
    # the overall minimum, which summarises the calibration-range networks.
    overall_min_rho = min(overall_min_rho_values)
    overall_min_rho_conservative = int(overall_min_rho * 100) / 100

    # Per-node (adaptive) method macros from the adaptive CSVs. Rho and speedup macros are
    # emitted per metric, only for entries whose per-metric work test selected sampling;
    # exact entries equal the ground truth by construction and carry no timing.
    dist_labels = {1000: "OneKm", 2000: "TwoKm", 5000: "FiveKm", 10000: "TenKm", 20000: "TwentyKm"}
    quartile_labels = {1: "QOne", 2: "QTwo", 3: "QThree", 4: "QFour"}
    adaptive_minima = []
    adaptive_max_std = 0.0
    closeness_quartile_rhos: list[float] = []
    twenty_km_speedups: list[float] = []
    for network in ["gla", "madrid", "cary", "woodlands"]:
        path = OUTPUT_DIR / f"{network}_validation_adaptive.csv"
        if not path.exists():
            continue
        adf = pd.read_csv(path)
        net_min = float(adf[["rho_closeness", "rho_betweenness"]].min().min())
        adaptive_minima.append(net_min)
        macros += f"\n% {network} per-node (adaptive) method\n"
        macros += f"\\newcommand{{\\{network}AdaptiveMinRho}}{{{net_min:.4f}}}\n"
        for col in ["rho_closeness_std", "rho_betweenness_std"]:
            if col in adf.columns:
                adaptive_max_std = max(adaptive_max_std, float(adf[col].max()))
        for _, arow in adf.iterrows():
            label = dist_labels.get(int(arow["distance"]))
            if label is None:
                continue
            if not _closeness_ran_exact(arow) and arow["mode"] != "exact":
                n_seeds_c = arow.get("n_seeds_closeness", float("nan"))
                if np.isfinite(n_seeds_c) and int(n_seeds_c) != 3:
                    raise ValueError(
                        f"{network} {arow['distance']}m: sampled closeness mean covers "
                        f"{int(n_seeds_c)} seeds, expected 3 (near-threshold seed exclusion)"
                    )
                macros += f"\\newcommand{{\\{network}Adaptive{label}RhoH}}{{{arow['rho_closeness']:.4f}}}\n"
                if np.isfinite(arow.get("speedup_closeness", float("nan"))):
                    macros += f"\\newcommand{{\\{network}Adaptive{label}SpeedupC}}{{{arow['speedup_closeness']:.1f}}}\n"
                    if int(arow["distance"]) == 20000:
                        twenty_km_speedups.append(float(arow["speedup_closeness"]))
                # Within-quartile closeness rho (reach quartiles), cited by the error-structure
                # discussion; range restriction depresses these relative to the network-wide rho.
                for qi, qlabel in quartile_labels.items():
                    qval = arow.get(f"h_spearman_q{qi}", float("nan"))
                    if np.isfinite(qval):
                        macros += f"\\newcommand{{\\{network}Adaptive{label}RhoH{qlabel}}}{{{qval:.3f}}}\n"
                        closeness_quartile_rhos.append(float(qval))
            if not _betweenness_ran_exact(arow):
                n_seeds_b = arow.get("n_seeds_betweenness", float("nan"))
                if np.isfinite(n_seeds_b) and int(n_seeds_b) != 3:
                    raise ValueError(
                        f"{network} {arow['distance']}m: sampled betweenness mean covers "
                        f"{int(n_seeds_b)} seeds, expected 3 (near-threshold seed exclusion)"
                    )
                macros += f"\\newcommand{{\\{network}Adaptive{label}RhoB}}{{{arow['rho_betweenness']:.4f}}}\n"
                if np.isfinite(arow.get("speedup_betweenness", float("nan"))):
                    macros += (
                        f"\\newcommand{{\\{network}Adaptive{label}SpeedupB}}{{{arow['speedup_betweenness']:.1f}}}\n"
                    )
                    if int(arow["distance"]) == 20000:
                        twenty_km_speedups.append(float(arow["speedup_betweenness"]))
    if adaptive_minima:
        adaptive_min = min(adaptive_minima)
        macros += f"\n\\newcommand{{\\adaptiveMinRho}}{{{adaptive_min:.4f}}}\n"
        # Conservative 2-dp floor for the abstract and conclusion, which should survive
        # regeneration without prose edits.
        macros += f"\\newcommand{{\\adaptiveMinRhoFloor}}{{{int(adaptive_min * 100) / 100}}}\n"
        macros += f"\\newcommand{{\\adaptiveMaxSeedStd}}{{{adaptive_max_std:.4f}}}\n"
    if closeness_quartile_rhos:
        macros += f"\\newcommand{{\\adaptiveClosenessQuartileMinRho}}{{{min(closeness_quartile_rhos):.3f}}}\n"
    if twenty_km_speedups:
        # Min/max over every sampled 20km cell (both metrics, all networks), so range
        # claims in the abstract and discussion cover the cells they describe.
        macros += f"\\newcommand{{\\adaptiveTwentyKmSpeedupMin}}{{{min(twenty_km_speedups):.1f}}}\n"
        macros += f"\\newcommand{{\\adaptiveTwentyKmSpeedupMax}}{{{max(twenty_km_speedups):.1f}}}\n"

    # Reach ratio (actual mean reach / canonical reach) at 20km for the suburbs:
    # quantifies how far each falls below the canonical grid model.
    r20 = math.pi * 20000**2 / GRID_SPACING**2
    for df_ratio, name in [(cary_df, "cary"), (woodlands_df, "woodlands")]:
        if df_ratio is None:
            continue
        row20 = df_ratio[df_ratio["distance"] == 20000]
        if row20.empty:
            continue
        ratio = float(row20.iloc[0]["mean_reach"]) / r20
        macros += f"\\newcommand{{\\{name}ReachRatioTwentyKm}}{{{ratio:.2f}}}\n"

    macros += f"""
% -----------------------------------------------------------------------------
% OVERALL (CROSS-NETWORK) METRICS
% -----------------------------------------------------------------------------

% Minimum observed rho across all validated networks (conservative, 2dp)
\\newcommand{{\\overallMinRho}}{{{overall_min_rho_conservative}}}
"""

    return macros


# =============================================================================
# VALIDATION TABLES (GLA + MADRID)
# =============================================================================


def _load_live_fraction(network: str) -> float:
    """Load live fraction from cached node info JSON."""
    path = CACHE_DIR / f"{network}_n_nodes.json"
    if path.exists():
        with open(path) as f:
            info = json.load(f)
            return info.get("live_fraction", 1.0)
    return 1.0


def generate_validation_table(
    df: pd.DataFrame,
    network_name: str,
    network_key: str,
    label: str,
    nnodes_macro: str,
    epsilon: float,
    include_shared_notes: bool = False,
) -> str:
    """Generate a LaTeX validation table for one network.

    Only requires the summary/validation CSV and the n_nodes JSON cache.
    """
    live_fraction = _load_live_fraction(network_key)

    latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Sampling validation on {network_name}
  ($\varepsilon = {epsilon}$, $\delta = 0.1$, $s = {GRID_SPACING:.0f}\,$m, $\varphi = {live_fraction:.2f}$).}}
\label{{{label}}}
\begin{{tabular}}{{rrrrrrr}}
\toprule
\textbf{{Dist.}} &
\textbf{{$p$}} & \textbf{{$\rho_c$}} & \textbf{{Spd$_c$}} &
& \textbf{{$\rho_b$}} & \textbf{{Spd$_b$}} \\
\midrule
"""

    for _, row in df.iterrows():
        p_val = row["hoeffding_p_close"]
        is_exact = p_val >= live_fraction

        if is_exact:
            p_pct = "exact"
            spd_c = "---"
            spd_b = "---"
        else:
            p_pct = f"{p_val * 100:.1f}\\%"
            spd_c = f"{row['speedup_closeness']:.1f}$\\times$" if np.isfinite(row["speedup_closeness"]) else "---"
            spd_b = (
                f"{row['speedup_betweenness']:.1f}$\\times$"
                if np.isfinite(row.get("speedup_betweenness", float("nan")))
                else "---"
            )

        rho_c = f"{row['rho_closeness']:.4f}"
        rho_b = f"{row['rho_betweenness']:.4f}" if np.isfinite(row.get("rho_betweenness", float("nan"))) else "---"

        latex += f"{int(row['distance'] // 1000)}\\,km & "
        latex += f"{p_pct} & {rho_c} & {spd_c} & "
        latex += f"& {rho_b} & {spd_b} \\\\\n"

    latex += rf"""\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\footnotesize
Network: {network_name}, {nnodes_macro} nodes.
"""
    if include_shared_notes:
        latex += r"""Rows marked ``exact'' have $p \geq \varphi$, the schedule's fallback threshold;
the rule is sized to exact closeness cost and applies to the whole call, so both
metrics run exact in those rows.
Subscripts: $c$ = closeness, $b$ = betweenness.
Notes apply to Tables~\ref{tab:validation}--\ref{tab:woodlands_validation}.
"""
    latex += "\\end{table}\n"
    return latex


def generate_validation_tables():
    """Generate validation tables for GLA and Madrid from cached CSVs."""
    print("\nGenerating validation tables...")

    # GLA
    gla_summary_path = OUTPUT_DIR / "gla_validation_summary.csv"
    if gla_summary_path.exists():
        gla_df = pd.read_csv(gla_summary_path)
        latex = generate_validation_table(
            gla_df,
            network_name="Greater London network",
            network_key="gla",
            label="tab:validation",
            nnodes_macro=r"\glaNnodes{}",
            epsilon=PAPER_EPSILON_CLOSENESS,
            include_shared_notes=True,
        )
        path = TABLES_DIR / "tab2_validation.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"  Saved: {path}")

    # Madrid
    madrid_path = OUTPUT_DIR / "madrid_validation.csv"
    if madrid_path.exists():
        madrid_df = pd.read_csv(madrid_path)
        latex = generate_validation_table(
            madrid_df,
            network_name="Greater Madrid network",
            network_key="madrid",
            label="tab:madrid_validation",
            nnodes_macro=r"\madridNnodes{}",
            epsilon=PAPER_EPSILON_CLOSENESS,
        )
        path = TABLES_DIR / "tab4_madrid_validation.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"  Saved: {path}")

    # Cary
    cary_path = OUTPUT_DIR / "cary_validation.csv"
    if cary_path.exists():
        cary_df = pd.read_csv(cary_path)
        latex = generate_validation_table(
            cary_df,
            network_name="Cary, NC (suburban) network",
            network_key="cary",
            label="tab:cary_validation",
            nnodes_macro=r"\caryNnodes{}",
            epsilon=PAPER_EPSILON_CLOSENESS,
        )
        path = TABLES_DIR / "tab5_cary_validation.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"  Saved: {path}")

    # The Woodlands (held-out)
    woodlands_path = OUTPUT_DIR / "woodlands_validation.csv"
    if woodlands_path.exists():
        woodlands_df = pd.read_csv(woodlands_path)
        latex = generate_validation_table(
            woodlands_df,
            network_name="The Woodlands, TX (held-out suburban) network",
            network_key="woodlands",
            label="tab:woodlands_validation",
            nnodes_macro=r"\woodlandsNnodes{}",
            epsilon=PAPER_EPSILON_CLOSENESS,
        )
        path = TABLES_DIR / "tab6_woodlands_validation.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"  Saved: {path}")


# =============================================================================
# PER-NODE METHOD AND ABLATION TABLES
# =============================================================================


def _closeness_ran_exact(row: pd.Series) -> bool:
    """True when the closeness work test selected exact computation for this row.

    The runtime plans each metric separately: a closeness-only call can route to exact
    computation while the betweenness call samples. Newer CSVs record the decision in a
    ``closeness_mode`` column; older CSVs lack it, and the decision is recovered from the
    recorded errors (an exact closeness run has zero error in every reach quartile).
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


def generate_adaptive_table() -> None:
    """Consolidated per-node method table across the four networks and all distances.

    Reads {network}_validation_adaptive.csv (produced by validate_adaptive.py). Rows are
    selected by the CSV mode columns rather than a hardcoded distance list. Entries whose
    per-metric work test selected exact computation are rendered ``exact`` with no timing.
    """
    frames = []
    for network, label in [("gla", "London"), ("madrid", "Madrid"), ("cary", "Cary"), ("woodlands", "Woodlands")]:
        path = OUTPUT_DIR / f"{network}_validation_adaptive.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["network"] = label
        frames.append(df)
    if not frames:
        print("  No adaptive validation CSVs found; skipping adaptive table.")
        return
    data = pd.concat(frames, ignore_index=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-node method on the four validation networks. Entries marked exact"
        r" record the work test declining to sample by design, where exact computation is"
        r" already cheaper (Section~\ref{sec:worktest}); they equal the ground truth, so no"
        r" timing is reported for them. All speedups include the pilot's cost."
        r" ($\varepsilon = " + str(PAPER_EPSILON_CLOSENESS) + r"$, $\delta = 0.1$; mean of three seeds;"
        r" each metric ran through its single-metric entry point, so the decision is recorded"
        r" per metric. Speedups are wall-clock ratios of the exact runtime to the sampled"
        r" runtime.)}",
        r"\label{tab:adaptive_validation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Network} & \textbf{Dist.} & \textbf{$\rho_c$} & \textbf{Speed-up$_c$} &"
        r" \textbf{$\rho_b$} & \textbf{Speed-up$_b$} \\",
        r"\midrule",
    ]
    prev_network = None
    for _, row in data.iterrows():
        if prev_network is not None and row["network"] != prev_network:
            lines.append(r"\addlinespace")
        prev_network = row["network"]
        dist_str = f"{int(row['distance']) // 1000}\\,km"
        if _closeness_ran_exact(row) or row["mode"] == "exact":
            c_cells = "exact & ---"
        else:
            spd_c = f"{row['speedup_closeness']:.1f}$\\times$" if np.isfinite(row["speedup_closeness"]) else "---"
            c_cells = f"{row['rho_closeness']:.4f} & {spd_c}"
        if _betweenness_ran_exact(row):
            b_cells = "exact & ---"
        else:
            spd_b = f"{row['speedup_betweenness']:.1f}$\\times$" if np.isfinite(row["speedup_betweenness"]) else "---"
            b_cells = f"{row['rho_betweenness']:.4f} & {spd_b}"
        lines.append(f"{row['network']} & {dist_str} & {c_cells} & {b_cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = TABLES_DIR / "tab7_adaptive_validation.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def generate_ablation_table() -> None:
    """Condensed canonical-schedule ablation table: per-network minima across distances.

    Reads the canonical validation CSVs (the ablation's rung-1 record) and reports each
    network's minimum rho per metric. The per-distance detail lives in the appendix tables.
    """
    sources = [
        ("gla_validation_summary.csv", "London"),
        ("madrid_validation.csv", "Madrid"),
        ("cary_validation.csv", "Cary"),
        ("woodlands_validation.csv", "Woodlands (held out)"),
    ]
    rows = []
    argmin_dists: set[int] = set()
    for fname, label in sources:
        path = OUTPUT_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        min_c = float(df["rho_closeness"].min())
        min_b = float(df["rho_betweenness"].dropna().min())
        argmin_dists.add(int(df.loc[df["rho_closeness"].idxmin(), "distance"]))
        argmin_dists.add(int(df.loc[df["rho_betweenness"].idxmin(), "distance"]))
        c_str = f"{min_c:.4f}$^{{\\dagger}}$" if min_c < 0.95 else f"{min_c:.4f}"
        b_str = f"{min_b:.4f}$^{{\\dagger}}$" if min_b < 0.95 else f"{min_b:.4f}"
        rows.append(f"{label} & {c_str} & {b_str} \\\\")
    if not rows:
        print("  No canonical validation CSVs found; skipping ablation table.")
        return
    # State where the minima occur from the data rather than as a fixed sentence.
    if argmin_dists == {20000}:
        argmin_sentence = r" All minima occur at 20\,km."
    else:
        dist_list = ", ".join(f"{d // 1000}\\,km" for d in sorted(argmin_dists))
        argmin_sentence = f" Minima occur at {dist_list}."
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation rung 1 (assumed reach): the canonical schedule's minimum"
        r" Spearman $\rho$ per network across 1--20\,km." + argmin_sentence + r" The held-out network fails the"
        r" $\rho \geq 0.95$ target for closeness; per-distance detail is in the appendix"
        r" (Tables~\ref{tab:validation}--\ref{tab:woodlands_validation}).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Network} & \textbf{Closeness (worst $\rho$)} & \textbf{Betweenness (worst $\rho$)} \\",
        r"\midrule",
    ]
    lines += rows
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.5em}",
        r"\footnotesize",
        r"$^{\dagger}$below the $\rho \geq 0.95$ target.",
        r"\end{table}",
        "",
    ]
    path = TABLES_DIR / "tab8_ablation.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("06_generate_macros.py - Generating LaTeX macros and paper tables")
    print("=" * 70)

    try:
        macros = generate_macros()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nRun the pipeline scripts first:")
        print("  python scripts/01_validate_gla.py")
        return 1

    output_path = TABLES_DIR / "model_macros.tex"
    with open(output_path, "w") as f:
        f.write(macros)

    print(f"\nGenerated: {output_path}")

    generate_validation_tables()
    generate_adaptive_table()
    generate_ablation_table()

    # Print summary
    print("\n" + "=" * 70)
    print("MACRO SUMMARY")
    print("=" * 70)

    print("\nDeterministic distance-based model:")
    print(f"  Epsilon:       {PAPER_EPSILON_CLOSENESS}")
    print(f"  Delta:         {HOEFFDING_DELTA}")
    print(f"  Grid spacing:  {GRID_SPACING}m")

    for dist in [5000, 10000, 20000]:
        det_p = compute_distance_p(dist, epsilon=PAPER_EPSILON_CLOSENESS, delta=HOEFFDING_DELTA)
        spd = 1.0 / det_p if det_p < 1.0 else 1.0
        print(f"  d={dist // 1000}km: p={det_p:.3f}, speedup={spd:.1f}x")

    gla_df = load_gla_summary()
    print("\nGLA validation:")
    for _, row in gla_df.iterrows():
        rho_b_str = (
            f", rho_b={row['rho_betweenness']:.3f}" if not np.isnan(row.get("rho_betweenness", float("nan"))) else ""
        )
        spd_c = row.get("speedup_closeness", float("nan"))
        spd_str = f", speedup_c={spd_c:.1f}x" if np.isfinite(spd_c) else ""
        print(f"  {int(row['distance'] / 1000)}km: rho_c={row['rho_closeness']:.3f}{rho_b_str}{spd_str}")

    madrid_df = load_madrid_validation()
    if madrid_df is not None:
        print("\nMadrid validation:")
        for _, row in madrid_df.iterrows():
            rho_b_str = (
                f", rho_b={row['rho_betweenness']:.3f}"
                if not np.isnan(row.get("rho_betweenness", float("nan")))
                else ""
            )
            spd_c = row.get("speedup_closeness", float("nan"))
            spd_str = f", speedup_c={spd_c:.1f}x" if np.isfinite(spd_c) else ""
            print(f"  {int(row['distance'] / 1000)}km: rho_c={row['rho_closeness']:.3f}{rho_b_str}{spd_str}")

    return 0


if __name__ == "__main__":
    exit(main())
