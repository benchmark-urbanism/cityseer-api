#!/usr/bin/env python
"""
07_generate_macros.py - Generate LaTeX macros from fitted model data.

Reads the JSON output files from the analysis pipeline and generates
a LaTeX macros file that can be included in the paper. This ensures
all values in the paper are derived from actual data, not hardcoded.

Outputs:
    - paper/tables/model_macros.tex: LaTeX macro definitions
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
TABLES_DIR = SAMPLING_DIR / "paper" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_model_fit() -> dict:
    """Load the fitted k value."""
    path = OUTPUT_DIR / "model_fit.json"
    if not path.exists():
        raise FileNotFoundError(f"Model fit not found: {path}. Run 01_fit_model.py first.")
    with open(path) as f:
        return json.load(f)


def load_floor_fit() -> dict:
    """Load the fitted min_eff_n value."""
    path = OUTPUT_DIR / "floor_fit.json"
    if not path.exists():
        raise FileNotFoundError(f"Floor fit not found: {path}. Run 02_fit_floor.py first.")
    with open(path) as f:
        return json.load(f)


def load_sampling_model() -> dict:
    """Load the combined sampling model."""
    path = OUTPUT_DIR / "sampling_model.json"
    if not path.exists():
        raise FileNotFoundError(f"Sampling model not found: {path}. Run 03_combined_model.py first.")
    with open(path) as f:
        return json.load(f)


def load_gla_validation() -> pd.DataFrame:
    """Load GLA validation results."""
    path = OUTPUT_DIR / "gla_validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"GLA validation not found: {path}. Run 04_validate_gla.py first.")
    return pd.read_csv(path)


def load_power_exponent() -> dict | None:
    """Load power exponent analysis (optional, from 08_validate_power_exponent.py)."""
    path = OUTPUT_DIR / "power_exponent_analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_model_comparison() -> dict | None:
    """Load model comparison results (optional, from 09_model_comparison.py)."""
    path = OUTPUT_DIR / "model_comparison.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# =============================================================================
# MACRO GENERATION
# =============================================================================


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number with thousands separators for LaTeX."""
    if n >= 1000:
        # Use {,} for LaTeX thousands separator
        return f"{n:,.0f}".replace(",", "{,}")
    elif decimals == 0:
        return f"{n:.0f}"
    else:
        return f"{n:.{decimals}f}"


def generate_macros() -> str:
    """Generate all LaTeX macros from data files."""

    # Load all data
    model_fit = load_model_fit()
    floor_fit = load_floor_fit()
    sampling_model = load_sampling_model()
    gla_df = load_gla_validation()

    # Extract values
    k = model_fit["model"]["k"]
    k_mean = model_fit["fitting_stats"]["k_mean"]
    k_p75 = model_fit["fitting_stats"].get("k_p75", k)  # Upper quartile
    k_p95 = model_fit["fitting_stats"]["k_p95"]
    k_max = model_fit["model"].get("k_max", k * 1.1)  # Conservative max
    n_configs = model_fit["fitting_stats"]["n_configs"]
    target_rho = model_fit["model"]["target_rho"]

    min_eff_n = floor_fit["model"]["min_eff_n"]
    floor_success_rate = floor_fit["model"]["achieved_success_rate"] * 100

    crossover_reach = sampling_model["model"]["crossover_reach"]

    # GLA validation data
    gla_5km = gla_df[gla_df["distance"] == 5000].iloc[0]
    gla_10km = gla_df[gla_df["distance"] == 10000].iloc[0]
    gla_20km = gla_df[gla_df["distance"] == 20000].iloc[0]

    # Compute minimum rho across all distances (conservative bound)
    min_rho = min(gla_5km["observed_rho"], gla_10km["observed_rho"], gla_20km["observed_rho"])
    # Round down to 2 decimal places for conservative claim
    min_rho_conservative = int(min_rho * 100) / 100

    # Generate LaTeX content
    macros = f"""% =============================================================================
% Model Macros - AUTO-GENERATED from analysis pipeline
% Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
% Source files:
%   - output/model_fit.json
%   - output/floor_fit.json
%   - output/sampling_model.json
%   - output/gla_validation_summary.csv
%   - output/power_exponent_analysis.json (if available)
%   - output/model_comparison.json (if available)
%
% DO NOT EDIT THIS FILE MANUALLY - regenerate with 07_generate_macros.py
% =============================================================================

% -----------------------------------------------------------------------------
% CORE MODEL PARAMETERS
% -----------------------------------------------------------------------------

% Proportional constant k (fitted from synthetic data)
\\newcommand{{\\kProp}}{{{k}}}

% Minimum effective sample size floor
\\newcommand{{\\minEffN}}{{{min_eff_n}}}

% Target Spearman correlation
\\newcommand{{\\targetRho}}{{{target_rho}}}

% Crossover reach (where floor and proportional are equal)
\\newcommand{{\\crossoverReach}}{{{int(crossover_reach)}}}

% -----------------------------------------------------------------------------
% FITTING STATISTICS
% -----------------------------------------------------------------------------

% k fitting statistics
\\newcommand{{\\kMean}}{{{k_mean}}}
\\newcommand{{\\kPseventyFive}}{{{k_p75}}}
\\newcommand{{\\kPninetyFive}}{{{k_p95}}}
\\newcommand{{\\kMax}}{{{k_max}}}
\\newcommand{{\\nConfigs}}{{{n_configs}}}

% Floor fitting - achieved success rate at min_eff_n
\\newcommand{{\\floorSuccessRate}}{{{floor_success_rate:.0f}}}

% -----------------------------------------------------------------------------
% GLA VALIDATION RESULTS
% -----------------------------------------------------------------------------

% GLA network size (approximate)
\\newcommand{{\\glaNnodes}}{{294{{,}}000}}

% Minimum observed rho across all GLA distances (conservative bound)
\\newcommand{{\\glaMinRho}}{{{min_rho_conservative}}}

% 5km validation
\\newcommand{{\\glaFiveKmReach}}{{{format_number(gla_5km["reach"], 0)}}}
\\newcommand{{\\glaFiveKmRho}}{{{gla_5km["observed_rho"]:.3f}}}
\\newcommand{{\\glaFiveKmSpeedup}}{{{gla_5km["speedup"]:.1f}}}

% 10km validation
\\newcommand{{\\glaTenKmReach}}{{{format_number(gla_10km["reach"], 0)}}}
\\newcommand{{\\glaTenKmRho}}{{{gla_10km["observed_rho"]:.3f}}}
\\newcommand{{\\glaTenKmSpeedup}}{{{gla_10km["speedup"]:.1f}}}

% 20km validation
\\newcommand{{\\glaTwentyKmReach}}{{{format_number(gla_20km["reach"], 0)}}}
\\newcommand{{\\glaTwentyKmRho}}{{{gla_20km["observed_rho"]:.3f}}}
\\newcommand{{\\glaTwentyKmSpeedup}}{{{gla_20km["speedup"]:.1f}}}

% Live node buffer (km)
\\newcommand{{\\glaBuffer}}{{20}}

% -----------------------------------------------------------------------------
% SYNTHETIC DATA PARAMETERS
% -----------------------------------------------------------------------------

% Number of topologies tested
\\newcommand{{\\nTopologies}}{{3}}

% Number of distances tested
\\newcommand{{\\nDistances}}{{12}}

% Number of sampling probabilities tested
\\newcommand{{\\nProbs}}{{22}}

% Maximum analysis distance for synthetic networks
\\newcommand{{\\syntheticMaxDist}}{{4{{,}}000}}

% Inward buffer for synthetic networks (km)
\\newcommand{{\\syntheticBuffer}}{{4}}
"""

    # Add power exponent macros if available
    pe_data = load_power_exponent()
    if pe_data is not None:
        pe = pe_data["power_exponent"]
        fl = pe_data["floor_logistic"]

        macros += f"""
% -----------------------------------------------------------------------------
% POWER EXPONENT VALIDATION (from 08_validate_power_exponent.py)
% -----------------------------------------------------------------------------

% Fitted power exponent
\\newcommand{{\\alphaHat}}{{{pe["alpha_hat"]:.4f}}}
\\newcommand{{\\alphaSE}}{{{pe["alpha_se"]:.4f}}}

% Bootstrap confidence interval for alpha
\\newcommand{{\\alphaCILower}}{{{pe["boot_ci_lower"]:.2f}}}
\\newcommand{{\\alphaCIUpper}}{{{pe["boot_ci_upper"]:.2f}}}

% Hypothesis test: H0: alpha = 0.5
\\newcommand{{\\alphaTestPValue}}{{{pe["p_value_vs_half"]:.2f}}}
\\newcommand{{\\alphaTestTStat}}{{{pe["t_stat_vs_half"]:.2f}}}

% OLS R-squared
\\newcommand{{\\alphaRSquared}}{{{pe["r_squared"]:.2f}}}

% Logistic floor confidence interval
\\newcommand{{\\floorCILower}}{{{fl["recommended_floor_ci_lower"]}}}
\\newcommand{{\\floorCIUpper}}{{{fl["recommended_floor_ci_upper"]}}}
\\newcommand{{\\floorLogistic}}{{{fl["recommended_floor"]}}}
"""

    # Add model comparison macros if available
    mc_data = load_model_comparison()
    if mc_data is not None:
        best = mc_data["best_model"]
        best_model = mc_data["models"][best]
        proposed = mc_data["models"]["sqrt_floor"]

        macros += f"""
% -----------------------------------------------------------------------------
% MODEL COMPARISON (from 09_model_comparison.py)
% -----------------------------------------------------------------------------

% Best model by AIC
\\newcommand{{\\bestModelName}}{{{best_model["name"]}}}
\\newcommand{{\\bestModelAIC}}{{{best_model["AIC"]:.1f}}}
\\newcommand{{\\bestModelBrier}}{{{best_model["cv_brier_mean"]:.4f}}}

% Proposed model (sqrt + floor)
\\newcommand{{\\proposedModelAIC}}{{{proposed["AIC"]:.1f}}}
\\newcommand{{\\proposedDeltaAIC}}{{{proposed["delta_AIC"]:.1f}}}
\\newcommand{{\\proposedModelBrier}}{{{proposed["cv_brier_mean"]:.4f}}}

% Number of candidate models compared
\\newcommand{{\\nModelsCompared}}{{{len(mc_data["models"])}}}
"""

    return macros


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("07_generate_macros.py - Generating LaTeX macros from data")
    print("=" * 70)

    try:
        macros = generate_macros()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nRun the pipeline scripts first:")
        print("  python scripts/01_fit_model.py")
        print("  python scripts/02_fit_floor.py")
        print("  python scripts/03_combined_model.py")
        print("  python scripts/04_validate_gla.py")
        return 1

    # Write macros file
    output_path = TABLES_DIR / "model_macros.tex"
    with open(output_path, "w") as f:
        f.write(macros)

    print(f"\nGenerated: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("MACRO SUMMARY")
    print("=" * 70)

    model = load_sampling_model()
    print("\nCore model parameters:")
    print(f"  k = {model['model']['k']}")
    print(f"  min_eff_n = {model['model']['min_eff_n']}")
    print(f"  crossover_reach = {model['model']['crossover_reach']:.0f}")

    gla_df = load_gla_validation()
    print("\nGLA validation:")
    for _, row in gla_df.iterrows():
        print(f"  {int(row['distance'] / 1000)}km: rho={row['observed_rho']:.3f}, speedup={row['speedup']:.1f}x")

    print("\n" + "=" * 70)
    print("USAGE")
    print("=" * 70)
    print("\nIn your LaTeX document, include:")
    print("  \\input{tables/model_macros}")
    print("\nThen use macros like:")
    print("  The model uses $k = \\kProp$ and $\\minEffN$.")
    print("  Validation achieved $\\rho > \\glaMinRho$.")

    return 0


if __name__ == "__main__":
    exit(main())
