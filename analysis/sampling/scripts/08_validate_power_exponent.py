#!/usr/bin/env python
"""
08_validate_power_exponent.py - Validate sqrt(reach) assumption and fit floor with uncertainty.

This script formally tests whether the power exponent in the sampling model
is consistent with 0.5 (square root scaling), and replaces the binned floor
fitting with logistic regression for proper confidence intervals.

Outputs:
    - output/power_exponent_analysis.json: Alpha estimate, CI, hypothesis test
    - output/floor_logistic_fit.json: Logistic regression floor with CI
    - paper/figures/fig_s1_diagnostics.pdf: Supplementary diagnostic plots
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
FIGURES_DIR = SAMPLING_DIR / "paper" / "figures"
CACHE_DIR = SAMPLING_DIR.parent.parent / "temp" / "sampling_cache"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CACHE = CACHE_DIR / "sampling_analysis_v17.pkl"

TARGET_RHO = 0.95
N_BOOTSTRAP = 2000

# Matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_synthetic_data() -> pd.DataFrame:
    """Load cached synthetic network sampling results."""
    if not SYNTHETIC_CACHE.exists():
        raise FileNotFoundError(f"Synthetic data cache not found at {SYNTHETIC_CACHE}")

    with open(SYNTHETIC_CACHE, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded synthetic data: {len(df)} rows")
    return df


# =============================================================================
# PHASE 1: CREATE THRESHOLD DATASET
# =============================================================================


def create_threshold_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (topology, distance), find the minimum p achieving rho >= target.
    Interpolate between bracketing probabilities for precision.
    """
    df_b = df[df["metric"] == "betweenness"].copy()

    threshold_data = []

    for (topo, dist), group in df_b.groupby(["topology", "distance"]):
        reach = group["mean_reach"].iloc[0]
        group_sorted = group.sort_values("sample_prob")

        achieving = group_sorted[group_sorted["spearman"] >= TARGET_RHO]

        if len(achieving) > 0:
            min_p = achieving["sample_prob"].iloc[0]
            achieved_rho = achieving["spearman"].iloc[0]

            # Find the bracketing lower probability for interpolation
            below_idx = group_sorted["sample_prob"] < min_p
            if below_idx.any():
                p_below = group_sorted.loc[below_idx, "sample_prob"].iloc[-1]
                rho_below = group_sorted.loc[below_idx, "spearman"].iloc[-1]
            else:
                p_below, rho_below = np.nan, np.nan
        else:
            min_p = 1.0
            achieved_rho = group_sorted["spearman"].max()
            p_below, rho_below = np.nan, np.nan

        # Linear interpolation for precise threshold
        if not np.isnan(p_below) and not np.isnan(rho_below) and achieved_rho != rho_below:
            p_interp = p_below + (TARGET_RHO - rho_below) * (min_p - p_below) / (achieved_rho - rho_below)
            p_interp = np.clip(p_interp, p_below, min_p)
        else:
            p_interp = min_p

        eff_n_interp = reach * p_interp

        threshold_data.append(
            {
                "topology": topo,
                "distance": dist,
                "reach": reach,
                "min_p": min_p,
                "p_interp": p_interp,
                "eff_n_threshold": reach * min_p,
                "eff_n_interp": eff_n_interp,
                "achieved_rho": achieved_rho,
                "log_reach": np.log(reach) if reach > 0 else np.nan,
                "log_eff_n_interp": np.log(eff_n_interp) if eff_n_interp > 0 else np.nan,
            }
        )

    df_thresh = pd.DataFrame(threshold_data)
    # Drop any rows where reach or eff_n is zero/nan
    df_thresh = df_thresh.dropna(subset=["log_reach", "log_eff_n_interp"])

    print(f"Threshold dataset: {len(df_thresh)} configurations")
    print(f"  Topologies: {df_thresh['topology'].unique().tolist()}")
    print(f"  Reach range: {df_thresh['reach'].min():.0f} - {df_thresh['reach'].max():.0f}")

    return df_thresh


# =============================================================================
# PHASE 3: FIT POWER EXPONENT
# =============================================================================


def fit_power_exponent(df_thresh: pd.DataFrame) -> dict:
    """
    Fit general power model: log(eff_n) = log(k) + alpha * log(reach)
    Test H0: alpha = 0.5
    """
    print("\n" + "=" * 60)
    print("POWER EXPONENT ESTIMATION")
    print("=" * 60)

    X = sm.add_constant(df_thresh["log_reach"])
    y = df_thresh["log_eff_n_interp"]

    model_ols = sm.OLS(y, X).fit()

    alpha_hat = model_ols.params.iloc[1]  # log_reach coefficient
    alpha_se = model_ols.bse.iloc[1]
    log_k_hat = model_ols.params.iloc[0]  # intercept
    k_hat = np.exp(log_k_hat)

    print(f"\nOLS fit: log(eff_n) = {log_k_hat:.4f} + {alpha_hat:.4f} * log(reach)")
    print(f"  Equivalent: eff_n = {k_hat:.2f} * reach^{alpha_hat:.4f}")
    print("\n  Alpha (power exponent):")
    print(f"    Point estimate: {alpha_hat:.4f}")
    print(f"    Standard error: {alpha_se:.4f}")
    print(f"    95% CI: [{alpha_hat - 1.96 * alpha_se:.4f}, {alpha_hat + 1.96 * alpha_se:.4f}]")
    print(f"    R-squared: {model_ols.rsquared:.4f}")

    # Hypothesis test: H0: alpha = 0.5
    t_stat = (alpha_hat - 0.5) / alpha_se
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=model_ols.df_resid))

    print("\n  Hypothesis test: H0: alpha = 0.5")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Decision: {'REJECT H0' if p_value < 0.05 else 'FAIL TO REJECT H0'}")

    # Practical significance check
    practical_diff = abs(alpha_hat - 0.5)
    print(f"    |alpha - 0.5| = {practical_diff:.4f} (practically significant if > 0.1)")

    return {
        "alpha_hat": round(float(alpha_hat), 4),
        "alpha_se": round(float(alpha_se), 4),
        "alpha_ci_lower": round(float(alpha_hat - 1.96 * alpha_se), 4),
        "alpha_ci_upper": round(float(alpha_hat + 1.96 * alpha_se), 4),
        "k_hat": round(float(k_hat), 2),
        "log_k_hat": round(float(log_k_hat), 4),
        "r_squared": round(float(model_ols.rsquared), 4),
        "t_stat_vs_half": round(float(t_stat), 4),
        "p_value_vs_half": round(float(p_value), 4),
        "reject_h0": bool(p_value < 0.05),
        "n_obs": int(model_ols.nobs),
        "ols_summary": str(model_ols.summary()),
    }


def bootstrap_alpha(df_thresh: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap CI for power exponent."""
    print(f"\nBootstrap CI (n={n_boot})...")

    np.random.seed(42)
    boot_alphas = []

    for _ in range(n_boot):
        indices = np.random.choice(len(df_thresh), size=len(df_thresh), replace=True)
        boot_df = df_thresh.iloc[indices]
        X_boot = sm.add_constant(boot_df["log_reach"])
        y_boot = boot_df["log_eff_n_interp"]
        try:
            model_boot = sm.OLS(y_boot, X_boot).fit()
            boot_alphas.append(float(model_boot.params.iloc[1]))
        except Exception:
            continue

    boot_alphas = np.array(boot_alphas)
    ci = np.percentile(boot_alphas, [2.5, 97.5])
    contains_half = ci[0] <= 0.5 <= ci[1]

    print(f"  Bootstrap alpha mean: {np.mean(boot_alphas):.4f}")
    print(f"  Bootstrap alpha std: {np.std(boot_alphas):.4f}")
    print(f"  Bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  CI contains 0.5: {contains_half}")

    return {
        "boot_alpha_mean": round(float(np.mean(boot_alphas)), 4),
        "boot_alpha_std": round(float(np.std(boot_alphas)), 4),
        "boot_ci_lower": round(float(ci[0]), 4),
        "boot_ci_upper": round(float(ci[1]), 4),
        "ci_contains_half": bool(contains_half),
        "n_bootstrap": n_boot,
    }


# =============================================================================
# PHASE 4: LOGISTIC REGRESSION FLOOR
# =============================================================================


def fit_floor_logistic(df: pd.DataFrame) -> dict:
    """
    Fit floor using logistic regression of success probability on effective sample size.
    More principled than binned approach, with proper confidence intervals.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION FLOOR FITTING")
    print("=" * 60)

    df_b = df[df["metric"] == "betweenness"].copy()
    df_b["success"] = (df_b["spearman"] >= TARGET_RHO).astype(int)

    # Fit logistic regression: P(success) = logistic(beta0 + beta1 * eff_n)
    X_logit = sm.add_constant(df_b["effective_n"])
    model_logit = sm.Logit(df_b["success"], X_logit).fit(disp=0)

    beta0 = model_logit.params.iloc[0]
    beta1 = model_logit.params.iloc[1]
    se_beta0 = model_logit.bse.iloc[0]
    se_beta1 = model_logit.bse.iloc[1]

    print("\nLogistic regression:")
    print(f"  P(rho >= {TARGET_RHO}) = logistic({beta0:.4f} + {beta1:.6f} * eff_n)")
    print(f"  beta0 = {beta0:.4f} (SE: {se_beta0:.4f})")
    print(f"  beta1 = {beta1:.6f} (SE: {se_beta1:.6f})")

    # Compute floor at different target success rates
    results = {}
    for target in [0.90, 0.95, 0.99]:
        logit_target = np.log(target / (1 - target))
        floor_point = (logit_target - beta0) / beta1

        # CI via simulation from parameter covariance
        np.random.seed(42)
        cov = model_logit.cov_params()
        samples = np.random.multivariate_normal([beta0, beta1], cov, size=10000)
        floor_samples = (logit_target - samples[:, 0]) / samples[:, 1]
        floor_samples = floor_samples[np.isfinite(floor_samples) & (floor_samples > 0)]

        ci_lower = float(np.percentile(floor_samples, 2.5))
        ci_upper = float(np.percentile(floor_samples, 97.5))

        # Delta method SE
        grad = np.array([-1 / beta1, -(logit_target - beta0) / beta1**2])
        var_floor = float(grad @ cov @ grad)
        se_floor = np.sqrt(var_floor)

        key = f"target_{int(target * 100)}"
        results[key] = {
            "target_success_rate": target,
            "floor_point": round(float(floor_point), 1),
            "floor_ci_lower": round(ci_lower, 1),
            "floor_ci_upper": round(ci_upper, 1),
            "floor_se": round(float(se_floor), 1),
        }

        print(f"\n  Floor for {target:.0%} success rate:")
        print(f"    Point estimate: {floor_point:.0f}")
        print(f"    95% CI (simulation): [{ci_lower:.0f}, {ci_upper:.0f}]")
        print(f"    SE (delta method): {se_floor:.0f}")

    # Primary recommendation (95% success rate)
    primary = results["target_95"]

    print(f"\n  RECOMMENDED FLOOR: min_eff_n = {primary['floor_point']:.0f}")
    print(f"  95% CI: [{primary['floor_ci_lower']:.0f}, {primary['floor_ci_upper']:.0f}]")

    return {
        "method": "logistic_regression",
        "beta0": round(float(beta0), 4),
        "beta1": round(float(beta1), 6),
        "n_obs": int(model_logit.nobs),
        "log_likelihood": round(float(model_logit.llf), 1),
        "pseudo_r_squared": round(float(model_logit.prsquared), 4),
        "results_by_target": results,
        "recommended_floor": int(round(primary["floor_point"])),
        "recommended_floor_ci_lower": int(round(primary["floor_ci_lower"])),
        "recommended_floor_ci_upper": int(round(primary["floor_ci_upper"])),
    }


# =============================================================================
# PHASE 6: DIAGNOSTIC PLOTS
# =============================================================================


def generate_diagnostics(df_thresh: pd.DataFrame, df: pd.DataFrame, alpha_results: dict, floor_results: dict):
    """Generate supplementary diagnostic figure (4 panels)."""
    print("\nGenerating diagnostic figure...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {"trellis": "#1f77b4", "tree": "#ff7f0e", "linear": "#2ca02c"}

    # --- Panel A: log(eff_n) vs log(reach) with regression ---
    ax = axes[0, 0]
    for topo in df_thresh["topology"].unique():
        subset = df_thresh[df_thresh["topology"] == topo]
        ax.scatter(
            subset["log_reach"], subset["log_eff_n_interp"], c=colors.get(topo, "gray"), label=topo, alpha=0.7, s=50
        )

    # Regression line
    alpha = alpha_results["alpha_hat"]
    log_k = alpha_results["log_k_hat"]
    x_line = np.linspace(df_thresh["log_reach"].min(), df_thresh["log_reach"].max(), 100)
    ax.plot(
        x_line,
        log_k + alpha * x_line,
        "k--",
        linewidth=2,
        label=f"OLS: alpha={alpha:.3f} (R2={alpha_results['r_squared']:.3f})",
    )

    # Reference line for alpha=0.5
    # Use same intercept for fair comparison
    y = df_thresh["log_eff_n_interp"]
    # Fit constrained model with alpha=0.5
    residuals_05 = y - 0.5 * df_thresh["log_reach"]
    intercept_05 = residuals_05.mean()
    ax.plot(x_line, intercept_05 + 0.5 * x_line, "r:", linewidth=1.5, label="sqrt model: alpha=0.5")

    ax.set_xlabel("log(reach)")
    ax.set_ylabel("log(eff_n at threshold)")
    ax.set_title("A) Power Law Check (slope = alpha)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Residuals from alpha=0.5 ---
    ax = axes[0, 1]
    df_thresh["resid_sqrt"] = df_thresh["log_eff_n_interp"] - intercept_05 - 0.5 * df_thresh["log_reach"]
    for topo in df_thresh["topology"].unique():
        subset = df_thresh[df_thresh["topology"] == topo]
        ax.scatter(subset["log_reach"], subset["resid_sqrt"], c=colors.get(topo, "gray"), label=topo, alpha=0.7, s=50)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("log(reach)")
    ax.set_ylabel("Residual from sqrt model")
    ax.set_title("B) Residuals vs alpha=0.5 (trend = wrong exponent)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Distribution of implied k ---
    ax = axes[1, 0]
    df_thresh["k_implied"] = df_thresh["eff_n_interp"] / np.sqrt(df_thresh["reach"])
    ax.hist(df_thresh["k_implied"], bins=15, edgecolor="black", alpha=0.7, color="#0072B2")
    ax.axvline(
        df_thresh["k_implied"].mean(), color="blue", linestyle="-", label=f"Mean: {df_thresh['k_implied'].mean():.2f}"
    )
    ax.axvline(
        df_thresh["k_implied"].quantile(0.75),
        color="red",
        linestyle="--",
        label=f"75th: {df_thresh['k_implied'].quantile(0.75):.2f}",
    )
    ax.axvline(
        df_thresh["k_implied"].quantile(0.95),
        color="orange",
        linestyle=":",
        label=f"95th: {df_thresh['k_implied'].quantile(0.95):.2f}",
    )
    ax.set_xlabel("Implied k = eff_n / sqrt(reach)")
    ax.set_ylabel("Count")
    ax.set_title("C) Distribution of Implied k Values")
    ax.legend(fontsize=8)

    # --- Panel D: Logistic floor fit ---
    ax = axes[1, 1]
    df_b["success"] = (df_b["spearman"] >= TARGET_RHO).astype(int)

    # Bin for empirical rates
    eff_n_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000, 2000]
    df_b["eff_n_bin"] = pd.cut(df_b["effective_n"], bins=eff_n_bins)
    success_by_bin = df_b.groupby("eff_n_bin", observed=True)["success"].agg(["mean", "count"]).reset_index()
    success_by_bin["se"] = np.sqrt(success_by_bin["mean"] * (1 - success_by_bin["mean"]) / success_by_bin["count"])
    bin_centers = [(b.left + b.right) / 2 for b in success_by_bin["eff_n_bin"]]

    ax.errorbar(
        bin_centers,
        success_by_bin["mean"],
        yerr=1.96 * success_by_bin["se"],
        fmt="o",
        capsize=4,
        markersize=6,
        color="#0072B2",
        label="Empirical (binned)",
    )

    # Logistic curve
    beta0 = floor_results["beta0"]
    beta1 = floor_results["beta1"]
    eff_n_plot = np.linspace(0, 2000, 200)
    from scipy.special import expit

    p_success = expit(beta0 + beta1 * eff_n_plot)
    ax.plot(eff_n_plot, p_success, "r-", linewidth=2, label="Logistic fit")

    # Mark floor
    floor = floor_results["recommended_floor"]
    floor_ci_lo = floor_results["recommended_floor_ci_lower"]
    floor_ci_hi = floor_results["recommended_floor_ci_upper"]
    ax.axvline(
        floor, color="green", linestyle="--", linewidth=1.5, label=f"Floor: {floor} [{floor_ci_lo}, {floor_ci_hi}]"
    )
    ax.axhspan(0, 0.95, alpha=0.05, color="red")
    ax.axhline(0.95, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("Effective Sample Size (eff_n)")
    ax.set_ylabel("P(rho >= 0.95)")
    ax.set_title("D) Logistic Floor Fit with CI")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Supplementary: Model Validation Diagnostics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig_s1_diagnostics.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("08_validate_power_exponent.py")
    print("Power exponent validation and logistic floor fitting")
    print("=" * 70)

    # Load data
    df = load_synthetic_data()

    # Create threshold dataset
    df_thresh = create_threshold_dataset(df)

    # Phase 3: Fit power exponent
    alpha_results = fit_power_exponent(df_thresh)

    # Phase 3.3: Bootstrap CI
    boot_results = bootstrap_alpha(df_thresh)
    alpha_results.update(boot_results)

    # Decision on alpha
    if not alpha_results["reject_h0"] and alpha_results["ci_contains_half"]:
        alpha_results["decision"] = "Use alpha = 0.5 (sqrt model) for parsimony"
        alpha_results["final_alpha"] = 0.5
    else:
        alpha_results["decision"] = f"Use estimated alpha = {alpha_results['alpha_hat']}"
        alpha_results["final_alpha"] = alpha_results["alpha_hat"]

    print(f"\n>>> DECISION: {alpha_results['decision']}")

    # Phase 4: Logistic floor
    floor_results = fit_floor_logistic(df)

    # Phase 6: Diagnostics
    generate_diagnostics(df_thresh, df, alpha_results, floor_results)

    # Save combined results
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "08_validate_power_exponent.py",
        "power_exponent": {k: v for k, v in alpha_results.items() if k != "ols_summary"},
        "floor_logistic": floor_results,
    }

    output_path = OUTPUT_DIR / "power_exponent_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nPower exponent:")
    print(
        f"  alpha = {alpha_results['alpha_hat']} "
        f"(95% CI: [{alpha_results['boot_ci_lower']}, {alpha_results['boot_ci_upper']}])"
    )
    print(f"  H0: alpha = 0.5, p = {alpha_results['p_value_vs_half']}")
    print(f"  Decision: {alpha_results['decision']}")
    print("\nLogistic floor (95% success):")
    print(
        f"  min_eff_n = {floor_results['recommended_floor']} "
        f"(95% CI: [{floor_results['recommended_floor_ci_lower']}, {floor_results['recommended_floor_ci_upper']}])"
    )

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {output_path}")
    print(f"  2. {FIGURES_DIR / 'fig_s1_diagnostics.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
