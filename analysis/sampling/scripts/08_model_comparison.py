#!/usr/bin/env python
"""
08_model_comparison.py - Compare candidate models for eff_n prediction.

Compares 5 candidate functional forms using AIC, BIC, and cross-validation
to justify (or revise) the sqrt(reach) + floor model.

Candidate models:
    1. sqrt:        eff_n = k * sqrt(r)
    2. sqrt_floor:  eff_n = max(k * sqrt(r), m)
    3. power:       eff_n = k * r^alpha
    4. power_floor: eff_n = max(k * r^alpha, m)
    5. log:         eff_n = k1 + k2 * log(r)

Outputs:
    - output/model_comparison.json: Comparison metrics for all models
    - paper/tables/tab_model_comparison.tex: LaTeX table for paper/supplementary
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
from scipy.optimize import differential_evolution
from scipy.special import expit
from utilities import CACHE_DIR, CACHE_VERSION, OUTPUT_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

TARGET_RHO = 0.95

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
    df_b = df[df["metric"] == "betweenness"].copy()
    df_b["success"] = (df_b["spearman"] >= TARGET_RHO).astype(int)
    print(f"Loaded synthetic data: {len(df_b)} betweenness rows")
    return df_b


# =============================================================================
# CANDIDATE MODELS
# =============================================================================


def eff_n_sqrt(reach, params):
    """eff_n = k * sqrt(r)"""
    k = params[0]
    return k * np.sqrt(reach)


def eff_n_sqrt_floor(reach, params):
    """eff_n = max(k * sqrt(r), m)"""
    k, m = params
    return np.maximum(k * np.sqrt(reach), m)


def eff_n_power(reach, params):
    """eff_n = k * r^alpha"""
    k, alpha = params
    return k * np.power(reach, alpha)


def eff_n_power_floor(reach, params):
    """eff_n = max(k * r^alpha, m)"""
    k, alpha, m = params
    return np.maximum(k * np.power(reach, alpha), m)


def eff_n_log(reach, params):
    """eff_n = k1 + k2 * log(r)"""
    k1, k2 = params
    return k1 + k2 * np.log(reach)


MODELS = {
    "sqrt": {
        "name": "Square root",
        "formula": r"$k \sqrt{r}$",
        "func": eff_n_sqrt,
        "n_params": 1,
        "bounds": [(1, 50)],
        "x0": [10.0],
    },
    "sqrt_floor": {
        "name": "Sqrt + floor (proposed)",
        "formula": r"$\max(k\sqrt{r}, m)$",
        "func": eff_n_sqrt_floor,
        "n_params": 2,
        "bounds": [(1, 50), (50, 1000)],
        "x0": [10.0, 350.0],
    },
    "power": {
        "name": "General power",
        "formula": r"$k \, r^{\alpha}$",
        "func": eff_n_power,
        "n_params": 2,
        "bounds": [(0.1, 100), (0.1, 0.9)],
        "x0": [5.0, 0.5],
    },
    "power_floor": {
        "name": "Power + floor",
        "formula": r"$\max(k \, r^{\alpha}, m)$",
        "func": eff_n_power_floor,
        "n_params": 3,
        "bounds": [(0.1, 100), (0.1, 0.9), (50, 1000)],
        "x0": [5.0, 0.5, 350.0],
    },
    "log": {
        "name": "Logarithmic",
        "formula": r"$k_1 + k_2 \log r$",
        "func": eff_n_log,
        "n_params": 2,
        "bounds": [(-500, 500), (10, 500)],
        "x0": [-100.0, 80.0],
    },
}


# =============================================================================
# MODEL FITTING
# =============================================================================


def fit_model(model_key: str, df: pd.DataFrame) -> dict:
    """
    Fit a candidate model by minimising binary cross-entropy loss.

    For each observation, the model predicts eff_n at the observed reach,
    which maps to P(success) via an auxiliary logistic link fitted to data.
    We maximise the log-likelihood of the binary success outcomes.
    """
    model_info = MODELS[model_key]
    func = model_info["func"]
    bounds = model_info["bounds"]

    reach = df["mean_reach"].values
    eff_n_obs = df["effective_n"].values
    y = df["success"].values

    # First fit a global logistic link: P(success | eff_n) = logistic(a + b * eff_n)
    # This is shared across models as the observation model
    import statsmodels.api as sm

    X_logit = sm.add_constant(eff_n_obs)
    logit_model = sm.Logit(y, X_logit).fit(disp=0)
    a_link = float(logit_model.params[0])
    b_link = float(logit_model.params[1])

    def neg_log_likelihood(params):
        """Compute negative log-likelihood."""
        eff_n_pred = func(reach, params)
        # Clip to avoid numerical issues
        eff_n_pred = np.clip(eff_n_pred, 1, None)

        # Map model's eff_n at each observation's reach + probability to predicted success
        # The model predicts what eff_n *should* be at each reach, but the observation
        # has a specific eff_n. We need a different approach:
        # For each observation, the model prescribes a threshold eff_n for its reach.
        # The observation succeeds if its actual eff_n >= model's threshold.
        # Instead, directly use: P(success) = logistic(a + b * min(eff_n_obs, model_eff_n))
        # Actually, the cleaner approach: evaluate model quality by how well the
        # model-prescribed eff_n predicts actual success at each observation.

        # Simple approach: for each observation, compute the model's predicted probability
        # that this observation succeeds, using the logistic link with the observation's
        # actual eff_n. Then penalise models where the prescribed sampling would give
        # observations with eff_n < model threshold (meaning model says "don't sample this low").
        #
        # Best approach: treat each observation's predicted success as
        # P(success | eff_n_obs) from the shared logistic, and compare models on how well
        # their prescribed eff_n_threshold matches the actual transition.

        # Simplest valid approach: compute model's eff_n at each observation's reach,
        # convert to P(success) via logistic link, compute log-likelihood of observed y.
        p_model = expit(a_link + b_link * eff_n_pred)
        p_model = np.clip(p_model, 1e-10, 1 - 1e-10)

        # But we want to evaluate the model's *prescriptive* quality:
        # "At each reach, what sampling is needed?"
        # Use observed eff_n to assess whether model threshold is met
        # P(success) should be high when above threshold, low when below
        # Use logistic of the ratio: logistic(scale * (eff_n_obs / eff_n_pred - 1))
        ratio = eff_n_obs / eff_n_pred
        p_success = expit(5.0 * (ratio - 0.8))  # Smooth step at 80% of threshold
        p_success = np.clip(p_success, 1e-10, 1 - 1e-10)

        ll = np.sum(y * np.log(p_success) + (1 - y) * np.log(1 - p_success))
        return -ll

    # Actually, let's use a simpler, more standard approach.
    # For model comparison, we evaluate how well each model's predicted eff_n
    # (at each reach) explains the observed rho values.
    # Use MSE on the threshold dataset instead.

    # Revert to threshold-based fitting: fit to the minimum eff_n achieving target
    # This is more interpretable and standard.

    def mse_loss(params):
        """MSE between model-predicted eff_n and observed threshold eff_n."""
        # We need the threshold dataset. Use a proxy: for each (topology, distance),
        # find the observation closest to the threshold.
        eff_n_pred = func(reach, params)
        # For each observation, compute predicted P(success) via logistic link
        p_pred = expit(a_link + b_link * eff_n_pred)
        p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)
        # Binary cross-entropy
        ll = np.sum(y * np.log(p_pred) + (1 - y) * np.log(1 - p_pred))
        return -ll

    # Fit using differential evolution for global optimisation
    result = differential_evolution(mse_loss, bounds, seed=42, maxiter=1000, tol=1e-8)

    n = len(df)
    k_params = model_info["n_params"]
    ll = -result.fun
    aic = 2 * k_params - 2 * ll
    bic = k_params * np.log(n) - 2 * ll

    print(f"  {model_key:20s}: LL={ll:.1f}, AIC={aic:.1f}, BIC={bic:.1f}, params={[round(p, 3) for p in result.x]}")

    return {
        "model": model_key,
        "name": model_info["name"],
        "formula": model_info["formula"],
        "params": [round(float(p), 4) for p in result.x],
        "n_params": k_params,
        "log_likelihood": round(float(ll), 1),
        "AIC": round(float(aic), 1),
        "BIC": round(float(bic), 1),
        "n_obs": n,
    }


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


def compute_cv_accuracy(model_key: str, df: pd.DataFrame, n_folds: int = 3) -> dict:
    """
    Leave-one-topology-out cross-validation.

    Since we only have 3 topologies, use each as a held-out fold.
    Measure: fraction of test observations where model correctly predicts success/failure.
    """
    model_info = MODELS[model_key]
    func = model_info["func"]
    bounds = model_info["bounds"]

    topologies = df["topology"].unique()
    fold_results = []

    for held_out in topologies:
        train = df[df["topology"] != held_out]
        test = df[df["topology"] == held_out]

        reach_train = train["mean_reach"].values
        eff_n_train = train["effective_n"].values
        y_train = train["success"].values

        # Fit logistic link on training data
        import statsmodels.api as sm

        X_logit = sm.add_constant(eff_n_train)
        logit_model = sm.Logit(y_train, X_logit).fit(disp=0)
        a_link = float(logit_model.params[0])
        b_link = float(logit_model.params[1])

        def neg_ll(params, _reach=reach_train, _a=a_link, _b=b_link, _y=y_train):
            eff_n_pred = func(_reach, params)
            p_pred = expit(_a + _b * eff_n_pred)
            p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)
            return -np.sum(_y * np.log(p_pred) + (1 - _y) * np.log(1 - p_pred))

        result = differential_evolution(neg_ll, bounds, seed=42, maxiter=500, tol=1e-8)

        # Evaluate on test set
        reach_test = test["mean_reach"].values
        y_test = test["success"].values

        eff_n_pred_test = func(reach_test, result.x)
        p_pred_test = expit(a_link + b_link * eff_n_pred_test)

        # Brier score (lower is better)
        brier = float(np.mean((p_pred_test - y_test) ** 2))

        # Log-likelihood on test
        p_clipped = np.clip(p_pred_test, 1e-10, 1 - 1e-10)
        test_ll = float(np.sum(y_test * np.log(p_clipped) + (1 - y_test) * np.log(1 - p_clipped)))
        test_ll_per_obs = test_ll / len(y_test)

        fold_results.append(
            {
                "held_out": held_out,
                "brier_score": brier,
                "test_ll_per_obs": test_ll_per_obs,
                "n_test": len(y_test),
            }
        )

    mean_brier = np.mean([r["brier_score"] for r in fold_results])
    std_brier = np.std([r["brier_score"] for r in fold_results])
    mean_ll = np.mean([r["test_ll_per_obs"] for r in fold_results])

    return {
        "cv_brier_mean": round(float(mean_brier), 4),
        "cv_brier_std": round(float(std_brier), 4),
        "cv_ll_per_obs_mean": round(float(mean_ll), 4),
        "fold_results": fold_results,
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================


def create_comparison_table(results: dict) -> str:
    """Create LaTeX table comparing all models."""

    # Sort by AIC
    model_keys = sorted(results.keys(), key=lambda k: results[k]["AIC"])
    best_aic = results[model_keys[0]]["AIC"]

    rows = []
    for key in model_keys:
        r = results[key]
        delta_aic = r["AIC"] - best_aic
        rows.append(
            {
                "Model": r["name"],
                "Formula": r["formula"],
                "n_params": r["n_params"],
                "AIC": r["AIC"],
                "BIC": r["BIC"],
                "Delta_AIC": delta_aic,
                "CV_Brier": r.get("cv_brier_mean", ""),
            }
        )

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model comparison for effective sample size prediction.
Models ranked by AIC; $\Delta$AIC $< 2$ indicates substantial support.}
\label{tab:model_comparison}
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Formula} & $n_p$ & \textbf{AIC} & $\Delta$\textbf{AIC} & \textbf{CV Brier} \\
\midrule
"""

    for row in rows:
        cv_str = f"{row['CV_Brier']:.4f}" if isinstance(row["CV_Brier"], float) else "--"
        latex += (
            f"{row['Model']} & {row['Formula']} & {row['n_params']}"
            f" & {row['AIC']:.1f} & {row['Delta_AIC']:.1f} & {cv_str} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
$n_p$: number of fitted parameters. CV Brier: leave-one-topology-out cross-validated Brier score (lower is better).\\
$\Delta$AIC $< 2$: substantial support; $2$--$10$: some support; $> 10$: essentially no support.
\end{table}
"""

    return latex


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("08_model_comparison.py - Comparing candidate sampling models")
    print("=" * 70)

    # Load data
    df = load_synthetic_data()

    # Fit all models
    print("\nFitting models...")
    print("-" * 70)
    fit_results = {}
    for model_key in MODELS:
        fit_results[model_key] = fit_model(model_key, df)

    # Cross-validation
    print("\nCross-validation (leave-one-topology-out)...")
    print("-" * 70)
    for model_key in MODELS:
        cv_result = compute_cv_accuracy(model_key, df)
        fit_results[model_key].update(cv_result)
        print(f"  {model_key:20s}: Brier={cv_result['cv_brier_mean']:.4f} +/- {cv_result['cv_brier_std']:.4f}")

    # Compute delta AIC and weights
    all_aics = {k: v["AIC"] for k, v in fit_results.items()}
    best_aic = min(all_aics.values())
    for key in fit_results:
        delta = fit_results[key]["AIC"] - best_aic
        fit_results[key]["delta_AIC"] = round(float(delta), 1)

    # AIC weights
    deltas = np.array([fit_results[k]["delta_AIC"] for k in fit_results])
    weights = np.exp(-0.5 * deltas)
    weights = weights / weights.sum()
    for i, key in enumerate(fit_results):
        fit_results[key]["AIC_weight"] = round(float(weights[i]), 4)

    # Model selection summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'AIC':>8} {'dAIC':>8} {'BIC':>8} {'Weight':>8} {'CV Brier':>10}")
    print("-" * 75)

    for key in sorted(fit_results, key=lambda k: fit_results[k]["AIC"]):
        r = fit_results[key]
        print(
            f"{r['name']:<25} {r['AIC']:>8.1f} {r['delta_AIC']:>8.1f} {r['BIC']:>8.1f} "
            f"{r['AIC_weight']:>8.4f} {r.get('cv_brier_mean', 0):>10.4f}"
        )

    # Check if proposed model (sqrt_floor) is well-supported
    proposed = fit_results["sqrt_floor"]
    best_key = min(fit_results, key=lambda k: fit_results[k]["AIC"])

    print(f"\nBest model by AIC: {fit_results[best_key]['name']}")
    print(f"Proposed model (sqrt+floor) delta AIC: {proposed['delta_AIC']}")

    if proposed["delta_AIC"] < 2:
        print(">>> sqrt+floor has SUBSTANTIAL support (delta AIC < 2)")
    elif proposed["delta_AIC"] < 10:
        print(">>> sqrt+floor has SOME support (2 <= delta AIC < 10)")
    else:
        print(f">>> sqrt+floor has MINIMAL support; consider {fit_results[best_key]['name']}")

    # Generate LaTeX table
    latex = create_comparison_table(fit_results)
    table_path = TABLES_DIR / "tab_model_comparison.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"\nSaved: {table_path}")

    # Save full results
    # Clean up fold_results for JSON serialisation
    clean_results = {}
    for key, val in fit_results.items():
        clean_val = {k: v for k, v in val.items() if k != "fold_results"}
        clean_results[key] = clean_val

    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "08_model_comparison.py",
        "best_model": best_key,
        "proposed_model_support": "substantial"
        if proposed["delta_AIC"] < 2
        else "some"
        if proposed["delta_AIC"] < 10
        else "minimal",
        "models": clean_results,
    }

    output_path = OUTPUT_DIR / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {output_path}")
    print(f"  2. {table_path}")

    return 0


if __name__ == "__main__":
    exit(main())
