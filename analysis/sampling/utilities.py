"""
Utility Functions for Sampling Analysis

Contains helper functions for cache inspection, data export, and exploratory analysis.

Functions:
- export_cache_to_json(): Export pickle cache to JSON format
- inspect_cache(): Inspect and debug cache structure
- run_extended_model_analysis(): Explore extended accuracy models

Consolidated from:
- export_cache_to_json.py
- inspect_cache.py
- extended_model_analysis.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize as scipy_optimize

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR.parent.parent / "temp" / "sampling_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"


# =============================================================================
# SECTION 1: Cache Export
# =============================================================================


def export_cache_to_json():
    """
    Export sampling analysis cache from pickle to JSON format.

    Converts the versioned pickle cache files to a single JSON file
    for easier inspection and cross-platform compatibility.
    """
    print("=" * 70)
    print("EXPORT CACHE TO JSON")
    print("=" * 70)

    shortest_file = CACHE_DIR / "sampling_analysis_v7.pkl"
    angular_file = CACHE_DIR / "sampling_analysis_angular_v7.pkl"

    if not shortest_file.exists():
        print(f"Error: {shortest_file} not found")
        return

    print(f"Loading: {shortest_file}")
    with open(shortest_file, "rb") as f:
        results_shortest = pickle.load(f)
    print(f"  Loaded {len(results_shortest)} shortest path results")

    if angular_file.exists():
        print(f"Loading: {angular_file}")
        with open(angular_file, "rb") as f:
            results_angular = pickle.load(f)
        print(f"  Loaded {len(results_angular)} angular results")
    else:
        print(f"Note: {angular_file} not found, skipping angular results")
        results_angular = []

    cache_data = {
        "results": results_shortest,
        "results_angular": results_angular,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_file = OUTPUT_DIR / "sampling_cache_v7.json"
    print(f"\nExporting to: {json_file}")
    with open(json_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Exported successfully!")
    print(f"File size: {json_file.stat().st_size / 1024:.1f} KB")


# =============================================================================
# SECTION 2: Cache Inspection
# =============================================================================


def inspect_cache(cache_name: str = "sampling_analysis_v7"):
    """
    Inspect the sampling analysis cache structure.

    Parameters
    ----------
    cache_name
        Name of the cache file (without .pkl extension)
    """
    print("=" * 70)
    print("INSPECT CACHE")
    print("=" * 70)

    pickle_file = CACHE_DIR / f"{cache_name}.pkl"
    print(f"Loading: {pickle_file}")

    if not pickle_file.exists():
        print(f"Error: {pickle_file} not found")
        return

    with open(pickle_file, "rb") as f:
        cache_data = pickle.load(f)

    print(f"\nCache type: {type(cache_data)}")

    if isinstance(cache_data, list):
        print(f"Length: {len(cache_data)}")
        if len(cache_data) > 0:
            print(f"First item type: {type(cache_data[0])}")
            if isinstance(cache_data[0], dict):
                print(f"First item keys: {cache_data[0].keys()}")
                print(f"\nSample entry (first item):")
                for k, v in cache_data[0].items():
                    print(f"  {k}: {v}")
    elif isinstance(cache_data, dict):
        print(f"Cache keys: {cache_data.keys()}")
        for key, value in cache_data.items():
            print(f"\nKey: {key}")
            print(f"  Type: {type(value)}")
            if isinstance(value, list):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"  First item keys: {value[0].keys()}")
                        print(f"  Sample entry:")
                        for k, v in list(value[0].items())[:5]:
                            print(f"    {k}: {v}")


# =============================================================================
# SECTION 3: Extended Model Analysis
# =============================================================================

# Model definitions


def _model_baseline(params, eff_n, p, reach):
    """Current model: rho = 1 - A / (B + eff_n)"""
    A, B = params
    return 1 - A / (B + eff_n)


def _model_additive(params, eff_n, p, reach):
    """Additive p penalty: rho = 1 - A / (B + eff_n) - C * (1 - p)"""
    A, B, C = params
    return 1 - A / (B + eff_n) - C * (1 - p)


def _model_multiplicative_k(params, eff_n, p, reach):
    """Multiplicative on eff_n: rho = 1 - A / (B + eff_n * p^k)"""
    A, B, k = params
    effective_adjusted = eff_n * (p**k)
    return 1 - A / (B + effective_adjusted)


def _model_log_p(params, eff_n, p, reach):
    """Log p penalty: rho = 1 - A / (B + eff_n) + C * log(p)"""
    A, B, C = params
    return 1 - A / (B + eff_n) + C * np.log(np.maximum(p, 0.01))


def _model_dual_terms(params, eff_n, p, reach):
    """Dual penalty terms: rho = 1 - A / (B + eff_n) - C / (D + p * 100)"""
    A, B, C, D = params
    return 1 - A / (B + eff_n) - C / (D + p * 100)


def _model_power_law(params, eff_n, p, reach):
    """Power law in both: rho = 1 - A / (B + eff_n^a * p^b)"""
    A, B, a, b = params
    effective_adjusted = (eff_n**a) * (p**b)
    return 1 - A / (B + effective_adjusted)


def _fit_model(model_func, initial_params, eff_n, p, reach, rho, bounds=None, param_names=None):
    """Fit model to data using least squares."""

    def residual(params):
        pred = model_func(params, eff_n, p, reach)
        pred = np.clip(pred, 0, 1)
        return np.sum((pred - rho) ** 2)

    if bounds is None:
        result = scipy_optimize.minimize(residual, initial_params, method="Nelder-Mead", options={"maxiter": 5000})
    else:
        result = scipy_optimize.minimize(
            residual, initial_params, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000}
        )

    fitted_params = result.x

    predictions = model_func(fitted_params, eff_n, p, reach)
    predictions = np.clip(predictions, 0, 1)

    residuals = rho - predictions
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((rho - np.mean(rho)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    n = len(rho)
    k = len(fitted_params)
    aic = n * np.log(ss_res / n + 1e-10) + 2 * k

    return {
        "params": fitted_params,
        "param_names": param_names or [f"p{i}" for i in range(len(fitted_params))],
        "predictions": predictions,
        "rmse": rmse,
        "r_squared": r_squared,
        "aic": aic,
        "n_params": k,
    }


def run_extended_model_analysis():
    """
    Run extended accuracy model analysis.

    Explores models that include sampling probability p as a second factor
    beyond just effective sample size (eff_n = reachability * p).

    The hypothesis is that at constant eff_n, accuracy degrades as p decreases
    (and reachability increases), suggesting eff_n alone may not be sufficient.
    """
    print("=" * 70)
    print("EXTENDED ACCURACY MODEL ANALYSIS")
    print("=" * 70)

    # Load data
    cache_file = OUTPUT_DIR / "sampling_cache_v7.json"
    if not cache_file.exists():
        print(f"Error: {cache_file} not found")
        print("Run export_cache_to_json() first to create the JSON cache")
        return

    with open(cache_file) as f:
        cache_data = json.load(f)

    df = pd.DataFrame(cache_data["results"])
    print(f"\nLoaded {len(df)} data points from cache")

    # Filter data
    df_sampled = df[(df["sample_prob"] < 1.0) & (df["effective_n"] > 5) & (df["spearman"] > 0.05)].copy()
    print(f"After filtering (p < 1.0, eff_n > 5, rho > 0.05): {len(df_sampled)} points")

    df_sampled["reach"] = df_sampled["mean_reach"]
    df_sampled["p"] = df_sampled["sample_prob"]
    df_sampled["rho"] = df_sampled["spearman"]

    df_harmonic = df_sampled[df_sampled["metric"] == "harmonic"].copy()
    df_betweenness = df_sampled[df_sampled["metric"] == "betweenness"].copy()

    print(f"\nHarmonic points: {len(df_harmonic)}")
    print(f"Betweenness points: {len(df_betweenness)}")

    # Define models to test
    models = [
        ("baseline", _model_baseline, [50, 50], ["A", "B"], [(0.1, 500), (0.1, 500)]),
        ("additive", _model_additive, [50, 50, 0.1], ["A", "B", "C"], [(0.1, 500), (0.1, 500), (-1, 1)]),
        ("multiplicative", _model_multiplicative_k, [50, 50, 0.5], ["A", "B", "k"], [(0.1, 500), (0.1, 500), (-2, 2)]),
        ("log_p", _model_log_p, [50, 50, 0.1], ["A", "B", "C"], [(0.1, 500), (0.1, 500), (-0.5, 0.5)]),
        (
            "dual_terms",
            _model_dual_terms,
            [50, 50, 5, 50],
            ["A", "B", "C", "D"],
            [(0.1, 500), (0.1, 500), (-50, 50), (0.1, 500)],
        ),
        (
            "power_law",
            _model_power_law,
            [50, 50, 1.0, 0.5],
            ["A", "B", "a", "b"],
            [(0.1, 500), (0.1, 500), (0.1, 2), (-1, 2)],
        ),
    ]

    # Fit models for each metric
    for metric_name, df_metric in [("Harmonic", df_harmonic), ("Betweenness", df_betweenness)]:
        print(f"\n{'=' * 70}")
        print(f"METRIC: {metric_name.upper()}")
        print("=" * 70)

        eff_n = df_metric["effective_n"].values
        p = df_metric["p"].values
        reach = df_metric["reach"].values
        rho = df_metric["rho"].values

        results = {}

        for name, model_func, init_params, param_names, bounds in models:
            print(f"\nFitting {name} model...")
            res = _fit_model(model_func, init_params, eff_n, p, reach, rho, bounds=bounds, param_names=param_names)
            results[name] = res

            params_str = ", ".join([f"{n}={v:.4f}" for n, v in zip(param_names, res["params"])])
            print(f"  {params_str}")
            print(f"  RMSE = {res['rmse']:.4f}, R² = {res['r_squared']:.4f}, AIC = {res['aic']:.1f}")

        # Summary comparison
        print(f"\n{'=' * 70}")
        print(f"MODEL COMPARISON SUMMARY - {metric_name}")
        print("=" * 70)
        print(f"{'Model':<20} {'RMSE':<10} {'R²':<10} {'AIC':<12} {'Params':<8}")
        print("-" * 60)

        sorted_results = sorted(results.items(), key=lambda x: x[1]["aic"])
        for name, res in sorted_results:
            print(f"{name:<20} {res['rmse']:<10.4f} {res['r_squared']:<10.4f} {res['aic']:<12.1f} {res['n_params']:<8}")

        print("-" * 60)
        best_model = sorted_results[0]
        print(f"Best model by AIC: {best_model[0]}")

    # Generate visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, (metric_name, df_metric) in enumerate([("Harmonic", df_harmonic), ("Betweenness", df_betweenness)]):
        eff_n = df_metric["effective_n"].values
        p = df_metric["p"].values
        reach = df_metric["reach"].values
        rho = df_metric["rho"].values

        res_baseline = _fit_model(
            _model_baseline,
            initial_params=[50, 50],
            eff_n=eff_n,
            p=p,
            reach=reach,
            rho=rho,
            param_names=["A", "B"],
            bounds=[(1, 500), (1, 500)],
        )

        # Left plot: Data with baseline model
        ax = axes[row, 0]
        colors_p = plt.cm.viridis(np.linspace(0, 1, len(df_metric["p"].unique())))
        p_colors = dict(zip(sorted(df_metric["p"].unique()), colors_p))

        for p_val in sorted(df_metric["p"].unique()):
            mask = df_metric["p"] == p_val
            ax.scatter(
                df_metric[mask]["effective_n"],
                df_metric[mask]["rho"],
                c=[p_colors[p_val]],
                label=f"p={p_val:.2f}",
                alpha=0.6,
                s=20,
            )

        eff_n_curve = np.linspace(5, max(eff_n), 100)
        pred_baseline = _model_baseline(res_baseline["params"], eff_n_curve, 0.5, 500)
        ax.plot(eff_n_curve, pred_baseline, "k-", linewidth=2, label=f"Baseline (RMSE={res_baseline['rmse']:.3f})")

        ax.set_xlabel("Effective Sample Size (eff_n)")
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"{metric_name}: rho vs eff_n (coloured by p)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Right plot: Residuals by p
        ax = axes[row, 1]
        residuals = rho - res_baseline["predictions"]

        p_vals = sorted(df_metric["p"].unique())
        positions = np.arange(len(p_vals))
        means = []
        stds = []
        for p_val in p_vals:
            mask = df_metric["p"] == p_val
            res_p = residuals[mask.values]
            means.append(np.mean(res_p))
            stds.append(np.std(res_p))

        ax.bar(positions, means, yerr=stds, alpha=0.7, capsize=3)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{pv:.2f}" for pv in p_vals], rotation=45)
        ax.set_xlabel("Sampling Probability (p)")
        ax.set_ylabel("Residual (observed - predicted)")
        ax.set_title(f"{metric_name}: Baseline Model Residuals by p")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "extended_model_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("EXTENDED MODEL ANALYSIS COMPLETE")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "export":
            export_cache_to_json()
        elif command == "inspect":
            cache_name = sys.argv[2] if len(sys.argv) > 2 else "sampling_analysis_v7"
            inspect_cache(cache_name)
        elif command == "extended":
            run_extended_model_analysis()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python utilities.py [export|inspect|extended]")
    else:
        print("Sampling Analysis Utilities")
        print("=" * 70)
        print("\nAvailable functions:")
        print("  export_cache_to_json() - Export pickle cache to JSON")
        print("  inspect_cache()        - Inspect cache structure")
        print("  run_extended_model_analysis() - Run extended model analysis")
        print("\nCommand line usage:")
        print("  python utilities.py export")
        print("  python utilities.py inspect [cache_name]")
        print("  python utilities.py extended")
