# %% [markdown]
# # Adaptive Sampling Analysis
#
# Compares full computation, uniform sampling, and adaptive per-distance sampling
# for network centrality analysis. Tests on both synthetic and real-world networks.
#
# Also includes spatial evaluation to verify accuracy holds across all areas,
# particularly in low-reachability regions where effective sample size is lower.
#
# Key questions:
# 1. Does adaptive sampling achieve target accuracy (rho >= 0.95)?
# 2. What speedup does adaptive sampling provide?
# 3. How does it compare to uniform sampling with equivalent compute budget?
# 4. Does accuracy hold spatially in low-reachability areas?

# %% Imports
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer import config
from cityseer.metrics import networks
from cityseer.tools import io
from scipy import stats as scipy_stats
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# %% Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PAPER_DIR = SCRIPT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
CACHE_DIR = SCRIPT_DIR / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Import substrate utilities
import sys

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.substrates import generate_keyed_template

# Cache version
CACHE_VERSION = "v2"

# Synthetic network configuration
TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 5
SYNTHETIC_DISTANCES = [500, 1000, 2000, 5000]

# Real-world network configuration
CITIES = {
    "london": {
        "name": "London (Soho)",
        "lng": -0.13396079424572427,
        "lat": 51.51371088849723,
        "buffer": 2000,
        "description": "Dense, irregular historical street pattern",
    },
    "madrid": {
        "name": "Madrid (Centro)",
        "lng": -3.7037902,
        "lat": 40.4167754,
        "buffer": 2000,
        "description": "Mediterranean grid with radial elements",
    },
    "phoenix": {
        "name": "Phoenix (Scottsdale)",
        "lng": -111.9261,
        "lat": 33.4942,
        "buffer": 2000,
        "description": "American suburban sprawl with cul-de-sacs",
    },
}
REALWORLD_DISTANCES = [200, 500, 1000, 2000, 5000]
REALWORLD_PROBS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_RUNS = 10

# Model parameters from sampling_reach.py
MODEL_HARMONIC_A = 32.40
MODEL_HARMONIC_B = 31.54
MODEL_BETWEENNESS_A = 48.31
MODEL_BETWEENNESS_B = 49.12

# Target accuracy
TARGET_RHO = 0.95


# %% Utility functions
def load_cache(name: str):
    """Load cached results if available."""
    path = CACHE_DIR / f"{name}_{CACHE_VERSION}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            path.unlink()
            return None
    return None


def save_cache(name: str, data):
    """Save results to cache."""
    path = CACHE_DIR / f"{name}_{CACHE_VERSION}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def rho_model(eff_n: float, a: float, b: float) -> float:
    """Predict Spearman rho from effective sample size."""
    return 1 - a / (b + eff_n)


def compute_accuracy(true_vals: np.ndarray, est_vals: np.ndarray) -> float:
    """Compute Spearman correlation between true and estimated values."""
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan
    rho, _ = scipy_stats.spearmanr(true_vals[mask], est_vals[mask])
    return rho


# %% Part 1: Synthetic Network Comparison
def run_synthetic_comparison(template_key: str, distances: list[int], target_rho: float):
    """
    Run full vs uniform vs adaptive comparison on a synthetic network.

    Returns dict with timing and accuracy results.
    """
    print(f"\n{'=' * 70}")
    print(f"TOPOLOGY: {template_key.upper()}")
    print(f"{'=' * 70}")

    # Generate network
    print(f"\nGenerating {template_key} substrate (tiles={SUBSTRATE_TILES})...")
    G, _, _ = generate_keyed_template(template_key=template_key, tiles=SUBSTRATE_TILES, decompose=None, plot=False)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
    print(f"Network: {network_structure.node_count()} nodes, {network_structure.edge_count} edges")

    # Probe reachability
    print("\nProbing reachability...")
    reach_estimates = config.probe_reachability(network_structure, distances)
    for d in distances:
        print(f"  {d}m: reach = {reach_estimates[d]:.0f}")

    # Compute adaptive sampling plan
    sample_probs = config.compute_sample_probs_for_target_rho(reach_estimates, target_rho, metric="both")

    # Calculate weighted mean p for uniform comparison
    total_reach = sum(reach_estimates.values())
    weighted_p = (
        sum((sample_probs.get(d) or 1.0) * reach_estimates[d] for d in distances) / total_reach
        if total_reach > 0
        else 0.5
    )

    print("\nAdaptive sampling plan:")
    for d in distances:
        p = sample_probs.get(d)
        print(f"  {d}m: p = {'full' if p is None or p >= 1.0 else f'{p:.0%}'}")
    print(f"  Mean p (reach-weighted): {weighted_p:.0%}")

    # 1. Full computation
    print("\n--- Full Computation ---")
    t0 = time.perf_counter()
    nodes_full = networks.node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    time_full = time.perf_counter() - t0
    print(f"Time: {time_full:.2f}s")

    # 2. Uniform sampling
    print(f"\n--- Uniform Sampling (p={weighted_p:.0%}) ---")
    t0 = time.perf_counter()
    nodes_uniform = networks.node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
        sample_probability=weighted_p,
        random_seed=42,
    )
    time_uniform = time.perf_counter() - t0
    print(f"Time: {time_uniform:.2f}s (speedup: {time_full / time_uniform:.1f}x)")

    # 3. Adaptive sampling
    print(f"\n--- Adaptive Sampling (target rho >= {target_rho}) ---")
    t0 = time.perf_counter()
    nodes_adaptive = networks.node_centrality_shortest_adaptive(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        target_rho=target_rho,
        compute_closeness=True,
        compute_betweenness=True,
        random_seed=42,
    )
    time_adaptive = time.perf_counter() - t0
    print(f"Time: {time_adaptive:.2f}s (speedup: {time_full / time_adaptive:.1f}x)")

    # Compute accuracy per distance
    uniform_acc = {"harmonic": {}, "betweenness": {}}
    adaptive_acc = {"harmonic": {}, "betweenness": {}}

    print("\nAccuracy comparison:")
    print(f"{'Distance':<10} {'Uniform H':<12} {'Uniform B':<12} {'Adaptive H':<12} {'Adaptive B':<12}")
    print("-" * 58)

    for d in distances:
        for metric, key_fmt in [("harmonic", "cc_harmonic_{}"), ("betweenness", "cc_betweenness_{}")]:
            key = key_fmt.format(d)
            full_vals = nodes_full[key].values
            uniform_vals = nodes_uniform[key].values
            adaptive_vals = nodes_adaptive[key].values

            uniform_acc[metric][d] = compute_accuracy(full_vals, uniform_vals)
            adaptive_acc[metric][d] = compute_accuracy(full_vals, adaptive_vals)

        print(
            f"{d}m        {uniform_acc['harmonic'][d]:.3f}        {uniform_acc['betweenness'][d]:.3f}        "
            f"{adaptive_acc['harmonic'][d]:.3f}        {adaptive_acc['betweenness'][d]:.3f}"
        )

    return {
        "topology": template_key,
        "n_nodes": network_structure.node_count(),
        "time_full": time_full,
        "time_uniform": time_uniform,
        "time_adaptive": time_adaptive,
        "uniform_p": weighted_p,
        "uniform_acc": uniform_acc,
        "adaptive_acc": adaptive_acc,
        "reach_estimates": reach_estimates,
        "sample_probs": sample_probs,
    }


# %% Part 2: Real-World Network Validation
def download_network(city_key: str, force: bool = False):
    """Download OSM network for a city."""
    city = CITIES[city_key]
    cache_key = f"network_{city_key}"

    if not force:
        cached = load_cache(cache_key)
        if cached is not None:
            print(f"  Loaded {city['name']} network from cache")
            network_structure = io.network_structure_from_gpd(cached["nodes_gdf"], cached["edges_gdf"])
            cached["network_structure"] = network_structure
            return cached

    print(f"  Downloading {city['name']} network from OSM...")

    poly_wgs, _ = io.buffered_point_poly(city["lng"], city["lat"], city["buffer"])
    G = io.osm_graph_from_poly(poly_wgs, simplify=True, final_clean_distances=(4, 8), remove_disconnected=100)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)

    cache_data = {
        "nodes_gdf": nodes_gdf,
        "edges_gdf": edges_gdf,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }
    save_cache(cache_key, cache_data)

    result = {**cache_data, "network_structure": network_structure}
    print(f"  Downloaded: {result['n_nodes']} nodes, {result['n_edges']} edges")
    return result


def run_realworld_validation(city_key: str, force_download: bool = False, force_compute: bool = False):
    """Run validation experiment for a city."""
    city = CITIES[city_key]
    cache_key = f"validation_{city_key}"

    if not force_compute:
        cached = load_cache(cache_key)
        if cached is not None:
            print(f"  Loaded {city['name']} validation from cache")
            return cached

    print(f"\nValidating on {city['name']}...")
    network_data = download_network(city_key, force=force_download)
    net = network_data["network_structure"]

    results = []

    for dist in REALWORLD_DISTANCES:
        print(f"  Distance {dist}m: computing ground truth...")
        true_result = net.local_node_centrality_shortest(
            distances=[dist], compute_closeness=True, compute_betweenness=True, pbar_disabled=True
        )
        true_harmonic = np.array(true_result.node_harmonic[dist])
        true_betweenness = np.array(true_result.node_betweenness[dist])
        mean_reach = float(np.mean(np.array(true_result.node_density[dist])))

        if mean_reach < 5:
            print(f"    Skipping (low reachability: {mean_reach:.0f})")
            continue

        print(f"    Mean reachability: {mean_reach:.0f}")

        for p in REALWORLD_PROBS:
            if p == 1.0:
                for metric in ["harmonic", "betweenness"]:
                    results.append(
                        {
                            "city": city_key,
                            "distance": dist,
                            "mean_reach": mean_reach,
                            "sample_prob": p,
                            "effective_n": mean_reach,
                            "metric": metric,
                            "spearman": 1.0,
                            "spearman_std": 0.0,
                        }
                    )
                continue

            metrics_data = {"harmonic": [], "betweenness": []}
            true_vals = {"harmonic": true_harmonic, "betweenness": true_betweenness}

            for seed in range(N_RUNS):
                r = net.local_node_centrality_shortest(
                    distances=[dist],
                    compute_closeness=True,
                    compute_betweenness=True,
                    sample_probability=p,
                    random_seed=seed,
                    pbar_disabled=True,
                )
                est_vals = {
                    "harmonic": np.array(r.node_harmonic[dist]),
                    "betweenness": np.array(r.node_betweenness[dist]),
                }
                for metric in ["harmonic", "betweenness"]:
                    sp = compute_accuracy(true_vals[metric], est_vals[metric])
                    if not np.isnan(sp):
                        metrics_data[metric].append(sp)

            for metric in ["harmonic", "betweenness"]:
                if metrics_data[metric]:
                    results.append(
                        {
                            "city": city_key,
                            "distance": dist,
                            "mean_reach": mean_reach,
                            "sample_prob": p,
                            "effective_n": mean_reach * p,
                            "metric": metric,
                            "spearman": np.mean(metrics_data[metric]),
                            "spearman_std": np.std(metrics_data[metric]),
                        }
                    )

    save_cache(cache_key, results)
    return results


# %% Generate Figures
def generate_realworld_figure(all_results: list[dict]):
    """Generate real-world validation figure."""
    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"london": "#1f77b4", "madrid": "#d62728", "phoenix": "#2ca02c"}

    for col, metric in enumerate(["harmonic", "betweenness"]):
        ax = axes[col]
        metric_df = df[df["metric"] == metric]
        sampled = metric_df[metric_df["sample_prob"] < 1.0]

        if metric == "harmonic":
            a, b = MODEL_HARMONIC_A, MODEL_HARMONIC_B
        else:
            a, b = MODEL_BETWEENNESS_A, MODEL_BETWEENNESS_B

        # Plot by city
        for city_key in sampled["city"].unique():
            city_df = sampled[sampled["city"] == city_key]
            ax.scatter(
                city_df["effective_n"],
                city_df["spearman"],
                c=colors.get(city_key, "gray"),
                alpha=0.6,
                s=40,
                label=CITIES[city_key]["name"],
            )

        # Model curve
        eff_n_range = np.linspace(10, max(sampled["effective_n"].max() * 1.1, 500), 100)
        pred_rho = rho_model(eff_n_range, a, b)
        ax.plot(eff_n_range, pred_rho, "k-", linewidth=2, label="Model prediction")

        ax.axhline(y=0.95, color="green", linestyle=":", alpha=0.7, label="rho = 0.95")
        ax.axhline(y=0.90, color="orange", linestyle=":", alpha=0.7, label="rho = 0.90")

        ax.set_xlabel("Effective Sample Size")
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"{metric.title()}")
        ax.set_xlim(0, sampled["effective_n"].max() * 1.1 if len(sampled) > 0 else 500)
        ax.set_ylim(0.5, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Real-World Validation: Observed vs Predicted Accuracy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "realworld_validation.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'realworld_validation.pdf'}")


# %% Generate LaTeX Tables
def generate_latex_tables(synthetic_results: list[dict], realworld_results: list[dict], network_stats: dict):
    """Generate LaTeX tables for the paper."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Synthetic results table
    synthetic_tex = f"""% Auto-generated table: Synthetic Network Comparison
% Generated by adaptive_sampling.py on {timestamp}
% DO NOT EDIT MANUALLY - regenerate with: python adaptive_sampling.py

\\begin{{table}}[htbp]
\\centering
\\caption{{Synthetic network comparison: Full vs uniform vs adaptive sampling. Uniform uses reach-weighted mean probability from adaptive plan for fair comparison.}}
\\label{{tab:synthetic_results}}
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
Topology & Nodes & Full (s) & Uniform (s) & Adaptive (s) & Speedup & $\\rho_H$ & $\\rho_B$ \\\\
\\midrule
"""
    for r in synthetic_results:
        speedup = r["time_full"] / r["time_adaptive"]
        mean_h = np.mean(list(r["adaptive_acc"]["harmonic"].values()))
        mean_b = np.mean(list(r["adaptive_acc"]["betweenness"].values()))
        synthetic_tex += f"{r['topology'].title()} & {r['n_nodes']} & {r['time_full']:.1f} & {r['time_uniform']:.1f} & {r['time_adaptive']:.1f} & {speedup:.1f}$\\times$ & {mean_h:.3f} & {mean_b:.3f} \\\\\n"

    synthetic_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(TABLES_DIR / "synthetic_results.tex", "w") as f:
        f.write(synthetic_tex)

    # Real-world networks table
    if network_stats:
        networks_tex = f"""% Auto-generated table: Real-World Network Characteristics
% Generated by adaptive_sampling.py on {timestamp}
% DO NOT EDIT MANUALLY - regenerate with: python adaptive_sampling.py

\\begin{{table}}[htbp]
\\centering
\\caption{{Real-world network characteristics.}}
\\label{{tab:realworld_networks}}
\\begin{{tabular}}{{lrrrrl}}
\\toprule
City & Centre & Buffer & Nodes & Edges & Characteristics \\\\
\\midrule
"""
        for city_key, stats in network_stats.items():
            city = CITIES[city_key]
            networks_tex += f"{city['name']} & ${city['lng']:.3f}$, ${city['lat']:.3f}$ & {city['buffer']}m & {stats['n_nodes']} & {stats['n_edges']} & {city['description']} \\\\\n"

        networks_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        with open(TABLES_DIR / "realworld_networks.tex", "w") as f:
            f.write(networks_tex)

    print(f"Saved LaTeX tables to: {TABLES_DIR}")


# %% Spatial Analysis Functions
def compute_morans_i(values: np.ndarray, coords: np.ndarray, k: int = 8) -> tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Uses k-nearest neighbors for spatial weights.

    Returns
    -------
    tuple[float, float]
        (Moran's I, p-value)
    """
    n = len(values)
    if n < 10:
        return np.nan, np.nan

    mean_val = np.mean(values)
    dev = values - mean_val
    var = np.var(values)
    if var == 0:
        return np.nan, np.nan

    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]  # Remove self

    numerator = 0.0
    weight_sum = 0.0
    for i in range(n):
        for j in indices[i]:
            numerator += dev[i] * dev[j]
            weight_sum += 1.0

    if weight_sum == 0:
        return np.nan, np.nan

    morans_i = (n / weight_sum) * (numerator / (n * var))
    expected_i = -1.0 / (n - 1)

    s1 = 2 * weight_sum
    s2 = 4 * k * n
    b2 = np.mean(dev**4) / var**2

    variance_i = (
        (n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * weight_sum**2))
        - b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * weight_sum**2)
    ) / ((n - 1) * (n - 2) * (n - 3) * weight_sum**2) - expected_i**2

    if variance_i <= 0:
        return morans_i, np.nan

    z_score = (morans_i - expected_i) / np.sqrt(variance_i)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))

    return morans_i, p_value


def run_spatial_analysis(city_key: str, distances: list[int], sample_prob: float, n_runs: int = 3):
    """
    Run spatial analysis for a city to test if accuracy holds across all areas.

    Key question: Does sampling accuracy hold in low-reachability regions?

    Returns
    -------
    tuple[list[dict], dict]
        (quartile_results, spatial_data) where spatial_data contains coords/residuals for mapping
    """
    city = CITIES[city_key]
    print(f"\n  Spatial analysis for {city['name']}...")

    # Load network
    network_data = download_network(city_key)
    net = network_data["network_structure"]
    nodes_gdf = network_data["nodes_gdf"]
    coords = np.column_stack([nodes_gdf.geometry.x.values, nodes_gdf.geometry.y.values])

    results = []
    spatial_data = {"city": city_key, "coords": coords, "distances": {}}

    for dist in distances:
        print(f"    Distance {dist}m...")

        # Ground truth
        true_result = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            pbar_disabled=True,
        )
        true_harmonic = np.array(true_result.node_harmonic[dist])
        true_betweenness = np.array(true_result.node_betweenness[dist])
        reachability = np.array(true_result.node_density[dist])

        # Sampled estimates (averaged over runs)
        est_harmonic_runs = []
        est_betweenness_runs = []
        for seed in range(n_runs):
            r = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=sample_prob,
                random_seed=seed,
                pbar_disabled=True,
            )
            est_harmonic_runs.append(np.array(r.node_harmonic[dist]))
            est_betweenness_runs.append(np.array(r.node_betweenness[dist]))

        est_harmonic = np.mean(est_harmonic_runs, axis=0)
        est_betweenness = np.mean(est_betweenness_runs, axis=0)

        # Residuals (relative error)
        harmonic_residual = np.where(true_harmonic > 0, (est_harmonic - true_harmonic) / true_harmonic, 0)
        betweenness_residual = np.where(
            true_betweenness > 0, (est_betweenness - true_betweenness) / true_betweenness, 0
        )

        # Compute local density using KDTree (Euclidean, for topology transition analysis)
        local_radius = 500  # meters
        tree = KDTree(coords)
        local_density = np.array([len(tree.query_ball_point(c, local_radius)) for c in coords])

        # Compute topology transition score: local_density / regional_density (reachability)
        # High score = locally dense relative to region (village core)
        # Low score = locally sparse relative to region (rural near city)
        with np.errstate(divide="ignore", invalid="ignore"):
            transition_score = np.where(reachability > 0, local_density / reachability, np.nan)

        # Store spatial data for mapping
        spatial_data["distances"][dist] = {
            "true_harmonic": true_harmonic,
            "true_betweenness": true_betweenness,
            "harmonic_residual": harmonic_residual,
            "betweenness_residual": betweenness_residual,
            "reachability": reachability,
            "local_density": local_density,
            "transition_score": transition_score,
        }

        # Moran's I
        morans_h, _ = compute_morans_i(harmonic_residual, coords)
        morans_b, _ = compute_morans_i(betweenness_residual, coords)

        # Analysis by reachability quartile
        reach_quartiles = pd.qcut(reachability, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

        for q_label in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            mask = reach_quartiles == q_label
            if mask.sum() < 10:
                continue

            q_true_h = true_harmonic[mask]
            q_est_h = est_harmonic[mask]
            q_true_b = true_betweenness[mask]
            q_est_b = est_betweenness[mask]

            valid_h = (q_true_h > 0) & np.isfinite(q_true_h) & np.isfinite(q_est_h)
            valid_b = (q_true_b > 0) & np.isfinite(q_true_b) & np.isfinite(q_est_b)

            rho_h = scipy_stats.spearmanr(q_true_h[valid_h], q_est_h[valid_h])[0] if valid_h.sum() > 10 else np.nan
            rho_b = scipy_stats.spearmanr(q_true_b[valid_b], q_est_b[valid_b])[0] if valid_b.sum() > 10 else np.nan

            mean_reach = reachability[mask].mean()

            results.append(
                {
                    "city": city_key,
                    "distance": dist,
                    "sample_prob": sample_prob,
                    "quartile": q_label,
                    "n_nodes": int(mask.sum()),
                    "mean_reachability": mean_reach,
                    "effective_n": mean_reach * sample_prob,
                    "rho_harmonic": rho_h,
                    "rho_betweenness": rho_b,
                    "morans_i_harmonic": morans_h,
                    "morans_i_betweenness": morans_b,
                }
            )

    return results, spatial_data


def generate_residual_maps(spatial_data_list: list[dict], distance: int):
    """
    Generate spatial residual maps showing where errors cluster.

    Each city gets a row with harmonic and betweenness residual maps.
    """
    n_cities = len(spatial_data_list)
    fig, axes = plt.subplots(n_cities, 2, figsize=(10, 4 * n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, 2)

    # Compute consistent scales across all cities for each metric
    cmaps = {"harmonic": "PuOr", "betweenness": "RdBu_r"}
    vmax_all = {}
    for metric in ["harmonic", "betweenness"]:
        all_residuals = []
        for sp_data in spatial_data_list:
            dist_data = sp_data["distances"].get(distance, {})
            if dist_data:
                all_residuals.append(dist_data[f"{metric}_residual"])
        if all_residuals:
            vmax_all[metric] = np.percentile(np.abs(np.concatenate(all_residuals)), 95)
        else:
            vmax_all[metric] = 0.1

    for row, sp_data in enumerate(spatial_data_list):
        city_key = sp_data["city"]
        coords = sp_data["coords"]
        dist_data = sp_data["distances"].get(distance, {})

        if not dist_data:
            continue

        for col, metric in enumerate(["harmonic", "betweenness"]):
            ax = axes[row, col]
            residuals = dist_data[f"{metric}_residual"]
            vmax = vmax_all[metric]
            clipped = np.clip(residuals, -vmax, vmax)

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=clipped,
                cmap=cmaps[metric],
                s=1.5,
                alpha=0.7,
                vmin=-vmax,
                vmax=vmax,
            )

            # Add colorbar for each metric
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02)
            cbar.set_label("Relative error", fontsize=6)
            cbar.ax.tick_params(labelsize=5)

            # Clean axes - remove borders, ticks, labels
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect("equal")

            # Add city name and Moran's I as text
            morans_i, _ = compute_morans_i(residuals, coords)
            ax.text(
                0.02, 0.98,
                f"{CITIES[city_key]['name']}\nMoran's I = {morans_i:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )

            # Column headers on first row only
            if row == 0:
                ax.set_title(f"{metric.title()}", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Spatial Residual Maps ({distance}m)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"spatial_residual_maps_{distance}m.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / f'spatial_residual_maps_{distance}m.png'}")


def generate_residual_vs_centrality(spatial_data_list: list[dict], distance: int):
    """
    Generate residual vs centrality analysis to test for regression-to-mean bias.

    Tests whether high-centrality nodes are systematically underestimated by sampling.
    Each city gets a row showing mean residual by centrality decile.

    Returns list of correlation results for CSV output.
    """
    n_cities = len(spatial_data_list)
    fig, axes = plt.subplots(n_cities, 2, figsize=(12, 3.5 * n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, 2)

    print(f"\n--- Residual vs Centrality Analysis ({distance}m) ---")
    print("Testing for regression-to-mean bias...")

    correlation_results = []

    for row, sp_data in enumerate(spatial_data_list):
        city_key = sp_data["city"]
        dist_data = sp_data["distances"].get(distance, {})

        if not dist_data:
            continue

        for col, metric in enumerate(["harmonic", "betweenness"]):
            ax = axes[row, col]
            true_vals = dist_data[f"true_{metric}"]
            residuals = dist_data[f"{metric}_residual"]

            # Filter valid values (exclude bottom 5% for betweenness to avoid denominator artifacts)
            if metric == "betweenness":
                threshold = np.percentile(true_vals[true_vals > 0], 5)
                valid = (true_vals > threshold) & np.isfinite(true_vals) & np.isfinite(residuals)
            else:
                valid = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(residuals)
            true_valid = true_vals[valid]
            resid_valid = residuals[valid]

            # Compute correlation
            corr_pearson, p_pearson = scipy_stats.pearsonr(true_valid, resid_valid)
            corr_spearman, p_spearman = scipy_stats.spearmanr(true_valid, resid_valid)

            # Store correlation results
            correlation_results.append({
                "city": city_key,
                "city_name": CITIES[city_key]["name"],
                "distance": distance,
                "metric": metric,
                "pearson_r": corr_pearson,
                "pearson_p": p_pearson,
                "spearman_rho": corr_spearman,
                "spearman_p": p_spearman,
                "n_valid": len(true_valid),
            })

            # Bin by centrality decile and compute mean residual per decile
            decile_labels = pd.qcut(true_valid, 10, labels=False, duplicates="drop")
            decile_df = pd.DataFrame({"decile": decile_labels, "true": true_valid, "residual": resid_valid})
            decile_means = decile_df.groupby("decile").agg({"true": "mean", "residual": "mean"}).reset_index()

            # Plot mean residual by decile
            ax.bar(
                decile_means["decile"],
                decile_means["residual"],
                color="steelblue" if metric == "harmonic" else "coral",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

            ax.set_xlabel("Centrality Decile (1=low, 10=high)", fontsize=9)
            ax.set_ylabel("Mean Relative Residual", fontsize=9)
            ax.set_xticks(range(len(decile_means)))
            ax.set_xticklabels([f"{i + 1}" for i in range(len(decile_means))], fontsize=8)
            ax.tick_params(axis="y", labelsize=8)

            # Add city name and correlation annotation
            ax.text(
                0.02, 0.98,
                f"{CITIES[city_key]['name']}\nr = {corr_pearson:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )

            # Column headers on first row only
            if row == 0:
                ax.set_title(f"{metric.title()}", fontsize=12, fontweight="bold")

            ax.grid(True, alpha=0.3, axis="y")

            print(
                f"  {CITIES[city_key]['name']} {metric}: "
                f"r = {corr_pearson:.3f} (p={p_pearson:.1e}), "
                f"ρ = {corr_spearman:.3f} (p={p_spearman:.1e})"
            )

    plt.suptitle(
        f"Residual by Centrality Decile ({distance}m)\n"
        "(Negative bars at high deciles = underestimation of peaks)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"residual_vs_centrality_{distance}m.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / f'residual_vs_centrality_{distance}m.png'}")

    return correlation_results


def generate_residual_scatter(spatial_data_list: list[dict], distance: int):
    """
    Generate scatter plots of residual vs true centrality.

    Shows the full distribution with density coloring to reveal patterns
    that might be hidden in the decile-binned bar charts.
    """
    n_cities = len(spatial_data_list)
    fig, axes = plt.subplots(n_cities, 2, figsize=(12, 4 * n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, 2)

    for row, sp_data in enumerate(spatial_data_list):
        city_key = sp_data["city"]
        dist_data = sp_data["distances"].get(distance, {})

        if not dist_data:
            continue

        for col, metric in enumerate(["harmonic", "betweenness"]):
            ax = axes[row, col]
            true_vals = dist_data[f"true_{metric}"]
            residuals = dist_data[f"{metric}_residual"]

            # Filter valid values
            valid = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(residuals)
            true_valid = true_vals[valid]
            resid_valid = residuals[valid]

            # Normalize true values to percentile for comparable x-axis
            true_pct = scipy_stats.rankdata(true_valid) / len(true_valid) * 100

            # Hexbin for density visualization
            hb = ax.hexbin(
                true_pct,
                resid_valid,
                gridsize=30,
                cmap="YlOrRd",
                mincnt=1,
                reduce_C_function=np.mean,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

            # Add trend line
            z = np.polyfit(true_pct, resid_valid, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 100, 100)
            ax.plot(x_line, p(x_line), "b--", linewidth=1.5, alpha=0.8, label=f"slope={z[0]:.4f}")

            ax.set_xlabel("Centrality Percentile", fontsize=9)
            ax.set_ylabel("Relative Residual", fontsize=9)
            ax.set_xlim(0, 100)

            # Clip y-axis to 95th percentile
            ylim = np.percentile(np.abs(resid_valid), 95) * 1.2
            ax.set_ylim(-ylim, ylim)

            # Add city name annotation
            corr, _ = scipy_stats.pearsonr(true_valid, resid_valid)
            ax.text(
                0.02, 0.98,
                f"{CITIES[city_key]['name']}\nr = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )

            # Column headers on first row only
            if row == 0:
                ax.set_title(f"{metric.title()}", fontsize=12, fontweight="bold")

            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Residual vs Centrality Scatter ({distance}m)\n"
        "(Density hexbin with trend line)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"residual_scatter_{distance}m.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / f'residual_scatter_{distance}m.png'}")


def analyze_topology_transition(spatial_data_list: list[dict], distance: int):
    """
    Analyze whether topology transition (local vs regional density mismatch)
    explains the spatial clustering of residuals.

    Hypothesis: Nodes where local density differs from regional density
    will have larger residuals because sampling misrepresents their context.
    """
    print(f"\n--- Topology Transition Analysis ({distance}m) ---")
    print("Testing: Do local/regional density mismatches cause residual clustering?")

    n_cities = len(spatial_data_list)
    fig, axes = plt.subplots(n_cities, 3, figsize=(15, 4 * n_cities))
    if n_cities == 1:
        axes = axes.reshape(1, 3)

    all_results = []

    for row, sp_data in enumerate(spatial_data_list):
        city_key = sp_data["city"]
        coords = sp_data["coords"]
        dist_data = sp_data["distances"].get(distance, {})

        if not dist_data:
            continue

        transition_score = dist_data["transition_score"]
        harmonic_residual = dist_data["harmonic_residual"]
        betweenness_residual = dist_data["betweenness_residual"]

        # Filter valid values
        valid = np.isfinite(transition_score) & np.isfinite(harmonic_residual)
        ts_valid = transition_score[valid]
        hr_valid = np.abs(harmonic_residual[valid])  # Use absolute residual
        br_valid = np.abs(betweenness_residual[valid])

        # Correlation between transition score deviation and residual magnitude
        # Use deviation from 1.0 (perfect local/regional match)
        ts_deviation = np.abs(np.log(ts_valid + 1e-6))  # Log scale, deviation from 0 = ratio of 1

        corr_h, p_h = scipy_stats.pearsonr(ts_deviation, hr_valid)
        corr_b, p_b = scipy_stats.pearsonr(ts_deviation, br_valid)

        all_results.append({
            "city": city_key,
            "city_name": CITIES[city_key]["name"],
            "distance": distance,
            "corr_harmonic": corr_h,
            "p_harmonic": p_h,
            "corr_betweenness": corr_b,
            "p_betweenness": p_b,
        })

        print(f"  {CITIES[city_key]['name']}:")
        print(f"    Transition deviation vs |harmonic residual|: r = {corr_h:.3f} (p = {p_h:.1e})")
        print(f"    Transition deviation vs |betweenness residual|: r = {corr_b:.3f} (p = {p_b:.1e})")

        # Plot 1: Transition score map
        ax = axes[row, 0]
        ts_clipped = np.clip(transition_score, 0.5, 2.0)  # Clip for visualization
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=ts_clipped, cmap="RdYlBu_r", s=0.5, alpha=0.6,
            vmin=0.5, vmax=2.0
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02)
        cbar.set_label("Transition score", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        ax.text(0.02, 0.98, f"{CITIES[city_key]['name']}\nlocal/regional",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
        if row == 0:
            ax.set_title("Transition Score\n(red=locally dense)", fontsize=10, fontweight="bold")

        # Plot 2: Harmonic residual map (for comparison)
        ax = axes[row, 1]
        hr_clipped = np.clip(harmonic_residual, -0.15, 0.15)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=hr_clipped, cmap="PuOr", s=0.5, alpha=0.6,
            vmin=-0.15, vmax=0.15
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02)
        cbar.set_label("Residual", fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title("Harmonic Residual", fontsize=10, fontweight="bold")

        # Plot 3: Scatter of transition deviation vs absolute residual
        ax = axes[row, 2]
        # Subsample for visualization
        n_sample = min(5000, len(ts_deviation))
        idx = np.random.choice(len(ts_deviation), n_sample, replace=False)
        ax.scatter(ts_deviation[idx], hr_valid[idx], s=1, alpha=0.3, c="steelblue", label="Harmonic")
        ax.scatter(ts_deviation[idx], br_valid[idx], s=1, alpha=0.3, c="coral", label="Betweenness")

        # Trend lines
        z_h = np.polyfit(ts_deviation, hr_valid, 1)
        z_b = np.polyfit(ts_deviation, br_valid, 1)
        x_line = np.linspace(0, ts_deviation.max(), 100)
        ax.plot(x_line, np.poly1d(z_h)(x_line), "b-", linewidth=2, label=f"H: r={corr_h:.2f}")
        ax.plot(x_line, np.poly1d(z_b)(x_line), "r-", linewidth=2, label=f"B: r={corr_b:.2f}")

        ax.set_xlabel("|log(transition score)|", fontsize=9)
        ax.set_ylabel("|Relative residual|", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Transition vs Residual", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Topology Transition Analysis ({distance}m)\n"
        "Does local/regional density mismatch explain residual clustering?",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"topology_transition_{distance}m.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / f'topology_transition_{distance}m.png'}")

    return all_results


def generate_spatial_report(spatial_results: list[dict], spatial_data_list: list[dict]):
    """Generate spatial analysis report and figures."""
    df = pd.DataFrame(spatial_results)

    print("\n" + "=" * 70)
    print("SPATIAL ANALYSIS RESULTS")
    print("=" * 70)
    print("Key question: Does accuracy hold in low-reachability areas?")

    # Summary by quartile (overall)
    print("\nAccuracy by reachability quartile (averaged across cities/distances):")
    quartile_order = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    summary = df.groupby("quartile").agg(
        {
            "mean_reachability": "mean",
            "effective_n": "mean",
            "rho_harmonic": "mean",
            "rho_betweenness": "mean",
        }
    )
    print(summary.to_string())

    # City-by-city breakdown
    print("\n" + "-" * 70)
    print("CITY-BY-CITY BREAKDOWN")
    print("-" * 70)
    print("Checking if the accuracy pattern is consistent across cities...")

    for city_key in df["city"].unique():
        city_df = df[df["city"] == city_key]
        city_name = CITIES[city_key]["name"]
        print(f"\n{city_name}:")
        print(f"  {'Quartile':<12} {'Mean Reach':>12} {'ρ Harmonic':>12} {'ρ Betw.':>12}")
        print(f"  {'-' * 48}")
        for q in quartile_order:
            q_data = city_df[city_df["quartile"] == q]
            if len(q_data) > 0:
                print(
                    f"  {q:<12} {q_data['mean_reachability'].mean():>12.0f} "
                    f"{q_data['rho_harmonic'].mean():>12.3f} {q_data['rho_betweenness'].mean():>12.3f}"
                )

    # Check accuracy gap
    q1_h = df[df["quartile"] == "Q1 (low)"]["rho_harmonic"].mean()
    q4_h = df[df["quartile"] == "Q4 (high)"]["rho_harmonic"].mean()
    q1_b = df[df["quartile"] == "Q1 (low)"]["rho_betweenness"].mean()
    q4_b = df[df["quartile"] == "Q4 (high)"]["rho_betweenness"].mean()

    print("\n" + "-" * 70)
    print("Low vs High reachability accuracy gap:")
    print(f"  Harmonic:    Q1={q1_h:.3f}, Q4={q4_h:.3f}, gap={q4_h - q1_h:+.3f}")
    print(f"  Betweenness: Q1={q1_b:.3f}, Q4={q4_b:.3f}, gap={q4_b - q1_b:+.3f}")

    # Moran's I summary
    print("\nSpatial autocorrelation (Moran's I) by city:")
    for city_key in df["city"].unique():
        city_df = df[df["city"] == city_key]
        morans_h = city_df.groupby("distance")["morans_i_harmonic"].first().mean()
        morans_b = city_df.groupby("distance")["morans_i_betweenness"].first().mean()
        print(f"  {CITIES[city_key]['name']}: H={morans_h:.3f}, B={morans_b:.3f}")

    morans_h = df.groupby(["city", "distance"])["morans_i_harmonic"].first().mean()
    morans_b = df.groupby(["city", "distance"])["morans_i_betweenness"].first().mean()
    print(f"  Overall average: H={morans_h:.3f}, B={morans_b:.3f}")

    # Generate residual maps and centrality analysis for each distance
    all_correlation_results = []
    all_transition_results = []
    if spatial_data_list:
        distances = list(spatial_data_list[0]["distances"].keys())
        for dist in distances:
            generate_residual_maps(spatial_data_list, dist)
            corr_results = generate_residual_vs_centrality(spatial_data_list, dist)
            all_correlation_results.extend(corr_results)
            generate_residual_scatter(spatial_data_list, dist)
            transition_results = analyze_topology_transition(spatial_data_list, dist)
            all_transition_results.extend(transition_results)

    # Save correlation summary table
    if all_correlation_results:
        corr_df = pd.DataFrame(all_correlation_results)
        corr_df.to_csv(TABLES_DIR / "residual_centrality_correlations.csv", index=False)
        print(f"\nSaved: {TABLES_DIR / 'residual_centrality_correlations.csv'}")

        # Print summary
        print("\n--- Regression-to-Mean Summary ---")
        print("Correlation between true centrality and relative residual:")
        print(f"  {'City':<12} {'Dist':<8} {'Harmonic r':<12} {'Betweenness r':<14}")
        print(f"  {'-' * 46}")
        for dist in sorted(corr_df["distance"].unique()):
            for city in corr_df["city_name"].unique():
                h_row = corr_df[(corr_df["distance"] == dist) & (corr_df["city_name"] == city) & (corr_df["metric"] == "harmonic")]
                b_row = corr_df[(corr_df["distance"] == dist) & (corr_df["city_name"] == city) & (corr_df["metric"] == "betweenness")]
                if len(h_row) > 0 and len(b_row) > 0:
                    print(f"  {city:<12} {dist:<8} {h_row['pearson_r'].values[0]:>+.3f}        {b_row['pearson_r'].values[0]:>+.3f}")

    # Save topology transition results
    if all_transition_results:
        trans_df = pd.DataFrame(all_transition_results)
        trans_df.to_csv(TABLES_DIR / "topology_transition_correlations.csv", index=False)
        print(f"\nSaved: {TABLES_DIR / 'topology_transition_correlations.csv'}")

        # Print summary
        print("\n--- Topology Transition Summary ---")
        print("Correlation between |log(transition_score)| and |residual|:")
        print(f"  {'City':<12} {'Dist':<8} {'Harmonic r':<12} {'Betweenness r':<14}")
        print(f"  {'-' * 46}")
        for _, row in trans_df.iterrows():
            print(f"  {row['city_name']:<12} {row['distance']:<8} {row['corr_harmonic']:>+.3f}        {row['corr_betweenness']:>+.3f}")

    # Generate quartile accuracy figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"london": "#1f77b4", "madrid": "#d62728", "phoenix": "#2ca02c"}

    for col, metric in enumerate(["harmonic", "betweenness"]):
        ax = axes[col]
        for city in df["city"].unique():
            city_df = df[df["city"] == city]
            agg = city_df.groupby("quartile")[f"rho_{metric}"].mean()
            values = [agg.get(q, np.nan) for q in quartile_order]
            ax.plot(range(4), values, "o-", label=CITIES[city]["name"], color=colors.get(city, "gray"), linewidth=2)

        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
        ax.set_xlabel("Reachability Quartile")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{metric.title()}")
        ax.set_ylim(0.75, 1.01)
        ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="Target ρ=0.95")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Spatial Validation: Accuracy by Local Reachability\n"
        "(Does accuracy hold in low-reachability areas?)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spatial_reachability_accuracy.pdf", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIGURES_DIR / 'spatial_reachability_accuracy.pdf'}")

    # Save data
    df.to_csv(TABLES_DIR / "spatial_analysis.csv", index=False)

    # Generate LaTeX table
    timestamp = datetime.now().isoformat(timespec="seconds")
    latex = f"""% Auto-generated table: Spatial Analysis - Accuracy by Reachability
% Generated by adaptive_sampling.py on {timestamp}

\\begin{{table}}[htbp]
\\centering
\\caption{{Accuracy by local reachability quartile. Q1 = lowest 25\\% reachability,
Q4 = highest 25\\%. Lower reachability means lower effective sample size.}}
\\label{{tab:spatial_reachability}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Quartile & Mean Reach & Eff. N & $\\rho_H$ & $\\rho_B$ \\\\
\\midrule
"""
    for q in quartile_order:
        q_data = df[df["quartile"] == q]
        latex += (
            f"{q} & {q_data['mean_reachability'].mean():,.0f} & "
            f"{q_data['effective_n'].mean():.0f} & "
            f"{q_data['rho_harmonic'].mean():.3f} & "
            f"{q_data['rho_betweenness'].mean():.3f} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(TABLES_DIR / "spatial_reachability.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {TABLES_DIR / 'spatial_reachability.tex'}")


# %% Main execution
print("=" * 70)
print("ADAPTIVE SAMPLING ANALYSIS")
print("=" * 70)
print(f"Target accuracy: rho >= {TARGET_RHO}")
print(f"Output: {FIGURES_DIR}")

synthetic_results = []
realworld_results = []
network_stats = {}

# Part 1: Synthetic networks
print("\n" + "=" * 70)
print("PART 1: SYNTHETIC NETWORK COMPARISON")
print("=" * 70)

for template in TEMPLATE_NAMES:
    result = run_synthetic_comparison(template, SYNTHETIC_DISTANCES, TARGET_RHO)
    synthetic_results.append(result)

# Part 2: Real-world networks
print("\n" + "=" * 70)
print("PART 2: REAL-WORLD NETWORK VALIDATION")
print("=" * 70)

for city_key in CITIES:
    network_data = download_network(city_key)
    network_stats[city_key] = {"n_nodes": network_data["n_nodes"], "n_edges": network_data["n_edges"]}
    results = run_realworld_validation(city_key)
    realworld_results.extend(results)

if realworld_results:
    generate_realworld_figure(realworld_results)

# Generate LaTeX tables
generate_latex_tables(synthetic_results, realworld_results, network_stats)

# Part 3: Spatial Analysis
print("\n" + "=" * 70)
print("PART 3: SPATIAL ANALYSIS")
print("=" * 70)
print("Testing whether accuracy holds spatially across all areas...")
print("(Reachability varies spatially - low-reach areas may have lower accuracy)")

SPATIAL_SAMPLE_PROB = 0.3
SPATIAL_DISTANCES = [500, 1000, 2000]
spatial_results = []
spatial_data_list = []

for city_key in CITIES:
    results, spatial_data = run_spatial_analysis(city_key, SPATIAL_DISTANCES, SPATIAL_SAMPLE_PROB, n_runs=3)
    spatial_results.extend(results)
    spatial_data_list.append(spatial_data)

if spatial_results:
    generate_spatial_report(spatial_results, spatial_data_list)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nSynthetic Network Results:")
print(f"{'Topology':<10} {'Full':<10} {'Uniform':<10} {'Adaptive':<10} {'Speedup':<10}")
print("-" * 50)
for r in synthetic_results:
    speedup = r["time_full"] / r["time_adaptive"]
    print(
        f"{r['topology']:<10} {r['time_full']:<10.2f} {r['time_uniform']:<10.2f} "
        f"{r['time_adaptive']:<10.2f} {speedup:<10.1f}x"
    )

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
