# %% [markdown]
# # Supplementary Figures
#
# Generates supplementary material figures for the adaptive sampling paper.
#
# Outputs:
# - figures/pernode_ciwidth_vs_effn.pdf (Per-node CI-width analysis)
# - figures/probe_sensitivity.pdf (Probe count sensitivity analysis)
# - tables/probe_sensitivity.csv (Probe sensitivity data)
#
# Consolidated from:
# - figure_pernode_ciwidth.py
# - analyze_probe_sensitivity.py

# %% Imports
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer import rustalgos
from cityseer.tools import io
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# %% Configuration
SCRIPT_DIR = Path(__file__).parent
PAPER_DIR = SCRIPT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
CACHE_DIR = SCRIPT_DIR / "cache"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache version
CACHE_VERSION = "v2"

# London network configuration (used by both analyses)
LONDON_CONFIG = {
    "name": "London (Soho)",
    "lng": -0.13396079424572427,
    "lat": 51.51371088849723,
    "buffer": 2000,
}

# Walking speed (m/s)
SPEED_M_S = 1.33

# Matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# %% Shared Utility Functions
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


def download_london_network(force: bool = False) -> dict:
    """Download OSM network for London."""
    cache_key = "network_london"

    if not force:
        cached = load_cache(cache_key)
        if cached is not None:
            print("  Loaded London network from cache")
            network_structure = io.network_structure_from_gpd(cached["nodes_gdf"], cached["edges_gdf"])
            cached["network_structure"] = network_structure
            return cached

    print("  Downloading London network from OSM...")

    poly_wgs, _ = io.buffered_point_poly(LONDON_CONFIG["lng"], LONDON_CONFIG["lat"], LONDON_CONFIG["buffer"])
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


def spatial_sample(
    network_structure: rustalgos.graph.NetworkStructure,
    n_samples: int,
    random_seed: int | None = None,
) -> tuple[list[int], float]:
    """
    Select spatially stratified sample of nodes.

    Parameters
    ----------
    network_structure
        Network to sample from.
    n_samples
        Number of nodes to sample.
    random_seed
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[int], float]
        (list of selected node indices, area in km^2)
    """
    import random

    if random_seed is not None:
        random.seed(random_seed)

    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    if not live_indices:
        return [], 0.0

    n_samples = min(n_samples, len(live_indices))

    all_xs = network_structure.node_xs
    all_ys = network_structure.node_ys
    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    area_km2 = x_range * y_range / 1_000_000

    cell_size = 1000.0
    n_cells_x = max(1, int(x_range / cell_size))
    n_cells_y = max(1, int(y_range / cell_size))

    def get_cell_id(x: float, y: float) -> tuple[int, int]:
        cx = min(int((x - x_min) / cell_size), n_cells_x - 1)
        cy = min(int((y - y_min) / cell_size), n_cells_y - 1)
        return (cx, cy)

    cell_ids = [get_cell_id(xs[i], ys[i]) for i in range(len(live_indices))]
    cells: dict[tuple[int, int], list[int]] = {}
    for idx, cell_id in zip(live_indices, cell_ids, strict=False):
        cells.setdefault(cell_id, []).append(idx)

    selected = []
    cell_lists = list(cells.values())
    random.shuffle(cell_lists)

    while len(selected) < n_samples:
        for cell_nodes in cell_lists:
            if cell_nodes and len(selected) < n_samples:
                idx = random.randrange(len(cell_nodes))
                selected.append(cell_nodes.pop(idx))

    return selected, area_km2


# =============================================================================
# SECTION 1: Per-Node CI-Width Analysis
# =============================================================================


def generate_pernode_ciwidth_figure():
    """
    Generate per-node CI-width vs reachability figure.

    Shows that within a single network at fixed sampling probability:
    - Nodes with LOW reachability have SMALL absolute CI-width
    - Nodes with HIGH reachability have LARGE absolute CI-width

    Coloured by RELATIVE CI-width to show the disparity:
    - Low-reachability nodes have HIGH relative CI-width (large % error)
    - High-reachability nodes have LOW relative CI-width (small % error)
    """
    print("=" * 70)
    print("SECTION 1: PER-NODE CI-WIDTH ANALYSIS")
    print("=" * 70)

    # Parameters
    n_runs = 20
    distances = [500, 1000]
    probability = 0.5

    # Load network
    print("\nLoading London network...")
    network_data = download_london_network()
    network_structure = network_data["network_structure"]
    n_nodes = network_data["n_nodes"]
    print(f"Network: {n_nodes} nodes")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, dist in enumerate(distances):
        print(f"\nDistance: {dist}m")

        # Run multiple times to get variance
        hillier_runs = []
        betweenness_runs = []

        print(f"  Running {n_runs} trials at p={probability}...")
        for seed in range(n_runs):
            result = network_structure.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=probability,
                random_seed=seed,
                pbar_disabled=True,
            )

            density = np.array(result.node_density[dist])
            farness = np.array(result.node_farness[dist])
            hillier = np.where(farness > 0, density**2 / farness, 0)
            hillier_runs.append(hillier)
            betweenness_runs.append(np.array(result.node_betweenness[dist]))

        hillier_arr = np.array(hillier_runs)
        betweenness_arr = np.array(betweenness_runs)

        hillier_std = np.std(hillier_arr, axis=0)
        betweenness_std = np.std(betweenness_arr, axis=0)

        # Get true values (p=1.0)
        print("  Computing ground truth (p=1.0)...")
        result_true = network_structure.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            pbar_disabled=True,
        )

        density_true = np.array(result_true.node_density[dist])
        farness_true = np.array(result_true.node_farness[dist])
        hillier_true = np.where(farness_true > 0, density_true**2 / farness_true, 0)
        betweenness_true = np.array(result_true.node_betweenness[dist])
        reachability = density_true

        effective_n = reachability * probability

        hillier_rel_ci = np.where(hillier_true > 0, hillier_std / hillier_true, 0)
        betweenness_rel_ci = np.where(betweenness_true > 0, betweenness_std / betweenness_true, 0)

        # Correlations
        valid_h = hillier_std > 0
        valid_b = betweenness_std > 0
        corr_h_abs = np.corrcoef(effective_n[valid_h], hillier_std[valid_h])[0, 1]
        corr_b_abs = np.corrcoef(effective_n[valid_b], betweenness_std[valid_b])[0, 1]

        # Plot Hillier closeness
        ax = axes[row, 0]
        rel_ci_capped = np.clip(hillier_rel_ci * 100, 0, 50)
        scatter = ax.scatter(
            effective_n,
            hillier_std,
            c=rel_ci_capped,
            cmap="RdYlGn_r",
            alpha=0.5,
            s=8,
            edgecolors="none",
            vmin=0,
            vmax=50,
        )
        ax.set_xlabel("Effective sample size (reachability × p)")
        ax.set_ylabel("Absolute CI-width (std)")
        ax.set_title(f"d={dist}m: Hillier Closeness (r={corr_h_abs:.2f})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Relative CI (%)")

        # Plot betweenness
        ax = axes[row, 1]
        rel_ci_capped = np.clip(betweenness_rel_ci * 100, 0, 50)
        scatter = ax.scatter(
            effective_n,
            betweenness_std,
            c=rel_ci_capped,
            cmap="RdYlGn_r",
            alpha=0.5,
            s=8,
            edgecolors="none",
            vmin=0,
            vmax=50,
        )
        ax.set_xlabel("Effective sample size (reachability × p)")
        ax.set_ylabel("Absolute CI-width (std)")
        ax.set_title(f"d={dist}m: Betweenness (r={corr_b_abs:.2f})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Relative CI (%)")

    fig.suptitle(
        f"London (Soho): Per-Node Absolute CI-Width vs Effective Sample Size (p={probability})\n"
        f"Colour = Relative CI-width (red=high %, green=low %)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = FIGURES_DIR / "pernode_ciwidth_vs_effn.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    plt.close()


# =============================================================================
# SECTION 2: Probe Sensitivity Analysis
# =============================================================================

# Probe sensitivity configuration
PROBE_COUNTS = [10, 20, 50, 100, 200]
PROBE_DISTANCES = [500, 1000, 2000]
PROBE_N_RUNS = 10


def probe_reachability_with_k(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    n_probes: int,
    speed_m_s: float = SPEED_M_S,
    random_seed: int | None = None,
) -> dict[int, float]:
    """
    Estimate reachability per distance using a specified number of probes.

    Parameters
    ----------
    network_structure
        The network to probe.
    distances
        Distance thresholds in metres.
    n_probes
        Number of probes to use.
    speed_m_s
        Walking speed for converting distance to seconds.
    random_seed
        Random seed for probe selection.

    Returns
    -------
    dict[int, float]
        Mean reachability (node count) for each distance threshold.
    """
    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    if not live_indices:
        return {d: 0.0 for d in distances}

    n_probes = min(n_probes, len(live_indices))
    probe_indices, _ = spatial_sample(network_structure, n_probes, random_seed=random_seed)

    reach_counts: dict[int, list[int]] = {d: [] for d in distances}
    max_seconds = int(max(distances) / speed_m_s)

    for src_idx in probe_indices:
        visited, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None
        )

        for d in distances:
            count = sum(1 for v_idx in visited if tree_map[v_idx].short_dist <= d and v_idx != src_idx)
            reach_counts[d].append(count)

    return {d: float(np.mean(counts)) if counts else 0.0 for d, counts in reach_counts.items()}


def compute_true_reachability(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    speed_m_s: float = SPEED_M_S,
) -> dict[int, float]:
    """Compute true mean reachability by running full computation."""
    print("  Computing true reachability (full computation)...")
    t0 = time.perf_counter()

    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    if not live_indices:
        return {d: 0.0 for d in distances}

    reach_counts: dict[int, list[int]] = {d: [] for d in distances}
    max_seconds = int(max(distances) / speed_m_s)

    for src_idx in live_indices:
        visited, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None
        )

        for d in distances:
            count = sum(1 for v_idx in visited if tree_map[v_idx].short_dist <= d and v_idx != src_idx)
            reach_counts[d].append(count)

    elapsed = time.perf_counter() - t0
    print(f"  True reachability computed in {elapsed:.1f}s")

    return {d: float(np.mean(counts)) if counts else 0.0 for d, counts in reach_counts.items()}


def run_probe_sensitivity_analysis():
    """Run the probe sensitivity analysis."""
    print("\n" + "=" * 70)
    print("SECTION 2: PROBE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Testing probe counts: {PROBE_COUNTS}")
    print(f"Distances: {PROBE_DISTANCES}m")
    print(f"Runs per probe count: {PROBE_N_RUNS}")

    # Load network
    print("\nLoading London network...")
    network_data = download_london_network()
    net = network_data["network_structure"]
    print(f"Network: {network_data['n_nodes']} nodes, {network_data['n_edges']} edges")

    # Check for cached results
    cache_key = "probe_sensitivity_results"
    cached_results = load_cache(cache_key)

    if cached_results is not None:
        print("\nLoaded cached sensitivity results")
        true_reachability = cached_results["true_reachability"]
        results = cached_results["results"]
    else:
        # Compute true reachability
        print("\n" + "-" * 70)
        true_reachability = compute_true_reachability(net, PROBE_DISTANCES)
        for d in PROBE_DISTANCES:
            print(f"  True mean reachability at {d}m: {true_reachability[d]:.1f} nodes")

        # Run sensitivity analysis
        print("\n" + "-" * 70)
        print("Running probe sensitivity analysis...")
        results = []

        for k in PROBE_COUNTS:
            print(f"\n  Testing k={k} probes ({PROBE_N_RUNS} runs)...")

            for run_idx in range(PROBE_N_RUNS):
                random_seed = 42 + run_idx

                t0 = time.perf_counter()
                estimated = probe_reachability_with_k(net, PROBE_DISTANCES, n_probes=k, random_seed=random_seed)
                probe_time = time.perf_counter() - t0

                for d in PROBE_DISTANCES:
                    true_val = true_reachability[d]
                    est_val = estimated[d]
                    if true_val > 0:
                        rel_error = (est_val - true_val) / true_val * 100
                        abs_rel_error = abs(rel_error)
                    else:
                        rel_error = 0.0
                        abs_rel_error = 0.0

                    results.append(
                        {
                            "k": k,
                            "run": run_idx,
                            "distance": d,
                            "true_reach": true_val,
                            "estimated_reach": est_val,
                            "rel_error_pct": rel_error,
                            "abs_rel_error_pct": abs_rel_error,
                            "probe_time_s": probe_time,
                        }
                    )

            # Summary for this k
            k_results = [r for r in results if r["k"] == k]
            mean_abs_error = np.mean([r["abs_rel_error_pct"] for r in k_results])
            std_error = np.std([r["rel_error_pct"] for r in k_results])
            mean_time = np.mean([r["probe_time_s"] for r in k_results]) / len(PROBE_DISTANCES)
            print(f"    Mean |error|: {mean_abs_error:.2f}%, Std: {std_error:.2f}%, Time: {mean_time:.3f}s")

        save_cache(cache_key, {"true_reachability": true_reachability, "results": results})

    return true_reachability, results


def generate_probe_sensitivity_figure(results: list[dict]):
    """Generate the probe sensitivity figure."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {500: "#1f77b4", 1000: "#ff7f0e", 2000: "#2ca02c"}
    markers = {500: "o", 1000: "s", 2000: "^"}

    # Left plot: Absolute relative error by probe count
    ax1 = axes[0]

    for d in PROBE_DISTANCES:
        d_df = df[df["distance"] == d]

        agg = (
            d_df.groupby("k")
            .agg(
                mean_error=("abs_rel_error_pct", "mean"),
                std_error=("abs_rel_error_pct", "std"),
                n_runs=("abs_rel_error_pct", "count"),
            )
            .reset_index()
        )

        agg["ci95"] = 1.96 * agg["std_error"] / np.sqrt(agg["n_runs"])

        ax1.errorbar(
            agg["k"],
            agg["mean_error"],
            yerr=agg["ci95"],
            marker=markers[d],
            color=colors[d],
            linewidth=2,
            markersize=8,
            capsize=4,
            label=f"{d}m",
        )

    ax1.axhline(y=5.0, color="gray", linestyle="--", alpha=0.7, label="5% error")
    ax1.axvline(x=50, color="green", linestyle=":", alpha=0.7, label="Default k=50")

    ax1.set_xlabel("Number of Probes (k)", fontsize=11)
    ax1.set_ylabel("Mean Absolute Relative Error (%)", fontsize=11)
    ax1.set_title("Probe Sensitivity: Error vs Probe Count", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 220)
    ax1.set_ylim(0, max(df["abs_rel_error_pct"].max() * 1.1, 15))
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(PROBE_COUNTS)

    # Right plot: Standard deviation
    ax2 = axes[1]

    for d in PROBE_DISTANCES:
        d_df = df[df["distance"] == d]

        agg = d_df.groupby("k").agg(estimate_std=("rel_error_pct", "std")).reset_index()

        ax2.plot(
            agg["k"],
            agg["estimate_std"],
            marker=markers[d],
            color=colors[d],
            linewidth=2,
            markersize=8,
            label=f"{d}m",
        )

    ax2.axvline(x=50, color="green", linestyle=":", alpha=0.7, label="Default k=50")

    ax2.set_xlabel("Number of Probes (k)", fontsize=11)
    ax2.set_ylabel("Std. Dev. of Relative Error (%)", fontsize=11)
    ax2.set_title("Probe Sensitivity: Variability vs Probe Count", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 220)
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(PROBE_COUNTS)

    plt.tight_layout()
    output_path = FIGURES_DIR / "probe_sensitivity.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    plt.close()


def print_probe_summary(true_reachability: dict[int, float], results: list[dict]):
    """Print summary statistics for probe sensitivity."""
    df = pd.DataFrame(results)

    print("\n" + "-" * 70)
    print("PROBE SENSITIVITY SUMMARY")
    print("-" * 70)

    print("\nTrue mean reachability:")
    for d in PROBE_DISTANCES:
        print(f"  {d}m: {true_reachability[d]:.1f} nodes")

    print(f"\n{'k':<8} {'Mean |Error|':>14} {'Std Error':>12} {'Max |Error|':>14}")
    print("-" * 50)

    for k in PROBE_COUNTS:
        k_df = df[df["k"] == k]
        mean_err = k_df["abs_rel_error_pct"].mean()
        std_err = k_df["rel_error_pct"].std()
        max_err = k_df["abs_rel_error_pct"].max()
        print(f"{k:<8} {mean_err:>14.2f}% {std_err:>12.2f}% {max_err:>14.2f}%")

    # Key finding
    k50_mean = df[df["k"] == 50]["abs_rel_error_pct"].mean()
    print(f"\nConclusion: k=50 achieves {k50_mean:.2f}% mean error")


def save_probe_results_table(results: list[dict]):
    """Save probe sensitivity results as CSV."""
    df = pd.DataFrame(results)

    agg = (
        df.groupby(["k", "distance"])
        .agg(
            mean_abs_error=("abs_rel_error_pct", "mean"),
            std_error=("rel_error_pct", "std"),
            n_runs=("run", "count"),
        )
        .reset_index()
    )

    output_path = TABLES_DIR / "probe_sensitivity.csv"
    agg.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SUPPLEMENTARY FIGURES GENERATION")
    print("=" * 70)

    # Section 1: Per-node CI-width analysis
    generate_pernode_ciwidth_figure()

    # Section 2: Probe sensitivity analysis
    true_reachability, results = run_probe_sensitivity_analysis()
    generate_probe_sensitivity_figure(results)
    print_probe_summary(true_reachability, results)
    save_probe_results_table(results)

    print("\n" + "=" * 70)
    print("SUPPLEMENTARY FIGURES COMPLETE")
    print("=" * 70)
