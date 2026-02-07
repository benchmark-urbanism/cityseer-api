"""
Utility Functions for Sampling Analysis

Provides shared functionality for network loading, caching, and analysis across
all sampling analysis scripts.

Sections:
1. Configuration and paths
2. Cache management (load/save pickle caches)
3. Network loading (OSM cities, Madrid regional, from cache)
4. Statistical utilities (accuracy metrics, Moran's I)
"""

import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: Configuration and Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
CACHE_DIR = SAMPLING_DIR / ".cache"
OUTPUT_DIR = SAMPLING_DIR / "output"
PAPER_DIR = SAMPLING_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

# Ensure directories exist
for d in [CACHE_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cache version for invalidation — bump this to force all caches to regenerate
# Versioned filenames (synthetic pkl, validation CSVs) auto-regenerate on bump.
# Network graphs (gla_graph.pkl, ground_truth_*.pkl) are unversioned and persist.
CACHE_VERSION = "v19"

# City configurations for OSM downloads
CITIES = {
    "london": {
        "name": "London (Soho)",
        "lng": -0.13396079424572427,
        "lat": 51.51371088849723,
        "buffer": 2000,
    },
    "madrid": {
        "name": "Madrid (Centro)",
        "lng": -3.7037902,
        "lat": 40.4167754,
        "buffer": 2000,
    },
    "phoenix": {
        "name": "Phoenix (Scottsdale)",
        "lng": -111.9260519,
        "lat": 33.4941704,
        "buffer": 2000,
    },
}

# Madrid regional network URL (for large-scale benchmarks)
MADRID_GPKG_URL = "https://github.com/songololo/ua-dataset-madrid/raw/main/data/street_network_w_edit.gpkg"

# Walking speed for time-distance conversions
SPEED_M_S = 1.33


# =============================================================================
# SECTION 2: Cache Management
# =============================================================================


def load_cache(name: str, version: str = CACHE_VERSION):
    """
    Load cached results if available.

    Parameters
    ----------
    name : str
        Cache key name
    version : str
        Cache version string for invalidation

    Returns
    -------
    Any or None
        Cached data if available and valid, None otherwise
    """
    path = CACHE_DIR / f"{name}_{version}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            path.unlink()  # Remove corrupted cache
            return None
    return None


def save_cache(name: str, data, version: str = CACHE_VERSION):
    """
    Save results to cache.

    Parameters
    ----------
    name : str
        Cache key name
    data : Any
        Data to cache
    version : str
        Cache version string
    """
    path = CACHE_DIR / f"{name}_{version}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


# =============================================================================
# SECTION 3: Network Loading
# =============================================================================


def download_osm_network(city_key: str, force: bool = False) -> dict:
    """
    Download OSM network for a city.

    Parameters
    ----------
    city_key : str
        Key from CITIES dict ('london', 'madrid', 'phoenix')
    force : bool
        If True, re-download even if cached

    Returns
    -------
    dict
        Contains 'nodes_gdf', 'edges_gdf', 'network_structure', 'n_nodes', 'n_edges'
    """
    from cityseer.tools import io

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


def load_madrid_regional_network(force: bool = False) -> tuple:
    """
    Load the Madrid metropolitan network from GitHub.

    This is a larger regional-scale network suitable for benchmarking
    at distances up to 20km.

    Parameters
    ----------
    force : bool
        If True, re-download even if cached

    Returns
    -------
    tuple
        (nodes_gdf, edges_gdf, network_structure)
    """
    from cityseer.tools import graphs, io

    cache_key = "network_madrid_regional"

    if not force:
        cached = load_cache(cache_key)
        if cached is not None:
            print("  Loaded Madrid regional network from cache")
            network_structure = io.network_structure_from_gpd(cached["nodes_gdf"], cached["edges_gdf"])
            return cached["nodes_gdf"], cached["edges_gdf"], network_structure

    print(f"  Downloading Madrid regional network from: {MADRID_GPKG_URL}")
    print("  (This may take a minute...)")

    # Load directly from GitHub URL
    edges_gdf = gpd.read_file(MADRID_GPKG_URL)
    print(f"    Downloaded: {len(edges_gdf)} edges, CRS: {edges_gdf.crs}")

    # Convert multipart geoms to single
    edges_gdf_singles = edges_gdf.explode(index_parts=False)

    # Generate networkx graph
    G_nx = io.nx_from_generic_geopandas(edges_gdf_singles)

    # Clean graph: remove degree-2 nodes and danglers
    G_nx = graphs.nx_remove_filler_nodes(G_nx)
    G = graphs.nx_remove_dangling_nodes(G_nx)

    print(f"    Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to cityseer network structure
    print("    Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, network_structure = io.network_structure_from_nx(G)
    print(f"    Network structure ready: {len(nodes_gdf)} nodes")

    # Cache the results
    cache_data = {
        "nodes_gdf": nodes_gdf,
        "edges_gdf": edges_gdf_out,
    }
    save_cache(cache_key, cache_data)

    return nodes_gdf, edges_gdf_out, network_structure


def load_network_from_legacy_cache(cache_path: Path) -> tuple:
    """
    Load a network from a legacy pickle cache file.

    Used for loading networks cached by older analysis scripts.

    Parameters
    ----------
    cache_path : Path
        Path to the pickle file containing nodes_gdf and edges_gdf

    Returns
    -------
    tuple
        (network_structure, nodes_gdf)
    """
    from cityseer.tools import io

    if not cache_path.exists():
        raise FileNotFoundError(f"Network cache not found at {cache_path}")

    print(f"  Loading network from {cache_path}...")
    with open(cache_path, "rb") as f:
        network_data = pickle.load(f)

    nodes_gdf = network_data["nodes_gdf"]
    edges_gdf = network_data["edges_gdf"]
    network_structure = io.network_structure_from_gpd(nodes_gdf, edges_gdf)

    print(f"    Loaded: {len(nodes_gdf)} nodes")
    return network_structure, nodes_gdf


# =============================================================================
# SECTION 4: Statistical Utilities
# =============================================================================


def compute_accuracy(true_vals: np.ndarray, est_vals: np.ndarray) -> float:
    """
    Compute Spearman correlation between true and estimated values.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth values
    est_vals : np.ndarray
        Estimated values

    Returns
    -------
    float
        Spearman correlation coefficient, or np.nan if insufficient data
    """
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan
    rho, _ = scipy_stats.spearmanr(true_vals[mask], est_vals[mask])
    return rho


def compute_accuracy_metrics(true_vals: np.ndarray, est_vals: np.ndarray) -> tuple:
    """
    Compute ranking and magnitude accuracy metrics.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth values
    est_vals : np.ndarray
        Estimated values

    Returns
    -------
    tuple
        (spearman, top_k_precision, scale_ratio, scale_iqr, max_abs_error)
    """
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    true_masked = true_vals[mask]
    est_masked = est_vals[mask]

    spearman, _ = scipy_stats.spearmanr(true_masked, est_masked)

    k = max(1, int(len(true_masked) * 0.1))
    true_top_k = set(np.argsort(true_masked)[-k:])
    est_top_k = set(np.argsort(est_masked)[-k:])
    top_k_precision = len(true_top_k & est_top_k) / k

    ratios = est_masked / true_masked
    scale_ratio = float(np.median(ratios))
    scale_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    max_abs_error = float(np.max(np.abs(true_masked - est_masked)))

    return spearman, top_k_precision, scale_ratio, scale_iqr, max_abs_error


def apply_live_buffer_nx(G: nx.MultiGraph, buffer_dist: float) -> nx.MultiGraph:
    """
    Mark only interior nodes as live on NetworkX graph.

    Applies an inward buffer from the convex hull of all nodes.
    Nodes inside the buffered zone are marked as live=True,
    nodes in the buffer zone are marked as live=False.

    Parameters
    ----------
    G : nx.MultiGraph
        NetworkX graph with 'x' and 'y' node attributes
    buffer_dist : float
        Inward buffer distance in metres

    Returns
    -------
    nx.MultiGraph
        Graph with 'live' attribute set on each node
    """
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    all_points = [Point(x, y) for x, y in coords]
    hull = unary_union(all_points).convex_hull
    live_zone = hull.buffer(-buffer_dist)

    for n in G.nodes():
        pt = Point(G.nodes[n]["x"], G.nodes[n]["y"])
        G.nodes[n]["live"] = live_zone.contains(pt)

    n_live = sum(1 for n in G.nodes() if G.nodes[n]["live"])
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    return G


def compute_morans_i(values: np.ndarray, coords: np.ndarray, k: int = 8) -> tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Uses k-nearest neighbors for spatial weights.

    Parameters
    ----------
    values : np.ndarray
        Values to test for spatial autocorrelation
    coords : np.ndarray
        Coordinates array of shape (n, 2)
    k : int
        Number of nearest neighbors for spatial weights

    Returns
    -------
    tuple[float, float]
        (Moran's I, p-value)
    """
    n = len(values)
    if n < 10:
        return np.nan, np.nan

    # Standardize values
    mean_val = np.mean(values)
    dev = values - mean_val
    var = np.var(values)
    if var == 0:
        return np.nan, np.nan

    # Build k-nearest neighbor weights
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # +1 includes self
    indices = indices[:, 1:]  # Remove self

    # Compute Moran's I
    numerator = 0.0
    weight_sum = 0.0

    for i in range(n):
        for j in indices[i]:
            numerator += dev[i] * dev[j]
            weight_sum += 1.0

    if weight_sum == 0:
        return np.nan, np.nan

    morans_i = (n / weight_sum) * (numerator / (n * var))

    # Expected value under null hypothesis
    expected_i = -1.0 / (n - 1)

    # Approximate variance under randomization assumption
    s1 = 2 * weight_sum
    s2 = 4 * k * n
    b2 = np.mean(dev**4) / var**2  # Kurtosis

    variance_i = (
        (n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * weight_sum**2))
        - b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * weight_sum**2)
    ) / ((n - 1) * (n - 2) * (n - 3) * weight_sum**2) - expected_i**2

    if variance_i <= 0:
        return morans_i, np.nan

    z_score = (morans_i - expected_i) / np.sqrt(variance_i)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))

    return morans_i, p_value


def spatial_sample(
    network_structure,
    n_samples: int,
    random_seed: int | None = None,
) -> tuple[list[int], float]:
    """
    Select spatially stratified sample of nodes.

    Uses grid-based stratification to ensure spatial coverage.

    Parameters
    ----------
    network_structure
        Network to sample from
    n_samples : int
        Number of nodes to sample
    random_seed : int | None
        Random seed for reproducibility

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
# SECTION 5: Model Utilities
# =============================================================================


def rho_model(eff_n: float, a: float, b: float) -> float:
    """
    Predict Spearman rho from effective sample size.

    Uses the hyperbolic model: rho = 1 - A / (B + eff_n)

    Parameters
    ----------
    eff_n : float
        Effective sample size (reach × p)
    a : float
        Model parameter A
    b : float
        Model parameter B

    Returns
    -------
    float
        Predicted Spearman rho
    """
    return 1 - a / (b + eff_n)


def required_eff_n_for_rho(target_rho: float, a: float, b: float) -> float:
    """
    Compute the effective_n required to achieve target rho.

    Inverts the model: eff_n = A / (1 - rho) - B

    Parameters
    ----------
    target_rho : float
        Target accuracy (Spearman rho)
    a : float
        Model parameter A
    b : float
        Model parameter B

    Returns
    -------
    float
        Required effective sample size
    """
    if target_rho >= 1.0:
        return float("inf")
    return a / (1 - target_rho) - b
