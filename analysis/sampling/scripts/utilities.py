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

import math
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
# Network graphs (gla_graph.pkl, gla_ground_truth_*.pkl) are unversioned and persist.
CACHE_VERSION = "v26"

# Canonical per-quartile key prefixes — single source of truth for fallback/perfect blocks
QUARTILE_KEYS = ("spearman", "mae", "max_error", "reach")

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


def compute_quartile_accuracy(
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    node_reach: np.ndarray,
) -> dict:
    """
    Compute accuracy metrics within reachability quartiles.

    Splits nodes into quartiles by their individual reachability and computes
    Spearman correlation and error metrics within each quartile, testing whether
    low-reachability nodes have systematically worse accuracy.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth centrality values (per node)
    est_vals : np.ndarray
        Estimated centrality values (per node)
    node_reach : np.ndarray
        Per-node reachability (node density at this distance)

    Returns
    -------
    dict
        Keys per quartile (q1-q4):
          spearman_q{i}    - Spearman rho
          mae_q{i}         - median absolute error
          max_error_q{i}   - max absolute error
          reach_q{i}       - median reach in quartile
    """
    nan_result = {}
    for q in range(1, 5):
        nan_result[f"spearman_q{q}"] = np.nan
        nan_result[f"mae_q{q}"] = np.nan
        nan_result[f"max_error_q{q}"] = np.nan
        nan_result[f"reach_q{q}"] = np.nan

    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals) & np.isfinite(node_reach)
    if mask.sum() < 40:
        return nan_result

    true_m = true_vals[mask]
    est_m = est_vals[mask]
    reach_m = node_reach[mask]

    quartile_edges = np.percentile(reach_m, [0, 25, 50, 75, 100])
    result = {}

    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        q_mask = (reach_m >= lo) & (reach_m < hi) if q < 3 else (reach_m >= lo) & (reach_m <= hi)

        if q_mask.sum() < 10:
            result[f"spearman_q{q + 1}"] = np.nan
            result[f"mae_q{q + 1}"] = np.nan
            result[f"max_error_q{q + 1}"] = np.nan
            result[f"reach_q{q + 1}"] = np.nan
        else:
            abs_errors = np.abs(true_m[q_mask] - est_m[q_mask])
            rho, _ = scipy_stats.spearmanr(true_m[q_mask], est_m[q_mask])
            result[f"spearman_q{q + 1}"] = rho
            result[f"mae_q{q + 1}"] = float(np.median(abs_errors))
            result[f"max_error_q{q + 1}"] = float(np.max(abs_errors))
            result[f"reach_q{q + 1}"] = float(np.median(reach_m[q_mask]))

    return result


# =============================================================================
# SECTION 4b: Betweenness Spatial Sampling
# =============================================================================

# Betweenness convergence parameters
BETW_RHO_THRESHOLD = 0.999  # Stop when ρ(current, previous) exceeds this
BETW_MAX_ROUNDS = 30  # Safety cap
BETW_MIN_ROUNDS = 3  # Always run at least this many rounds
BETW_BASE_SEED = 42


def build_spatial_grid(net, cell_size: float) -> dict[tuple[int, int], list[int]]:
    """Partition live nodes into spatial grid cells.

    Parameters
    ----------
    net
        cityseer NetworkStructure
    cell_size : float
        Grid cell size in metres (typically distance / 2)

    Returns
    -------
    dict[tuple[int, int], list[int]]
        Mapping from (cx, cy) grid cell to list of live node indices
    """
    all_xs = net.node_xs
    all_ys = net.node_ys
    live_indices = [i for i in net.node_indices() if net.is_node_live(i)]

    if not live_indices:
        return {}

    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]
    x_min = min(xs)
    y_min = min(ys)

    cells: dict[tuple[int, int], list[int]] = {}
    for idx, x, y in zip(live_indices, xs, ys):
        cx = int((x - x_min) / cell_size)
        cy = int((y - y_min) / cell_size)
        cells.setdefault((cx, cy), []).append(idx)

    return cells


class CellSampler:
    """Without-replacement sampler across grid cells.

    Each cell maintains a shuffled deck of node indices. Each call to
    ``sample_round()`` draws one node per cell. When a cell's deck is
    exhausted it reshuffles — so every node is used before any repeats.
    """

    def __init__(self, cells: dict[tuple[int, int], list[int]], rng):
        import random as _random

        self.rng: _random.Random = rng
        self.decks: dict[tuple[int, int], list[int]] = {}
        self.pools: dict[tuple[int, int], list[int]] = {}
        for key, nodes in cells.items():
            if nodes:
                self.pools[key] = list(nodes)
                deck = list(nodes)
                rng.shuffle(deck)
                self.decks[key] = deck

    def sample_round(self) -> list[int]:
        """Draw one node per cell (without replacement until exhausted)."""
        sources: list[int] = []
        for key, deck in self.decks.items():
            if not deck:
                deck = list(self.pools[key])
                self.rng.shuffle(deck)
                self.decks[key] = deck
            sources.append(deck.pop())
        return sources


def select_spatial_sources(
    net,
    n_sources: int,
    cell_size: float,
    rng,
) -> list[int]:
    """Select spatially distributed live nodes using grid stratification.

    Partitions live nodes into grid cells of ``cell_size`` metres and draws
    from each cell in round-robin order (without replacement within each
    cell) until ``n_sources`` nodes have been selected.

    Parameters
    ----------
    net
        cityseer NetworkStructure
    n_sources : int
        Number of source nodes to select
    cell_size : float
        Grid cell size in metres (typically distance / 2)
    rng : random.Random
        Random number generator for reproducibility

    Returns
    -------
    list[int]
        Selected live node indices (length == min(n_sources, n_live))
    """
    import random as _random

    cells = build_spatial_grid(net, cell_size)
    if not cells:
        return []

    # Build shuffled decks per cell
    decks: dict[tuple[int, int], list[int]] = {}
    for key, nodes in cells.items():
        if nodes:
            deck = list(nodes)
            rng.shuffle(deck)
            decks[key] = deck

    cell_keys = list(decks.keys())
    rng.shuffle(cell_keys)

    selected: list[int] = []
    while len(selected) < n_sources and decks:
        empty_keys: list[tuple[int, int]] = []
        for key in cell_keys:
            if key not in decks:
                continue
            deck = decks[key]
            if not deck:
                empty_keys.append(key)
                continue
            selected.append(deck.pop())
            if len(selected) >= n_sources:
                break
        # Remove exhausted cells
        for key in empty_keys:
            del decks[key]
        # If all cells are exhausted, stop
        if not any(decks.get(k) for k in cell_keys if k in decks):
            break

    return selected


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


def mean_quartiles(quartile_list: list[dict], quartile_keys: tuple = QUARTILE_KEYS) -> dict:
    """
    Average quartile accuracy results across multiple runs.

    Parameters
    ----------
    quartile_list : list[dict]
        List of quartile dicts from compute_quartile_accuracy
    quartile_keys : tuple
        Quartile key prefixes (default: QUARTILE_KEYS)

    Returns
    -------
    dict
        Averaged quartile results
    """
    if not quartile_list:
        result = {}
        for prefix in quartile_keys:
            for q in range(1, 5):
                result[f"{prefix}_q{q}"] = np.nan
        return result
    result = {}
    for key in quartile_list[0]:
        vals = [q[key] for q in quartile_list if not np.isnan(q[key])]
        result[key] = float(np.mean(vals)) if vals else np.nan
    return result


# =============================================================================
# SECTION 6: Hoeffding / EW Bound Utilities
# =============================================================================

# Default parameters for the Hoeffding/EW bound
HOEFFDING_EPSILON = 0.05  # Normalised additive error tolerance
HOEFFDING_DELTA = 0.1  # Failure probability (90% confidence)


def compute_hoeffding_p(
    reach: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute sampling probability from the Hoeffding/EW bound.

    k = log(2r / delta) / (2 * epsilon^2)
    p = min(1, k / r)

    Parameters
    ----------
    reach : float
        Mean network reach (nodes within distance)
    epsilon : float
        Normalised additive error tolerance
    delta : float
        Failure probability

    Returns
    -------
    float
        Recommended sampling probability in [0, 1]
    """
    if reach <= 0 or epsilon <= 0:
        return 1.0
    k = math.log(2 * reach / delta) / (2 * epsilon**2)
    return min(1.0, k / reach)


def compute_hoeffding_eff_n(
    reach: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute effective sample size from the Hoeffding/EW bound.

    k = log(2r / delta) / (2 * epsilon^2)

    Parameters
    ----------
    reach : float
        Mean network reach (nodes within distance)
    epsilon : float
        Normalised additive error tolerance
    delta : float
        Failure probability

    Returns
    -------
    float
        Required effective sample size
    """
    if reach <= 0 or epsilon <= 0:
        return reach
    return math.log(2 * reach / delta) / (2 * epsilon**2)


def ew_predicted_epsilon(
    n_eff: float,
    reach: float,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute the EW-predicted maximum normalised epsilon.

    eps = sqrt(log(2r / delta) / (2 * n_eff))

    Parameters
    ----------
    n_eff : float
        Effective sample size
    reach : float
        Mean network reach
    delta : float
        Failure probability

    Returns
    -------
    float
        Predicted maximum additive error
    """
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))
