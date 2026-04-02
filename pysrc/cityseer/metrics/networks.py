r"""
Compute network centralities.

Two centrality methods are available, using shortest-path (metric) or simplest-path (angular) heuristics:

- [`centrality_shortest`](#centrality-shortest)
- [`centrality_simplest`](#centrality-simplest)

Metrics are specified as ``{name: expression}`` dicts using variables ``c`` (cost) and ``p`` (normalised
progress). For shortest paths, ``c`` is metric distance and ``p = c / threshold``. For simplest paths,
``c`` is angular cost and ``p`` is normalised time progress.

Four categories of metrics are supported:

- **closeness**: per-reached-node accumulation (e.g. ``{"harmonic": "1/c", "density": "1"}``)
- **betweenness**: target seed weight in Brandes backpropagation (e.g. ``{"betweenness": "1"}``)
- **cycles**: circuit rank (boolean flag)
- **postprocess**: derived from computed columns in Python (e.g. ``{"hillier": "density**2 / farness"}``)

Pass ``None`` for defaults or ``{}`` to skip a category.

When `segment_weighted=True`, node weights are set to the primal edge (street segment) lengths so that centrality
measures reflect total reachable street length rather than node counts. This requires a dual graph representation.

When `sample=True`, only a subset of nodes are used as sources for centrality computation, with results
corrected to approximate the full computation.

:::note
The reasons for picking one approach over another are varied:

- Centralities can be distorted by messy graph topologies such as unnecessary intermediate points along streets
(used to describe road curvature) or overly complex representations of street intersections. Clean the network
first using the [`graph`](/tools/graphs) module (see the
[automatic graph cleaning](/guide#automatic-graph-cleaning) for examples).
- `harmonic` centrality can produce inflated values when nodes are very close together, because the
inverse-distance calculation amplifies small distances. This is more likely with simplest-path measures or short
distance thresholds.
- Simplest (angular) measures require a dual graph representation. Convert primal graphs with
  [`graphs.nx_to_dual`](/tools/graphs#nx-to-dual) before ingesting them.
- Metrics should only be compared across networks that use the same graph representation (both primal or both
dual), because the differing number of nodes and edges between representations affects the metric values. For
example, a four-way intersection consisting of one node with four edges on a primal graph translates to four
nodes and six edges on the dual. This effect is amplified for denser regions of the network.
- Standard closeness and normalised closeness do not work well with distance-bounded analysis. Use harmonic
closeness or Hillier normalisation instead.
:::

"""

from __future__ import annotations

import logging
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import config, rustalgos, sampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# separate out so that ast parser can parse function def
SPEED_M_S = config.SPEED_M_S


def _require_dual_for_angular(
    network_structure: rustalgos.graph.NetworkStructure,
    context: str,
) -> None:
    if not network_structure.is_dual:
        raise ValueError(
            f"{context} requires a dual graph for angular analysis. "
            "Convert the graph with cityseer.tools.graphs.nx_to_dual(...) before ingesting it."
        )


class _SegmentWeightContext:
    """Context manager that sets node weights to segment lengths and restores them on exit."""

    def __init__(
        self,
        network_structure: rustalgos.graph.NetworkStructure,
        nodes_gdf: gpd.GeoDataFrame,
        segment_weighted: bool,
    ):
        self.network_structure = network_structure
        self.segment_weighted = segment_weighted
        self._saved_weights: list[tuple[int, float]] | None = None
        if segment_weighted:
            if not network_structure.is_dual:
                raise ValueError("segment_weighted requires a dual graph where each node represents a street segment.")
            if "primal_edge" not in nodes_gdf.columns:
                raise ValueError("segment_weighted requires primal_edge geometries in nodes_gdf (from a dual graph).")
            node_idxs = network_structure.node_indices()
            self._saved_weights = [(i, network_structure.get_node_weight(i)) for i in node_idxs]
            ns_indices = nodes_gdf["ns_node_idx"].values
            seg_lengths = nodes_gdf["primal_edge"].length.values
            for ns_idx, seg_len in zip(ns_indices, seg_lengths, strict=True):
                network_structure.set_node_weight(int(ns_idx), float(seg_len))

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if self._saved_weights is not None:
            for idx, w in self._saved_weights:
                self.network_structure.set_node_weight(idx, w)


DEFAULT_SHORTEST_CLOSENESS = {"density": "1", "farness": "c", "harmonic": "1/c", "decay": "exp(-4 * p)"}
DEFAULT_SHORTEST_BETWEENNESS = {"betweenness": "1", "betweenness_decay": "exp(-4 * p)"}
DEFAULT_SHORTEST_POSTPROCESS = {"hillier": "density**2 / farness"}

DEFAULT_SIMPLEST_CLOSENESS = {"density": "1", "farness": "1 + c / 90", "harmonic": "1 / (1 + c / 90)"}
DEFAULT_SIMPLEST_BETWEENNESS = {"betweenness": "1"}
DEFAULT_SIMPLEST_POSTPROCESS = {"hillier": "density**2 / farness"}


def _safe_eval(expr: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate a simple arithmetic expression over named numpy arrays.

    Only allows: variable references, numeric literals, and the operators
    +, -, *, /, ** (power). No function calls, attribute access, subscripts,
    or other Python constructs are permitted.

    Raises ValueError if the expression contains disallowed constructs or
    references undefined variables.
    """
    import ast

    tree = ast.parse(expr, mode="eval")

    def _eval_node(node: ast.AST) -> np.ndarray:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Unsupported operator in postprocess expression: {ast.dump(node.op)}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError(f"Unsupported unary operator in postprocess expression: {ast.dump(node.op)}")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise NameError(f"Unknown variable in postprocess expression: '{node.id}'")
            return variables[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return np.full_like(next(iter(variables.values())), node.value, dtype=np.float64)
        raise ValueError(f"Disallowed construct in postprocess expression: {ast.dump(node)}")

    return _eval_node(tree)


def _extract_results(
    results: dict[int, rustalgos.centrality.CentralityResult],
    nodes_gdf: gpd.GeoDataFrame,
    postprocess: dict[str, str],
    angular: bool = False,
) -> gpd.GeoDataFrame:
    """Extract metrics from CentralityResult objects into GeoDataFrame columns."""
    if not results:
        return nodes_gdf
    ref_result = next(iter(results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)
    temp_data: dict[str, object] = {}
    for d, res in results.items():
        metrics = res.metrics
        for name, per_dist in metrics.items():
            if d in per_dist:
                data_key = config.prep_gdf_key(name, d, angular=angular)
                temp_data[data_key] = per_dist[d]
    # Postprocessing: evaluate expressions over computed columns
    if postprocess:
        all_distances = list(results.keys())
        for pp_name, pp_expr in postprocess.items():
            for d in all_distances:
                namespace: dict[str, np.ndarray] = {}
                for name in next(iter(results.values())).metrics:
                    col_key = config.prep_gdf_key(name, d, angular=angular)
                    val = temp_data.get(col_key)
                    if val is not None:
                        namespace[name] = val  # type: ignore[assignment]
                try:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        data_key = config.prep_gdf_key(pp_name, d, angular=angular)
                        temp_data[data_key] = _safe_eval(pp_expr, namespace)
                except NameError:
                    pass  # skip if dependencies not computed
    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def centrality_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    closeness: dict[str, str] | None = None,
    betweenness: dict[str, str] | None = None,
    cycles: bool = True,
    postprocess: dict[str, str] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    segment_weighted: bool = False,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute centrality using shortest paths with a single Dijkstra per source.

    Metrics are specified as ``{name: expression}`` dicts. Expressions use two variables:

    - ``c``: the raw cost (metric distance in metres for shortest-path analysis)
    - ``p``: normalised progress from 0 at the source to 1 at the distance threshold (``p = c / threshold``)

    Pass ``None`` for defaults or ``{}`` to skip a category.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.
    distances: list[int]
        Distance thresholds in metres at which to compute centrality measures.
    minutes: list[float]
        Walking times in minutes; converted to distance thresholds using `speed_m_s`.
    closeness: dict[str, str]
        Closeness metric expressions. Each entry is ``{name: expr(c, p)}``, accumulated per
        reached node. ``None`` uses defaults: density, farness, harmonic, decay.
    betweenness: dict[str, str]
        Betweenness metric expressions. Each entry is ``{name: expr(c, p)}``, used as the weight
        assigned to each destination when accumulating betweenness contributions along shortest
        paths. ``None`` uses defaults: betweenness, betweenness_decay.
    cycles: bool
        If True, compute circuit rank (cycle count) for each node. Default True.
    postprocess: dict[str, str]
        Derived metrics computed in Python from the closeness/betweenness results.
        ``None`` uses default: ``{"hillier": "density**2 / farness"}``.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for betweenness path equality, as a percentage (e.g. 1.0 = 1%).
    segment_weighted: bool
        If True, weight by primal edge (street segment) lengths. Requires a dual graph.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, enables adaptive sampling at longer distance thresholds.
    epsilon: float
        Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.06).

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional centrality columns.

    Examples
    --------
    ```python
    from cityseer.tools import mock, graphs, io
    from cityseer.metrics import networks

    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
    nodes_gdf = networks.centrality_shortest(
        network_structure,
        nodes_gdf,
        distances=[400, 800],
    )
    print(nodes_gdf[["cc_harmonic_400", "cc_betweenness_800"]])
    ```
    """
    logger.info("Computing centrality (shortest).")
    if closeness is None:
        closeness = dict(DEFAULT_SHORTEST_CLOSENESS)
    if betweenness is None:
        betweenness = dict(DEFAULT_SHORTEST_BETWEENNESS)
    if postprocess is None:
        postprocess = dict(DEFAULT_SHORTEST_POSTPROCESS)
    closeness_items = list(closeness.items())
    betweenness_items = list(betweenness.items())

    resolved_distances, _seconds = rustalgos.pair_distances_and_time(speed_m_s, distances, minutes)
    node_count = network_structure.street_node_count()

    eps = epsilon if epsilon is not None else sampling.HOEFFDING_EPSILON
    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        lives = network_structure.node_lives
        live_fraction = sum(lives) / len(lives) if lives else 1.0
        for d in sorted(resolved_distances):
            p = sampling.compute_distance_p(d, epsilon=eps)
            if p >= live_fraction:
                full_distances.append(d)
            else:
                sampled_distances.append((d, p))

    results: dict[int, rustalgos.centrality.CentralityResult] = {}

    with _SegmentWeightContext(network_structure, nodes_gdf, segment_weighted):
        if full_distances:
            dist_label = ", ".join(f"{d}m" for d in full_distances)
            logger.info(f"  Full: {dist_label}")
            partial_func = partial(
                network_structure.centrality_shortest,
                distances=full_distances,
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                compute_cycles=cycles,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                segment_weighted=segment_weighted,
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality full: {dist_label}",
            )
            for d in full_distances:
                results[d] = result  # type: ignore

        for d, p in sampled_distances:
            logger.info(f"  Sampled {d}m: p={p:.0%}")
            partial_func = partial(
                network_structure.centrality_shortest,
                distances=[d],
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                compute_cycles=cycles,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                segment_weighted=segment_weighted,
                sample_probability=p,
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality p={p:.0%}: {d}m",
            )
            results[d] = result  # type: ignore

    return _extract_results(results, nodes_gdf, postprocess)


def build_od_matrix(
    od_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    origin_col: str,
    destination_col: str,
    weight_col: str,
    zone_id_col: str | None = None,
    max_snap_dist: float = 500.0,
) -> rustalgos.centrality.OdMatrix:
    """Build an OdMatrix from OD flow data and zone boundaries.

    Computes zone centroids, snaps them to the nearest network nodes,
    and constructs a sparse OD weight matrix for use with `betweenness_od`.

    Parameters
    ----------
    od_df : pd.DataFrame
        Origin-destination flow data with columns for origin zone, destination zone, and weight.
    zones_gdf : gpd.GeoDataFrame
        Zone boundaries (polygons) or centroids (points). Must be in a projected CRS
        matching the network, or in EPSG:4326 (will be auto-reprojected).
    network_structure : rustalgos.graph.NetworkStructure
        The network to snap zone centroids to.
    origin_col : str
        Column in od_df containing origin zone identifiers.
    destination_col : str
        Column in od_df containing destination zone identifiers.
    weight_col : str
        Column in od_df containing trip weights (e.g., number of bicycle commuters).
    zone_id_col : str | None
        Column in zones_gdf containing zone identifiers matching origin_col/destination_col.
        If None, uses the GeoDataFrame index.
    max_snap_dist : float
        Maximum distance (in CRS units, typically metres) for snapping a centroid to a network node.
        Centroids beyond this distance are excluded with a warning.

    Returns
    -------
    rustalgos.centrality.OdMatrix
        Sparse OD matrix ready for use with `betweenness_od`.
    """
    from scipy.spatial import KDTree

    geom_types = set(zones_gdf.geometry.geom_type)
    centroids = zones_gdf.geometry.centroid if geom_types & {"Polygon", "MultiPolygon"} else zones_gdf.geometry

    zones_work = zones_gdf.copy()
    zones_work["_centroid"] = centroids
    if zones_work.crs is not None and zones_work.crs.to_epsg() == 4326:
        node_xys = network_structure.node_xys
        mean_x = np.mean([xy[0] for xy in node_xys[:100]])
        target_crs = 27700 if 100_000 < mean_x < 700_000 else 32630
        logger.info(f"Reprojecting zone centroids from EPSG:4326 to EPSG:{target_crs}")
        centroid_gdf = gpd.GeoDataFrame({"geometry": zones_work["_centroid"]}, crs=zones_work.crs)  # type: ignore
        centroid_gdf = centroid_gdf.to_crs(epsg=target_crs)
        zones_work["_centroid"] = centroid_gdf.geometry

    zone_ids = zones_work[zone_id_col].values if zone_id_col is not None else zones_work.index.values
    centroid_coords = np.array([(g.x, g.y) for g in zones_work["_centroid"]])

    # Snap centroids to nearest network nodes via KDTree
    node_xys = network_structure.node_xys
    tree = KDTree(node_xys)
    distances_snap, indices = tree.query(centroid_coords)

    zone_to_node: dict = {}
    n_excluded = 0
    for i, zone_id in enumerate(zone_ids):
        if distances_snap[i] > max_snap_dist:
            n_excluded += 1
            continue
        zone_to_node[zone_id] = int(indices[i])

    if n_excluded > 0:
        logger.warning(f"{n_excluded} zone centroids exceeded max_snap_dist={max_snap_dist}m and were excluded")
    logger.info(
        f"Snapped {len(zone_to_node)} zone centroids to network nodes "
        f"(median distance: {np.median(distances_snap):.0f}m)"
    )

    # Build COO arrays
    origins_arr: list[int] = []
    dests_arr: list[int] = []
    weights_arr: list[float] = []

    for _, row in od_df.iterrows():
        o_zone = row[origin_col]
        d_zone = row[destination_col]
        w = row[weight_col]

        if pd.isna(w) or w <= 0:
            continue
        if o_zone not in zone_to_node or d_zone not in zone_to_node:
            continue

        origins_arr.append(zone_to_node[o_zone])
        dests_arr.append(zone_to_node[d_zone])
        weights_arr.append(float(w))

    logger.info(f"Built OD matrix: {len(origins_arr)} pairs, {sum(weights_arr):.0f} total trips")

    return rustalgos.centrality.OdMatrix(origins_arr, dests_arr, weights_arr)


def betweenness_od(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    od_matrix: rustalgos.centrality.OdMatrix,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    betweenness: dict[str, str] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute OD-weighted betweenness centrality using the shortest path heuristic.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.
    od_matrix
        An [`OdMatrix`](/rustalgos/centrality#odmatrix) mapping (origin, destination) node pairs to trip weights.
        Build with [`build_od_matrix`](/metrics/networks#build-od-matrix).
    distances: list[int]
        Distance thresholds in metres at which to compute betweenness.
    minutes: list[float]
        Walking times in minutes; converted to distance thresholds using `speed_m_s`.
    betweenness: dict[str, str]
        Betweenness metric expressions. ``None`` uses defaults: betweenness, betweenness_decay.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for path equality, as a percentage.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional betweenness columns.

    """
    logger.info("Computing OD-weighted betweenness centrality.")
    if betweenness is None:
        betweenness = dict(DEFAULT_SHORTEST_BETWEENNESS)
    betweenness_items = list(betweenness.items())
    partial_func = partial(
        network_structure.betweenness_od_shortest,
        od_matrix=od_matrix,
        distances=distances,
        minutes=minutes,
        betweenness_exprs=betweenness_items,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
    )
    result: rustalgos.centrality.CentralityResult = config.wrap_progress(  # type: ignore[assignment]
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    resolved_distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    results = {d: result for d in resolved_distances}
    return _extract_results(results, nodes_gdf, {})


def centrality_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    closeness: dict[str, str] | None = None,
    betweenness: dict[str, str] | None = None,
    postprocess: dict[str, str] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    segment_weighted: bool = False,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute centrality using simplest (angular) paths with a single Dijkstra per source.

    Expressions use ``c`` (angular cost) and ``p`` (normalised time progress).

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.
    distances: list[int]
        Distance thresholds in metres at which to compute centrality measures.
    minutes: list[float]
        Walking times in minutes; converted to distance thresholds using `speed_m_s`.
    closeness: dict[str, str]
        Closeness metric expressions. ``None`` uses defaults: density, farness, harmonic.
    betweenness: dict[str, str]
        Betweenness metric expressions. ``None`` uses defaults: betweenness.
    postprocess: dict[str, str]
        Derived metrics. ``None`` uses default: ``{"hillier": "density**2 / farness"}``.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for angular betweenness path equality, as a percentage.
    segment_weighted: bool
        If True, weight by primal edge (street segment) lengths. Requires a dual graph.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, enables adaptive sampling at longer distance thresholds.
    epsilon: float
        Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.06).

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional centrality columns.

    Examples
    --------
    ```python
    from cityseer.tools import mock, graphs, io
    from cityseer.metrics import networks

    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    G_dual = graphs.nx_to_dual(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G_dual)
    nodes_gdf = networks.centrality_simplest(
        network_structure,
        nodes_gdf,
        distances=[400, 800],
    )
    print(nodes_gdf[["cc_harmonic_400_ang", "cc_betweenness_800_ang"]])
    ```
    """
    _require_dual_for_angular(network_structure, "centrality_simplest")
    logger.info("Computing centrality (simplest).")
    if closeness is None:
        closeness = dict(DEFAULT_SIMPLEST_CLOSENESS)
    if betweenness is None:
        betweenness = dict(DEFAULT_SIMPLEST_BETWEENNESS)
    if postprocess is None:
        postprocess = dict(DEFAULT_SIMPLEST_POSTPROCESS)
    closeness_items = list(closeness.items())
    betweenness_items = list(betweenness.items())

    resolved_distances, _seconds = rustalgos.pair_distances_and_time(speed_m_s, distances, minutes)
    node_count = network_structure.street_node_count()

    eps = epsilon if epsilon is not None else sampling.HOEFFDING_EPSILON
    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        lives = network_structure.node_lives
        live_fraction = sum(lives) / len(lives) if lives else 1.0
        for d in sorted(resolved_distances):
            p = sampling.compute_distance_p(d, epsilon=eps)
            if p >= live_fraction:
                full_distances.append(d)
            else:
                sampled_distances.append((d, p))

    results: dict[int, rustalgos.centrality.CentralityResult] = {}

    with _SegmentWeightContext(network_structure, nodes_gdf, segment_weighted):
        if full_distances:
            dist_label = ", ".join(f"{d}m" for d in full_distances)
            logger.info(f"  Full: {dist_label}")
            partial_func = partial(
                network_structure.centrality_simplest,
                distances=full_distances,
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                segment_weighted=segment_weighted,
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality simplest full: {dist_label}",
            )
            for d in full_distances:
                results[d] = result  # type: ignore

        for d, p in sampled_distances:
            logger.info(f"  Sampled {d}m: p={p:.0%}")
            partial_func = partial(
                network_structure.centrality_simplest,
                distances=[d],
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                segment_weighted=segment_weighted,
                sample_probability=p,
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality simplest p={p:.0%}: {d}m",
            )
            results[d] = result  # type: ignore

    return _extract_results(results, nodes_gdf, postprocess, angular=True)


# =============================================================================
# Convenience wrappers — closeness-only and betweenness-only
# =============================================================================


def closeness_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using shortest paths.

    Wraps `centrality_shortest` with betweenness disabled.
    """
    return centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        betweenness={},
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def closeness_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using simplest (angular) paths.

    Wraps `centrality_simplest` with betweenness disabled.
    """
    return centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        betweenness={},
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def betweenness_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using shortest paths.

    Wraps `centrality_shortest` with closeness disabled.
    """
    return centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        closeness={},
        cycles=False,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def betweenness_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using simplest (angular) paths.

    Wraps `centrality_simplest` with closeness disabled.
    """
    return centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        closeness={},
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )
