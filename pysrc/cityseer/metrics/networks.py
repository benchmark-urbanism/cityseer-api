r"""
Compute network centralities.

Two centrality methods are available, using shortest-path (metric) or simplest-path (angular) heuristics:

- [`centrality_shortest`](#centrality_shortest)
- [`centrality_simplest`](#centrality_simplest)

Metrics are specified as ``{name: expression}`` dicts using variables ``c`` (cost) and ``p`` (normalised
progress). For shortest paths, ``c`` is metric distance and ``p = c / threshold``. For simplest paths,
``c`` is angular cost and ``p`` is normalised time progress.

Four categories of metrics are supported:

- **closeness**: per-reached-node accumulation (e.g. ``{"harmonic": "1/c", "density": "1"}``)
- **betweenness**: target seed weight in Brandes backpropagation (e.g. ``{"betweenness": "1"}``)
- **cycles**: circuit rank (boolean flag)
- **postprocess**: derived from computed columns in Python (e.g. ``{"hillier": "density**2 / farness"}``)

Pass ``None`` for defaults or ``{}`` to skip a category.

Per-node ``weight`` values (default ``1.0``, set on the nodes ``GeoDataFrame`` or read from NetworkX node
attributes) apply gravity-style weighting to centrality: closeness weights each reachable node by its destination
weight (so ``density`` becomes ``sum_j w_j`` rather than a plain count), and betweenness weights each
origin-destination pair by the product of its endpoint weights. The same weighting is applied identically whether
or not sampling is used. Land-use, mixed-use, and statistical aggregations are intentionally *not* node-weighted.

When `segment_weighted=True`, node weights are temporarily set to the primal edge (street segment) lengths so that
centrality measures reflect total reachable street length rather than node counts (closeness by destination length,
betweenness by the product of endpoint lengths). This is a convenience preset over the per-node ``weight``
mechanism and requires a dual graph representation.

When `sample=True`, only a subset of nodes are used as sources for centrality computation, with results
corrected to approximate the full computation.

:::note
The reasons for picking one approach over another are varied:

- Columns prefixed ``cc_`` are managed by cityseer: recomputing a metric for the same distance overwrites the
matching ``cc_`` columns in place (intended for re-runs). Don't store your own data under this prefix.
- Centralities can be distorted by messy graph topologies such as unnecessary intermediate points along streets
(used to describe road curvature) or overly complex representations of street intersections. Clean the network
first using the [`graph`](/tools/graphs) module (see the
[automatic graph cleaning](/guide/fundamentals#automatic-graph-cleaning) for examples).
- `harmonic` centrality can produce inflated values when nodes are very close together, because the
inverse-distance calculation amplifies small distances. This is more likely with simplest-path measures or short
distance thresholds.
- Simplest (angular) measures require a dual graph representation. Convert primal graphs with
  [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) before ingesting them.
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
import warnings
from functools import partial
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import config, rustalgos, sampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# separate out so that ast parser can parse function def
SPEED_M_S = config.SPEED_M_S
MIN_THRESH_WT = config.MIN_THRESH_WT


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


def _plan_adaptive_sampling(
    network_structure: rustalgos.graph.NetworkStructure,
    resolved_distances: list[int],
    epsilon: float,
    has_betweenness: bool,
    random_seed: int | None = None,
) -> tuple[list[int], list[tuple[int, np.ndarray]]]:
    """Split distances into exact and sampled, with per-node inclusion probabilities.

    A pilot polls the network (sampled sources, one bounded Dijkstra each; see
    ``cityseer.sampling.estimate_polled_reach``) to measure per-node reach at every distance,
    respecting barriers and dead ends that a Euclidean count would miss. Per-node probabilities
    ``q = min(1, k(r)/r)`` derive from the lower confidence bound on reach, so estimation error
    lands on the oversampling side. Sparse areas receive high probabilities and dense areas low
    ones, so every catchment accumulates approximately the Hoeffding-required number of
    effective samples. A distance is sampled only when the estimated sampled work
    (``sum(q * r)``) undercuts exact work (``sum(r)`` over live nodes for closeness-only calls;
    over all nodes when betweenness is requested, since exact betweenness sources every node).
    Work sums use the upper confidence bound on reach: a node the poll never hit is censused
    (q = 1), so pricing it at its point estimate of zero would hide real traversal cost and
    select sampling where exact computation is cheaper. Sampling engages only when predicted
    work clears ``WORK_TEST_MARGIN``, since the work model omits constant overheads.
    """
    full_distances: list[int] = []
    sampled: list[tuple[int, np.ndarray]] = []
    lives = np.asarray(network_structure.node_lives, dtype=bool)
    reach_lcb, _reach_point, reach_ucb = sampling.estimate_polled_reach(
        network_structure, sorted(resolved_distances), random_seed=random_seed
    )
    for d in sorted(resolved_distances):
        q = sampling.compute_node_p(reach_lcb[d], epsilon=epsilon)
        reach_est = reach_ucb[d]
        sampled_work = float(np.sum(q * reach_est))
        exact_work = float(np.sum(reach_est)) if has_betweenness else float(np.sum(reach_est[lives]))
        if sampled_work >= sampling.WORK_TEST_MARGIN * exact_work:
            full_distances.append(d)
        else:
            sampled.append((d, q))
    return full_distances, sampled


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
                    if col_key in temp_data:
                        namespace[name] = cast(np.ndarray, temp_data[col_key])
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

    Tip: compute only what you need — a smaller ``closeness`` / ``betweenness`` dict, ``{}`` to skip a whole
    category, or ``cycles=False`` — evaluates fewer expressions and emits fewer columns.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).
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
        Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.05).

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
    sampled_distances: list[tuple[int, np.ndarray]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        full_distances, sampled_distances = _plan_adaptive_sampling(
            network_structure, resolved_distances, eps, bool(betweenness_items), random_seed=random_seed
        )

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
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality full: {dist_label}",
            )
            for d in full_distances:
                results[d] = result

        for d, q in sampled_distances:
            mean_q = float(np.mean(q))
            logger.info(f"  Sampled {d}m: mean q={mean_q:.0%}")
            partial_func = partial(
                network_structure.centrality_shortest,
                distances=[d],
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                compute_cycles=cycles,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                sample_probability=1.0,
                sampling_weights=[float(v) for v in q],
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality q~{mean_q:.0%}: {d}m",
            )
            results[d] = result

    return _extract_results(results, nodes_gdf, postprocess)


def _snap_coords_to_nodes(
    network_structure: rustalgos.graph.NetworkStructure,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap ``(x, y)`` coordinates to their nearest network node via a KDTree.

    Returns ``(node_indices, snap_distances)`` aligned to the input rows.
    """
    from scipy.spatial import KDTree

    tree = KDTree(network_structure.node_xys)
    snap_dists, indices = tree.query(coords)
    return indices, snap_dists


def _snap_weighted_points(
    network_structure: rustalgos.graph.NetworkStructure,
    points_gdf: gpd.GeoDataFrame,
    weight_col: str,
    max_snap_dist: float,
    label: str,
) -> list[tuple[int, float]]:
    """Snap weighted point features to network nodes, returning ``(node_idx, weight)`` pairs.

    Rows beyond ``max_snap_dist`` from any node, or with NaN / non-positive weight, are dropped
    (with a logged count). Weights are not aggregated here; the Rust layer sums weights sharing a
    node so a single Dijkstra is run per node.
    """
    if len(points_gdf) == 0:
        return []
    coords = np.array([(g.x, g.y) for g in points_gdf.geometry])
    indices, snap_dists = _snap_coords_to_nodes(network_structure, coords)
    pairs: list[tuple[int, float]] = []
    n_far = 0
    n_bad_weight = 0
    weights = points_gdf[weight_col].to_numpy()
    for i in range(len(points_gdf)):
        if snap_dists[i] > max_snap_dist:
            n_far += 1
            continue
        w = float(weights[i])
        if np.isnan(w) or w <= 0:
            n_bad_weight += 1
            continue
        pairs.append((int(indices[i]), w))
    if n_far:
        logger.warning(f"{n_far} {label} exceeded max_snap_dist={max_snap_dist}m and were excluded.")
    if n_bad_weight:
        logger.info(f"Dropped {n_bad_weight} {label} with NaN or non-positive weight.")
    return pairs


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

    # Snap centroids to nearest network nodes
    indices, distances_snap = _snap_coords_to_nodes(network_structure, centroid_coords)

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
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.
    od_matrix
        An [`OdMatrix`](/rustalgos/centrality#odmatrix) mapping (origin, destination) node pairs to trip weights.
        Build with [`build_od_matrix`](/metrics/networks#build_od_matrix).
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
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    resolved_distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    results = {d: result for d in resolved_distances}
    return _extract_results(results, nodes_gdf, {})


def betweenness_demand(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    origins_gdf: gpd.GeoDataFrame,
    destinations_gdf: gpd.GeoDataFrame,
    origin_weight_col: str,
    destination_weight_col: str,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    decay_fn: str = "exp(-4 * p)",
    closest_destination: bool = False,
    metric_name: str = "demand",
    max_snap_dist: float = 100.0,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute demand-weighted (flow) betweenness from a spatial interaction model.

    Trips are allocated between weighted origins (e.g. population) and weighted destinations (e.g.
    attractors) using a **singly (origin-)constrained** spatial interaction model, then routed along
    shortest network paths so that intermediate nodes accumulate the flow that passes through them.
    For each origin :math:`o` and reachable destination :math:`d` the allocated flow is

    .. math::
        W_{od} = W_o \cdot \frac{W_d \cdot f(c_{od})}{\sum_{d'} W_{d'} \cdot f(c_{od'})}

    where :math:`f` is ``decay_fn`` and :math:`c_{od}` is the network distance. Each origin's full
    weight is conserved and distributed across reachable destinations (destination totals are not
    constrained — that would require a doubly-constrained / Furness model). The gravity model is the
    classic instance of this form, recovered with an exponential ``decay_fn``.

    This is the modelled-matrix counterpart to [`betweenness_od`](#betweenness_od): rather than
    supplying an explicit OD matrix, the per-pair weights are derived from the network distances
    revealed during routing, computed in a single traversal per origin.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).
    nodes_gdf
        A nodes `GeoDataFrame`; flow betweenness columns are written to it and it is returned.
    origins_gdf
        A `GeoDataFrame` of demand origins (points or centroids).
    destinations_gdf
        A `GeoDataFrame` of demand destinations / attractors (points or centroids).
    origin_weight_col
        Column in `origins_gdf` giving each origin's weight (e.g. population).
    destination_weight_col
        Column in `destinations_gdf` giving each destination's attractiveness weight.
    distances: list[int]
        Distance thresholds in metres at which to compute flow betweenness.
    minutes: list[float]
        Walking times in minutes; converted to distance thresholds using `speed_m_s`.
    decay_fn: str
        Distance-decay expression for the allocation, using `c` (metric cost) and `p` (normalised
        progress = `c / threshold`). Defaults to `"exp(-4 * p)"` (scale-free, re-normalised per
        threshold). For a classic gravity model on absolute distance use e.g. `"exp(-0.002 * c)"`.
    closest_destination: bool
        If `True`, each origin routes its full weight to its single nearest reachable destination
        instead of allocating across all of them.
    metric_name: str
        Name used for the output column (`cc_{metric_name}_{distance}`). Defaults to `"demand"`.
    max_snap_dist: float
        Maximum distance for snapping origin/destination points to network nodes. Points farther
        than this are dropped (with a logged count).
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for shortest-path equality, as a percentage.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` with a flow-betweenness column added per distance threshold.
    """
    logger.info("Computing demand-weighted (flow) betweenness centrality.")
    origins = _snap_weighted_points(network_structure, origins_gdf, origin_weight_col, max_snap_dist, "origins")
    destinations = _snap_weighted_points(
        network_structure, destinations_gdf, destination_weight_col, max_snap_dist, "destinations"
    )
    logger.info(f"Snapped {len(origins)} origins and {len(destinations)} destinations.")
    partial_func = partial(
        network_structure.betweenness_demand_shortest,
        origins=origins,
        destinations=destinations,
        decay_fn=decay_fn,
        distances=distances,
        minutes=minutes,
        closest_destination=closest_destination,
        metric_name=metric_name,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
    )
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    resolved_distances = config.log_thresholds(distances=distances, minutes=minutes, speed_m_s=speed_m_s)
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

    Tip: compute only what you need — pass a smaller ``closeness`` / ``betweenness`` dict, or ``{}`` to skip a
    whole category — to evaluate fewer expressions and emit fewer columns.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).
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
        Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.05).

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
    sampled_distances: list[tuple[int, np.ndarray]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        full_distances, sampled_distances = _plan_adaptive_sampling(
            network_structure, resolved_distances, eps, bool(betweenness_items), random_seed=random_seed
        )

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
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality simplest full: {dist_label}",
            )
            for d in full_distances:
                results[d] = result

        for d, q in sampled_distances:
            mean_q = float(np.mean(q))
            logger.info(f"  Sampled {d}m: mean q={mean_q:.0%}")
            partial_func = partial(
                network_structure.centrality_simplest,
                distances=[d],
                closeness_exprs=closeness_items,
                betweenness_exprs=betweenness_items,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                sample_probability=1.0,
                sampling_weights=[float(v) for v in q],
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=node_count,
                rust_struct=network_structure,
                partial_func=partial_func,
                desc=f"centrality simplest q~{mean_q:.0%}: {d}m",
            )
            results[d] = result

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


# ---------------------------------------------------------------------------
# Deprecated 4.24 compatibility shims
#
# These reproduce the pre-v5 functional API (names, parameters, output column
# names and values) by translating old-style calls into the expression engine.
# They add no algorithms of their own and are scheduled for removal a few major
# releases on. See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).
# ---------------------------------------------------------------------------


def _legacy_decay_expr(min_threshold_wt: float) -> str:
    """Reproduce legacy beta-weighting as a normalised-progress expression.

    The old weighting was `exp(-beta * c)` with `beta` derived per distance as
    `-ln(min_threshold_wt) / d`. Since `p = c / d`, this is exactly `exp(-k * p)`
    with `k = -ln(min_threshold_wt)` (`k = 4` for the default `min_threshold_wt`).
    """
    k = -float(np.log(min_threshold_wt))
    return f"exp(-{k} * p)"


def node_centrality_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Deprecated 4.24 alias for [`centrality_shortest`](#centrality_shortest).

    .. deprecated:: 5.0
        Use `centrality_shortest` with `closeness` / `betweenness` expression dicts. This shim preserves
        the 4.24 output (columns `cc_density`, `cc_farness`, `cc_harmonic`, `cc_beta`, `cc_cycles`,
        `cc_hillier`, `cc_betweenness`, `cc_betweenness_beta`) and will be removed in a future major release.
        See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).
    """
    warnings.warn(
        "node_centrality_shortest is deprecated since 5.0; use centrality_shortest "
        "with closeness/betweenness expression dicts. This shim will be removed in a "
        "future major release.",
        DeprecationWarning,
        stacklevel=2,
    )
    if betas is not None and distances is None:
        distances = rustalgos.distances_from_betas(betas, min_threshold_wt)
    decay_expr = _legacy_decay_expr(min_threshold_wt)
    closeness = {"density": "1", "farness": "c", "harmonic": "1/c", "beta": decay_expr} if compute_closeness else {}
    betweenness = {"betweenness": "1", "betweenness_beta": decay_expr} if compute_betweenness else {}
    postprocess = {"hillier": "density**2 / farness"} if compute_closeness else {}
    return centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        closeness=closeness,
        betweenness=betweenness,
        cycles=compute_closeness,
        postprocess=postprocess,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def node_centrality_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    tolerance: float | None = None,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Deprecated 4.24 alias for [`centrality_simplest`](#centrality_simplest).

    .. deprecated:: 5.0
        Use `centrality_simplest` with `closeness` / `betweenness` expression dicts. This shim preserves the
        4.24 output (angular columns `cc_density_ang`, `cc_farness_ang`, `cc_harmonic_ang`, `cc_hillier_ang`,
        `cc_betweenness_ang`) and will be removed in a future major release. See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).
    """
    warnings.warn(
        "node_centrality_simplest is deprecated since 5.0; use centrality_simplest with "
        "closeness/betweenness expression dicts. This shim will be removed in a future major release.",
        DeprecationWarning,
        stacklevel=2,
    )
    if betas is not None and distances is None:
        distances = rustalgos.distances_from_betas(betas, min_threshold_wt)
    # old angular scaling: farness = offset + c / unit (defaults 1 and 90 match the modern defaults)
    farness_expr = f"{farness_scaling_offset} + c / {angular_scaling_unit}"
    closeness = (
        {"density": "1", "farness": farness_expr, "harmonic": f"1 / ({farness_expr})"} if compute_closeness else {}
    )
    betweenness = {"betweenness": "1"} if compute_betweenness else {}
    postprocess = {"hillier": "density**2 / farness"} if compute_closeness else {}
    return centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        closeness=closeness,
        betweenness=betweenness,
        postprocess=postprocess,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def segment_centrality(*_args, **_kwargs) -> gpd.GeoDataFrame:
    """Removed in 5.0; raises with guidance.

    .. deprecated:: 5.0
        The continuous-segment engine (`segment_density` / `harmonic` / `beta` / `betweenness`) was removed
        at the low level, so the old numbers cannot be reproduced. The nearest equivalent is
        `centrality_shortest(..., segment_weighted=True)` — a different calculation. See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).
    """
    raise NotImplementedError(
        "segment_centrality was removed in v5: its continuous-segment engine is gone, so the old "
        "segment_density/harmonic/beta/betweenness cannot be reproduced. Nearest equivalent: "
        "centrality_shortest(..., segment_weighted=True) — a different calculation. See COMPATIBILITY.md."
    )
