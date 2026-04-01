r"""
Compute network centralities.

Two centrality methods are available, using shortest-path (metric) or simplest-path (angular) heuristics:

- [`centrality_shortest`](#centrality-shortest)
- [`centrality_simplest`](#centrality-simplest)

These methods wrap the underlying `rust` optimised functions for computing centralities. Multiple classes of measures
and distances are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar
strategies.

When `segment_weighted=True`, node weights are set to the primal edge (street segment) lengths so that centrality
measures reflect total reachable street length rather than node counts. This requires a dual graph representation.

When `sample=True`, adaptive sampling uses the Hoeffding bound to select a distance-dependent sampling probability.
The `epsilon` parameter controls the error tolerance (lower = more samples, higher accuracy).
The default for when sampling is enabled is 0.06.

| Distance | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.08 | ε=0.1 |
|----------|--------|--------|--------|--------|-------|
| 1 km     | 100%   | 100%   | 100%   | 100%   | 100%  |
| 2 km     | 100%   | 100%   | 100%   | 100%   | 100%  |
| 5 km     | 100%   | 100%   | 58.7%  | 33.0%  | 21.1% |
| 10 km    | 100%   | 37.3%  | 16.6%  | 9.3%   | 6.0%  |
| 20 km    | 41.5%  | 10.4%  | 4.6%   | 2.6%   | 1.7%  |

Sampling is exact (100%) at short distances and becomes progressively sparser at longer distances where
reachability is high enough to maintain relative accuracy. The theoretical speedup is approximately 1/p.
When comparing centrality values across different locations, use the same epsilon to ensure consistent
error tolerances and comparable sampling rates.

:::note
The reasons for picking one approach over another are varied:

- Centralities compute the measures relative to each reachable node within the threshold distances. For
this reason, they can be susceptible to distortions caused by messy graph topologies such redundant and varied
concentrations of degree=2 nodes (e.g. to describe roadway geometry) or needlessly complex representations of
street intersections. In these cases, the network should first be cleaned using methods such as those available in
the [`graph`](/tools/graphs) module (see the [network preparation guide](/guide#network-preparation) for examples).
- `harmonic` centrality can be problematic on graphs where nodes are erroneously placed too close
together or where impedances otherwise approach zero, as may be the case for simplest-path measures or small
distance thesholds. This happens because the outcome of the division step can balloon towards $\infty$ once
impedances decrease below 1.
- Simplest (angular) measures require a dual graph representation. Convert primal graphs with
  [`graphs.nx_to_dual`](/tools/graphs#nx-to-dual) before ingesting them.
- Measures should only be directly compared on the same topology because different topologies can otherwise affect
the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed
on primal graphs because this does not account for the impact of differing topologies. Dual graph representations
can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a
four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual.
This effect is amplified for denser regions of the network.
- The usual formulations of closeness or normalised closeness are discouraged because these do not behave
suitably for localised graphs. Harmonic closeness or Hillier normalisation (which resembles a simplified form of
Improved Closeness Centrality proposed by Wasserman and Faust) should be used instead.
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


def centrality_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    decay_fn: str | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    segment_weighted: bool = False,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute centrality using shortest paths with a single Dijkstra per source.

    .. versionchanged:: 4.25.0
        Renamed from ``node_centrality_shortest``. Added ``segment_weighted`` parameter.
        The old ``segment_centrality`` function has been removed; use ``segment_weighted=True`` instead.

    When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style Dijkstra traversal
    per source produces the data for both closeness accumulation and betweenness backpropagation, halving computation
    time compared to computing them separately.

    The decay closeness and betweenness decay metrics are computed using a decay function expressed as a string with
    the variable `p`, which represents normalised progress from the source (`p = 0`) to the distance threshold
    (`p = 1`), where `p = cost / max_cost`. By default, `decay_fn` is `"exp(-4 * p)"` (exponential decay reaching
    ~1.8% at the threshold). Helper functions for constructing decay expressions are available in the
    `cityseer.decay` module.

    When ``sample=True``, sampling probability is derived from each distance threshold using a canonical grid network
    model (see ``sampling.compute_distance_p``). This produces deterministic, reach-agnostic sample fractions that are
    comparable across networks.

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
    compute_closeness: bool
        Compute closeness centralities. True by default.
    compute_betweenness: bool
        Compute betweenness centralities. True by default.
    decay_fn: str
        An expression string for the decay function, using the variable `p` (normalised progress from 0 to 1, where
        `p = cost / max_cost`). At the source `p = 0` and at the distance threshold `p = 1`. Default is
        `"exp(-4 * p)"` (exponential decay reaching ~1.8% at the threshold). Use `"1"` for flat (unweighted) decay
        metrics, or provide a custom expression. Helper functions are available in the `cityseer.decay` module.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for betweenness path equality, as a percentage (e.g. 1.0 = 1%).
        Paths within this percentage of the shortest are treated as near-equal for multi-predecessor
        Brandes betweenness. A tiny internal epsilon is always enforced as a minimum for
        floating-point stability.
    segment_weighted: bool
        If True, set node weights to primal edge (street segment) lengths so that centrality
        measures are proportional to street length rather than node count. Requires a dual graph.
        Default is False.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, uses distance-based Bernoulli sampling with inverse-probability weighting (IPW). The
        sampling probability is derived from each distance threshold using a canonical grid model (see
        ``sampling.compute_distance_p``). At distances where the sampling probability exceeds the live
        fraction, exact computation is used instead.
    epsilon: float
        Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.

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
    resolved_distances, _seconds = rustalgos.pair_distances_and_time(speed_m_s, distances, minutes)
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    eps = epsilon if epsilon is not None else sampling.HOEFFDING_EPSILON
    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        # Sampling runs dead buffer nodes as sources, so the break-even point
        # is p < n_live / n_total. Above this threshold exact mode is faster.
        lives = network_structure.node_lives
        live_fraction = sum(lives) / len(lives) if lives else 1.0
        for d in sorted(resolved_distances):
            p = sampling.compute_distance_p(d, epsilon=eps)
            if p >= live_fraction:
                full_distances.append(d)
            else:
                sampled_distances.append((d, p))

    results: dict[int, rustalgos.centrality.CentralityShortestResult] = {}

    with _SegmentWeightContext(network_structure, nodes_gdf, segment_weighted):
        if full_distances:
            dist_label = ", ".join(f"{d}m" for d in full_distances)
            logger.info(f"  Full: {dist_label}")
            partial_func = partial(
                network_structure.centrality_shortest,
                distances=full_distances,
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                decay_fn=decay_fn,
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
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                decay_fn=decay_fn,
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

    if not results:
        return nodes_gdf

    ref_result = next(iter(results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)

    if compute_closeness:
        for measure_key, attr_key in [
            ("decay", "node_decay"),
            ("cycles", "node_cycles"),
            ("density", "node_density"),
            ("farness", "node_farness"),
            ("harmonic", "node_harmonic"),
        ]:
            for d, res in results.items():
                data_key = config.prep_gdf_key(measure_key, d)
                temp_data[data_key] = getattr(res, attr_key)[d]
        for d, res in results.items():
            data_key = config.prep_gdf_key("hillier", d)
            with np.errstate(divide="ignore", invalid="ignore"):
                temp_data[data_key] = res.node_density[d] ** 2 / res.node_farness[d]

    if compute_betweenness:
        for measure_key, attr_key in [
            ("betweenness", "node_betweenness"),
            ("betweenness_decay", "node_betweenness_decay"),
        ]:
            for d, res in results.items():
                data_key = config.prep_gdf_key(measure_key, d)
                temp_data[data_key] = getattr(res, attr_key)[d]

    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


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
    decay_fn: str | None = None,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute OD-weighted betweenness centrality using the shortest path heuristic.

    Weights betweenness by origin-destination trip counts from a sparse OD matrix. Only source nodes with outbound
    trips are traversed, and each shortest-path contribution is scaled by the corresponding OD weight. Closeness
    metrics are not computed.

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
    decay_fn: str
        An expression string for the decay function, using the variable `p` (normalised progress from 0 to 1, where
        `p = cost / max_cost`). At the source `p = 0` and at the distance threshold `p = 1`. Default is
        `"exp(-4 * p)"` (exponential decay reaching ~1.8% at the threshold). Use `"1"` for flat (unweighted) decay
        metrics, or provide a custom expression. Helper functions are available in the `cityseer.decay` module.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional betweenness columns.

    """
    logger.info("Computing OD-weighted betweenness centrality.")
    partial_func = partial(
        network_structure.betweenness_od_shortest,
        od_matrix=od_matrix,
        distances=distances,
        minutes=minutes,
        decay_fn=decay_fn,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
    )
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)
    temp_data = {}
    for measure_key, attr_key in [
        ("betweenness", "node_betweenness"),
        ("betweenness_decay", "node_betweenness_decay"),
    ]:
        for distance in distances:
            data_key = config.prep_gdf_key(measure_key, distance)
            temp_data[data_key] = getattr(result, attr_key)[distance]
    temp_df = pd.DataFrame(temp_data, index=result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def centrality_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    speed_m_s: float = SPEED_M_S,
    tolerance: float | None = None,
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    segment_weighted: bool = False,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute centrality using simplest (angular) paths with a single Dijkstra per source.

    .. versionchanged:: 4.25.0
        Renamed from ``node_centrality_simplest``. Added ``segment_weighted`` parameter.

    When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style
    Dijkstra traversal per source produces the data for both closeness accumulation and
    betweenness backpropagation.

    This function does not accept a `decay_fn` parameter; angular (simplest-path) centralities use
    angular cost rather than distance-based decay weighting.

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
    compute_closeness: bool
        Compute closeness centralities. True by default.
    compute_betweenness: bool
        Compute betweenness centralities. True by default.
    speed_m_s: float
        Speed in metres per second for converting `minutes` to distance thresholds.
    tolerance: float
        Relative tolerance for angular betweenness path equality, as a percentage (e.g. 1.0 = 1%).
    angular_scaling_unit: float
        Scaling unit for angular cost normalisation.
    farness_scaling_offset: float
        Offset for farness calculation.
    segment_weighted: bool
        If True, set node weights to primal edge (street segment) lengths so that centrality
        measures are proportional to street length rather than node count. Requires a dual graph.
        Default is False.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, uses distance-based Bernoulli sampling with inverse-probability weighting (IPW).
    epsilon: float
        Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.

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
    resolved_distances, _seconds = rustalgos.pair_distances_and_time(speed_m_s, distances, minutes)
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

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

    results: dict[int, rustalgos.centrality.CentralitySimplestResult] = {}

    with _SegmentWeightContext(network_structure, nodes_gdf, segment_weighted):
        if full_distances:
            dist_label = ", ".join(f"{d}m" for d in full_distances)
            logger.info(f"  Full: {dist_label}")
            partial_func = partial(
                network_structure.centrality_simplest,
                distances=full_distances,
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                angular_scaling_unit=angular_scaling_unit,
                farness_scaling_offset=farness_scaling_offset,
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
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                angular_scaling_unit=angular_scaling_unit,
                farness_scaling_offset=farness_scaling_offset,
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

    if not results:
        return nodes_gdf

    ref_result = next(iter(results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)

    if compute_closeness:
        for d, res in results.items():
            temp_data[config.prep_gdf_key("density", d, angular=True)] = res.node_density[d]
            temp_data[config.prep_gdf_key("harmonic", d, angular=True)] = res.node_harmonic[d]
            temp_data[config.prep_gdf_key("farness", d, angular=True)] = res.node_farness[d]
            with np.errstate(divide="ignore", invalid="ignore"):
                temp_data[config.prep_gdf_key("hillier", d, angular=True)] = (
                    res.node_density[d] ** 2 / res.node_farness[d]
                )

    if compute_betweenness:
        for d, res in results.items():
            data_key = config.prep_gdf_key("betweenness", d, angular=True)
            temp_data[data_key] = res.node_betweenness[d]

    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


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

    Wraps `centrality_shortest` with `compute_closeness=True` and `compute_betweenness=False`.
    Uses exponential decay (`"exp(-4 * p)"`) by default; pass `decay_fn` to
    `centrality_shortest` for a custom decay function.
    """
    return centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        compute_closeness=True,
        compute_betweenness=False,
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
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using simplest (angular) paths.

    Wraps `centrality_simplest` with `compute_closeness=True` and `compute_betweenness=False`.
    """
    return centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        compute_closeness=True,
        compute_betweenness=False,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        angular_scaling_unit=angular_scaling_unit,
        farness_scaling_offset=farness_scaling_offset,
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

    Wraps `centrality_shortest` with `compute_closeness=False` and `compute_betweenness=True`.
    Uses exponential decay (`"exp(-4 * p)"`) by default; pass `decay_fn` to
    `centrality_shortest` for a custom decay function.
    """
    return centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        compute_closeness=False,
        compute_betweenness=True,
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

    Wraps `centrality_simplest` with `compute_closeness=False` and `compute_betweenness=True`.
    """
    return centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        minutes=minutes,
        compute_closeness=False,
        compute_betweenness=True,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )
