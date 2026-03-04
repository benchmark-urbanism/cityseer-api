r"""
Compute network centralities.

There are three network centrality methods available depending on whether you're using a node-based or segment-based
approach, with the former available in both shortest and simplest (angular) variants.

- [`node_centrality_shortest`](#node-centrality-shortest)
- [`node_centrality_simplest`](#node-centrality-simplest)
- [`segment_centrality`](#segment-centrality)

These methods wrap the underlying `rust` optimised functions for computing centralities. Multiple classes of measures
and distances are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar
strategies.

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

- Node based centralities compute the measures relative to each reachable node within the threshold distances. For
this reason, they can be susceptible to distortions caused by messy graph topologies such redundant and varied
concentrations of degree=2 nodes (e.g. to describe roadway geometry) or needlessly complex representations of
street intersections. In these cases, the network should first be cleaned using methods such as those available in
the [`graph`](/tools/graphs) module (see the [graph cleaning guide](/guide#graph-cleaning) for examples). If a
network topology has varied intensities of nodes but the street segments are less spurious, then segmentised methods
can be preferable because they are based on segment distances: segment aggregations remain the same regardless of
the number of intervening nodes, however, are not immune from situations such as needlessly complex representations
of roadway intersections or a proliferation of walking paths in greenspaces;
- Node-based `harmonic` centrality can be problematic on graphs where nodes are erroneously placed too close
together or where impedances otherwise approach zero, as may be the case for simplest-path measures or small
distance thesholds. This happens because the outcome of the division step can balloon towards $\infty$ once
impedances decrease below 1.
- Note that `cityseer`'s implementation of simplest (angular) measures work on both primal and dual graphs (node only).
- Measures should only be directly compared on the same topology because different topologies can otherwise affect
the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed
on primal graphs because this does not account for the impact of differing topologies. Dual graph representations
can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a
four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual.
This effect is amplified for denser regions of the network.
- Segmentised versions of centrality measures should not be computed on dual graph topologies because street segment
lengths would be duplicated for each permutation of dual edge spanning street intersections. By way of example,
the contribution of a single edge segment at a four-way intersection would be duplicated three times.
- The usual formulations of closeness or normalised closeness are discouraged because these do not behave
suitably for localised graphs. Harmonic closeness or Hillier normalisation (which resembles a simplified form of
Improved Closeness Centrality proposed by Wasserman and Faust) should be used instead.
- Network decomposition can be a useful strategy when working at small distance thresholds, and confers advantages
such as more regularly spaced snapshots and fewer artefacts at small distance thresholds where street edges
intersect distance thresholds. However, the regular spacing of the decomposed segments will introduce spikes in the
distributions of node-based centrality measures when working at very small distance thresholds. Segmentised versions
may therefore be preferable when working at small thresholds on decomposed networks.
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
MIN_THRESH_WT = config.MIN_THRESH_WT
SPEED_M_S = config.SPEED_M_S


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
    tolerance: float = 0.0,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute node centrality using shortest paths with a single Dijkstra per source.

    When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style Dijkstra traversal
    per source produces the data for both closeness accumulation and betweenness backpropagation, halving computation
    time compared to computing them separately.

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
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
    betas: list[float]
        A list of $\beta$ to be used for the exponential decay function for weighted metrics.
    minutes: list[float]
        A list of walking times in minutes to be used for calculations.
    compute_closeness: bool
        Compute closeness centralities. True by default.
    compute_betweenness: bool
        Compute betweenness centralities. True by default.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, uses distance-based sampling. If False, computes exact centrality.
    epsilon: float
        Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional centrality columns.
    """
    logger.info("Computing node centrality (shortest).")
    resolved_distances, _betas, _seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    eps = epsilon if epsilon is not None else sampling.HOEFFDING_EPSILON
    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        for d in sorted(resolved_distances):
            p = sampling.compute_distance_p(d, epsilon=eps)
            if p >= 1.0:
                full_distances.append(d)
            else:
                sampled_distances.append((d, p))

    results: dict[int, rustalgos.centrality.CentralityShortestResult] = {}

    if full_distances:
        dist_label = ", ".join(f"{d}m" for d in full_distances)
        logger.info(f"  Full: {dist_label}")
        partial_func = partial(
            network_structure.centrality_shortest,
            distances=full_distances,
            compute_closeness=compute_closeness,
            compute_betweenness=compute_betweenness,
            min_threshold_wt=min_threshold_wt,
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
            results[d] = result  # type: ignore[assignment]

    for d, p in sampled_distances:
        logger.info(f"  Sampled {d}m: p={p:.0%}")
        partial_func = partial(
            network_structure.centrality_shortest,
            distances=[d],
            compute_closeness=compute_closeness,
            compute_betweenness=compute_betweenness,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            tolerance=tolerance,
            sample_probability=p,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count,
            rust_struct=network_structure,
            partial_func=partial_func,
            desc=f"centrality p={p:.0%}: {d}m",
        )
        results[d] = result  # type: ignore[assignment]

    if not results:
        return nodes_gdf

    ref_result = next(iter(results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)

    if compute_closeness:
        for measure_key, attr_key in [
            ("beta", "node_beta"),
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
            ("betweenness_beta", "node_betweenness_beta"),
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
        centroid_gdf = gpd.GeoDataFrame({"geometry": zones_work["_centroid"]}, crs=zones_work.crs)  # type: ignore[no-matching-overload]
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
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
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
        Build with [`config.build_od_matrix`](/config#build-od-matrix).
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
    betas: list[float]
        A list of $\\beta$ to be used for the exponential decay function for weighted metrics.
    minutes: list[float]
        A list of walking times in minutes to be used for calculations.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.

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
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    distances = config.log_thresholds(
        distances=distances,
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)
    temp_data = {}
    for measure_key, attr_key in [
        ("betweenness", "node_betweenness"),
        ("betweenness_beta", "node_betweenness_beta"),
    ]:
        for distance in distances:
            data_key = config.prep_gdf_key(measure_key, distance)
            temp_data[data_key] = getattr(result, attr_key)[distance]
    temp_df = pd.DataFrame(temp_data, index=result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


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
    tolerance: float = 0.0,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    r"""Compute node centrality using simplest (angular) paths with a single Dijkstra per source.

    When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style Dijkstra traversal
    per source produces the data for both closeness accumulation and betweenness backpropagation.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
    betas: list[float]
        A list of $\beta$ to be used for the exponential decay function for weighted metrics.
    minutes: list[float]
        A list of walking times in minutes to be used for calculations.
    compute_closeness: bool
        Compute closeness centralities. True by default.
    compute_betweenness: bool
        Compute betweenness centralities. True by default.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.
    angular_scaling_unit: float
        Scaling unit for angular cost normalisation.
    farness_scaling_offset: float
        Offset for farness calculation.
    tolerance: float
        Relative tolerance for betweenness path equality.
    random_seed: int
        Optional seed for reproducible sampling.
    sample: bool
        If True, uses distance-based sampling. If False, computes exact centrality.
    epsilon: float
        Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `nodes_gdf` parameter is returned with additional centrality columns.
    """
    logger.info("Computing node centrality (simplest).")
    resolved_distances, _betas, _seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    eps = epsilon if epsilon is not None else sampling.HOEFFDING_EPSILON
    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    if not sample:
        full_distances = sorted(resolved_distances)
    else:
        logger.warning("Sampling is experimental: API and behaviour may change in future releases.")
        for d in sorted(resolved_distances):
            p = sampling.compute_distance_p(d, epsilon=eps)
            if p >= 1.0:
                full_distances.append(d)
            else:
                sampled_distances.append((d, p))

    results: dict[int, rustalgos.centrality.CentralitySimplestResult] = {}

    if full_distances:
        dist_label = ", ".join(f"{d}m" for d in full_distances)
        logger.info(f"  Full: {dist_label}")
        partial_func = partial(
            network_structure.centrality_simplest,
            distances=full_distances,
            compute_closeness=compute_closeness,
            compute_betweenness=compute_betweenness,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            angular_scaling_unit=angular_scaling_unit,
            farness_scaling_offset=farness_scaling_offset,
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
            results[d] = result  # type: ignore[assignment]

    for d, p in sampled_distances:
        logger.info(f"  Sampled {d}m: p={p:.0%}")
        partial_func = partial(
            network_structure.centrality_simplest,
            distances=[d],
            compute_closeness=compute_closeness,
            compute_betweenness=compute_betweenness,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            angular_scaling_unit=angular_scaling_unit,
            farness_scaling_offset=farness_scaling_offset,
            tolerance=tolerance,
            sample_probability=p,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count,
            rust_struct=network_structure,
            partial_func=partial_func,
            desc=f"centrality simplest p={p:.0%}: {d}m",
        )
        results[d] = result  # type: ignore[assignment]

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
        for measure_key, attr_key in [
            ("betweenness", "node_betweenness"),
            ("betweenness_beta", "node_betweenness_beta"),
        ]:
            for d, res in results.items():
                data_key = config.prep_gdf_key(measure_key, d, angular=True)
                temp_data[data_key] = getattr(res, attr_key)[d]

    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def segment_centrality(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
) -> gpd.GeoDataFrame:
    r"""
    Compute segment-based network centrality using the shortest path heuristic.

    > Simplest path heuristics introduce conceptual and practical complications and support is deprecated since v4.

    > For conceptual and practical reasons, segment based centralities are not weighted by node weights.

    Parameters
    ----------
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the method.
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$
        for distance-weighted metrics will be determined implicitly using `min_threshold_wt`. If the `distances`
        parameter is not provided, then the `beta` or `minutes` parameters must be provided instead.
    betas: list[float]
        A list of $\beta$ to be used for the exponential decay function for weighted metrics. The $d_{max}$ thresholds
        for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the
        `distances` or `minutes` parameter must be provided instead.
    minutes: list[float]
        A list of walking times in minutes to be used for calculations. The $d_{max}$ thresholds for unweighted metrics
        and $\beta$ for distance-weighted metrics will be determined implicitly using the `speed_m_s` and
        `min_threshold_wt` parameters. If the `minutes` parameter is not provided, then the `distances` or `betas`
        parameters must be provided instead.
    compute_closeness: bool
        Compute closeness centralities. True by default.
    compute_betweenness: bool
        Compute betweenness centralities. True by default.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas)
        for more information.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.

    Examples
    --------
    Segment path centralities are available with the following keys:

    | key                 | formula | notes |
    | ------------------- | :-----: |------ |
    | seg_density     | $$\sum_{(a, b)}^{edges}d_{b} - d_{a}$$ | A summation of edge lengths. |
    | seg_harmonic    | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\ln(b) -\ln(a)$$ | A continuous form of
    harmonic closeness centrality applied to edge lengths. |
    | seg_beta        | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\frac{\exp(-\beta\cdot b) -\exp(-\beta\cdot a)}{-\beta}$$ | A
    continuous form of beta-weighted (gravity index) centrality applied to edge lengths. |
    | seg_betweenness | | A continuous form of betweenness: Resembles `segment_beta` applied to edges situated
    on shortest paths between all nodes $j$ and $k$ passing through $i$. |

    """
    logger.info("Computing shortest path segment centrality.")
    partial_func = partial(
        network_structure.segment_centrality,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    # wraps progress bar
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=network_structure, partial_func=partial_func
    )
    # unpack
    distances = config.log_thresholds(
        distances=distances,
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    if compute_closeness is True:
        for measure_key, attr_key in [
            ("seg_density", "segment_density"),
            ("seg_harmonic", "segment_harmonic"),
            ("seg_beta", "segment_beta"),
        ]:
            for distance in distances:
                data_key = config.prep_gdf_key(measure_key, distance)
                temp_data[data_key] = getattr(result, attr_key)[distance]
    if compute_betweenness is True:
        for distance in distances:
            data_key = config.prep_gdf_key("seg_betweenness", distance)
            temp_data[data_key] = result.segment_betweenness[distance]  # type: ignore

    temp_df = pd.DataFrame(temp_data, index=result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]

    return nodes_gdf


# =============================================================================
# Convenience wrappers — closeness-only and betweenness-only
# =============================================================================


def closeness_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using shortest paths. Wraps `node_centrality_shortest`."""
    return node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=True,
        compute_betweenness=False,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )


def closeness_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using simplest (angular) paths. Wraps `node_centrality_simplest`."""
    return node_centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=True,
        compute_betweenness=False,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
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
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using shortest paths. Wraps `node_centrality_shortest`."""
    return node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=False,
        compute_betweenness=True,
        min_threshold_wt=min_threshold_wt,
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
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
    random_seed: int | None = None,
    sample: bool = False,
    epsilon: float | None = None,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using simplest (angular) paths. Wraps `node_centrality_simplest`."""
    return node_centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=False,
        compute_betweenness=True,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
        random_seed=random_seed,
        sample=sample,
        epsilon=epsilon,
    )
