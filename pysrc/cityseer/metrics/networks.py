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

See the accompanying paper on `arXiv` for additional information about methods for computing centrality measures.

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
import warnings
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import config, rustalgos

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
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
    sample_probability: float | None = None,
    sampling_weights: list[float] | None = None,
    random_seed: int | None = None,
    n_betweenness_samples: int | None = None,
    epsilon_betweenness: float | None = None,
) -> gpd.GeoDataFrame:
    r"""
    Compute node-based network centrality using the shortest path heuristic.

    :::note
    Node weights are taken into account when computing centralities. These would typically be initialised at 1 unless
    manually specified.
    :::

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
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
        A value like 0.02 (2%) captures pedestrian indifference to near-equal routes. Only affects betweenness;
        closeness always uses exact shortest paths.
    sample_probability: float
        Probability of sampling a node as a source for centrality calculations. When used alone, provides uniform
        random sampling across all nodes. When combined with `sampling_weights`, the final probability for each
        node is `sample_probability * sampling_weights[node_idx]`.
    sampling_weights: list[float]
        Optional array of per-node sampling weights for manual (non-adaptive) sampling control. Must have length
        equal to the number of nodes, with values in the range [0.0, 1.0]. Use this to bias which nodes are
        selected as source nodes for centrality calculations (e.g., to oversample high-population areas). When
        provided, the sampling probability for each node becomes `sample_probability * sampling_weights[node_idx]`.
        This parameter is not available in the adaptive variants, which automatically calibrate sampling
        probabilities per distance threshold.
    random_seed: int
        Optional seed for deterministic sampling.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.

    Examples
    --------
    The following keys use the shortest-path heuristic:

    | key                   | formula | notes |
    | ----------------------| :------:| ----- |
    | density          | $$\sum_{j\neq{i}}^{n}1$$ | A summation of nodes. |
    | harmonic         | $$\sum_{j\neq{i}}^{n}\frac{1}{d_{(i,j)}}$$ | Harmonic closeness is an appropriate form
    of closeness centrality for localised implementations constrained by the threshold $d_{max}$. |
    | hillier          | $$\frac{(n-1)^2}{\sum_{j \neq i}^{n} d_{(i,j)}}$$ | The square of node density divided by
    farness. This is also a simplified form of Improved Closeness Centrality. |
    | beta             | $$\sum_{j\neq{i}}^{n} \\ \exp(-\beta\cdot d[i,j])$$ | Also known as the gravity index.
    This is a spatial impedance metric differentiated from other closeness centralities by the use of an
    explicit $\beta$ parameter, which can be used to model the decay in walking tolerance as distances
    increase. |
    | cycles           | $$\sum_{j\neq{i}j=cycle}^{n}1$$ | A summation of network cycles. |
    | farness          | $$\sum_{j\neq{i}}^{n}d_{(i,j)}$$ | A summation of distances in metres. |
    | betweenness      | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | Betweenness centrality summing all
    shortest-paths traversing each node $i$. |
    | betweenness_beta | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n} \\ \exp(-\beta\cdot d[j,k])$$ | Applies a
    spatial impedance decay function to betweenness centrality. $d$ represents the full distance from
    any $j$ to $k$ node pair passing through node $i$. |

    """
    warnings.warn(
        "node_centrality_shortest is deprecated. Use closeness_shortest and/or betweenness_shortest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info("Computing shortest path node centrality.")
    # Resolve distances for logging
    resolved_distances = config.log_thresholds(
        distances=distances,
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    temp_data: dict[str, object] = {}
    node_keys_py = None
    gdf_idx = None

    # Closeness: use existing source-sampling Dijkstra
    if compute_closeness is True:
        partial_func = partial(
            network_structure.closeness_shortest,
            distances=distances,
            betas=betas,
            minutes=minutes,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            sample_probability=sample_probability,
            sampling_weights=sampling_weights,
            random_seed=random_seed,
        )
        closeness_result = config.wrap_progress(
            total=network_structure.street_node_count(),
            rust_struct=network_structure,
            partial_func=partial_func,
            desc="closeness",
        )
        config.log_sampling(
            sample_probability=sample_probability,
            distances=resolved_distances,
            reachability_totals=closeness_result.reachability_totals,  # type: ignore
            sampled_source_count=closeness_result.sampled_source_count,  # type: ignore
        )
        node_keys_py = closeness_result.node_keys_py
        gdf_idx = nodes_gdf.index.intersection(node_keys_py)
        for measure_key, attr_key in [
            ("beta", "node_beta"),
            ("cycles", "node_cycles"),
            ("density", "node_density"),
            ("farness", "node_farness"),
            ("harmonic", "node_harmonic"),
        ]:
            for distance in resolved_distances:
                data_key = config.prep_gdf_key(measure_key, distance)
                temp_data[data_key] = getattr(closeness_result, attr_key)[distance]
        for distance in resolved_distances:
            data_key = config.prep_gdf_key("hillier", distance)
            temp_data[data_key] = (
                closeness_result.node_density[distance] ** 2
                / closeness_result.node_farness[distance]  # type: ignore
            )

    # Betweenness: use R-K path sampling via separate Brandes Dijkstra (per distance)
    if compute_betweenness is True:
        for d in resolved_distances:
            partial_func_b = partial(
                network_structure.betweenness_shortest,
                distance=d,
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                n_samples=n_betweenness_samples,
                random_seed=random_seed,
            )
            betweenness_result = config.wrap_progress(
                total=n_betweenness_samples or network_structure.street_node_count(),
                rust_struct=network_structure,
                partial_func=partial_func_b,
                desc=f"betweenness: {d}m",
            )
            if node_keys_py is None:
                node_keys_py = betweenness_result.node_keys_py
                gdf_idx = nodes_gdf.index.intersection(node_keys_py)
            for measure_key, attr_key in [
                ("betweenness", "node_betweenness"),
                ("betweenness_beta", "node_betweenness_beta"),
            ]:
                data_key = config.prep_gdf_key(measure_key, d)
                temp_data[data_key] = getattr(betweenness_result, attr_key)[d]

    if temp_data and node_keys_py is not None and gdf_idx is not None:
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
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    tolerance: float = 0.0,
    sample_probability: float | None = None,
    sampling_weights: list[float] | None = None,
    random_seed: int | None = None,
    n_betweenness_samples: int | None = None,
    epsilon_betweenness: float | None = None,
) -> gpd.GeoDataFrame:
    r"""
    Compute node-based network centrality using the simplest path (angular) heuristic.

    :::note
    Node weights are taken into account when computing centralities. These would typically be initialised at 1 unless
    manually specified.
    :::

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
    angular_scaling_unit: float
        The number by which to divide angular distances for scaling. 90 by default. For example, if the cumulative
        angular distance for a given route is 180 then this will be scaled per 180 / 90 = 2.
    farness_scaling_offset: float
        A number by which to offset the scaled angular distance for computing farness. 1 by default. For example, if the
        scaled angular distance is 2, then an offset of 1 will be applied as 1 + 2 = 3. This offset is only applied when
        calculating farness. Harmonic closeness always uses an offset of 1 to prevent division by zero.
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
        A value like 0.02 (2%) captures pedestrian indifference to near-equal routes.
    sample_probability: float
        Probability of sampling a node as a source for centrality calculations. When used alone, provides uniform
        random sampling across all nodes. When combined with `sampling_weights`, the final probability for each
        node is `sample_probability * sampling_weights[node_idx]`. For automatic per-distance sampling calibration,
        Use [`closeness_simplest`](#closeness-simplest) for automatic per-distance adaptive sampling.
    sampling_weights: list[float]
        Optional array of per-node sampling weights for manual (non-adaptive) sampling control. Must have length
        equal to the number of nodes, with values in the range [0.0, 1.0]. Use this to bias which nodes are
        selected as source nodes for centrality calculations (e.g., to oversample high-population areas). When
        provided, the sampling probability for each node becomes `sample_probability * sampling_weights[node_idx]`.
        This parameter is not available in the adaptive variants, which automatically calibrate sampling
        probabilities per distance threshold.
    random_seed: int
        Optional seed for deterministic sampling.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.

    Examples
    --------
    The following keys use the simplest-path heuristic:

    | key                   | formula | notes |
    | ----------------------| :------:| ----- |
    | density_ang | $$\sum_{j\neq{i}}^{n}1$$ | A summation of nodes. |
    | harmonic_ang    | $$\sum_{j\neq{i}}^{n}\frac{1}{d_{(i,j)}}$$ | Harmonic closeness is an appropriate form
    of closeness centrality for localised implementations constrained by the threshold $d_{max}$. |
    | hillier_ang | $$\frac{(n-1)^2}{\sum_{j \neq i}^{n} d_{(i,j)}}$$ | The square of node density divided by
    farness. This is also a simplified form of Improved Closeness Centrality. |
    | farness_ang | $$\sum_{j\neq{i}}^{n}d_{(i,j)}$$ | A summation of distances in metres. |
    | betweenness_ang | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | Betweenness centrality summing
    all shortest-paths traversing each node $i$. |

    The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular`
    parameter is explicitly set to `True`:

    """
    warnings.warn(
        "node_centrality_simplest is deprecated. Use closeness_simplest and/or betweenness_simplest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info("Computing simplest path node centrality.")
    resolved_distances = config.log_thresholds(
        distances=distances,
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
    )
    temp_data: dict[str, object] = {}
    node_keys_py = None
    gdf_idx = None

    # Closeness: use existing source-sampling Dijkstra
    if compute_closeness is True:
        partial_func = partial(
            network_structure.closeness_simplest,
            distances=distances,
            betas=betas,
            minutes=minutes,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            angular_scaling_unit=angular_scaling_unit,
            farness_scaling_offset=farness_scaling_offset,
            sample_probability=sample_probability,
            sampling_weights=sampling_weights,
            random_seed=random_seed,
        )
        closeness_result = config.wrap_progress(
            total=network_structure.street_node_count(),
            rust_struct=network_structure,
            partial_func=partial_func,
            desc="closeness",
        )
        config.log_sampling(
            sample_probability=sample_probability,
            distances=resolved_distances,
            reachability_totals=closeness_result.reachability_totals,  # type: ignore
            sampled_source_count=closeness_result.sampled_source_count,  # type: ignore
        )
        node_keys_py = closeness_result.node_keys_py
        gdf_idx = nodes_gdf.index.intersection(node_keys_py)
        for distance in resolved_distances:
            temp_data[config.prep_gdf_key("density", distance, angular=True)] = (
                closeness_result.node_density[distance]  # type: ignore
            )
        for distance in resolved_distances:
            temp_data[config.prep_gdf_key("harmonic", distance, angular=True)] = (
                closeness_result.node_harmonic[distance]  # type: ignore
            )
        for distance in resolved_distances:
            temp_data[config.prep_gdf_key("hillier", distance, angular=True)] = (
                closeness_result.node_density[distance] ** 2
                / closeness_result.node_farness[distance]  # type: ignore
            )
        for distance in resolved_distances:
            temp_data[config.prep_gdf_key("farness", distance, angular=True)] = (
                closeness_result.node_farness[distance]  # type: ignore
            )

    # Betweenness: use R-K path sampling via separate Brandes Dijkstra (per distance)
    if compute_betweenness is True:
        for d in resolved_distances:
            partial_func_b = partial(
                network_structure.betweenness_simplest,
                distance=d,
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                n_samples=n_betweenness_samples,
                random_seed=random_seed,
            )
            betweenness_result = config.wrap_progress(
                total=n_betweenness_samples or network_structure.street_node_count(),
                rust_struct=network_structure,
                partial_func=partial_func_b,
                desc=f"betweenness: {d}m",
            )
            if node_keys_py is None:
                node_keys_py = betweenness_result.node_keys_py
                gdf_idx = nodes_gdf.index.intersection(node_keys_py)
            temp_data[config.prep_gdf_key("betweenness", d, angular=True)] = (
                betweenness_result.node_betweenness[d]  # type: ignore
            )

    if temp_data and node_keys_py is not None and gdf_idx is not None:
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
# Primary API — separate closeness and betweenness functions
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
    probe_density: float = config.DEFAULT_PROBE_DENSITY,
    epsilon: float = config.HOEFFDING_EPSILON,
    delta: float = config.HOEFFDING_DELTA,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using shortest paths with adaptive source sampling.

    Parameters
    ----------
    network_structure
        A NetworkStructure.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances: list[int]
        Distance thresholds (meters).
    betas: list[float]
        Decay parameters (beta).
    minutes: list[float]
        Time thresholds (minutes).
    min_threshold_wt: float
        Minimum weight for beta/distance conversion.
    speed_m_s: float
        Travel speed (m/s).
    random_seed: int
        Optional seed for reproducible sampling.
    probe_density: float
        Probes per km² for reachability estimation.
    epsilon: float
        Hoeffding approximation error bound.
    delta: float
        Hoeffding failure probability.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input GeoDataFrame with closeness columns added.
    """
    logger.info("Computing adaptive closeness centrality (shortest).")
    resolved_distances, _betas, _seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    # Probe reachability and compute per-distance sampling probabilities
    reach_estimates = config.probe_reachability(
        network_structure, resolved_distances, probe_density=probe_density, speed_m_s=speed_m_s
    )
    sample_probs = config.compute_sample_probs(reach_estimates, epsilon=epsilon, delta=delta)
    config.log_adaptive_sampling_plan(resolved_distances, reach_estimates, sample_probs, epsilon=epsilon, delta=delta)

    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    for d in sorted(resolved_distances):
        p = sample_probs.get(d)
        if p is None or p >= 1.0:
            full_distances.append(d)
        else:
            sampled_distances.append((d, p))

    closeness_results: dict[int, rustalgos.centrality.ClosenessShortestResult] = {}

    if full_distances:
        dist_label = ", ".join(f"{d}m" for d in full_distances)
        logger.info(f"  Closeness full: {dist_label}")
        partial_func = partial(
            network_structure.closeness_shortest,
            distances=full_distances,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count, rust_struct=network_structure, partial_func=partial_func,
            desc=f"closeness full: {dist_label}",
        )
        for d in full_distances:
            closeness_results[d] = result  # type: ignore[assignment]

    for d, p in sampled_distances:
        logger.info(f"  Closeness {d}m: p={p:.0%}")
        partial_func = partial(
            network_structure.closeness_shortest,
            distances=[d],
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            sample_probability=p,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count, rust_struct=network_structure, partial_func=partial_func,
            desc=f"closeness p={p:.0%}: {d}m",
        )
        closeness_results[d] = result  # type: ignore[assignment]

    ref_result = next(iter(closeness_results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)

    for measure_key, attr_key in [
        ("beta", "node_beta"),
        ("cycles", "node_cycles"),
        ("density", "node_density"),
        ("farness", "node_farness"),
        ("harmonic", "node_harmonic"),
    ]:
        for d, res in closeness_results.items():
            data_key = config.prep_gdf_key(measure_key, d)
            temp_data[data_key] = getattr(res, attr_key)[d]
    for d, res in closeness_results.items():
        data_key = config.prep_gdf_key("hillier", d)
        temp_data[data_key] = res.node_density[d] ** 2 / res.node_farness[d]

    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


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
    probe_density: float = config.DEFAULT_PROBE_DENSITY,
    epsilon: float = config.HOEFFDING_EPSILON,
    delta: float = config.HOEFFDING_DELTA,
) -> gpd.GeoDataFrame:
    """Compute closeness centrality using simplest paths with adaptive source sampling.

    Parameters
    ----------
    network_structure
        A NetworkStructure.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances: list[int]
        Distance thresholds (meters).
    betas: list[float]
        Decay parameters (beta).
    minutes: list[float]
        Time thresholds (minutes).
    min_threshold_wt: float
        Minimum weight for beta/distance conversion.
    speed_m_s: float
        Travel speed (m/s).
    angular_scaling_unit: float
        Scaling unit for angular cost.
    farness_scaling_offset: float
        Offset for farness calculation.
    random_seed: int
        Optional seed for reproducible sampling.
    probe_density: float
        Probes per km² for reachability estimation.
    epsilon: float
        Hoeffding approximation error bound.
    delta: float
        Hoeffding failure probability.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input GeoDataFrame with closeness columns added.
    """
    logger.info("Computing adaptive closeness centrality (simplest).")
    resolved_distances, _betas, _seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    reach_estimates = config.probe_reachability(
        network_structure, resolved_distances, probe_density=probe_density, speed_m_s=speed_m_s
    )
    sample_probs = config.compute_sample_probs(reach_estimates, epsilon=epsilon, delta=delta)
    config.log_adaptive_sampling_plan(resolved_distances, reach_estimates, sample_probs, epsilon=epsilon, delta=delta)

    full_distances: list[int] = []
    sampled_distances: list[tuple[int, float]] = []
    for d in sorted(resolved_distances):
        p = sample_probs.get(d)
        if p is None or p >= 1.0:
            full_distances.append(d)
        else:
            sampled_distances.append((d, p))

    closeness_results: dict[int, rustalgos.centrality.ClosenessSimplestResult] = {}

    if full_distances:
        dist_label = ", ".join(f"{d}m" for d in full_distances)
        logger.info(f"  Closeness full: {dist_label}")
        partial_func = partial(
            network_structure.closeness_simplest,
            distances=full_distances,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            angular_scaling_unit=angular_scaling_unit,
            farness_scaling_offset=farness_scaling_offset,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count, rust_struct=network_structure, partial_func=partial_func,
            desc=f"closeness full: {dist_label}",
        )
        for d in full_distances:
            closeness_results[d] = result  # type: ignore[assignment]

    for d, p in sampled_distances:
        logger.info(f"  Closeness {d}m: p={p:.0%}")
        partial_func = partial(
            network_structure.closeness_simplest,
            distances=[d],
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            angular_scaling_unit=angular_scaling_unit,
            farness_scaling_offset=farness_scaling_offset,
            sample_probability=p,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count, rust_struct=network_structure, partial_func=partial_func,
            desc=f"closeness p={p:.0%}: {d}m",
        )
        closeness_results[d] = result  # type: ignore[assignment]

    ref_result = next(iter(closeness_results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)

    for d, res in closeness_results.items():
        temp_data[config.prep_gdf_key("density", d, angular=True)] = res.node_density[d]
        temp_data[config.prep_gdf_key("harmonic", d, angular=True)] = res.node_harmonic[d]
        temp_data[config.prep_gdf_key("farness", d, angular=True)] = res.node_farness[d]
        temp_data[config.prep_gdf_key("hillier", d, angular=True)] = (
            res.node_density[d] ** 2 / res.node_farness[d]
        )

    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def betweenness_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
    n_samples: int | None = None,
    random_seed: int | None = None,
    probe_density: float = config.DEFAULT_PROBE_DENSITY,
    epsilon: float = config.HOEFFDING_EPSILON,
    delta: float = config.HOEFFDING_DELTA,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using R-K path sampling (shortest paths) with adaptive budgeting.

    Probes network reachability, computes per-distance R-K sample budgets from epsilon,
    and runs betweenness_shortest per distance with the appropriate n_samples.

    Parameters
    ----------
    network_structure
        A NetworkStructure.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances: list[int]
        Distance thresholds (meters).
    betas: list[float]
        Decay parameters (beta).
    minutes: list[float]
        Time thresholds (minutes).
    min_threshold_wt: float
        Minimum weight for beta/distance conversion.
    speed_m_s: float
        Travel speed (m/s).
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
        A value like 0.02 (2%) captures pedestrian indifference to near-equal routes.
    n_samples: int
        Explicit number of samples (overrides adaptive budgeting if provided).
    random_seed: int
        Optional seed for reproducible sampling.
    probe_density: float
        Probes per km² for reachability estimation.
    epsilon: float
        R-K approximation error bound.
    delta: float
        R-K failure probability.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input GeoDataFrame with betweenness columns added.
    """
    logger.info("Computing adaptive R-K betweenness centrality (shortest).")
    resolved_distances = config.log_thresholds(
        distances=distances, betas=betas, minutes=minutes,
        min_threshold_wt=min_threshold_wt, speed_m_s=speed_m_s,
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    # Probe reachability for budget computation and saturation check
    reach_estimates = config.probe_reachability(
        network_structure, resolved_distances, probe_density=probe_density, speed_m_s=speed_m_s
    )
    if n_samples is not None:
        budgets = {d: n_samples for d in resolved_distances}
    else:
        budgets = config.compute_betweenness_budgets(reach_estimates, epsilon=epsilon, delta=delta)

    betweenness_results: dict[int, rustalgos.centrality.BetweennessShortestResult] = {}

    for d in sorted(resolved_distances):
        n = budgets.get(d)
        if n is None:
            # Tipping point: R-K budget >= reach, pair space too small for sampling to converge
            logger.info(f"  Betweenness {d}m: reach={reach_estimates.get(d, 0):.0f} -> exact Brandes (tipping point)")
            partial_func = partial(
                network_structure.betweenness_exact_shortest,
                distances=[d],
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
            )
            result = config.wrap_progress(
                total=node_count, rust_struct=network_structure, partial_func=partial_func,
                desc=f"betweenness exact: {d}m",
            )
        else:
            # Sampling regime: R-K path sampling with Euclidean pair selection
            logger.info(f"  Betweenness {d}m: reach={reach_estimates.get(d, 0):.0f}, n_samples={n} -> sampling")
            partial_func = partial(
                network_structure.betweenness_shortest,
                distance=d,
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
                n_samples=n,
                random_seed=random_seed,
            )
            result = config.wrap_progress(
                total=n, rust_struct=network_structure, partial_func=partial_func,
                desc=f"betweenness n={n}: {d}m",
            )
        betweenness_results[d] = result  # type: ignore[assignment]

    ref_result = next(iter(betweenness_results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)
    for measure_key, attr_key in [
        ("betweenness", "node_betweenness"),
        ("betweenness_beta", "node_betweenness_beta"),
    ]:
        for d, res in betweenness_results.items():
            data_key = config.prep_gdf_key(measure_key, d)
            temp_data[data_key] = getattr(res, attr_key)[d]
    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def betweenness_simplest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
    n_samples: int | None = None,
    random_seed: int | None = None,
    probe_density: float = config.DEFAULT_PROBE_DENSITY,
    epsilon: float = config.HOEFFDING_EPSILON,
    delta: float = config.HOEFFDING_DELTA,
) -> gpd.GeoDataFrame:
    """Compute betweenness centrality using R-K path sampling (simplest paths) with adaptive budgeting.

    Parameters
    ----------
    network_structure
        A NetworkStructure.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances: list[int]
        Distance thresholds (meters).
    betas: list[float]
        Decay parameters (beta).
    minutes: list[float]
        Time thresholds (minutes).
    min_threshold_wt: float
        Minimum weight for beta/distance conversion.
    speed_m_s: float
        Travel speed (m/s).
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
        A value like 0.02 (2%) captures pedestrian indifference to near-equal routes.
    n_samples: int
        Explicit number of samples (overrides adaptive budgeting if provided).
    random_seed: int
        Optional seed for reproducible sampling.
    probe_density: float
        Probes per km² for reachability estimation.
    epsilon: float
        R-K approximation error bound.
    delta: float
        R-K failure probability.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input GeoDataFrame with betweenness columns added.
    """
    logger.info("Computing adaptive R-K betweenness centrality (simplest).")
    resolved_distances = config.log_thresholds(
        distances=distances, betas=betas, minutes=minutes,
        min_threshold_wt=min_threshold_wt, speed_m_s=speed_m_s,
    )
    node_count = network_structure.street_node_count()
    temp_data: dict[str, object] = {}

    if n_samples is not None:
        budgets = {d: n_samples for d in resolved_distances}
    else:
        reach_estimates = config.probe_reachability(
            network_structure, resolved_distances, probe_density=probe_density, speed_m_s=speed_m_s
        )
        budgets = config.compute_betweenness_budgets(reach_estimates, epsilon=epsilon, delta=delta)

    betweenness_results: dict[int, rustalgos.centrality.BetweennessSimplestResult] = {}

    for d in sorted(resolved_distances):
        n = budgets.get(d) or 100
        logger.info(f"  Betweenness {d}m: n_samples={n}")
        partial_func = partial(
            network_structure.betweenness_simplest,
            distance=d,
            min_threshold_wt=min_threshold_wt,
            speed_m_s=speed_m_s,
            tolerance=tolerance,
            n_samples=n,
            random_seed=random_seed,
        )
        result = config.wrap_progress(
            total=node_count, rust_struct=network_structure, partial_func=partial_func,
            desc=f"betweenness n={n}: {d}m",
        )
        betweenness_results[d] = result  # type: ignore[assignment]

    ref_result = next(iter(betweenness_results.values()))
    node_keys_py = ref_result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)
    for d, res in betweenness_results.items():
        data_key = config.prep_gdf_key("betweenness", d, angular=True)
        temp_data[data_key] = res.node_betweenness[d]
    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf


def betweenness_exact_shortest(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    tolerance: float = 0.0,
) -> gpd.GeoDataFrame:
    """Compute exact Brandes betweenness centrality from all sources (no sampling).

    Parameters
    ----------
    network_structure
        A NetworkStructure.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances: list[int]
        Distance thresholds (meters).
    betas: list[float]
        Decay parameters (beta).
    minutes: list[float]
        Time thresholds (minutes).
    min_threshold_wt: float
        Minimum weight for beta/distance conversion.
    speed_m_s: float
        Travel speed (m/s).
    tolerance: float
        Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are
        treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only.
        A value like 0.02 (2%) captures pedestrian indifference to near-equal routes.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input GeoDataFrame with betweenness columns added.
    """
    logger.info("Computing exact Brandes betweenness centrality (shortest).")
    resolved_distances = config.log_thresholds(
        distances=distances, betas=betas, minutes=minutes,
        min_threshold_wt=min_threshold_wt, speed_m_s=speed_m_s,
    )
    partial_func = partial(
        network_structure.betweenness_exact_shortest,
        distances=distances,
        betas=betas,
        minutes=minutes,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        tolerance=tolerance,
    )
    result = config.wrap_progress(
        total=network_structure.street_node_count(),
        rust_struct=network_structure,
        partial_func=partial_func,
        desc="betweenness (exact)",
    )
    node_keys_py = result.node_keys_py
    gdf_idx = nodes_gdf.index.intersection(node_keys_py)
    temp_data: dict[str, object] = {}
    for measure_key, attr_key in [
        ("betweenness", "node_betweenness"),
        ("betweenness_beta", "node_betweenness_beta"),
    ]:
        for distance in resolved_distances:
            data_key = config.prep_gdf_key(measure_key, distance)
            temp_data[data_key] = getattr(result, attr_key)[distance]
    temp_df = pd.DataFrame(temp_data, index=node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]
    return nodes_gdf
