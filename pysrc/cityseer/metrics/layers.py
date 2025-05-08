from __future__ import annotations

import logging
from functools import partial

import geopandas as gpd
import pandas as pd

from .. import config, rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# separate out so that ast parser can parse function def
MIN_THRESH_WT = config.MIN_THRESH_WT
SPEED_M_S = config.SPEED_M_S


def build_data_map(
    data_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 200,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    n_nearest_candidates: int = 20,
) -> rustalgos.data.DataMap:
    """
    Assign a `GeoDataFrame` to a [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).

    A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the
    network. Points will be assigned to the closest street edge. Polygons will be assigned to the closest
    `n_nearest_candidates` adjacent street edges.
    up to

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    barriers_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing barriers. These barriers will be considered during the assignment of data points to the network.
    n_nearest_candidates: int
        The number of nearest street edge candidates to consider when assigning data points to the network. This is used
        to determine the best assignments based on proximity. Edges are sorted by distance and the closest
        `n_nearest_candidates` are considered.

    Returns
    -------
    data_map: rustalgos.data.DataMap
        A [`rustalgos.data.DataMap`](/rustalgos#datamap) instance.
    """
    # check for unique index
    if data_gdf.index.duplicated().any():
        raise ValueError("The data GeoDataFrame index must contain unique entries.")
    # create data map
    data_map = rustalgos.data.DataMap()
    # prepare the data_map
    logger.info("Assigning data to network.")
    for data_key, data_row in data_gdf.iterrows():  # type: ignore
        data_id = None if data_id_col is None else data_row[data_id_col]  # type: ignore
        data_map.insert(
            data_key,
            data_row[data_gdf.active_geometry_name].wkt,  # type: ignore
            data_id,  # type: ignore
        )
    # barrier geoms
    barriers_wkt: list[str] | None = None
    if barriers_gdf is not None:
        barriers_wkt = []
        for _, row in barriers_gdf.iterrows():  # type: ignore
            barriers_wkt.append(row.geometry.wkt)  # type: ignore
    if barriers_wkt is not None:
        network_structure.set_barriers(barriers_wkt)  # type: ignore
    data_map.assign_data_to_network(network_structure, max_netw_assign_dist, n_nearest_candidates)
    network_structure.unset_barriers()  # type: ignore

    return data_map


def compute_accessibilities(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    accessibility_keys: list[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 200,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    n_nearest_candidates: int = 20,
    spatial_tolerance: int = 0,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    jitter_scale: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute land-use accessibilities for the specified land-use classification keys over the street network.

    The landuses are aggregated and computed over the street network relative to the network nodes, with the implication
    that the measures are generated from the same locations as those used for centrality computations.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    landuse_column_label: str
        The column label from which to take landuse categories, e.g. a column labelled "landuse_categories" might
        contain "shop", "pub", "school", etc.
    accessibility_keys: tuple[str]
        Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use
        schema used for the `landuse_labels` parameter, e.g. "pub". The calculations will be performed in both
        weighted `wt` and non_weighted `nw` variants.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
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
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    barriers_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing barriers. These barriers will be considered during the assignment of data points to the network.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas)
        for more information.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics. Three
        columns will be returned for each input landuse class and distance combination; a simple count of reachable
        locations, a distance weighted count of reachable locations, and the smallest distance to the nearest location.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_nearest_assign`.

    Examples
    --------
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs, io

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
    print(nodes_gdf.head())
    landuses_gdf = mock.mock_landuse_categorical_data(G)
    print(landuses_gdf.head())
    nodes_gdf, landuses_gdf = layers.compute_accessibilities(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        accessibility_keys=["a", "c"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    print(nodes_gdf.columns)
    # weighted form
    print(nodes_gdf["cc_c_400_wt"])
    # non-weighted form
    print(nodes_gdf["cc_c_400_nw"])
    # nearest distance to landuse
    print(nodes_gdf["cc_c_nearest_max_800"])
    ```

    """
    logger.info(f"Computing land-use accessibility for: {', '.join(accessibility_keys)}")
    # assign to network
    data_map = build_data_map(
        data_gdf,
        network_structure,
        max_netw_assign_dist,
        data_id_col,
        barriers_gdf=barriers_gdf,
        n_nearest_candidates=n_nearest_candidates,
    )
    # extract landuses
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    landuses_map = dict(data_gdf[landuse_column_label])  # type: ignore
    # call the underlying function
    partial_func = partial(
        data_map.accessibility,
        network_structure=network_structure,
        landuses_map=landuses_map,  # type: ignore
        accessibility_keys=accessibility_keys,
        distances=distances,
        betas=betas,
        minutes=minutes,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    acc_result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
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
    gdf_idx = nodes_gdf.index.intersection(acc_result.node_keys_py)  # type: ignore
    # create a dictionary to hold the data
    temp_data = {}
    # unpack accessibility data
    for acc_key in accessibility_keys:
        for dist_key in distances:
            ac_nw_data_key = config.prep_gdf_key(acc_key, dist_key, angular, weighted=False)
            temp_data[ac_nw_data_key] = acc_result.result[acc_key].unweighted[dist_key]  # type: ignore
            ac_wt_data_key = config.prep_gdf_key(acc_key, dist_key, angular, weighted=True)
            temp_data[ac_wt_data_key] = acc_result.result[acc_key].weighted[dist_key]  # type: ignore
            if dist_key == max(distances):
                ac_dist_data_key = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                temp_data[ac_dist_data_key] = acc_result.result[acc_key].distance[dist_key]  # type: ignore

    temp_df = pd.DataFrame(temp_data, index=acc_result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]  # type: ignore

    return nodes_gdf, data_gdf


def compute_mixed_uses(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 200,
    compute_hill: bool | None = True,
    compute_hill_weighted: bool | None = True,
    compute_shannon: bool | None = False,
    compute_gini: bool | None = False,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    n_nearest_candidates: int = 20,
    spatial_tolerance: int = 0,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    jitter_scale: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute landuse metrics.

    This function wraps the underlying `rust` optimised functions for aggregating and computing various mixed-use.
    These are computed simultaneously for any required combinations of measures (and distances). By default, hill and
    hill weighted measures will be computed, by the available flags e.g. `compute_hill` or `compute_shannon` can be used
    to configure which classes of measures should run.

    See the accompanying paper on `arXiv` for additional information about methods for computing mixed-use measures
    at the pedestrian scale.

    The data is aggregated and computed over the street network, with the implication that mixed-use and land-use
    accessibility aggregations are generated from the same locations as for centrality computations, which can
    therefore be correlated or otherwise compared. The outputs of the calculations are written to the corresponding
    node indices in the same `node_gdf` `GeoDataFrame` used for centrality methods, and which will display the
    calculated metrics under correspondingly labelled columns.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    landuse_column_label: str
        The column label from which to take landuse categories, e.g. a column labelled "landuse_categories" might
        contain "shop", "pub", "school", etc., landuse categories.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    compute_hill: bool
        Compute Hill diversity. This is the recommended form of diversity index. Computed for q of 0, 1, and 2.
    compute_hill_weighted: bool
        Compute distance weighted Hill diversity. This is the recommended form of diversity index. Computed for q of 0,
        1, and 2.
    compute_shannon: bool
        Compute shannon entropy. Hill diversity of q=1 is generally preferable.
    compute_gini: bool
        Compute the gini form of diversity index. Hill diversity of q=2 is generally preferable.
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
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    barriers_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing barriers. These barriers will be considered during the assignment of data points to the network.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas)
        for more information.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calculated metrics.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_nearest_assign`.

    Examples
    --------
    | key | formula | notes |
    |-----|:-------:|-------|
    | hill | $$q\geq{0},\ q\neq{1} \\ \big(\sum_{i}^{S}p_{i}^q\big)^{1/(1-q)} \\
    lim_{q\to1} \\ exp\big(-\sum_{i}^{S}\ p_{i}\ log\ p_{i}\big)$$ | Hill diversity: this is the
    preferred form of diversity metric because it adheres to the replication principle and uses units of effective
    species instead of measures of information or uncertainty. The `q` parameter controls the degree of emphasis on
    the _richness_ of species as opposed to the _balance_ of species. Over-emphasis on balance can be misleading in
    an urban context, for which reason research finds support for using `q=0`: this reduces to a simple count of
    distinct land-uses.|
    | hill_wt | $$\big[\sum_{i}^{S}d_{i}\big(\frac{p_{i}}{\bar{T}}\big)^{q} \big]^{1/(1-q)} \\
    \bar{T} = \sum_{i}^{S}d_{i}p_{i}$$ | This is a distance-weighted variant of Hill Diversity based
    on the distances from the point of computation to the nearest example of a particular land-use. It therefore
    gives a locally representative indication of the intensity of mixed-uses. $d_{i}$ is a negative exponential
    function where $\beta$ controls the strength of the decay. ($\beta$ is provided by the `Network Layer`, see
    [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas).)|
    | shannon | $$ -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or_information entropy_) is
    one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is
    effectively a transformation of Shannon diversity into units of effective species.|
    | gini | $$ 1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index.
    It can behave problematically because it does not adhere to the replication principle and places emphasis on the
    balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an
    emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a
    transformation of Gini-Simpson diversity into units of effective species.|

    :::note
    `hill_wt` at `q=0` is generally the best choice for granular landuse data, or else `q=1` or
    `q=2` for increasingly crude landuse classifications schemas.
    :::

    A worked example:
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs, io

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
    print(nodes_gdf.head())
    landuses_gdf = mock.mock_landuse_categorical_data(G)
    print(landuses_gdf.head())
    nodes_gdf, landuses_gdf = layers.compute_mixed_uses(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    # the data is written to the GeoDataFrame
    print(nodes_gdf.columns)
    # access accordingly, e.g. hill diversity at q=0 and 800m
    print(nodes_gdf["cc_hill_q0_800_nw"])
    ```
    :::warning
    Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that
    has been used. Meaningful comparisons from one location to another are only possible where the same schemas have
    been applied.
    :::

    """
    logger.info("Computing mixed-use measures.")
    # assign to network
    data_map = build_data_map(
        data_gdf,
        network_structure,
        max_netw_assign_dist,
        data_id_col,
        barriers_gdf=barriers_gdf,
        n_nearest_candidates=n_nearest_candidates,
    )
    # extract landuses
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    landuses_map = dict(data_gdf[landuse_column_label])  # type: ignore
    partial_func = partial(
        data_map.mixed_uses,
        network_structure=network_structure,
        landuses_map=landuses_map,  # type: ignore
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_hill=compute_hill,
        compute_hill_weighted=compute_hill_weighted,
        compute_shannon=compute_shannon,
        compute_gini=compute_gini,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
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
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)  # type: ignore
    # create a dictionary to hold the data
    temp_data = {}
    # unpack mixed-uses data
    for dist_key in distances:
        for q_key in [0, 1, 2]:
            if compute_hill:
                hill_nw_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular, weighted=False)
                temp_data[hill_nw_data_key] = result.hill[q_key][dist_key]  # type: ignore
            if compute_hill_weighted:
                hill_wt_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular, weighted=True)
                temp_data[hill_wt_data_key] = result.hill_weighted[q_key][dist_key]  # type: ignore
        if compute_shannon:
            shannon_data_key = config.prep_gdf_key("shannon", dist_key, angular)
            temp_data[shannon_data_key] = result.shannon[dist_key]  # type: ignore
        if compute_gini:
            gini_data_key = config.prep_gdf_key("gini", dist_key, angular)
            temp_data[gini_data_key] = result.gini[dist_key]  # type: ignore

    temp_df = pd.DataFrame(temp_data, index=result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]  # type: ignore

    return nodes_gdf, data_gdf


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_labels: list[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 200,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    spatial_tolerance: int = 0,
    n_nearest_candidates: int = 20,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    jitter_scale: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute numerical statistics over the street network.

    This function wraps the underlying `rust` optimised function for computing statistical measures. The data is
    aggregated and computed over the street network relative to the network nodes, with the implication
    that statistical aggregations are generated from the same locations as for centrality computations, which can
    therefore be correlated or otherwise compared.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    stats_column_labels: list[str]
        The column labels corresponding to the columns in `data_gdf` from which to take numerical information.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
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
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    barriers_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing barriers. These barriers will be considered during the assignment of data points to the network.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas)
        for more information.
    speed_m_s: float
        The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and
        distance thresholds $d_{max}$.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_nearest_assign`.

    Examples
    --------
    A worked example:

    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs, io

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
    print(nodes_gdf.head())
    numerical_gdf = mock.mock_numerical_data(G, num_arrs=3)
    print(numerical_gdf.head())
    nodes_gdf, numerical_gdf = layers.compute_stats(
        data_gdf=numerical_gdf,
        stats_column_label="mock_numerical_1",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    print(nodes_gdf.columns)
    # weighted form
    print(nodes_gdf["cc_mock_numerical_1_mean_400_wt"])
    # non-weighted form
    print(nodes_gdf["cc_mock_numerical_1_mean_400_nw"])
    ```

    :::note
    The following stat types will be available for each `stats_key` for each of the
    computed distances:
    - `max` and `min`
    - `sum` and `sum_wt`
    - `mean` and `mean_wt`
    - `variance` and `variance_wt`
    :::

    """
    logger.info("Computing statistics.")
    # assign to network
    data_map = build_data_map(
        data_gdf,
        network_structure,
        max_netw_assign_dist,
        data_id_col,
        barriers_gdf=barriers_gdf,
        n_nearest_candidates=n_nearest_candidates,
    )
    # extract stats columns
    stats_maps = []
    for stats_column_label in stats_column_labels:
        if stats_column_label not in data_gdf.columns:
            raise ValueError("The specified numerical stats column name can't be found in the GeoDataFrame.")
        stats_maps.append(dict(data_gdf[stats_column_label]))  # type: ignore
    # stats
    partial_func = partial(
        data_map.stats,
        network_structure=network_structure,
        numerical_maps=stats_maps,
        distances=distances,
        betas=betas,
        minutes=minutes,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    stats_result = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
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
    gdf_idx = nodes_gdf.index.intersection(stats_result.node_keys_py)  # type: ignore
    # create a dictionary to hold the data
    temp_data = {}
    # unpack the numerical arrays
    for idx, stats_column_label in enumerate(stats_column_labels):
        for dist_key in distances:
            k = config.prep_gdf_key(f"{stats_column_label}_sum", dist_key, angular=angular, weighted=False)
            temp_data[k] = stats_result.result[idx].sum[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_sum", dist_key, angular=angular, weighted=True)
            temp_data[k] = stats_result.result[idx].sum_wt[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_mean", dist_key, angular=angular, weighted=False)
            temp_data[k] = stats_result.result[idx].mean[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_mean", dist_key, angular=angular, weighted=True)
            temp_data[k] = stats_result.result[idx].mean_wt[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_count", dist_key, angular=angular, weighted=False)
            temp_data[k] = stats_result.result[idx].count[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_count", dist_key, angular=angular, weighted=True)
            temp_data[k] = stats_result.result[idx].count_wt[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_var", dist_key, angular=angular, weighted=False)
            temp_data[k] = stats_result.result[idx].variance[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_var", dist_key, angular=angular, weighted=True)
            temp_data[k] = stats_result.result[idx].variance_wt[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_max", dist_key, angular=angular)
            temp_data[k] = stats_result.result[idx].max[dist_key]  # type: ignore
            k = config.prep_gdf_key(f"{stats_column_label}_min", dist_key, angular=angular)
            temp_data[k] = stats_result.result[idx].min[dist_key]  # type: ignore

    temp_df = pd.DataFrame(temp_data, index=stats_result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]  # type: ignore

    return nodes_gdf, data_gdf
