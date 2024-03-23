from __future__ import annotations

import logging
from functools import partial

import geopandas as gpd

from cityseer import config, rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_gdf_to_network(
    data_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int | float,
    data_id_col: str | None = None,
) -> tuple[rustalgos.DataMap, gpd.GeoDataFrame]:
    """
    Assign a `GeoDataFrame` to a [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure).

    A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the
    network. Data points will be assigned to the two closest network nodes — one in either direction — based on the
    closest adjacent street edge. This facilitates a dynamic spatial aggregation strategy which will select the shortest
    distance to a data point relative to either direction of approach.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    network_structure: rustalgos.NetworkStructure
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure).
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.

    Returns
    -------
    data_map: rustalgos.DataMap
        A [`rustalgos.DataMap`](/rustalgos#datamap) instance.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    :::note
    The `max_assign_dist` parameter should not be set overly low. The `max_assign_dist` parameter sets a crow-flies
    distance limit on how far the algorithm will search in its attempts to encircle the data point. If the
    `max_assign_dist` is too small, then the algorithm is potentially hampered from finding a starting node; or, if a
    node is found, may have to terminate exploration prematurely because it can't travel sufficiently far from the
    data point to explore the surrounding network. If too many data points are not being successfully assigned to the
    correct street edges, then this distance should be increased. Conversely, if most of the data points are
    satisfactorily assigned, then it may be possible to decrease this threshold. A distance of around 400m may provide
    a good starting point.
    :::

    :::note
    The precision of assignment improves on decomposed networks (see
    [graphs.nx_decompose](/tools/graphs#nx-decompose)), which offers the additional benefit of a more granular
    representation of variations of metrics along street-fronts.
    :::

    ![Example assignment of data to a network](/images/assignment.png)
    _Example assignment on a non-decomposed graph._

    ![Example assignment of data to a network](/images/assignment_decomposed.png)
    _Assignment of data to network nodes becomes more contextually precise on decomposed graphs._

    """
    data_map = rustalgos.DataMap()
    calculate_assigned = False
    # add column to data_gdf
    if not ("nearest_assign" in data_gdf.columns and "next_nearest_assign" in data_gdf.columns):
        calculate_assigned = True
        data_gdf["nearest_assign"] = None
        data_gdf["next_nearest_assign"] = None
    # prepare the data_map
    for data_key, data_row in data_gdf.iterrows():  # type: ignore
        if not isinstance(data_key, str):
            raise ValueError("Data keys must be string instances.")
        data_id: str | None = None if data_id_col is None else str(data_row[data_id_col])  # type: ignore
        data_map.insert(
            data_key,
            # get key from GDF in case of different geom column name
            data_row[data_gdf.geometry.name].x,  # type: ignore
            data_row[data_gdf.geometry.name].y,  # type: ignore
            data_id,
            data_row["nearest_assign"],  # type: ignore
            data_row["next_nearest_assign"],  # type: ignore
        )
    # only compute if not already computed
    if calculate_assigned is True:
        for data_key in data_map.entry_keys():  # pylint: disable=not-an-iterable
            data_coord = data_map.get_data_coord(data_key)
            nearest_idx, next_nearest_idx = network_structure.assign_to_network(data_coord, max_netw_assign_dist)
            if nearest_idx is not None:
                data_map.set_nearest_assign(data_key, nearest_idx)
                data_gdf.at[data_key, "nearest_assign"] = nearest_idx
            if next_nearest_idx is not None:
                data_map.set_next_nearest_assign(data_key, next_nearest_idx)
                data_gdf.at[data_key, "next_nearest_assign"] = next_nearest_idx
    if data_map.none_assigned():
        logger.warning("No assignments for nearest assigned direction.")
    return data_map, data_gdf


def compute_accessibilities(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    accessibility_keys: list[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    data_id_col: str | None = None,
    angular: bool = False,
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
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
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: list[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas) for
        more information.
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
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs, io

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
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
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist, data_id_col)
    if not config.QUIET_MODE:
        logger.info(f'Computing land-use accessibility for: {", ".join(accessibility_keys)}')
    # extract landuses
    landuses_map: dict[str, str] = data_gdf[landuse_column_label].to_dict()  # type: ignore
    # call the underlying function
    partial_func = partial(
        data_map.accessibility,
        network_structure=network_structure,
        landuses_map=landuses_map,
        accessibility_keys=accessibility_keys,
        distances=distances,
        betas=betas,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    result = config.wrap_progress(total=network_structure.node_count(), rust_struct=data_map, partial_func=partial_func)
    # unpack accessibility data
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)  # pylint: disable=unpacking-non-sequence
    for acc_key in accessibility_keys:
        for dist_key in distances:
            ac_nw_data_key = config.prep_gdf_key(acc_key, dist_key, angular, weighted=False)
            nodes_gdf[ac_nw_data_key] = result[acc_key].unweighted[dist_key]  # type: ignore
            ac_wt_data_key = config.prep_gdf_key(acc_key, dist_key, angular, weighted=True)
            nodes_gdf[ac_wt_data_key] = result[acc_key].weighted[dist_key]  # type: ignore
            if dist_key == max(distances):
                ac_dist_data_key = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                nodes_gdf[ac_dist_data_key] = result[acc_key].distance[dist_key]  # type: ignore

    return nodes_gdf, data_gdf


def compute_mixed_uses(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int = 400,
    compute_hill: bool | None = True,
    compute_hill_weighted: bool | None = True,
    compute_shannon: bool | None = False,
    compute_gini: bool | None = False,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    data_id_col: str | None = None,
    angular: bool = False,
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
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
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
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
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: list[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas) for
        more information.
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
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
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
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist, data_id_col)
    if not config.QUIET_MODE:
        logger.info("Computing mixed-use measures.")
    # extract landuses
    landuses_map: dict[str, str] = data_gdf[landuse_column_label].to_dict()  # type: ignore
    partial_func = partial(
        data_map.mixed_uses,
        network_structure=network_structure,
        landuses_map=landuses_map,
        distances=distances,
        betas=betas,
        compute_hill=compute_hill,
        compute_hill_weighted=compute_hill_weighted,
        compute_shannon=compute_shannon,
        compute_gini=compute_gini,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    result = config.wrap_progress(total=network_structure.node_count(), rust_struct=data_map, partial_func=partial_func)
    # unpack mixed-uses data
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)  # pylint: disable=unpacking-non-sequence
    for dist_key in distances:
        for q_key in [0, 1, 2]:
            if compute_hill:
                hill_nw_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular, weighted=False)
                nodes_gdf[hill_nw_data_key] = result.hill[q_key][dist_key]  # type: ignore
            if compute_hill_weighted:
                hill_wt_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular, weighted=True)
                nodes_gdf[hill_wt_data_key] = result.hill_weighted[q_key][dist_key]  # type: ignore
        if compute_shannon:
            shannon_data_key = config.prep_gdf_key("shannon", dist_key, angular)
            nodes_gdf[shannon_data_key] = result.shannon[dist_key]  # type: ignore
        if compute_gini:
            gini_data_key = config.prep_gdf_key("gini", dist_key, angular)
            nodes_gdf[gini_data_key] = result.gini[dist_key]  # type: ignore

    return nodes_gdf, data_gdf


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    data_id_col: str | None = None,
    angular: bool = False,
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
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
    stats_column_label: str
        The column label corresponding to the column in `data_gdf` from which to take numerical information.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: list[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas) for
        more information.
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
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    A worked example:

    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs, io

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
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
    if stats_column_label not in data_gdf.columns:
        raise ValueError("The specified numerical stats column name can't be found in the GeoDataFrame.")
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist, data_id_col)
    if not config.QUIET_MODE:
        logger.info("Computing statistics.")
    # extract landuses
    stats_map: dict[str, float] = data_gdf[stats_column_label].to_dict()  # type: ignore
    # stats
    partial_func = partial(
        data_map.stats,
        network_structure=network_structure,
        numerical_map=stats_map,
        distances=distances,
        betas=betas,
        angular=angular,
        spatial_tolerance=spatial_tolerance,
        min_threshold_wt=min_threshold_wt,
        jitter_scale=jitter_scale,
    )
    # wraps progress bar
    result = config.wrap_progress(total=network_structure.node_count(), rust_struct=data_map, partial_func=partial_func)
    # unpack the numerical arrays
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)  # pylint: disable=unpacking-non-sequence
    for dist_key in distances:
        k = config.prep_gdf_key(f"{stats_column_label}_sum", dist_key, angular=angular, weighted=False)
        nodes_gdf[k] = result.sum[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_sum", dist_key, angular=angular, weighted=True)
        nodes_gdf[k] = result.sum_wt[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_mean", dist_key, angular=angular, weighted=False)
        nodes_gdf[k] = result.mean[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_mean", dist_key, angular=angular, weighted=True)
        nodes_gdf[k] = result.mean_wt[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_count", dist_key, angular=angular, weighted=False)
        nodes_gdf[k] = result.count[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_count", dist_key, angular=angular, weighted=True)
        nodes_gdf[k] = result.count_wt[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_var", dist_key, angular=angular, weighted=False)
        nodes_gdf[k] = result.variance[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_var", dist_key, angular=angular, weighted=True)
        nodes_gdf[k] = result.variance_wt[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_max", dist_key, angular=angular)
        nodes_gdf[k] = result.max[dist_key]  # type: ignore
        k = config.prep_gdf_key(f"{stats_column_label}_min", dist_key, angular=angular)
        nodes_gdf[k] = result.min[dist_key]  # type: ignore

    return nodes_gdf, data_gdf
