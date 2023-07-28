from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import numpy.typing as npt

from cityseer import config, rustalgos
from cityseer.metrics import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_gdf_to_network(
    data_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int | float,
    data_id_col: str | None = None,
) -> tuple[rustalgos.DataMap, gpd.GeoDataFrame]:
    """
    Assign a `GeoDataFrame` to a [`structures.NetworkStructure`](/structures#networkstructure).

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
    network_structure: structures.NetworkStructure
        A [`structures.NetworkStructure`](/structures#networkstructure).
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.

    Returns
    -------
    data_map: structures.DataMap
        A [`structures.DataMap`](/structures#datamap) instance.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    :::note
    The `max_assign_dist` parameter should not be set overly low. The `max_assign_dist` parameter sets a crow-flies
    distance limit on how far the algorithm will search in its attempts to encircle the data point. If the
    `max_assign_dist` is too small, then the algorithm is potentially hampered from finding a starting node; or, if a
    node is found, may have to terminate exploration prematurely because it can't travel sufficiently far from the data
    point to explore the surrounding network. If too many data points are not being successfully assigned to the correct
    street edges, then this distance should be increased. Conversely, if most of the data points are satisfactorily
    assigned, then it may be possible to decrease this threshold. A distance of around 400m may provide a good starting
    point.
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
    for data_key, data_row in data_gdf.iterrows():
        if not isinstance(data_key, str):
            raise ValueError("Data keys must be string instances.")
        data_id = None if data_id_col is None else str(data_row[data_id_col])
        data_map.insert(
            data_key,
            data_row["geometry"].x,
            data_row["geometry"].y,
            data_id,
            data_row["nearest_assign"],
            data_row["next_nearest_assign"],
        )
    # only compute if not already computed
    if calculate_assigned is True:
        for data_key in data_map.entry_keys():
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
    accessibility_keys: list[str] | tuple[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    data_id_col: str | None = None,
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute land-use accessibilities for the specified land-use classification keys.

    See [`compute_landuses`](#compute-landuses) for additional information.

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
    accessibility_keys: tuple[str]
        Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use
        schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both
        `weighted` and `non_weighted` variants.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`structures.NetworkStructure`](/structures#networkstructure). Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: float | ndarray[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`distance_from_beta`](/metrics/networks#clip-weights-curve).
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
    print(nodes_gdf.head())
    landuses_gdf = mock.mock_landuse_categorical_data(G)
    print(landuses_gdf.head())
    # some of the more commonly used measures can be accessed through simplified interfaces, e.g.
    nodes_gdf, landuses_gdf = layers.compute_accessibilities(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        accessibility_keys=["a", "c"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    print(nodes_gdf.columns)
    print(nodes_gdf["cc_metric_c_400_weighted"])  # weighted form
    print(nodes_gdf["cc_metric_c_400_non_weighted"])  # non-weighted form
    ```

    """
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist, data_id_col)
    if not config.QUIET_MODE:
        logger.info(f'Computing land-use accessibility for: {", ".join(accessibility_keys)}')
    # extract landuses
    landuses_map = data_gdf[landuse_column_label].to_dict()
    # call the underlying function
    accessibility_data = data_map.accessibility(
        network_structure,
        landuses_map,
        accessibility_keys,
        distances,
        betas,
        angular,
        spatial_tolerance,
        min_threshold_wt,
        jitter_scale,
    )
    # unpack accessibility data
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)
    for acc_key in accessibility_keys:
        for dist_key in distances:
            ac_nw_data_key = config.prep_gdf_key(f"{acc_key}_{dist_key}_non_weighted")
            ac_wt_data_key = config.prep_gdf_key(f"{acc_key}_{dist_key}_weighted")
            nodes_gdf[ac_nw_data_key] = accessibility_data[acc_key].unweighted[dist_key]  # non-weighted
            nodes_gdf[ac_wt_data_key] = accessibility_data[acc_key].weighted[dist_key]  # weighted

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
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute landuse metrics.

    This function wraps the underlying `numba` optimised functions for aggregating and computing various mixed-use and
    land-use accessibility measures. These are computed simultaneously for any required combinations of measures
    (and distances). Situations requiring only a single measure can instead make use of the simplified
    [`hill_diversity`](#hill-diversity), [`hill_branch_wt_diversity`](#hill-branch-wt-diversity), and
    [`compute_accessibilities`](#compute-accessibilities) functions.

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
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`structures.NetworkStructure`](/structures#networkstructure). Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: float | ndarray[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    mixed_use_keys: tuple[str]
        Mixed-use metrics to compute, containing any combination of the `key` values from the following table.
        See examples below for additional information.
    cl_disparity_wt_matrix: ndarray[float]
        An optional pairwise `NxN` disparity matrix numerically describing the degree of disparity between any pair
        of distinct land-uses. This parameter is only required if computing mixed-uses using
        `hill_pairwise_disparity` or `raos_pairwise_disparity`.
    qs: tuple[float]
        The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of
        the Hill diversity mixed-use measures and is otherwise ignored.
    spatial_tolerance: int
        Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint
        locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above
         weights corresponding to the given spatial tolerance and normalises to the new range. For background, see
        [`distance_from_beta`](/metrics/networks#clip-weights-curve).
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.

    Examples
    --------
    | key | formula | notes |
    |-----|:-------:|-------|
    | hill | $$q\geq{0},\ q\neq{1} \ \big(\sum_{i}^{S}p_{i}^q\big)^{1/(1-q)} \
    lim_{q\to1} \ exp\big(-\sum_{i}^{S}\ p_{i}\ log\ p_{i}\big)$$ | Hill diversity: this is the
    preferred form of diversity metric because it adheres to the replication principle and uses units of effective
    species instead of measures of information or uncertainty. The `q` parameter controls the degree of emphasis on
    the _richness_ of species as opposed to the _balance_ of species. Over-emphasis on balance can be misleading in
    an urban context, for which reason research finds support for using `q=0`: this reduces to a simple count of
    distinct land-uses.|
    | hill_branch_wt | $$\big[\sum_{i}^{S}d_{i}\big(\frac{p_{i}}{\bar{T}}\big)^{q} \big]^{1/(1-q)} \
    \bar{T} = \sum_{i}^{S}d_{i}p_{i}$$ | This is a distance-weighted variant of Hill Diversity based
    on the distances from the point of computation to the nearest example of a particular land-use. It therefore
    gives a locally representative indication of the intensity of mixed-uses. $d_{i}$ is a negative exponential
    function where $\beta$ controls the strength of the decay. ($\beta$ is provided by the `Network Layer`, see
    [`distance_from_beta`](/metrics/networks#distance-from-beta).)|
    | hill_pairwise_wt | $$\big[\sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} \big(  \frac{p_{i} p_{j}}{Q}
    \big)^{q} \big]^{1/(1-q)} \ Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | This is a
    pairwise-distance-weighted variant of Hill Diversity based on the respective distances between the closest
    examples of the pairwise distinct land-use combinations as routed through the point of computation.
    $d_{ij}$ represents a negative exponential function where $\beta$ controls the strength of the decay.
    ($\beta$ is provided by the `Network Layer`, see
    [`distance_from_beta`](/metrics/networks#distance-from-beta).)|
    | hill_pairwise_disparity | $$\big[ \sum_{i}^{S} \sum_{j\neq{i}}^{S} w_{ij} \big(  \frac{p_{i}
    p_{j}}{Q} \big)^{q} \big]^{1/(1-q)} \ Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} w_{ij} p_{i}
    p_{j}$$ | This is a disparity-weighted variant of Hill Diversity based on the pairwise disparities between
    land-uses. This variant requires the use of a disparity matrix provided through the `cl_disparity_wt_matrix`
    parameter.|
    | shannon | $$ -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or_information entropy_) is
    one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is
    effectively a transformation of Shannon diversity into units of effective species.|
    | gini_simpson | $$ 1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index.
    It can behave problematically because it does not adhere to the replication principle and places emphasis on the
    balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an
    emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a
    transformation of Gini-Simpson diversity into units of effective species.|
    | raos_pairwise_disparity | $$ \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | Rao diversity
    is a pairwise disparity measure and requires the use of a disparity matrix provided through the
    `cl_disparity_wt_matrix` parameter. It suffers from the same issues as Gini-Simpson. It is preferable to use
    disparity weighted Hill diversity with `q=2`.|

    :::note
    `hill_branch_wt` paired with `q=0` is generally the best choice for granular landuse data, or else `q=1` or
    `q=2` for increasingly crude landuse classifications schemas.
    :::

    A worked example:
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
    print(nodes_gdf.head())
    landuses_gdf = mock.mock_landuse_categorical_data(G)
    print(landuses_gdf.head())
    # compute some metrics - here we'll use the full interface, see below for simplified interfaces
    nodes_gdf, landuses_gdf = layers.compute_mixed_uses(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
        mixed_use_keys=["hill"],
        qs=[0, 1],
    )
    print(nodes_gdf.columns)  # the data is written to the GeoDataFrame
    print(nodes_gdf["cc_metric_hill_q0_800"])  # access accordingly, e.g. hill diversity at q=0 and 800m
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
        logger.info(f"Computing mixed-use measures.")
    # extract landuses
    landuses_map = data_gdf[landuse_column_label].to_dict()
    mixed_uses_data = data_map.mixed_uses(
        network_structure,
        landuses_map,
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
    # unpack mixed-uses data
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)
    for dist_key in distances:
        for q_key in [0, 1, 2]:
            if compute_hill:
                hill_nw_data_key = config.prep_gdf_key(f"q{q_key}_{dist_key}_hill")
                nodes_gdf[hill_nw_data_key] = mixed_uses_data.hill[q_key][dist_key]
            if compute_hill_weighted:
                hill_wt_data_key = config.prep_gdf_key(f"q{q_key}_{dist_key}_hill_weighted")
                nodes_gdf[hill_wt_data_key] = mixed_uses_data.hill_weighted[q_key][dist_key]
        if compute_shannon:
            shannon_data_key = config.prep_gdf_key(f"{dist_key}_shannon")
            nodes_gdf[shannon_data_key] = mixed_uses_data.shannon[dist_key]
        if compute_gini:
            gini_data_key = config.prep_gdf_key(f"{dist_key}_gini")
            nodes_gdf[gini_data_key] = mixed_uses_data.gini[dist_key]

    return nodes_gdf, data_gdf


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_label: str | list[str] | tuple[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    data_id_col: str | None = None,
    spatial_tolerance: int = 0,
    min_threshold_wt: float | None = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute stats.

    This function wraps the underlying `numba` optimised functions for computing statistical measures. The data is
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
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`structures.NetworkStructure`](/structures#networkstructure). Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: float | ndarray[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    data_id_col: str
        An optional column name for data point keys. This is used for deduplicating points representing a shared source
        of information. For example, where a single greenspace is represented by many entrances as datapoints, only the
        nearest entrance (from a respective location) will be considered (during aggregations) when the points share a
        datapoint identifier.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.

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
    from cityseer.tools import mock, graphs

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
    print(nodes_gdf.head())
    numerical_gdf = mock.mock_numerical_data(G, num_arrs=3)
    print(numerical_gdf.head())

    # some of the more commonly used measures can be accessed through simplified interfaces, e.g.
    nodes_gdf, numerical_gdf = layers.compute_stats(
        data_gdf=numerical_gdf,
        stats_column_labels=["mock_numerical_1"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    print(nodes_gdf.columns)
    print(nodes_gdf["cc_metric_mock_numerical_1_mean_weighted_400"])  # weighted form
    print(nodes_gdf["cc_metric_mock_numerical_1_sum_200"])  # non-weighted form
    ```

    :::note
    The following stat types will be available for each `stats_key` for each of the
    computed distances:
    - `max` and `min`
    - `sum` and `sum_weighted`
    - `mean` and `mean_weighted`
    - `variance` and `variance_weighted`
    :::

    """
    if stats_column_label not in data_gdf.columns:
        raise ValueError("The specified numerical stats column name can't be found in the GeoDataFrame.")
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist, data_id_col)
    if not config.QUIET_MODE:
        logger.info(f"Computing mixed-use measures.")
    # extract landuses
    stats_map = data_gdf[stats_column_label].to_dict()
    # stats
    stats_result = data_map.stats(
        network_structure, stats_map, distances, betas, angular, spatial_tolerance, min_threshold_wt, jitter_scale
    )
    # unpack the numerical arrays
    distances, betas = rustalgos.pair_distances_and_betas(distances, betas)
    for dist_key in distances:
        nodes_gdf[config.prep_gdf_key(f"sum_{dist_key}")] = stats_result.sum[dist_key]
        nodes_gdf[config.prep_gdf_key(f"sum_wt_{dist_key}")] = stats_result.sum_wt[dist_key]
        nodes_gdf[config.prep_gdf_key(f"mean_{dist_key}")] = stats_result.mean[dist_key]
        nodes_gdf[config.prep_gdf_key(f"mean_wt_{dist_key}")] = stats_result.mean_wt[dist_key]
        nodes_gdf[config.prep_gdf_key(f"count_{dist_key}")] = stats_result.count[dist_key]
        nodes_gdf[config.prep_gdf_key(f"count_wt_{dist_key}")] = stats_result.count_wt[dist_key]
        nodes_gdf[config.prep_gdf_key(f"variance_{dist_key}")] = stats_result.variance[dist_key]
        nodes_gdf[config.prep_gdf_key(f"variance_wt_{dist_key}")] = stats_result.variance_wt[dist_key]
        nodes_gdf[config.prep_gdf_key(f"max_{dist_key}")] = stats_result.max[dist_key]
        nodes_gdf[config.prep_gdf_key(f"min_{dist_key}")] = stats_result.min[dist_key]

    return nodes_gdf, data_gdf
