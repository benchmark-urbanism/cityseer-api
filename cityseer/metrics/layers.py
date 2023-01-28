from __future__ import annotations

import logging
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from numba_progress import ProgressBar  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import cctypes, config, structures
from cityseer.algos import data
from cityseer.metrics import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_gdf_to_network(
    data_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: Union[int, float],
    data_id_col: Optional[str] = None,
) -> tuple[structures.DataMap, gpd.GeoDataFrame]:
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
    network_structure.validate()
    data_map = structures.DataMap(len(data_gdf))
    data_map.xs = data_gdf.geometry.x.values.astype(np.float32)
    data_map.ys = data_gdf.geometry.y.values.astype(np.float32)
    if data_id_col is not None:
        lab_enc = LabelEncoder()
        data_map.data_id = lab_enc.fit_transform(data_gdf[data_id_col])  # type: ignore
    data_map.validate(False)
    if "nearest_assign" not in data_gdf:
        if not config.QUIET_MODE:
            progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=len(data_gdf))
        else:
            progress_proxy = None
        data_map_nearest_arr, data_map_next_nearest_arr = data.assign_to_network(
            data_map.xs,
            data_map.ys,
            data_map.nearest_assign,
            data_map.next_nearest_assign,
            network_structure.nodes.xs,
            network_structure.nodes.ys,
            network_structure.edges.end,
            network_structure.node_edge_map,
            np.float32(max_netw_assign_dist),
            progress_proxy=progress_proxy,
        )
        if progress_proxy is not None:
            progress_proxy.close()
        data_map.nearest_assign = data_map_nearest_arr
        data_map.next_nearest_assign = data_map_next_nearest_arr
        data_gdf["nearest_assign"] = data_map_nearest_arr
        data_gdf["next_nearest_assign"] = data_map_next_nearest_arr
    else:
        data_map.nearest_assign = data_gdf["nearest_assign"].values.astype(np.int_)
        data_map.next_nearest_assign = data_gdf["next_nearest_assign"].values.astype(np.int_)
    if np.all(data_map.nearest_assign == -1):
        logger.warning("No assignments for nearest assigned direction.")
    if np.all(data_map.next_nearest_assign == -1):
        logger.warning("No assignments for nearest assigned direction.")
    return data_map, data_gdf


def compute_landuses():
    """
    Please use the compute_accessibilities or compute_mixed_uses functions instead.
    """
    raise DeprecationWarning(
        "The compute_landuses function has been deprecated. Please use the compute_accessibilities or "
        "compute_mixed_uses functions instead."
    )


def compute_accessibilities(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    accessibility_keys: Union[list[str], tuple[str]],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[cctypes.DistancesType] = None,
    betas: Optional[cctypes.BetasType] = None,
    data_id_col: Optional[str] = None,
    spatial_tolerance: int = 0,
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
    network_structure.validate()
    _distances, _betas = networks.pair_distances_betas(distances, betas)
    data_map, data_gdf = assign_gdf_to_network(
        data_gdf, network_structure, max_netw_assign_dist=max_netw_assign_dist, data_id_col=data_id_col
    )
    # remember, most checks on parameter integrity occur in underlying function
    # so, don't duplicate here
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    # get the landuse encodings
    lab_enc = LabelEncoder()
    encoded_labels: npt.NDArray[np.int_] = lab_enc.fit_transform(data_gdf[landuse_column_label])  # type: ignore
    data_gdf[f"{landuse_column_label}_encoded"] = encoded_labels  # type: ignore
    # figure out the corresponding indices for the landuse classes that are present in the dataset
    # these indices are passed as keys which will be matched against the integer landuse encodings
    acc_keys: list[int] = []
    for ac_label in accessibility_keys:
        if ac_label not in lab_enc.classes_:  # type: ignore
            logger.warning(f"No instances of accessibility label '{ac_label}' in the data.")
        else:
            acc_keys.append(lab_enc.transform([ac_label]))  # type: ignore
    if not config.QUIET_MODE:
        logger.info(f'Computing land-use accessibility for: {", ".join(accessibility_keys)}')
    if not config.QUIET_MODE:
        progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=network_structure.nodes.count)
    else:
        progress_proxy = None
    # determine max impedance weights
    max_curve_wts = networks.clip_weights_curve(_distances, _betas, spatial_tolerance)
    # call the underlying function
    # pylint: disable=duplicate-code
    accessibility_data, accessibility_data_wt = data.accessibility(
        network_structure.nodes.xs,
        network_structure.nodes.ys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        data_map.xs,
        data_map.ys,
        data_map.nearest_assign,
        data_map.next_nearest_assign,
        data_map.data_id,
        distances=_distances,
        betas=_betas,
        max_curve_wts=max_curve_wts,
        landuse_encodings=encoded_labels,
        accessibility_keys=np.array(acc_keys, dtype=np.int_),
        jitter_scale=np.float32(jitter_scale),
        angular=angular,
        progress_proxy=progress_proxy,
    )
    if progress_proxy is not None:
        progress_proxy.close()
    # unpack accessibility data
    for ac_label in accessibility_keys:
        for d_idx, d_key in enumerate(_distances):
            ac_nw_data_key = config.prep_gdf_key(f"{ac_label}_{d_key}_non_weighted")
            ac_wt_data_key = config.prep_gdf_key(f"{ac_label}_{d_key}_weighted")
            # sometimes target classes are not present in data
            if ac_label in lab_enc.classes_:  # type: ignore
                ac_code: int = lab_enc.transform([ac_label])  # type: ignore
                ac_idx: int = acc_keys.index(ac_code)
                nodes_gdf[ac_nw_data_key] = accessibility_data[ac_idx][d_idx]  # non-weighted
                nodes_gdf[ac_wt_data_key] = accessibility_data_wt[ac_idx][d_idx]  # weighted
            else:
                nodes_gdf[ac_nw_data_key] = int(0)
                nodes_gdf[ac_wt_data_key] = float(0.0)

    return nodes_gdf, data_gdf


def compute_mixed_uses(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    mixed_use_keys: Union[list[str], tuple[str]],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[cctypes.DistancesType] = None,
    betas: Optional[cctypes.BetasType] = None,
    cl_disparity_wt_matrix: Optional[npt.NDArray[np.float32]] = None,
    qs: Optional[cctypes.QsType] = None,
    spatial_tolerance: int = 0,
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
    network_structure.validate()
    _distances, _betas = networks.pair_distances_betas(distances, betas)
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=max_netw_assign_dist)
    mixed_uses_options = [
        "hill",
        "hill_branch_wt",
        "hill_pairwise_wt",
        "hill_pairwise_disparity",
        "shannon",
        "gini_simpson",
        "raos_pairwise_disparity",
    ]
    # remember, most checks on parameter integrity occur in underlying function
    # so, don't duplicate here
    if landuse_column_label not in data_gdf.columns:
        raise ValueError("The specified landuse column name can't be found in the GeoDataFrame.")
    # get the landuse encodings
    lab_enc = LabelEncoder()
    encoded_labels: npt.NDArray[np.int_] = lab_enc.fit_transform(data_gdf[landuse_column_label])  # type: ignore
    data_gdf[f"{landuse_column_label}_encoded"] = encoded_labels  # type: ignore
    # if necessary, check the disparity matrix
    if cl_disparity_wt_matrix is None:
        cl_disparity_wt_matrix = np.full((0, 0), np.nan)
    elif (
        not isinstance(cl_disparity_wt_matrix, np.ndarray)
        or cl_disparity_wt_matrix.ndim != 2
        or cl_disparity_wt_matrix.shape[0] != cl_disparity_wt_matrix.shape[1]
        or len(cl_disparity_wt_matrix) != len(lab_enc.classes_)  # type: ignore
    ):
        raise TypeError(
            "Disparity weights must be a square pairwise NxN matrix in list, tuple, or numpy.ndarray form. "
            "The number of edge-wise elements should match the number of unique class labels."
        )
    # warn if no qs provided
    if qs is None:
        qs = tuple([])
    if isinstance(qs, (int, float)):
        qs = [qs]
    if not isinstance(qs, (list, tuple, np.ndarray)):
        raise TypeError("Please provide a float, list, tuple, or numpy.ndarray of q values.")
    # extrapolate the requested mixed use measures
    mu_hill_keys: list[int] = []
    mu_other_keys: list[int] = []
    if mixed_use_keys is not None:
        for mu in mixed_use_keys:
            if mu not in mixed_uses_options:
                raise ValueError(f'Invalid mixed-use option: {mu}. Must be one of {", ".join(mixed_uses_options)}.')
            idx = mixed_uses_options.index(mu)
            if idx < 4:
                mu_hill_keys.append(idx)
            else:
                mu_other_keys.append(idx - 4)
        if not config.QUIET_MODE:
            logger.info(f'Computing mixed-use measures: {", ".join(mixed_use_keys)}')
    if not config.QUIET_MODE:
        progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=network_structure.nodes.count)
    else:
        progress_proxy = None
    # determine max impedance weights
    max_curve_wts = networks.clip_weights_curve(_distances, _betas, spatial_tolerance)
    # call the underlying function
    # pylint: disable=duplicate-code
    mixed_use_hill_data, mixed_use_other_data = data.mixed_uses(
        network_structure.nodes.xs,
        network_structure.nodes.ys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        data_map.xs,
        data_map.ys,
        data_map.nearest_assign,
        data_map.next_nearest_assign,
        distances=_distances,
        betas=_betas,
        max_curve_wts=max_curve_wts,
        landuse_encodings=encoded_labels,
        qs=np.array(qs, dtype=np.float32),
        mixed_use_hill_keys=np.array(mu_hill_keys, dtype=np.int_),
        mixed_use_other_keys=np.array(mu_other_keys, dtype=np.int_),
        cl_disparity_wt_matrix=np.array(cl_disparity_wt_matrix, dtype=np.float32),
        jitter_scale=np.float32(jitter_scale),
        angular=angular,
        progress_proxy=progress_proxy,
    )
    if progress_proxy is not None:
        progress_proxy.close()
    # write the results to the GeoDataFrame
    # unpack mixed use hill
    for mu_h_idx, mu_h_key in enumerate(mu_hill_keys):
        mu_h_label = mixed_uses_options[mu_h_key]
        for q_idx, q_key in enumerate(qs):
            for d_idx, d_key in enumerate(_distances):
                mu_h_data_key = config.prep_gdf_key(f"{mu_h_label}_q{q_key}_{d_key}")
                nodes_gdf[mu_h_data_key] = mixed_use_hill_data[mu_h_idx][q_idx][d_idx]
    # unpack mixed use other
    for mu_o_idx, mu_o_key in enumerate(mu_other_keys):
        mu_o_label = mixed_uses_options[mu_o_key + 4]
        # no qs
        for d_idx, d_key in enumerate(_distances):
            mu_o_data_key = config.prep_gdf_key(f"{mu_o_label}_{d_key}")
            nodes_gdf[mu_o_data_key] = mixed_use_other_data[mu_o_idx][d_idx]

    return nodes_gdf, data_gdf


def hill_diversity(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[cctypes.DistancesType] = None,
    betas: Optional[cctypes.BetasType] = None,
    qs: cctypes.QsType = None,
    spatial_tolerance: int = 0,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute hill diversity for the provided `landuse_labels` at the specified values of `q`.

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
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#etwork-structure-from-nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`structures.NetworkStructure`](/structures#etworkstructure). Best generated with the
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
    nodes_gdf, landuses_gdf = layers.hill_diversity(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
        qs=[0, 1],
    )
    print(nodes_gdf.columns)
    print(nodes_gdf["cc_metric_hill_q1_400"])  # e.g. distance weighted hill at q=1 and 400m
    ```

    """
    return compute_mixed_uses(
        data_gdf,
        landuse_column_label,
        ["hill"],
        nodes_gdf,
        network_structure,
        max_netw_assign_dist=max_netw_assign_dist,
        distances=distances,
        betas=betas,
        qs=qs,
        spatial_tolerance=spatial_tolerance,
        jitter_scale=jitter_scale,
        angular=angular,
    )


def hill_branch_wt_diversity(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[cctypes.DistancesType] = None,
    betas: Optional[cctypes.BetasType] = None,
    qs: cctypes.QsType = None,
    spatial_tolerance: int = 0,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute distance-weighted hill diversity for the provided `landuse_labels` at the specified values of `q`.

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
    nodes_gdf, landuses_gdf = layers.hill_branch_wt_diversity(
        data_gdf=landuses_gdf,
        landuse_column_label="categorical_landuses",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
        qs=[0, 1],
    )
    print(nodes_gdf.columns)
    print(nodes_gdf["cc_metric_hill_branch_wt_q1_400"])  # e.g. distance weighted hill at q=1 and 400m
    ```

    """
    return compute_mixed_uses(
        data_gdf,
        landuse_column_label,
        ["hill_branch_wt"],
        nodes_gdf,
        network_structure,
        max_netw_assign_dist=max_netw_assign_dist,
        distances=distances,
        betas=betas,
        qs=qs,
        spatial_tolerance=spatial_tolerance,
        jitter_scale=jitter_scale,
        angular=angular,
    )


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_labels: Union[str, list[str], tuple[str], npt.NDArray[np.unicode_]],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[cctypes.DistancesType] = None,
    betas: Optional[cctypes.BetasType] = None,
    data_id_col: Optional[str] = None,
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
    network_structure.validate()
    _distances, _betas = networks.pair_distances_betas(distances, betas)
    data_map, data_gdf = assign_gdf_to_network(
        data_gdf, network_structure, max_netw_assign_dist=max_netw_assign_dist, data_id_col=data_id_col
    )
    data_map.validate(True)
    # check keys
    if not isinstance(stats_column_labels, (str, list, tuple, np.ndarray)):
        raise TypeError("Stats keys should be a string else a list, tuple, or np.ndarray of strings.")
    # wrap single keys
    if isinstance(stats_column_labels, str):
        stats_column_labels = [stats_column_labels]
    # check data arrays
    for col_label in stats_column_labels:
        if col_label not in data_gdf.columns:
            raise ValueError(f"Column label {col_label} not found in provided GeoDataFrame.")
    stats_data_arrs: npt.NDArray[np.float32] = data_gdf[stats_column_labels].values.T  # type: ignore
    # call the underlying function
    if not config.QUIET_MODE:
        progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=network_structure.nodes.count)
    else:
        progress_proxy = None
    # pylint: disable=duplicate-code
    (
        stats_sum,
        stats_sum_wt,
        stats_mean,
        stats_mean_wt,
        stats_variance,
        stats_variance_wt,
        stats_max,
        stats_min,
    ) = data.aggregate_stats(
        network_structure.nodes.xs,
        network_structure.nodes.ys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        data_map.xs,
        data_map.ys,
        data_map.nearest_assign,
        data_map.next_nearest_assign,
        data_map.data_id,
        distances=_distances,
        betas=_betas,
        numerical_arrays=stats_data_arrs,  # type: ignore
        jitter_scale=np.float32(jitter_scale),
        angular=angular,
        progress_proxy=progress_proxy,
    )
    if progress_proxy is not None:
        progress_proxy.close()
    # unpack the numerical arrays
    for num_idx, stats_key in enumerate(stats_column_labels):
        for stats_type_key, stats in zip(
            [
                "max",
                "min",
                "sum",
                "sum_weighted",
                "mean",
                "mean_weighted",
                "variance",
                "variance_weighted",
            ],
            [
                stats_max,
                stats_min,
                stats_sum,
                stats_sum_wt,
                stats_mean,
                stats_mean_wt,
                stats_variance,
                stats_variance_wt,
            ],
        ):
            for d_idx, d_key in enumerate(_distances):
                stats_data_key = config.prep_gdf_key(f"{stats_key}_{stats_type_key}_{d_key}")
                nodes_gdf[stats_data_key] = stats[num_idx][d_idx]

    return nodes_gdf, data_gdf
