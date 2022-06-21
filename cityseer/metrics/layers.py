from __future__ import annotations

import logging
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from numba_progress import ProgressBar  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import config, structures, types
from cityseer.algos import data
from cityseer.metrics import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
DataLayer Class representing data samples.

Categorical data, such as land-use classifications, and numerical data, such as valuations or census statistics,
can be assigned to the network as a [`DataLayer`](/metrics/layers/#datalayer). A `DataLayer` represents the spatial
locations of data points, and is used to calculate various mixed-use, land-use accessibility, and statistical
measures. Importantly, these measures are computed directly over the street network and offer distance-weighted
variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise
based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian
walking distances into account.

The coordinates of data points should correspond as precisely as possible to the location of the feature in space;
or, in the case of buildings, should ideally correspond to the location of the building entrance.

:::note
Note that in many cases, the [`DataLayerFromDict`](#datalayerfromdict) class will provide a more convenient
alternative for instantiating this class.
:::

"""


def assign_gdf_to_network(
    data_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: Union[int, float],
) -> tuple[structures.DataMap, gpd.GeoDataFrame]:
    """
    Assign a GeoDataFrame to a [`NetworkStructure`](/structures#networkstructure).

    A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the
    network. Data points will be assigned to the two closest network nodes — one in either direction — based on the
    closest adjacent street edge. This facilitates a dynamic spatial aggregation strategy which will select the shortest
    distance to a data point relative to either direction of approach.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe).
    network_structure: structures.NetworkStructure
        A [`NetworkStructure`](/structures#networkstructure).
    max_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network
        nodes.

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
    [graphs.nx_decompose](/tools/graphs/#nx-decompose)), which offers the additional benefit of a more granular
    representation of variations of metrics along street-fronts.
    :::

    ![Example assignment of data to a network](/images/assignment.png)
    _Example assignment on a non-decomposed graph._

    ![Example assignment of data to a network](/images/assignment_decomposed.png)
    _Assignment of data to network nodes becomes more contextually precise on decomposed graphs._

    """
    data_map = structures.DataMap(len(data_gdf))
    data_map.xs = data_gdf.geometry.x.values.astype(np.float32)
    data_map.ys = data_gdf.geometry.y.values.astype(np.float32)
    if "nearest_assign" not in data_gdf:
        if not config.QUIET_MODE:
            progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=len(data_gdf))
        else:
            progress_proxy = None
        data.assign_to_network(
            data_map,
            network_structure,
            np.float32(max_netw_assign_dist),
            progress_proxy=progress_proxy,
        )
        if progress_proxy is not None:
            progress_proxy.close()
        data_gdf["nearest_assign"] = data_map.nearest_assign
        data_gdf["next_nearest_assign"] = data_map.next_nearest_assign
    else:
        data_map.nearest_assign = data_gdf["nearest_assign"].values.astype(np.int_)
        data_map.next_nearest_assign = data_gdf["next_nearest_assign"].values.astype(np.int_)
    return data_map, data_gdf


def compute_landuses(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[types.DistancesType] = None,
    betas: Optional[types.BetasType] = None,
    mixed_use_keys: Optional[Union[list[str], tuple[str]]] = None,
    accessibility_keys: Optional[Union[list[str], tuple[str]]] = None,
    cl_disparity_wt_matrix: Optional[npt.NDArray[np.float32]] = None,
    qs: types.QsType = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute landuse metrics.

    This method wraps the underlying `numba` optimised functions for aggregating and computing various mixed-use and
    land-use accessibility measures. These are computed simultaneously for any required combinations of measures
    (and distances). Situations requiring only a single measure can instead make use of the simplified
    [`hill_diversity`](#hill-diversity), [`hill_branch_wt_diversity`](#hill-branch-wt-diversity), and
    [`DataLayer.compute_accessibilities`](#compute-accessibilities) methods.

    See the accompanying paper on `arXiv` for additional information about methods for computing mixed-use measures
    at the pedestrian scale.

    The data is aggregated and computed over the street network, with the implication that mixed-use and land-use
    accessibility aggregations are generated from the same locations as for centrality computations, which can
    therefore be correlated or otherwise compared. The outputs of the calculations are written to the corresponding
    node indices in the same `NetworkLayer.metrics_state` dictionary used for centrality methods, and will be
    categorised by the respective keys and parameters.

    For example, if `hill` and `shannon` mixed-use keys; `shops` and `factories` accessibility keys are computed on
    a `Network Layer` instantiated with 800m and 1600m distance thresholds, then the dictionary would assume the
    following structure:

    ```python
    NetworkLayer.metrics_state.mixed_uses = {
        # note that hill measures have q keys
        'hill': {
            # here, q=0
            0: {
                800: [...],
                1600: [...]
            },
            # here, q=1
            1: {
                800: [...],
                1600: [...]
            }
        },
        # non-hill measures do not have q keys
        'shannon': {
            800: [...],
            1600: [...]
        }
    }
    # accessibility keys are computed in both weighted and unweighted forms
    NetworkLayer.metrics_state.accessibility.weighted = {
        'shops': {
            800: [...],
            1600: [...]
        },
        'factories': {
            800: [...],
            1600: [...]
        }
    }
    NetworkLayer.metrics_state.accessibility.non_weighted = {
        'shops': {
            800: [...],
            1600: [...]
        },
        'factories': {
            800: [...],
            1600: [...]
        }
    }
    ```

    Parameters
    ----------
    landuse_labels: tuple[str]
        Land-use labels corresponding to the length and order of the data points. The labels should correspond to
        descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only required if
        computing mixed-uses or land-use accessibilities.
    mixed_use_keys: tuple[str]
        Mixed-use metrics to compute, containing any combination of the `key` values from the following table, by
        default None. See examples below for additional information.
    accessibility_keys
        Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use
        schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both
        `weighted` and `non_weighted` variants. By default None.
    cl_disparity_wt_matrix: ndarray[float]
        An optional pairwise `NxN` disparity matrix numerically describing the degree of disparity between any pair
        of distinct land-uses. This parameter is only required if computing mixed-uses using
        `hill_pairwise_disparity` or `raos_pairwise_disparity`.  The number and order of land-uses should match
        those implicitly generated by [`encode_categorical`](#encode_categorical). By default None.
    qs: tuple[float]
        The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of
        the Hill diversity mixed-use measures and is otherwise ignored. By default None.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`. Default of zero.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances. By default False

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
    [`distance_from_beta`](/metrics/networks/#distance-from-beta).)|
    | hill_pairwise_wt | $$\big[\sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} \big(  \frac{p_{i} p_{j}}{Q}
    \big)^{q} \big]^{1/(1-q)} \ Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | This is a
    pairwise-distance-weighted variant of Hill Diversity based on the respective distances between the closest
    examples of the pairwise distinct land-use combinations as routed through the point of computation.
    $d_{ij}$ represents a negative exponential function where $\beta$ controls the strength of the decay.
    ($\beta$ is provided by the `Network Layer`, see
    [`distance_from_beta`](/metrics/networks/#distance-from-beta).)|
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

    # generate the network layer
    cc_netw = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

    # prepare a mock data dictionary
    data_dict = mock.mock_data_dict(G, random_seed=25)
    # prepare some mock land-use classifications
    landuses = mock.mock_categorical_data(len(data_dict), random_seed=25)

    # generate a data layer
    L = layers.DataLayerFromDict(data_dict)
    # assign to the network
    L.assign_to_network(cc_netw, max_dist=500)
    # compute some metrics - here we'll use the full interface, see below for simplified interfaces
    # FULL INTERFACE
    # ==============
    L.compute_landuses(landuse_labels=landuses,
                        mixed_use_keys=['hill'],
                        qs=[0, 1],
                        accessibility_keys=['c', 'd', 'e'])
    # note that the above measures can optionally be run individually using simplified interfaces, e.g.
    # SIMPLIFIED INTERFACES
    # =====================
    # L.hill_diversity(landuses, qs=[0])
    # L.compute_accessibilities(landuses, ['a', 'b'])

    # let's prepare some keys for accessing the computational outputs
    # distance idx: any of the distances with which the NetworkLayer was initialised
    distance_idx = 200
    # q index: any of the invoked q parameters
    q_idx = 0
    # a node idx
    node_idx = 0

    # the data is available at cc_netw.metrics_state
    print(cc_netw.metrics_state.mixed_uses['hill'][q_idx][distance_idx][node_idx])
    # prints: 4.0
    print(cc_netw.metrics_state.accessibility.weighted['d'][distance_idx][node_idx])
    # prints: 0.019174593
    print(cc_netw.metrics_state.accessibility.non_weighted['d'][distance_idx][node_idx])
    # prints: 1.0
    ```

    Note that the data can also be unpacked to a dictionary using
    [`NetworkLayer.metrics_to_dict`](/metrics/networks/#networklayer-metrics-to-dict), or transposed to a `networkX`
    graph using [`NetworkLayer.to_nx_multigraph`](/metrics/networks/#networklayer-to-nx-multigraph).

    :::warning
    Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that
    has been used. Meaningful comparisons from one location to another are only possible where the same schemas have
    been applied.
    :::

    """
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
    # remember, most checks on parameter integrity occur in underlying method
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
    # figure out the corresponding indices for the landuse classes that are present in the dataset
    # these indices are passed as keys which will be matched against the integer landuse encodings
    acc_keys: list[int] = []
    if accessibility_keys is not None:
        for ac_label in accessibility_keys:
            if ac_label not in lab_enc.classes_:
                logger.warning(f"No instances of accessibility label: {ac_label} present in the data.")
            else:
                acc_keys.append(lab_enc.transform([ac_label]))  # type: ignore
        if not config.QUIET_MODE:
            logger.info(f'Computing land-use accessibility for: {", ".join(accessibility_keys)}')
    if not config.QUIET_MODE:
        progress_proxy = ProgressBar(update_interval=0.25, notebook=False, total=network_structure.nodes.count)
    else:
        progress_proxy = None
    # call the underlying method
    # pylint: disable=duplicate-code
    (mixed_use_hill_data, mixed_use_other_data, accessibility_data, accessibility_data_wt,) = data.aggregate_landuses(
        network_structure,
        data_map,
        distances=_distances,
        betas=_betas,
        landuse_encodings=encoded_labels,
        qs=np.array(qs, dtype=np.float32),
        mixed_use_hill_keys=np.array(mu_hill_keys, dtype=np.int_),
        mixed_use_other_keys=np.array(mu_other_keys, dtype=np.int_),
        accessibility_keys=np.array(acc_keys, dtype=np.int_),
        cl_disparity_wt_matrix=np.array(cl_disparity_wt_matrix, dtype=np.float32),
        jitter_scale=np.float32(jitter_scale),
        angular=angular,
        progress_proxy=progress_proxy,
    )
    if progress_proxy is not None:
        progress_proxy.close()
    # write the results to the NetworkLayer's metrics dict
    # keys will check for pre-existing, whereas qs and distance keys will overwrite
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
    # unpack accessibility data
    for ac_idx, ac_code in enumerate(acc_keys):
        ac_label: str = lab_enc.inverse_transform(ac_code)[0]  # type: ignore
        # non-weighted
        for d_idx, d_key in enumerate(_distances):
            ac_nw_data_key = config.prep_gdf_key(f"{ac_label}_{d_key}_non_weighted")
            nodes_gdf[ac_nw_data_key] = accessibility_data[ac_idx][d_idx]
        # weighted
        for d_idx, d_key in enumerate(_distances):
            ac_wt_data_key = config.prep_gdf_key(f"{ac_label}_{d_key}_weighted")
            nodes_gdf[ac_wt_data_key] = accessibility_data_wt[ac_idx][d_idx]

    return nodes_gdf, data_gdf


def hill_diversity(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[types.DistancesType] = None,
    betas: Optional[types.BetasType] = None,
    qs: types.QsType = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Compute hill diversity for the provided `landuse_labels` at the specified values of `q`.

    See [`DataLayer.compute_landuses`](#datalayer-compute-landuses) for additional information.

    Parameters
    ----------
    landuse_labels: tuple[str]
        Land-use labels corresponding to the length and order of the data points. The labels should correspond to
        descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only required if
        computing mixed-uses or land-use accessibilities.
    qs: tuple[float]
        The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of
        the Hill diversity mixed-use measures and is otherwise ignored. By default None.

    Examples
    --------
    The data key is `hill`, e.g.:

    `NetworkLayer.metrics_state.mixed_uses['hill'][q_key][distance_key][node_idx]`

    """
    return compute_landuses(
        data_gdf,
        landuse_column_label,
        nodes_gdf,
        network_structure,
        max_netw_assign_dist=max_netw_assign_dist,
        distances=distances,
        betas=betas,
        mixed_use_keys=["hill"],
        qs=qs,
        jitter_scale=jitter_scale,
        angular=angular,
    )


def hill_branch_wt_diversity(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[types.DistancesType] = None,
    betas: Optional[types.BetasType] = None,
    qs: types.QsType = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Compute distance-weighted hill diversity for the provided `landuse_labels` at the specified values of `q`.

    See [`DataLayer.compute_landuses`](#datalayer-compute-landuses) for additional information.

    Parameters
    ----------
    landuse_labels: tuple[str]
        Land-use labels corresponding to the length and order of the data points. The labels should correspond to
        descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only required if
        computing mixed-uses or land-use accessibilities.
    qs: tuple[float]
        The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of
        the Hill diversity mixed-use measures and is otherwise ignored. By default None.

    Examples
    --------
    The data key is `hill_branch_wt`, e.g.:

    `NetworkLayer.metrics_state.mixed_uses['hill_branch_wt'][q_key][distance_key][node_idx]`

    """
    return compute_landuses(
        data_gdf,
        landuse_column_label,
        nodes_gdf,
        network_structure,
        max_netw_assign_dist=max_netw_assign_dist,
        distances=distances,
        betas=betas,
        mixed_use_keys=["hill_branch_wt"],
        qs=qs,
        jitter_scale=jitter_scale,
        angular=angular,
    )


def compute_accessibilities(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    accessibility_keys: Union[list[str], tuple[str]],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[types.DistancesType] = None,
    betas: Optional[types.BetasType] = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Compute land-use accessibilities for the specified land-use classification keys.

    See [`DataLayer.compute_landuses`](#datalayer-compute-landuses) for additional information.

    Parameters
    ----------
    landuse_labels: tuple[str]
        Land-use labels corresponding to the length and order of the data points. The labels should correspond to
        descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only required if
        computing mixed-uses or land-use accessibilities.
    accessibility_keys
        Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use
        schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both
        `weighted` and `non_weighted` variants. By default None.

    Examples
    --------
    The data keys will correspond to the `accessibility_keys` specified, e.g. where computing `retail`
    accessibility:

    ```python
    NetworkLayer.metrics_state.accessibility.weighted['retail'][distance_key][node_idx]
    NetworkLayer.metrics_state.accessibility.non_weighted['retail'][distance_key][node_idx]
    ```

    """
    return compute_landuses(
        data_gdf,
        landuse_column_label,
        nodes_gdf,
        network_structure,
        max_netw_assign_dist=max_netw_assign_dist,
        distances=distances,
        betas=betas,
        accessibility_keys=accessibility_keys,
        jitter_scale=jitter_scale,
        angular=angular,
    )


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_labels: Union[str, list[str], tuple[str], npt.NDArray[np.unicode_]],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    max_netw_assign_dist: int = 400,
    distances: Optional[types.DistancesType] = None,
    betas: Optional[types.BetasType] = None,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Compute stats.

    This method wraps the underlying `numba` optimised functions for computing statistical measures. The data is
    aggregated and computed over the street network relative to the `Network Layer` nodes, with the implication
    that statistical aggregations are generated from the same locations as for centrality computations, which can
    therefore be correlated or otherwise compared. The outputs of the calculations are written to the corresponding
    node indices in the same `NetworkLayer.metrics_state` dictionary used for centrality methods, and will be
    categorised by the respective keys and parameters.

    For example, if a `valuations` stats key is computed on a `Network Layer` instantiated with 800m and 1600m
    distance thresholds, then the dictionary would assume the following structure:

    ```python
    NetworkLayer.metrics_state.stats = {
        # stats grouped by each stats key
        'valuations': {
            # each stat will have the following key-value pairs
            'max': {
                800: [...],
                1600: [...]
            },
            'min': {
                800: [...],
                1600: [...]
            },
            'sum': {
                800: [...],
                1600: [...]
            },
            'sum_weighted': {
                800: [...],
                1600: [...]
            },
            'mean': {
                800: [...],
                1600: [...]
            },
            'mean_weighted': {
                800: [...],
                1600: [...]
            },
            'variance': {
                800: [...],
                1600: [...]
            },
            'variance_weighted': {
                800: [...],
                1600: [...]
            }
        }
    }
    ```

    Parameters
    ----------
    stats_keys: str | tuple[str]
        If computing a single stat: a `str` key describing the stats computed for the `stats_data` parameter.
        If computing multiple stats: a `tuple` of keys. Computed stats will be stored under the supplied key. See
        examples below.
    stats_data: ndarray[int | float]
        If computing a single stat: a 1d array of numerical data, where the length corresponds to the number of data
        points in the `DataLayer`. If computing multiple stats keys: an array of numerical data, where the first
        dimension corresponds to the number of keys in the `stats_keys` parameter and the second dimension
        corresponds to number of data points in the `DataLayer`. e.g:
        ```python
        # if computing three keys for a DataLayer containg 5 data points
        stats_keys = ['valuations', 'floors', 'occupants']
        stats_data = [
            [50000, 60000, 55000, 42000, 46000],  # valuations
            [3, 3, 2, 3, 5],  # floors
            [420, 300, 220, 250, 600]  # occupants
        ]
        ```
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`. By default zero.
    angular
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances. By default False.

    Examples
    --------
    If computing three sets of stats data corresponding to stats_keys ['valuations', 'floors', 'occupants'], then
    the computed data is arranged as follows:

    ```python
    NetworkLayer.NetworkLayer.metrics_state.stats['valuations'][stat_type][distance_key][node_idx]
    NetworkLayer.NetworkLayer.metrics_state.stats['floors'][stat_type][distance_key][node_idx]
    NetworkLayer.NetworkLayer.metrics_state.stats['occupants'][stat_type][distance_key][node_idx]
    ```

    A worked example:
    ```python
    from cityseer.metrics import networks, layers
    from cityseer.tools import mock, graphs

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)

    # generate the network layer
    cc_netw = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

    # prepare a mock data dictionary
    data_dict = mock.mock_data_dict(G, random_seed=25)
    # let's prepare some numerical data
    stats_data = mock.mock_numerical_data(len(data_dict), num_arrs=1, random_seed=25)

    # generate a data layer
    L = layers.DataLayerFromDict(data_dict)
    # assign to the network
    L.assign_to_network(cc_netw, max_dist=500)
    # compute some metrics
    L.compute_stats(stats_keys='mock_stat',
                    stats_data=stats_data)
    # let's prepare some keys for accessing the computational outputs
    # distance idx: any of the distances with which the NetworkLayer was initialised
    distance_idx = 200
    # a node idx
    node_idx = 0

    # the data is available at cc_netw.metrics_state
    print(cc_netw.metrics_state.stats['mock_stat']['mean_weighted'][distance_idx][node_idx])
    # prints: 71.29311697979952
    ```

    Note that the data can also be unpacked to a dictionary using
    [`NetworkLayer.metrics_to_dict`](/metrics/networks/#networklayer-metrics-to-dict), or transposed to a `networkX`
    graph using [`NetworkLayer.to_nx_multigraph`](/metrics/networks/#networklayer-to-nx-multigraph).

    :::note
    Per the above worked example, the following stat types will be available for each `stats_key` for each of the
    computed distances:
    - `max` and `min`
    - `sum` and `sum_weighted`
    - `mean` and `mean_weighted`
    - `variance` and `variance_weighted`
    :::

    """
    _distances, _betas = networks.pair_distances_betas(distances, betas)
    data_map, data_gdf = assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=max_netw_assign_dist)
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
    # call the underlying method
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
        network_structure,
        data_map,
        distances=_distances,
        betas=_betas,
        numerical_arrays=stats_data_arrs,
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
