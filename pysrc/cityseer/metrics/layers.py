"""Compute land-use accessibility, mixed-use diversity, and statistical aggregations over the street network.

Data points (land uses, numerical attributes) are assigned to the nearest street edges and then summarised within
walking-distance catchments around each node, measured along the actual street network rather than as straight-line
distances. Because these summaries are computed at the same node locations used for centrality, you can directly
compare how well-connected a location is with how accessible different amenities are from that location. An optional
``decay_fn`` parameter controls how distance affects the weighting; see the [`cityseer.decay`](/api/decay) module for
preset helpers. ``decay_fn`` also accepts a ``{label: expression}`` dict to compute several decay variants in a single
network traversal, with each label appended to that variant's output column names.

For practical worked examples, see the [Cityseer Examples](https://cityseer.benchmarkurbanism.com/examples)
site, including the
[OSM Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility/osm-accessibility),
[Mixed Uses](https://cityseer.benchmarkurbanism.com/examples/accessibility/gpd-mixed-uses), and
[Statistical Aggregations](https://cityseer.benchmarkurbanism.com/examples/stats/gpd-stats) recipes.
"""

from __future__ import annotations

import logging
from functools import partial

import geopandas as gpd
import pandas as pd

from .. import config, rustalgos

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


def _resolve_decay_fns(decay_fn: str | dict[str, str] | None) -> tuple[list[str], list[str], bool]:
    """Resolve ``decay_fn`` into aligned ``(labels, expressions, legacy_tail)``.

    - ``None``: the legacy default — both an unweighted (``_nw``) and a beta-weighted (``_wt``)
      column, matching pre-v5 behaviour. Pass a single ``decay_fn`` to get just one column.
    - ``str``: a single decay with an empty label (output columns unsuffixed).
    - ``dict``: ``{label: expression}`` — each label suffixes its output columns, and all decays
      are computed in a single shared network traversal.

    ``legacy_tail`` is True only for the ``None`` default; it places the ``_nw``/``_wt`` suffix at
    the end of the column name (``cc_key_dist_nw``), as in pre-v5. The dict form embeds the label
    in the metric name instead (``cc_key_label_dist``).
    """
    if decay_fn is None:
        # legacy default: unweighted (_nw) + beta-weighted (_wt), matching pre-v5 output.
        return ["nw", "wt"], ["1", "exp(-4 * p)"], True
    if isinstance(decay_fn, str):
        return [""], [decay_fn], False
    if isinstance(decay_fn, dict):
        if not decay_fn:
            raise ValueError("decay_fn dict must contain at least one {label: expression} entry.")
        labels = [str(label) for label in decay_fn]
        exprs = [str(expr) for expr in decay_fn.values()]
        return labels, exprs, False
    raise TypeError("decay_fn must be a string, a dict of {label: expression}, or None.")


def _decay_suffix(label: str) -> str:
    """Column-key suffix for a decay label; empty label yields no suffix (back-compatible)."""
    return f"_{label}" if label else ""


def _layer_col_key(metric_key: str, dist: int, angular: bool, decay_label: str, legacy_tail: bool) -> str:
    """Build the output column key for a decay variant.

    The legacy default appends the ``_nw``/``_wt`` suffix at the end (``cc_key_dist_nw``, pre-v5);
    the dict/per-label form embeds the label in the metric name (``cc_key_label_dist``).
    """
    if legacy_tail:
        return config.prep_gdf_key(metric_key, dist, angular) + _decay_suffix(decay_label)
    return config.prep_gdf_key(f"{metric_key}{_decay_suffix(decay_label)}", dist, angular)


# Statistical measures produced by compute_stats, mapped to their `Stats` result accessor.
# The dict key is both the user-facing token and the output column suffix.
_STATS_ACCESSORS: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "count": "count",
    "var": "variance",
    "median": "median",
    "mad": "mad",
    "max": "max",
    "min": "min",
}


def _resolve_stats_measures(measures: list[str] | None) -> list[str]:
    """Validate the ``measures`` selection for compute_stats; ``None`` means all measures."""
    if measures is None:
        return list(_STATS_ACCESSORS)
    if not measures:
        raise ValueError("measures must contain at least one measure, or be None for all.")
    invalid = [m for m in measures if m not in _STATS_ACCESSORS]
    if invalid:
        raise ValueError(f"Unknown stats measure(s): {invalid}. Allowed: {list(_STATS_ACCESSORS)}.")
    return list(measures)


def build_data_map(
    data_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 100,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    n_nearest_candidates: int = 50,
) -> rustalgos.data.DataMap:
    """
    Assign a `GeoDataFrame` to a [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).

    A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the
    network. Points will be assigned to the closest street edge. Polygons will be assigned to the closest
    `n_nearest_candidates` adjacent street edges.

    Parameters
    ----------
    data_gdf: GeoDataFrame
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing data points. The coordinates of data points should correspond as precisely as possible to the
        location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the
        building entrance.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function.
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
        A [`rustalgos.data.DataMap`](/rustalgos/data#datamap) instance.
    """
    # check for unique index
    if data_gdf.index.duplicated().any():
        raise ValueError("The data GeoDataFrame index must contain unique entries.")
    # create data map
    data_map = rustalgos.data.DataMap()
    # prepare the data_map
    logger.info("Assigning data to network.")
    for data_key, data_row in data_gdf.iterrows():
        data_id = None if data_id_col is None else data_row[data_id_col]
        data_map.insert(
            data_key,
            data_row[data_gdf.active_geometry_name].wkt,
            data_id,
        )
    # barrier geoms
    barriers_wkt: list[str] | None = None
    if barriers_gdf is not None:
        barriers_wkt = []
        for _, row in barriers_gdf.iterrows():
            barriers_wkt.append(row.geometry.wkt)
    if barriers_wkt is not None:
        network_structure.set_barriers(barriers_wkt)
    data_map.assign_data_to_network(network_structure, max_netw_assign_dist, n_nearest_candidates)
    network_structure.unset_barriers()

    return data_map


def compute_accessibilities(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    accessibility_keys: list[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 100,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    n_nearest_candidates: int = 50,
    speed_m_s: float = SPEED_M_S,
    decay_fn: str | dict[str, str] | None = None,
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
        schema used for the `landuse_labels` parameter, e.g. "pub".
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int]
        Distance thresholds in metres for the network traversal. Metrics are computed for each
        threshold independently. If not provided, the `minutes` parameter must be provided instead.
    minutes: list[float]
        Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`.
        If not provided, the `distances` parameter must be provided instead.
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
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    speed_m_s: float
        Walking speed in metres per second used to convert `minutes` to distance thresholds.
    decay_fn: str | dict[str, str]
        An optional decay function expression using the variable `p`, where `p` is the normalised
        distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the
        accessibility count weighting. When omitted (`None`), the legacy default computes both an
        unweighted (`_nw`) and a decay-weighted (`_wt`) column; pass a single expression such as
        `"1"` (flat) to compute one unsuffixed column. For
        distance-weighted metrics, provide an expression such as `"exp(-4 * p)"` for exponential
        decay, or use the `cityseer.decay` module helpers to generate expressions from absolute
        distance units; see [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of
        `{label: expression}` to compute several decays in a single network traversal; each label
        is appended to that variant's output column names (a plain string or `None` adds no suffix).

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calculated metrics. Two
        columns will be returned for each input landuse class and distance combination; a count of reachable
        locations, and the smallest distance to the nearest location.
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
    # the default emits an unweighted (_nw) and a weighted (_wt) column;
    # pass a single decay_fn (e.g. "1") to compute just one and save time
    print(nodes_gdf["cc_c_400_nw"])
    # nearest distance to landuse (decay-independent: one column)
    print(nodes_gdf["cc_c_nearest_max_800"])
    ```

    For worked examples with real-world data, see the
    [OSM Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility/osm-accessibility)
    recipe.

    """
    if angular:
        _require_dual_for_angular(network_structure, "compute_accessibilities")
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
    landuses_map = dict(data_gdf[landuse_column_label])
    # call the underlying function
    decay_labels, decay_exprs, legacy_tail = _resolve_decay_fns(decay_fn)
    partial_func = partial(
        data_map.accessibility_decays,
        network_structure=network_structure,
        landuses_map=landuses_map,
        accessibility_keys=accessibility_keys,
        decay_fns=decay_exprs,
        distances=distances,
        minutes=minutes,
        angular=angular,
        speed_m_s=speed_m_s,
    )
    # wraps progress bar (returns one AccessibilityResult per decay label)
    acc_results = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
    )
    # unpack
    distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(acc_results[0].node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    # unpack accessibility data (one result per decay label; empty label -> no suffix)
    for decay_label, acc_result in zip(decay_labels, acc_results, strict=True):
        for acc_key in accessibility_keys:
            for dist_key in distances:
                ac_data_key = _layer_col_key(acc_key, dist_key, angular, decay_label, legacy_tail)
                temp_data[ac_data_key] = acc_result.result[acc_key].count[dist_key]
                if dist_key == max(distances):
                    # nearest distance is decay-independent: one unsuffixed column
                    ac_dist_data_key = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                    temp_data[ac_dist_data_key] = acc_result.result[acc_key].distance[dist_key]

    temp_df = pd.DataFrame(temp_data, index=acc_results[0].node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]

    return nodes_gdf, data_gdf


def compute_mixed_uses(
    data_gdf: gpd.GeoDataFrame,
    landuse_column_label: str,
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 100,
    compute_hill: bool | None = True,
    compute_shannon: bool | None = False,
    compute_gini: bool | None = False,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    n_nearest_candidates: int = 50,
    speed_m_s: float = SPEED_M_S,
    decay_fn: str | dict[str, str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""
    Compute landuse metrics.

    This function wraps the underlying `rust` optimised functions for aggregating and computing various mixed-use.
    These are computed simultaneously for any required combinations of measures (and distances). By default, hill
    measures will be computed, but the available flags e.g. `compute_hill` or `compute_shannon` can be used
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
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    compute_hill: bool
        Compute Hill diversity. This is the recommended form of diversity index. Computed for q of 0, 1, and 2.
    compute_shannon: bool
        Compute shannon entropy. Hill diversity of q=1 is generally preferable.
    compute_gini: bool
        Compute the gini form of diversity index. Hill diversity of q=2 is generally preferable.
    distances: list[int]
        Distance thresholds in metres for the network traversal. Metrics are computed for each
        threshold independently. If not provided, the `minutes` parameter must be provided instead.
    minutes: list[float]
        Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`.
        If not provided, the `distances` parameter must be provided instead.
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
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    speed_m_s: float
        Walking speed in metres per second used to convert `minutes` to distance thresholds.
    decay_fn: str | dict[str, str]
        An optional decay function expression using the variable `p`, where `p` is the normalised
        distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the
        Hill diversity weighting. When omitted (`None`), the legacy default computes both an
        unweighted (`_nw`) and a decay-weighted (`_wt`) variant; pass a single expression such as
        `"1"` (flat) to compute one unsuffixed variant. For
        distance-weighted metrics, provide an expression such as `"exp(-4 * p)"` for exponential
        decay, or use the `cityseer.decay` module helpers to generate expressions from absolute
        distance units; see [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of
        `{label: expression}` to compute several decays in a single network traversal; each label
        is appended to that variant's output column names (a plain string or `None` adds no suffix).

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
    | shannon | $$ -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or_information entropy_) is
    one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is
    effectively a transformation of Shannon diversity into units of effective species.|
    | gini | $$ 1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index.
    It can behave problematically because it does not adhere to the replication principle and places emphasis on the
    balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an
    emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a
    transformation of Gini-Simpson diversity into units of effective species.|

    :::note
    `hill` at `q=0` is generally the best choice for granular landuse data, or else `q=1` or
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
    # the default emits _nw and _wt; pass a single decay_fn to compute just one and save time
    print(nodes_gdf["cc_hill_q0_800_nw"])
    ```
    :::warning
    Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that
    has been used. Meaningful comparisons from one location to another are only possible where the same schemas have
    been applied.
    :::

    For a worked example, see the
    [Mixed Uses](https://cityseer.benchmarkurbanism.com/examples/accessibility/gpd-mixed-uses)
    recipe.

    """
    if angular:
        _require_dual_for_angular(network_structure, "compute_mixed_uses")
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
    landuses_map = dict(data_gdf[landuse_column_label])
    decay_labels, decay_exprs, legacy_tail = _resolve_decay_fns(decay_fn)
    partial_func = partial(
        data_map.mixed_uses_decays,
        network_structure=network_structure,
        landuses_map=landuses_map,
        decay_fns=decay_exprs,
        distances=distances,
        minutes=minutes,
        compute_hill=compute_hill,
        compute_shannon=compute_shannon,
        compute_gini=compute_gini,
        angular=angular,
        speed_m_s=speed_m_s,
    )
    # wraps progress bar (returns one MixedUsesResult per decay label)
    results = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
    )
    # unpack
    distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(results[0].node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    # unpack mixed-uses data (one result per decay label; empty label -> no suffix)
    # note: shannon and gini are decay-independent, so they repeat across labels.
    for decay_label, result in zip(decay_labels, results, strict=True):
        for dist_key in distances:
            for q_key in [0, 1, 2]:
                if compute_hill:
                    hill_data_key = _layer_col_key(f"hill_q{q_key}", dist_key, angular, decay_label, legacy_tail)
                    temp_data[hill_data_key] = result.hill[q_key][dist_key]
            if compute_shannon:
                # shannon is decay-independent: one unsuffixed column
                shannon_data_key = config.prep_gdf_key("shannon", dist_key, angular)
                temp_data[shannon_data_key] = result.shannon[dist_key]
            if compute_gini:
                # gini is decay-independent: one unsuffixed column
                gini_data_key = config.prep_gdf_key("gini", dist_key, angular)
                temp_data[gini_data_key] = result.gini[dist_key]

    temp_df = pd.DataFrame(temp_data, index=results[0].node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]

    return nodes_gdf, data_gdf


def compute_stats(
    data_gdf: gpd.GeoDataFrame,
    stats_column_labels: list[str],
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: rustalgos.graph.NetworkStructure,
    max_netw_assign_dist: int = 100,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    data_id_col: str | None = None,
    barriers_gdf: gpd.GeoDataFrame | None = None,
    angular: bool = False,
    n_nearest_candidates: int = 50,
    speed_m_s: float = SPEED_M_S,
    decay_fn: str | dict[str, str] | None = None,
    measures: list[str] | None = None,
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
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the function.
    network_structure
        A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure). Best generated with the
        [`io.network_structure_from_nx`](/tools/io#network_structure_from_nx) function.
    max_netw_assign_dist: int
        The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.
    distances: list[int]
        Distance thresholds in metres for the network traversal. Metrics are computed for each
        threshold independently. If not provided, the `minutes` parameter must be provided instead.
    minutes: list[float]
        Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`.
        If not provided, the `distances` parameter must be provided instead.
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
    n_nearest_candidates: int
        The number of nearest candidates to consider when assigning respective data points to the nearest adjacent
        streets.
    speed_m_s: float
        Walking speed in metres per second used to convert `minutes` to distance thresholds.
    decay_fn: str | dict[str, str]
        An optional decay function expression using the variable `p`, where `p` is the normalised
        distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the
        statistical weighting. When omitted (`None`), the legacy default computes both an
        unweighted (`_nw`) and a decay-weighted (`_wt`) variant; pass a single expression such as
        `"1"` (flat) to compute one unsuffixed variant. For
        distance-weighted metrics, provide an expression such as `"exp(-4 * p)"` for exponential
        decay, or use the `cityseer.decay` module helpers. Values are clamped to [0, 1]. Supported
        functions include `exp`, `ln`, `log`, `log10`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `sin`,
        `cos`, `tan`, and the `^` operator (`min`/`max` are not supported; to negate a power write
        `-((x)^2)` not `-(x)^2`). When multiple distances are specified, `p` is normalised independently per
        threshold. See [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of
        `{label: expression}` to compute several decays in a single network traversal; each label
        is appended to that variant's output column names (a plain string or `None` adds no suffix).
    measures: list[str]
        An optional subset of statistical measures to compute, chosen from `"sum"`, `"mean"`,
        `"count"`, `"var"`, `"median"`, `"mad"`, `"max"`, and `"min"`. Defaults to `None`, which
        computes all of them. Restricting the set keeps the output `GeoDataFrame` smaller and skips
        the weighted median / MAD sort when neither `"median"` nor `"mad"` is requested.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calculated metrics.
    data_gdf: GeoDataFrame
        The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_nearest_assign`.

    Examples
    --------
    Default exponential decay at multiple scales:

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
        stats_column_labels=["mock_numerical_1"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[200, 400, 800],
    )
    print(nodes_gdf.columns)
    # mean at 400m; the default emits _nw and _wt. Pass a single decay_fn for just one,
    # and measures=[...] to compute only the statistics you need — both save time
    print(nodes_gdf["cc_mock_numerical_1_mean_400_nw"])
    ```

    Custom decay using the `p` variable directly (Gaussian peaking at 400m within a 1200m cutoff):

    ```python
    nodes_gdf, numerical_gdf = layers.compute_stats(
        data_gdf=numerical_gdf,
        stats_column_labels=["mock_numerical_1"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[1200],
        decay_fn="exp(-((p - 0.333)^2) / (2 * 0.125^2))",  # Gaussian peaking at 400m
    )
    ```

    Using the `cityseer.decay` helper module for the same Gaussian curve:

    ```python
    from cityseer import decay

    nodes_gdf, numerical_gdf = layers.compute_stats(
        data_gdf=numerical_gdf,
        stats_column_labels=["mock_numerical_1"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[1200],
        decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
    )
    ```

    Flat (unweighted) metrics:

    ```python
    nodes_gdf, numerical_gdf = layers.compute_stats(
        data_gdf=numerical_gdf,
        stats_column_labels=["mock_numerical_1"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[800],
        decay_fn="1",
    )
    ```

    :::note
    The following stat types will be available for each `stats_key` for each of the
    computed distances:
    - `max` and `min`
    - `sum`
    - `mean`
    - `count`
    - `median`
    - `variance`
    - `mad` (median absolute deviation)

    The decay function (default exponential, or custom via `decay_fn`) controls how
    distance affects the weighting. Use `decay_fn="1"` for flat (unweighted) metrics.
    :::

    For a worked example, see the
    [Statistical Aggregations](https://cityseer.benchmarkurbanism.com/examples/stats/gpd-stats)
    recipe.

    """
    if angular:
        _require_dual_for_angular(network_structure, "compute_stats")
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
        stats_maps.append(dict(data_gdf[stats_column_label]))
    # stats
    decay_labels, decay_exprs, legacy_tail = _resolve_decay_fns(decay_fn)
    selected_measures = _resolve_stats_measures(measures)
    partial_func = partial(
        data_map.stats_decays,
        network_structure=network_structure,
        numerical_maps=stats_maps,
        decay_fns=decay_exprs,
        distances=distances,
        minutes=minutes,
        angular=angular,
        speed_m_s=speed_m_s,
        measures=measures,
    )
    # wraps progress bar (returns one StatsResult per decay label)
    stats_results = config.wrap_progress(
        total=network_structure.street_node_count(), rust_struct=data_map, partial_func=partial_func
    )
    # unpack
    distances = config.log_thresholds(
        distances=distances,
        minutes=minutes,
        speed_m_s=speed_m_s,
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(stats_results[0].node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    # unpack the numerical arrays (one result per decay label; empty label -> no suffix).
    # Only the selected measures are emitted; each maps to its `Stats` result accessor.
    for decay_label, stats_result in zip(decay_labels, stats_results, strict=True):
        for idx, stats_column_label in enumerate(stats_column_labels):
            stat_obj = stats_result.result[idx]
            for measure in selected_measures:
                measure_data = getattr(stat_obj, _STATS_ACCESSORS[measure])
                for dist_key in distances:
                    k = _layer_col_key(f"{stats_column_label}_{measure}", dist_key, angular, decay_label, legacy_tail)
                    temp_data[k] = measure_data[dist_key]

    temp_df = pd.DataFrame(temp_data, index=stats_results[0].node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]

    return nodes_gdf, data_gdf
