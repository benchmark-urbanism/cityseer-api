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
from functools import partial

import geopandas as gpd
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
    jitter_scale: float = 0.0,
    sample_probability: float | None = None,
    sampling_weights: list[float] | None = None,
    random_seed: int | None = None,
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
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.
    sample_probability: float
        Probability of sampling a node as a source for centrality calculations. When used alone, provides uniform
        random sampling. When combined with `sampling_weights`, the final probability for each node is
        `sample_probability * sampling_weights[node_idx]`.
    sampling_weights: list[float]
        Optional array of per-node sampling weights. Must have length equal to the number of nodes, with values
        in the range [0.0, 1.0]. Use this to bias sampling toward certain nodes (e.g., by normalized population).
        When provided, the sampling probability for each node becomes `sample_probability * sampling_weights[node_idx]`.
    random_seed: int
        Optional seed for deterministic sampling and random cost jitter.

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
    logger.info("Computing shortest path node centrality.")
    # wrap the main function call for passing to the progress wrapper
    partial_func = partial(
        network_structure.local_node_centrality_shortest,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
        sample_probability=sample_probability,
        sampling_weights=sampling_weights,
        random_seed=random_seed,
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
    config.log_sampling(
        sample_probability=sample_probability,
        distances=distances,
        reachability_totals=result.reachability_totals,  # type: ignore
        sampled_source_count=result.sampled_source_count,  # type: ignore
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    # set the index to the gdf index
    if compute_closeness is True:
        for measure_key, attr_key in [
            ("beta", "node_beta"),
            ("cycles", "node_cycles"),
            ("density", "node_density"),
            ("farness", "node_farness"),
            ("harmonic", "node_harmonic"),
        ]:
            for distance in distances:
                data_key = config.prep_gdf_key(measure_key, distance)
                temp_data[data_key] = getattr(result, attr_key)[distance]
        for distance in distances:
            data_key = config.prep_gdf_key("hillier", distance)
            # existing columns
            temp_data[data_key] = result.node_density[distance] ** 2 / result.node_farness[distance]  # type: ignore
    if compute_betweenness is True:
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
    jitter_scale: float = 0.0,
    sample_probability: float | None = None,
    sampling_weights: list[float] | None = None,
    random_seed: int | None = None,
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
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.
    sample_probability: float
        Probability of sampling a node as a source for centrality calculations. When used alone, provides uniform
        random sampling. When combined with `sampling_weights`, the final probability for each node is
        `sample_probability * sampling_weights[node_idx]`.
    sampling_weights: list[float]
        Optional array of per-node sampling weights. Must have length equal to the number of nodes, with values
        in the range [0.0, 1.0]. Use this to bias sampling toward certain nodes (e.g., by normalized population).
        When provided, the sampling probability for each node becomes `sample_probability * sampling_weights[node_idx]`.
    random_seed: int
        Optional seed for deterministic sampling and random cost jitter.

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
    logger.info("Computing simplest path node centrality.")
    partial_func = partial(
        network_structure.local_node_centrality_simplest,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        angular_scaling_unit=angular_scaling_unit,
        farness_scaling_offset=farness_scaling_offset,
        jitter_scale=jitter_scale,
        sample_probability=sample_probability,
        sampling_weights=sampling_weights,
        random_seed=random_seed,
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
    config.log_sampling(
        sample_probability=sample_probability,
        distances=distances,
        reachability_totals=result.reachability_totals,  # type: ignore
        sampled_source_count=result.sampled_source_count,  # type: ignore
    )
    # intersect computed keys with those available in the gdf index (stations vs. streets)
    gdf_idx = nodes_gdf.index.intersection(result.node_keys_py)
    # create a dictionary to hold the data
    temp_data = {}
    if compute_closeness is True:
        for distance in distances:
            data_key = config.prep_gdf_key("density", distance, angular=True)
            temp_data[data_key] = result.node_density[distance]  # type: ignore
        for distance in distances:
            data_key = config.prep_gdf_key("harmonic", distance, angular=True)
            temp_data[data_key] = result.node_harmonic[distance]  # type: ignore
        for distance in distances:
            data_key = config.prep_gdf_key("hillier", distance, angular=True)
            temp_data[data_key] = (
                result.node_density[distance] ** 2 / result.node_farness[distance]  # type: ignore
            )
        for distance in distances:
            data_key = config.prep_gdf_key("farness", distance, angular=True)
            temp_data[data_key] = result.node_farness[distance]  # type: ignore
    if compute_betweenness is True:
        for distance in distances:
            data_key = config.prep_gdf_key("betweenness", distance, angular=True)
            temp_data[data_key] = result.node_betweenness[distance]  # type: ignore

    temp_df = pd.DataFrame(temp_data, index=result.node_keys_py)
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
    jitter_scale: float = 0.0,
    random_seed: int | None = None,
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
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a
        range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the
        shortest path calculations to provide random variation to the paths traced through the network. When working
        with shortest paths in metres, the random value represents distance in metres. When using a simplest path
        heuristic, the jitter will represent angular change in degrees.
    random_seed: int
        Optional seed for random cost jitter.

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
        network_structure.local_segment_centrality,
        distances=distances,
        betas=betas,
        minutes=minutes,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
        random_seed=random_seed,
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
# Adaptive Sampling Functions
# =============================================================================
# These functions run centrality with per-distance adaptive sampling, where
# sampling probability is calibrated separately for each distance threshold.
# This allows aggressive sampling at large distances (high reach) while
# maintaining accuracy at short distances (low reach).


def _run_adaptive_centrality(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int],
    target_rho: float,
    centrality_func: str,  # "shortest" or "simplest"
    compute_closeness: bool,
    compute_betweenness: bool,
    min_threshold_wt: float,
    speed_m_s: float,
    jitter_scale: float,
    random_seed: int | None,
    n_probes: int,
    # simplest-only params
    angular_scaling_unit: float | None = None,
    farness_scaling_offset: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Internal: Run centrality with per-distance adaptive sampling.

    This function handles the shared logic for both shortest and simplest
    path adaptive centrality computation.
    """
    # Determine which metric model to use for sampling calibration
    # Use the more conservative model when computing both metrics
    if compute_closeness and compute_betweenness:
        metric = "both"
    elif compute_betweenness:
        metric = "betweenness"
    else:
        metric = "harmonic"

    # 1. Probe reachability
    logger.info(f"Probing reachability ({n_probes} samples)...")
    reach_estimates = config.probe_reachability(network_structure, distances, n_probes=n_probes, speed_m_s=speed_m_s)

    # 2. Compute sampling probabilities using appropriate metric model
    sample_probs = config.compute_sample_probs_for_target_rho(reach_estimates, target_rho, metric=metric)

    # 3. Log the plan
    config.log_adaptive_sampling_plan(distances, reach_estimates, sample_probs, target_rho, metric=metric)

    # 4. Run per-distance
    logger.info("Running per-distance centrality...")

    # Track all results to merge
    all_results: dict[
        int, rustalgos.centrality.CentralityShortestResult | rustalgos.centrality.CentralitySimplestResult
    ] = {}

    for d in sorted(distances):
        p = sample_probs.get(d)
        # Use None (full computation) if p >= 1.0 or None
        effective_p = p if (p is not None and p < 1.0) else None

        logger.info(f"  {d}m: {'full' if effective_p is None else f'p={effective_p:.0%}'}...")

        if centrality_func == "shortest":
            result = network_structure.local_node_centrality_shortest(
                distances=[d],
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                jitter_scale=jitter_scale,
                sample_probability=effective_p,
                random_seed=random_seed,
                pbar_disabled=True,  # Disable per-distance progress bars
            )
        else:  # simplest
            result = network_structure.local_node_centrality_simplest(
                distances=[d],
                compute_closeness=compute_closeness,
                compute_betweenness=compute_betweenness,
                min_threshold_wt=min_threshold_wt,
                speed_m_s=speed_m_s,
                angular_scaling_unit=angular_scaling_unit,
                farness_scaling_offset=farness_scaling_offset,
                jitter_scale=jitter_scale,
                sample_probability=effective_p,
                random_seed=random_seed,
                pbar_disabled=True,
            )

        all_results[d] = result

        # Log actual accuracy achieved
        if effective_p is not None and result.sampled_source_count > 0:
            total_reach = result.reachability_totals[0] if result.reachability_totals else 0
            mean_reach = total_reach / result.sampled_source_count
            eff_n = mean_reach * effective_p
            exp_rho, _ = config.get_expected_spearman(eff_n, metric=metric)
            logger.info(f"    actual: reach={mean_reach:.0f}, eff_n={eff_n:.0f}, expected ρ={exp_rho:.2f}")

    # 5. Merge results into GeoDataFrame
    # Get reference result for node keys
    ref_result = next(iter(all_results.values()))
    gdf_idx = nodes_gdf.index.intersection(ref_result.node_keys_py)

    temp_data: dict[str, object] = {}

    if centrality_func == "shortest":
        if compute_closeness:
            for measure_key, attr_key in [
                ("beta", "node_beta"),
                ("cycles", "node_cycles"),
                ("density", "node_density"),
                ("farness", "node_farness"),
                ("harmonic", "node_harmonic"),
            ]:
                for d, res in all_results.items():
                    data_key = config.prep_gdf_key(measure_key, d)
                    temp_data[data_key] = getattr(res, attr_key)[d]
            for d, res in all_results.items():
                data_key = config.prep_gdf_key("hillier", d)
                temp_data[data_key] = res.node_density[d] ** 2 / res.node_farness[d]
        if compute_betweenness:
            for measure_key, attr_key in [
                ("betweenness", "node_betweenness"),
                ("betweenness_beta", "node_betweenness_beta"),
            ]:
                for d, res in all_results.items():
                    data_key = config.prep_gdf_key(measure_key, d)
                    temp_data[data_key] = getattr(res, attr_key)[d]
    else:  # simplest
        if compute_closeness:
            for d, res in all_results.items():
                temp_data[config.prep_gdf_key("density", d, angular=True)] = res.node_density[d]
                temp_data[config.prep_gdf_key("harmonic", d, angular=True)] = res.node_harmonic[d]
                temp_data[config.prep_gdf_key("farness", d, angular=True)] = res.node_farness[d]
                temp_data[config.prep_gdf_key("hillier", d, angular=True)] = (
                    res.node_density[d] ** 2 / res.node_farness[d]
                )
        if compute_betweenness:
            for d, res in all_results.items():
                temp_data[config.prep_gdf_key("betweenness", d, angular=True)] = res.node_betweenness[d]

    temp_df = pd.DataFrame(temp_data, index=ref_result.node_keys_py)
    nodes_gdf.loc[gdf_idx, temp_df.columns] = temp_df.loc[gdf_idx, temp_df.columns]

    logger.info("Adaptive centrality complete.")
    return nodes_gdf


def node_centrality_shortest_adaptive(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int],
    target_rho: float = 0.95,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    jitter_scale: float = 0.0,
    random_seed: int | None = None,
    n_probes: int = 50,
) -> gpd.GeoDataFrame:
    """
    Compute shortest-path node centrality with per-distance adaptive sampling.

    This function automatically calibrates sampling probability for each distance
    threshold to achieve a target accuracy level (Spearman ρ). Short distances
    use full or near-full computation (where reach is low), while long distances
    use aggressive sampling (where high reach provides statistical power).

    This can provide substantial speedups for analyses spanning multiple scales
    (e.g., 500m to 20km) while maintaining consistent accuracy across all distances.

    Parameters
    ----------
    network_structure
        A NetworkStructure. Best generated with io.network_structure_from_nx.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances
        Distance thresholds in metres. Unlike the standard function, only distances
        (not betas or minutes) are supported for adaptive sampling.
    target_rho
        Target Spearman ρ correlation for ranking accuracy. Default 0.95.
        Higher values (e.g., 0.97) provide better accuracy but less speedup.
        Note: Model was fitted on closeness; betweenness may have higher variance.
    compute_closeness
        Compute closeness centralities. True by default.
    compute_betweenness
        Compute betweenness centralities. True by default.
    min_threshold_wt
        Minimum threshold weight for beta computation.
    speed_m_s
        Walking speed in m/s for distance-to-time conversion.
    jitter_scale
        Scale of random jitter for path calculations.
    random_seed
        Optional seed for reproducible sampling.
    n_probes
        Number of nodes to probe for reachability estimation. Default 50.

    Returns
    -------
    nodes_gdf
        The input GeoDataFrame with centrality columns added.

    See Also
    --------
    node_centrality_shortest : Standard (non-adaptive) version with uniform sampling.

    Examples
    --------
    ```python
    # Compute centrality across scales with automatic sampling
    nodes_gdf = node_centrality_shortest_adaptive(
        network_structure,
        nodes_gdf,
        distances=[500, 2000, 5000, 20000],
        target_rho=0.95,
    )
    ```
    """
    logger.info("Computing adaptive shortest path node centrality.")
    return _run_adaptive_centrality(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        target_rho=target_rho,
        centrality_func="shortest",
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
        random_seed=random_seed,
        n_probes=n_probes,
    )


def node_centrality_simplest_adaptive(
    network_structure: rustalgos.graph.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int],
    target_rho: float = 0.95,
    compute_closeness: bool = True,
    compute_betweenness: bool = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
    angular_scaling_unit: float = 90,
    farness_scaling_offset: float = 1,
    jitter_scale: float = 0.0,
    random_seed: int | None = None,
    n_probes: int = 50,
) -> gpd.GeoDataFrame:
    """
    Compute simplest-path (angular) node centrality with per-distance adaptive sampling.

    This function automatically calibrates sampling probability for each distance
    threshold to achieve a target accuracy level (Spearman ρ). Short distances
    use full or near-full computation (where reach is low), while long distances
    use aggressive sampling (where high reach provides statistical power).

    Parameters
    ----------
    network_structure
        A NetworkStructure. Best generated with io.network_structure_from_nx.
    nodes_gdf
        A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.
    distances
        Distance thresholds in metres.
    target_rho
        Target Spearman ρ correlation for ranking accuracy. Default 0.95.
    compute_closeness
        Compute closeness centralities. True by default.
    compute_betweenness
        Compute betweenness centralities. True by default.
    min_threshold_wt
        Minimum threshold weight for beta computation.
    speed_m_s
        Walking speed in m/s for distance-to-time conversion.
    angular_scaling_unit
        Scaling unit for angular distance. Default 90 degrees.
    farness_scaling_offset
        Offset for farness calculation. Default 1.
    jitter_scale
        Scale of random jitter for path calculations.
    random_seed
        Optional seed for reproducible sampling.
    n_probes
        Number of nodes to probe for reachability estimation. Default 50.

    Returns
    -------
    nodes_gdf
        The input GeoDataFrame with centrality columns added.

    See Also
    --------
    node_centrality_simplest : Standard (non-adaptive) version with uniform sampling.
    """
    logger.info("Computing adaptive simplest path node centrality.")
    return _run_adaptive_centrality(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=distances,
        target_rho=target_rho,
        centrality_func="simplest",
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        speed_m_s=speed_m_s,
        jitter_scale=jitter_scale,
        random_seed=random_seed,
        n_probes=n_probes,
        angular_scaling_unit=angular_scaling_unit,
        farness_scaling_offset=farness_scaling_offset,
    )
