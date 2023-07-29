r"""
Cityseer networks module for calculating network centralities using optimised JIT compiled functions.

There are two network centrality methods available depending on whether you're using a node-based or segment-based
approach:

- [`node_centrality`](#node-centrality)
- [`segment_centrality`](#segment-centrality)

These methods wrap the underlying `numba` optimised functions for computing centralities, and provides access to
all of the underlying node-based or segment-based centrality methods. Multiple selected measures and distances are
computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar strategies.

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
- Note that `cityseer`'s implementation of simplest (angular) measures work on both primal (node or segment based)
and dual graphs (node only).
- Measures should only be directly compared on the same topology because different topologies can otherwise affect
the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed
on primal graphs because this does not account for the impact of differing topologies. Dual graph representations
can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a
four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual.
This effect is amplified for denser regions of the network.
- Segmentised versions of centrality measures should not be computed on dual graph topologies because street segment
lengths would be duplicated for each permutation of dual edge spanning street intersections. By way of example,
the contribution of a single edge segment at a four-way intersection would be duplicated three times.
- Global closeness is strongly discouraged because it does not behave suitably for localised graphs. Harmonic
closeness or improved closeness should be used instead. Note that Global closeness ($\frac{nodes}{farness}$) and
improved closeness ($\frac{nodes}{farness / nodes}$) can be recovered from the available metrics, if so desired,
through additional (manual) steps.
- Network decomposition can be a useful strategy when working at small distance thresholds, and confers advantages
such as more regularly spaced snapshots and fewer artefacts at small distance thresholds where street edges
intersect distance thresholds. However, the regular spacing of the decomposed segments will introduce spikes in the
distributions of node-based centrality measures when working at very small distance thresholds. Segmentised versions
may therefore be preferable when working at small thresholds on decomposed networks.
:::

"""

from __future__ import annotations

import logging
from typing import Any
import threading
import time
from queue import Queue

import geopandas as gpd
from tqdm import tqdm
from cityseer import config, rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# type hack until networkx supports type-hinting
MultiGraph = Any

# separate out so that ast parser can parse function def
MIN_THRESH_WT = config.MIN_THRESH_WT


# def _cast_beta(beta: cctypes.BetasType) -> npt.NDArray[np.float32]:
"""Type checks and casts beta parameter to a numpy array of beta."""

# def distance_from_beta(beta: cctypes.BetasType, min_threshold_wt: float = MIN_THRESH_WT) -> npt.NDArray[np.int_]:
r"""
Map decay parameters $\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.

:::note
It is generally not necessary to utilise this function directly.
:::

Parameters
----------
beta: float | ndarray[float]
    $\beta$ value/s to convert to distance thresholds $d_{max}$.
min_threshold_wt: float
    An optional cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$.

Returns
-------
betas: ndarray[float]
    A numpy array of distance thresholds $d_{max}$.

Examples
--------
```python
from cityseer.metrics import networks
# a list of betas
betas = [0.01, 0.02]
# convert to distance thresholds
d_max = networks.distance_from_beta(betas)
print(d_max)
# prints: array([400., 200.])
```

Weighted measures such as the gravity index, weighted betweenness, and weighted land-use accessibilities are
computed using a negative exponential decay function in the form of:

$$
weight = exp(-\beta \cdot distance)
$$

The strength of the decay is controlled by the $\beta$ parameter, which reflects a decreasing willingness to walk
correspondingly farther distances. For example, if $\beta=0.005$ were to represent a person's willingness to walk
to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at
13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity; thus, once a
sufficiently small weight is encountered it becomes computationally expensive to consider locations any farther
away. The minimum weight at which this cutoff occurs is represented by $w_{min}$, and the corresponding maximum
distance threshold by $d_{max}$.

![Example beta decays](/images/betas.png)

Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
the `betas` parameter, then this function will be called in order to extrapolate the distance thresholds implicitly,
using:

$$
d_{max} = \frac{log(w_{min})}{-\beta}
$$

The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking
thresholds, for example:

| $\beta$ | $d_{max}$ |
|:-------:|:---------:|
| 0.02 | 200m |
| 0.01 | 400m |
| 0.005 | 800m |
| 0.0025 | 1600m |

Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly, for example:

| $\beta$ | $w_{min}$ | $d_{max}$ |
|:-------:|:---------:|:---------:|
| 0.02 | 0.01 | 230m |
| 0.01 | 0.01 | 461m |
| 0.005 | 0.01 | 921m |
| 0.0025 | 0.01 | 1842m |

"""

# def _cast_distance(distance: cctypes.DistancesType) -> npt.NDArray[np.int_]:
"""Type checks and casts distance parameter to a numpy array of distance."""

# def beta_from_distance(distance: cctypes.DistancesType, min_threshold_wt: float = MIN_THRESH_WT,):
r"""
Map distance thresholds $d_{max}$ to equivalent decay parameters $\beta$ at the specified cutoff weight $w_{min}$.

See [`distance_from_beta`](#distance-from-beta) for additional discussion.

:::note
It is generally not necessary to utilise this function directly.
:::

Parameters
----------
distance: int | ndarray[int]
    $d_{max}$ value/s to convert to decay parameters $\beta$.
min_threshold_wt: float
    The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

Returns
-------
ndarray[float]
    A numpy array of decay parameters $\beta$.

Examples
--------
```python
from cityseer.metrics import networks
# a list of betas
distances = [400, 200]
# convert to betas
betas = networks.beta_from_distance(distances)
print(betas)  # prints: array([0.01, 0.02])
```

Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
the `distances` parameter, then this function will be called in order to extrapolate the decay parameters
implicitly, using:

$$
\beta = -\frac{log(w_{min})}{d_{max}}
$$

The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ parameters, for
example:

| $d_{max}$ | $\beta$ |
|:---------:|:-------:|
| 200m | 0.02 |
| 400m | 0.01 |
| 800m | 0.005 |
| 1600m | 0.0025 |

"""


# def avg_distance_for_beta(beta: cctypes.BetasType, min_threshold_wt: float = MIN_THRESH_WT) -> npt.NDArray[np.float32]:
r"""
Calculate the mean distance for a given $\beta$ parameter.

Parameters
----------
beta: float | ndarray[float]
    $\beta$ representing a spatial impedance / distance decay for which to compute the average walking distance.
min_threshold_wt: float
    The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

Returns
-------
ndarray[float]
    The average walking distance for a given $\beta$.

Examples
--------
```python
from cityseer.metrics import networks
import numpy as np

distances = np.array([100, 200, 400, 800, 1600])
print('distances', distances)
# distances [ 100  200  400  800 1600]

betas = networks.beta_from_distance(distances)
print('betas', betas)
# betas [0.04   0.02   0.01   0.005  0.0025]

print('avg', networks.avg_distance_for_beta(betas))
# avg [ 35.11949  70.23898 140.47797 280.95593 561.91187]
```

"""


# def pair_distances_betas(
r"""
Pair distances and betas, where one or the other parameter is provided.

Parameters
----------
distances: list[int] | tuple[int]
    Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
    (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
    then the `beta` parameter must be provided instead.
betas: float | ndarray[float]
    A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
    `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
    provided, then the `distance` parameter must be provided instead.
min_threshold_wt: float
    The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
    `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.

Returns
-------
distances: int | ndarray[int]
    Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
    (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
    then the `beta` parameter must be provided instead.
betas: float | ndarray[float]
    A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
    `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
    provided, then the `distance` parameter must be provided instead.

Examples
--------
:::warning
Networks should be buffered according to the largest distance threshold that will be used for analysis. This
protects nodes near network boundaries from edge falloffs. Nodes outside the area of interest but within these
buffered extents should be set to 'dead' so that centralities or other forms of measures are not calculated.
Whereas metrics are not calculated for 'dead' nodes, they can still be traversed by network analysis algorithms
when calculating shortest paths and landuse accessibilities.
:::

"""


# def clip_weights_curve(
r"""
Calculate the upper bounds for clipping weights produced by spatial impedance functions.

Determine the upper weights threshold of the distance decay curve for a given $\beta$ based on the
`spatial_tolerance` parameter. This is used by downstream functions to determine the upper extent at which weights
derived for spatial impedance functions are flattened and normalised. This functionality is only intended for
situations where the location of datapoints is uncertain for a given spatial tolerance.

:::warning
Use distance based clipping with caution for smaller distance thresholds. For example, if using a 200m distance
threshold clipped by 100m, then substantial distortion is introduced by the process of clipping and normalising the
distance decay curve. More generally, smaller distance thresholds should generally be avoided for situations where
datapoints are not located with high spatial precision.
:::

Parameters
----------
distances: ndarray[int]
    An array of distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
betas: ndarray[float32]
    An array of $\beta$ to be used for the exponential decay function for weighted metrics.
spatial_tolerance: int
    The spatial buffer distance corresponding to the tolerance for spatial inaccuracy.

Returns
-------
max_curve_wts: ndarray[float]
    An array of maximum weights at which curves for corresponding $\beta$ will be clipped.

"""


# provides access to the underlying centrality.local_centrality method
def node_centrality_shortest(
    network_structure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    jitter_scale: float = 0.0,
) -> gpd.GeoDataFrame:
    r"""
    Compute node-based network centrality.

    Parameters
    ----------
    measures: tuple[str]
        A tuple of centrality measures to compute. Centrality keys can be selected from the available centrality
        measure `key` values in the table beneath. Each centrality measure will be computed for all distance
        thresholds $d_{max}$.
    network_structure
        A [`structures.NetworkStructure`](/structures#networkstructure). Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) method.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) method. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the method.
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: float | ndarray[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.

    Examples
    --------
    The following keys use the shortest-path heuristic, and are available when the `angular` parameter is set to the
    default value of `False`:

    | key                   | formula | notes |
    | ----------------------| :------:| ----- |
    | node_density          | $$\sum_{j\neq{i}}^{n}1$$ | A summation of nodes. |
    | node_farness          | $$\sum_{j\neq{i}}^{n}d_{(i,j)}$$ | A summation of distances in metres. |
    | node_cycles           | $$\sum_{j\neq{i}j=cycle}^{n}1$$ | A summation of network cycles. |
    | node_harmonic         | $$\sum_{j\neq{i}}^{n}\frac{1}{Z_{(i,j)}}$$ | Harmonic closeness is an appropriate form
    of closeness centrality for localised implementations constrained by the threshold $d_{max}$. |
    | node_beta             | $$\sum_{j\neq{i}}^{n}\exp(-\beta\cdot d[i,j])$$ | Also known as the gravity index.
    This is a spatial impedance metric differentiated from other closeness centralities by the use of an
    explicit $\beta$ parameter, which can be used to model the decay in walking tolerance as distances
    increase. |
    | node_betweenness      | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | Betweenness centrality summing all
    shortest-paths traversing each node $i$. |
    | node_betweenness_beta | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}\exp(-\beta\cdot d[j,k])$$ | Applies a
    spatial impedance decay function to betweenness centrality. $d$ represents the full distance from
    any $j$ to $k$ node pair passing through node $i$. |

    The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular`
    parameter is explicitly set to `True`:

    | key                      | formula | notes |
    | ------------------------ | :-----: | ----- |
    | node_harmonic_angular    | $$\sum_{j\neq{i}}^{n}\frac{1}{Z_{(i,j)}}$$ | The simplest-path implementation of
    harmonic closeness uses angular-distances for the impedance parameter. Angular-distances are normalised by 180 and
    added to 1 to avoid division by zero: ${Z = 1 + (angularchange/180)}$. |
    | node_betweenness_angular | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | The simplest-path version of
    betweenness centrality. This is distinguished from the shortest-path version by use of a simplest-path heuristic
    (shortest angular distance). |

    """
    if distances is None:
        distances = rustalgos.distances_from_betas(betas, min_threshold_wt=min_threshold_wt)

    def wrapper(queue):
        result = network_structure.local_node_centrality_shortest(
            distances=distances,
            compute_closeness=compute_closeness,
            compute_betweenness=compute_betweenness,
            min_threshold_wt=min_threshold_wt,
            jitter_scale=jitter_scale,
        )
        queue.put(result)

    # start and track progress
    result_queue = Queue()
    thread = threading.Thread(target=wrapper, args=(result_queue,))
    pbar = tqdm(total=network_structure.node_count())
    thread.start()
    while thread.is_alive():
        time.sleep(1)
        pbar.update(network_structure.progress() - pbar.n)
    pbar.update(network_structure.node_count() - pbar.n)
    pbar.close()
    node_result_short = result_queue.get()
    thread.join()
    # unpack
    if compute_closeness is True:
        for measure_name in ["node_beta", "node_cycles", "node_density", "node_farness", "node_harmonic"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(node_result_short, measure_name)[distance]
    if compute_betweenness is True:
        for measure_name in ["node_betweenness", "node_betweenness_beta"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(node_result_short, measure_name)[distance]
    return nodes_gdf


def node_centrality_simplest(
    network_structure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    min_threshold_wt: float = MIN_THRESH_WT,
    jitter_scale: float = 0.0,
) -> gpd.GeoDataFrame:
    """ """
    if distances is None:
        distances = rustalgos.distances_from_betas(betas, min_threshold_wt=min_threshold_wt)
    node_result = network_structure.local_node_centrality_simplest(
        distances=distances,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        jitter_scale=jitter_scale,
    )
    if compute_closeness is True:
        for measure_name in ["node_harmonic"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(node_result, measure_name)[distance]
    if compute_betweenness is True:
        for measure_name in ["node_betweenness"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(node_result, measure_name)[distance]
    return nodes_gdf


# provides access to the underlying centrality.local_centrality method
def segment_centrality(
    network_structure: rustalgos.NetworkStructure,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    compute_closeness: bool | None = True,
    compute_betweenness: bool | None = True,
    jitter_scale: float = 0.0,
    min_threshold_wt: float = MIN_THRESH_WT,
) -> gpd.GeoDataFrame:
    r"""
    Compute segment-based network centrality.

    Parameters
    ----------
    measures: tuple[str]
        A tuple of centrality measures to compute. Centrality keys can be selected from the available centrality
        measure `key` values in the table beneath. Each centrality measure will be computed for all distance
        thresholds $d_{max}$.
    network_structure
        A [`structures.NetworkStructure`](/structures#networkstructure). Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) method.
    nodes_gdf
        A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe)
        representing nodes. Best generated with the
        [`graphs.network_structure_from_nx`](/tools/graphs#network-structure-from-nx) method. The outputs of
        calculations will be written to this `GeoDataFrame`, which is then returned from the method.
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: float | ndarray[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    jitter_scale: float
        The scale of random jitter to add to shortest path calculations, useful for situations with highly
        rectilinear grids. `jitter_scale` is passed to the `scale` parameter of `np.random.normal`.
    angular: bool
        Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
        and distances.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.

    Returns
    -------
    nodes_gdf: GeoDataFrame
        The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.

    Examples
    --------
    The following keys use the shortest-path heuristic, and are available when the `angular` parameter is set to the
    default value of `False`:

    | key                 | formula | notes |
    | ------------------- | :-----: |------ |
    | segment_density     | $$\sum_{(a, b)}^{edges}d_{b} - d_{a}$$ | A summation of edge lengths. |
    | segment_harmonic    | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\ln(b) -\ln(a)$$ | A continuous form of
    harmonic closeness centrality applied to edge lengths. |
    | segment_beta        | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\frac{\exp(-\beta\cdot b) -\exp(-\beta\cdot a)}{-\beta}$$ | A  # pylint: disable=line-too-long
    continuous form of beta-weighted (gravity index) centrality applied to edge lengths. |
    | segment_betweenness | | A continuous form of betweenness: Resembles `segment_beta` applied to edges situated
    on shortest paths between all nodes $j$ and $k$ passing through $i$. |

    The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular`
    parameter is explicitly set to `True`.

    | key                       | formula | notes |
    | ------------------------- | :-----: | ----- |
    | segment_harmonic_hybrid   | $$\sum_{(a, b)}^{edges}\frac{d_{b} - d_{a}}{Z}$$ | Weights angular
    harmonic centrality by the lengths of the edges. See `node_harmonic_angular`. |
    | segment_betweeness_hybrid | | A continuous form of angular betweenness: Resembles `segment_harmonic_hybrid`
    applied to edges situated on shortest paths between all nodes $j$ and $k$ passing through $i$. |

    """
    if distances is None:
        distances = rustalgos.distances_from_betas(betas, min_threshold_wt=min_threshold_wt)
    segment_result = network_structure.local_segment_centrality(
        distances=distances,
        compute_closeness=compute_closeness,
        compute_betweenness=compute_betweenness,
        min_threshold_wt=min_threshold_wt,
        jitter_scale=jitter_scale,
    )
    if compute_closeness is True:
        for measure_name in ["segment_density", "segment_harmonic", "segment_beta"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(segment_result, measure_name)[distance]
    if compute_betweenness is True:
        for measure_name in ["segment_betweenness"]:
            for distance in distances:
                data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                nodes_gdf[data_key] = getattr(segment_result, measure_name)[distance]
    return nodes_gdf
