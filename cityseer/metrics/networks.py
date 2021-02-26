'''
Centrality methods
'''
import logging
from typing import Union

import networkx as nx
import numpy as np
from numba.typed import Dict

from cityseer.algos import centrality, checks
from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def distance_from_beta(beta: Union[float, list, np.ndarray],
                       min_threshold_wt: float = checks.def_min_thresh_wt) -> np.ndarray:
    """
    Maps decay parameters $\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.

    ::: warning Note
    It is generally not necessary to utilise this function directly. It will be called internally, if necessary, when invoking [NetworkLayer](#network-layer) or [NetworkLayerFromNX](#network-layer-from-nx).
    :::

    Parameters
    ----------
    beta
        $\beta$ value/s to convert to distance thresholds $d_{max}$.
    min_threshold_wt
        An optional cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$, default of 0.01831563888873418.

    Returns
    -------
    np.ndarray
        A numpy array of distance thresholds $d_{max}$.

    Notes
    -----

    ```python
    from cityseer.metrics import networks
    # a list of betas
    betas = [-0.01, -0.02]
    # convert to distance thresholds
    d_max = networks.distance_from_beta(betas)
    print(d_max)
    # prints: array([400., 200.])
    ```

    Weighted measures such as the gravity index, weighted betweenness, and weighted land-use accessibilities are computed using a negative exponential decay function in the form of:

    $$weight = exp(\beta \cdot distance)$$

    The strength of the decay is controlled by the $\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances. For example, if $\beta=-0.005$ were to represent a person's willingness to walk to a bus stop, then a location $100m$ distant would be weighted at $60\%$ and a location $400m$ away would be weighted at $13.5\%$. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity; thus, once a sufficiently small weight is encountered it becomes computationally expensive to consider locations any farther away. The minimum weight at which this cutoff occurs is represented by $w_{min}$, and the corresponding maximum distance threshold by $d_{max}$.

    ![Example beta decays](../.vitepress/plots/images/betas.png)
    
    [NetworkLayer](#network-layer) and [NetworkLayerFromNX](/metrics/networks.html#network-layer-from-nx) can be invoked with either `distances` or `betas` parameters, but not both. If using the `betas` parameter, then this function will be called in order to extrapolate the distance thresholds implicitly, using:

    $$d_{max} = \frac{log\Big(w_{min}\Big)}{\beta}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking thresholds, for example:

    | $\beta$ | $d_{max}$ |
    |-----------|---------|
    | $-0.02$ | $200m$ |
    | $-0.01$ | $400m$ |
    | $-0.005$ | $800m$ |
    | $-0.0025$ | $1600m$ |

    Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly, for example:

    | $\beta$ | $w_{min}$ | $d_{max}$ |
    |----------|----------|----------|
    | $-0.02$ | $0.01$ | $230m$ |
    | $-0.01$ | $0.01$ | $461m$ |
    | $-0.005$ | $0.01$ | $921m$ |
    | $-0.0025$ | $0.01$ | $1842m$ |

    """
    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]
    if not isinstance(beta, (list, tuple, np.ndarray)):
        raise TypeError('Please provide a beta or a list, tuple, or numpy.ndarray of betas.')
    # check that the betas do not have leading negatives
    for b in beta:
        if b > 0:
            raise ValueError('Please provide the beta value with the leading negative.')
        elif b == 0:
            # ints have no concept of -0, so catch betas that are positive 0 or -0 (in int form)
            # i.e. betas of -0.0 will successfully result in np.inf as opposed to -np.inf
            if np.log(min_threshold_wt) / b == -np.inf:
                raise ValueError('Please provide zeros in float form with a leading negative.')
    # cast to numpy
    beta = np.array(beta)
    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / beta


def beta_from_distance(distance: Union[float, list, np.ndarray],
                       min_threshold_wt: float = checks.def_min_thresh_wt) -> np.ndarray:
    """

    Maps distance thresholds $d_{max}$ to equivalent decay parameters $\beta$ at the specified cutoff weight $w_{min}$. See [distance_from_beta](#distance-from-beta) for additional discussion.

    ::: warning Note
    It is generally not necessary to utilise this function directly. It will be called internally, if necessary, when invoking [NetworkLayer](#network-layer) or [NetworkLayerFromNX](#network-layer-from-nx).
    :::

    Parameters
    ----------
    distance
        $d_{max}$ value/s to convert to decay parameters $\beta$.
    min_threshold_wt
        The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$, default of 0.01831563888873418.

    Returns
    -------
    np.ndarray
        A numpy array of decay parameters $\beta$.

    Notes
    -----
    
    ```python
    from cityseer.metrics import networks
    # a list of betas
    distances = [400, 200]
    # convert to betas
    betas = networks.beta_from_distance(distances)
    print(betas)  # prints: array([-0.01, -0.02])
    ```

    [NetworkLayer](#network-layer) and [NetworkLayerFromNX](#network-layer-from-nx) can be invoked with either `distances` or `betas` parameters, but not both. If using the `distances` parameter, then this function will be called in order to extrapolate the decay parameters implicitly, using:

    $$\beta = \frac{log\Big(w_{min}\Big)}{d_{max}}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ parameters, for example:

    | $d_{max}$ | $\beta$ |
    |-----------|---------|
    | $200m$ | $-0.02$ |
    | $400m$ | $-0.01$ |
    | $800m$ | $-0.005$ |
    | $1600m$ | $-0.0025$ |

    """
    # cast to list form
    if isinstance(distance, (int, float)):
        distance = [distance]
    if not isinstance(distance, (list, tuple, np.ndarray)):
        raise TypeError('Please provide a distance or a list, tuple, or numpy.ndarray of distances.')
    # check that the betas do not have leading negatives
    for d in distance:
        if d <= 0:
            raise ValueError('Please provide a positive distance value.')
    # cast to numpy
    distance = np.array(distance)
    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / distance


# TODO: sub-class network layer for nd; nd-ang; seg; seg-ang
# do all setup logic here in class and keep backend lean and modular

class NetworkLayer:
    """
    Network layers are used for network centrality computations and provide the backbone for landuse and statistical aggregations. [`NetworkLayerFromNX`](#network-layer-from-nx) should be used instead if converting from a `NetworkX`
    graph to a `NetworkLayer`.

    A `NetworkLayer` requires either a set of distances $d_{max}$ or equivalent exponential decay parameters
    $\beta$, but not both. The unprovided parameter will be calculated implicitly in order to keep weighted and
    unweighted metrics in lockstep. The `min_threshold_wt` parameter can be used to generate custom mappings from
    one to the other: see [distance_from_beta](#distance-from-beta) for more information. These distances and betas
    are used for any subsequent centrality and land-use calculations.

    ```python
    from cityseer.metrics import networks
    from cityseer.tools import mock, graphs

    # prepare a mock graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)

    # if initialised with distances: 
    # betas for weighted metrics will be generated implicitly
    N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
    print(N.distances)  # prints: [200, 400, 800, 1600]
    print(N.betas)  # prints: [-0.02, -0.01, -0.005, -0.0025]

    # if initialised with betas: 
    # distances for non-weighted metrics will be generated implicitly
    N = networks.NetworkLayerFromNX(G, betas=[-0.02, -0.01, -0.005, -0.0025])
    print(N.distances)  # prints: [200, 400, 800, 1600]
    print(N.betas)  # prints: [-0.02, -0.01, -0.005, -0.0025]
    ```
    """

    def __init__(self,
                 node_uids: Union[list, tuple],
                 node_data: np.ndarray,
                 edge_data: np.ndarray,
                 node_edge_map: Dict,
                 distances: Union[list, tuple, np.ndarray] = None,
                 betas: Union[list, tuple, np.ndarray] = None,
                 min_threshold_wt: float = checks.def_min_thresh_wt):
        """
        Parameters
        ----------
        node_uids
            A `list` or `tuple` of node identifiers corresponding to each node. This list must be in the same order and
            of the same length as the `node_data`.
        node_data
            A 2d `numpy` array representing the graph's nodes. The indices of the second dimension correspond as
            follows:

            | idx | property |
            |-----|:---------|
            | 0 | `x` coordinate |
            | 1 | `y` coordinate |
            | 2 | `bool` describing whether the node is `live`. Metrics are only computed for `live` nodes. |

            The `x` and `y` node attributes determine the spatial coordinates of the node, and should be in a suitable
            projected (flat) coordinate reference system in metres. [`nX_wgs_to_utm`](/util/graphs.html#nx-wgs-to-utm)
            can be used for converting a `networkX` graph from WGS84 `lng`, `lat` geographic coordinates to the local
            UTM `x`, `y` projected coordinate system.

            When calculating local network centralities or land-use accessibilities, it is best-practice to buffer the
            network by a distance equal to the maximum distance threshold to be considered. This prevents problematic
            results arising due to boundary roll-off effects.
            
            The `live` node attribute identifies nodes falling within the areal boundary of interest as opposed to those
            that fall within the surrounding buffered area. Calculations are only performed for `live=True` nodes, thus
            reducing frivolous computation while also cleanly identifying which nodes are in the buffered roll-off area.
            If some other process will be used for filtering the nodes, or if boundary roll-off is not being considered,
            then set all nodes to `live=True`.
        edge_data
            A 2d `numpy` array representing the graph's edges. Each edge will be described separately for each direction
            of travel. The indices of the second dimension correspond as follows:
            | idx | property |
            |-----|:---------|
            | 0 | start node `idx` |
            | 1 | end node `idx` |
            | 2 | the segment length in metres |
            | 3 | the sum of segment's angular change |
            | 4 | an 'impedance factor' which can be applied to magnify or reduce the effect of the edge's impedance on
            shortest-path calculations. e.g. for gradients or other such considerations. Use with caution. |
            | 5 | the edge's entry angular bearing |
            | 6 | the edge's exit angular bearing |
        node_edge_map
            A `numba` `Dict` with `node_data` indices as keys and `numba` `List` types as values containing the out-edge
            indices for each node.
        distances
            A distance, or `list`, `tuple`, or `numpy` array of distances corresponding to the local $d_{max}$
            thresholds to be used for centrality (and land-use) calculations. The $\beta$ parameters (for
            distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then
            the `beta` parameter must be provided instead. Use a distance of `np.inf` where no distance threshold should
            be enforced.
        betas
            A $\beta$, or `list`, `tuple`, or `numpy` array of $\beta$ to be used for the exponential decay function for
            weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the
            `betas` parameter is not provided, then the `distance` parameter must be provided instead.
        min_threshold_wt
            The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
            `distance` and `beta` parameters. See [distance_from_beta](#distance-from-beta) for more information.

        Returns
        -------
        NetworkLayer
            A `NetworkLayer`.

        Properties
        ----------
        """
        self._uids = node_uids
        self._node_data = node_data
        self._edge_data = edge_data
        self._node_edge_map = node_edge_map
        self._distances = distances
        self._betas = betas
        self._min_threshold_wt = min_threshold_wt
        self.metrics = {
            'centrality': {},
            'mixed_uses': {},
            'accessibility': {
                'non_weighted': {},
                'weighted': {}
            },
            'stats': {},
            'models': {}
        }
        # for storing originating networkX graph
        self._networkX_multigraph = None
        # check the data structures
        if len(self._uids) != len(self._node_data):
            raise ValueError('The number of indices does not match the number of nodes.')
        # check network maps
        checks.check_network_maps(self._node_data, self._edge_data, self._node_edge_map)
        # if distances, check the types and generate the betas
        if self._distances is not None and self._betas is None:
            if isinstance(self._distances, (int, float)):
                self._distances = [self._distances]
            if isinstance(self._distances, (list, tuple, np.ndarray)):
                if len(self._distances) == 0:
                    raise ValueError('Please provide at least one distance.')
            else:
                raise TypeError('Please provide a distance, or a list, tuple, or numpy.ndarray of distances.')
            # generate the betas
            self._betas = beta_from_distance(self._distances,
                                             min_threshold_wt=self._min_threshold_wt)
        # if betas, generate the distances
        elif self._betas is not None and self._distances is None:
            if isinstance(self._betas, (float)):
                self._betas = [self._betas]
            if isinstance(self._betas, (list, tuple, np.ndarray)):
                if len(self._betas) == 0:
                    raise ValueError('Please provide at least one beta.')
            else:
                raise TypeError('Please provide a beta, or a list, tuple, or numpy.ndarray of betas.')
            self._distances = distance_from_beta(self._betas,
                                                 min_threshold_wt=self._min_threshold_wt)
        else:
            raise ValueError('Please provide either distances or betas, but not both.')

    @property
    def uids(self):
        """Node uids corresponding to each node in the graph's node_map."""
        return self._uids

    @property
    def distances(self):
        """The distance threshold/s at which the class has been initialised."""
        return self._distances

    @property
    def betas(self):
        """The distance decay $\beta$ thresholds (spatial impedance) at which the class is initialised."""
        return self._betas

    @property
    def node_x_arr(self):
        return self._node_data[:, 0]

    @property
    def node_y_arr(self):
        return self._node_data[:, 1]

    @property
    def node_live_arr(self):
        return self._node_data[:, 2]

    @property
    def edge_lengths_arr(self):
        return self._edge_data[:, 2]

    @property
    def edge_angles_arr(self):
        return self._edge_data[:, 3]

    @property
    def edge_impedance_factors_arr(self):
        return self._edge_data[:, 4]

    @property
    def edge_in_bearings_arr(self):
        return self._edge_data[:, 5]

    @property
    def edge_out_bearings_arr(self):
        return self._edge_data[:, 6]

    @property
    def networkX_multigraph(self):
        """If initialised with `NetworkLayerFromNX`, the `networkX` `MultiGraph` from which the graph is derived."""
        return self._networkX_multigraph

    @networkX_multigraph.setter
    def networkX_multigraph(self, networkX_multigraph):
        self._networkX_multigraph = networkX_multigraph

    # for retrieving metrics to a dictionary
    def metrics_to_dict(self):
        '''
        metrics are stored in arrays, this method unpacks per uid
        '''
        m = {}
        for i, uid in enumerate(self._uids):
            m[uid] = {
                'x': self.node_x_arr[i],
                'y': self.node_y_arr[i],
                'live': self.node_live_arr[i] == 1
            }
            # unpack centralities
            m[uid]['centrality'] = {}
            for m_key, m_val in self.metrics['centrality'].items():
                m[uid]['centrality'][m_key] = {}
                for d_key, d_val in m_val.items():
                    m[uid]['centrality'][m_key][d_key] = d_val[i]

            m[uid]['mixed_uses'] = {}
            for m_key, m_val in self.metrics['mixed_uses'].items():
                m[uid]['mixed_uses'][m_key] = {}
                if 'hill' in m_key:
                    for q_key, q_val in m_val.items():
                        m[uid]['mixed_uses'][m_key][q_key] = {}
                        for d_key, d_val in q_val.items():
                            m[uid]['mixed_uses'][m_key][q_key][d_key] = d_val[i]
                else:
                    for d_key, d_val in m_val.items():
                        m[uid]['mixed_uses'][m_key][d_key] = d_val[i]
            m[uid]['accessibility'] = {
                'non_weighted': {},
                'weighted': {}
            }
            for cat in ['non_weighted', 'weighted']:
                for cl_key, cl_val in self.metrics['accessibility'][cat].items():
                    m[uid]['accessibility'][cat][cl_key] = {}
                    for d_key, d_val in cl_val.items():
                        m[uid]['accessibility'][cat][cl_key][d_key] = d_val[i]
            m[uid]['stats'] = {}
            for th_key, th_val in self.metrics['stats'].items():
                m[uid]['stats'][th_key] = {}
                for stat_key, stat_val in th_val.items():
                    m[uid]['stats'][th_key][stat_key] = {}
                    for d_key, d_val in stat_val.items():
                        m[uid]['stats'][th_key][stat_key][d_key] = d_val[i]
        return m

    # for unpacking to a networkX graph
    def to_networkX(self):
        metrics_dict = self.metrics_to_dict()
        return graphs.nX_from_graph_maps(self._uids,
                                         self._node_data,
                                         self._edge_data,
                                         self._node_edge_map,
                                         self._networkX_multigraph,
                                         metrics_dict)

    # deprecated method
    def compute_centrality(self, **kwargs):
        raise DeprecationWarning('The compute_centrality method has been deprecated. '
                                 'It has been split into two: '
                                 'use "node_centrality" for node based measures '
                                 'and "segment_centrality" for segmentised measures.'
                                 'See the documentation for further information.')

    # provides access to the underlying centrality.local_centrality method
    def node_centrality(self,
                        measures: Union[list, tuple] = None,
                        angular: bool = False):
        # see centrality.local_centrality for integrity checks on closeness and betweenness keys
        # typos are caught below
        if not angular:
            heuristic = 'shortest (non-angular)'
            options = (
                'node_density',
                'node_farness',
                'node_cycles',
                'node_harmonic',
                'node_beta',
                'node_betweenness',
                'node_betweenness_beta'
            )
        else:
            heuristic = 'simplest (angular)'
            options = (
                'node_harmonic_angular',
                'node_betweenness_angular'
            )
        if measures is None:
            raise ValueError(f'Please select at least one measure to compute.')
        measure_keys = []
        for measure in measures:
            if measure not in options:
                raise ValueError(f'Invalid network measure: {measure}. '
                                 f'Must be one of {", ".join(options)} when using {heuristic} path heuristic.')
            if measure in measure_keys:
                raise ValueError(f'Please remove duplicate measure: {measure}.')
            measure_keys.append(measure)
        measure_keys = tuple(measure_keys)
        if not checks.quiet_mode:
            logger.info(f'Computing {", ".join(measure_keys)} centrality measures using {heuristic} path heuristic.')
        measures_data = centrality.local_node_centrality(
            self._node_data,
            self._edge_data,
            self._node_edge_map,
            np.array(self._distances),
            np.array(self._betas),
            measure_keys,
            angular,
            suppress_progress=checks.quiet_mode)
        # write the results
        # writing metrics to dictionary will check for pre-existing
        # but writing sub-distances arrays will overwrite prior
        for measure_idx, measure_name in enumerate(measure_keys):
            if measure_name not in self.metrics['centrality']:
                self.metrics['centrality'][measure_name] = {}
            for d_idx, d_key in enumerate(self._distances):
                self.metrics['centrality'][measure_name][d_key] = measures_data[measure_idx][d_idx]

    # provides access to the underlying centrality.local_centrality method
    def segment_centrality(self,
                           measures: Union[list, tuple] = None,
                           angular: bool = False):
        # see centrality.local_centrality for integrity checks on closeness and betweenness keys
        # typos are caught below
        if not angular:
            heuristic = 'shortest (non-angular)'
            options = (
                'segment_density',
                'segment_harmonic',
                'segment_beta',
                'segment_betweenness'
            )
        else:
            heuristic = 'simplest (angular)'
            options = (
                'segment_harmonic_hybrid',
                'segment_betweeness_hybrid'
            )
        if measures is None:
            raise ValueError(f'Please select at least one measure to compute.')
        measure_keys = []
        for measure in measures:
            if measure not in options:
                raise ValueError(f'Invalid network measure: {measure}. '
                                 f'Must be one of {", ".join(options)} when using {heuristic} path heuristic.')
            if measure in measure_keys:
                raise ValueError(f'Please remove duplicate measure: {measure}.')
            measure_keys.append(measure)
        measure_keys = tuple(measure_keys)
        if not checks.quiet_mode:
            logger.info(f'Computing {", ".join(measure_keys)} centrality measures using {heuristic} path heuristic.')
        measures_data = centrality.local_segment_centrality(
            self._node_data,
            self._edge_data,
            self._node_edge_map,
            np.array(self._distances),
            np.array(self._betas),
            measure_keys,
            angular,
            suppress_progress=checks.quiet_mode)
        # write the results
        # writing metrics to dictionary will check for pre-existing
        # but writing sub-distances arrays will overwrite prior
        for measure_idx, measure_name in enumerate(measure_keys):
            if measure_name not in self.metrics['centrality']:
                self.metrics['centrality'][measure_name] = {}
            for d_idx, d_key in enumerate(self._distances):
                self.metrics['centrality'][measure_name][d_key] = measures_data[measure_idx][d_idx]


class NetworkLayerFromNX(NetworkLayer):
    def __init__(self,
                 networkX_graph: nx.MultiGraph,
                 distances: Union[list, tuple, np.ndarray] = None,
                 betas: Union[list, tuple, np.ndarray] = None,
                 min_threshold_wt: float = checks.def_min_thresh_wt):
        node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(networkX_graph)
        super().__init__(node_uids,
                         node_data,
                         edge_data,
                         node_edge_map,
                         distances,
                         betas,
                         min_threshold_wt)
        # keep reference to networkX graph
        self.networkX = networkX_graph
