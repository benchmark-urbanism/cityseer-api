'''
Centrality methods
'''
import logging
from typing import Union

import networkx as nx
import numpy as np
from numba.typed import Dict

from cityseer.algos import centrality, checks
from cityseer.util import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def distance_from_beta(beta: Union[float, list, np.ndarray],
                       min_threshold_wt: float = checks.def_min_thresh_wt) -> np.ndarray:
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

class Network_Layer:

    def __init__(self,
                 node_uids: Union[list, tuple],
                 node_data: np.ndarray,
                 edge_data: np.ndarray,
                 node_edge_map: Dict,
                 distances: Union[list, tuple, np.ndarray] = None,
                 betas: Union[list, tuple, np.ndarray] = None,
                 min_threshold_wt: float = checks.def_min_thresh_wt):
        '''
        NODE MAP:
        0 - x
        1 - y
        2 - live

        EDGE MAP:
        0 - start node
        1 - end node
        2 - length in metres
        3 - sum of angular travel along length
        4 - impedance factor
        5 - in bearing
        6 - out bearing
        '''
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
        self._networkX = None
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
        return self._uids

    @property
    def distances(self):
        return self._distances

    @property
    def betas(self):
        return self._betas

    @property
    def min_threshold_wt(self):
        return self._min_threshold_wt

    @property
    def x_arr(self):
        return self._node_data[:, 0]

    @property
    def y_arr(self):
        return self._node_data[:, 1]

    @property
    def live(self):
        return self._node_data[:, 2]

    @property
    def edge_lengths(self):
        return self._edge_data[:, 2]

    @property
    def edge_angles(self):
        return self._edge_data[:, 3]

    @property
    def edge_impedance_factor(self):
        return self._edge_data[:, 4]

    @property
    def edge_in_bearing(self):
        return self._edge_data[:, 5]

    @property
    def edge_out_bearing(self):
        return self._edge_data[:, 6]

    @property
    def networkX(self):
        return self._networkX

    @networkX.setter
    def networkX(self, value):
        self._networkX = value

    # for retrieving metrics to a dictionary
    def metrics_to_dict(self):
        '''
        metrics are stored in arrays, this method unpacks per uid
        '''
        m = {}
        for i, uid in enumerate(self._uids):
            m[uid] = {
                'x': self.x_arr[i],
                'y': self.y_arr[i],
                'live': self.live[i] == 1
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
                                         self._networkX,
                                         metrics_dict)

    # deprecated method
    def compute_centrality(self, **kwargs):
        raise DeprecationWarning('The compute_centrality method has been deprecated. '
                                 'It has been split into two: '
                                 'use "compute_node_centrality" for node based measures '
                                 'and "compute_segment_centrality" for segmentised measures.'
                                 'See the documentation for further information.')

    # provides access to the underlying centrality.local_centrality method
    def compute_node_centrality(self,
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
    def compute_segment_centrality(self,
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


class Network_Layer_From_nX(Network_Layer):
    def __init__(self,
                 networkX_graph: nx.Graph,
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
