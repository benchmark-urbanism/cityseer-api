'''
Centrality methods
'''
import logging
from typing import Union
import numpy as np
import networkx as nx
from cityseer.algos import centrality, data, types
from cityseer.util import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def_min_thresh_wt = 0.01831563888873418


def distance_from_beta(beta: Union[float, list, np.ndarray],
                       min_threshold_wt: float = def_min_thresh_wt) -> np.ndarray:
    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]
    # check that the betas do not have leading negatives
    for b in beta:
        if b >= 0:
            raise ValueError('Please provide the beta/s without the leading negative.')
    # cast to numpy
    beta = np.array(beta)
    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / beta


class Network_Layer:

    def __init__(self,
                 node_uids: Union[list, tuple],
                 node_map: np.ndarray,
                 edge_map: np.ndarray,
                 distances: Union[list, tuple, np.ndarray] = None,
                 betas: Union[list, tuple, np.ndarray] = None,
                 min_threshold_wt: float = def_min_thresh_wt,
                 angular: bool = False):

        '''
        NODE MAP:
        0 - x
        1 - y
        2 - live
        3 - edge indx
        4 - weight

        EDGE MAP:
        0 - start node
        1 - end node
        2 - length in metres
        3 - impedance

        INDEX MAP
        0 - x_arr
        1 - x_idx - corresponds to original index of non-sorted x_arr
        2 - y_arr
        3 - y_idx
        '''

        self.uids = node_uids
        self.nodes = node_map
        self.edges = edge_map
        self.index = data.generate_index(self.x_arr, self.y_arr)
        self.distances = distances
        self.betas = betas
        self.min_threshold_wt = min_threshold_wt
        self.angular = angular
        self.metrics = {
            'centrality': {},
            'landuses': {}
        }
        self.networkX = None

        # if distances, check the types and generate the betas
        if self.distances is not None and self.betas is None:
            if self.distances == []:
                raise ValueError('A list of local centrality distance thresholds is required.')
            if isinstance(self.distances, (int, float)):
                self.distances = [self.distances]
            if not isinstance(self.distances, (list, tuple, np.ndarray)):
                raise TypeError('Please provide a distance or a list, tuple, or numpy.ndarray of distances.')
            # generate the betas
            self.betas = []
            for d in self.distances:
                self.betas.append(np.log(self.min_threshold_wt) / d)
        # if betas, generate the distances
        elif self.betas is not None and self.distances is None:
            self.distances = distance_from_beta(self.betas, min_threshold_wt=self.min_threshold_wt)
        else:
            raise ValueError('Please provide either distances or betas, but not both.')

        # check the data structures
        if len(self.uids) != len(self.nodes):
            raise ValueError('The number of indices does not match the number of nodes.')
        if len(self.nodes) != len(self.index):
            raise ValueError('The data map and index map are not the same lengths.')
        types.check_network_types(self.nodes, self.edges)
        types.check_index_map(self.index)

    @property
    def x_arr(self):
        return self.nodes[:, 0]

    @property
    def y_arr(self):
        return self.nodes[:, 1]

    @property
    def live(self):
        return self.nodes[:, 2]

    def to_networkX(self):
        return graphs.networkX_from_graph_maps(self.uids, self.nodes, self.edges, self.networkX, self.metrics)

    def compute_centrality(self, close_metrics: Union[list, tuple] = None, between_metrics: Union[list, tuple] = None):
        '''
        This method provides full access to the underlying network.network_centralities method
        '''

        if close_metrics is None and between_metrics is None:
            raise ValueError(
                f'Neither closeness nor betweenness metrics specified, please specify at least one metric to compute.')

        closeness_options = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity',
                             'cycles']
        closeness_map = []
        if close_metrics is not None:
            for cl in close_metrics:
                if cl not in closeness_options:
                    raise ValueError(f'Invalid closeness option: {cl}. Must be one of {", ".join(closeness_options)}.')
                closeness_map.append(closeness_options.index(cl))
        # improved closeness is extrapolated from node density and farness_distance, so these may have to be added regardless:
        # assign to new variable so as to keep closeness_map pure for later use in unpacking the results
        closeness_map_extra = closeness_map
        if close_metrics is not None and 'improved' in close_metrics:
            closeness_map_extra = list(set(closeness_map_extra + [
                closeness_options.index('node_density'),
                closeness_options.index('farness_distance')]))

        betweenness_options = ['betweenness', 'betweenness_gravity']
        betweenness_map = []
        if between_metrics is not None:
            for bt in between_metrics:
                if bt not in betweenness_options:
                    raise ValueError(
                        f'Invalid betweenness option: {bt}. Must be one of {", ".join(betweenness_options)}.')
                betweenness_map.append(betweenness_options.index(bt))

        closeness_data, betweenness_data = centrality.local_centrality(
            self.nodes,
            self.edges,
            np.array(self.distances),
            np.array(self.betas),
            np.array(closeness_map_extra),
            np.array(betweenness_map),
            self.angular)

        # write the results
        # keys will check for pre-existing
        # distances will overwrite
        if close_metrics is not None:
            for cl_key, cl_idx in zip(close_metrics, closeness_map):
                if cl_key not in self.metrics['centrality']:
                    self.metrics['centrality'][cl_key] = {}
                for d_idx, d_key in enumerate(self.distances):
                    self.metrics['centrality'][cl_key][d_key] = closeness_data[cl_idx][d_idx]

        if between_metrics is not None:
            for bt_key, bt_idx in zip(between_metrics, betweenness_map):
                if bt_key not in self.metrics['centrality']:
                    self.metrics['centrality'][bt_key] = {}
                for d_idx, d_key in enumerate(self.distances):
                    self.metrics['centrality'][bt_key][d_key] = betweenness_data[bt_idx][d_idx]

    def harmonic_closeness(self):
        return self.compute_centrality(close_metrics=['harmonic'])

    def gravity(self):
        return self.compute_centrality(close_metrics=['gravity'])

    def betweenness(self):
        return self.compute_centrality(between_metrics=['betweenness'])

    def betweenness_gravity(self):
        return self.compute_centrality(between_metrics=['betweenness_gravity'])


class Network_Layer_From_NetworkX(Network_Layer):

    def __init__(self,
                 networkX_graph: nx.Graph,
                 distances: Union[list, tuple, np.ndarray] = None,
                 betas: Union[list, tuple, np.ndarray] = None,
                 min_threshold_wt: float = def_min_thresh_wt,
                 angular: bool = False):
        node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(networkX_graph)

        super().__init__(node_uids, node_map, edge_map, distances, betas, min_threshold_wt, angular)

        self.networkX = networkX_graph
