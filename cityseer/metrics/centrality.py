'''
Centrality methods
'''
import logging
from typing import Union, Tuple, Any
import numpy as np
from cityseer.algos import networks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


default_min_threshold_wt = 0.01831563888873418


def distance_from_beta(beta:Union[float, list, np.ndarray], min_threshold_wt:float=default_min_threshold_wt) -> Tuple[np.ndarray, float]:

    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]

    # check that the betas do not have leading negatives
    for b in beta:
        if b < 0:
            raise ValueError('Please provide the beta/s without the leading negative.')

    # cast to numpy
    beta = np.array(beta)

    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / -beta, min_threshold_wt


Enumerable = Union[list, tuple, np.ndarray]
def compute_centrality(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable, close_metrics:list=None,
        between_metrics:list=None, min_threshold_wt:float=default_min_threshold_wt, angular:bool=False) -> Tuple[Any, ...]:
    '''
    This method provides full access to the underlying network.network_centralities method
    '''

    if node_map.shape[1] != 5:
        raise AttributeError('The node map must have a dimensionality of Nx5, consisting of x, y, live, link idx, and weight attributes.')

    if edge_map.shape[1] != 4:
        raise AttributeError('The link map must have a dimensionality of Nx4, consisting of start, end, length, and impedance attributes.')

    if distances == []:
        raise ValueError('A list of local centrality distance thresholds is required.')

    if isinstance(distances, (int, float)):
        distances = [distances]

    if not isinstance(distances, (list, tuple, np.ndarray)):
        raise TypeError('Please provide a distance or a list, tuple, or numpy.ndarray of distances.')

    betas = []
    for d in distances:
        betas.append(np.log(min_threshold_wt) / d)

    if close_metrics is None and between_metrics is None:
        raise ValueError(f'Neither closeness nor betweenness metrics specified, please specify at least one metric to compute.')

    closeness_options = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity', 'cycles']
    closeness_map = []
    if close_metrics:
        for cl in close_metrics:
            if cl not in closeness_options:
                raise ValueError(f'Invalid closeness option: {cl}. Must be one of {", ".join(closeness_options)}.')
            closeness_map.append(closeness_options.index(cl))
    # improved closeness is extrapolated from node density and farness_distance, so these may have to be added regardless:
    # assign to new variable so as to keep closeness_map pure for later use in unpacking the results
    closeness_map_extra = closeness_map
    if close_metrics and 'improved' in close_metrics:
        closeness_map_extra = list(set(closeness_map_extra + [
            closeness_options.index('node_density'),
            closeness_options.index('farness_distance')]))

    betweenness_options = ['betweenness', 'betweenness_gravity']
    betweenness_map = []
    if between_metrics:
        for bt in between_metrics:
            if bt not in betweenness_options:
                raise ValueError(f'Invalid betweenness option: {bt}. Must be one of {", ".join(betweenness_options)}.')
            betweenness_map.append(betweenness_options.index(bt))

    closeness_data, betweenness_data = networks.network_centralities(node_map, edge_map, np.array(distances),
                                                                     np.array(betas), np.array(closeness_map_extra), np.array(betweenness_map), angular)

    # return statement tuple unpacking supported from Python 3.8... till then, unpack first
    return_data = list((*closeness_data[closeness_map], *betweenness_data[betweenness_map]))
    # unpack if single dimension
    if len(distances) == 1:
        for i in range(len(return_data)):
            return_data[i] = return_data[i][0]
    # add betas
    return_data = *return_data, betas
    return return_data


def harmonic_closeness(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable) -> np.ndarray:
    harmonic, betas = compute_centrality(node_map, edge_map, distances, close_metrics=['harmonic'])
    # discard betas - unused
    return harmonic


def gravity(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable=None, betas:Enumerable=None,
            min_threshold_wt=default_min_threshold_wt) -> np.ndarray:
    # establish distances and min weights
    if distances is not None and betas is None:
        dist = distances
        threshold_wt = min_threshold_wt
    elif betas is not None and distances is None:
        dist, threshold_wt = distance_from_beta(betas, min_threshold_wt=min_threshold_wt)
    else:
        raise ValueError('Please provide either distances or betas, but not both.')

    gravity, betas = compute_centrality(node_map, edge_map, dist, close_metrics=['gravity'], min_threshold_wt=threshold_wt)
    # discard betas - unused
    return gravity


def angular_harmonic_closeness(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable) -> np.ndarray:
    harmonic, betas = compute_centrality(node_map, edge_map, distances, close_metrics=['harmonic'], angular=True)
    # discard betas - unused
    return harmonic


def betweenness(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable) -> np.ndarray:
    betw, betas = compute_centrality(node_map, edge_map, distances, between_metrics=['betweenness'])
    # discard betas - unused
    return betw


def betweenness_gravity(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable=None, betas:Enumerable=None,
                        min_threshold_wt=default_min_threshold_wt) -> np.ndarray:
    # establish distances and min weights
    if distances is not None and betas is None:
        dist = distances
        threshold_wt = min_threshold_wt
    elif betas is not None and distances is None:
        dist, threshold_wt = distance_from_beta(betas, min_threshold_wt=min_threshold_wt)
    else:
        raise ValueError('Please provide either distances or betas, but not both.')

    betw, betas = compute_centrality(node_map, edge_map, dist, between_metrics=['betweenness_gravity'], min_threshold_wt=threshold_wt)
    # discard betas - unused
    return betw


def angular_betweenness(node_map:np.ndarray, edge_map:np.ndarray, distances:Enumerable) -> np.ndarray:
    betw, betas = compute_centrality(node_map, edge_map, distances, between_metrics=['betweenness'], angular=True)
    # discard betas - unused
    return betw
