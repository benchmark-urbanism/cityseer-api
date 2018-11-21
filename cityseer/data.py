import logging
from typing import Tuple
from tqdm import tqdm
import utm
import numpy as np
from numba.pycc import CC
from numba import njit


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cc = CC('data')


def data_dict_wgs_to_utm(data_dict:dict) -> dict:

    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    logger.info('Converting data dictionary from WGS to UTM.')
    data_dict_copy = data_dict.copy()

    logger.info('Processing node x, y coordinates.')
    for k, v in tqdm(data_dict_copy.items()):
        # x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at data dictionary key {k}.')
        x = v['x']
        # y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at data dictionary key {k}.')
        y = v['y']
        # check for unintentional use of conversion
        if x > 180 or y > 90:
            raise AttributeError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # remember - accepts and returns in y, x order
        y, x = utm.from_latlon(y, x)[:2]
        # write back to graph
        data_dict_copy[k]['x'] = x
        data_dict_copy[k]['y'] = y

    return data_dict_copy


def data_dict_to_data_map(data_dict:dict) -> Tuple[list, np.ndarray]:

    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    data_labels = []
    data_map = np.full((len(data_dict), 6), np.nan)

    for i, (k, v) in enumerate(data_dict.items()):
        # set key to data labels
        data_labels.append(k)
        # DATA MAP INDEX POSITION 0 = x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered entry missing "x" coordinate attribute at index {i}.')
        data_map[i][0] = v['x']
        # DATA MAP INDEX POSITION 1 = y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered entry missing "y" coordinate attribute at index {i}.')
        data_map[i][1] = v['y']
        # DATA MAP INDEX POSITION 2 = live or not
        if 'live' in v:
            data_map[i][2] = v['live']
        else:
            data_map[i][2] = True
        # DATA MAP INDEX POSITION 3 = optional data class - leave as np.nan if not present
        if 'class' in v:
            data_map[i][3] = v['class']
        # DATA MAP INDEX POSITION 4 = assigned network index - leave as default np.nan
        # pass
        # DATA MAP INDEX POSITION 5 = distance from assigned network index - leave as default np.nan
        # pass

    return data_labels, data_map


@cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64)')
@njit
def assign_to_network(data_map, node_map, max_dist):
    '''
    Each data point is assigned to the closest network node.

    This is designed to be done once prior to windowed iteration of the graph.

    Crow-flies operations are performed inside the iterative data aggregation step because pre-computation would be memory-prohibitive due to an N*M matrix.

    Note that the assignment to a network index is a starting reference for the data aggregation step, and that if the prior point on the shortest path is closer, then the distance will be calculated via the prior point instead.

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx
    4 - weight

    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - data class
    4 - assigned network index
    5 - distance from assigned network index
    '''

    if data_map.shape[1] != 6:
        raise AttributeError('The data map must have a dimensionality of Nx6, with the first four indices consisting of x, y, live, and class attributes. This method will populate indices 5 and 6.')

    if node_map.shape[1] != 5:
        raise AttributeError('The node map must have a dimensionality of Nx5, consisting of x, y, live, link idx, and weight attributes.')

    netw_x_arr = node_map[:,0]
    netw_y_arr = node_map[:,1]
    data_x_arr = data_map[:,0]
    data_y_arr = data_map[:,1]

    # iterate each data point
    for data_idx in range(len(data_map)):
        # iterate each network id
        for network_idx in range(len(node_map)):
            # get the distance
            dist = np.sqrt(
                (netw_x_arr[network_idx] - data_x_arr[data_idx]) ** 2 +
                (netw_y_arr[network_idx] - data_y_arr[data_idx]) ** 2)
            # only proceed if it is less than the max dist cutoff
            if dist > max_dist:
                continue
            # if no adjacent network point has yet been assigned for this data point
            # then proceed to record this adjacency and the corresponding distance
            elif np.isnan(data_map[data_idx][5]):
                data_map[data_idx][5] = dist
                data_map[data_idx][4] = network_idx
            # otherwise, only update if the new distance is less than any prior distances
            elif dist < data_map[data_idx][5]:
                data_map[data_idx][5] = dist
                data_map[data_idx][4] = network_idx

    return data_map
