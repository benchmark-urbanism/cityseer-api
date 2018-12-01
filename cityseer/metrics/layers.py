'''

'''
import logging
from typing import Tuple

import numpy as np
import utm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cityseer.algos import data, checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dict_wgs_to_utm(data_dict: dict) -> dict:
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


def data_map_from_dict(data_dict: dict) -> Tuple[tuple, np.ndarray, tuple]:
    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    # extract data classes, if present, and convert to ints
    logger.info('Extracting data classes, if present')
    classes_raw = set([v['class'] for v in data_dict.values() if 'class' in v])
    # use sklearn's label encoder
    le = LabelEncoder()
    le.fit(list(classes_raw))
    # map the int encodings to the respective classes
    classes_raw = le.inverse_transform(range(len(classes_raw)))

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
            data_map[i][3] = le.transform([v['class']])[0]
        # DATA MAP INDEX POSITION 4 = assigned network index - leave as default np.nan
        # pass
        # DATA MAP INDEX POSITION 5 = distance from assigned network index - leave as default np.nan
        # pass

    return tuple(data_labels), data_map, tuple(classes_raw)


class Data_Layer:

    def __init__(self, data_uids: [list, tuple], data_map: np.ndarray, class_labels: [list, tuple]):

        '''
        DATA MAP:
        0 - x
        1 - y
        2 - live
        3 - data class - integer form encoded from original raw classes
        4 - assigned network index - nearest
        5 - assigned network index - next-nearest

        INDEX MAP
        0 - x_arr
        1 - x_idx - corresponds to original index of non-sorted x_arr
        2 - y_arr
        3 - y_idx - corresponds to original index of non-sorted y_arr
        '''

        self.uids = data_uids  # original labels / indices for each data point
        self.data = data_map  # data map per above
        self.class_labels = class_labels  # original raw data classes

        # check the data structures
        if len(self.uids) != len(self.data):
            raise ValueError('The number of data labels does not match the number of data points.')
        if len(self.class_labels) != len(set(self.class_codes)):
            raise ValueError('The number of data class labels does not match the number of data class codes.')
        checks.check_data_map(self.data)

    @property
    def x_arr(self):
        return self.data[:, 0]

    @property
    def y_arr(self):
        return self.data[:, 1]

    @property
    def live(self):
        return self.data[:, 2]

    @property
    def class_codes(self):
        return self.data[:, 3]

    def assign_to_network(self, Network_Layer, max_dist):
        data.assign_to_network(self.data, Network_Layer.nodes, Network_Layer.edges, max_dist)


class Data_Layer_From_Dict(Data_Layer):

    def __init__(self, data_dict):
        d_labels, d_map, d_classes = data_map_from_dict(data_dict)
        super().__init__(d_labels, d_map, d_classes)
