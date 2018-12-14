'''

'''
import logging
from typing import Tuple, Union

import numpy as np
import utm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cityseer.algos import data, checks, diversity

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

    def __init__(self,
                 data_uids: Union[list, tuple, np.ndarray],
                 data_map: np.ndarray,
                 class_labels: Union[list, tuple, np.ndarray],
                 cl_disparity_wt_matrix: Union[list, tuple, np.ndarray] = None,
                 qs: Union[list, tuple, np.ndarray] = None):

        '''
        DATA MAP:
        0 - x
        1 - y
        2 - live
        3 - data class - integer form encoded from original raw classes
        4 - assigned network index - nearest
        5 - assigned network index - next-nearest
        '''

        self._uids = data_uids  # original labels / indices for each data point
        self._data = data_map  # data map per above
        self._class_labels = class_labels  # original raw data classes
        self._qs = qs  # selected Hill's Q parameters for mixed-use hill measures
        self._cl_disparity_wt_matrix = cl_disparity_wt_matrix  # matrix of pairwise weights between classifications

        self._Network = None

        # checks
        checks.check_data_map(self._data, check_assigned=False)

        if len(self._uids) != len(self._data):
            raise ValueError('The number of data labels does not match the number of data points.')

        if len(self._class_labels) != len(set(self.class_codes)):
            raise ValueError('The number of data class labels does not match the number of unique data class codes.')

        # warn if no qs provided
        if self._qs is None:
            logger.warning(
                'This data class was initialised without any values of q. At least one value of q is required if making use of any "hill" mixed use metrics.')
            self._qs = []
        if isinstance(self._qs, (int, float)):
            self._qs = [self._qs]
        if not isinstance(self._qs, (list, tuple, np.ndarray)):
            raise TypeError('Please provide a float, list, tuple, or numpy.ndarray of q values.')

        if self._cl_disparity_wt_matrix is None:
            logger.warning(
                'No class disparity weights matrix provided: This is required if making use of the "hill_pairwise_disparity" or "raos_pairwise_disparity" measures.')
            self._cl_disparity_wt_matrix = [[]]
        elif not isinstance(self._cl_disparity_wt_matrix, (list, tuple, np.ndarray)) or \
                self._cl_disparity_wt_matrix.ndim != 2 or \
                self._cl_disparity_wt_matrix.shape[0] != self._cl_disparity_wt_matrix.shape[1] or \
                len(self._cl_disparity_wt_matrix) != len(self.class_labels):
            raise TypeError(
                'Disparity weights must be a pairwise NxN matrix in list, tuple, or numpy.ndarray form. The number of edge-wise elements should match the number of unique class labels.')

    @property
    def uids(self):
        return self._uids

    @property
    def x_arr(self):
        return self._data[:, 0]

    @property
    def y_arr(self):
        return self._data[:, 1]

    @property
    def live(self):
        return self._data[:, 2]

    @property
    def class_codes(self):
        return self._data[:, 3]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def qs(self):
        return self._qs

    @property
    def cl_disparity_wt_matrix(self):
        return self._cl_disparity_wt_matrix

    @property
    def Network(self):
        return self._Network

    def assign_to_network(self, Network_Layer, max_dist):
        self._Network = Network_Layer
        data.assign_to_network(self._data, self.Network._nodes, self.Network._edges, max_dist)

    def compute_landuses(self,
                         mixed_use_metrics: Union[list, tuple] = None,
                         accessibility_labels: Union[list, tuple] = None):
        '''
        This method provides full access to the underlying diversity.local_landuses method
        '''

        # checks on parameter integrity occur in underlying method

        if self.Network is None:
            raise ValueError('Assign this data layer to a network prior to computing mixed-uses or accessibilities.')

        # extrapolate the requested mixed use measures
        mixed_uses_options = ['hill',
                              'hill_branch_wt',
                              'hill_pairwise_wt',
                              'hill_pairwise_disparity',
                              'shannon',
                              'gini_simpson',
                              'raos_pairwise_disparity']
        mixed_use_hill_keys = []
        mixed_use_other_keys = []
        if mixed_use_metrics is not None:
            for mu in mixed_use_metrics:
                if mu not in mixed_uses_options:
                    raise ValueError(f'Invalid mixed-use option: {mu}. Must be one of {", ".join(mixed_uses_options)}.')
                idx = mixed_uses_options.index(mu)
                if idx < 4:
                    mixed_use_hill_keys.append(idx)
                else:
                    mixed_use_other_keys.append(idx - 4)

        accessibility_keys = []
        if accessibility_labels is not None:
            for ac_label in accessibility_labels:
                if ac_label not in self.class_labels:
                    raise ValueError('Class label not found in Data Layer class labels.')
                accessibility_keys.append(self.class_labels.index(ac_label))

        mixed_use_hill_data, mixed_use_other_data, accessibility_data, accessibility_data_wt = \
            diversity.local_landuses(self.Network._nodes,
                                     self.Network._edges,
                                     self._data,
                                     np.array(self.Network.distances),
                                     np.array(self.Network.betas),
                                     np.array(self.qs),
                                     np.array(mixed_use_hill_keys),
                                     np.array(mixed_use_other_keys),
                                     np.array(accessibility_keys),
                                     np.array(self.cl_disparity_wt_matrix),
                                     self.Network.angular)

        # write the results to the Network's metrics dict
        # keys will check for pre-existing, whereas qs and distance keys will overwrite
        # unpack mixed use hill
        if mixed_use_hill_keys is not None:
            for mu_h_idx, mu_h_key in enumerate(mixed_use_hill_keys):
                mu_h_label = mixed_uses_options[mu_h_key]
                if mu_h_label not in self.Network.metrics['mixed_uses']:
                    self.Network.metrics['mixed_uses'][mu_h_label] = {}
                for q_idx, q_key in enumerate(self.qs):
                    self.Network.metrics['mixed_uses'][mu_h_label][q_key] = {}
                    for d_idx, d_key in enumerate(self.Network.distances):
                        self.Network.metrics['mixed_uses'][mu_h_label][q_key][d_key] = \
                            mixed_use_hill_data[mu_h_idx][q_idx][d_idx]
        # unpack mixed use other
        if mixed_use_other_keys is not None:
            for mu_o_idx, mu_o_key in enumerate(mixed_use_other_keys):
                mu_o_label = mixed_uses_options[mu_o_key + 4]
                if mu_o_label not in self.Network.metrics['mixed_uses']:
                    self.Network.metrics['mixed_uses'][mu_o_label] = {}
                # no qs
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['mixed_uses'][mu_o_label][d_key] = mixed_use_other_data[mu_o_idx][d_idx]
        # unpack accessibility data
        for ac_idx, ac_code in enumerate(accessibility_keys):
            ac_label = self.class_labels[ac_code]
            for k, data in zip(['non_weighted', 'weighted'], [accessibility_data, accessibility_data_wt]):
                if ac_label not in self.Network.metrics['accessibility'][k]:
                    self.Network.metrics['accessibility'][k][ac_label] = {}
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['accessibility'][k][ac_label][d_key] = data[ac_idx][d_idx]

    def hill_diversity(self):
        return self.compute_landuses(mixed_use_metrics=['hill'])

    def hill_branch_wt_diversity(self):
        return self.compute_landuses(mixed_use_metrics=['hill_branch_wt'])

    def hill_pairwise_wt_diversity(self):
        return self.compute_landuses(mixed_use_metrics=['hill_pairwise_wt'])

    def compute_accessibilities(self, accessibility_labels: Union[list, tuple]):
        return self.compute_landuses(accessibility_labels=accessibility_labels)


class Data_Layer_From_Dict(Data_Layer):

    def __init__(self,
                 data_dict: dict,
                 cl_disparity_wt_matrix: Union[list, tuple, np.ndarray] = None,
                 qs: Union[list, tuple, np.ndarray] = None
                 ):
        data_uids, data_map, class_labels = data_map_from_dict(data_dict)

        super().__init__(data_uids, data_map, class_labels, cl_disparity_wt_matrix, qs)
