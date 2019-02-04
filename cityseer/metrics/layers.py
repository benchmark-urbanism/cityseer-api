import logging
from typing import Tuple, List, Union

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


def encode_categorical(classes: Union[list, tuple, np.ndarray]) -> Tuple[tuple, np.ndarray]:
    if not isinstance(classes, (list, tuple, np.ndarray)):
        raise TypeError('This method requires an iterable object.')

    # use sklearn's label encoder
    le = LabelEncoder()
    le.fit(classes)
    # map the int encodings to the respective classes
    classes_int = le.transform(classes)

    return tuple(le.classes_), classes_int


def data_map_from_dict(data_dict: dict) -> Tuple[tuple, np.ndarray]:
    '''
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    data_uids = []
    data_map = np.full((len(data_dict), 4), np.nan)
    for i, (k, v) in enumerate(data_dict.items()):
        # set key to data labels
        data_uids.append(k)
        # DATA MAP INDEX POSITION 0 = x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered entry missing "x" coordinate attribute at index {i}.')
        data_map[i][0] = v['x']
        # DATA MAP INDEX POSITION 1 = y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered entry missing "y" coordinate attribute at index {i}.')
        data_map[i][1] = v['y']

    return tuple(data_uids), data_map


class Data_Layer:
    '''
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''

    def __init__(self,
                 data_uids: Union[list, tuple, np.ndarray],
                 data_map: np.ndarray):

        self._uids = data_uids  # original labels / indices for each data point
        self._data = data_map  # data map per above
        self._Network = None

        # checks
        checks.check_data_map(self._data, check_assigned=False)

        if len(self._uids) != len(self._data):
            raise ValueError('The number of data labels does not match the number of data points.')

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
    def Network(self):
        return self._Network

    def assign_to_network(self, Network_Layer, max_dist):
        self._Network = Network_Layer
        data.assign_to_network(self._data, self.Network._nodes, self.Network._edges, max_dist)

    def compute_aggregated(self,
                           landuse_labels: Union[list, tuple, np.ndarray] = None,
                           mixed_use_metrics: Union[list, tuple] = None,
                           accessibility_labels: Union[list, tuple] = None,
                           cl_disparity_wt_matrix: Union[list, tuple, np.ndarray] = None,
                           qs: Union[list, tuple, np.ndarray] = None,
                           numerical_labels: Union[list, tuple] = None,
                           numerical_arrays: Union[List[Union[list, tuple, np.ndarray]],
                                                   Tuple[Union[list, tuple, np.ndarray]],
                                                   np.ndarray] = None):
        '''
        This method provides full access to the underlying diversity.local_landuses method
        '''

        if self.Network is None:
            raise ValueError('Assign this data layer to a network prior to computing mixed-uses or accessibilities.')

        mixed_uses_options = ['hill',
                              'hill_branch_wt',
                              'hill_pairwise_wt',
                              'hill_pairwise_disparity',
                              'shannon',
                              'gini_simpson',
                              'raos_pairwise_disparity']

        if numerical_labels is not None and numerical_arrays is None \
                or numerical_labels is None and numerical_arrays is not None \
                or numerical_labels is not None and numerical_arrays is not None and \
                len(numerical_arrays) != len(numerical_labels):
            raise ValueError('An equal number of stats labels and stats data arrays is required.')

        if numerical_arrays is None:
            numerical_arrays = np.array([[]])
        elif not isinstance(numerical_arrays, (list, tuple, np.ndarray)):
            raise ValueError('Stats data must be in the form of a list, tuple, or numpy array.')
        else:
            numerical_arrays = np.array(numerical_arrays)
            if numerical_arrays.ndim == 1:
                numerical_arrays = np.array([numerical_arrays])
            if numerical_arrays.shape[1] != len(self._data):
                raise ValueError('The length of all data arrays must match the number of data points.')

        if landuse_labels is None:
            landuse_classes = ()
            landuse_encodings = ()
            qs = ()
            mixed_use_hill_keys = ()
            mixed_use_other_keys = ()
            accessibility_keys = ()
            cl_disparity_wt_matrix = [[]]  # (()) causes error because numpy conversion creates single dimension array

        # remember, most checks on parameter integrity occur in underlying method
        # so, don't duplicate here
        else:
            if len(landuse_labels) != len(self._data):
                raise ValueError('The number of landuse labels should match the number of data points.')

            # get the landuse encodings
            landuse_classes, landuse_encodings = encode_categorical(landuse_labels)

            # if necessary, check the disparity matrix
            if cl_disparity_wt_matrix is None:
                cl_disparity_wt_matrix = [[]]
            elif not isinstance(cl_disparity_wt_matrix, (list, tuple, np.ndarray)) or \
                    cl_disparity_wt_matrix.ndim != 2 or \
                    cl_disparity_wt_matrix.shape[0] != cl_disparity_wt_matrix.shape[1] or \
                    len(cl_disparity_wt_matrix) != len(landuse_classes):
                raise TypeError(
                    'Disparity weights must be a square pairwise NxN matrix in list, tuple, or numpy.ndarray form. '
                    'The number of edge-wise elements should match the number of unique class labels.')

            # warn if no qs provided
            if qs is None:
                qs = ()
            if isinstance(qs, (int, float)):
                qs = (qs)
            if not isinstance(qs, (list, tuple, np.ndarray)):
                raise TypeError('Please provide a float, list, tuple, or numpy.ndarray of q values.')

            # extrapolate the requested mixed use measures
            mixed_use_hill_keys = []
            mixed_use_other_keys = []
            if mixed_use_metrics is not None:
                for mu in mixed_use_metrics:
                    if mu not in mixed_uses_options:
                        raise ValueError(
                            f'Invalid mixed-use option: {mu}. Must be one of {", ".join(mixed_uses_options)}.')
                    idx = mixed_uses_options.index(mu)
                    if idx < 4:
                        mixed_use_hill_keys.append(idx)
                    else:
                        mixed_use_other_keys.append(idx - 4)

            accessibility_keys = []
            if accessibility_labels is not None:
                for ac_label in accessibility_labels:
                    if ac_label not in landuse_classes:
                        logger.warning(f'No instances of accessibility label: {ac_label} present in the data.')
                    else:
                        accessibility_keys.append(landuse_classes.index(ac_label))

        # call the underlying method
        mixed_use_hill_data, mixed_use_other_data, \
        accessibility_data, accessibility_data_wt, \
        stats_mean, stats_mean_wt, \
        stats_variance, stats_variance_wt, \
        stats_max, stats_min = data.local_aggregator(self.Network._nodes,
                                                     self.Network._edges,
                                                     self._data,
                                                     distances=np.array(self.Network.distances),
                                                     betas=np.array(self.Network.betas),
                                                     landuse_encodings=np.array(landuse_encodings),
                                                     qs=np.array(qs),
                                                     mixed_use_hill_keys=np.array(mixed_use_hill_keys),
                                                     mixed_use_other_keys=np.array(mixed_use_other_keys),
                                                     accessibility_keys=np.array(accessibility_keys),
                                                     cl_disparity_wt_matrix=np.array(cl_disparity_wt_matrix),
                                                     numerical_arrays=numerical_arrays,
                                                     angular=self.Network.angular)

        # write the results to the Network's metrics dict
        # keys will check for pre-existing, whereas qs and distance keys will overwrite
        # unpack mixed use hill
        for mu_h_idx, mu_h_key in enumerate(mixed_use_hill_keys):
            mu_h_label = mixed_uses_options[mu_h_key]
            if mu_h_label not in self.Network.metrics['mixed_uses']:
                self.Network.metrics['mixed_uses'][mu_h_label] = {}
            for q_idx, q_key in enumerate(qs):
                self.Network.metrics['mixed_uses'][mu_h_label][q_key] = {}
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['mixed_uses'][mu_h_label][q_key][d_key] = \
                        mixed_use_hill_data[mu_h_idx][q_idx][d_idx]

        # unpack mixed use other
        for mu_o_idx, mu_o_key in enumerate(mixed_use_other_keys):
            mu_o_label = mixed_uses_options[mu_o_key + 4]
            if mu_o_label not in self.Network.metrics['mixed_uses']:
                self.Network.metrics['mixed_uses'][mu_o_label] = {}
            # no qs
            for d_idx, d_key in enumerate(self.Network.distances):
                self.Network.metrics['mixed_uses'][mu_o_label][d_key] = mixed_use_other_data[mu_o_idx][d_idx]

        # unpack accessibility data
        for ac_idx, ac_code in enumerate(accessibility_keys):
            ac_label = landuse_classes[ac_code]  # ac_code is index of ac_label
            for k, ac_data in zip(['non_weighted', 'weighted'], [accessibility_data, accessibility_data_wt]):
                if ac_label not in self.Network.metrics['accessibility'][k]:
                    self.Network.metrics['accessibility'][k][ac_label] = {}
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['accessibility'][k][ac_label][d_key] = ac_data[ac_idx][d_idx]

        # unpack the numerical arrays
        if numerical_labels:
            for num_idx, num_label in enumerate(numerical_labels):
                if num_label not in self.Network.metrics['stats']:
                    self.Network.metrics['stats'][num_label] = {}
                for k, stats_data in zip(['max',
                                          'min',
                                          'mean',
                                          'mean_weighted',
                                          'variance',
                                          'variance_weighted'],
                                         [stats_max,
                                          stats_min,
                                          stats_mean,
                                          stats_mean_wt,
                                          stats_variance,
                                          stats_variance_wt]):
                    if k not in self.Network.metrics['stats'][num_label]:
                        self.Network.metrics['stats'][num_label][k] = {}
                    for d_idx, d_key in enumerate(self.Network.distances):
                        self.Network.metrics['stats'][num_label][k][d_key] = stats_data[num_idx][d_idx]

    def hill_diversity(self,
                       landuse_labels: Union[list, tuple, np.ndarray],
                       qs: Union[list, tuple, np.ndarray] = None):
        return self.compute_aggregated(landuse_labels, mixed_use_metrics=['hill'], qs=qs)

    def hill_branch_wt_diversity(self,
                                 landuse_labels: Union[list, tuple, np.ndarray],
                                 qs: Union[list, tuple, np.ndarray] = None):
        return self.compute_aggregated(landuse_labels, mixed_use_metrics=['hill_branch_wt'], qs=qs)

    def compute_accessibilities(self,
                                landuse_labels: Union[list, tuple, np.ndarray],
                                accessibility_labels: Union[list, tuple]):
        return self.compute_aggregated(landuse_labels, accessibility_labels=accessibility_labels)

    def compute_stats_single(self,
                             numerical_label: str,
                             numerical_array: Union[list, tuple, np.ndarray]):

        return self.compute_aggregated(numerical_labels=[numerical_label], numerical_arrays=[numerical_array])

    def compute_stats_multiple(self,
                               numerical_labels: List[str],
                               numerical_arrays: Union[list, tuple, np.ndarray]):

        return self.compute_aggregated(numerical_labels=numerical_labels, numerical_arrays=numerical_arrays)


class Data_Layer_From_Dict(Data_Layer):

    def __init__(self, data_dict: dict):
        data_uids, data_map = data_map_from_dict(data_dict)

        super().__init__(data_uids, data_map)
