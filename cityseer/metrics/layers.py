import logging
from typing import Tuple, List, Union

import numpy as np
import utm
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from cityseer.algos import data, checks
from cityseer.metrics import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dict_wgs_to_utm(data_dict: dict) -> dict:
    """
    Converts data dictionary `x` and `y` values from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates
    to the local UTM projected coordinate system.

    Parameters
    ----------
    data_dict
        A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary with `x` and `y` key-value pairs.
        ```python
        example_data_dict = {
            'uid_01': {
                'x': 6000956.463188213,
                'y': 600693.4059810264
            },
            'uid_02': {
                'x': 6000753.336609659,
                'y': 600758.7916663144
            }
        }
        ```

    Returns
    -------
    dict
        Returns a copy of the source dictionary with the `x` and `y` values converted to the local UTM coordinate
        system.

    Notes
    ------
    ```python
    from cityseer.tools import mock
    from cityseer.metrics import layers

    # let's generate a mock data dictionary
    G_wgs = mock.mock_graph(wgs84_coords=True)
    # mock_data_dict takes on the same extents on the graph parameter
    data_dict_WGS = mock.mock_data_dict(G_wgs, random_seed=25)
    # the dictionary now contains wgs coordinates
    for i, (key, value) in enumerate(data_dict_WGS.items()):
        print(key, value)
        # prints:
        # 0 {'x': -0.09600470559254023, 'y': 51.592916036617794}
        # 1 {'x': -0.10621770551738155, 'y': 51.58888719412964}
        if i == 1:
            break
            
    # any data dictionary that follows this template can be passed to dict_wgs_to_utm()
    data_dict_UTM = layers.dict_wgs_to_utm(data_dict_WGS)
    # the coordinates have now been converted to UTM
    for i, (key, value) in enumerate(data_dict_UTM.items()):
        print(key, value)
        # prints:
        # 0 {'x': 701144.5207785056, 'y': 5719758.706109629}
        # 1 {'x': 700455.0000341447, 'y': 5719282.703221394}
        if i == 1:
            break
    ```
    """
    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    logger.info('Converting data dictionary from WGS to UTM.')
    data_dict_copy = data_dict.copy()

    logger.info('Processing node x, y coordinates.')
    for k, v in tqdm(data_dict_copy.items(), disable=checks.quiet_mode):
        # x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at data dictionary key {k}.')
        lng = v['x']
        # y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at data dictionary key {k}.')
        lat = v['y']
        # check for unintentional use of conversion
        if abs(lng) > 180 or abs(lat) > 90:
            raise AttributeError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # be cognisant of parameter and return order
        # returns in easting, northing order
        easting, northing = utm.from_latlon(lat, lng)[:2]
        # write back to graph
        data_dict_copy[k]['x'] = easting
        data_dict_copy[k]['y'] = northing

    return data_dict_copy


def encode_categorical(classes: Union[list, tuple, np.ndarray]) -> Tuple[tuple, np.ndarray]:
    """
    Converts a list of land-use classes (or other categorical data) to an integer encoded version based on the unique
    elements.

    ::: warning Note
    It is generally not necessary to utilise this function directly. It will be called implicitly if calculating
    land-use metrics.
    :::

    Parameters
    ----------
    classes
        A `list`, `tuple` or `numpy` array of classes to be encoded.

    Returns
    -------
    tuple
        A `tuple` of unique class descriptors extracted from the input classes.
    np.ndarray
        A `numpy` array of the encoded classes. The value of the `int` encoding will correspond to the order of the
        `class_descriptors`.

    Notes
    -----
    ```python
    from cityseer.metrics import layers

    classes = ['cat', 'dog', 'cat', 'owl', 'dog']

    class_descriptors, class_encodings = layers.encode_categorical(classes)
    print(class_descriptors)  # prints: ('cat', 'dog', 'owl')
    print(list(class_encodings))  # prints: [0, 1, 0, 2, 1]
    ```
    """
    if not isinstance(classes, (list, tuple, np.ndarray)):
        raise TypeError('This method requires an iterable object.')

    # use sklearn's label encoder
    le = LabelEncoder()
    le.fit(classes)
    # map the int encodings to the respective classes
    classes_int = le.transform(classes)

    return tuple(le.classes_), classes_int


def data_map_from_dict(data_dict: dict) -> Tuple[tuple, np.ndarray]:
    """
    Converts a data dictionary into a `numpy` array for use by `DataLayer` classes.

    ::: warning Note
    It is generally not necessary to use this function directly. This function will be called implicitly when invoking [DataLayerFromDict](#class-datalayerfromdict)
    :::

    Parameters
    ----------
    data_dict
        A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary with `x` and `y` key-value pairs. The coordinates must be in a projected coordinate system matching that of the [`NetworkLayer`](/metrics/networks#class-networklayer) to which the data will be assigned.

        ```python
        example_data_dict = {
            'uid_01': {
                'x': 6000956.463188213,
                'y': 600693.4059810264
            },
            'uid_02': {
                'x': 6000753.336609659,
                'y': 600758.7916663144
            }
        }
        ```

    Returns
    -------
    tuple
        A tuple of data `uids` corresponding to the data point identifiers in the source `data_dict`.
    np.ndarray
        A 2d numpy array representing the data points. The indices of the second dimension correspond as follows:

        | idx | property |
        |-----|:----------|
        | 0   | `x` coordinate |
        | 1   | `y` coordinate |
        | 2   | assigned network index - nearest |
        | 3   | assigned network index - next-nearest |

        The arrays at indices `2` and `3` will be initialised with `np.nan`. These will be populated when the [DataLayer.assign_to_network](#datalayer-assign-to-network) method is invoked.
    """
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


class DataLayer:
    """
    Categorical data, such as land-use classifications and numerical data, can be assigned to the network as a
    [`DataLayer`](/metrics/layers#class-datalayer). A `DataLayer` represents the spatial locations of data points, and
    can be used to calculate various mixed-use, land-use accessibility, and statistical measures. Importantly, these
    measures are computed directly over the street network and offer distance-weighted variants; the combination of
    which, makes them more contextually sensitive than methods otherwise based on simpler crow-flies aggregation
    methods.

    The coordinates of data points should correspond as precisely as possible to the location of the feature in space;
    or, in the case of buildings, should ideally correspond to the location of the building entrance.

    Note that in many cases, the [`DataLayerFromDict`](#class-#datalayerfromdict) class will provide a more convenient
    alternative for instantiating this class.
    """
    def __init__(self,
                 data_uids: Union[list, tuple],
                 data_map: np.ndarray):
        """


        Parameters
        ----------
        data_uids
            A `list` or `tuple` of data identifiers corresponding to each data point. This list must be in the same
            order and of the same length as the `data_map`.
        data_map
            A 2d `numpy` array representing the data points. The length of the first dimension should match that of the
            `data_uids`. The indices of the second dimension correspond as follows:

            | idx | property |
            |-----|:----------|
            | 0 | `x` coordinate |
            | 1 | `y` coordinate |
            | 2 | assigned network index - nearest |
            | 3 | assigned network index - next-nearest |

            The arrays at indices `2` and `3` will be populated when the
            [DataLayer.assign_to_network](#datalayer-assign-to-network) method is invoked.

        Returns
        -------
        DataLayer
            Returns a `DataLayer`.

        Properties
        ----------
        """

        self._uids = data_uids  # original labels / indices for each data point
        self._data = data_map  # data map per above
        self._Network = None

        # checks
        checks.check_data_map(self._data, check_assigned=False)

        if len(self._uids) != len(self._data):
            raise ValueError('The number of data labels does not match the number of data points.')

    @property
    def uids(self):
        """Unique ids corresponding to each location in the data_map."""
        return self._uids

    @property
    def data_x_arr(self):
        return self._data[:, 0]

    @property
    def data_y_arr(self):
        return self._data[:, 1]

    @property
    def Network(self):
        return self._Network

    def assign_to_network(self,
                          Network_Layer: networks.NetworkLayer,
                          max_dist: Union[int, float]):
        """
        Once created, a [`DataLayer`](#class-datalayer) should be assigned to a [`NetworkLayer`](/metrics/networks#class-networklayer). The
        `NetworkLayer` provides the backbone for the localised spatial aggregation of data points over the street
        network. The measures will be computed over the same distance thresholds as used for the `NetworkLayer`.

        The data points will be assigned to the two closest network nodes — one in either direction — based on the
        closest adjacent street edge. This enables a dynamic spatial aggregation method that more accurately describes
        distances over the network to data points, relative to the direction of approach.

        Parameters
        ----------
        Network_Layer
            A [`NetworkLayer`](/metrics/networks#class-networklayer).
        max_dist
            The maximum distance to consider when assigning respective data points to the nearest adjacent network
            nodes.

        Notes
        -----
        ::: tip Hint
        The `max_dist` parameter should not be set too small. There are two steps in the assignment process: the first,
        identifies the closest street node; the second, sets-out from this node and attempts to wind around the data
        point — akin to circling the block. It will then review the discovered graph edges from which it is able to
        identify the closest adjacent street-front. The `max_dist` parameter sets a crow-flies distance limit on how far
        the algorithm will search in its attempts to encircle the data point. If the `max_dist` is too small, then the
        algorithm is potentially hampered from finding a starting node; or, if a node is found, may have to terminate
        exploration prematurely because it can't travel far enough away from the data point to explore the surrounding
        network. If too many data points are not being successfully assigned to the correct street edges, then this
        distance should be increased. Conversely, if most of the data points are satisfactorily assigned, then it may be
        possible to decrease this threshold. A distance of $\approx 400m$ provides a good starting point.
        :::

        ::: warning Note
        The precision of assignment improves on decomposed networks (see
        [graphs.nX_decompose](/tools/graphs.html#nx-decompose)), which offers the additional benefit of a more granular
        representation of variations in metrics along street-fronts.
        :::

        ![Example assignment of data to a network](../.vitepress/plots/images/assignment.png)
        _Example assignment on a non-decomposed graph._

        ![Example assignment of data to a network](../.vitepress/plots/images/assignment_decomposed.png)
        _Assignment of data to network nodes becomes more contextually precise on decomposed graphs._
        """
        self._Network = Network_Layer
        data.assign_to_network(self._data,
                               self.Network._node_data,
                               self.Network._edge_data,
                               self.Network._node_edge_map,
                               max_dist,
                               suppress_progress=checks.quiet_mode)

    def compute_aggregated(self,
                           landuse_labels: Union[list, tuple, np.ndarray] = None,
                           mixed_use_keys: Union[list, tuple] = None,
                           accessibility_keys: Union[list, tuple] = None,
                           cl_disparity_wt_matrix: Union[list, tuple, np.ndarray] = None,
                           qs: Union[list, tuple, np.ndarray] = None,
                           stats_keys: Union[list, tuple] = None,
                           stats_data_arrs: Union[List[Union[list, tuple, np.ndarray]],
                                                  Tuple[Union[list, tuple, np.ndarray]],
                                                  np.ndarray] = None,
                           angular: bool = False):
        """
        This method wraps the underlying `numba` optimised functions for aggregating and computing various mixed-use,
        land-use accessibility, and statistical measures. These are computed simultaneously for any required
        combinations of measures (and distances), which can have significant speed implications. Situations requiring
        only a single measure can instead make use of the simplified [`DataLayer.hill_diversity`](#datalayer-hill-diversity),
        [`DataLayer.hill_branch_wt_diversity`](#datalayer-hill-branch-wt-diversity),
        [`DataLayer.compute_accessibilities`](#datalayer-compute-accessibilities),
        [`DataLayer.compute_stats_single`](#datalayer-compute-stats-single), and
        [`DataLayer.compute_stats_multiple`](#datalayer-compute-stats-multiple) methods.

        The data is aggregated and computed over the street network relative to the `Network Layer` nodes, with the
        implication that mixed-use, accessibility, and statistical aggregations are generated from the same locations as
        for centrality computations, which can therefore be correlated or otherwise compared. The outputs of the
        calculations are written to the corresponding node indices in the same `NetworkLayer.metrics` dictionary used
        for centrality methods, and will be categorised by the respective keys and parameters.

        For example, if `hill` and `shannon` mixed-use keys; `shops` and `factories` accessibility keys; and a
        `valuations` stats keys are computed on a `Network Layer` instantiated with $800m$ and $1600m$ distance
        thresholds, then the dictionary would assume the following structure:

        ```python
        NetworkLayer.metrics = {
            'mixed_uses': {
                # note that hill measures have q keys
                'hill': {
                    # here, q=0
                    0: {
                        800: [...],
                        1600: [...]
                    },
                    # here, q=1
                    1: {
                        800: [...],
                        1600: [...]
                    }
                },
                # non-hill measures do not have q keys
                'shannon': {
                    800: [...],
                    1600: [...]
                }
            },
            'accessibility': {
                # accessibility keys are computed in both weighted and unweighted forms
                'weighted': {
                    'shops': {
                        800: [...],
                        1600: [...]
                    },
                    'factories': {
                        800: [...],
                        1600: [...]
                    }
                },
                'non_weighted': {
                    'shops': {
                        800: [...],
                        1600: [...]
                    },
                    'factories': {
                        800: [...],
                        1600: [...]
                    }
                }
            },
            'stats': {
                # stats grouped by each stats key
                'valuations': {
                    # each stat will have the following key-value pairs
                    'max': {
                        800: [...],
                        1600: [...]
                    },
                    'min': {
                        800: [...],
                        1600: [...]
                    },
                    'sum': {
                        800: [...],
                        1600: [...]
                    },
                    'sum_weighted': {
                        800: [...],
                        1600: [...]
                    },
                    'mean': {
                        800: [...],
                        1600: [...]
                    },
                    'mean_weighted': {
                        800: [...],
                        1600: [...]
                    },
                    'variance': {
                        800: [...],
                        1600: [...]
                    },
                    'variance_weighted': {
                        800: [...],
                        1600: [...]
                    }
                }
            }
        }
        ```

        Parameters
        ----------
        landuse_labels
            A set of land-use labels corresponding to the length and order of the data points. The labels should
            correspond to descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only
            required if computing mixed-uses or land-use accessibilities, by default None.
        mixed_use_keys
            An optional list of strings describing which mixed-use metrics to compute, containing any combination of
            `key` values from the following table, by default None. See **Notes** for additional information.
        accessibility_keys
            An optional `list` or `tuple` of land-use classifications for which to calculate accessibilities. The keys
            should be selected from the same land-use schema used for the `landuse_labels` parameter, e.g. "retail". The
            calculations will be performed in both `weighted` and `non_weighted` variants, by default None.
        cl_disparity_wt_matrix
            A pairwise `NxN` disparity matrix numerically describing the degree of disparity between any pair of
            distinct land-uses. This parameter is only required if computing mixed-uses using `hill_pairwise_disparity`
            or `raos_pairwise_disparity`.  The number and order of land-uses should match those implicitly generated by
            [`encode_categorical`](#encode-categorical), by default None.
        qs
            The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of
            the Hill diversity mixed-use measures, by default None.
        stats_keys
            A `list` or `tuple` of keys corresponding to the number of nested arrays passed to the `stats_data_arrs`
            parameter. The computed stats will be saved to the `N.metrics` dictionary using these keys. This parameter
            is only required if computing stats for a `stats_data_arrs` parameter, by default None
        stats_data_arrs
            A 2d `list`, `tuple` or `numpy` array of numerical data, where the first dimension corresponds to the number
            of keys in the `stats_keys` parameter and the second dimension corresponds to number of data points in the
            `DataLayer`, by default None. e.g:
            ```python
            # for a DataLayer containg 5 data points
            stats_keys = ['valuations', 'floors', 'occupants']
            stats_data_arrs = [
                [50000, 60000, 55000, 42000, 46000],  # valuations
                [3, 3, 2, 3, 5],  # floors
                [420, 300, 220, 250, 600]  # occupants
            ]
            ```
        angular
            Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations
            and distances, by default False

        Notes
        -----
        | key | formula | notes |
        |-----|:-------:|-------|
        | hill | $\scriptstyle\big(\sum_{i}^{S}p_{i}^q\big)^{1/(1-q)}\ q\geq0,\ q\neq1 \\ \scriptstyle lim_{q\to1}\
        exp\big(-\sum_{i}^{S}\ p_{i}\ log\ p_{i}\big)$ | Hill diversity: this is the preferred form of diversity
        metric because it adheres to the replication principle and uses units of effective species instead of measures
        of information or uncertainty. The `q` parameter controls the degree of emphasis on the _richness_ of species as
        opposed to the _balance_ of species. Over-emphasis on balance can be misleading in an urban context, for which
        reason research finds support for using `q=0`: this reduces to a simple count of distinct land-uses.|
        | hill_branch_wt | $\scriptstyle\big[\sum_{i}^{S}d_{i}\big(\frac{p_{i}}{\bar{T}}\big)^{q} \big]^{1/(1-q)} \\
        \scriptstyle\bar{T} = \sum_{i}^{S}d_{i}p_{i}$ | This is a distance-weighted variant of Hill Diversity based
        on the distances from the point of computation to the nearest example of a particular land-use. It therefore
        gives a locally representative indication of the intensity of mixed-uses. $d_{i}$ is a negative exponential
        function where $-\beta$ controls the strength of the decay. ($-\beta$ is provided by the `Network Layer`, see
        [`distance_from_beta`](/metrics/networks#distance-from-beta).)|
        | hill_pairwise_wt | $\scriptstyle\big[ \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} \big(  \frac{p_{i} p_{j}}{Q}
        \big)^{q} \big]^{1/(1-q)} \\ \scriptstyle Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$ | This is a
        pairwise-distance-weighted variant of Hill Diversity based on the respective distances between the closest
        examples of the pairwise distinct land-use combinations as routed through the point of computation.
        $d_{ij}$ represents a negative exponential function where $-\beta$ controls the strength of the decay.
        ($-\beta$ is provided by the `Network Layer`, see
        [`distance_from_beta`](/metrics/networks#distance-from-beta).)|
        | hill_pairwise_disparity | $\scriptstyle\big[ \sum_{i}^{S} \sum_{j\neq{i}}^{S} w_{ij} \big(  \frac{p_{i}
        p_{j}}{Q} \big)^{q} \big]^{1/(1-q)} \\ \scriptstyle Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} w_{ij} p_{i}
        p_{j}$ | This is a disparity-weighted variant of Hill Diversity based on the pairwise disparities between
        land-uses. This variant requires the use of a disparity matrix provided through the `cl_disparity_wt_matrix`
        parameter.|
        | shannon | $\scriptstyle -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$ | Shannon diversity (or_information entropy_) is
        one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is
        effectively a transformation of Shannon diversity into units of effective species.|
        | gini_simpson | $\scriptstyle 1 - \sum_{i}^{S} p_{i}^2$ | Gini-Simpson is another classic diversity index.
        It can behave problematically because it does not adhere to the replication principle and places emphasis on the
        balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an
        emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a
        transformation of Gini-Simpson diversity into units of effective species.|
        | raos_pairwise_disparity | $\scriptstyle \sum_{i}^{S} \sum_{j \neq{i}}^{S} d_{ij} p_{i} p_{j}$ | Rao diversity
        is a pairwise disparity measure and requires the use of a disparity matrix provided through the
        `cl_disparity_wt_matrix` parameter. It suffers from the same issues as Gini-Simpson. It is preferable to use
        disparity weighted Hill diversity with `q=2`.|

        ::: tip Hint
        The available choices of land-use diversity measures may seem overwhelming. `hill_branch_wt` paired with `q=0`
        is generally the best choice for granular landuse data, or else `q=1` or `q=2` for increasingly crude landuse
        classifications schemas.
        :::

        A worked example:
        ```python
        from cityseer.metrics import networks, layers
        from cityseer.tools import mock, graphs

        # prepare a mock graph
        G = mock.mock_graph()
        G = graphs.nX_simple_geoms(G)

        # generate the network layer
        N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

        # prepare a mock data dictionary
        data_dict = mock.mock_data_dict(G, random_seed=25)
        # prepare some mock land-use classifications
        landuses = mock.mock_categorical_data(len(data_dict), random_seed=25)
        # let's also prepare some numerical data
        stats_data = mock.mock_numerical_data(len(data_dict), num_arrs=1, random_seed=25)

        # generate a data layer
        L = layers.DataLayerFromDict(data_dict)
        # assign to the network
        L.assign_to_network(N, max_dist=500)
        # compute some metrics
        L.compute_aggregated(landuse_labels=landuses,
                            mixed_use_keys=['hill'],
                            qs=[0, 1],
                            accessibility_keys=['c', 'd', 'e'],
                            stats_keys=['mock_stat'],
                            stats_data_arrs=stats_data)
        # note that the above measures can be run individually using simplified interfaces, e.g.
        # L.hill_diversity(landuses, [0])
        # L.compute_accessibilities(landuses, ['a', 'b'])
        L.compute_stats_single('mock_stat', stats_data[0])  # this method requires a 1d array

        # let's prepare some keys for accessing the computational outputs
        # distance idx: any of the distances with which the NetworkLayer was initialised
        distance_idx = 200
        # q index: any of the invoked q parameters
        q_idx = 0
        # a node idx
        node_idx = 0

        # the data is available at N.metrics
        print(N.metrics['mixed_uses']['hill'][q_idx][distance_idx][node_idx])
        # prints: 4.0
        print(N.metrics['accessibility']['weighted']['d'][distance_idx][node_idx])
        # prints: 0.019168843947614676
        print(N.metrics['accessibility']['non_weighted']['d'][distance_idx][node_idx])
        # prints: 1.0
        print(N.metrics['stats']['mock_stat']['mean_weighted'][distance_idx][node_idx])
        # prints: 71297.82967202332
        ```

        Note that the data can also be unpacked to a dictionary using [`NetworkLayer.metrics_to_dict`](/metrics/networks#networklayer-metrics-to-dict), or transposed to a `networkX` graph using [`NetworkLayer.to_networkX`](/metrics/networks#networklayer-to-networkx).

        ::: danger Caution
        Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that has been used. Meaningful comparisons from one location to another are only possible where the same schemas have been applied.
        :::
        """
        if self.Network is None:
            raise ValueError('Assign this data layer to a network prior to computing mixed-uses or accessibilities.')

        mixed_uses_options = ['hill',
                              'hill_branch_wt',
                              'hill_pairwise_wt',
                              'hill_pairwise_disparity',
                              'shannon',
                              'gini_simpson',
                              'raos_pairwise_disparity']

        if (stats_keys is not None and stats_data_arrs is None) \
                or (stats_keys is None and stats_data_arrs is not None) \
                or (stats_keys is not None and stats_data_arrs is not None and
                    len(stats_data_arrs) != len(stats_keys)):
            raise ValueError('An equal number of stats labels and stats data arrays is required.')

        if stats_data_arrs is None:
            stats_data_arrs = np.full((0, 0), np.nan)
        elif not isinstance(stats_data_arrs, (list, tuple, np.ndarray)):
            raise ValueError('Stats data must be in the form of a list, tuple, or numpy array.')
        else:
            stats_data_arrs = np.array(stats_data_arrs)
            if stats_data_arrs.ndim == 1:
                stats_data_arrs = np.array([stats_data_arrs])
            if stats_data_arrs.shape[1] != len(self._data):
                raise ValueError('The length of all data arrays must match the number of data points.')

        if landuse_labels is None:
            if mixed_use_keys or accessibility_keys:
                raise ValueError(
                    'Computation of mixed use or accessibility measures requires the landuse_labels parameter.')

            landuse_classes = ()
            landuse_encodings = ()
            qs = ()
            mu_hill_keys = ()
            mu_other_keys = ()
            acc_keys = ()
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
                cl_disparity_wt_matrix = np.full((0, 0), np.nan)
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
            mu_hill_keys = []
            mu_other_keys = []
            if mixed_use_keys is not None:
                for mu in mixed_use_keys:
                    if mu not in mixed_uses_options:
                        raise ValueError(
                            f'Invalid mixed-use option: {mu}. Must be one of {", ".join(mixed_uses_options)}.')
                    idx = mixed_uses_options.index(mu)
                    if idx < 4:
                        mu_hill_keys.append(idx)
                    else:
                        mu_other_keys.append(idx - 4)
                if not checks.quiet_mode:
                    logger.info(f'Computing mixed-use measures: {", ".join(mixed_use_keys)}')

            acc_keys = []
            if accessibility_keys is not None:
                for ac_label in accessibility_keys:
                    if ac_label not in landuse_classes:
                        logger.warning(f'No instances of accessibility label: {ac_label} present in the data.')
                    else:
                        acc_keys.append(landuse_classes.index(ac_label))
                if not checks.quiet_mode:
                    logger.info(f'Computing land-use accessibility for: {", ".join(accessibility_keys)}')

        if stats_keys is not None and not checks.quiet_mode:
            logger.info(f'Computing stats for: {", ".join(stats_keys)}')

        # call the underlying method
        mixed_use_hill_data, mixed_use_other_data, \
        accessibility_data, accessibility_data_wt, \
        stats_sum, stats_sum_wt, \
        stats_mean, stats_mean_wt, \
        stats_variance, stats_variance_wt, \
        stats_max, stats_min = data.local_aggregator(self.Network._node_data,
                                                     self.Network._edge_data,
                                                     self.Network._node_edge_map,
                                                     self._data,
                                                     distances=np.array(self.Network.distances),
                                                     betas=np.array(self.Network.betas),
                                                     landuse_encodings=np.array(landuse_encodings),
                                                     qs=np.array(qs),
                                                     mixed_use_hill_keys=np.array(mu_hill_keys),
                                                     mixed_use_other_keys=np.array(mu_other_keys),
                                                     accessibility_keys=np.array(acc_keys),
                                                     cl_disparity_wt_matrix=np.array(cl_disparity_wt_matrix),
                                                     numerical_arrays=stats_data_arrs,
                                                     angular=angular,
                                                     suppress_progress=checks.quiet_mode)

        # write the results to the Network's metrics dict
        # keys will check for pre-existing, whereas qs and distance keys will overwrite
        # unpack mixed use hill
        for mu_h_idx, mu_h_key in enumerate(mu_hill_keys):
            mu_h_label = mixed_uses_options[mu_h_key]
            if mu_h_label not in self.Network.metrics['mixed_uses']:
                self.Network.metrics['mixed_uses'][mu_h_label] = {}
            for q_idx, q_key in enumerate(qs):
                self.Network.metrics['mixed_uses'][mu_h_label][q_key] = {}
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['mixed_uses'][mu_h_label][q_key][d_key] = \
                        mixed_use_hill_data[mu_h_idx][q_idx][d_idx]

        # unpack mixed use other
        for mu_o_idx, mu_o_key in enumerate(mu_other_keys):
            mu_o_label = mixed_uses_options[mu_o_key + 4]
            if mu_o_label not in self.Network.metrics['mixed_uses']:
                self.Network.metrics['mixed_uses'][mu_o_label] = {}
            # no qs
            for d_idx, d_key in enumerate(self.Network.distances):
                self.Network.metrics['mixed_uses'][mu_o_label][d_key] = mixed_use_other_data[mu_o_idx][d_idx]

        # unpack accessibility data
        for ac_idx, ac_code in enumerate(acc_keys):
            ac_label = landuse_classes[ac_code]  # ac_code is index of ac_label
            for k, ac_data in zip(['non_weighted', 'weighted'], [accessibility_data, accessibility_data_wt]):
                if ac_label not in self.Network.metrics['accessibility'][k]:
                    self.Network.metrics['accessibility'][k][ac_label] = {}
                for d_idx, d_key in enumerate(self.Network.distances):
                    self.Network.metrics['accessibility'][k][ac_label][d_key] = ac_data[ac_idx][d_idx]

        # unpack the numerical arrays
        if stats_keys:
            for num_idx, stats_key in enumerate(stats_keys):
                if stats_key not in self.Network.metrics['stats']:
                    self.Network.metrics['stats'][stats_key] = {}
                for k, stats_data in zip(['max',
                                          'min',
                                          'sum',
                                          'sum_weighted',
                                          'mean',
                                          'mean_weighted',
                                          'variance',
                                          'variance_weighted'],
                                         [stats_max,
                                          stats_min,
                                          stats_sum,
                                          stats_sum_wt,
                                          stats_mean,
                                          stats_mean_wt,
                                          stats_variance,
                                          stats_variance_wt]):
                    if k not in self.Network.metrics['stats'][stats_key]:
                        self.Network.metrics['stats'][stats_key][k] = {}
                    for d_idx, d_key in enumerate(self.Network.distances):
                        self.Network.metrics['stats'][stats_key][k][d_key] = stats_data[num_idx][d_idx]

    def hill_diversity(self,
                       landuse_labels: Union[list, tuple, np.ndarray],
                       qs: Union[list, tuple, np.ndarray] = None):
        """
        Compute hill diversity for the provided `landuse_labels` at the specified values of `q`. See
        [`DataLayer.compute_aggregated`](#datalayer-compute-aggregated) for additional information.

        Parameters
        ----------
        landuse_labels
            A set of land-use labels corresponding to the length and order of the data points. The labels should
            correspond to descriptors from the land-use schema, such as "retail" or "commercial".
        qs
            The values of `q` for which to compute Hill diversity, by default None

        Notes
        -----
        The data key is `hill`, e.g.:

        `NetworkLayer.metrics['mixed_uses']['hill'][<<q key>>][<<distance key>>][<<node idx>>]`
        """
        return self.compute_aggregated(landuse_labels, mixed_use_keys=['hill'], qs=qs)

    def hill_branch_wt_diversity(self,
                                 landuse_labels: Union[list, tuple, np.ndarray],
                                 qs: Union[list, tuple, np.ndarray] = None):
        """
        Compute distance-weighted hill diversity for the provided `landuse_labels` at the specified values of `q`. See
        [`DataLayer.compute_aggregated`](#datalayer-compute-aggregated) for additional information.

        Parameters
        ----------
        landuse_labels
            A set of land-use labels corresponding to the length and order of the data points. The labels should
            correspond to descriptors from the land-use schema, such as "retail" or "commercial".
        qs
            The values of `q` for which to compute Hill diversity, by default None

        Notes
        -----
        The data key is `hill_branch_wt`, e.g.:

        `NetworkLayer.metrics['mixed_uses']['hill_branch_wt'][<<q key>>][<<distance key>>][<<node idx>>]`
        """
        return self.compute_aggregated(landuse_labels, mixed_use_keys=['hill_branch_wt'], qs=qs)

    def compute_accessibilities(self,
                                landuse_labels: Union[list, tuple, np.ndarray],
                                accessibility_keys: Union[list, tuple]):
        """
        Compute land-use accessibilities for the specified land-use classification keys. See
        [`DataLayer.compute_aggregated`](#datalayer-compute-aggregated) for additional information.

        Parameters
        ----------
        landuse_labels
            A set of land-use labels corresponding to the length and order of the data points. The labels should
            correspond to descriptors from the land-use schema, such as "retail" or "commercial".
        accessibility_keys
            The land-use keys for which to compute accessibilies. The keys should be selected from the same land-use
            schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both
            `weighted` and `non_weighted` variants.

        Notes
        -----
        The data keys will correspond to the `accessibility_keys` specified, e.g. where computing `retail`
        accessibility:

        `NetworkLayer.metrics['accessibility']['weighted']['retail'][<<distance key>>][<<node idx>>]`<br>
        `NetworkLayer.metrics['accessibility']['non_weighted']['retail'][<<distance key>>][<<node idx>>]`
        """
        return self.compute_aggregated(landuse_labels, accessibility_keys=accessibility_keys)

    def compute_stats_single(self,
                             stats_key: str,
                             stats_data_arr: Union[list, tuple, np.ndarray]):
        """
        Compute stats for a single `stats_key` parameter. As with the landuse and mixed-use measures: stats will be
        computed for each distance that was specified when initialising the `NetworkLayer`.

        Parameters
        ----------
        stats_key
            A `str` key describing the stats computed for the `stats_data_arr` parameter. The computed stats will be
            saved to the `N.metrics` dictionary under this key.
        stats_data_arr
            A 1d `list`, `tuple` or `numpy` array of numerical data, where the length corresponds to the number of data
            points in the `DataLayer`.
        
        Notes
        -----
        The data key will correspond to the `stats_key` parameter, e.g. where using `occupants` as the key:
        
        `NetworkLayer.metrics['stats']['occupants'][<<stat type>>][<<distance key>>][<<node idx>>]`

        ::: tip Hint
        Per the above worked example, the following stat types will be available for each `stats_key` for each of the computed distances:
        - `max` and `min`
        - `sum` and `sum_weighted`
        - `mean` and `mean_weighted`
        - `variance` and `variance_weighted`
        :::
        """
        if stats_data_arr.ndim != 1:
            raise ValueError(
                'The stats_data_arr must be a single dimensional array with a length corresponding to the number of data points in the DataLayer.')
        return self.compute_aggregated(stats_keys=[stats_key], stats_data_arrs=[stats_data_arr])

    def compute_stats_multiple(self,
                               stats_keys: Union[list, tuple],
                               stats_data_arrs: Union[list, tuple, np.ndarray]):
        """
        

        Parameters
        ----------
        stats_keys
            A `list` or `tuple` of keys corresponding to the number of nested arrays passed to the `stats_data_arrs`
            parameter. The computed stats will be saved to the `N.metrics` dictionary under these keys.
        stats_data_arrs
            A 2d `list`, `tuple` or `numpy` array of numerical data, where the first dimension corresponds to the number
            of keys in the `stats_keys` parameter and the second dimension corresponds to number of data points in the
            `DataLayer`. See the below example.

        Notes
        -----
        ```python
        # for a DataLayer containg 5 data points
        stats_keys = ['valuations', 'floors', 'occupants']
        stats_data_arrs = [
            [50000, 60000, 55000, 42000, 46000],  # valuations
            [3, 3, 2, 3, 5],  # floors
            [420, 300, 220, 250, 600]  # occupants
        ]
        ```
        The data keys will correspond to the `stats_keys` parameter:

        `NetworkLayer.metrics['stats']['valuations'][<<stat type>>][<<distance key>>][<<node idx>>]`<br>
        `NetworkLayer.metrics['stats']['floors'][<<stat type>>][<<distance key>>][<<node idx>>]`<br>
        `NetworkLayer.metrics['stats']['occupants'][<<stat type>>][<<distance key>>][<<node idx>>]`
        """

        return self.compute_aggregated(stats_keys=stats_keys, stats_data_arrs=stats_data_arrs)

    def model_singly_constrained(self,
                                 key: str,
                                 i_data_map: np.ndarray,
                                 j_data_map: np.ndarray,
                                 i_weights: Union[list, tuple, np.ndarray],
                                 j_weights: Union[list, tuple, np.ndarray],
                                 angular: bool = False):
        """
        Undocumented method for computing singly-constrained interactions.
        """

        j_assigned, netw_flows = data.singly_constrained(self.Network._node_data,
                                                         self.Network._edge_data,
                                                         self.Network._node_edge_map,
                                                         distances=np.array(self.Network.distances),
                                                         betas=np.array(self.Network.betas),
                                                         i_data_map=i_data_map,
                                                         j_data_map=j_data_map,
                                                         i_weights=np.array(i_weights),
                                                         j_weights=np.array(j_weights),
                                                         angular=angular,
                                                         suppress_progress=checks.quiet_mode)

        # write the results to the Network's metrics dict
        if key not in self.Network.metrics['models']:
            self.Network.metrics['models'][key] = {}
        for d_idx, d_key in enumerate(self.Network.distances):
            if d_key not in self.Network.metrics['models'][key]:
                self.Network.metrics['models'][key][d_key] = {}
            self.Network.metrics['models'][key][d_key]['assigned_trips'] = j_assigned[d_idx]
            self.Network.metrics['models'][key][d_key]['network_flows'] = netw_flows[d_idx]


class DataLayerFromDict(DataLayer):
    """
    Directly transposes an appropriately prepared data dictionary into a `DataLayer`. This `class` calls
    [`data_map_from_dict`](#data-map-from-dict) internally. Methods and properties are inherited from the parent
    [`DataLayer`](#class-datalayer) class, which can be referenced for more information.
    """
    def __init__(self, data_dict: dict) -> DataLayer:
        """
        Parameters
        ----------
        data_dict
            A dictionary representing distinct data points, where each `key` represents a `uid` and each value
            represents a nested dictionary with `x` and `y` key-value pairs. The coordinates must be in a projected
            coordinate system matching that of the
            [`NetworkLayer`](/metrics/networks#class-networklayer) to which the data will
            be assigned.

        Returns
        -------
        DataLayer
            Returns a [`DataLayer`](#class-datalayer).

        Notes
        -----
        Example dictionary:
        ```python
        example_data_dict = {
            'uid_01': {
                'x': 6000956.463188213,
                'y': 600693.4059810264
            },
            'uid_02': {
                'x': 6000753.336609659,
                'y': 600758.7916663144
            }
        }
        ```
        """
        data_uids, data_map = data_map_from_dict(data_dict)

        super().__init__(data_uids, data_map)
