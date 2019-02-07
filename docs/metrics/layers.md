---

---

<RenderMath></RenderMath>


cityseer.metrics.layers
=======================


dict\_wgs\_to\_utm
------------------

<FuncSignature>dict_wgs_to_utm(data_dict)</FuncSignature>

Converts data dictionary `x` and `y` values from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the local UTM projected coordinate system.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="data_dict" type="dict">

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

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="dict" type="dict">

Returns a copy of the source dictionary with the `x` and `y` values converted to the local UTM coordinate system.

</FuncElement>

```python
from cityseer.util import mock
from cityseer.metrics import layers

# let's generate a mock data dictionary
G_wgs = mock.mock_graph(wgs84_coords=True)
# mock_data_dict takes on the same extents on the graph parameter
data_dict_WGS = mock.mock_data_dict(G_wgs, random_seed=25)
# the dictionary now contains wgs coordinates
for i, (key, value) in enumerate(data_dict_WGS.items()):
    print(key, value)
    # prints:
    # 0 {'x': -1.4528142077605772, 'y': 54.145396945391695, 'live': False}
    # 1 {'x': -1.463037735961814, 'y': 54.1411197101806, 'live': False}
    if i == 1:
        break
        
# any data dictionary that follows this template can be passed to dict_wgs_to_utm()
data_dict_UTM = layers.dict_wgs_to_utm(data_dict_WGS)
# the coordinates have now been converted to UTM
for i, (key, value) in enumerate(data_dict_UTM.items()):
    print(key, value)
    # prints:
    # 0 {'x': 6000804.828421302, 'y': 601059.7404369968, 'live': False}
    # 1 {'x': 6000314.41378748, 'y': 600402.3539826316, 'live': False}
    if i == 1:
        break
```


encode\_categorical
-------------------

<FuncSignature>encode_categorical(classes)</FuncSignature>

Converts a list of land-use classes (or other categorical data) to an integer encoded version based on the unique elements.

::: warning Note
It is generally not necessary to utilise this function directly. It will be called implicitly if calculating land-use metrics.
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="classes" type="list, tuple, np.ndarray">

A `list`, `tuple` or `numpy` array of classes to be encoded.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="class_descriptors" type="tuple">

A `tuple` of unique class descriptors extracted from the input classes.

</FuncElement>

<FuncElement name="class_encodings" type="np.ndarray">

A `numpy` array of the encoded classes. The value of the `int` encoding will correspond to the order of the `class_descriptors`.

</FuncElement>

```python
from cityseer.metrics import layers

classes = ['cat', 'dog', 'cat', 'owl', 'dog']

class_descriptors, class_encodings = layers.encode_categorical(classes)
print(class_descriptors)  # prints: ('cat', 'dog', 'owl')
print(list(class_encodings))  # prints: [0, 1, 0, 2, 1]
```


data\_map\_from\_dict
---------------------

<FuncSignature>data_map_from_dict(data_dict)</FuncSignature>

Converts a data dictionary into a `numpy` array for use by `Data_Layer` classes.

::: warning Note
It is generally not necessary to use this function directly. This function will be called implicitly when invoking [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx)
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="data_dict" type="dict">

A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary with `x` and `y` key-value pairs. The coordinates must be in a projected coordinate system matching that of the [`Network_Layer`](http://localhost:8080/cityseer/metrics/networks.html#network-layer) to which the data will be assigned.

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

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="data_uids" type="tuple">

A tuple of data `uids` corresponding to the data point identifiers in the source `data_dict`.

</FuncElement>

<FuncElement name="data_map" type="np.ndarray">

A 2d numpy array representing the data points. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | assigned network index - nearest |
| 3 | assigned network index - next-nearest | 

The arrays at indices `2` and `3` will be initialised with `np.nan`. These will be populated when the [@assign_to_network](#assign-to-network) method is invoked.

</FuncElement>


Data\_Layer <Chip text="class"/>
-----------

<FuncSignature>Data_Layer(data_uids, data_map)</FuncSignature>

Data layers represent the locations of data points. By assigning a `Data_Layer` to a `Network_Layer` it then becomes possible to calculate various mixed-use and land-use accessibility measures, as well as a range of spatially sensitive statistics. These measures are computed directly over the street network, which is more contextually sensitive than methods based on crow-flies aggregation methods.

The coordinates of data points should correspond as precisely as possible to the location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the building entrance.

Note that in many cases, the [`Data_Layer_From_Dict`](#data-layer-from-dict) class will provide a more convenient alternative for instantiating this class.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="data_uids" type="list, tuple">

A `list` or `tuple` of data identifiers corresponding to each data point. This list must be in the same order and of the same length as the `data_map`.

</FuncElement>

<FuncElement name="node_map" type="np.ndarray">

A 2d `numpy` array representing the data points. The length of the first dimension should match that of the `data_uids`. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | assigned network index - nearest |
| 3 | assigned network index - next-nearest | 

The arrays at indices `2` and `3` will be populated when the [@assign_to_network](#assign-to-network) method is invoked.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="Data_Layer" type="class">

A `Data_Layer`.

</FuncElement>


@assign\_to\_network
---------------------

Once created, a [`Data_Layer`](#data-layer) should be assigned to a [`Network_Layer`](#network-layer). The `Network_Layer` provides the backbone for the localised spatial aggregation of data points over the street network. The measures will be computed over the same distance thresholds as used for the `Network_Layer`.

The data points will be assigned to the two closest network nodes — one in either direction — based on the closest adjacent street edge. This enables a dynamic spatial aggregation method that accurately describes distances over the network to data points, relative to the direction of approach.

<FuncSignature>Data_Layer.assign_to_network(Network_Layer, max_dist)</FuncSignature>

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="Network_Layer" type="networks.Network_Layer">

A [`Network_Layer`](#network-layer).

</FuncElement>

<FuncElement name="max_dist" type="int, float">

The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.

</FuncElement>

::: tip Hint
The `max_dist` parameter should not be set too small. There are two steps in the assignment process: the first, identifies the closest street node; the second, sets-out from this node and attempts to wind around the data point — akin to circling the block. It will then review the discovered graph edges from which it is able to identify the closest adjacent street-front. The `max-dist` parameter sets a crow-flies distance limit on how far the algorithm will search in its attempts to encircle the data point. If the `max_dist` is too small, then the algorithm is potentially hampered from finding a starting node; or, if a node is found, may have to terminate exploration prematurely because it can't travel far enough away from the data point to explore the surrounding network. If too many data points are not being successfully assigned to the correct street edges, then this distance should be increased. Conversely, if most of the data points are satisfactorily assigned, then it may be possible to decrease this threshold. A distance of $\approx 500m$ provides a good starting point.
:::

::: warning Note
The precision of assignment improves on decomposed networks (see [graphs.nX_decompose](/util/graphs.html#nx-decompose)), which offers the further benefit of a more granular representation of variations in metrics along street-fronts.
:::

<img src="/plots/assignment.png" alt="Example assignment of data to a network" class="left"><img src="/plots/assignment_decomposed.png" alt="Example assignment on a decomposed network" class="right">

_Assignment of data to network nodes becomes more contextually precise on decomposed graphs (right) as opposed to non-decomposed graphs (left)._

@compute\_aggregated
---------------------

<FuncSignature>
<pre>
Data_Layer.compute_aggregated(landuse_labels,
                              mixed_use_keys,
                              accessibility_keys,
                              cl_disparity_wt_matrix,
                              qs,
                              stats_keys,
                              stats_data_arrs)
</pre>
</FuncSignature>

This method wraps the underlying `numba` optimised functions for aggregating and computing various mixed-use, land-use accessibility, and statistical measures. These are computed simultaneously for any required combinations of measures (and distances), which can have significant speed implications. Situations requiring only a single measure can instead make use of the simplified [`@hill_diversity`](#hill-diversity), [`@hill_branch_wt_diversity`](#hill-branch-wt-diversity), [`@compute_accessibilities`](#compute-accessibilities), [`@compute_stats_single`](#compute-stats-single), and [`@compute_stats_multiple`](#compute-stats-multiple) methods.

The data is aggregated and computed over the street network relative to the `Network Layer` nodes, with the implication that mixed-use, accessibility, and statistical aggregations are generated from the same locations as for centrality computations, which can therefore be correlated or otherwise compared. The outputs of the calculations are written to the corresponding node indices in the same `Network_Layer.metrics` dictionary used for centrality methods, and will be categorised by the respective keys and parameters.

For example, if `hill` and `shannon` mixed-use keys; `shops` and `factories` accessibility keys; and a `valuations` stats keys are computed at $800m$ and $1600m$, then the dictionary would assume the following structure:

```python
Network_Layer.metrics = {
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

A working example:

```python
from cityseer.metrics import networks, layers
from cityseer.util import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_auto_edge_params(G)

# generate the network layer
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600])

# prepare a mock data dictionary
data_dict = mock.mock_data_dict(G, random_seed=25)
# prepare some mock land-use classifications
landuses = mock.mock_categorical_data(len(data_dict), random_seed=25)
# let's also prepare some numerical data
stats_data = mock.mock_numerical_data(len(data_dict), num_arrs=1, random_seed=25)

# generate a data layer
L = layers.Data_Layer_From_Dict(data_dict)
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
# distance idx: any of the distances with which the Network_Layer was initialised
distance_idx = 200
# q index: any of the invoked q parameters
q_idx = 0
# a node idx
node_idx = 0

# the data is available at N.metrics
print(N.metrics['mixed_uses']['hill'][q_idx][distance_idx][node_idx])
# prints: 4.0
print(N.metrics['accessibility']['weighted']['d'][distance_idx][node_idx])
# prints: 0.019168845139732035
print(N.metrics['accessibility']['non_weighted']['d'][distance_idx][node_idx])
# prints: 1.0
print(N.metrics['stats']['mock_stat']['mean_weighted'][distance_idx][node_idx])
# prints: 47135.72553719515
```

Note that the data can also be unpacked to a dictionary using [`Network_Layer.metrics_to_dict`](/metrics/networks.html#metrics-to-dict), or transposed to a `networkX` graph using [`Network_Layer.to_networkX`](/metrics/networks.html#to-networkx).

::: danger Caution

Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that has been used. Meaningful comparisons from one location to another are only possible where the same schemas have been applied.

:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="landuse_labels" type="list, tuple, np.ndarray">

A set of land-use labels corresponding to the length and order of the data points. The labels should correspond to descriptors from the land-use schema, such as "retail" or "commercial". This parameter is only required if computing mixed-uses or land-use accessibilities.

</FuncElement>

<FuncElement name="mixed_use_keys" type="list, tuple">

An optional list of strings describing which mixed-use metrics to compute, containing any combination of `key` values from the following table.

</FuncElement>

| key | formula | notes |
|-----|---------|-------|
| hill | $$\Big(\sum_{i}^{S}p_{i}^q\Big)^{1/(1-q)}\ q\geq0,\ q\neq1$$ $$lim_{q\to1}\ exp\Big(-\sum_{i}^{S}\ p_{i}\ log\ p_{i}\Big)$$ | Hill diversity: this is the preferred form of diversity metric because it adheres to the replication principle and uses units of effective species instead of measures of information or uncertainty. The `q` parameter controls the degree of emphasis on the _richness_ of species as opposed to the _balance_ of species. Over-emphasis on balance can be misleading in an urban context, for which reason research finds support for using `q=0`: this reduces to a simple count of distinct land-uses. |
| hill_branch_wt | $$\Bigg[\sum_{i}^{S}d_{i}\bigg(\frac{p_{i}}{\bar{T}}\bigg)^{q} \Bigg]^{1/(1-q)}$$ $$\bar{T} = \sum_{i}^{S}d_{i}p_{i}$$ | This is a distance-weighted variant of Hill Diversity based on the distances from the point of computation to the nearest example of a particular land-use. It therefore gives a locally representative indication of the intensity of mixed-uses. $d_{i}$ is a negative exponential function where $-\beta$ controls the strength of the decay. ($-\beta$ is provided by the `Network Layer`, see [distance_from_beta](/metrics/networks.html#distance-from-beta).)|
| hill_pairwise_disparity | $$\Bigg[ \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} \bigg(  \frac{p_{i} p_{j}}{Q} \bigg)^{q} \Bigg]^{1/(1-q)}$$ $$Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | This is a pairwise-distance-weighted variant of Hill Diversity based on the respective distances between the closest examples of the pairwise distinct land-use combinations as routed through the point of computation. $d_{ij}$ represents a negative exponential function where $-\beta$ controls the strength of the decay. ($-\beta$ is provided by the `Network Layer`, see [distance_from_beta](/metrics/networks.html#distance-from-beta).) |
| hill_pairwise_disparity | $$\Bigg[ \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} \bigg(  \frac{p_{i} p_{j}}{Q} \bigg)^{q} \Bigg]^{1/(1-q)}$$ $$Q = \sum_{i}^{S} \sum_{j\neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | This is a disparity-weighted variant of Hill Diversity based on the pairwise disparities between land-uses. This variant requires the use of a disparity matrix provided through the `cl_disparity_wt_matrix` parameter. |
| shannon | $$-\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or _information entropy_) is one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is effectively a transformation of Shannon diversity into units of effective species. |
| gini_simpson | $$1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index. It can behave problematically because it does not adhere to the replication principle and places emphasis on the balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a transformation of Gini-Simpson diversity into units of effective species. |
| raos_pairwise_disparity | $$\sum_{i}^{S} \sum_{j \neq{i}}^{S} d_{ij} p_{i} p_{j}$$ | Rao diversity is a pairwise disparity measure and requires the use of a disparity matrix provided through the `cl_disparity_wt_matrix` parameter. It suffers from the same issues as Gini-Simpson. It is preferable to use disparity weighted Hill diversity with `q=2`. |

::: tip Hint
The available choices of land-use diversity measures can seem overwhelming. `hill_branch_wt` paired with `q=0` is generally the best choice unless there is reason to use another.
:::

<FuncElement name="accessibility_keys" type="list, tuple">

An optional `list` or `tuple` of land-use classifications for which to calculate accessibilities. The keys should be selected from the same land-use schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both `weighted` and `non_weighted` variants.

</FuncElement>

<FuncElement name="cl_disparity_wt_matrix" type="list, tuple, np.ndarray">

A pairwise `NxN` disparity matrix numerically describing the degree of disparity between any pair of distinct land-uses. This parameter is only required if computing mixed-uses using `hill_pairwise_disparity` or `raos_pairwise_disparity`.  The number and order of land-uses should match those implicitly generated by [`encode_categorical`](#encode-categorical). 

</FuncElement>

<FuncElement name="qs" type="list, tuple, np.ndarray">

The values of `q` for which to compute Hill diversity. This parameter is only required if computing one of the Hill diversity mixed-use measures. 

</FuncElement>

<FuncElement name="stats_keys" type="list, tuple">

A `list` or `tuple` of keys corresponding to the number of nested arrays passed to the `stats_data_arrs` parameter. The computed stats will be saved to the `N.metrics` dictionary under these keys. This parameter is only required if computing stats for a `stats_data_arrs` parameter.

</FuncElement>

<FuncElement name="stats_data_arrs" type="list, tuple, np.ndarray">

A 2d `list`, `tuple` or `numpy` array of numerical data, where the first dimension corresponds to the number of keys in the `stats_keys` parameter and the second dimension corresponds to number of data points in the `Data_Layer`. See the below example.

</FuncElement>

```python
# for a Data_Layer containg 5 data points
stats_keys = ['valuations', 'floors', 'occupants']
stats_data_arrs = [
    [50000, 60000, 55000, 42000, 46000],  # valuations
    [3, 3, 2, 3, 5],  # floors
    [420, 300, 220, 250, 600]  # occupants
]
```


@hill\_diversity
----------------

<FuncSignature>Data_Layer.hill_diversity(landuse_labels, qs)</FuncSignature>

Compute hill diversity for the provided `landuse_labels` at the specified values of `q`. See [@compute_aggregated](#compute-aggregated) for additional information.

<FuncElement name="landuse_labels" type="list, tuple, np.ndarray">

A set of land-use labels corresponding to the length and order of the data points. The labels should correspond to descriptors from the land-use schema, such as "retail" or "commercial".

</FuncElement>

<FuncElement name="qs" type="list, tuple, np.ndarray">

The values of `q` for which to compute Hill diversity.

</FuncElement>

The data key is `hill`, e.g.:

`Network_Layer.metrics['mixed_uses']['hill'][<<q key>>][<<distance key>>][<<node idx>>]`


@hill\_branch\_wt\_diversity
----------------------------

<FuncSignature>Data_Layer.hill_branch_wt_diversity(landuse_labels, qs)</FuncSignature>

Compute distance-weighted hill diversity for the provided `landuse_labels` at the specified values of `q`. See [@compute_aggregated](#compute-aggregated) for additional information.

<FuncElement name="landuse_labels" type="list, tuple, np.ndarray">

A set of land-use labels corresponding to the length and order of the data points. The labels should correspond to descriptors from the land-use schema, such as "retail" or "commercial".

</FuncElement>

<FuncElement name="qs" type="list, tuple, np.ndarray">

The values of `q` for which to compute Hill diversity.

</FuncElement>

The data key is `hill_branch_wt`, e.g.:

`Network_Layer.metrics['mixed_uses']['hill_branch_wt'][<<q key>>][<<distance key>>][<<node idx>>]`


@compute\_accessibilities
-------------------------

<FuncSignature>
<pre>
Data_Layer.compute_accessibilities(landuse_labels,
                                   accessibility_keys)
</pre>
</FuncSignature>

Compute land-use accessibilities for the specified land-use classification keys. See [@compute_aggregated](#compute-aggregated) for additional information.

<FuncElement name="landuse_labels" type="list, tuple, np.ndarray">

A set of land-use labels corresponding to the length and order of the data points. The labels should correspond to descriptors from the land-use schema, such as "retail" or "commercial".

</FuncElement>

<FuncElement name="accessibility_keys" type="list, tuple">

The land-use keys for which to compute accessibilies. The keys should be selected from the same land-use schema used for the `landuse_labels` parameter, e.g. "retail". The calculations will be performed in both `weighted` and `non_weighted` variants.

</FuncElement>

The data keys will correspond to the `accessibility_keys` specified, e.g. where computing `retail` accessibility:

`Network_Layer.metrics['accessibility']['weighted']['retail'][<<distance key>>][<<node idx>>]`
`Network_Layer.metrics['accessibility']['non_weighted']['retail'][<<distance key>>][<<node idx>>]`


@compute\_stats\_single
-----------------------
<FuncSignature>
<pre>
Data_Layer.compute_stats_single(stats_key, stats_data_arr)
</pre>
</FuncSignature>

Compute stats for a single `stats_key` parameter.

<FuncElement name="stats_key" type="str">

A `str` key describing the stats computed for the `stats_data_arr` parameter. The computed stats will be saved to the `N.metrics` dictionary under this key.

</FuncElement>

<FuncElement name="stats_data_arr" type="list, tuple, np.ndarray">

A 1d `list`, `tuple` or `numpy` array of numerical data, where the length corresponds to the number of data points in the `Data_Layer`.

</FuncElement>

The data key will correspond to the `stats_key` parameter, e.g. where using `occupants` as the key:

`Network_Layer.metrics['stats']['occupants'][<<stat type>>][<<distance key>>][<<node idx>>]`


@compute\_stats\_multiple
-------------------------

<FuncSignature>
<pre>
Data_Layer.compute_stats_multiple(stats_keys, stats_data_arr)
</pre>
</FuncSignature>

<FuncElement name="stats_keys" type="list, tuple">

A `list` or `tuple` of keys corresponding to the number of nested arrays passed to the `stats_data_arrs` parameter. The computed stats will be saved to the `N.metrics` dictionary under these keys.

</FuncElement>

<FuncElement name="stats_data_arrs" type="list, tuple, np.ndarray">

A 2d `list`, `tuple` or `numpy` array of numerical data, where the first dimension corresponds to the number of keys in the `stats_keys` parameter and the second dimension corresponds to number of data points in the `Data_Layer`. See the below example.

</FuncElement>

```python
# for a Data_Layer containg 5 data points
stats_keys = ['valuations', 'floors', 'occupants']
stats_data_arrs = [
    [50000, 60000, 55000, 42000, 46000],  # valuations
    [3, 3, 2, 3, 5],  # floors
    [420, 300, 220, 250, 600]  # occupants
]
```

The data keys will correspond to the `stats_keys` parameter:

`Network_Layer.metrics['stats']['valuations'][<<stat type>>][<<distance key>>][<<node idx>>]`
`Network_Layer.metrics['stats']['floors'][<<stat type>>][<<distance key>>][<<node idx>>]`
`Network_Layer.metrics['stats']['occupants'][<<stat type>>][<<distance key>>][<<node idx>>]`


Data_Layer_From_Dict <Chip text="class"/>
--------------------

<FuncSignature>Data_Layer_From_Dict(data_dict)</FuncSignature>

Directly transposes an appropriately prepared data dictionary into a `Data_Layer`. This `class` calls [`data_map_from_dict`](#data-map-from-dict) internally. Methods and properties are inherited from the parent [`Data_Layer`](#data-layer) class, which can be referenced for more information.

<FuncElement name="data_dict" type="dict">

A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary with `x` and `y` key-value pairs. The coordinates must be in a projected coordinate system matching that of the [`Network_Layer`](http://localhost:8080/cityseer/metrics/networks.html#network-layer) to which the data will be assigned.

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

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="Data_Layer" type="class">

A [`Data_Layer`](#data-layer).

</FuncElement>