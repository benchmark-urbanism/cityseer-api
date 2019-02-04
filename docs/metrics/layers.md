---

---

<RenderMath></RenderMath>


cityseer.metrics.layers
=======================


dict\_wgs\_to\_utm
------------------

<FuncSignature>dict_wgs_to_utm(data_dict)</FuncSignature>

Converts a data dictionary's `x` and `y` values from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the local UTM projected coordinate system.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="data_dict" type="dict">

A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary of `key-value` pairs consisting of `x` and `y` coordinate attributes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="dict" type="dict">

Returns a copy of the dictionary with the `x` and `y` values converted to the local UTM coordinate system.

</FuncElement>

```python
from cityseer.util import mock
from cityseer.metrics import layers

# let's generate a mock data dictionary
G_wgs = mock.mock_graph(wgs84_coords=True)
# mock_data_dict takes on the same extents on the graph parameter
data_dict_WGS = mock.mock_data_dict(G_wgs, random_seed=0)
# the dictionary now contains wgs coordinates
for i, (key, value) in enumerate(data_dict_WGS.items()):
    print(key, value)
    # prints:
    # 0 {'x': -1.458369781174891, 'y': 54.14683122127234}
    # 1 {'x': -1.457436968983548, 'y': 54.144993424483964}
    if i == 1:
        break
        
# any data dictionary that follows this template can be passed to dict_wgs_to_utm()
data_dict_UTM = layers.dict_wgs_to_utm(data_dict_WGS)
# the coordinates have now been converted to UTM
for i, (key, value) in enumerate(data_dict_UTM.items()):
    print(key, value)
    # prints:
    # 0 {'x': 6000956.463188213, 'y': 600693.4059810264}
    # 1 {'x': 6000753.336609659, 'y': 600758.7916663144}
    if i == 1:
        break
```


encode\_categorical
-------------------

<FuncSignature>encode_categorical(classes)</FuncSignature>

::: warning Note
It is generally not necessary to utilise this method directly. It will be called internally, if necessary, when calculating land-use metrics.
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
It is generally not necessary to use this method directly. This method will be called internally when invoking [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx)
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="data_dict" type="dict">

A dictionary representing distinct data points, where each `key` represents a `uid` and each value represents a nested dictionary of `key-value` pairs consisting of `x` and `y` coordinate attributes. The coordinates must be in a projected coordinate system matching that of the [`Network_Layer`](http://localhost:8080/cityseer/metrics/networks.html#network-layer) to which the data will be assigned.

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

The arrays at indices `2` and `3` will be initialised with `np.nan`.

</FuncElement>


Data\_Layer <Chip text="class"/>
-----------

<FuncSignature>Data_Layer(data_uids, data_map)</FuncSignature>

Data layers represent 


@assign\_to\_network
---------------------

<FuncSignature>Data_Layer.assign_to_network(Network_Layer, max_dist)</FuncSignature>


@compute\_aggregated
---------------------

<FuncSignature>
<pre>
Data_Layer.compute_aggregated(landuse_labels,
                              mixed_use_metrics,
                              accessibility_labels,
                              cl_disparity_wt_matrix,
                              qs,
                              numerical_labels,
                              numerical_arrays)
</pre>
</FuncSignature>


@hill\_diversity
----------------

<FuncSignature>Data_Layer.hill_diversity(landuse_labels, qs)</FuncSignature>


@hill\_branch\_wt\_diversity
----------------------------

<FuncSignature>Data_Layer.hill_branch_wt_diversity(landuse_labels, qs)</FuncSignature>


@compute\_accessibilities
-------------------------

<FuncSignature>
<pre>
Data_Layer.compute_accessibilities(landuse_labels,
                                   accessibility_labels)
</pre>
</FuncSignature>


@compute\_stats\_single
-----------------------
<FuncSignature>
<pre>
Data_Layer.compute_stats_single(numerical_label,
                                numerical_array)
</pre>
</FuncSignature>

@compute\_stats\_multiple
-------------------------

<FuncSignature>
<pre>
Data_Layer.compute_stats_multiple(numerical_labels,
                                  numerical_arrays)
</pre>
</FuncSignature>


Data_Layer_From_Dict <Chip text="class"/>
--------------------

<FuncSignature>Data_Layer_From_Dict(data_dict)</FuncSignature>

Directly transposes an appropriately prepared data dictionary into a `Data_Layer`. This `class` calls [`data_map_from_dict`](#data-map-from-dict) internally. Methods and properties are inherited from the parent [`Data_Layer`](#data-layer) class.
