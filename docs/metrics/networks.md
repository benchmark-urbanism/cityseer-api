---
---

<RenderMath></RenderMath>

centrality
==========

distance\_from\_beta
--------------------

<FuncSignature>
<pre>
distance_from_beta(beta,
                   min_threshold_wt=0.01831563888873418)
</pre>              
</FuncSignature>

Maps decay parameters $-\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.

::: danger Caution
It is generally not necessary to use this method directly. This method will be called internally when invoking [Network_Layer](/metrics/networks.html#network-layer) or [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx).
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="beta" type="float, list[float], numpy.ndarray">

$-\beta$ value/s to convert to distance thresholds $d_{max}$.

</FuncElement>
<FuncElement name="min_threshold_wt" type="float">

The cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="betas" type="numpy.ndarray">

A numpy array of distance thresholds $d_{max}$.

</FuncElement>

```python

from cityseer.metrics import networks
# a list of betas
betas = [-0.01, -0.02]
# convert to distance thresholds
d_max = networks.distance_from_beta(betas)
print(d_max)  # prints: array([400., 200.])

```

Weighted measures such sa gravity, weighted betweenness, and weighted land-use accessibilities are computed using a negative exponential decay function in the form of:

$$weight = exp(-\beta \cdot distance)$$

The strength of the decay is controlled by the $-\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
For example, if $-\beta=0.005$ were to represent a person's willingness to walk to a bus stop, then a location $100m$ distant would be weighted at $60\\%$ and a location $400m$ away would be weighted at $13.5\\%$. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. It becomes computationally expensive to consider locations farther away than a selected cutoff weight $w_{min}$, which corresponds to the maximum distance threshold $d_{max}$.

<img src="../plots/betas.png" alt="Example beta decays" class="centre">

[Network_Layer](/metrics/networks.html#network-layer) and [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx) can be invoked with either `distances` or `betas` parameters, but not both. If using the `betas` parameter, then this method will be used to extrapolate the distance thresholds implicitly. If using distances, then the $-\beta$ values will likewise be set automatically, using:

$$-\beta = \frac{log\Big(w_{min}\Big)}{d_{max}}$$

The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking thresholds, for example:

| $-\beta$ | $d_{max}$ |
|-----------|---------|
| $-0.02$ | $200m$ |
| $-0.01$ | $400m$ |
| $-0.005$ | $800m$ |
| $-0.0025$ | $1600m$ |

People may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context: customised decays can be modelled by specifying the desired $-\beta$ values and the $w_{min}$ at which to cap the distance thresholds, for example:

| $-\beta$ | $w_{min}$ | $d_{max}$ |
|----------|----------|----------|
| $-0.02$ | $0.01$ | $230m$ |
| $-0.01$ | $0.01$ | $461m$ |
| $-0.005$ | $0.01$ | $921m$ |
| $-0.0025$ | $0.01$ | $1842m$ |


Network\_Layer
--------------

<FuncSignature>
<pre>
Network_Layer(node_uids,
              node_map,
              edge_map,
              distances=None,
              betas=None,
              min_threshold_wt=0.01831563888873418,
              angular=False)
</pre>
</FuncSignature>

Network layers are used for network centrality computations and provide the backbone for landuse and statistical aggregations. Use [`Network_Layer_From_nX`](#network-layer-from-nx) instead, when converting directly from a `NetworkX` graph to a `Network_Layer`.

A `Network_Layer` requires either a set of distances $d_{max}$ or equivalent exponential decay parameters $-\beta$, but not both. The unprovided parameter will be calculated implicitly in order to keep weighted and unweighted metrics in lockstep. The `min_threshold_wt` parameter can be used to generate custom mappings from one to the other: see [distance_from_beta](#distance-from-beta) for more information.

```python

from cityseer.metrics import networks
from cityseer.util import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_auto_edge_params(G)

# if initialised with distances: 
# betas for weighted metrics will be generated implicitly
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600])
print(N.distances)  # prints: [200, 400, 800, 1600]
print(N.betas)  # prints: [-0.02, -0.01, -0.005, -0.0025]

# if initialised with betas: 
# distances for non-weighted metrics will be generated implicitly
N = networks.Network_Layer_From_nX(G, betas=[-0.02, -0.01, -0.005, -0.0025])
print(N.distances)  # prints: [200, 400, 800, 1600]
print(N.betas)  # prints: [-0.02, -0.01, -0.005, -0.0025]
```

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="node_uids" type="tuple">

A tuple of node ids corresponding to the node identifiers.

</FuncElement>

<FuncElement name="node_map" type="np.ndarray">

A 2d numpy array representing the graph's nodes. The indices of the second dimension correspond to the following:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | start `idx` for the corresponding edge map |
| 4 | `weight` applied to the node | 

</FuncElement>

<FuncElement name="edge_map" type="np.ndarray">

A 2d numpy array representing the graph's edges. The indices of the second dimension correspond to the following:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | `length` in metres for enforcing $d_{max}$ |
| 3 | `impedance` for shortest path calculations |

</FuncElement>

<FuncElement name="distances" type="int, float, list, tuple, np.ndarray">

A distance, or `list`, `tuple`, or `numpy` array of distances corresponding to the local $d_{max}$ thresholds to be used for centrality (and land-use) calculations. The $-\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.

</FuncElement>

<FuncElement name="betas" type="float, list, tuple, np.ndarray">

A $-\beta$, or `list`, `tuple`, or `numpy` array of $-\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.

</FuncElement>

<FuncElement name="min_threshold_wt" type="float">

The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [distance_from_beta](#distance-from-beta) for more information.

</FuncElement>

<FuncElement name="angular" type="bool">

Set the `angular` parameter to `True` when using angular impedances. This adds a step to the shortest-path algorithm, which prevents the side-stepping of sharp angular turns.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="Network_Layer" type="class">

A `Network_Layer`.

</FuncElement>


### Node attributes

The `x` and `y` node attributes determine the spatial coordinates of the node, and should be in a suitable projected (flat) coordinate reference system in metres. Use [`nX_wgs_to_utm`](/util/graphs.html#nx-wgs-to-utm) if required in order to convert from WGS84 `lng`, `lat` geographic coordinates to a local UTM `x`, `y` projected coordinate system.

The `live` node attribute is optional, but recommended. It is used for identifying nodes falling within the original boundary of interest as opposed to those that fall within the surrounding buffered area. (That is assuming you have buffered your extents! See the hint box.) Centrality calculations are only performed for `live` nodes, thus reducing frivolous computation. Note that the algorithms still have access to the full buffered network.

The `idx` node attribute maps the current node to the starting edge-index for associated out-edges. Note that there may be more than one edge associated with any particular node. 

The `weight` parameter allows centrality calculations to be weighted by external considerations, e.g. edge lengths, building density, etc. Where weights are not to be considered, they should be set to a default value of `1`.

### Edge attributes

The start and end edge `idx` attributes point to the corresponding node indices in the node map.

The `length` edge attribute always corresponds to the edge lengths in metres. This is used when calculating the distances traversed by shortest-path algorithms so that the maximum distance thresholds can be enforced.

The `impedance` edge attribute represents the friction to movement across the network: it is used by the shortest-path algorithm when calculating the shortest-routes from each origin node $i$ to all surrounding nodes $j$. For shortest-path centralities, the `impedance` attribute will generally have the same value as the `length` attribute, but this need not be the case. One such example is simplest-path centralities, where impedance represents the angular change in the direction.

::: tip Hint
When calculating local network centralities, it is best-practice to buffer the global network by a distance equal to the maximum distance threshold to be considered. This prevents misleading results arising due to boundary roll-off effects.
:::


Network\_Layer\_From\_nX
------------------------

<FuncSignature>
<pre>
Network_Layer_From_nX(networkX_graph,
                      distances=None,
                      betas=None,
                      min_threshold_wt=0.01831563888873418,
                      angular=False)
</pre>
</FuncSignature>

Transposes a `networkX` graph a `Network_Layer`. This `class` simplifies the conversion of `NetworkX` graphs by calling [`graph_maps_from_nX`](/util/graphs.html#graph-maps-from-nx) internally and then instancing a [`Network_Layer`](#network-layer) class.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph.

`x` and `y` node attributes are required. `weight` and `live` node attributes are optional.

`length` and `impedance` edge attributes are required.

</FuncElement>

<FuncElement name="distances" type="int, float, list, tuple, np.ndarray">

A distance, or `list`, `tuple`, or `numpy` array of distances corresponding to the local $d_{max}$ thresholds to be used for centrality (and land-use) calculations. The $-\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.

</FuncElement>

<FuncElement name="betas" type="float, list, tuple, np.ndarray">

A $-\beta$, or `list`, `tuple`, or `numpy` array of $-\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.

</FuncElement>

<FuncElement name="min_threshold_wt" type="float">

The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [distance_from_beta](#distance-from-beta) for more information.

</FuncElement>

<FuncElement name="angular" type="bool">

Set the `angular` parameter to `True` when using angular impedances. This adds a step to the shortest-path algorithm, which prevents the side-stepping of sharp angular turns.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="Network_Layer" type="class">

A `Network_Layer`.

</FuncElement>

Please refer to the parent [`Network_Layer`](#network-layer) class for more information.


N.metrics\_to\_dict
-------------------

<FuncSignature>Network_Layer.metrics_to_dict()</FuncSignature>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="metrics_dict" type="dict">

Unpacks all calculated metrics from the  `Network_Layer` class into a `python` dictionary. The dictionary `keys` will correspond to the node `uids`.

</FuncElement>


```python
from cityseer.metrics import networks
from cityseer.util import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_auto_edge_params(G)

# generate the network layer and compute some metrics
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600])
N.harmonic_closeness()

# let's select a random node id
random_idx = 6
random_uid = N.uids[random_idx]

# the data is directly available at N.metrics
# in this case the data is stored in arrays corresponding to the node indices
print(N.metrics['centrality']['harmonic'][200][random_idx])
# prints: 0.02312025367905132

# let's convert the data to a dictionary
# the unpacked data is now stored by the uid of the node identifier
data_dict = N.metrics_to_dict()
print(data_dict[random_uid]['centrality']['harmonic'][200])
# prints: 0.02312025367905132
```


N.to\_networkX
--------------

<FuncSignature>Network_Layer.to_networkX()</FuncSignature>

Converts a `Network Layer` to a `networkX` graph.

<FuncHeading>Returns</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph.

`x`, `y`, `live`, `weight` node attributes will be copied from the `node_map` to the graph nodes. `length` and `impedance` attributes will be copied from the `edge_map` to the graph edges. 

Any data from computed metrics will be copied to the graph.

</FuncElement>

```python
from cityseer.metrics import networks
from cityseer.util import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_auto_edge_params(G)

# generate the network layer and compute some metrics
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600])
# compute some-or-other metrics
N.harmonic_closeness()
# convert back to networkX
G_post = N.to_networkX()

# let's select a random node id
random_idx = 6
random_uid = N.uids[random_idx]

print(N.metrics['centrality']['harmonic'][200][random_idx])
# prints: 0.02312025367905132

# the metrics have been copied to the new networkX graph
print(G_post.nodes[random_uid]['metrics']['centrality']['harmonic'][200])
# prints: 0.02312025367905132
```

<img src="../plots/graph_before.png" alt="Graph before conversion" class="left"><img src="../plots/graph_after.png" alt="graph after conversion back to networkX" class="right">

_A `networkX` graph before conversion to a `Network Layer` (left) and after conversion back to `networkX` (right)._


N.compute\_centrality
---------------------

<FuncSignature>N.compute_centrality(close_metrics=None, between_metrics=None)</FuncSignature>

Wraps underlying `numba` optimised functions for computing network centralities. Provides access to all available methods, which are computed simultaneously for any required combinations (and distances), which has potentially significant speed implications. Situations requiring only a single measure, can make use of the simplified [`N.gravity`](#n-gravity), [`N.harmonic closeness`](#n-harmonic-closeness), [`N.betweenness`](#n-betweenness), or [`N.weighted betweenness`](##n-betweenness-gravity) methods.

The calculated metrics will be written to the `Network_Layer` and is accessible from the `metrics` property, using the following pattern:
 
`Network_Layer.metrics['centrality'][<<centrality key>>][<<distance key>>][<<node idx>>]`

```python
from cityseer.metrics import networks
from cityseer.util import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_auto_edge_params(G)

# generate the network layer and compute some metrics
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600])
N.compute_centrality(close_metrics=['harmonic'])

# distance idx: any of the distance with which the Network_Layer was initialised
distance_idx = 200
# let's select a random node idx
random_idx = 6

# the data is directly available at N.metrics
# in this case we need the 'harmonic' key and any of the initialised distances
print(N.metrics['centrality']['harmonic'][distance_idx][random_idx])
# prints: 0.02312025367905132
```

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="close_metrics" type="list[str], tuple[str]">

A list of strings, containing any combination of the following `key` values:

</FuncElement>

| key | formula | notes |
|-----|---------|-------|
| node_density | $$\sum_{j\neq{i}} w_{j}$$ | The default $w=1$ reduces to a simple node count, however, this is technically a density measure because of the $d_{max}$ threshold constraint. Setting $w$ equal to street lengths converts the measure to a street density measure. |
| farness_impedance | $$\sum_{j\neq{i}} \frac{Z_{(i,j)}}{w_{j}}$$ | $w=1$ reduces to the sum of impedances $Z$ within the threshold $d_{max}$. Be cautious with weights where $w=0$ because this would return `np.inf`. |
| farness_distance | $$\sum_{j\neq{i}}d_{(i,j)}$$ | A summation of distances in metres within $d_{max}$. |
| harmonic | $$\sum_{j\neq{i}}\frac{w_{j}}{Z_{(i,j)}}$$ | Reduces to _harmonic closeness_ where $w=1$. Harmonic closeness is the appropriate form of closeness centrality for localised implementations constrained by the threshold $d_{max}$. (Conventional forms of closeness centrality should not be used in this context.) |
| improved | $$\frac{(N-1)\_{w_{j}}^2}{\sum_{j\neq{i}}d_{(i,j)}}$$ | A simplified variant of _"improved" closeness_. Like _harmonic closeness_, this variant behaves appropriately on localised implementations. |
| gravity | $$\sum_{j\neq{i}} exp(-\beta \cdot d[i,j]) \cdot w_{j}$$ | Reduces to _gravity centrality_ where $w=1$. Gravity is differentiated from other closeness centralities by the use of an explicit $-\beta$ parameter modelling distance decays. |
| cycles | $$\sum_{j\neq{i}}^{cycles} exp(-\beta \cdot d[i, j])$$ | A summation of distance-weighted network cycles within the threshold $d_{max}$ |

<FuncElement name="between_metrics" type="list[str], tuple[str]">

A list of strings, containing any combination of the following `key` values:

</FuncElement>

| key | formula | notes |
|-----|---------|-------|
| betweenness | $$\sum_{j\neq{i}} \sum_{k\neq{j}\neq{i}} w_{(j, k)}$$ | The default $w=1$ reduces to betweenness centrality within the $d_{max}$ threshold constraint. For betweenness measures, $w$ is a blended average of the weights for any $j$, $k$ node pair passing through node $i$. | 
| betweenness_gravity | $$\sum_{j\neq{i}} \sum_{k\neq{j}\neq{i}} w_{(j, k)} \cdot exp(-\beta \cdot d[j,k])$$ | Adds a distance decay to betweenness. $d$ represents the full distance from any $j$ to $k$ node pair passing through node $i$.

::: tip Hint
The following four methods are simplified wrappers for some of the more commonly used forms of network centrality. Note that for cases requiring more than one form of centrality, it may be substantially faster to compute all variants at once by using the underlying [N.compute_centrality](#n-compute-centrality) method directly. 
:::


N.harmonic\_closeness
---------------------

<FuncSignature>N.harmonic_closeness()</FuncSignature>

Compute harmonic closeness. See [N.compute_centrality](#n-compute-centrality) for more information.

The data key is `harmonic`, e.g.

`Network_Layer.metrics['centrality']['harmonic'][<<distance key>>][<<node idx>>]`

N.gravity
---------

<FuncSignature>N.gravity()</FuncSignature>

Compute gravity centrality. See [N.compute_centrality](#n-compute-centrality) for more information.

The data key is `gravity`, e.g.

`Network_Layer.metrics['centrality']['gravity'][<<distance key>>][<<node idx>>]`

N.betweenness
-------------

<FuncSignature>N.betweenness()</FuncSignature>

Compute betweenness. See [N.compute_centrality](#n-compute-centrality) for more information.

The data key is `betweenness`, e.g.

`Network_Layer.metrics['centrality']['betweenness'][<<distance key>>][<<node idx>>]`

N.betweenness\_gravity
----------------------

<FuncSignature>N.betweenness_gravity()</FuncSignature>

Compute gravity weighted betweenness. See [N.compute_centrality](#n-compute-centrality) for more information.

The data key is `betweenness_gravity`, e.g.

`Network_Layer.metrics['centrality']['betweenness_gravity'][<<distance key>>][<<node idx>>]`
