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

Maps $-\beta$ decay parameters to equivalent $d_{max}$ distance thresholds at a specified minimum weight of $w_{min}$.

::: danger Caution
It is generally not necessary to use this method directly. This method will be called internally when invoking [Network_Layer](/metrics/networks.html#network-layer) or [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx).
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="beta" type="float, list[float], numpy.ndarray">

$-\beta$ value/s to convert to distance thresholds $d_{max}$.

</FuncElement>
<FuncElement name="min_threshold_wt" type="float">

The minimum weight $w_{min}$ at which to set the distance threshold $d_{max}$.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="betas" type="numpy.ndarray">

A numpy array of $d_{max}$ distances.

</FuncElement>

Weighted centralities, i.e. gravity and weighted betweenness, and land-use accessibilities are computed using a negative exponential decay function of the form:

$$weight = exp(-\beta \cdot distance)$$

The strength of the decay is controlled by the $-\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
For example, if $-\beta=0.005$ were to represent a person's willingness to walk to a bus stop, then a location $100m$ distant would be weighted at $60\\%$ and a location $400m$ away would be weighted at $13.5\\%$. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At small weights, it becomes computationally expensive to consider locations any farther away, for which reason the distances are capped at $d_{max}$ corresponding to the minimum considered weight of $w_{min}$.

<img src="../plots/betas.png" alt="Example beta decays" class="centre">

[Network_Layer](/metrics/networks.html#network-layer) and [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx) can be invoked with either `distances` or `betas` parameters, but not both. If using the `betas` parameter, then this method will be used internally to extrapolate the distance thresholds. If using distances, then the $-\beta$ values will likewise be set automatically, using:

$$\beta = \frac{log\Big(w_{min}\Big)}{d_{max}}$$

The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking thresholds, for example:

| $-\beta$ | $d_{max}$ |
|-----------|:---------|
| $-0.02$ | $200m$ |
| $-0.01$ | $400m$ |
| $-0.005$ | $800m$ |
| $-0.0025$ | $1600m$ |

People may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context: customised behaviour can be modelled by specifying the desired $-\beta$ values and the $w_{min}$ at which to cap the distance thresholds, for example:

| $-\beta$ | $w_{min}$ | $d_{max}$ |
|----------|:----------|:----------|
| $-0.02$ | $0.01$ | $230m$ |
| $-0.01$ | $0.01$ | $461m$ |
| $-0.005$ | $0.01$ | $921m$ |
| $-0.0025$ | $0.01$ | $1842m$ |

```python

from cityseer.metrics import networks
# a list of betas
betas = [-0.01, -0.02]
# convert to distance thresholds
networks.distance_from_beta(betas)
# prints: array([400., 200.])

```


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

Network layers are used for network centrality computations and provide the backbone for landuse and statistical aggregations. Use [`Network_Layer_From_nX`](#network-layer-from-nx) instead if converting directly from a `NetworkX` graph to a `Network_Layer`.

A `Network_Layer` requires either a set of distances $d_{max}$ or equivalent exponential decay parameters $-\beta$, but not both. The unprovided parameter will be calculated implicitly in order to keep weighted and unweighted metrics in lockstep. The `min_threshold_wt` parameter can be used to generate custom mappings from one to the other, see [distance_from_beta](#distance-from-beta) for more information.

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

A 2d numpy array containing the graph's nodes. The indices of the second dimension correspond to the following:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | start `idx` for the corresponding edge map |
| 4 | `weight` applied to the node | 

</FuncElement>

<FuncElement name="edge_map" type="np.ndarray">

A 2d numpy array containing the graph's edges. The indices of the second dimension correspond to the following:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | `length` in metres |
| 3 | `impedance` for shortest path calculations |

</FuncElement>

<FuncElement name="distances" type="[int, float, list, tuple, np.ndarray]">

A distance, or `list`, `tuple`, or `numpy` array of distances corresponding to the local $d_{max}$ thresholds to be used for centrality (and land-use) calculations. If provided, then the $-\beta$ parameters (for distance-weighted metrics) will be calculated implicitly. If not provided, then the `beta` parameter must be provided explicitly.

</FuncElement>

<FuncElement name="betas" type="[float, list, tuple, np.ndarray]">

A $-\beta$, or `list`, `tuple`, or `numpy` array of $-\beta$ to be used for the exponential decay function for weighted metrics. If provided, then the `distance` parameters for unweighted metrics will be calculated implicitly. If not provided, then the `distance` parameter must be provided instead.

</FuncElement>

<FuncElement name="min_threshold_wt" type="float">

The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [distance_from_beta](#distance-from-beta) for more information.

</FuncElement>

<FuncElement name="angular" type="bool">

Set the `angular` parameter to `True` if using angular impedances. This adds a step to the shortest-path algorithm that prevents the side-stepping of sharp angular turns.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

The `x` and `y` node attributes determine the spatial coordinates of the node, and should be in a suitable projected (flat) coordinate reference system in metres. Use [`nX_wgs_to_utm`](/util/graphs.html#nx-wgs-to-utm) if required in order to convert from a WGS84 `lng`, `lat` geographic coordinate system to a local UTM `x`, `y` projected coordinate system.

The `live` node attribute is optional, but recommended. It is used for identifying which nodes fall within the original boundary of interest as opposed to those that fall within the surrounding buffered area. (That is assuming you have buffered your extents! See the hint box.) Centrality calculations are only performed for `live` nodes, thus reducing frivolous computation. Note that the algorithms still have access to the full buffered network.

The `idx` node attribute maps the current node to the starting edge-index for associated out-edges. 

The `weight` parameter can be used to 





The optional edge attribute `length` represents the original edge length in metres. If not provided, lengths will be computed using crow-flies distances between either end of the edges. This attribute must be positive.

If provided, the optional edge attribute `weight` will be used for shortest path calculations instead of distances in metres. If decomposing the network, then the `weight` attribute will be divided into the number of decomposed sub-edges. This attribute must be positive.

::: warning Note
This method assumes that all graph preparation, e.g. cleaning and simplification, has happened upstream of this method. If generating data from sources such as [Open Street Map](https://www.openstreetmap.org), then consider using tools such as [roadmap-processing](https://github.com/aicenter/roadmap-processing) for initial fetching, cleaning, and simplification of the data. Whereas simplification (assuming accurate distances are maintained via the `length` attribute) helps reduce topological distortions in centrality methods, another option is to use a sufficiently fine level of decomposition to likewise temper node density variances.
:::

::: tip Hint
When calculating local network centralities, it is best-practice to buffer the global network by a distance equal to the maximum distance threshold to be considered. This prevents misleading results arising due to a boundary roll-off effect.
:::

Network\_Layer\_From\_nX
------------------------

metrics\_to\_dict
-----------------

to\_networkX
------------

compute\_centrality
-------------------

harmonic\_closeness
-------------------

gravity
-------

betweenness
-----------

betweenness\_gravity
--------------------


