---
---

<RenderMath></RenderMath>

centrality <Chip text="beta" :important="true"/>
==========

distance\_from\_beta() <Chip text='v0.1+'/>
----------------------

<FuncSignature>distance_from_beta(beta, min_threshold_wt=0.01831563888873418)</FuncSignature>

A convenience method mapping $-\beta$ decay parameters to equivalent $d_{max}$ distance thresholds at a specified minimum weight of $w_{min}$, which can then be passed to [`centrality.compute_centrality`](#compute-centrality).

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
<FuncElement name="min_threshold_wt" type="float">

The $w_{min}$ threshold.

</FuncElement>

::: warning Note
There is no need to use this method unless overriding the automatically generated default $\beta$ values in [`centrality.compute_centrality`](#compute-centrality).
:::

::: danger Important
Pass both $d_{max}$ and $w_{min}$ to [`centrality.compute_centrality`](#compute-centrality).
:::

The weighted variants of centrality, i.e. gravity and weighted betweenness, are computed using a negative exponential decay function of the form:

$$weight = exp(-\beta \cdot distance)$$

The strength of the decay is controlled by the $-\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
For example, if $-\beta=0.005$ were to represent a person's willingness to walk to a bus stop, then a location $100m$ distant would be weighted at $60\\%$ and a location $400m$ away would be weighted at $13.5\\%$. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At infinitesimal weights, it becomes computationally expensive to consider locations any farther away. The threshold $w_{min}$ corresponds to the minimum weight at which the maximum distance $d_{max}$ is set.

The [`centrality.compute_centrality`](#compute-centrality) method computes the $-\beta$ parameters automatically using the following formula:

$$\beta = \frac{log\Big(1 / w_{min}\Big)}{d_{max}}$$

For example, using the default `min_threshold_wt` of $w_{min}=0.01831563888873418$ at $d_{max}$ walking thresholds of $400m$, $800m$, and $1600m$, would yield the following $\beta$:

| $d_{max}$ | $-\beta$ |
|-----------|:----------|
| $400m$ | $-0.01$ |
| $800m$ | $-0.005$ |
| $1600m$ | $-0.0025$ |

People may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context. This method can therefore be used to model specific $-\beta$ parameters or set custom $w_{min}$ cutoffs. The returned effective $d_{max}$ and $w_{min}$ values can then be passed to [`centrality.compute_centrality`](#compute-centrality). For example, the following $-\beta$ and $w_{min}$ thresholds would yield the following effective $d_{max}$ distances:

| $-\beta$ | $w_{min}$ | $d_{max}$ |
|----------|:----------|:----------|
| $-0.01$ | $0.01$ | $461m$ |
| $-0.005$ | $0.01$ | $921m$ |
| $-0.0025$ | $0.01$ | $1842m$ |


graph\_from\_networkx() <Chip text='v0.1+'/>
-----------------------

<FuncSignature>graph_from_networkx(network_x_graph, wgs84_coords=False, decompose=False, geom=None)</FuncSignature>

A convenience method for generating a `node_map` and `edge_map` from a [NetworkX](https://networkx.github.io/documentation/networkx-1.10/index.html) undirected Graph, which can then be passed to [`centrality.compute_centrality`](#compute-centrality).

<FuncHeading>Parameters</FuncHeading>
<FuncElement name="network_x_graph" type="networkx.Graph">

A NetworkX undirected `Graph`. Requires `x` and `y` node attributes for spatial coordinates. Accepts optional `label` and `live` node attributes. Accepts optional `length` and `weight` edge attributes. See notes.

</FuncElement>
<FuncElement name="wgs84_coords" type="bool">

Set to `True` if the `x` and `y` node attribute keys reference [`WGS84`](https://epsg.io/4326) lng, lat values instead of a projected coordinate system.

</FuncElement>
<FuncElement name="decompose" type="int, float">

Optionally generate a decomposed version of the graph wherein edges are broken into smaller sections no longer than the specified distance in metres. This evens out the density of nodes to reduce topological distortions in the graph, and provides a more granular representation of variations along street (edge) lengths.

</FuncElement>
<FuncElement name="geom" type="shapely.geometry.Polygon">

An optional `shapely` [`Polygon`](https://shapely.readthedocs.io/en/latest/manual.html#polygons) geometry defining the original boundary of interest. Must be in the same Coordinate Reference System as the `x` and `y` attributes supplied in the graph. Used for identifying `live` nodes in the event that a `live` attribute was not provided with the graph. If both the `geom` and the `live` attribute are provided, then the `live` attribute takes precedence and the `geom` is then only used for handling certain edge cases for decomposed nodes.

</FuncElement>
<FuncHeading>Returns</FuncHeading>
<FuncElement name="node_labels" type="list">

A list of node labels in the same order as the node map.

</FuncElement>

<FuncElement name="node_map" type="numpy.ndarray">

Node data in the form of an $n \times 4$ array, where each row represents a node consisting of: `[x, y, live, edge_index]`

</FuncElement>
<FuncElement name="edge_map" type="numpy.ndarray">

Edge data in the form of an $e \times 4$ array, where each row represents an edge consisting of: `[start_node, end_node, length, weight]`.

</FuncElement>

The node attributes `x` and `y` determine the spatial coordinates of the node, and should be in a suitable projected (flat) coordinate reference system in metres unless the `wgs84_coords` parameter is set to `True`.

If provided, the `label` node attribute will be saved to the `node_labels` list. If not provided, labels will be generated automatically. If decomposing the graph, newly generated nodes will be assigned a new label.

The `live` node attribute is optional, but recommended. It is used for identifying which nodes fall within the original boundary of interest as opposed to those that fall within the surrounding buffered area. (That is assuming you have buffered your extents! See the hint box.) If provided, centrality calculations are only performed for `live` nodes, thus reducing frivolous computation. Note that the algorithms still have access to the full buffered network. If providing a `geom` in lieu of the `live` attribute, then the live tag will be generated internally. The `geom` method can be computationally intensive for large graphs or complex boundaries, in which case, consider using a simplified geometry or run the analysis for the entire area and discard buffered nodes afterwards.

The optional edge attribute `length` represents the original edge length in metres. If not provided, lengths will be computed using crow-flies distances between either end of the edges. This attribute must be positive.

If provided, the optional edge attribute `weight` will be used for shortest path calculations instead of distances in metres. If decomposing the network, then the `weight` attribute will be divided into the number of decomposed sub-edges. This attribute must be positive.

::: warning Note
This method assumes that all graph preparation, e.g. cleaning and simplification, has happened upstream of this method. If generating data from sources such as [Open Street Map](https://www.openstreetmap.org), then consider using tools such as [roadmap-processing](https://github.com/aicenter/roadmap-processing) for initial fetching, cleaning, and simplification of the data. Whereas simplification (assuming accurate distances are maintained via the `length` attribute) helps reduce topological distortions in centrality methods, another option is to use a sufficiently fine level of decomposition to likewise temper node density variances.
:::

::: tip Hint
When calculating local network centralities, it is best-practice for the area of interest to have been buffered by a distance equal to the maximum distance threshold to be considered. This prevents misleading results arising due to a boundary roll-off effect.
:::

::: danger Important
Graph decomposition provides a more granular representation of variations along street lengths. However, setting the `decompose` parameter too small in relation to the size of the graph can increase the computation time unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller $20m$, and $50m$ may already be sufficient for many cases. On the other-hand, it may be feasible to go as small as possible for very small networks, such as building circulation networks.
:::


compute\_centrality()
---------------------

