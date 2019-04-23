---

---

cityseer.util.graphs
====================

A collection of convenience functions for the preparation and conversion of [`NetworkX`](https://networkx.github.io/) graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.


nX\_simple\_geoms
-----------------

<FuncSignature>nX_simple_geoms(networkX_graph)</FuncSignature>

Generates straight-line geometries for each edge based on the the `x` and `y` coordinates of the adjacent nodes. The edge geometry will be stored to the edge `geom` attribute. The [nX_auto_edge_params](#nx-auto-edge-params) function can then be used to auto-generate edge `length` and `impedance` attributes.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph with `x` and `y` node attributes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

Returns a `networkX` graph with `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge `geom` attributes.

</FuncElement>


nX\_wgs\_to\_utm
----------------

<FuncSignature>nX_wgs_to_utm(networkX_graph, force_zone_number=None)</FuncSignature>

Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all other nodes and geometries contained in the graph. This ensures consistent behaviour when a graph spans a UTM boundary.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge attributes containing `LineString` geoms to be converted.

</FuncElement>

<FuncElement name="force_zone_number" type="int">

An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched UTM zones may introduce substantial distortions in the results. 

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A `networkX` graph with `x` and `y` node attributes converted to the local UTM coordinate system. Edge `geom` attributes will also be converted, if found.

</FuncElement>


nX\_remove\_dangling\_nodes <Chip text="v0.8.0"/>
---------------------------

<FuncSignature>
<pre>
nX_remove_dangling_nodes(networkX_graph,
                         despine=25,
                         remove_disconnected=True)
</pre>
</FuncSignature>

Optionally removes short dead-ends and disconnected graph components, which may be prevalent on poor quality network datasets.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncElement name="despine" type="float">

The maximum cutoff distance for removal of dead-ends. Use `0` where no de-spining should occur.

</FuncElement>

<FuncElement name="remove_disconnected" type="bool">

Whether to remove disconnected components. When set to `True`, only the largest connected component will be returned.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A `networkX` graph with disconnected components optionally removed, and dead-ends removed where less than the `despine` distance.

</FuncElement>


nX\_remove\_filler\_nodes
-------------------------

<FuncSignature>nX_remove_filler_nodes(networkX_graph)</FuncSignature>

Removes frivolous nodes where $degree=2$: such nodes represent no route-choices other than continuing-on to the next edge. The edges on either side of the deleted nodes will be removed and replaced with a new unified edge, with `geom` attributes welded together.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A `networkX` graph with nodes of $degree=2$ removed. Adjacent edges will be combined into a single new edge with associated `geom` attributes welded together.

</FuncElement>

::: tip Hint
Frivolous nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` [`Linestrings`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) to accurately describe arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus reducing potential side-effects when computing network centralities as a function of varied node intensities.
:::


nX\_consolidate\_spatial <Chip text="v0.8.4"/>
------------------------

<FuncSignature>nX_consolidate_spatial(networkX_graph, buffer_dist=14)</FuncSignature>

Consolidates nearby nodes within a specified spatial buffer distance.

::: tip Hint

Compare with [`nX_consolidate_parallel`](#nx-consolidate-parallel).

:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncElement name="buffer_dist" type="float">

The buffer distance to be used for consolidating nearby nodes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A `networkX` graph. Nodes located within the `buffer_dist` distance from other nodes will be consolidated into new parent nodes. The coordinates of the parent nodes will be derived from the aggregate centroid of the constituent nodes. 

</FuncElement>


nX\_consolidate\_parallel <Chip text="v0.8.7"/>
-------------------------

<FuncSignature>nX_consolidate_parallel(networkX_graph, buffer_dist=14)</FuncSignature>

Consolidates nearby nodes within a spatial buffer distance, but only if adjacent nodes are found that are also within the buffer distance from each other. This method targets parallel links and may better preserve overall network topology.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncElement name="buffer_dist" type="float">

The buffer distance to be used for consolidating nearby nodes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A `networkX` graph. Nodes located within the `buffer_dist` distance from other nodes will be consolidated into new parent nodes. The coordinates of the parent nodes will be derived from the aggregate centroid of the constituent nodes. 

</FuncElement>


nX\_decompose
-------------

<FuncSignature>nX_decompose(networkX_graph, decompose_max)</FuncSignature>

Decomposes a graph so that no edge is longer than a set maximum. Decomposition provides a more granular representation of potential variations along street lengths, while reducing network centrality side-effects that arise as a consequence of varied node densities.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms. Optional `live` node attributes.

</FuncElement>
<FuncElement name="decompose_max" type="nx.Graph">

The maximum length threshold for decomposed edges.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `impedance` edge attributes will be set to match the lengths of the new edges.

</FuncElement>

```python
from cityseer.util import mock, graphs

G = mock.mock_graph()
G_simple = graphs.nX_simple_geoms(G)
G_decomposed = graphs.nX_decompose(G_simple, 100)
```

<img src="../images/plots/graph_simple.png" alt="Example graph" class="left"><img src="../images/plots/graph_decomposed.png" alt="Example decomposed graph" class="right">

_Simple graph (left) and the equivalent $100m$ decomposed graph (right)._

::: warning Note
Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller $20m$, and $50m$ may already be sufficient for the majority of cases.
:::

::: tip Hint

This function will automatically orient the `geom` attribute LineStrings in the correct direction before splitting into sub-geometries; i.e. there is no need to order the geometry's coordinates in a particular direction.

:::


nX\_to\_dual
------------

<FuncSignature>nX_to_dual(networkX_graph)</FuncSignature>

Converts a primal graph representation, where intersections are represented as nodes and streets as edges, to the dual representation, such that edges are converted to nodes and intersections become edges. This is necessary for the computation of simplest-path (angular) metrics. Primal edge `geom` attributes will be welded to adjacent edges and split into the new dual edge `geom` attributes, from which the `length` and `impedance` attributes will be set. The angular impedances will be calculated by summing the angular change over the length of the geometry.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms. Optional `live` node attributes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding to the mid-points of the original primal edges.

If `live` node attributes were provided, then the `live` attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set to `live=True`. Otherwise, all dual nodes wil be set to `live=True`.

The primal `geom` edge attributes will be split and welded to form the new dual `geom` edge attributes, from which the `length` and angular `impedance` edge attributes will be set. A `parent_primal_node` edge attribute will be added, corresponding to the node identifier of the primal graph.

</FuncElement>

```python
from cityseer.util import mock, graphs

G = mock.mock_graph()
G_simple = graphs.nX_simple_geoms(G)
G_dual = graphs.nX_to_dual(G_simple)
```

<img src="../images/plots/graph_dual.png" alt="Example dual graph" class="centre" style="max-height:450px;">

_Dual graph (blue) overlaid on the source primal graph (red). Edge `length` and angular `impedance` attributes are set automatically from the `geom` attribute's length and total angular change._

::: warning Note
Shortest-path centralities decompose more gracefully than angular-based simplest-path centralities for pedestrian walking distances, at which scale they also demonstrate stronger correlations to mixed-uses.
:::

::: tip Hint

This function will automatically orient the `geom` attribute LineStrings in the correct direction when splitting and welding; i.e. there is no need to order the geometry's coordinates in a particular direction.

:::


nX\_auto\_edge\_params
----------------------

<FuncSignature>nX_auto_edge_params(networkX_graph)</FuncSignature>

Generates edge `length` and `impedance` attributes from the `geom` edge attribute.

::: danger Caution
Do not use this function on dual graphs. Dual graphs prepared with [nX_to_dual](#nx-to-dual) will already have `length` and `impedance` edge attributes set, and calling this function would consequently overwrite the angular impedance attributes. It is also not necessary to call this method on decomposed graphs, because [nX_decompose](#nx-decompose) will set the `length` and `impedance` attributes as part of the process of decomposition.
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

Returns a `NetworkX` graph with added `length` and `impedance` edge attributes. Both `length` and `impedance` will be set to the length of the street edge in metres, as deduced from the `Linestring` geometry. 

</FuncElement>


nX\_m\_weighted\_nodes
----------------------

<FuncSignature>nX_m_weighted_nodes(networkX_graph)</FuncSignature>

Adds a `weight` node attribute and sets the value to the aggregate half-lengths of the adjacent street edges. Subsequent centrality computations will be weighted accordingly.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="graph" type="nx.Graph">

Returns a `NetworkX` graph with an added `weight` node attribute, which will be set to the aggregate half-lengths of adjacent street edges.

</FuncElement>

::: tip Hint
Certain centrality calculations can be weighted by street-lengths, which helps to mitigate undesirable side-effects arising as a function of varied node densities due to the topological representation of the graph. That said, weighted nodes achieve a similar effect to graph decomposition, which is generally preferable because it offers a more granular representation of edges, thereby accommodating variations in intensities along street-fronts.
:::


graph\_maps\_from\_nX
---------------------

<FuncSignature>graph_maps_from_nX(networkX_graph)</FuncSignature>

Transposes a `networkX` graph into `numpy` arrays for use by `Network_Layer` classes.

::: warning Note
It is generally not necessary to use this function directly. This function will be called internally when invoking [Network_Layer_From_nX](/metrics/networks.html#network-layer-from-nx)
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates.

`x` and `y` node attributes are required. The `weight` node attribute is optional, and a default of `1` will be used if not present. The `live` node attribute is optional, but recommended. See [`Network_Layer`](#network-layer) for more information about what these attributes represent.

`length` and `impedance` edge attributes are required. See [`Network_Layer`](#network-layer) for more information about what these attributes represent.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="node_uids" type="tuple">

A tuple of node `uids` corresponding to the node identifiers in the source `networkX` graph.

</FuncElement>

<FuncElement name="node_map" type="np.ndarray">

A 2d `numpy` array representing the graph's nodes. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | start `idx` for the corresponding edge map |
| 4 | `weight` applied to the node | 

</FuncElement>

<FuncElement name="edge_map" type="np.ndarray">

A 2d `numpy` array representing the graph's edges. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | `length` in metres |
| 3 | `impedance` for shortest path calculations |

</FuncElement>


nX\_from\_graph\_maps
---------------------

<FuncSignature>
<pre>
nX_from_graph_maps(node_uids,
                   node_map,
                   edge_map,
                   networkX_graph=None,
                   metrics_dict=None)
</pre>
</FuncSignature>

Transposes `cityseer` graph maps into a `networkX` graph.

::: warning Note
It is generally not necessary to use this function directly. This function will be called internally when invoking [Network_Layer.to_networkX](/metrics/networks.html#to-networkX)
:::

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="node_uids" type="tuple">

A tuple of node ids corresponding to the node identifiers for the target `networkX` graph.

</FuncElement>

<FuncElement name="node_map" type="np.ndarray">

A 2d `numpy` array representing the graph's nodes. The indices of the second dimension should correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | start `idx` for the corresponding edge map |
| 4 | `weight` applied to the node | 

</FuncElement>

<FuncElement name="edge_map" type="np.ndarray">

A 2d `numpy` array representing the graph's edges. The indices of the second dimension should correspond as follows:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | `length` in metres |
| 3 | `impedance` for shortest path calculations |

</FuncElement>

<FuncElement name="networkX_graph" type="nx.Graph">

An optional `networkX` graph to use as a backbone for unpacking the data. The number of nodes and edges should correspond to the `cityseer` data maps and the node identifiers should correspond to the `node_uids`. If not provided, then a new `networkX` graph will be returned. This function is intended to be used for situations where `cityseer` data is being transposed back to a source `networkX` graph.

</FuncElement>


<FuncElement name="metrics_dict" type="dict">

An optional dictionary with keys corresponding to the identifiers in `node_uids`. The dictionary's `values` will be unpacked to the corresponding nodes in the `networkX` graph.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph. If a backbone graph was provided, a copy of the same graph will be returned with the data overridden as described below. If no graph was provided, then a new graph will be generated.

`x`, `y`, `live`, `weight` node attributes will be copied from the `node_map` to the graph nodes. `length` and `impedance` attributes will be copied from the `edge_map` to the graph edges. 

If a `metrics_dict` is provided, all data will be copied to the graph nodes based on matching node identifiers.

</FuncElement>