
nX\_to\_dual
------------

<FuncSignature>nX_to_dual(networkX_graph)</FuncSignature>

Converts a primal graph representation, where intersections are represented as nodes and streets as edges, to the dual representation, such that edges are converted to nodes and intersections become edges. Primal edge `geom` attributes will be welded to adjacent edges and split into the new dual edge `geom` attributes.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph in UTM coordinates, containing `x` and `y` node attributes, and a `geom` edge attribute containing `LineString` geoms. Optional `live` node attributes.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="graph" type="nx.Graph">

A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding to the mid-points of the original primal edges.

If `live` node attributes were provided, then the `live` attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set to `live=True`. Otherwise, all dual nodes wil be set to `live=True`.

The primal `geom` edge attributes will be split and welded to form the new dual `geom` edge attributes. A `parent_primal_node` edge attribute will be added, corresponding to the node identifier of the primal graph.

</FuncElement>

```python
from cityseer.tools import mock, graphs

G = mock.mock_graph()
G_simple = graphs.nX_simple_geoms(G)
G_dual = graphs.nX_to_dual(G_simple)
```

<img src="../images/plots/graph_dual.png" alt="Example dual graph" class="centre" style="max-height:450px;">

_Dual graph (blue) overlaid on the source primal graph (red)._

::: tip Hint

This function will automatically orient the `geom` attribute LineStrings in the correct direction when splitting and welding; i.e. there is no need to order the geometry's coordinates in a particular direction.

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

`x` and `y` node attributes are required. The `live` node attribute is optional, but recommended. The `ghosted` attribute should be applied to 'ghosted' nodes on decomposed graphs -- this will be added automatically if using [`nX_decompose`](#nx_decompose). See [`Network_Layer`](#network-layer) for more information about what these attributes represent.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="node_uids" type="tuple">

A tuple of node `uids` corresponding to the node identifiers in the source `networkX` graph.

</FuncElement>

<FuncElement name="node_data" type="np.ndarray">

A 2d `numpy` array representing the graph's nodes. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | `ghosted` describing whether the node is a 'ghosted' or 'decomposed' node that is not essential to the network topology. |

</FuncElement>

<FuncElement name="edge_data" type="np.ndarray">

A 2d `numpy` array representing the graph's edges. Each edge will be described separately for each direction of travel. The indices of the second dimension correspond as follows:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | the segment length in metres |
| 3 | the sum of segment's angular change |
| 4 | an 'impedance factor' which can be applied to magnify or reduce the effect of the edge's impedance on shortest-path calculations. e.g. for gradients or other such considerations. Use with caution. |
| 5 | the edge's entry angular bearing |
| 6 | the edge's exit angular bearing |

All edge attributes will be generated automatically, however, the impedance factor parameter can be over-ridden by supplying a `imp_factor` attribute on the input graph's edges.

</FuncElement>

<FuncElement name="node_edge_map" type="numba.typed.Dict">

A `numba` `Dict` with `node_data` indices as keys and `numba` `List` types as values containing the out-edge indices for each node.  

</FuncElement>

nX\_from\_graph\_maps
---------------------

<FuncSignature>
<pre>
nX_from_graph_maps(node_uids,
                   node_data,
                   edge_data,
                   node_edge_map,
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

<FuncElement name="node_data" type="np.ndarray">

A 2d `numpy` array representing the graph's nodes. The indices of the second dimension should correspond as follows:

| idx | property |
|-----|:----------|
| 0 | `x` coordinate |
| 1 | `y` coordinate |
| 2 | `bool` describing whether the node is `live` |
| 3 | `ghosted` describing whether the node is a 'ghosted' or 'decomposed' node that is not essential to the network topology. |

</FuncElement>

<FuncElement name="edge_data" type="np.ndarray">

A 2d `numpy` array representing the graph's directional edges. The indices of the second dimension should correspond as follows:

| idx | property |
|-----|:----------|
| 0 | start node `idx` |
| 1 | end node `idx` |
| 2 | the segment length in metres |
| 3 | the sum of segment's angular change |
| 4 | 'impedance factor' applied to magnify or reduce the edge impedance. |
| 5 | the edge's entry angular bearing |
| 6 | the edge's exit angular bearing |

</FuncElement>

<FuncElement name="node_edge_map" type="numba.typed.Dict">

A `numba` `Dict` with `node_data` indices as keys and `numba` `List` types as values containing the out-edge indices for each node.  

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

`x`, `y`, `live`, `ghosted` node attributes will be copied from `node_data` to the graph nodes. `length`, `angle_sum`, `imp_factor`, `start_bearing`, and `end_bearing` attributes will be copied from the `edge_data` to the graph edges.

If a `metrics_dict` is provided, all data will be copied to the graph nodes based on matching node identifiers.

</FuncElement>
