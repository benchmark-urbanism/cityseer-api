# cityseer.tools.graphs

A collection of convenience functions for the preparation and conversion of [`NetworkX`](https://networkx.github.io/)
graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and
manipulated directly, if so desired.

### nX\_simple\_geoms

<FuncSignature>

nX_simple_geoms(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Generates straight-line geometries for each edge based on the the `x` and `y` coordinates of the adjacent nodes.
The edge geometry will be stored to the edge `geom` attribute.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` with `x` and `y` node attributes.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge `geom` attributes.

</FuncElement>


### nX\_from\_osm

<FuncSignature>

nX_from_osm(osm_json: str) -> nx.MultiGraph

</FuncSignature>

Generates a `NetworkX` `MultiGraph` from [`Open Street Map`](https://www.openstreetmap.org) data.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="osm_json" type="str">

A `json` string response from the [OSM overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API), consisting of `nodes` and `ways`.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `NetworkX` `MultiGraph` with `x` and `y` attributes in [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates.

</FuncElement>


### nX\_wgs\_to\_utm

<FuncSignature>

nX_wgs_to_utm(networkX_multigraph = None) -> nx.MultiGraph

</FuncSignature>

Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the
local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries
will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all
other nodes and geometries contained in the graph. This ensures consistent behaviour in cases where a graph spans
a UTM boundary.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge attributes containing `LineString` geoms to be converted.

</FuncElement>

<FuncElement name="force_zone_number" type="int">

An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched UTM zones may introduce substantial distortions in the results. Defaults to None.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge `geom` attributes are present, these will also be converted.

</FuncElement>


### nX\_remove\_dangling\_nodes

<FuncSignature>

nX_remove_dangling_nodes(networkX_multigraph = True) -> nx.MultiGraph

</FuncSignature>

Optionally removes short dead-ends or disconnected graph components, which may be prevalent on poor quality network
datasets.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.

</FuncElement>

<FuncElement name="despine" type="float">

The maximum cutoff distance for removal of dead-ends. Use `0` where no despining should occur. Defaults to None.

</FuncElement>

<FuncElement name="remove_disconnected" type="bool">

Whether to remove disconnected components. If set to `True`, only the largest connected component will be returned. Defaults to True.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with disconnected components optionally removed, and dead-ends removed where less than the `despine` parameter distance.

</FuncElement>


### nX\_remove\_filler\_nodes

<FuncSignature>

nX_remove_filler_nodes(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Removes nodes of degree=2: such nodes represent no route-choices other than traversal to the next edge.
The edges on either side of the deleted nodes will be removed and replaced with a new spliced edge.

:::tip Note
Filler nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented
through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` [`Linestrings`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings)
to describe arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus
reducing side-effects when computing network centralities, which arise as a function of varied node intensities.
:::

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with nodes of degree=2 removed. Adjacent edges will be combined into a unified new edge with associated `geom` attributes spliced together.

</FuncElement>


### nX\_consolidate\_nodes

<FuncSignature>

nX_consolidate_nodes(networkX_multigraph = 100) -> nx.MultiGraph

</FuncSignature>

Consolidates nodes if they are within a buffer distance of each other. Several parameters provide more control over
the conditions used for deciding whether or not to merge nodes. The algorithm proceeds in two steps:
- Firstly, nodes within the buffer distance of each other are merged. A new centroid will be determined and all
existing edge endpoints will be updated accordingly. The new centroid for the merged nodes can be based on:
- The centroid of the node group;
- Else, all nodes of degree greater or equal to `cent_min_degree`;
- Else, all nodes with aggregate adjacent edge lengths greater than a factor of `cent_min_len_factor` of the node
with the greatest aggregate length for adjacent edges.
- Secondly, the merging of nodes creates parallel edges which may start and end at a shared node on either side.
These edges are replaced by a single new edge, with the new geometry selected from either:
- An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
- Else, the shortest edge, with longer edges discarded.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.

</FuncElement>

<FuncElement name="buffer_dist" type="float">

The buffer distance to be used for consolidating nearby nodes. Defaults to 5.

</FuncElement>

<FuncElement name="min_node_group" type="int">

The minimum number of nodes to consider a valid group for consolidation. Defaults to 2.

</FuncElement>

<FuncElement name="min_node_degree" type="int">

The least number of edges a node should have in order to be considered for consolidation. Defaults to 1.

</FuncElement>

<FuncElement name="min_cumulative_degree" type="int">

An optional minimum cumulative degree to consider a valid node group for consolidation. Defaults to None.

</FuncElement>

<FuncElement name="max_cumulative_degree" type="int, optional">

An optional maximum cumulative degree to consider a valid node group for consolidation. Defaults to None.

</FuncElement>

<FuncElement name="neighbour_policy" type="str">

Whether all nodes within the buffer distance are merged, or only "direct" or "indirect" neighbours. Defaults to None.

</FuncElement>

<FuncElement name="crawl" type="bool, optional">

Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the buffer distance from the prior node. Defaults to True.

</FuncElement>

<FuncElement name="cent_min_degree" type="int">

The minimum node degree for a node to be considered when calculating the new centroid for the merged node cluster. Defaults to 3.

</FuncElement>

<FuncElement name="cent_min_len_factor" type="float">

The minimum aggregate adjacent edge lengths an existing node should have to be considered when calculating the centroid for the new node cluster. Expressed as a factor of the node with the greatest aggregate adjacent edge lengths. Defaults to None.

</FuncElement>

<FuncElement name="merge_edges_by_midline" type="bool">

Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be retained as the new geometry and the longer edges will be discarded. Defaults to True.

</FuncElement>

<FuncElement name="multi_edge_len_factor" type="float">

In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest length but with the exception that (longer) edges still shorter than multi_edge_min_len are removed regardless. Defaults to 1.5.

</FuncElement>

<FuncElement name="multi_edge_min_len" type="float">

See `multi_edge_len_factor`. Defaults to 100.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with consolidated nodes.

</FuncElement>

  

<FuncHeading>

Example

</FuncHeading>


  See the guide on [graph cleaning](/guide/cleaning) for more information.
  
  ![Example raw graph from OSM](../.vitepress/plots/images/graph_cleaning_1.png)
  _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._
  
  ![Example cleaned graph](../.vitepress/plots/images/graph_cleaning_5.png)
  _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._

### nX\_split\_opposing\_geoms

<FuncSignature>

nX_split_opposing_geoms(networkX_multigraph = 100) -> nx.MultiGraph

</FuncSignature>

Projects nodes to pierce opposing edges within a buffer distance. The pierced nodes facilitate subsequent node
merging for scenarios such as divided boulevards.

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.

</FuncElement>

<FuncElement name="buffer_dist" type="float">

The buffer distance to be used for splitting nearby nodes. Defaults to 5.

</FuncElement>

<FuncElement name="merge_edges_by_midline" type="bool">

Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be retained as the new geometry and the longer edges will be discarded. Defaults to True.

</FuncElement>

<FuncElement name="multi_edge_len_factor" type="float">

In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest length but with the exception that (longer) edges still shorter than multi_edge_min_len are removed regardless. Defaults to 1.5.

</FuncElement>

<FuncElement name="multi_edge_min_len" type="float">

See `multi_edge_len_factor`. Defaults to 100.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A `networkX` `MultiGraph` with consolidated nodes.

</FuncElement>


### nX\_decompose

<FuncSignature>

nX_decompose(networkX_multigraph,
             decompose_max: float) -> nx.MultiGraph

</FuncSignature>

Decomposes a graph so that no edge is longer than a set maximum. Decomposition provides a more granular
representation of potential variations along street lengths, while reducing network centrality side-effects that
arise as a consequence of varied node densities.

::: warning Note

Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time
unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller
20m, and 50m may already be sufficient for the majority of cases.

:::

::: tip Hint

This function will automatically orient the `geom` attribute LineStrings in the correct direction before splitting
into sub-geometries; i.e. there is no need to order the geometry's coordinates in a particular direction.

:::

<FuncHeading>

Arguments

</FuncHeading>


<FuncElement name="networkX_multigraph" type="nx.MultiGraph">

A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.

</FuncElement>

  
<FuncElement name="decompose_max" type="float">

The maximum length threshold for decomposed edges.

</FuncElement>

  

<FuncHeading>

Returns

</FuncHeading>


<FuncElement name="nx.MultiGraph">

A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `impedance` edge attributes will be set to match the lengths of the new edges.

</FuncElement>

  

<FuncHeading>

Example

</FuncHeading>


  
  ```python
  from cityseer.tools import mock, graphs, plot
  
  G = mock.mock_graph()
  G_simple = graphs.nX_simple_geoms(G)
  G_decomposed = graphs.nX_decompose(G_simple, 100)
  plot.plot_nX(G_decomposed)
  ```
  
  ![Example graph](../.vitepress/plots/images/graph_simple.png)
  _Example graph prior to decomposition._
  
  ![Example decomposed graph](../.vitepress/plots/images/graph_decomposed.png)
  _Example graph after decomposition._

### nX\_to\_dual

<FuncSignature>

nX_to_dual(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Converts a primal `MultiGraph` to a dual `MultiGraph`.
Note that a `MultiGraph` is useful for primal but not for dual, so the output `MultiGraph` will have single edges.
e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
primal. The same type of situation does not arise in the dual because the nodes map to distinct streets regardless.

### graph\_maps\_from\_nX

<FuncSignature>

graph_maps_from_nX(networkX_multigraph: nx.MultiGraph) -> Tuple[tuple, np.ndarray, np.ndarray, Dict]

</FuncSignature>

Generates node and edge data maps from a `MultiGraph`.
Calculates length and angle attributes, as well as in and out bearings and stores these in the data maps.

### nX\_from\_graph\_maps

<FuncSignature>

nX_from_graph_maps(node_uids = None) -> nx.MultiGraph

</FuncSignature>

Writes cityseer data graph maps back to a `MultiGraph`. Can write back to an existing `MultiGraph` if provided.

