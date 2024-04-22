---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# graphs


 Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.


<div class="function">

## nx_simple_geoms


<div class="content">
<span class="name">nx_simple_geoms</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Inferring geometries from node to node. Infers straight-lined geometries connecting the `x` and `y` coordinates of each node-pair. The resultant edge geometry will be stored to each edge's `geom` attribute.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge `geom` attributes.</div>
</div>


</div>


<div class="function">

## nx_remove_filler_nodes


<div class="content">
<span class="name">nx_remove_filler_nodes</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Remove nodes of degree=2. Nodes of degree=2 represent no route-choice options other than traversal to the next edge. These are frequently found on network topologies as a means of describing roadway geometry, but are meaningless from a network topology point of view. This method will find and deleted these nodes, and replaces the two edges on either side with a new spliced edge. The new edge's `geom` attribute will retain the geometric properties of the original edges.

:::note
Filler nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented
through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` `Linestrings` to describe
arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus reducing
side-effects as a function of varied node intensities when computing network centralities.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with nodes of degree=2 removed. Adjacent edges will be combined into a unified new edge with associated `geom` attributes spliced together.</div>
</div>


</div>


<div class="function">

## nx_remove_dangling_nodes


<div class="content">
<span class="name">nx_remove_dangling_nodes</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">despine</span>
    <span class="pc">:</span>
    <span class="pa"> int = 15</span>
  </div>
  <div class="param">
    <span class="pn">remove_disconnected</span>
    <span class="pc">:</span>
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">cleanup_filler_nodes</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Remove disconnected components and optionally removes short dead-end street stubs.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">despine</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum cutoff distance for removal of dead-ends. Use `0` where no despining should occur.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">remove_disconnected</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Remove disconnected components with fewer nodes than specified by this parameter. Defaults to 100. Set to 0 to keep all disconnected components.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cleanup_filler_nodes</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to cleanup filler nodes. True by default.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with disconnected components optionally removed, and dead-ends removed where less than the `despine` parameter distance.</div>
</div>


</div>


<div class="function">

## nx_merge_parallel_edges


<div class="content">
<span class="name">nx_merge_parallel_edges</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">merge_edges_by_midline</span>
    <span class="pc">:</span>
    <span class="pa"> bool</span>
  </div>
  <div class="param">
    <span class="pn">contains_buffer_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Check a MultiGraph for duplicate edges; which, if found, will be merged. The shortest of these parallel edges is selected and buffered by `contains_buffer_dist`. If this buffer contains an adjacent edge, then the adjacent edge is merged. Edges falling outside this buffer are retained.

 When candidate edges are found for merging, they are replaced by a single new edge. The new geometry selected from either:
- An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
- Else, the shortest edge is retained, with longer edges discarded.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">merge_edges_by_midline</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be retained as the new geometry and the longer edges will be discarded. Defaults to True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">contains_buffer_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are sufficiently adjacent to be merged.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with consolidated nodes.</div>
</div>


</div>


<div class="function">

## nx_snap_endpoints


<div class="content">
<span class="name">nx_snap_endpoints</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Snaps geom endpoints to adjacent node coordinates.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph`.</div>
</div>


</div>


<div class="function">

## nx_iron_edges


<div class="content">
<span class="name">nx_iron_edges</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Simplifies edges.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with simplified edges.</div>
</div>


</div>


<div class="function">

## nx_consolidate_nodes


<div class="content">
<span class="name">nx_consolidate_nodes</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">buffer_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float = 12</span>
  </div>
  <div class="param">
    <span class="pn">neighbour_policy</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">crawl</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">centroid_by_itx</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">merge_edges_by_midline</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">contains_buffer_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 25</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Consolidates nodes if they are within a buffer distance of each other. Several parameters provide more control over the conditions used for deciding whether or not to merge nodes. The algorithm proceeds in two steps:

 Nodes within the buffer distance of each other are merged. If selecting `centroid_by_itx` then the new centroid will try to use intersections to determine the new centroid for the nodes. It will first attempt to find intersections with two through-routes, else will use intersections with one through-route.

 The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced by a single new edge, with the new geometry selected from either:
- An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
- Else, the shortest edge, with longer edges discarded; See [`nx_merge_parallel_edges`](#nx-merge-parallel-edges) for more information.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">buffer_dist</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The buffer distance to be used for consolidating nearby nodes. Defaults to 5.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">neighbour_policy</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Whether all nodes within the buffer distance are merged, or only &quot;direct&quot; or &quot;indirect&quot; neighbours. Defaults to None which will consider all nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">crawl</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the buffer distance from the prior node. Defaults to False.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">centroid_by_itx</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to favour intersections when selecting the combined centroid of merged nodes. Intersections with two straight through-routes will be favoured if found, otherwise intersections with one straight through-route are used where available. True by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">merge_edges_by_midline</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be retained as the new geometry and the longer edges will be discarded. Defaults to True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">contains_buffer_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are sufficiently adjacent to be merged. This is run after node consolidation has completed.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">nx.MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with consolidated nodes.</div>
</div>

### Notes

 See the guide on [graph cleaning](/guide#graph-cleaning) for more information.

![Example raw graph from OSM](/images/graph_cleaning_1.png) _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

![Example cleaned graph](/images/graph_cleaning_5.png) _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._

</div>


<div class="function">

## nx_split_opposing_geoms


<div class="content">
<span class="name">nx_split_opposing_geoms</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">buffer_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float = 12</span>
  </div>
  <div class="param">
    <span class="pn">merge_edges_by_midline</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">contains_buffer_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 25</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Split edges opposite nodes on parallel edge segments if within a buffer distance. This facilitates merging parallel roadways through subsequent use of [`nx_consolidate-nodes`](#nx-consolidate-nodes).

 The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced by a single new edge, with the new geometry selected from either:
- An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
- Else, the shortest edge, with longer edges discarded; See [`nx_merge_parallel_edges`](#nx-merge-parallel-edges) for more information.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">buffer_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The buffer distance to be used for splitting nearby nodes. Defaults to 5.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">merge_edges_by_midline</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be retained as the new geometry and the longer edges will be discarded. Defaults to True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">contains_buffer_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are sufficiently adjacent to be merged.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with consolidated nodes.</div>
</div>


</div>


<div class="function">

## nx_decompose


<div class="content">
<span class="name">nx_decompose</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">decompose_max</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Decomposes a graph so that no edge is longer than a set maximum. Decomposition provides a more granular representation of potential variations along street lengths, while reducing network centrality side-effects that arise as a consequence of varied node densities.

:::note
Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time
unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller
20m, and 50m may already be sufficient for the majority of cases.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">decompose_max</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The maximum length threshold for decomposed edges.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `imp_factor` edge attributes will be set to match the lengths of the new edges.</div>
</div>

### Notes

```python
from cityseer.tools import mock, graphs, plot

G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
G_decomposed = graphs.nx_decompose(G_simple, 100)
plot.plot_nx(G_decomposed)
```


![Example graph](/images/graph_simple.png) _Example graph prior to decomposition._

![Example decomposed graph](/images/graph_decomposed.png) _Example graph after decomposition._

</div>


<div class="function">

## nx_to_dual


<div class="content">
<span class="name">nx_to_dual</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Convert a primal graph representation to the dual representation. Primal graphs represent intersections as nodes and streets as edges. This method will invert this representation so that edges are converted to nodes and intersections become edges. Primal edge `geom` attributes will be welded to adjacent edges and split into the new dual edge `geom` attributes.

:::note
Note that a `MultiGraph` is useful for primal but not for dual, so the output `MultiGraph` will have single edges.
e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
primal. The same type of situation does not arise in the dual because the nodes map to distinct edges regardless.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding to the mid-points of the original primal edges. If `live` node attributes were provided, then the `live` attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set to `live=True`. Otherwise, all dual nodes wil be set to `live=True`. The primal edges will be split and welded to form the new dual `geom` edges. The primal `LineString` `geom` will be saved to the dual node's `primal_edge` attribute. `primal_edge_node_a`, `primal_edge_node_b`, and `primal_edge_idx` attributes will be added to the new (dual) nodes, and a `primal_node_id` edge attribute will be added to the new (dual) edges.</div>
</div>

### Notes

```python
from cityseer.tools import graphs, mock, plot

G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=False)
```


![Example dual graph](/images/graph_dual.png) _Dual graph (blue) overlaid on the source primal graph (red)._

</div>


<div class="function">

## nx_weight_by_dissolved_edges


<div class="content">
<span class="name">nx_weight_by_dissolved_edges</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">dissolve_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 20</span>
  </div>
  <div class="param">
    <span class="pn">max_ang_diff</span>
    <span class="pc">:</span>
    <span class="pa"> int = 45</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Generates graph node weightings based on the ratio of directly adjacent edges to total nearby edges. This is used to control for unintended amplification of centrality measures where redundant network representations (e.g. duplicitious segments such as adjacent street, sidewalk, cycleway, busway) tend to inflate centrality scores. This method is intended for 'messier' network representations (e.g. OSM).

 > This method is only recommended for primal graph representations.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dissolve_distance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 A distance to use when buffering edges to calculate the weighting. 20m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_ang_diff</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Only count a nearby adjacent edge as duplicitous if the angular difference between edges is less than `max_ang_diff`. 45 degrees by default.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` graph. The nodes will have a new `weight` parameter indicating the node's contribution given the locally 'dissolved' context.</div>
</div>


</div>


<div class="function">

## nx_generate_vis_lines


<div class="content">
<span class="name">nx_generate_vis_lines</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Generates a `line_geom` property for nodes consisting `MultiLineString` geoms for visualisation purposes. This method can be used if preferring to visualise the outputs as lines instead of points. The lines are assembled from the adjacent half segments.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` graph. The nodes will have a new `line_geom` parameter containing `shapely` `MultiLineString` geoms.</div>
</div>


</div>



</section>
