# Table of Contents

* [cityseer.tools.graphs](#cityseer.tools.graphs)
  * [nX\_simple\_geoms](#cityseer.tools.graphs.nX_simple_geoms)
  * [nX\_from\_osm](#cityseer.tools.graphs.nX_from_osm)
  * [nX\_wgs\_to\_utm](#cityseer.tools.graphs.nX_wgs_to_utm)
  * [nX\_remove\_dangling\_nodes](#cityseer.tools.graphs.nX_remove_dangling_nodes)
  * [nX\_remove\_filler\_nodes](#cityseer.tools.graphs.nX_remove_filler_nodes)
  * [nX\_consolidate\_nodes](#cityseer.tools.graphs.nX_consolidate_nodes)
  * [nX\_split\_opposing\_geoms](#cityseer.tools.graphs.nX_split_opposing_geoms)
  * [nX\_decompose](#cityseer.tools.graphs.nX_decompose)
  * [nX\_to\_dual](#cityseer.tools.graphs.nX_to_dual)
  * [graph\_maps\_from\_nX](#cityseer.tools.graphs.graph_maps_from_nX)
  * [nX\_from\_graph\_maps](#cityseer.tools.graphs.nX_from_graph_maps)

---
sidebar_label: graphs
title: cityseer.tools.graphs
---

General graph manipulation

<a name="cityseer.tools.graphs.nX_simple_geoms"></a>
#### nX\_simple\_geoms

<FuncSignature>

nX_simple_geoms(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Adds straight LineStrings between coordinates of nodes. Useful for converting OSM graphs to geometry.

<a name="cityseer.tools.graphs.nX_from_osm"></a>
#### nX\_from\_osm

<FuncSignature>

nX_from_osm(osm_json) -> nx.MultiGraph

</FuncSignature>

Parses an osm response into a NetworkX graph.

<a name="cityseer.tools.graphs.nX_wgs_to_utm"></a>
#### nX\_wgs\_to\_utm

<FuncSignature>

nX_wgs_to_utm(networkX_multigraph =None) -> nx.MultiGraph

</FuncSignature>

Converts a WGS CRS graph to a local UTM CRS.

<a name="cityseer.tools.graphs.nX_remove_dangling_nodes"></a>
#### nX\_remove\_dangling\_nodes

<FuncSignature>

nX_remove_dangling_nodes(networkX_multigraph = True) -> nx.MultiGraph

</FuncSignature>

Strips out all dead-end edges that are shorter than the despine parameter.
Removes disconnected components if remove_disconnected is set to True.

<a name="cityseer.tools.graphs.nX_remove_filler_nodes"></a>
#### nX\_remove\_filler\_nodes

<FuncSignature>

nX_remove_filler_nodes(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Iterates a networkX graph's nodes and removes nodes of degree = 2 (and with two neighbours).
The associated edges are replaced with new spliced geometry.

<a name="cityseer.tools.graphs.nX_consolidate_nodes"></a>
#### nX\_consolidate\_nodes

<FuncSignature>

nX_consolidate_nodes(networkX_multigraph = 100) -> nx.MultiGraph

</FuncSignature>

Consolidates nodes if they are within a buffer distance of each other.
Several parameters provide more control over the conditions used for deciding whether or not to merge nodes.
e.g. min_node_threshold, min_node_degree, min_cumulative_degree, and max_cumulative_degree.
The neighbour_policy controls whether all nodes are merged or only "direct" or "indirect" neighbours.
The crawl parameter controls whether the algorithm will recursively explore adjacent nodes (within buffer distance).
The merge_edges_by_midline controls whether edges are merged by an imaginary centreline.
If set to False, then the shortest edge is used instead.
The cent_min_degree parameter and cent_min_len control how the new centroids are selected, and are described in
the _squash_adjacent method.
The max_len_discrepancy and discrepancy_min_len parameters are used for deciding when to keep separate edges, and
are described in the _merge_parallel_edges method.

<a name="cityseer.tools.graphs.nX_split_opposing_geoms"></a>
#### nX\_split\_opposing\_geoms

<FuncSignature>

nX_split_opposing_geoms(networkX_multigraph = 100) -> nx.MultiGraph

</FuncSignature>

Projects nodes to pierce opposing edges within the buffer distance.
The pierced nodes are used for facilitating node merging for scenarios such as divided boulevards.
The max_len_discrepancy and discrepancy_min_len parameters are used for deciding when to keep separate edges, and
are described in the _merge_parallel_edges method.

<a name="cityseer.tools.graphs.nX_decompose"></a>
#### nX\_decompose

<FuncSignature>

nX_decompose(networkX_multigraph, decompose_max: float) -> nx.MultiGraph

</FuncSignature>

Decomposes a MultiGraph so that no edge is longer than the specified decompose_max.

<a name="cityseer.tools.graphs.nX_to_dual"></a>
#### nX\_to\_dual

<FuncSignature>

nX_to_dual(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph

</FuncSignature>

Converts a primal MultiGraph to a dual MultiGraph.
Note that a MultiGraph is useful for primal but not for dual, so the output MultiGraph will have single edges.
e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
primal. The same type of situation does not arise in the dual because the nodes map to distinct streets regardless.

<a name="cityseer.tools.graphs.graph_maps_from_nX"></a>
#### graph\_maps\_from\_nX

<FuncSignature>

graph_maps_from_nX(networkX_multigraph: nx.MultiGraph) -> Tuple[tuple, np.ndarray, np.ndarray, Dict]

</FuncSignature>

Generates node and edge data maps from a MultiGraph.
Calculates length and angle attributes, as well as in and out bearings and stores these in the data maps.

<a name="cityseer.tools.graphs.nX_from_graph_maps"></a>
#### nX\_from\_graph\_maps

<FuncSignature>

nX_from_graph_maps(node_uids = None) -> nx.MultiGraph

</FuncSignature>

Writes cityseer data graph maps back to a MultiGraph. Can write back to an existing MultiGraph if provided.

