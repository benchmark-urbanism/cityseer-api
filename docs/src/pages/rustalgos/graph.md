---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# graph


 Graph data structures and utilities for network analysis.


<div class="class">


## NodePayload



 Payload for a network node.



<div class="function">

## NodePayload


<div class="content">
<span class="name">NodePayload</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Validates the payload. Returns Ok(()) if valid, Err(PyValueError) otherwise.

</div>

 
</div>


<div class="class">


## EdgePayload



 Payload for a network edge.



<div class="function">

## EdgePayload


<div class="content">
<span class="name">EdgePayload</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Validates the payload. Returns Ok(()) if valid, Err(PyValueError) otherwise.

</div>

 
</div>


<div class="class">


## NetworkStructure



 Main network structure.



<div class="function">

## NetworkStructure


<div class="content">
<span class="name">NetworkStructure</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## dijkstra_tree_shortest


<div class="content">
<span class="name">dijkstra_tree_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">src_idx</span>
  </div>
  <div class="param">
    <span class="pn">max_seconds</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## dijkstra_tree_simplest


<div class="content">
<span class="name">dijkstra_tree_simplest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">src_idx</span>
  </div>
  <div class="param">
    <span class="pn">max_seconds</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## centrality_shortest


<div class="content">
<span class="name">centrality_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">closeness_exprs=None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness_exprs=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_cycles=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">sample_probability=None</span>
  </div>
  <div class="param">
    <span class="pn">sampling_weights=None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute node centrality using shortest paths with a single Dijkstra per source. Closeness and betweenness metrics are specified as lists of (name, expression) pairs. Expressions use variables `c` (metric distance) and `p` (normalised progress = c / threshold). Each expression is parsed once per thread via `meval` and evaluated per reached node (closeness) or per shortest path (betweenness).

 When `sample_probability` is set, Bernoulli sampling with inverse-probability weighting (IPW) is used.

</div>

 

<div class="function">

## centrality_simplest


<div class="content">
<span class="name">centrality_simplest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">closeness_exprs=None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness_exprs=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">sample_probability=None</span>
  </div>
  <div class="param">
    <span class="pn">sampling_weights=None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute node centrality using simplest (angular) paths on the dual graph. Angular routing is evaluated on two directed states per segment. Each source segment seeds both orientations into a single Brandes traversal.

 Expressions use `c` (angular cost) and `p` (normalised time progress = agg_seconds / max_seconds).

</div>

 

<div class="function">

## betweenness_od_shortest


<div class="content">
<span class="name">betweenness_od_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">od_matrix</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness_exprs=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute OD-weighted betweenness centrality using shortest paths. Uses Brandes multi-predecessor Dijkstra from each source that has outbound OD trips. Betweenness expressions use `c` (metric distance) and `p` (normalised progress = c / threshold).

</div>

 

<div class="function">

## betweenness_demand_shortest


<div class="content">
<span class="name">betweenness_demand_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">origins</span>
  </div>
  <div class="param">
    <span class="pn">destinations</span>
  </div>
  <div class="param">
    <span class="pn">decay_fn</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">closest_destination=False</span>
  </div>
  <div class="param">
    <span class="pn">metric_name=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Demand-weighted (flow) betweenness from a singly / origin-constrained spatial interaction model. Each origin distributes its full weight across reachable destinations in proportion to `W_d * decay(c)`, where `decay` is the supplied expression evaluated on `c` (metric cost) and `p` (normalised progress to the threshold) — the gravity model is one instance of this spatial interaction form. The allocated origin-destination flows are then routed along shortest paths via Brandes back-propagation, accumulating flow betweenness at intermediate nodes. Origins and destinations are each aggregated by node first, so several snapped points sharing a node contribute their summed weight (and a node only triggers one Dijkstra). When `closest_destination` is true, an origin routes its full weight to its single nearest reachable destination instead of allocating across all of them.

</div>

 

<div class="function">

## progress_init


<div class="content">
<span class="name">progress_init</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## progress


<div class="content">
<span class="name">progress</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## set_is_dual


<div class="content">
<span class="name">set_is_dual</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">is_dual</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## set_is_directed


<div class="content">
<span class="name">set_is_directed</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">is_directed</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## add_street_node


<div class="content">
<span class="name">add_street_node</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_key</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
  </div>
  <div class="param">
    <span class="pn">live</span>
  </div>
  <div class="param">
    <span class="pn">weight</span>
  </div>
  <div class="param">
    <span class="pn">z=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## add_transport_node


<div class="content">
<span class="name">add_transport_node</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_key</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
  </div>
  <div class="param">
    <span class="pn">linking_radius=None</span>
  </div>
  <div class="param">
    <span class="pn">z=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## get_node_payload_py


<div class="content">
<span class="name">get_node_payload_py</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## get_node_weight


<div class="content">
<span class="name">get_node_weight</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## set_node_weight


<div class="content">
<span class="name">set_node_weight</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <div class="param">
    <span class="pn">weight</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Set the weight of a node.

</div>

 

<div class="function">

## is_node_live


<div class="content">
<span class="name">is_node_live</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## set_node_live


<div class="content">
<span class="name">set_node_live</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <div class="param">
    <span class="pn">live</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Set the live status of a node (e.g. based on a boundary polygon).

</div>

 

<div class="function">

## node_count


<div class="content">
<span class="name">node_count</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns the total count of all nodes (street and transport).

</div>

 

<div class="function">

## node_bound


<div class="content">
<span class="name">node_bound</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns an upper bound on node indices (all valid indices are < node_bound). Use this instead of node_count() when allocating index-addressed vectors, because StableGraph may have gaps after node removal.

</div>

 

<div class="function">

## edge_bound


<div class="content">
<span class="name">edge_bound</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns an upper bound on edge indices (all valid indices are < edge_bound).

</div>

 

<div class="function">

## street_node_count


<div class="content">
<span class="name">street_node_count</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns the count of non-transport (street) nodes.

</div>

 

<div class="function">

## node_indices


<div class="content">
<span class="name">node_indices</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns a list of indices for all nodes (street and transport).

</div>

 

<div class="function">

## node_keys_py


<div class="content">
<span class="name">node_keys_py</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns a list of original keys for all nodes (street and transport).

</div>

 

<div class="function">

## street_node_indices


<div class="content">
<span class="name">street_node_indices</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Returns a list of indices for non-transport (street) nodes.

</div>

 

<div class="function">

## add_street_edge


<div class="content">
<span class="name">add_street_edge</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_key_py</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key_py</span>
  </div>
  <div class="param">
    <span class="pn">geom_wkt</span>
  </div>
  <div class="param">
    <span class="pn">imp_factor=None</span>
  </div>
  <div class="param">
    <span class="pn">shared_primal_node_key=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Adds a street edge with geometry. Calculates length, bearings, and angle sum from WKT. Sets seconds to NaN.

</div>

 

<div class="function">

## remove_street_node


<div class="content">
<span class="name">remove_street_node</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Remove a street node and all its connected edges from the StableGraph. StableGraph::remove_node() cascades to all edges connected to the node, and preserves existing indices for other nodes (no swap-and-compact). This means node indices held externally (e.g. by the QGIS plugin's `node_idx` dict) remain valid after removal.

 Returns an error if the node does not exist or is a transport node.

</div>

 

<div class="function">

## remove_street_edge


<div class="content">
<span class="name">remove_street_edge</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Remove a specific directed edge identified by its start/end node indices and edge_idx. Other edge indices remain stable after removal (StableGraph guarantee).

</div>

 

<div class="function">

## add_transport_edge


<div class="content">
<span class="name">add_transport_edge</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_key_py</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key_py</span>
  </div>
  <div class="param">
    <span class="pn">seconds</span>
  </div>
  <div class="param">
    <span class="pn">imp_factor=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Adds an abstract transport edge defined by travel time (seconds). Length is set to NaN. Geometry-related fields are NaN/None.

</div>

 

<div class="function">

## edge_references


<div class="content">
<span class="name">edge_references</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## get_edge_payload_py


<div class="content">
<span class="name">get_edge_payload_py</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## get_edge_length


<div class="content">
<span class="name">get_edge_length</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## get_edge_impedance


<div class="content">
<span class="name">get_edge_impedance</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 The type of the None singleton.

</div>

 

<div class="function">

## build_edge_rtree


<div class="content">
<span class="name">build_edge_rtree</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Builds the R-tree for street edge geometries using their bounding boxes. Deduplicates edges based on sorted node pairs and geometric equality. Stores (start_node_idx, end_node_idx, start_node_point, end_node_point, edge_geom) in the R-tree data payload.

</div>

 

<div class="function">

## set_barriers


<div class="content">
<span class="name">set_barriers</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">barriers_wkt</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Sets barrier geometries from WKT strings and builds the R-tree. Replaces any existing barriers.

</div>

 

<div class="function">

## unset_barriers


<div class="content">
<span class="name">unset_barriers</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Removes all barrier geometries and the associated R-tree.

</div>

 

<span class="name">node_xs</span>


 

<span class="name">node_zs</span>


 

<span class="name">node_xys</span>


 

<span class="name">node_xyzs</span>


 

<span class="name">node_ys</span>


 

<span class="name">street_node_lives</span>


 
</div>



</section>
