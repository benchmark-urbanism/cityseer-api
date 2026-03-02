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

</div>

 

<div class="function">

## dijkstra_tree_segment


<div class="content">
<span class="name">dijkstra_tree_segment</span><div class="signature multiline">
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

</div>

 

<div class="function">

## closeness_shortest


<div class="content">
<span class="name">closeness_shortest</span><div class="signature multiline">
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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
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
    <span class="pn">source_indices=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## closeness_simplest


<div class="content">
<span class="name">closeness_simplest</span><div class="signature multiline">
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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">angular_scaling_unit=None</span>
  </div>
  <div class="param">
    <span class="pn">farness_scaling_offset=None</span>
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
    <span class="pn">source_indices=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## segment_centrality


<div class="content">
<span class="name">segment_centrality</span><div class="signature multiline">
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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## betweenness_shortest


<div class="content">
<span class="name">betweenness_shortest</span><div class="signature multiline">
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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">source_indices=None</span>
  </div>
  <div class="param">
    <span class="pn">sample_probability=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute Brandes betweenness centrality from all sources or a specified subset. When `source_indices` is None, iterates all live source nodes (exact). When `source_indices` is provided with `sample_probability`, applies IPW scaling (1/(2p)) for an unbiased estimate — matching the closeness framework. When `source_indices` is provided without `sample_probability`, scales by n_live / (2 * n_sources) for backwards compatibility.

 Returns a `BetweennessShortestResult` with betweenness fields populated.

</div>

 

<div class="function">

## betweenness_simplest


<div class="content">
<span class="name">betweenness_simplest</span><div class="signature multiline">
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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">source_indices=None</span>
  </div>
  <div class="param">
    <span class="pn">sample_probability=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute Brandes betweenness centrality using simplest (angular) paths. Mirrors `betweenness_shortest` but uses `dijkstra_brandes_simplest` which optimises for minimum angular change (with sidestepping prevention). Distance thresholds are still in metres (physical distance along the simplest path).

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
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
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


 Compute OD-weighted betweenness centrality using shortest paths. Uses Brandes multi-predecessor Dijkstra from each source that has outbound OD trips. For each OD destination, backpropagates credit through all equal shortest paths, weighted by the OD flow weight and split by sigma (path count).

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
  <span class="pt">)</span>
</div>
</div>

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
  <span class="pt">)</span>
</div>
</div>

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
  <span class="pt">)</span>
</div>
</div>


 Adds a street edge with geometry. Calculates length, bearings, and angle sum from WKT. Sets seconds to NaN.

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

 

<span class="name">node_ys</span>


 

<span class="name">node_xys</span>


 

<span class="name">node_xs</span>


 

<span class="name">street_node_lives</span>


 
</div>



</section>
