---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# graph


 Graph data structures and utilities for network analysis.


<div class="class">


## NodePayload




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

</div>

 
</div>


<div class="class">


## EdgePayload




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

</div>

 
</div>


<div class="class">


## NetworkStructure




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
  <div class="param">
    <span class="pn">jitter_scale=None</span>
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
  <div class="param">
    <span class="pn">jitter_scale=None</span>
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
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## local_node_centrality_shortest


<div class="content">
<span class="name">local_node_centrality_shortest</span><div class="signature multiline">
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
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## local_node_centrality_simplest


<div class="content">
<span class="name">local_node_centrality_simplest</span><div class="signature multiline">
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
    <span class="pn">angular_scaling_unit=None</span>
  </div>
  <div class="param">
    <span class="pn">farness_scaling_offset=None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## local_segment_centrality


<div class="content">
<span class="name">local_segment_centrality</span><div class="signature multiline">
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
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

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

## add_node


<div class="content">
<span class="name">add_node</span><div class="signature multiline">
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

## get_node_payload


<div class="content">
<span class="name">get_node_payload</span><div class="signature multiline">
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

</div>

 

<div class="function">

## add_edge


<div class="content">
<span class="name">add_edge</span><div class="signature multiline">
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
    <span class="pn">start_nd_key</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key</span>
  </div>
  <div class="param">
    <span class="pn">length</span>
  </div>
  <div class="param">
    <span class="pn">angle_sum</span>
  </div>
  <div class="param">
    <span class="pn">imp_factor</span>
  </div>
  <div class="param">
    <span class="pn">in_bearing</span>
  </div>
  <div class="param">
    <span class="pn">out_bearing</span>
  </div>
  <div class="param">
    <span class="pn">seconds</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

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

## get_edge_payload


<div class="content">
<span class="name">get_edge_payload</span><div class="signature multiline">
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

## assign_to_network


<div class="content">
<span class="name">assign_to_network</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## find_nearest


<div class="content">
<span class="name">find_nearest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## road_distance


<div class="content">
<span class="name">road_distance</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">nd_a_idx</span>
  </div>
  <div class="param">
    <span class="pn">nd_b_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## closest_intersections


<div class="content">
<span class="name">closest_intersections</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">pred_map</span>
  </div>
  <div class="param">
    <span class="pn">last_nd_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>



</section>
