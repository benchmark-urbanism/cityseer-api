---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# observe


 Observe module for computing observations derived from `networkX` graphs. These methods are generally sufficiently simple that further computational optimisation is not required. Network centrality methods (which do require further computational optimisation due to their complexity) are handled separately in the [`networks`](/metrics/networks) module.


<div class="class">


## ContinuityEntry



 State management for an individual street continuity entry. This corresponds to an individual street name, route name or number, or highway type.



<div class="function">

## ContinuityEntry


<div class="content">
<span class="name">ContinuityEntry</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">entry_name</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Instances a continuity entry.

</div>

 

<div class="function">

## generate_key

<div class="decorator">@staticmethod</div>

<div class="content">
<span class="name">generate_key</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">start_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Generate a unique key given uncertainty of start and end node order.

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
    <span class="pn">length</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Adds edge details to a continuity entry.

</div>

 
</div>


<div class="class">


## StreetContinuityReport



 State management for a collection of street continuity metrics. Each key in the `entries` attribute corresponds to a `ContinuityEntry`.



<div class="function">

## StreetContinuityReport


<div class="content">
<span class="name">StreetContinuityReport</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">method</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Instance a street continuity report.

</div>

 

<div class="function">

## scaffold_entry


<div class="content">
<span class="name">scaffold_entry</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">entry_name</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Adds a new continuity entry to the report's entries.

</div>

 

<div class="function">

## report_by_count


<div class="content">
<span class="name">report_by_count</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">n_items</span>
    <span class="pc">:</span>
    <span class="pa"> int = 10</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Print a report sorted by entry counts.

</div>

 

<div class="function">

## report_by_length


<div class="content">
<span class="name">report_by_length</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">n_items</span>
    <span class="pc">:</span>
    <span class="pa"> int = 10</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Print a report sorted by entry lengths.

</div>

 
</div>


<div class="function">

## street_continuity


<div class="content">
<span class="name">street_continuity</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
  </div>
  <div class="param">
    <span class="pn">method</span>
    <span class="pc">:</span>
    <span class="pa"> str | tuple[str, str]</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">MultiGraph</span>
  <span class="pr">StreetContinuityReport</span>
  <span class="pt">]</span>
</div>
</div>


 Compute the street continuity for a given graph. This requires a graph with `names`, `routes`, or `highways` edge keys corresponding to the selected `method` parameter. These keys are available if importing an OSM network with [`osm_graph_from_poly`](/tools/io#osm-graph-from-poly) or if importing OS Open Roads data with [nx_from_open_roads](/tools/io#nx-from-open-roads).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms. Edges should contain &quot;names&quot;, &quot;routes&quot;, or &quot;highways&quot; keys corresponding to the specified `method` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">method</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The type of continuity metric to compute, where available options are &quot;names&quot;, &quot;routes&quot;, or &quot;highways&quot;.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A copy of the input `networkX` `MultiGraph` with new edge keys corresponding to the calculated route continuity metric. The metrics will be stored in 'length' and 'count' forms for the specified method, with keys formatted according to `f&quot;{method}_cont_by_{form}&quot;`. For example, when computing &quot;names&quot; continuity, the `names_cont_by_count` and `names_cont_by_length` keys will be added to the returned `networkX` `MultiGraph`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">StreetContinuityReport</div>
  </div>
  <div class="desc">

 An instance of [`StreetContinuityReport`](/metrics/observe#streetcontinuityreport) containing the computed state for the selected method.</div>
</div>


</div>


<div class="function">

## hybrid_street_continuity


<div class="content">
<span class="name">hybrid_street_continuity</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">MultiGraph</span>
  <span class="pr">StreetContinuityReport</span>
  <span class="pt">]</span>
</div>
</div>


 Compute the street continuity for a given graph using a hybridisation of routes and names continuity. Hybrid continuity merges route continuity and street continuity information where a route overlaps a street continuity.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom` edge attributes containing `LineString` geoms. Edges should contain &quot;names&quot;, &quot;routes&quot;, or &quot;highways&quot; keys corresponding to the specified `method` parameter.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A copy of the input `networkX` `MultiGraph` with new edge keys corresponding to the calculated route continuity metric. The metrics will be stored in 'hybrid_cont_by_length' and 'hybrid_cont_by_count' keys.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">StreetContinuityReport</div>
  </div>
  <div class="desc">

 An instance of [`StreetContinuityReport`](/metrics/observe#streetcontinuityreport) containing the computed state for the &quot;hybrid&quot; method.</div>
</div>


</div>



</section>
