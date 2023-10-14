---
layout: ../../layouts/PageLayout.astro
---

# util


 Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.


<div class="function">

## measure_bearing


<div class="content">
<span class="name">measure_bearing</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">xy_1</span><span class="p">:</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]]</span>,</span><span class="param">	<span class="n">xy_2</span><span class="p">:</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]]</span></span><span class="return-annotation">) -> <span class="nb">float</span>:</span></span>
</div>


 Measures the angular bearing between two coordinate pairs.

</div>


<div class="function">

## measure_coords_angle


<div class="content">
<span class="name">measure_coords_angle</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">coords_1</span><span class="p">:</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]]</span>,</span><span class="param">	<span class="n">coords_2</span><span class="p">:</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]]</span>,</span><span class="param">	<span class="n">coords_3</span><span class="p">:</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]]</span></span><span class="return-annotation">) -> <span class="nb">float</span>:</span></span>
</div>


 Measures angle between three coordinate pairs.

</div>


<div class="function">

## measure_cumulative_angle


<div class="content">
<span class="name">measure_cumulative_angle</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">linestring_coords</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span></span><span class="return-annotation">) -> <span class="nb">float</span>:</span></span>
</div>


 Measures the cumulative angle along a LineString geom's coords.

</div>


<div class="function">

## substring


<div class="content">
<span class="name">substring</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">geom</span>, </span><span class="param"><span class="n">start_dist</span>, </span><span class="param"><span class="n">end_dist</span>, </span><span class="param"><span class="n">normalized</span><span class="o">=</span><span class="kc">False</span></span><span class="return-annotation">):</span></span>
</div>


 Temporary copy of shapely substring method until issue #1699 is fixed (re: z coords).

</div>


<div class="function">

## snap_linestring_startpoint


<div class="content">
<span class="name">snap_linestring_startpoint</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">linestring_coords</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]],</span> <span class="n">shapely</span><span class="o">.</span><span class="n">coords</span><span class="o">.</span><span class="n">CoordinateSequence</span><span class="p">]</span>,</span><span class="param">	<span class="n">x_y</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]</span></span><span class="return-annotation">) -> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>:</span></span>
</div>


 Snaps a LineString's start-point coordinate to a specified x_y coordinate.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">linestring_coords</div>
    <div class="type">tuple | list | np.ndarray</div>
  </div>
  <div class="desc">

 A list, tuple, or numpy array of x, y coordinate tuples.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x_y</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple of floats representing the target x, y coordinates against which to align the linestring start point.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">linestring_coords</div>
  </div>
  <div class="desc">

 A list of linestring coords aligned to the specified starting point.</div>
</div>


</div>


<div class="function">

## snap_linestring_endpoint


<div class="content">
<span class="name">snap_linestring_endpoint</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">linestring_coords</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]],</span> <span class="n">shapely</span><span class="o">.</span><span class="n">coords</span><span class="o">.</span><span class="n">CoordinateSequence</span><span class="p">]</span>,</span><span class="param">	<span class="n">x_y</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]</span></span><span class="return-annotation">) -> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>:</span></span>
</div>


 Snaps a LineString's end-point coordinate to a specified x_y coordinate.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">linestring_coords</div>
    <div class="type">tuple | list | np.ndarray</div>
  </div>
  <div class="desc">

 A list, tuple, or numpy array of x, y coordinate tuples.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x_y</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple of floats representing the target x, y coordinates against which to align the linestring end point.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">linestring_coords</div>
  </div>
  <div class="desc">

 A list of linestring coords aligned to the specified ending point.</div>
</div>


</div>


<div class="function">

## align_linestring_coords


<div class="content">
<span class="name">align_linestring_coords</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">linestring_coords</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]],</span> <span class="n">shapely</span><span class="o">.</span><span class="n">coords</span><span class="o">.</span><span class="n">CoordinateSequence</span><span class="p">]</span>,</span><span class="param">	<span class="n">x_y</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]</span>,</span><span class="param">	<span class="n">reverse</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">tolerance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span></span><span class="return-annotation">) -> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>:</span></span>
</div>


 Align a LineString's coordinate order to either start or end at a specified x_y coordinate within a given tolerance.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">linestring_coords</div>
    <div class="type">tuple | list | np.ndarray</div>
  </div>
  <div class="desc">

 A list, tuple, or numpy array of x, y coordinate tuples.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x_y</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple of floats representing the target x, y coordinates against which to align the linestring coords.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">reverse</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If reverse=False the coordinate order will be aligned to start from the given x_y coordinate. If reverse=True the coordinate order will be aligned to end at the given x_y coordinate.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Distance tolerance in metres for matching the x_y coordinate to the linestring_coords.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">linestring_coords</div>
  </div>
  <div class="desc">

 A list of linestring coords aligned to the specified endpoint.</div>
</div>


</div>


<div class="function">

## snap_linestring_endpoints


<div class="content">
<span class="name">snap_linestring_endpoints</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">networkx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">multigraph</span><span class="o">.</span><span class="n">MultiGraph</span>,</span><span class="param">	<span class="n">start_nd_key</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">end_nd_key</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">linestring_coords</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>,</span><span class="param">	<span class="n">tolerance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span></span><span class="return-annotation">) -> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>:</span></span>
</div>


 Snaps edge geom coordinate sequence to the nodes on either side.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes and edge `geom` attributes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">start_nd_key</div>
    <div class="type">NodeKey</div>
  </div>
  <div class="desc">

 A node key corresponding to the edge's start node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">end_nd_key</div>
    <div class="type">NodeKey</div>
  </div>
  <div class="desc">

 A node key corresponding to the edge's end node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">linestring_coords</div>
    <div class="type">tuple | list | np.ndarray</div>
  </div>
  <div class="desc">

 A list, tuple, or numpy array of x, y coordinate tuples.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Distance tolerance in metres for matching the x_y coordinate to the linestring_coords.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">linestring_coords</div>
  </div>
  <div class="desc">

 A list of linestring coords aligned to the specified ending point.</div>
</div>


</div>


<div class="function">

## weld_linestring_coords


<div class="content">
<span class="name">weld_linestring_coords</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">linestring_coords_a</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]],</span> <span class="n">shapely</span><span class="o">.</span><span class="n">coords</span><span class="o">.</span><span class="n">CoordinateSequence</span><span class="p">]</span>,</span><span class="param">	<span class="n">linestring_coords_b</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">ndarray</span><span class="p">[</span><span class="n">Any</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">float64</span><span class="p">]],</span> <span class="n">shapely</span><span class="o">.</span><span class="n">coords</span><span class="o">.</span><span class="n">CoordinateSequence</span><span class="p">]</span>,</span><span class="param">	<span class="n">force_xy</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">tolerance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.01</span></span><span class="return-annotation">) -> <span class="nb">list</span><span class="p">[</span><span class="n">typing</span><span class="o">.</span><span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]]]</span>:</span></span>
</div>


 Welds two linestrings. Finds a matching start / end point combination and merges the coordinates accordingly. If the optional force_xy is provided then the weld will be performed at the x_y end of the LineStrings. The force_xy parameter is useful for looping geometries or overlapping geometries where it can happen that welding works from either of the two ends, thus potentially mis-aligning the start point unless explicit.

</div>


<div class="class">


## EdgeInfo



 Encapsulates EdgeInfo logic.



<div class="function">

## EdgeInfo


<div class="content">
<span class="name">EdgeInfo</span><span class="signature pdoc-code condensed">()</span>
</div>


 Initialises a network information structure.

</div>

 

<span class="name">names</span>



 Returns a set of street names.

 

<span class="name">routes</span>



 Returns a set of routes - e.g. route numbers.

 

<span class="name">highways</span>



 Returns a set of highway types - e.g. footway.

 

<div class="function">

## gather_edge_info


<div class="content">
<span class="name">gather_edge_info</span><span class="signature pdoc-code condensed">(<span class="param"><span class="bp">self</span>, </span><span class="param"><span class="n">edge_data</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]</span></span><span class="return-annotation">):</span></span>
</div>


 Gather edge data from provided edge_data.

</div>

 

<div class="function">

## set_edge_info


<div class="content">
<span class="name">set_edge_info</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="bp">self</span>,</span><span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">networkx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">multigraph</span><span class="o">.</span><span class="n">MultiGraph</span>,</span><span class="param">	<span class="n">start_node_key</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">end_node_key</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">edge_idx</span><span class="p">:</span> <span class="nb">int</span></span><span class="return-annotation">):</span></span>
</div>


 Set accumulated edge data to specified graph and edge.

</div>

 
</div>


<div class="function">

## add_node


<div class="content">
<span class="name">add_node</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">nodes_names</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>,</span><span class="param">	<span class="n">x</span><span class="p">:</span> <span class="nb">float</span>,</span><span class="param">	<span class="n">y</span><span class="p">:</span> <span class="nb">float</span>,</span><span class="param">	<span class="n">live</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">bool</span><span class="p">]</span>:</span></span>
</div>


 Add a node to a networkX `MultiGraph`. Assembles a new name from source node names. Checks for duplicates. Returns new name and is_dupe

</div>


<div class="function">

## create_nodes_strtree


<div class="content">
<span class="name">create_nodes_strtree</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="n">shapely</span><span class="o">.</span><span class="n">strtree</span><span class="o">.</span><span class="n">STRtree</span><span class="p">,</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]]]</span>:</span></span>
</div>


 Create a nodes-based STRtree spatial index.

</div>


<div class="function">

## create_edges_strtree


<div class="content">
<span class="name">create_edges_strtree</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="n">shapely</span><span class="o">.</span><span class="n">strtree</span><span class="o">.</span><span class="n">STRtree</span><span class="p">,</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]]]</span>:</span></span>
</div>


 Create an edges-based STRtree spatial index.

</div>


<div class="function">

## blend_metrics


<div class="content">
<span class="name">blend_metrics</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nodes_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>,</span><span class="param">	<span class="n">edges_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>,</span><span class="param">	<span class="n">method</span><span class="p">:</span> <span class="nb">str</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Blends metrics from a nodes GeoDataFrame into an edges GeoDataFrame. This is useful for situations where it is preferable to visualise the computed metrics as LineStrings instead of points. The line will be assigned the value from the adjacent two nodes based on the selected "min", "max", or "avg" method.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A nodes `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edges_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">method</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The method used for determining the line value from the adjacent points. Must be one of &quot;min&quot;, &quot;max&quot;, or &quot;avg&quot;.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">merged_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 An edges `GeoDataFrame` created by merging the node metrics from the provided nodes `GeoDataFrame` into the provided edges `GeoDataFrame`.</div>
</div>


</div>



