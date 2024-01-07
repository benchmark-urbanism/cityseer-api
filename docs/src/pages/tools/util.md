---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# util


 Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.


<div class="function">

## measure_bearing


<div class="content">
<span class="name">measure_bearing</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">xy_1</span>
  </div>
  <div class="param">
    <span class="pn">xy_2</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Measures the angular bearing between two coordinate pairs.

</div>


<div class="function">

## measure_coords_angle


<div class="content">
<span class="name">measure_coords_angle</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">coords_1</span>
  </div>
  <div class="param">
    <span class="pn">coords_2</span>
  </div>
  <div class="param">
    <span class="pn">coords_3</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Measures angle between three coordinate pairs. Angular change is from one line segment to the next, across the intermediary coord.

</div>


<div class="function">

## measure_angle_diff_betw_linestrings


<div class="content">
<span class="name">measure_angle_diff_betw_linestrings</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords_a</span>
  </div>
  <div class="param">
    <span class="pn">linestring_coords_b</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Measures the angular difference between the bearings of two sets of linestring coords.

</div>


<div class="function">

## measure_cumulative_angle


<div class="content">
<span class="name">measure_cumulative_angle</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Measures the cumulative angle along a LineString geom's coords.

</div>


<div class="function">

## measure_max_angle


<div class="content">
<span class="name">measure_max_angle</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Measures the maximum angle along a LineString geom's coords.

</div>


<div class="function">

## snap_linestring_startpoint


<div class="content">
<span class="name">snap_linestring_startpoint</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords</span>
  </div>
  <div class="param">
    <span class="pn">x_y</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Union[list[Union[tuple[float</span>
  <span class="pr">float]</span>
  <span class="pr">tuple[float</span>
  <span class="pr">float</span>
  <span class="pr">float]</span>
  <span class="pr">ndarray[Any</span>
  <span class="pr">float64]]]]</span>
  <span class="pr">CoordinateSequence</span>
  <span class="pt">]</span>
</div>
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
<span class="name">snap_linestring_endpoint</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords</span>
  </div>
  <div class="param">
    <span class="pn">x_y</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Union[list[Union[tuple[float</span>
  <span class="pr">float]</span>
  <span class="pr">tuple[float</span>
  <span class="pr">float</span>
  <span class="pr">float]</span>
  <span class="pr">ndarray[Any</span>
  <span class="pr">float64]]]]</span>
  <span class="pr">CoordinateSequence</span>
  <span class="pt">]</span>
</div>
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
<span class="name">align_linestring_coords</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords</span>
  </div>
  <div class="param">
    <span class="pn">x_y</span>
  </div>
  <div class="param">
    <span class="pn">reverse</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.5</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Union[list[Union[tuple[float</span>
  <span class="pr">float]</span>
  <span class="pr">tuple[float</span>
  <span class="pr">float</span>
  <span class="pr">float]</span>
  <span class="pr">ndarray[Any</span>
  <span class="pr">float64]]]]</span>
  <span class="pr">CoordinateSequence</span>
  <span class="pt">]</span>
</div>
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
<span class="name">snap_linestring_endpoints</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
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
    <span class="pn">linestring_coords</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.5</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Union[list[Union[tuple[float</span>
  <span class="pr">float]</span>
  <span class="pr">tuple[float</span>
  <span class="pr">float</span>
  <span class="pr">float]</span>
  <span class="pr">ndarray[Any</span>
  <span class="pr">float64]]]]</span>
  <span class="pr">CoordinateSequence</span>
  <span class="pt">]</span>
</div>
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
<span class="name">weld_linestring_coords</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">linestring_coords_a</span>
  </div>
  <div class="param">
    <span class="pn">linestring_coords_b</span>
  </div>
  <div class="param">
    <span class="pn">force_xy</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.01</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Union[list[Union[tuple[float</span>
  <span class="pr">float]</span>
  <span class="pr">tuple[float</span>
  <span class="pr">float</span>
  <span class="pr">float]</span>
  <span class="pr">ndarray[Any</span>
  <span class="pr">float64]]]]</span>
  <span class="pr">CoordinateSequence</span>
  <span class="pt">]</span>
</div>
</div>


 Welds two linestrings. Finds a matching start / end point combination and merges the coordinates accordingly. If the optional force_xy is provided then the weld will be performed at the x_y end of the LineStrings. The force_xy parameter is useful for looping geometries or overlapping geometries where it can happen that welding works from either of the two ends, thus potentially mis-aligning the start point unless explicit.

</div>


<div class="class">


## EdgeInfo



 Encapsulates EdgeInfo logic.



<div class="function">

## EdgeInfo


<div class="content">
<span class="name">EdgeInfo</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>


 Initialises a network information structure.

</div>

 

<span class="name">names</span>


 

<span class="name">routes</span>


 

<span class="name">highways</span>


 

<div class="function">

## gather_edge_info


<div class="content">
<span class="name">gather_edge_info</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">edge_data</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Gather edge data from provided edge_data.

</div>

 

<div class="function">

## set_edge_info


<div class="content">
<span class="name">set_edge_info</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">nx_multigraph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
  </div>
  <div class="param">
    <span class="pn">start_node_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">end_node_key</span>
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


 Set accumulated edge data to specified graph and edge.

</div>

 
</div>


<div class="function">

## add_node


<div class="content">
<span class="name">add_node</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">nodes_names</span>
    <span class="pc">:</span>
    <span class="pa"> list[str]</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">live</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>


 Add a node to a networkX `MultiGraph`. Assembles a new name from source node names. Checks for duplicates. Returns new name and is_dupe

</div>


<div class="function">

## create_nodes_strtree


<div class="content">
<span class="name">create_nodes_strtree</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">STRtree</span>
  <span class="pr">list[dict[str</span>
  <span class="pt">]</span>
</div>
</div>


 Create a nodes-based STRtree spatial index.

</div>


<div class="function">

## create_edges_strtree


<div class="content">
<span class="name">create_edges_strtree</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">STRtree</span>
  <span class="pr">list[dict[str</span>
  <span class="pt">]</span>
</div>
</div>


 Create an edges-based STRtree spatial index.

</div>


<div class="function">

## blend_metrics


<div class="content">
<span class="name">blend_metrics</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nodes_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">edges_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">method</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Blends metrics from a nodes GeoDataFrame into an edges GeoDataFrame. This is useful for situations where it is preferable to visualise the computed metrics as LineStrings instead of points. The line will be assigned the value from the adjacent two nodes based on the selected "min", "max", or "avg" method.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A nodes `GeoDataFrame` as derived from [`network_structure_from_nx`](tools/io#network-structure-from-nx).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edges_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](tools/io#network-structure-from-nx).</div>
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


<div class="function">

## project_geom


<div class="content">
<span class="name">project_geom</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">geom</span>
  </div>
  <div class="param">
    <span class="pn">from_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Projects an input shapely geometry.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">geom</div>
    <div class="type">shapely.geometry</div>
  </div>
  <div class="desc">

 A GeoDataFrame containing building polygons.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">from_crs_code</div>
    <div class="type">int | str</div>
  </div>
  <div class="desc">

 The EPSG code from which to convert the projection.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int | str</div>
  </div>
  <div class="desc">

 The EPSG code into which to convert the projection.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">shapely.geometry</div>
  </div>
  <div class="desc">

 A shapely geometry in the specified `to_crs_code` projection.</div>
</div>


</div>


<div class="function">

## extract_utm_epsg_code


<div class="content">
<span class="name">extract_utm_epsg_code</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">lng</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">lat</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>


 Finds the UTM coordinate reference system for a given longitude and latitude.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">lng</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The longitude for which to find the appropriate UTM EPSG code.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">lat</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The latitude for which to find the appropriate UTM EPSG code.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The EPSG coordinate reference code for the UTM projection.</div>
</div>


</div>



</section>
