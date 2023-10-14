---
layout: ../../layouts/PageLayout.astro
---

# io


 Functions for fetching and converting graphs and network structures.


<div class="function">

## nx_epsg_conversion


<div class="content">
<span class="name">nx_epsg_conversion</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>, </span><span class="param"><span class="n">from_epsg_code</span><span class="p">:</span> <span class="nb">int</span>, </span><span class="param"><span class="n">to_epsg_code</span><span class="p">:</span> <span class="nb">int</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Convert a graph from the `from_epsg_code` EPSG CRS to the `to_epsg_code` EPSG CRS. The `to_epsg_code` must be for a projected CRS. If edge `geom` attributes are found, the associated `LineString` geometries will also be converted.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes in the `from_epsg_code` coordinate system. Optional `geom` edge attributes containing `LineString` geoms to be converted.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">from_epsg_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An integer representing a valid EPSG code specifying the CRS from which the graph must be converted. For example, [4326](https://epsg.io/4326) if converting data from an OpenStreetMap response.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_epsg_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An integer representing a valid EPSG code specifying the CRS into which the graph must be projected. For example, [27700](https://epsg.io/27700) if converting to British National Grid.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the specified `to_epsg_code` coordinate system. Edge `geom` attributes will also be converted if found.</div>
</div>


</div>


<div class="function">

## nx_wgs_to_utm


<div class="content">
<span class="name">nx_wgs_to_utm</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>, </span><span class="param"><span class="n">force_zone_number</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Convert a graph from WGS84 geographic coordinates to UTM projected coordinates. Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all other nodes and geometries contained in the graph. This ensures consistent behaviour in cases where a graph spans a UTM boundary.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge attributes containing `LineString` geoms to be converted.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">force_zone_number</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched UTM zones may introduce substantial distortions in the results.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge `geom` attributes are present, these will also be converted.</div>
</div>


</div>


<div class="function">

## buffered_point_poly


<div class="content">
<span class="name">buffered_point_poly</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">lng</span><span class="p">:</span> <span class="nb">float</span>,</span><span class="param">	<span class="n">lat</span><span class="p">:</span> <span class="nb">float</span>,</span><span class="param">	<span class="n">buffer</span><span class="p">:</span> <span class="nb">int</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="n">shapely</span><span class="o">.</span><span class="n">geometry</span><span class="o">.</span><span class="n">polygon</span><span class="o">.</span><span class="n">Polygon</span><span class="p">,</span> <span class="n">shapely</span><span class="o">.</span><span class="n">geometry</span><span class="o">.</span><span class="n">polygon</span><span class="o">.</span><span class="n">Polygon</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">str</span><span class="p">]</span>:</span></span>
</div>


 Buffer a point and return a `shapely` Polygon in WGS and UTM coordinates. This function can be used to prepare a `poly_wgs` `Polygon` for passing to [`osm_graph_from_poly()`](#osm-graph-from-poly).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">lng</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The longitudinal WGS coordinate in degrees.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">lat</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The latitudinal WGS coordinate in degrees.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">buffer</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The buffer distance in metres.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">poly_wgs</div>
    <div class="type">Polygon</div>
  </div>
  <div class="desc">

 A `shapely` `Polygon` in WGS coordinates.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">poly_utm</div>
    <div class="type">Polygon</div>
  </div>
  <div class="desc">

 A `shapely` `Polygon` in UTM coordinates.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">utm_zone_number</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The UTM zone number used for conversion.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">utm_zone_letter</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The UTM zone letter used for conversion.</div>
</div>


</div>


<div class="function">

## fetch_osm_network


<div class="content">
<span class="name">fetch_osm_network</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">osm_request</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">timeout</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">300</span>,</span><span class="param">	<span class="n">max_tries</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span></span><span class="return-annotation">) -> <span class="n">requests</span><span class="o">.</span><span class="n">models</span><span class="o">.</span><span class="n">Response</span> <span class="o">|</span> <span class="kc">None</span>:</span></span>
</div>


 Fetches an OSM response.
:::note
This function requires a valid OSM request. If you prepare a polygonal extents then it may be easier to use
[`osm_graph_from_poly()`](#osm-graph-from-poly), which would call this method on your behalf and then
builds a graph automatically.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">osm_request</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A valid OSM request as a string. Use [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) for testing custom queries.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">timeout</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Timeout duration for API call in seconds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_tries</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of attempts to fetch a response before raising.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">requests.Response</div>
  </div>
  <div class="desc">

 An OSM API response.</div>
</div>


</div>


<div class="function">

## osm_graph_from_poly


<div class="content">
<span class="name">osm_graph_from_poly</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">poly_geom</span><span class="p">:</span> <span class="n">shapely</span><span class="o">.</span><span class="n">geometry</span><span class="o">.</span><span class="n">polygon</span><span class="o">.</span><span class="n">Polygon</span>,</span><span class="param">	<span class="n">poly_epsg_code</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">4326</span>,</span><span class="param">	<span class="n">to_epsg_code</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">buffer_dist</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">15</span>,</span><span class="param">	<span class="n">custom_request</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">simplify</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">remove_parallel</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">iron_edges</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">remove_disconnected</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">timeout</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">300</span>,</span><span class="param">	<span class="n">max_tries</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Prepares a `networkX` `MultiGraph` from an OSM request for the specified shapely polygon. This function will retrieve the OSM response and will automatically unpack this into a `networkX` graph. Simplification will be applied by default, but can be disabled.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">poly_geom</div>
    <div class="type">shapely.Polygon</div>
  </div>
  <div class="desc">

 A shapely Polygon representing the extents for which to fetch the OSM network.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">poly_epsg_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An integer representing a valid EPSG code for the provided polygon. For example, [4326](https://epsg.io/4326) if using WGS lng / lat, or [27700](https://epsg.io/27700) if using the British National Grid.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_epsg_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional integer representing a valid EPSG code for the generated network returned from this function. If this parameter is provided, then the network will be converted to the specified EPSG coordinate reference system. If not provided, then the OSM network will be projected into a local UTM coordinate reference system.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">buffer_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 A distance to use for buffering and cleaning operations. 15m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">custom_request</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional custom OSM request. If provided, this must include a &quot;geom_osm&quot; string formatting key for inserting the geometry passed to the OSM API query. See the discussion below.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">simplify</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to automatically simplify the OSM graph. Set to False for manual cleaning.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">remove_parallel</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Ignored if simplify is False. Whether to remove parallel roadway segments.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">iron_edges</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Ignored if simplify is False.  Whether to straighten the ends of street segments. This can help to reduce the number of artefacts from segment kinks from merging `LineStrings`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">remove_disconnected</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Ignored if simplify is False.  Whether to remove disconnected components from the network.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">timeout</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Timeout duration for API call in seconds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_tries</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of attempts to fetch a response before raising.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` `MultiGraph` with `x` and `y` node attributes that have been converted to UTM. The network will be simplified if the `simplify` parameter is `True`.</div>
</div>

### Notes

 The default OSM request will attempt to find all walkable routes. It will ignore motorways and will try to work with pedestrianised routes and walkways.

 If you wish to provide your own OSM request, then provide a valid OSM API request as a string. The string must contain a `geom_osm` f-string formatting key. This allows for the geometry parameter passed to the OSM API to be injected into the request. It is also recommended to not use the `skel` output option so that `cityseer` can use street name and highway reference information for cleaning purposes. See [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) for experimenting with custom queries.

 For example, to return only drivable roads, then use a request similar to the following. Notice the `geom_osm` f-string interpolation key and the use of `out qt;` instead of `out skel qt;`.

```python
custom_request = f'''
[out:json];
(
way["highway"]
["area"!="yes"]
["highway"!~"footway|pedestrian|steps|bus_guideway|escape|raceway|proposed|planned|abandoned|platform|construction"]
(poly:"{geom_osm}");
);
out body;
>;
out qt;
'''
```


</div>


<div class="function">

## nx_from_osm


<div class="content">
<span class="name">nx_from_osm</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">osm_json</span><span class="p">:</span> <span class="nb">str</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Generate a `NetworkX` `MultiGraph` from [Open Street Map](https://www.openstreetmap.org) data.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">osm_json</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A `json` string response from the [OSM overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API), consisting of `nodes` and `ways`.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` `MultiGraph` with `x` and `y` attributes in [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates.</div>
</div>


</div>


<div class="function">

## nx_from_osm_nx


<div class="content">
<span class="name">nx_from_osm_nx</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multidigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">node_attributes</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">edge_attributes</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">tolerance</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.001</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Copy an [`OSMnx`](https://osmnx.readthedocs.io/) directed `MultiDiGraph` to an undirected `cityseer` `MultiGraph`. See the [`OSMnx`](/guide#osm-and-networkx) section of the guide for a more general discussion (and example) on workflows combining `OSMnx` with `cityseer`.

 `x` and `y` node attributes will be copied directly and `geometry` edge attributes will be copied to a `geom` edge attribute. The conversion process will snap the `shapely` `LineString` endpoints to the corresponding start and end node coordinates.

 Note that `OSMnx` `geometry` attributes only exist for simplified edges: if a `geometry` edge attribute is not found, then a simple (straight) `shapely` `LineString` geometry will be inferred from the respective start and end nodes.

 Other attributes will be ignored to avoid potential downstream misinterpretations of the attributes as a consequence of subsequent steps of graph manipulation, i.e. to avoid situations where attributes may fall out of lock-step with the state of the graph. If particular attributes need to be copied across, and assuming cognisance of downstream implications, then these can be manually specified by providing a list of node attributes keys per the `node_attributes` parameter or edge attribute keys per the `edge_attributes` parameter.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multidigraph</div>
    <div class="type">MultiDiGraph</div>
  </div>
  <div class="desc">

 A `OSMnx` derived `networkX` `MultiDiGraph` containing `x` and `y` node attributes, with optional `geometry` edge attributes containing `LineString` geoms (for simplified edges).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">node_attributes</div>
    <div class="type">tuple[str]</div>
  </div>
  <div class="desc">

 Optional node attributes to copy to the new MultiGraph. (In addition to the default `x` and `y` attributes.)</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_attributes</div>
    <div class="type">tuple[str]</div>
  </div>
  <div class="desc">

 Optional edge attributes to copy to the new MultiGraph. (In addition to the optional `geometry` attribute.)</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Tolerance at which to raise errors for mismatched geometry end-points vis-a-vis corresponding node coordinates. Prior to conversion, this method will check edge geometry end-points for alignment with the corresponding end-point nodes. Where these don't align within the given tolerance an exception will be raised. Otherwise, if within the tolerance, the conversion function will snap the geometry end-points to the corresponding node coordinates so that downstream exceptions are not subsequently raised. It is preferable to minimise graph manipulation prior to conversion to a `cityseer` compatible `MultiGraph` otherwise particularly large tolerances may be required, and this may lead to some unexpected or undesirable effects due to aggressive snapping.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attributes.</div>
</div>


</div>


<div class="function">

## nx_from_open_roads


<div class="content">
<span class="name">nx_from_open_roads</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">open_roads_path</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="n">pathlib</span><span class="o">.</span><span class="n">Path</span>,</span><span class="param">	<span class="n">target_bbox</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">],</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span></span><span class="return-annotation">) -> <span class="n">networkx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">multigraph</span><span class="o">.</span><span class="n">MultiGraph</span>:</span></span>
</div>


 Generates a `networkX` `MultiGraph` from an OS Open Roads dataset.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">open_roads_path</div>
    <div class="type">str | Path</div>
  </div>
  <div class="desc">

 A valid relative filepath from which to load the OS Open Roads dataset.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">target_bbox</div>
    <div class="type">tuple[int]</div>
  </div>
  <div class="desc">

 A tuple of integers or floats representing the `[s, w, n, e]` bounding box extents for which to load the dataset. Set to `None` for no bounding box.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attributes.</div>
</div>


</div>


<div class="function">

## network_structure_from_nx


<div class="content">
<span class="name">network_structure_from_nx</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">crs</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="nb">int</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span><span class="p">,</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span><span class="p">,</span> <span class="n">NetworkStructure</span><span class="p">]</span>:</span></span>
</div>


 Transpose a `networkX` `MultiGraph` into a `GeoDataFrame` and `NetworkStructure` for use by `cityseer`. Calculates length and angle attributes, as well as in and out bearings, and stores this information in the returned data maps.
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
    <div class="name">crs</div>
    <div class="type">str | int</div>
  </div>
  <div class="desc">

 CRS for initialising the returned structures. This is used for initialising the GeoPandas [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe).  # pylint: disable=line-too-long</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with `live`, `weight`, and `geometry` attributes. The original `networkX` graph's node keys will be used for the `GeoDataFrame` index.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edges_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with `ns_edge_idx`, `start_ns_node_idx`, `end_ns_node_idx`, `edge_idx`, `nx_start_node_key`, `nx_end_node_key`, `length`, `angle_sum`, `imp_factor`, `in_bearing`, `out_bearing`, `total_bearing`, `geom` attributes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.NetworkStructure</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos#networkstructure) instance.</div>
</div>


</div>


<div class="function">

## nx_from_geopandas


<div class="content">
<span class="name">nx_from_geopandas</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nodes_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>,</span><span class="param">	<span class="n">edges_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Write nodes and edges `GeoDataFrames` to a `networkX` `MultiGraph`.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with `live`, `weight`, and Point `geometry` attributes. The index will be used for the returned `networkX` graph's node keys.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edges_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `networkX` graph with geometries and attributes as copied from the input `GeoDataFrames`.</div>
</div>


</div>



