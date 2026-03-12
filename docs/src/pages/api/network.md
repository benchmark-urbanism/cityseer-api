---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# network


<div class="class">


## CityNetwork



 High-level interface for urban network analysis. Wraps network construction, centrality computation, and land-use analysis into a single object that manages graph topology, node attributes, and coordinate reference systems. The network is built as a dual graph where street segments become nodes and intersections become edges, enabling both shortest-path (metric) and simplest-path (angular) centrality analysis.

 Construct instances via the class methods rather than calling ``__init__`` directly:

- [`from_geopandas`](#from-geopandas) -- from a GeoDataFrame of LineString geometries
- [`from_wkts`](#from-wkts) -- from a dictionary of WKT strings or Shapely geometries
- [`from_nx`](#from-nx) -- from a cityseer-compatible NetworkX MultiGraph
- [`from_osm`](#from-osm) -- from OpenStreetMap via a bounding polygon
- [`load`](#load) -- from a previously saved parquet/pickle pair

 Most methods return ``self`` to support method chaining:

```python
cn = (
    CityNetwork.from_geopandas(edges_gdf, crs=32632)
    .set_boundary(boundary_polygon)
    .centrality_shortest(distances=[500, 1000, 2000])
)
```


:::note
The underlying graph construction automatically cleans input geometries by removing short self-loops, near-duplicate
edges, and short danglers. Use the [`feature_status`](#feature-status) property to inspect which input features were
filtered and why.
:::



<div class="function">

## CityNetwork


<div class="content">
<span class="name">CityNetwork</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">nodes_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">_state</span>
  </div>
  <div class="param">
    <span class="pn">_crs</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<span class="name">network_structure</span><span class="annotation">: NetworkStructure</span>


 

<span class="name">nodes_gdf</span><span class="annotation">: geopandas.geodataframe.GeoDataFrame</span>


 

<div class="function">

## to_geopandas


<div class="content">
<span class="name">to_geopandas</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Return a GeoDataFrame with the original input LineString geometries. The returned GeoDataFrame contains all computed columns (centrality metrics, layer results, etc.) joined to the original edge geometries rather than the midpoint representations used internally.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A new GeoDataFrame indexed by feature id with LineString geometries.</div>
</div>


</div>

 

<span class="name">is_dual</span><span class="annotation">: bool</span>


 

<span class="name">crs</span><span class="annotation">: pyproj.crs.crs.CRS | None</span>


 

<span class="name">node_count</span><span class="annotation">: int</span>


 

<span class="name">feature_status</span><span class="annotation">: pandas.Series</span>


 

<div class="function">

## from_wkts

<div class="decorator">@classmethod</div>

<div class="content">
<span class="name">from_wkts</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">cls</span>
  </div>
  <div class="param">
    <span class="pn">wkts</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">crs</span>
  </div>
  <div class="param">
    <span class="pn">boundary</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Construct a CityNetwork from a dictionary of WKT strings or Shapely geometries.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">wkts</div>
    <div class="type">dict[Any, str] | dict[Any, BaseGeometry]</div>
  </div>
  <div class="desc">

 A mapping from feature identifiers to WKT strings or Shapely LineString geometries. Input geometries may include z (elevation) coordinates, which are preserved and used for slope-based walking impedance calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">crs</div>
    <div class="type">Any</div>
  </div>
  <div class="desc">

 A projected coordinate reference system (EPSG code, CRS object, or proj string).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">boundary</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 Optional polygon in the same projected CRS; nodes inside are marked as ``live``, nodes outside as ``dead``.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">network</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 A new CityNetwork instance.</div>
</div>


</div>

 

<div class="function">

## from_geopandas

<div class="decorator">@classmethod</div>

<div class="content">
<span class="name">from_geopandas</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">cls</span>
  </div>
  <div class="param">
    <span class="pn">gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">crs</span>
  </div>
  <div class="param">
    <span class="pn">boundary</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Construct a CityNetwork from a GeoDataFrame of LineString geometries. Extra columns from the input GeoDataFrame are carried through to the internal nodes GeoDataFrame. The CRS is read from the GeoDataFrame unless explicitly overridden.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame with LineString or MultiLineString geometries. The index must be unique. Input geometries may include z (elevation) coordinates, which are preserved and used for slope-based walking impedance calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">crs</div>
    <div class="type">Any</div>
  </div>
  <div class="desc">

 Optional projected CRS override. If ``None``, uses the GeoDataFrame's CRS.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">boundary</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 Optional polygon in the same projected CRS; nodes inside are marked as ``live``, nodes outside as ``dead``.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">network</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 A new CityNetwork instance.</div>
</div>


</div>

 

<div class="function">

## from_nx

<div class="decorator">@classmethod</div>

<div class="content">
<span class="name">from_nx</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">cls</span>
  </div>
  <div class="param">
    <span class="pn">graph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">boundary</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Construct a CityNetwork from a cityseer-compatible NetworkX MultiGraph. The input graph must be a *primal* edge graph (not a dual graph) with ``geom`` attributes on edges and a ``crs`` attribute on the graph. Node ``live`` attributes are preserved.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">graph</div>
    <div class="type">nx.MultiGraph</div>
  </div>
  <div class="desc">

 A cityseer-compatible primal NetworkX graph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">boundary</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 Optional polygon in the same projected CRS; nodes inside are marked as ``live``, nodes outside as ``dead``.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">network</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 A new CityNetwork instance.</div>
</div>

### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">ValueError</div>
  </div>
  <div class="desc">

 If the input graph is a dual graph.</div>
</div>


</div>

 

<div class="function">

## from_osm

<div class="decorator">@classmethod</div>

<div class="content">
<span class="name">from_osm</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">cls</span>
  </div>
  <div class="param">
    <span class="pn">poly_geom</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">poly_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int = 4326</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">simplify</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">boundary</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry | None = None</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Construct a CityNetwork from OpenStreetMap data within a bounding polygon. Downloads the road network via OSMnx and converts it to a dual CityNetwork. Requires the ``osmnx`` package.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">poly_geom</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 A Shapely polygon defining the area of interest.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">poly_crs_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 EPSG code for ``poly_geom``. Defaults to 4326 (WGS84).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Target projected EPSG code. If ``None``, an appropriate UTM zone is inferred.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">simplify</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to simplify the OSM graph topology. Defaults to ``True``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">boundary</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 Optional polygon for live/dead node assignment (in the target projected CRS).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">**kwargs</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 Additional keyword arguments passed to [`io.osm_graph_from_poly`](/tools/io#osm-graph-from-poly).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">network</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 A new CityNetwork instance.</div>
</div>


</div>

 

<div class="function">

## update


<div class="content">
<span class="name">update</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Update the network topology with new or modified geometries. Performs an incremental diff against the current state: unchanged features retain their node indices, added features are inserted, and removed features are deleted. Previously computed centrality columns are cleared since they are invalidated by topology changes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data</div>
    <div class="type">dict[Any, str] | dict[Any, BaseGeometry] | GeoDataFrame</div>
  </div>
  <div class="desc">

 The complete updated set of geometries (not just the diff).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining.</div>
</div>


</div>

 

<div class="function">

## set_boundary


<div class="content">
<span class="name">set_boundary</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">polygon</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.base.BaseGeometry</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Set live/dead node status based on a boundary polygon. Nodes whose midpoints fall inside the polygon are marked ``live``; others are marked ``dead``. Dead nodes are excluded from centrality source computations but remain reachable as targets.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">polygon</div>
    <div class="type">BaseGeometry</div>
  </div>
  <div class="desc">

 A Shapely polygon in the same projected CRS as the network.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining.</div>
</div>


</div>

 

<div class="function">

## set_all_live


<div class="content">
<span class="name">set_all_live</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Mark all nodes as live, clearing any boundary restriction.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining.</div>
</div>


</div>

 

<div class="function">

## save


<div class="content">
<span class="name">save</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">path</span>
    <span class="pc">:</span>
    <span class="pa"> str | pathlib.Path</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Save the network to disk as a parquet/pickle pair. Creates two files: ``<path>.nodes.parquet`` (the nodes GeoDataFrame with all computed columns) and ``<path>.state.pkl`` (source WKTs, boundary, and feature status). Use [`load`](#load) to restore.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">path</div>
    <div class="type">str | Path</div>
  </div>
  <div class="desc">

 Base file path. File extensions are replaced automatically.</div>
</div>


</div>

 

<div class="function">

## load

<div class="decorator">@classmethod</div>

<div class="content">
<span class="name">load</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">cls</span>
  </div>
  <div class="param">
    <span class="pn">path</span>
    <span class="pc">:</span>
    <span class="pa"> str | pathlib.Path</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Load a previously saved CityNetwork from disk. Rebuilds the full graph topology from the saved source WKTs and merges any previously computed columns (centrality metrics, layer results) from the saved nodes GeoDataFrame.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">path</div>
    <div class="type">str | Path</div>
  </div>
  <div class="desc">

 Base file path (same as was passed to [`save`](#save)).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">network</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 The restored CityNetwork instance.</div>
</div>


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
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Compute shortest-path (metric) node centrality. Wraps [`node_centrality_shortest`](/metrics/networks#node-centrality-shortest). All keyword arguments are forwarded; see that function for the full parameter list including ``distances``, ``betas``, ``minutes``, ``compute_closeness``, ``compute_betweenness``, ``sample``, and ``epsilon``.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
</div>


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
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Compute simplest-path (angular) node centrality. Wraps [`node_centrality_simplest`](/metrics/networks#node-centrality-simplest). All keyword arguments are forwarded; see that function for the full parameter list.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
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
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Compute segment-based centrality. Wraps [`segment_centrality`](/metrics/networks#segment-centrality). All keyword arguments are forwarded; see that function for the full parameter list.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
</div>


</div>

 

<div class="function">

## build_od_matrix


<div class="content">
<span class="name">build_od_matrix</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">od_df</span>
    <span class="pc">:</span>
    <span class="pa"> pandas.DataFrame</span>
  </div>
  <div class="param">
    <span class="pn">zones_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">OdMatrix</span>
  <span class="pt">]</span>
</div>
</div>


 Build an origin-destination matrix for OD-weighted betweenness. Wraps [`build_od_matrix`](/metrics/networks#build-od-matrix). See that function for the full parameter list.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">od_df</div>
    <div class="type">pd.DataFrame</div>
  </div>
  <div class="desc">

 Origin-destination flow data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">zones_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 Zone polygons corresponding to the OD matrix.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">od_matrix</div>
    <div class="type">OdMatrix</div>
  </div>
  <div class="desc">

 An OD matrix for use with [`betweenness_od`](#betweenness-od).</div>
</div>


</div>

 

<div class="function">

## betweenness_od


<div class="content">
<span class="name">betweenness_od</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">od_matrix</span>
    <span class="pc">:</span>
    <span class="pa"> OdMatrix</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Compute OD-weighted betweenness centrality. Wraps [`betweenness_od`](/metrics/networks#betweenness-od). See that function for the full parameter list.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">od_matrix</div>
    <div class="type">OdMatrix</div>
  </div>
  <div class="desc">

 An OD matrix from [`build_od_matrix`](#build-od-matrix).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
</div>


</div>

 

<div class="function">

## compute_accessibilities


<div class="content">
<span class="name">compute_accessibilities</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute land-use accessibility metrics. Wraps [`compute_accessibilities`](/metrics/layers#compute-accessibilities).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame of land-use points with categorical columns.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self with accessibility columns added to ``nodes_gdf``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input data GeoDataFrame with nearest network assignments.</div>
</div>


</div>

 

<div class="function">

## compute_mixed_uses


<div class="content">
<span class="name">compute_mixed_uses</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute mixed-use diversity metrics. Wraps [`compute_mixed_uses`](/metrics/layers#compute-mixed-uses).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame of land-use points with categorical columns.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self with mixed-use columns added to ``nodes_gdf``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input data GeoDataFrame with nearest network assignments.</div>
</div>


</div>

 

<div class="function">

## compute_stats


<div class="content">
<span class="name">compute_stats</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute statistical aggregations of numerical data over the network. Wraps [`compute_stats`](/metrics/layers#compute-stats).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame of data points with numerical columns.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self with statistical columns added to ``nodes_gdf``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input data GeoDataFrame with nearest network assignments.</div>
</div>


</div>

 

<div class="function">

## add_gtfs


<div class="content">
<span class="name">add_gtfs</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">gtfs_path</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">*</span>
  </div>
  <div class="param">
    <span class="pn">crs</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 400</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CityNetwork</span>
  <span class="pt">]</span>
</div>
</div>


 Add GTFS public transport data to the network. Wraps [`io.add_transport_gtfs`](/tools/io#add-transport-gtfs).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">gtfs_path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Path to a GTFS zip file or directory.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">crs</div>
    <div class="type">Any</div>
  </div>
  <div class="desc">

 Optional CRS override for the GTFS data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Maximum distance (metres) for snapping stops to the network. Defaults to 400.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining.</div>
</div>


</div>

 

<div class="function">

## to_nx


<div class="content">
<span class="name">to_nx</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">MultiGraph</span>
  <span class="pt">]</span>
</div>
</div>


 Convert the network to a cityseer-compatible NetworkX MultiGraph.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">graph</div>
    <div class="type">nx.MultiGraph</div>
  </div>
  <div class="desc">

 A primal edge graph with ``geom`` attributes on edges and ``crs`` on the graph.</div>
</div>


</div>

 
</div>



</section>
