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

 ### Dual graph architecture

 ``CityNetwork`` always constructs a dual graph internally. In the dual representation, each street segment becomes a node (positioned at the segment midpoint) and edges connect segments that share a common intersection. This enables both shortest-path and simplest-path (angular) analysis from a single topology:

- **Shortest-path** analysis uses metric distances along street segments.
- **Simplest-path** analysis uses cumulative angular change along streets and at intersections as the routing cost.

 Because the dual is built automatically, there is no need to call ``nx_to_dual`` when using ``CityNetwork``. Although the topology is dual internally, results are visualised and exported as the original street segment geometries via [`to_geopandas`](#to-geopandas), so each row in the output corresponds to one input street.

 ### Working with results

 All computed metrics are written to the internal ``nodes_gdf`` GeoDataFrame. Since ``CityNetwork`` uses a dual graph, each row in ``nodes_gdf`` represents a street segment, with a Point geometry at the segment midpoint.

 To retrieve results with the original LineString geometries, use [`to_geopandas`](#to-geopandas):

```python
cn = CityNetwork.from_geopandas(edges_gdf, crs=32632)
cn.centrality_shortest(distances=[800])

# Midpoint geometries (internal representation)
cn.nodes_gdf["cc_harmonic_800"]

# Original LineString geometries with the same computed columns
result_gdf = cn.to_geopandas()
result_gdf["cc_harmonic_800"]
```

 Column names follow the ``cc_{metric}_{distance}`` convention described in the [Column Naming Conventions](/intro#column-naming-conventions) section.

 ### Feature cleaning

 Input geometries are automatically cleaned during construction. Short self-loops, near-duplicate edges, and short danglers are removed. The ``feature_status`` property returns a Series indicating the status of each input feature:

```python
cn = CityNetwork.from_geopandas(edges_gdf, crs=32632)
print(cn.feature_status.value_counts())
# active              142
# short_dangler         3
# duplicate             1
```

 ### Saving and loading

 Networks can be serialised to disk and restored later, preserving all computed metrics:

```python
cn.save("my_network")
# Creates: my_network.nodes.parquet, my_network.state.pkl

cn_restored = CityNetwork.load("my_network")
```

 ### Incremental updates

 The [`update`](#update) method performs an incremental topology diff: unchanged features keep their node indices, added features are inserted, and removed features are deleted. Previously computed centrality columns are cleared since they are invalidated by topology changes.

```python
# Initial build
cn = CityNetwork.from_geopandas(edges_gdf, crs=32632)
cn.centrality_shortest(distances=[800])

# Update with modified geometries
cn.update(updated_edges_gdf)
cn.centrality_shortest(distances=[800])
```

 ### Typical workflow

```python
import geopandas as gpd
from cityseer.network import CityNetwork
from cityseer import decay

# 1. Load street network edges
edges_gdf = gpd.read_file("streets.gpkg")

# 2. Build the network
cn = CityNetwork.from_geopandas(edges_gdf, crs="EPSG:32632")

# 3. Compute centrality at multiple scales
cn.centrality_shortest(distances=[400, 800, 1600])
cn.centrality_simplest(distances=[400, 800, 1600])

# 4. Compute land-use accessibility
landuses_gdf = gpd.read_file("landuses.gpkg")
cn, landuses_gdf = cn.compute_accessibilities(
    data_gdf=landuses_gdf,
    landuse_column_label="category",
    accessibility_keys=["retail", "park"],
    distances=[400, 800],
    decay_fn=decay.exponential(),
)

# 5. Compute statistical aggregations
prices_gdf = gpd.read_file("property_prices.gpkg")
cn, prices_gdf = cn.compute_stats(
    data_gdf=prices_gdf,
    stats_column_labels=["price"],
    distances=[800, 1600],
    decay_fn=decay.gaussian(peak=400, cutoff=1600),
)

# 6. Export results with original LineString geometries
result_gdf = cn.to_geopandas()
result_gdf.to_file("results.gpkg")
```




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

### Notes

```python
cn.centrality_shortest(distances=[800])
result_gdf = cn.to_geopandas()

# result_gdf has LineString geometries (not midpoint Points)
print(result_gdf.geometry.geom_type.unique())  # ['LineString']

# All computed columns are present
print(result_gdf["cc_harmonic_800"])

# Export to file
result_gdf.to_file("centrality_results.gpkg")
```


</div>

 

<span class="name">is_dual</span><span class="annotation">: bool</span>


 

<span class="name">is_directed</span><span class="annotation">: bool</span>


 

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
  <div class="param">
    <span class="pn">directed</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">oneway_fids</span>
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

<div class="param-set">
  <div class="def">
    <div class="name">directed</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If ``True``, build a directed network. Requires ``oneway_fids``. Features in ``oneway_fids`` are one-way (in LineString coordinate order); all other features are bidirectional.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">oneway_fids</div>
    <div class="type">set[Any] | None</div>
  </div>
  <div class="desc">

 Feature IDs that are one-way when ``directed=True``. Ignored if ``directed=False``.</div>
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

 If ``directed=True`` but ``oneway_fids`` is not provided.</div>
</div>

### Notes

```python
from shapely.geometry import LineString
from cityseer.network import CityNetwork

wkts = {
    "street_a": LineString([(0, 0), (100, 0)]),
    "street_b": LineString([(100, 0), (100, 100)]),
    "street_c": LineString([(100, 0), (200, 0)]),
}
cn = CityNetwork.from_wkts(wkts, crs=32632)
cn.centrality_shortest(distances=[200])
```


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
  <div class="param">
    <span class="pn">directed</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
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

<div class="param-set">
  <div class="def">
    <div class="name">directed</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If ``True``, build a directed network. Requires a boolean ``oneway`` column in the GeoDataFrame. Features with ``oneway=True`` are one-way in LineString coordinate order; features with ``oneway=False`` are bidirectional.</div>
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

 If ``directed=True`` but the GeoDataFrame has no ``oneway`` column.</div>
</div>

### Notes

```python
import geopandas as gpd
from shapely.geometry import LineString
from cityseer.network import CityNetwork

gdf = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (100, 100)]),
            LineString([(100, 0), (200, 0)]),
        ]
    },
    crs="EPSG:32632",
)
cn = CityNetwork.from_geopandas(gdf)
cn.centrality_shortest(distances=[200])
print(cn.nodes_gdf[["cc_harmonic_200", "cc_betweenness_200"]])
```

 With directed one-way streets:

```python
gdf = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
        ],
        "oneway": [True, False],
    },
    crs="EPSG:32632",
)
cn = CityNetwork.from_geopandas(gdf, directed=True)
```


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


 Construct a CityNetwork from a cityseer-compatible NetworkX graph. The input graph must be a *primal* edge graph (not a dual graph) with ``geom`` attributes on edges and a ``crs`` attribute on the graph. Node ``live`` attributes are preserved.

 When a ``MultiDiGraph`` is passed, directed mode is enabled automatically: each directed edge becomes its own one-way dual node (in the coordinate order of the directed edge). Two-way streets should be represented as two reciprocal edges (A to B and B to A), which produce two separate dual nodes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">graph</div>
    <div class="type">nx.MultiGraph | nx.MultiDiGraph</div>
  </div>
  <div class="desc">

 A cityseer-compatible primal NetworkX graph. ``MultiDiGraph`` enables directed routing.</div>
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

### Notes

 From an undirected graph:

```python
import networkx as nx
from shapely.geometry import LineString
from cityseer.network import CityNetwork

G = nx.MultiGraph(crs="EPSG:32632")
G.add_node("a", x=0.0, y=0.0)
G.add_node("b", x=100.0, y=0.0)
G.add_node("c", x=200.0, y=0.0)
G.add_edge("a", "b", geom=LineString([(0, 0), (100, 0)]))
G.add_edge("b", "c", geom=LineString([(100, 0), (200, 0)]))

cn = CityNetwork.from_nx(G)
```

 From a directed MultiDiGraph (e.g. via OSMnx):

```python
G = nx.MultiDiGraph(crs="EPSG:32632")
G.add_node("a", x=0.0, y=0.0)
G.add_node("b", x=100.0, y=0.0)
# One-way: a -> b only
G.add_edge("a", "b", key=0, geom=LineString([(0, 0), (100, 0)]))
cn = CityNetwork.from_nx(G)
assert cn.is_directed
```


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


 Construct a CityNetwork from OpenStreetMap data within a bounding polygon. Downloads the road network and converts it to a dual CityNetwork.

 For directed (one-way) routing with OSM data, fetch a directed graph via `OSMnx <https://osmnx.readthedocs.io/>`_ and pass it to :meth:`from_nx` or convert it with :func:`io.nx_from_osm_nx(directed=True) <cityseer.tools.io.nx_from_osm_nx>`.
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

### Notes

```python
from shapely.geometry import box
from cityseer.network import CityNetwork

# Bounding box in WGS84 (lon/lat)
polygon = box(-0.13, 51.51, -0.12, 51.52)
cn = CityNetwork.from_osm(polygon, to_crs_code=32630)
cn.centrality_shortest(distances=[400, 800])
```


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

 For directed networks built via ``from_geopandas(directed=True)``, the incoming GeoDataFrame must include the ``oneway`` column. Direction changes (even without geometry changes) trigger a rebuild.
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

### Notes

```python
cn.centrality_shortest(distances=[800])
cn.save("my_network")
# Creates: my_network.nodes.parquet, my_network.state.pkl

# Later, restore the full network with all metrics
cn_restored = CityNetwork.load("my_network")
print(cn_restored.nodes_gdf["cc_harmonic_800"])
```


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


 Compute shortest-path (metric) node centrality. Wraps [`node_centrality_shortest`](/metrics/networks#node-centrality-shortest). All keyword arguments are forwarded; see that function for the full parameter list including ``distances``, ``minutes``, ``compute_closeness``, ``compute_betweenness``, ``decay_fn``, ``sample``, and ``epsilon``.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
</div>

### Notes

```python
from cityseer import decay

# Multiple distance thresholds
cn.centrality_shortest(distances=[400, 800, 1600])

# With custom decay and sampling for large networks
cn.centrality_shortest(
    distances=[800, 2000, 5000],
    decay_fn=decay.exponential(steepness=4),
    sample=True,
    epsilon=0.06,
)

# Closeness only (skip betweenness for speed)
cn.centrality_shortest(distances=[800], compute_betweenness=False)

# Using walking time thresholds instead of distances
cn.centrality_shortest(minutes=[5, 10, 20])
```

 Output columns per distance ``d`` (see [Column Naming Conventions](/intro#column-naming-conventions)):

| Column | Description |
| --- | --- |
| ``cc_density_{d}`` | Count of reachable nodes. |
| ``cc_harmonic_{d}`` | Harmonic closeness. |
| ``cc_farness_{d}`` | Sum of distances to reachable nodes. |
| ``cc_hillier_{d}`` | Hillier normalisation (density² / farness). |
| ``cc_cycles_{d}`` | Circuit rank (meshedness). |
| ``cc_decay_{d}`` | Decay-weighted closeness. |
| ``cc_betweenness_{d}`` | Betweenness centrality. |
| ``cc_betweenness_decay_{d}`` | Decay-weighted betweenness. |

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

 This method does not accept a ``decay_fn`` parameter; angular centralities use angular cost rather than distance-based decay.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">self</div>
    <div class="type">CityNetwork</div>
  </div>
  <div class="desc">

 Returns self for method chaining. Results are written to ``nodes_gdf``.</div>
</div>

### Notes

```python
cn.centrality_simplest(distances=[400, 800, 1600])
```

 Output columns per distance ``d`` (note the ``_ang`` suffix):

| Column | Description |
| --- | --- |
| ``cc_density_{d}_ang`` | Count of reachable nodes (angular routing). |
| ``cc_harmonic_{d}_ang`` | Harmonic closeness (cumulative angular change as impedance). |
| ``cc_farness_{d}_ang`` | Sum of cumulative angular changes to reachable nodes. |
| ``cc_hillier_{d}_ang`` | Hillier normalisation (density² / farness). |
| ``cc_betweenness_{d}_ang`` | Betweenness (simplest angular paths). |

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


 Compute land-use accessibility metrics. Wraps [`compute_accessibilities`](/metrics/layers#compute-accessibilities). All additional keyword arguments are forwarded; see that function for the full parameter list including ``landuse_column_label``, ``accessibility_keys``, ``distances``, ``minutes``, ``decay_fn``, and ``angular``.
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

### Notes

```python
from cityseer import decay

cn, landuses_gdf = cn.compute_accessibilities(
    data_gdf=landuses_gdf,
    landuse_column_label="category",
    accessibility_keys=["retail", "cafe", "park"],
    distances=[400, 800],
    decay_fn=decay.exponential(),
)
# Count of reachable "retail" within 800m
print(cn.nodes_gdf["cc_retail_800"])
# Nearest distance to "park" at the maximum threshold
print(cn.nodes_gdf["cc_park_nearest_max_800"])
```


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


 Compute mixed-use diversity metrics. Wraps [`compute_mixed_uses`](/metrics/layers#compute-mixed-uses). All additional keyword arguments are forwarded; see that function for the full parameter list including ``landuse_column_label``, ``distances``, ``minutes``, ``compute_hill``, ``compute_shannon``, ``compute_gini``, ``decay_fn``, and ``angular``.
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

### Notes

```python
cn, landuses_gdf = cn.compute_mixed_uses(
    data_gdf=landuses_gdf,
    landuse_column_label="category",
    distances=[400, 800],
)
# Hill diversity at q=0 (count of distinct land-uses) at 800m
print(cn.nodes_gdf["cc_hill_q0_800"])
```


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


 Compute statistical aggregations of numerical data over the network. Wraps [`compute_stats`](/metrics/layers#compute-stats). All additional keyword arguments are forwarded; see that function for the full parameter list including ``stats_column_labels``, ``distances``, ``minutes``, ``decay_fn``, and ``angular``.
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

### Notes

```python
from cityseer import decay

cn, prices_gdf = cn.compute_stats(
    data_gdf=prices_gdf,
    stats_column_labels=["price", "floor_area"],
    distances=[800, 1600],
    decay_fn=decay.exponential(),
)
# Weighted mean of "price" at 800m
print(cn.nodes_gdf["cc_price_mean_800"])
# Weighted sum of "floor_area" at 1600m
print(cn.nodes_gdf["cc_floor_area_sum_1600"])
```


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
  <span class="pr">MultiDiGraph</span>
  <span class="pt">]</span>
</div>
</div>


 Convert the network to a NetworkX MultiGraph (or MultiDiGraph if directed). If the network was built with [`from_nx`](#from-nx), returns a copy of the original graph with computed centrality and layer columns added to each edge's data dictionary. Otherwise builds a new cityseer-compatible undirected graph from the internal GeoDataFrame.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">graph</div>
    <div class="type">nx.MultiGraph | nx.MultiDiGraph</div>
  </div>
  <div class="desc">

 A primal edge graph with computed metrics added to edge data.</div>
</div>

### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">NotImplementedError</div>
  </div>
  <div class="desc">

 If the network is directed but was not built via [`from_nx`](#from-nx) (no source graph to export).</div>
</div>

### Notes

```python
cn = CityNetwork.from_nx(G)
cn.centrality_shortest(distances=[800])

# Round-trip: get back a NetworkX graph with metrics on edges
G_with_metrics = cn.to_nx()
u, v, k, data = list(G_with_metrics.edges(keys=True, data=True))[0]
print(data["cc_harmonic_800"])
```


</div>

 
</div>



</section>
