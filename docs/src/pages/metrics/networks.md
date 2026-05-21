---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# networks


 Compute network centralities. Two centrality methods are available, using shortest-path (metric) or simplest-path (angular) heuristics:

- [`centrality_shortest`](#centrality-shortest)
- [`centrality_simplest`](#centrality-simplest)

 Metrics are specified as ``{name: expression}`` dicts using variables ``c`` (cost) and ``p`` (normalised progress). For shortest paths, ``c`` is metric distance and ``p = c / threshold``. For simplest paths, ``c`` is angular cost and ``p`` is normalised time progress.

 Four categories of metrics are supported:

- **closeness**: per-reached-node accumulation (e.g. ``{"harmonic": "1/c", "density": "1"}``)
- **betweenness**: target seed weight in Brandes backpropagation (e.g. ``{"betweenness": "1"}``)
- **cycles**: circuit rank (boolean flag)
- **postprocess**: derived from computed columns in Python (e.g. ``{"hillier": "density**2 / farness"}``)

 Pass ``None`` for defaults or ``{}`` to skip a category.

 Per-node ``weight`` values (default ``1.0``, set on the nodes ``GeoDataFrame`` or read from NetworkX node attributes) apply gravity-style weighting to centrality: closeness weights each reachable node by its destination weight (so ``density`` becomes ``sum_j w_j`` rather than a plain count), and betweenness weights each origin-destination pair by the product of its endpoint weights. The same weighting is applied identically whether or not sampling is used. Land-use, mixed-use, and statistical aggregations are intentionally *not* node-weighted.

 When `segment_weighted=True`, node weights are temporarily set to the primal edge (street segment) lengths so that centrality measures reflect total reachable street length rather than node counts (closeness by destination length, betweenness by the product of endpoint lengths). This is a convenience preset over the per-node ``weight`` mechanism and requires a dual graph representation.

 When `sample=True`, only a subset of nodes are used as sources for centrality computation, with results corrected to approximate the full computation.

:::note
The reasons for picking one approach over another are varied:

- Centralities can be distorted by messy graph topologies such as unnecessary intermediate points along streets
(used to describe road curvature) or overly complex representations of street intersections. Clean the network
first using the [`graph`](/tools/graphs) module (see the
[automatic graph cleaning](/guide#automatic-graph-cleaning) for examples).
- `harmonic` centrality can produce inflated values when nodes are very close together, because the
inverse-distance calculation amplifies small distances. This is more likely with simplest-path measures or short
distance thresholds.
- Simplest (angular) measures require a dual graph representation. Convert primal graphs with
[`graphs.nx_to_dual`](/tools/graphs#nx-to-dual) before ingesting them.
- Metrics should only be compared across networks that use the same graph representation (both primal or both
dual), because the differing number of nodes and edges between representations affects the metric values. For
example, a four-way intersection consisting of one node with four edges on a primal graph translates to four
nodes and six edges on the dual. This effect is amplified for denser regions of the network.
- Standard closeness and normalised closeness do not work well with distance-bounded analysis. Use harmonic
closeness or Hillier normalisation instead.
:::


<div class="function">

## centrality_shortest


<div class="content">
<span class="name">centrality_shortest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">closeness</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">cycles</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">postprocess</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">segment_weighted</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute centrality using shortest paths with a single Dijkstra per source. Metrics are specified as ``{name: expression}`` dicts. Expressions use two variables:

- ``c``: the raw cost (metric distance in metres for shortest-path analysis)
- ``p``: normalised progress from 0 at the source to 1 at the distance threshold (``p = c / threshold``)

 Pass ``None`` for defaults or ``{}`` to skip a category.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds in metres at which to compute centrality measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking times in minutes; converted to distance thresholds using `speed_m_s`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">closeness</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Closeness metric expressions. Each entry is ``{name: expr(c, p)}``, accumulated per reached node. ``None`` uses defaults: density, farness, harmonic, decay.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betweenness</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Betweenness metric expressions. Each entry is ``{name: expr(c, p)}``, used as the weight assigned to each destination when accumulating betweenness contributions along shortest paths. ``None`` uses defaults: betweenness, betweenness_decay.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cycles</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If True, compute circuit rank (cycle count) for each node. Default True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">postprocess</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Derived metrics computed in Python from the closeness/betweenness results. ``None`` uses default: ``{&quot;hillier&quot;: &quot;density**2 / farness&quot;}``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in metres per second for converting `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Relative tolerance for betweenness path equality, as a percentage (e.g. 1.0 = 1%).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">segment_weighted</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If True, weight by primal edge (street segment) lengths. Requires a dual graph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Optional seed for reproducible sampling.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If True, enables adaptive sampling at longer distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.06).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `nodes_gdf` parameter is returned with additional centrality columns.</div>
</div>

### Notes

```python
from cityseer.tools import mock, graphs, io
from cityseer.metrics import networks

G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
nodes_gdf = networks.centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[400, 800],
)
print(nodes_gdf[["cc_harmonic_400", "cc_betweenness_800"]])
```


</div>


<div class="function">

## build_od_matrix


<div class="content">
<span class="name">build_od_matrix</span><div class="signature multiline">
  <span class="pt">(</span>
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
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">origin_col</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">destination_col</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">weight_col</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">zone_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">max_snap_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float = 500.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">OdMatrix</span>
  <span class="pt">]</span>
</div>
</div>


 Build an OdMatrix from OD flow data and zone boundaries. Computes zone centroids, snaps them to the nearest network nodes, and constructs a sparse OD weight matrix for use with `betweenness_od`.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">od_df</div>
    <div class="type">pd.DataFrame</div>
  </div>
  <div class="desc">

 Origin-destination flow data with columns for origin zone, destination zone, and weight.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">zones_gdf</div>
    <div class="type">gpd.GeoDataFrame</div>
  </div>
  <div class="desc">

 Zone boundaries (polygons) or centroids (points). Must be in a projected CRS matching the network, or in EPSG:4326 (will be auto-reprojected).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.graph.NetworkStructure</div>
  </div>
  <div class="desc">

 The network to snap zone centroids to.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Column in od_df containing origin zone identifiers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">destination_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Column in od_df containing destination zone identifiers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">weight_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Column in od_df containing trip weights (e.g., number of bicycle commuters).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">zone_id_col</div>
    <div class="type">str | None</div>
  </div>
  <div class="desc">

 Column in zones_gdf containing zone identifiers matching origin_col/destination_col. If None, uses the GeoDataFrame index.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_snap_dist</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Maximum distance (in CRS units, typically metres) for snapping a centroid to a network node. Centroids beyond this distance are excluded with a warning.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">rustalgos.centrality.OdMatrix</div>
  </div>
  <div class="desc">

 Sparse OD matrix ready for use with `betweenness_od`.</div>
</div>


</div>


<div class="function">

## betweenness_od


<div class="content">
<span class="name">betweenness_od</span><div class="signature multiline">
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
    <span class="pn">od_matrix</span>
    <span class="pc">:</span>
    <span class="pa"> OdMatrix</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute OD-weighted betweenness centrality using the shortest path heuristic.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">od_matrix</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 An [`OdMatrix`](/rustalgos/centrality#odmatrix) mapping (origin, destination) node pairs to trip weights. Build with [`build_od_matrix`](/metrics/networks#build-od-matrix).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds in metres at which to compute betweenness.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking times in minutes; converted to distance thresholds using `speed_m_s`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betweenness</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Betweenness metric expressions. ``None`` uses defaults: betweenness, betweenness_decay.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in metres per second for converting `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Relative tolerance for path equality, as a percentage.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `nodes_gdf` parameter is returned with additional betweenness columns.</div>
</div>


</div>


<div class="function">

## centrality_simplest


<div class="content">
<span class="name">centrality_simplest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">closeness</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">postprocess</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">segment_weighted</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute centrality using simplest (angular) paths with a single Dijkstra per source. Expressions use ``c`` (angular cost) and ``p`` (normalised time progress).
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. The outputs of calculations will be written to this `GeoDataFrame`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds in metres at which to compute centrality measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking times in minutes; converted to distance thresholds using `speed_m_s`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">closeness</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Closeness metric expressions. ``None`` uses defaults: density, farness, harmonic.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betweenness</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Betweenness metric expressions. ``None`` uses defaults: betweenness.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">postprocess</div>
    <div class="type">dict[str, str]</div>
  </div>
  <div class="desc">

 Derived metrics. ``None`` uses default: ``{&quot;hillier&quot;: &quot;density**2 / farness&quot;}``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in metres per second for converting `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Relative tolerance for angular betweenness path equality, as a percentage.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">segment_weighted</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If True, weight by primal edge (street segment) lengths. Requires a dual graph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Optional seed for reproducible sampling.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If True, enables adaptive sampling at longer distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.06).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `nodes_gdf` parameter is returned with additional centrality columns.</div>
</div>

### Notes

```python
from cityseer.tools import mock, graphs, io
from cityseer.metrics import networks

G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G_dual = graphs.nx_to_dual(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G_dual)
nodes_gdf = networks.centrality_simplest(
    network_structure,
    nodes_gdf,
    distances=[400, 800],
)
print(nodes_gdf[["cc_harmonic_400_ang", "cc_betweenness_800_ang"]])
```


</div>


<div class="function">

## closeness_shortest


<div class="content">
<span class="name">closeness_shortest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute closeness centrality using shortest paths. Wraps `centrality_shortest` with betweenness disabled.

</div>


<div class="function">

## closeness_simplest


<div class="content">
<span class="name">closeness_simplest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute closeness centrality using simplest (angular) paths. Wraps `centrality_simplest` with betweenness disabled.

</div>


<div class="function">

## betweenness_shortest


<div class="content">
<span class="name">betweenness_shortest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute betweenness centrality using shortest paths. Wraps `centrality_shortest` with closeness disabled.

</div>


<div class="function">

## betweenness_simplest


<div class="content">
<span class="name">betweenness_simplest</span><div class="signature multiline">
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
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute betweenness centrality using simplest (angular) paths. Wraps `centrality_simplest` with closeness disabled.

</div>



</section>
