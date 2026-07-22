---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# networks


 Compute network centralities. If you are using `cityseer` for the first time, use the [`CityNetwork`](/api/network) class instead of this module: it builds the network automatically (including cleaning and the dual graph) and exposes the same centrality methods. The functions here are the lower-level functional API, for direct control over the ``NetworkStructure`` and nodes ``GeoDataFrame``.

 Two centrality functions are available, using shortest-path (metric) or simplest-path (angular) heuristics:

- [`centrality_shortest`](#centrality_shortest)
- [`centrality_simplest`](#centrality_simplest)

 [`node_centrality_shortest`](#node_centrality_shortest), [`node_centrality_simplest`](#node_centrality_simplest), and [`segment_centrality`](#segment_centrality) are **deprecated**. They are backwards-compatibility shims for pre-5.0 code and will be removed in a future major release; do not use them in new work.

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
Cautions that apply when computing centralities with these lower-level functions:

- Columns prefixed ``cc_`` are managed by cityseer: recomputing a metric for the same distance overwrites the
matching ``cc_`` columns in place (intended for re-runs). Don't store your own data under this prefix.
- Centralities can be distorted by messy graph topologies such as unnecessary intermediate points along streets
(used to describe road curvature) or overly complex representations of street intersections. Clean the network
first using the [`graph`](/tools/graphs) module (see the
[automatic graph cleaning](/guide/fundamentals#automatic-graph-cleaning) for examples).
- `harmonic` closeness sums inverse distances (``1/c``), so a pair of nodes separated by only a few metres
contributes a very large value, and a pair below 1 m can inflate a node's score severely. `CityNetwork`
construction removes near-duplicate edges and short self-loops automatically; when building the network manually,
consolidate nearby nodes (see [`nx_consolidate_nodes`](/tools/graphs#nx_consolidate_nodes)) before computing
harmonic closeness.
- Simplest (angular) measures require a dual graph representation. `CityNetwork` builds the dual automatically;
this step only applies to the manual method, where primal graphs must be converted with
[`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) before ingestion.
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

 Tip: compute only what you need — a smaller ``closeness`` / ``betweenness`` dict, ``{}`` to skip a whole category, or ``cycles=False`` — evaluates fewer expressions and emits fewer columns.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).</div>
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

 Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.05).</div>
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
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float = 500.0</span>
  </div>
  <div class="param">
    <span class="pn">barriers_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame | None = None</span>
  </div>
  <div class="param">
    <span class="pn">n_nearest_candidates</span>
    <span class="pc">:</span>
    <span class="pa"> int = 50</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">OdMatrix</span>
  <span class="pt">]</span>
</div>
</div>


 Build an OdMatrix from OD flow data and zone boundaries. Computes zone centroids, assigns them to the network with the shared data-layer workflow ([`build_data_map`](/metrics/layers#build_data_map) — the same representation-aware assignment used by accessibility, mixed-uses, stats, and `betweenness_demand`), and constructs a sparse OD weight matrix for use with `betweenness_od`. Each zone is represented by its nearest assigned network node.
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

 Zone boundaries (polygons) or centroids (points). Must be in a projected CRS matching the network, or in ``EPSG:4326`` (will be auto-reprojected).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.graph.NetworkStructure</div>
  </div>
  <div class="desc">

 The network to assign zone centroids to.</div>
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
    <div class="name">max_netw_assign_dist</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Maximum distance (in CRS units, typically metres) for assigning a centroid to the network. Centroids with no valid assignment within this distance are excluded with a warning.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">barriers_gdf</div>
    <div class="type">gpd.GeoDataFrame | None</div>
  </div>
  <div class="desc">

 Optional barriers to respect during assignment, as in the data layers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest candidate edges to consider when assigning centroids to the network, as in the data layers.</div>
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

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).</div>
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

 An [`OdMatrix`](/rustalgos/centrality#odmatrix) mapping (origin, destination) node pairs to trip weights. Build with [`build_od_matrix`](/metrics/networks#build_od_matrix).</div>
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

## betweenness_demand


<div class="content">
<span class="name">betweenness_demand</span><div class="signature multiline">
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
    <span class="pn">origins_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">destinations_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">origin_weight_col</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">destination_weight_col</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str = 'exp(-4 * p)'</span>
  </div>
  <div class="param">
    <span class="pn">closest_destination</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">participation</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.0</span>
  </div>
  <div class="param">
    <span class="pn">metric_name</span>
    <span class="pc">:</span>
    <span class="pa"> str = 'demand'</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float = 100.0</span>
  </div>
  <div class="param">
    <span class="pn">barriers_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame | None = None</span>
  </div>
  <div class="param">
    <span class="pn">n_nearest_candidates</span>
    <span class="pc">:</span>
    <span class="pa"> int = 50</span>
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


 Compute demand-weighted (flow) betweenness from a spatial interaction model. Trips are allocated between weighted origins (e.g. population) and weighted destinations (e.g. attractors) using a **singly (origin-)constrained** spatial interaction model, then routed along shortest network paths so that intermediate nodes accumulate the flow that passes through them. For each origin $o$ and reachable destination $d$ the allocated flow is


$$
W_{od} = W_o \cdot \frac{W_d \cdot f(c_{od})}{K + \sum_{d'} W_{d'} \cdot f(c_{od'})}
$$


 where $f$ is ``decay_fn``, $c_{od}$ is the network distance, and $K$ is a stay-home alternative in the destination choice set, derived from the ``participation`` share. At full participation ($K = 0$, the default) each origin's full weight is conserved and distributed across reachable destinations (destination totals are not constrained — that would require a doubly-constrained / Furness model), and the gravity model is the classic instance of this form, recovered with an exponential ``decay_fn``. Below full participation each origin participates at rate $A_o / (K + A_o)$, where $A_o$ is its accessibility $\sum_{d'} W_{d'} f(c_{od'})$, so trip generation falls where accessibility is low.

 This is the modelled-matrix counterpart to [`betweenness_od`](#betweenness_od): rather than supplying an explicit OD matrix, the per-pair weights are derived from the network distances revealed during routing, computed in a single traversal per origin.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A nodes `GeoDataFrame`; flow betweenness columns are written to it and it is returned.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origins_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` of demand origins (points or centroids).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">destinations_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` of demand destinations / attractors (points or centroids).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_weight_col</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 Column in `origins_gdf` giving each origin's weight (e.g. population).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">destination_weight_col</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 Column in `destinations_gdf` giving each destination's attractiveness weight.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds in metres at which to compute flow betweenness.</div>
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
    <div class="name">decay_fn</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Distance-decay expression for the allocation, using `c` (metric cost) and `p` (normalised progress = `c / threshold`). Defaults to `&quot;exp(-4 * p)&quot;` (scale-free, re-normalised per threshold). For a classic gravity model on absolute distance use e.g. `&quot;exp(-0.002 * c)&quot;`. Because the allocation is normalised per origin, this expression only shapes destination choice; it cannot scale an origin's total outflow. Use `betweenness` expressions for that.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">closest_destination</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If `True`, each origin routes its participating weight to its single nearest reachable destination instead of allocating across all of them.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">participation</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The share of people at a *typical* location who make a trip, in $(0, 1]$. The default `1.0` is full participation: every origin's full weight travels (the classic conserved model, at no extra cost). Below `1.0`, a stay-home option enters the destination choice set — think of staying home as one phantom destination competing with everything an origin can reach: `participation=0.2` means &quot;at a location of median accessibility, one in five people travels&quot;, and locations with better or worse access participate proportionately more or less, so trip generation becomes accessibility-elastic. The underlying stay-home weight is derived internally per distance threshold from the run's own median origin accessibility ($K = A_{med} \cdot (1 - s) / s$, logged per run), so the setting transfers across datasets and thresholds. For pedestrian flows, walking mode shares suggest starting around `0.2` (European cities range roughly 0.15 to 0.3); use a local travel survey's share when available. Results are not knife-edge in this setting. Costs one extra traversal sweep when below `1.0`, and note that output flows are then participating weights rather than total weights.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">metric_name</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Name used for the output column (`cc_{metric_name}_{distance}`). Defaults to `&quot;demand&quot;`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Maximum assignment distance for origin/destination points. Points are assigned to the network with the same workflow as the data layers ([`build_data_map`](/metrics/layers#build_data_map): representation-aware nearest-street assignment, with assignment offsets included in all routed distances — allocation and radius cutoffs alike); points with no valid assignment within this distance are dropped.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">barriers_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 Optional barriers to respect during assignment, as in the data layers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest candidate edges to consider when assigning points to the network, as in the data layers.</div>
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

 Relative tolerance for shortest-path equality, as a percentage. Paths within this margin of the shortest are treated as ties and flow splits across them, so this is the multipath control — the counterpart of a detour ratio in other tools (a 5% tolerance corresponds to a 1.05 detour ratio). Small tolerances can improve conserved-flow fits by spreading flow off knife-edge shortest paths; large ones blur the routing.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `nodes_gdf` with a flow-betweenness column added per distance threshold.</div>
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

 Tip: compute only what you need — pass a smaller ``closeness`` / ``betweenness`` dict, or ``{}`` to skip a whole category — to evaluate fewer expressions and emit fewer columns.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/graph#networkstructure).</div>
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

 Error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON`` (0.05).</div>
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


<div class="function">

## node_centrality_shortest


<div class="content">
<span class="name">node_centrality_shortest</span><div class="signature multiline">
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
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.01831563888873418</span>
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


 Deprecated 4.24 alias for [`centrality_shortest`](#centrality_shortest).
### Deprecated

Since version 5.0. Use `centrality_shortest` with `closeness` / `betweenness` expression dicts. This shim preserves
the 4.24 output (columns `cc_density`, `cc_farness`, `cc_harmonic`, `cc_beta`, `cc_cycles`,
`cc_hillier`, `cc_betweenness`, `cc_betweenness_beta`) and will be removed in a future major release.
See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).


</div>


<div class="function">

## node_centrality_simplest


<div class="content">
<span class="name">node_centrality_simplest</span><div class="signature multiline">
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
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.01831563888873418</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
  </div>
  <div class="param">
    <span class="pn">angular_scaling_unit</span>
    <span class="pc">:</span>
    <span class="pa"> float = 90</span>
  </div>
  <div class="param">
    <span class="pn">farness_scaling_offset</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1</span>
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


 Deprecated 4.24 alias for [`centrality_simplest`](#centrality_simplest).
### Deprecated

Since version 5.0. Use `centrality_simplest` with `closeness` / `betweenness` expression dicts. This shim preserves the
4.24 output (angular columns `cc_density_ang`, `cc_farness_ang`, `cc_harmonic_ang`, `cc_hillier_ang`,
`cc_betweenness_ang`) and will be removed in a future major release. See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).


</div>


<div class="function">

## segment_centrality


<div class="content">
<span class="name">segment_centrality</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">*_args</span>
  </div>
  <div class="param">
    <span class="pn">**_kwargs</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Removed in 5.0; raises with guidance.
### Deprecated

Since version 5.0. The continuous-segment engine (`segment_density` / `harmonic` / `beta` / `betweenness`) was removed
at the low level, so the old numbers cannot be reproduced. The nearest equivalent is
`centrality_shortest(..., segment_weighted=True)` — a different calculation. See [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).


</div>



</section>
