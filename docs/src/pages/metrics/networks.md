---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# networks


 Compute network centralities. There are two network centrality methods available in both shortest and simplest (angular) variants.

- [`centrality_shortest`](#centrality-shortest)
- [`centrality_simplest`](#centrality-simplest)

 These methods wrap the underlying `rust` optimised functions for computing centralities. Multiple classes of measures and distances are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar strategies.

 When `sample=True`, adaptive sampling uses the Hoeffding bound to select a distance-dependent sampling probability. The `epsilon` parameter controls the error tolerance (lower = more samples, higher accuracy). The default for when sampling is enabled is 0.06.

| Distance | ε=0.02 | ε=0.04 | ε=0.06 | ε=0.08 | ε=0.1 |
|----------|--------|--------|--------|--------|-------|
| 1 km     | 100%   | 100%   | 100%   | 100%   | 100%  |
| 2 km     | 100%   | 100%   | 100%   | 100%   | 100%  |
| 5 km     | 100%   | 100%   | 58.7%  | 33.0%  | 21.1% |
| 10 km    | 100%   | 37.3%  | 16.6%  | 9.3%   | 6.0%  |
| 20 km    | 41.5%  | 10.4%  | 4.6%   | 2.6%   | 1.7%  |

Sampling is exact (100%) at short distances and becomes progressively sparser at longer distances where reachability is high enough to maintain relative accuracy. The theoretical speedup is approximately 1/p. When comparing centrality values across different locations, use the same epsilon to ensure consistent error tolerances and comparable sampling rates.

:::note
The reasons for picking one approach over another are varied:

- Centralities compute the measures relative to each reachable node within the threshold distances. For
this reason, they can be susceptible to distortions caused by messy graph topologies such redundant and varied
concentrations of degree=2 nodes (e.g. to describe roadway geometry) or needlessly complex representations of
street intersections. In these cases, the network should first be cleaned using methods such as those available in
the [`graph`](/tools/graphs) module (see the [network preparation guide](/guide#network-preparation) for examples).
- `harmonic` centrality can be problematic on graphs where nodes are erroneously placed too close
together or where impedances otherwise approach zero, as may be the case for simplest-path measures or small
distance thesholds. This happens because the outcome of the division step can balloon towards $\infty$ once
impedances decrease below 1.
- Simplest (angular) measures require a dual graph representation. Convert primal graphs with
[`graphs.nx_to_dual`](/tools/graphs#nx-to-dual) before ingesting them.
- Measures should only be directly compared on the same topology because different topologies can otherwise affect
the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed
on primal graphs because this does not account for the impact of differing topologies. Dual graph representations
can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a
four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual.
This effect is amplified for denser regions of the network.
- The usual formulations of closeness or normalised closeness are discouraged because these do not behave
suitably for localised graphs. Harmonic closeness or Hillier normalisation (which resembles a simplified form of
Improved Closeness Centrality proposed by Wasserman and Faust) should be used instead.
- Network decomposition can be a useful strategy when working at small distance thresholds, and confers advantages
such as more regularly spaced snapshots and fewer artefacts at small distance thresholds where street edges
intersect distance thresholds.
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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
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


 Compute centrality using shortest paths with a single Dijkstra per source. When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style Dijkstra traversal per source produces the data for both closeness accumulation and betweenness backpropagation, halving computation time compared to computing them separately.

 The decay closeness and betweenness decay metrics are computed using a decay function expressed as a string with the variable `p`, which represents normalised progress from the source (`p = 0`) to the distance threshold (`p = 1`), where `p = cost / max_cost`. By default, `decay_fn` is `"exp(-4 * p)"` (exponential decay reaching ~1.8% at the threshold). Helper functions for constructing decay expressions are available in the `cityseer.decay` module.

 .. versionchanged:: 4.24.0 The `cycles` output now measures the circuit rank of the locally reachable subgraph (`m - n + c`), computed per source and then target-aggregated using the same source/IPW framework as the other shortest-path metrics. This provides a more stable measure of network meshedness (independent loops / city blocks) than the older tree-cycle heuristic.

 When ``sample=True``, sampling probability is derived from each distance threshold using a canonical grid network model (see ``sampling.compute_distance_p``). This produces deterministic, reach-agnostic sample fractions that are comparable across networks.
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
    <div class="name">compute_closeness</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute closeness centralities. True by default. The `cycles` output measures the circuit rank of the source's locally reachable subgraph and target-aggregates that loopiness contribution over all sources that can reach each node within the threshold.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_betweenness</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute betweenness centralities. True by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">decay_fn</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An expression string for the decay function, using the variable `p` (normalised progress from 0 to 1, where `p = cost / max_cost`). At the source `p = 0` and at the distance threshold `p = 1`. Default is `&quot;exp(-4 * p)&quot;` (exponential decay reaching ~1.8% at the threshold). Use `&quot;1&quot;` for flat (unweighted) decay metrics, or provide a custom expression. Helper functions are available in the `cityseer.decay` module.</div>
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

 Relative tolerance for betweenness path equality, as a percentage (e.g. 1.0 = 1%). Paths within this percentage of the shortest are treated as near-equal for multi-predecessor Brandes betweenness. A tiny internal epsilon is always enforced as a minimum for floating-point stability.</div>
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

 If True, uses distance-based Bernoulli sampling with inverse-probability weighting (IPW). The sampling probability is derived from each distance threshold using a canonical grid model (see ``sampling.compute_distance_p``). At distances where the sampling probability exceeds the live fraction, exact computation is used instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.</div>
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

 For worked examples with real-world data, see the [Metric Centrality](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_metric_centrality.html) and [OSM Centrality](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/osm_centrality.html) recipes.

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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
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


 Compute OD-weighted betweenness centrality using the shortest path heuristic. Weights betweenness by origin-destination trip counts from a sparse OD matrix. Only source nodes with outbound trips are traversed, and each shortest-path contribution is scaled by the corresponding OD weight. Closeness metrics are not computed.
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
    <div class="name">decay_fn</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An expression string for the decay function, using the variable `p` (normalised progress from 0 to 1, where `p = cost / max_cost`). At the source `p = 0` and at the distance threshold `p = 1`. Default is `&quot;exp(-4 * p)&quot;` (exponential decay reaching ~1.8% at the threshold). Use `&quot;1&quot;` for flat (unweighted) decay metrics, or provide a custom expression. Helper functions are available in the `cityseer.decay` module.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in metres per second for converting `minutes` to distance thresholds.</div>
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


 Compute centrality using simplest (angular) paths with a single Dijkstra per source. When both `compute_closeness` and `compute_betweenness` are True, a single Brandes-style Dijkstra traversal per source produces the data for both closeness accumulation and betweenness backpropagation.

 This function does not accept a `decay_fn` parameter; angular (simplest-path) centralities use angular cost rather than distance-based decay weighting.

 .. versionchanged:: 4.24.0 Angular routing now uses endpoint-aware dual-graph traversal instead of bearing-based angular costs. This requires a dual graph representation (convert with [`graphs.nx_to_dual`](/tools/graphs#nx-to-dual)). The `tolerance` parameter now uses the same relative-percentage semantics as shortest-path betweenness, but applies to angular route cost instead of metric distance. User-facing `tolerance=0.0` means no additional tolerance beyond a tiny internal epsilon used for floating-point stability. Closeness values are nearly identical; betweenness values may differ slightly.
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
    <div class="name">compute_closeness</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute closeness centralities. True by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_betweenness</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute betweenness centralities. True by default.</div>
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

 Relative tolerance for angular betweenness path equality, as a percentage (e.g. 1.0 = 1%). Paths whose angular route cost is within this percentage of the best angular route are treated as near-equal for multi-predecessor Brandes betweenness. A tiny internal epsilon is always enforced as a minimum for floating-point stability.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angular_scaling_unit</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Scaling unit for angular cost normalisation.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">farness_scaling_offset</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Offset for farness calculation.</div>
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

 If True, uses distance-based Bernoulli sampling with inverse-probability weighting (IPW). The sampling probability is derived from each distance threshold using a canonical grid model (see ``sampling.compute_distance_p``). At distances where the sampling probability exceeds the live fraction, exact computation is used instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Normalised additive error tolerance for sampling. Defaults to ``sampling.HOEFFDING_EPSILON``.</div>
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

 For a worked example, see the [Angular Centrality](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_angular_centrality.html) recipe.

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


 Compute closeness centrality using shortest paths. Wraps `centrality_shortest` with `compute_closeness=True` and `compute_betweenness=False`. Uses exponential decay (`"exp(-4 * p)"`) by default; pass `decay_fn` to `centrality_shortest` for a custom decay function.

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


 Compute closeness centrality using simplest (angular) paths. Wraps `centrality_simplest` with `compute_closeness=True` and `compute_betweenness=False`.

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


 Compute betweenness centrality using shortest paths. Wraps `centrality_shortest` with `compute_closeness=False` and `compute_betweenness=True`. Uses exponential decay (`"exp(-4 * p)"`) by default; pass `decay_fn` to `centrality_shortest` for a custom decay function.

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


 Compute betweenness centrality using simplest (angular) paths. Wraps `centrality_simplest` with `compute_closeness=False` and `compute_betweenness=True`.

</div>



</section>
