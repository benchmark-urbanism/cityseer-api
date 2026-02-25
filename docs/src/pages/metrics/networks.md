---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# networks


 Compute network centralities. There are three network centrality methods available depending on whether you're using a node-based or segment-based approach, with the former available in both shortest and simplest (angular) variants.

- [`node_centrality_shortest`](#node-centrality-shortest)
- [`node_centrality_simplest`](#node-centrality-simplest)
- [`segment_centrality`](#segment-centrality)

 These methods wrap the underlying `rust` optimised functions for computing centralities. Multiple classes of measures and distances are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar strategies.

 See the accompanying paper on `arXiv` for additional information about methods for computing centrality measures.

:::note
The reasons for picking one approach over another are varied:

- Node based centralities compute the measures relative to each reachable node within the threshold distances. For
this reason, they can be susceptible to distortions caused by messy graph topologies such redundant and varied
concentrations of degree=2 nodes (e.g. to describe roadway geometry) or needlessly complex representations of
street intersections. In these cases, the network should first be cleaned using methods such as those available in
the [`graph`](/tools/graphs) module (see the [graph cleaning guide](/guide#graph-cleaning) for examples). If a
network topology has varied intensities of nodes but the street segments are less spurious, then segmentised methods
can be preferable because they are based on segment distances: segment aggregations remain the same regardless of
the number of intervening nodes, however, are not immune from situations such as needlessly complex representations
of roadway intersections or a proliferation of walking paths in greenspaces;
- Node-based `harmonic` centrality can be problematic on graphs where nodes are erroneously placed too close
together or where impedances otherwise approach zero, as may be the case for simplest-path measures or small
distance thesholds. This happens because the outcome of the division step can balloon towards $\infty$ once
impedances decrease below 1.
- Note that `cityseer`'s implementation of simplest (angular) measures work on both primal and dual graphs (node only).
- Measures should only be directly compared on the same topology because different topologies can otherwise affect
the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed
on primal graphs because this does not account for the impact of differing topologies. Dual graph representations
can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a
four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual.
This effect is amplified for denser regions of the network.
- Segmentised versions of centrality measures should not be computed on dual graph topologies because street segment
lengths would be duplicated for each permutation of dual edge spanning street intersections. By way of example,
the contribution of a single edge segment at a four-way intersection would be duplicated three times.
- The usual formulations of closeness or normalised closeness are discouraged because these do not behave
suitably for localised graphs. Harmonic closeness or Hillier normalisation (which resembles a simplified form of
Improved Closeness Centrality proposed by Wasserman and Faust) should be used instead.
- Network decomposition can be a useful strategy when working at small distance thresholds, and confers advantages
such as more regularly spaced snapshots and fewer artefacts at small distance thresholds where street edges
intersect distance thresholds. However, the regular spacing of the decomposed segments will introduce spikes in the
distributions of node-based centrality measures when working at very small distance thresholds. Segmentised versions
may therefore be preferable when working at small thresholds on decomposed networks.
:::


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
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
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
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Deprecated. Use closeness_shortest and/or betweenness_shortest instead.

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
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.01831563888873418</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1.33333</span>
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

 An [`OdMatrix`](/rustalgos/centrality#odmatrix) mapping (origin, destination) node pairs to trip weights. Build with [`config.build_od_matrix`](/config#build-od-matrix).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A list of $\beta$ to be used for the exponential decay function for weighted metrics.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A list of walking times in minutes to be used for calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and distance thresholds $d_{max}$.</div>
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
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
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
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Deprecated. Use closeness_simplest and/or betweenness_simplest instead.

</div>


<div class="function">

## segment_centrality


<div class="content">
<span class="name">segment_centrality</span><div class="signature multiline">
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
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
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
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute segment-based network centrality using the shortest path heuristic. > Simplest path heuristics introduce conceptual and practical complications and support is deprecated since v4.

 > For conceptual and practical reasons, segment based centralities are not weighted by node weights.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method. The outputs of calculations will be written to this `GeoDataFrame`, which is then returned from the method.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ for distance-weighted metrics will be determined implicitly using `min_threshold_wt`. If the `distances` parameter is not provided, then the `beta` or `minutes` parameters must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A list of $\beta$ to be used for the exponential decay function for weighted metrics. The $d_{max}$ thresholds for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distances` or `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A list of walking times in minutes to be used for calculations. The $d_{max}$ thresholds for unweighted metrics and $\beta$ for distance-weighted metrics will be determined implicitly using the `speed_m_s` and `min_threshold_wt` parameters. If the `minutes` parameter is not provided, then the `distances` or `betas` parameters must be provided instead.</div>
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
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas) for more information.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `speed_m_s` parameter can be configured to generate custom mappings between walking times and distance thresholds $d_{max}$.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics.</div>
</div>

### Notes

 Segment path centralities are available with the following keys:

| key                 | formula | notes |
| ------------------- | :-----: |------ |
| seg_density     | $$\sum_{(a, b)}^{edges}d_{b} - d_{a}$$ | A summation of edge lengths. |
| seg_harmonic    | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\ln(b) -\ln(a)$$ | A continuous form of harmonic closeness centrality applied to edge lengths. |
| seg_beta        | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\frac{\exp(-\beta\cdot b) -\exp(-\beta\cdot a)}{-\beta}$$ | A continuous form of beta-weighted (gravity index) centrality applied to edge lengths. |
| seg_betweenness | | A continuous form of betweenness: Resembles `segment_beta` applied to edges situated on shortest paths between all nodes $j$ and $k$ passing through $i$. |

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
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">probe_density</span>
    <span class="pc">:</span>
    <span class="pa"> float = 4.0</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">delta</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">sample_rate</span>
    <span class="pc">:</span>
    <span class="pa"> dict[int, float] | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute closeness centrality using shortest paths with adaptive source sampling. Uses spatially stratified sampling with IPW correction. The inclusion probability passed to Rust is the marginal rate ``actual_p = n_sources / n_live`` rather than per-node cell-specific probabilities, making the estimator approximately unbiased. Set ``sample=False`` to disable sampling and compute exact centrality from all sources. Supply ``sample_rate`` to use fixed per-distance fractions, independent of reachability.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A NetworkStructure.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds (meters).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Decay parameters (beta).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Time thresholds (minutes).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Minimum weight for beta/distance conversion.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Travel speed (m/s).</div>
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
    <div class="name">probe_density</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Probes per km² for reachability estimation.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding approximation error bound.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">delta</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding failure probability.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If False, disables adaptive sampling and computes exact centrality from all sources.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample_rate</div>
    <div class="type">dict[int, float] | None</div>
  </div>
  <div class="desc">

 Fixed sampling fractions keyed by distance (meters). When provided, overrides the Hoeffding probe path: each distance uses the given fraction of live nodes as sources, regardless of reachability. Distances absent from the dict are computed in full. Enables reach-agnostic comparison across graphs.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input GeoDataFrame with closeness columns added.</div>
</div>


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
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">probe_density</span>
    <span class="pc">:</span>
    <span class="pa"> float = 4.0</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">delta</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">sample_rate</span>
    <span class="pc">:</span>
    <span class="pa"> dict[int, float] | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute closeness centrality using simplest paths with adaptive source sampling. Uses spatially stratified sampling with IPW correction. The inclusion probability passed to Rust is the marginal rate ``actual_p = n_sources / n_live`` rather than per-node cell-specific probabilities, making the estimator approximately unbiased. Set ``sample=False`` to disable sampling and compute exact centrality from all sources. Supply ``sample_rate`` to use fixed per-distance fractions, independent of reachability.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A NetworkStructure.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds (meters).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Decay parameters (beta).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Time thresholds (minutes).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Minimum weight for beta/distance conversion.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Travel speed (m/s).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angular_scaling_unit</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Scaling unit for angular cost.</div>
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
    <div class="name">probe_density</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Probes per km² for reachability estimation.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding approximation error bound.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">delta</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding failure probability.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If False, disables adaptive sampling and computes exact centrality from all sources.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample_rate</div>
    <div class="type">dict[int, float] | None</div>
  </div>
  <div class="desc">

 Fixed sampling fractions keyed by distance (meters). When provided, overrides the Hoeffding probe path: each distance uses the given fraction of live nodes as sources, regardless of reachability. Distances absent from the dict are computed in full. Enables reach-agnostic comparison across graphs.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input GeoDataFrame with closeness columns added.</div>
</div>


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
    <span class="pa"> float = 0.0</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">probe_density</span>
    <span class="pc">:</span>
    <span class="pa"> float = 4.0</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.05</span>
  </div>
  <div class="param">
    <span class="pn">delta</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">sample_rate</span>
    <span class="pc">:</span>
    <span class="pa"> dict[int, float] | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute betweenness centrality using shortest paths with adaptive source sampling. Uses spatially stratified sampling with IPW correction. The inclusion probability passed to Rust is the marginal rate ``actual_p = n_sources / n_live`` rather than per-node cell-specific probabilities, making the estimator approximately unbiased. Set ``sample=False`` to disable sampling and compute exact centrality from all sources. Supply ``sample_rate`` to use fixed per-distance fractions, independent of reachability.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A NetworkStructure.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds (meters).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Decay parameters (beta).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Time thresholds (minutes).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Minimum weight for beta/distance conversion.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Travel speed (m/s).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Relative tolerance for betweenness path equality. Paths within `tolerance` fraction of the shortest are treated as near-equal for multi-predecessor Brandes betweenness. Set to 0.0 for exact shortest paths only. A value like 0.02 (2%) captures pedestrian indifference to near-equal routes.</div>
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
    <div class="name">probe_density</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Probes per km² for reachability estimation.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding approximation error bound.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">delta</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding failure probability.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If False, disables adaptive sampling and computes exact centrality from all sources.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample_rate</div>
    <div class="type">dict[int, float] | None</div>
  </div>
  <div class="desc">

 Fixed sampling fractions keyed by distance (meters). When provided, overrides the Hoeffding probe path: each distance uses the given fraction of live nodes as sources, regardless of reachability. Distances absent from the dict are computed in full. Enables reach-agnostic comparison across graphs.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input GeoDataFrame with betweenness columns added.</div>
</div>


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
    <span class="pa"> float = 0.0</span>
  </div>
  <div class="param">
    <span class="pn">random_seed</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">probe_density</span>
    <span class="pc">:</span>
    <span class="pa"> float = 4.0</span>
  </div>
  <div class="param">
    <span class="pn">epsilon</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.05</span>
  </div>
  <div class="param">
    <span class="pn">delta</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">sample</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">sample_rate</span>
    <span class="pc">:</span>
    <span class="pa"> dict[int, float] | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute betweenness centrality using simplest paths with adaptive source sampling. Uses spatially stratified sampling with IPW correction. The inclusion probability passed to Rust is the marginal rate ``actual_p = n_sources / n_live`` rather than per-node cell-specific probabilities, making the estimator approximately unbiased. Set ``sample=False`` to disable sampling and compute exact centrality from all sources. Supply ``sample_rate`` to use fixed per-distance fractions, independent of reachability.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A NetworkStructure.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A GeoDataFrame representing nodes. Results are written to this GeoDataFrame.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distance thresholds (meters).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Decay parameters (beta).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Time thresholds (minutes).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Minimum weight for beta/distance conversion.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Travel speed (m/s).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">tolerance</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Relative tolerance for near-equal angular path detection. 0.0 = exact simplest paths only.</div>
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
    <div class="name">probe_density</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Probes per km² for reachability estimation.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">epsilon</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding approximation error bound.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">delta</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Hoeffding failure probability.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If False, disables adaptive sampling and computes exact centrality from all sources.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">sample_rate</div>
    <div class="type">dict[int, float] | None</div>
  </div>
  <div class="desc">

 Fixed sampling fractions keyed by distance (meters). When provided, overrides the Hoeffding probe path: each distance uses the given fraction of live nodes as sources, regardless of reachability. Distances absent from the dict are computed in full. Enables reach-agnostic comparison across graphs.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input GeoDataFrame with betweenness columns added.</div>
</div>


</div>



</section>
