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
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute node-based network centrality using the shortest path heuristic.
:::note
Node weights are taken into account when computing centralities. These would typically be initialised at 1 unless
manually specified.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method.</div>
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

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
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
    <div class="name">jitter_scale</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The scale of random jitter to add to shortest path calculations, useful for situations with highly rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the shortest path calculations to provide random variation to the paths traced through the network. When working with shortest paths in metres, the random value represents distance in metres. When using a simplest path heuristic, the jitter will represent angular change in degrees.</div>
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

 The following keys use the shortest-path heuristic:

| key                   | formula | notes |
| ----------------------| :------:| ----- |
| density          | $$\sum_{j\neq{i}}^{n}1$$ | A summation of nodes. |
| harmonic         | $$\sum_{j\neq{i}}^{n}\frac{1}{d_{(i,j)}}$$ | Harmonic closeness is an appropriate form of closeness centrality for localised implementations constrained by the threshold $d_{max}$. |
| hillier          | $$\frac{(n-1)^2}{\sum_{j \neq i}^{n} d_{(i,j)}}$$ | The square of node density divided by farness. This is also a simplified form of Improved Closeness Centrality. |
| beta             | $$\sum_{j\neq{i}}^{n} \\ \exp(-\beta\cdot d[i,j])$$ | Also known as the gravity index. This is a spatial impedance metric differentiated from other closeness centralities by the use of an explicit $\beta$ parameter, which can be used to model the decay in walking tolerance as distances increase. |
| cycles           | $$\sum_{j\neq{i}j=cycle}^{n}1$$ | A summation of network cycles. |
| farness          | $$\sum_{j\neq{i}}^{n}d_{(i,j)}$$ | A summation of distances in metres. |
| betweenness      | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | Betweenness centrality summing all shortest-paths traversing each node $i$. |
| betweenness_beta | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n} \\ \exp(-\beta\cdot d[j,k])$$ | Applies a spatial impedance decay function to betweenness centrality. $d$ represents the full distance from any $j$ to $k$ node pair passing through node $i$. |

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
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute node-based network centrality using the simplest path (angular) heuristic.
:::note
Node weights are taken into account when computing centralities. These would typically be initialised at 1 unless
manually specified.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method.</div>
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

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
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
    <div class="name">jitter_scale</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The scale of random jitter to add to shortest path calculations, useful for situations with highly rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the shortest path calculations to provide random variation to the paths traced through the network. When working with shortest paths in metres, the random value represents distance in metres. When using a simplest path heuristic, the jitter will represent angular change in degrees.</div>
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

 The following keys use the simplest-path heuristic:

| key                   | formula | notes |
| ----------------------| :------:| ----- |
| density_ang | $$\sum_{j\neq{i}}^{n}1$$ | A summation of nodes. |
| harmonic_ang    | $$\sum_{j\neq{i}}^{n}\frac{1}{d_{(i,j)}}$$ | Harmonic closeness is an appropriate form of closeness centrality for localised implementations constrained by the threshold $d_{max}$. |
| hillier_ang | $$\frac{(n-1)^2}{\sum_{j \neq i}^{n} d_{(i,j)}}$$ | The square of node density divided by farness. This is also a simplified form of Improved Closeness Centrality. |
| farness_ang | $$\sum_{j\neq{i}}^{n}d_{(i,j)}$$ | A summation of distances in metres. |
| betweenness_ang | $$\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$$ | Betweenness centrality summing all shortest-paths traversing each node $i$. |

The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular` parameter is explicitly set to `True`:

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
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
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

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) method.</div>
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

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
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
    <div class="name">jitter_scale</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The scale of random jitter to add to shortest path calculations, useful for situations with highly rectilinear grids or for smoothing metrics on messy network representations. A random sample is drawn from a range of zero to one and is then multiplied by the specified `jitter_scale`. This random value is added to the shortest path calculations to provide random variation to the paths traced through the network. When working with shortest paths in metres, the random value represents distance in metres. When using a simplest path heuristic, the jitter will represent angular change in degrees.</div>
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
| seg_beta        | $$\sum_{(a, b)}^{edges}\int_{a}^{b}\frac{\exp(-\beta\cdot b) -\exp(-\beta\cdot a)}{-\beta}$$ | A  # pylint: disable=line-too-long continuous form of beta-weighted (gravity index) centrality applied to edge lengths. |
| seg_betweenness | | A continuous form of betweenness: Resembles `segment_beta` applied to edges situated on shortest paths between all nodes $j$ and $k$ passing through $i$. |

</div>



</section>
