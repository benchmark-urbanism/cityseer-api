---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# layers


<div class="function">

## assign_gdf_to_network


<div class="content">
<span class="name">assign_gdf_to_network</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int | float</span>
  </div>
  <div class="param">
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">DataMap</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Assign a `GeoDataFrame` to a [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the network. Data points will be assigned to the two closest network nodes — one in either direction — based on the closest adjacent street edge. This facilitates a dynamic spatial aggregation strategy which will select the shortest distance to a data point relative to either direction of approach.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing data points. The coordinates of data points should correspond as precisely as possible to the location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the building entrance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.NetworkStructure</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_id_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional column name for data point keys. This is used for deduplicating points representing a shared source of information. For example, where a single greenspace is represented by many entrances as datapoints, only the nearest entrance (from a respective location) will be considered (during aggregations) when the points share a datapoint identifier.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">data_map</div>
    <div class="type">rustalgos.DataMap</div>
  </div>
  <div class="desc">

 A [`rustalgos.DataMap`](/rustalgos#datamap) instance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.</div>
</div>

### Notes

:::note
The `max_assign_dist` parameter should not be set overly low. The `max_assign_dist` parameter sets a crow-flies
distance limit on how far the algorithm will search in its attempts to encircle the data point. If the
`max_assign_dist` is too small, then the algorithm is potentially hampered from finding a starting node; or, if a
node is found, may have to terminate exploration prematurely because it can't travel sufficiently far from the
data point to explore the surrounding network. If too many data points are not being successfully assigned to the
correct street edges, then this distance should be increased. Conversely, if most of the data points are
satisfactorily assigned, then it may be possible to decrease this threshold. A distance of around 400m may provide
a good starting point.
:::

:::note
The precision of assignment improves on decomposed networks (see
[graphs.nx_decompose](/tools/graphs#nx-decompose)), which offers the additional benefit of a more granular
representation of variations of metrics along street-fronts.
:::

![Example assignment of data to a network](/images/assignment.png) _Example assignment on a non-decomposed graph._

![Example assignment of data to a network](/images/assignment_decomposed.png) _Assignment of data to network nodes becomes more contextually precise on decomposed graphs._

</div>


<div class="function">

## compute_accessibilities


<div class="content">
<span class="name">compute_accessibilities</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">landuse_column_label</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">accessibility_keys</span>
    <span class="pc">:</span>
    <span class="pa"> list[str]</span>
  </div>
  <div class="param">
    <span class="pn">nodes_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 400</span>
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
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 0</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute land-use accessibilities for the specified land-use classification keys over the street network. The landuses are aggregated and computed over the street network relative to the network nodes, with the implication that the measures are generated from the same locations as those used for centrality computations.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing data points. The coordinates of data points should correspond as precisely as possible to the location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the building entrance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">landuse_column_label</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The column label from which to take landuse categories, e.g. a column labelled &quot;landuse_categories&quot; might contain &quot;shop&quot;, &quot;pub&quot;, &quot;school&quot;, etc.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">accessibility_keys</div>
    <div class="type">tuple[str]</div>
  </div>
  <div class="desc">

 Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use schema used for the `landuse_labels` parameter, e.g. &quot;pub&quot;. The calculations will be performed in both weighted `wt` and non_weighted `nw` variants.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of calculations will be written to this `GeoDataFrame`, which is then returned from the function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.</div>
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
    <div class="name">data_id_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional column name for data point keys. This is used for deduplicating points representing a shared source of information. For example, where a single greenspace is represented by many entrances as datapoints, only the nearest entrance (from a respective location) will be considered (during aggregations) when the points share a datapoint identifier.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angular</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations and distances.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">spatial_tolerance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above weights corresponding to the given spatial tolerance and normalises to the new range. For background, see [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).</div>
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

 The input `node_gdf` parameter is returned with additional columns populated with the calcualted metrics. Three columns will be returned for each input landuse class and distance combination; a simple count of reachable locations, a distance weighted count of reachable locations, and the smallest distance to the nearest location.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.</div>
</div>

### Notes

```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
print(nodes_gdf.head())
landuses_gdf = mock.mock_landuse_categorical_data(G)
print(landuses_gdf.head())
nodes_gdf, landuses_gdf = layers.compute_accessibilities(
    data_gdf=landuses_gdf,
    landuse_column_label="categorical_landuses",
    accessibility_keys=["a", "c"],
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[200, 400, 800],
)
print(nodes_gdf.columns)
# weighted form
print(nodes_gdf["cc_c_400_wt"])
# non-weighted form
print(nodes_gdf["cc_c_400_nw"])
# nearest distance to landuse
print(nodes_gdf["cc_c_nearest_max_800"])
```


</div>


<div class="function">

## compute_mixed_uses


<div class="content">
<span class="name">compute_mixed_uses</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">landuse_column_label</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">nodes_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 400</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill_weighted</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_shannon</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = False</span>
  </div>
  <div class="param">
    <span class="pn">compute_gini</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = False</span>
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
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 0</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute landuse metrics. This function wraps the underlying `rust` optimised functions for aggregating and computing various mixed-use. These are computed simultaneously for any required combinations of measures (and distances). By default, hill and hill weighted measures will be computed, by the available flags e.g. `compute_hill` or `compute_shannon` can be used to configure which classes of measures should run.

 See the accompanying paper on `arXiv` for additional information about methods for computing mixed-use measures at the pedestrian scale.

 The data is aggregated and computed over the street network, with the implication that mixed-use and land-use accessibility aggregations are generated from the same locations as for centrality computations, which can therefore be correlated or otherwise compared. The outputs of the calculations are written to the corresponding node indices in the same `node_gdf` `GeoDataFrame` used for centrality methods, and which will display the calculated metrics under correspondingly labelled columns.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing data points. The coordinates of data points should correspond as precisely as possible to the location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the building entrance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">landuse_column_label</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The column label from which to take landuse categories, e.g. a column labelled &quot;landuse_categories&quot; might contain &quot;shop&quot;, &quot;pub&quot;, &quot;school&quot;, etc., landuse categories.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of calculations will be written to this `GeoDataFrame`, which is then returned from the function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_hill</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute Hill diversity. This is the recommended form of diversity index. Computed for q of 0, 1, and 2.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_hill_weighted</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute distance weighted Hill diversity. This is the recommended form of diversity index. Computed for q of 0, 1, and 2.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_shannon</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute shannon entropy. Hill diversity of q=1 is generally preferable.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">compute_gini</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Compute the gini form of diversity index. Hill diversity of q=2 is generally preferable.</div>
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
    <div class="name">data_id_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional column name for data point keys. This is used for deduplicating points representing a shared source of information. For example, where a single greenspace is represented by many entrances as datapoints, only the nearest entrance (from a respective location) will be considered (during aggregations) when the points share a datapoint identifier.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angular</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations and distances.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">spatial_tolerance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above weights corresponding to the given spatial tolerance and normalises to the new range. For background, see [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).</div>
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

 The input `node_gdf` parameter is returned with additional columns populated with the calculated metrics.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_nearest_assign`.</div>
</div>

### Notes

| key | formula | notes |
|-----|:-------:|-------|
| hill | $$q\geq{0},\ q\neq{1} \\ \big(\sum_{i}^{S}p_{i}^q\big)^{1/(1-q)} \\ lim_{q\to1} \\ exp\big(-\sum_{i}^{S}\ p_{i}\ log\ p_{i}\big)$$ | Hill diversity: this is the preferred form of diversity metric because it adheres to the replication principle and uses units of effective species instead of measures of information or uncertainty. The `q` parameter controls the degree of emphasis on the _richness_ of species as opposed to the _balance_ of species. Over-emphasis on balance can be misleading in an urban context, for which reason research finds support for using `q=0`: this reduces to a simple count of distinct land-uses.|
| hill_wt | $$\big[\sum_{i}^{S}d_{i}\big(\frac{p_{i}}{\bar{T}}\big)^{q} \big]^{1/(1-q)} \\ \bar{T} = \sum_{i}^{S}d_{i}p_{i}$$ | This is a distance-weighted variant of Hill Diversity based on the distances from the point of computation to the nearest example of a particular land-use. It therefore gives a locally representative indication of the intensity of mixed-uses. $d_{i}$ is a negative exponential function where $\beta$ controls the strength of the decay. ($\beta$ is provided by the `Network Layer`, see [`rustalgos.distances_from_beta`](/rustalgos#distances-from-betas).)|
| shannon | $$ -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or_information entropy_) is one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is effectively a transformation of Shannon diversity into units of effective species.|
| gini | $$ 1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index. It can behave problematically because it does not adhere to the replication principle and places emphasis on the balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a transformation of Gini-Simpson diversity into units of effective species.|

:::note
`hill_wt` at `q=0` is generally the best choice for granular landuse data, or else `q=1` or
`q=2` for increasingly crude landuse classifications schemas.
:::

 A worked example:
```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
print(nodes_gdf.head())
landuses_gdf = mock.mock_landuse_categorical_data(G)
print(landuses_gdf.head())
nodes_gdf, landuses_gdf = layers.compute_mixed_uses(
    data_gdf=landuses_gdf,
    landuse_column_label="categorical_landuses",
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[200, 400, 800],
)
# the data is written to the GeoDataFrame
print(nodes_gdf.columns)
# access accordingly, e.g. hill diversity at q=0 and 800m
print(nodes_gdf["cc_hill_q0_800_nw"])
```

:::warning
Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that
has been used. Meaningful comparisons from one location to another are only possible where the same schemas have
been applied.
:::

</div>


<div class="function">

## compute_stats


<div class="content">
<span class="name">compute_stats</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">stats_column_label</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">nodes_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">max_netw_assign_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int = 400</span>
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
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 0</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute numerical statistics over the street network. This function wraps the underlying `rust` optimised function for computing statistical measures. The data is aggregated and computed over the street network relative to the network nodes, with the implication that statistical aggregations are generated from the same locations as for centrality computations, which can therefore be correlated or otherwise compared.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing data points. The coordinates of data points should correspond as precisely as possible to the location of the feature in space; or, in the case of buildings, should ideally correspond to the location of the building entrance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">stats_column_label</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The column label corresponding to the column in `data_gdf` from which to take numerical information.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing nodes. Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function. The outputs of calculations will be written to this `GeoDataFrame`, which is then returned from the function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_netw_assign_dist</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum distance to consider when assigning respective data points to the nearest adjacent network nodes.</div>
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
    <div class="name">data_id_col</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional column name for data point keys. This is used for deduplicating points representing a shared source of information. For example, where a single greenspace is represented by many entrances as datapoints, only the nearest entrance (from a respective location) will be considered (during aggregations) when the points share a datapoint identifier.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angular</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to use a simplest-path heuristic in-lieu of a shortest-path heuristic when calculating aggregations and distances.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">spatial_tolerance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Tolerance in metres indicating a spatial buffer for datapoint accuracy. Intended for situations where datapoint locations are not precise. If greater than zero, weighted functions will clip the spatial impedance curve above weights corresponding to the given spatial tolerance and normalises to the new range. For background, see [`rustalgos.clip_weights_curve`](/rustalgos#clip-weights-curve).</div>
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

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `data_gdf` is returned with two additional columns: `nearest_assigned` and `next_neareset_assign`.</div>
</div>

### Notes

 A worked example:

```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
print(nodes_gdf.head())
numerical_gdf = mock.mock_numerical_data(G, num_arrs=3)
print(numerical_gdf.head())
nodes_gdf, numerical_gdf = layers.compute_stats(
    data_gdf=numerical_gdf,
    stats_column_label="mock_numerical_1",
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[200, 400, 800],
)
print(nodes_gdf.columns)
# weighted form
print(nodes_gdf["cc_mock_numerical_1_mean_400_wt"])
# non-weighted form
print(nodes_gdf["cc_mock_numerical_1_mean_400_nw"])
```


:::note
The following stat types will be available for each `stats_key` for each of the
computed distances:
- `max` and `min`
- `sum` and `sum_wt`
- `mean` and `mean_wt`
- `variance` and `variance_wt`
:::

</div>



</section>
