---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# layers


 Compute land-use accessibility, mixed-use diversity, and statistical aggregations over the street network. Data points (land uses, numerical attributes) are assigned to the nearest street edges and then summarised within walking-distance catchments around each node, measured along the actual street network rather than as straight-line distances. Because these summaries are computed at the same node locations used for centrality, you can directly compare how well-connected a location is with how accessible different amenities are from that location. An optional ``decay_fn`` parameter controls how distance affects the weighting; see the [`cityseer.decay`](/api/decay) module for preset helpers. ``decay_fn`` also accepts a ``{label: expression}`` dict to compute several decay variants in a single network traversal, with each label appended to that variant's output column names.

 For practical worked examples, see the [Cityseer Examples](https://benchmark-urbanism.github.io/cityseer-examples/) site, including the [OSM Accessibility](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/osm_accessibility.html), [Mixed Uses](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/gpd_mixed_uses.html), and [Statistical Aggregations](https://benchmark-urbanism.github.io/cityseer-examples/recipes/stats/gpd_stats.html) recipes.


<div class="function">

## build_data_map


<div class="content">
<span class="name">build_data_map</span><div class="signature multiline">
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
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
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
  <span class="pr">DataMap</span>
  <span class="pt">]</span>
</div>
</div>


 Assign a `GeoDataFrame` to a [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). A `NetworkStructure` provides the backbone for the calculation of land-use and statistical aggregations over the network. Points will be assigned to the closest street edge. Polygons will be assigned to the closest `n_nearest_candidates` adjacent street edges.
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
    <div class="type">None</div>
  </div>
  <div class="desc">

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
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

<div class="param-set">
  <div class="def">
    <div class="name">barriers_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing barriers. These barriers will be considered during the assignment of data points to the network.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest street edge candidates to consider when assigning data points to the network. This is used to determine the best assignments based on proximity. Edges are sorted by distance and the closest `n_nearest_candidates` are considered.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">data_map</div>
    <div class="type">rustalgos.data.DataMap</div>
  </div>
  <div class="desc">

 A [`rustalgos.data.DataMap`](/rustalgos#datamap) instance.</div>
</div>


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
    <span class="pa"> int = 100</span>
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
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">barriers_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str | dict[str, str] | None = None</span>
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

 Land-use keys for which to compute accessibilities. The keys should be selected from the same land-use schema used for the `landuse_labels` parameter, e.g. &quot;pub&quot;.</div>
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

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
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

 Distance thresholds in metres for the network traversal. Metrics are computed for each threshold independently. If not provided, the `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`. If not provided, the `distances` parameter must be provided instead.</div>
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
    <div class="name">barriers_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing barriers. These barriers will be considered during the assignment of data points to the network.</div>
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
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest candidates to consider when assigning respective data points to the nearest adjacent streets.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Walking speed in metres per second used to convert `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">decay_fn</div>
    <div class="type">str | dict[str, str]</div>
  </div>
  <div class="desc">

 An optional decay function expression using the variable `p`, where `p` is the normalised distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the accessibility count weighting. Default is `&quot;1&quot;` (flat, no distance weighting). For distance-weighted metrics, provide an expression such as `&quot;exp(-4 * p)&quot;` for exponential decay, or use the `cityseer.decay` module helpers to generate expressions from absolute distance units; see [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of `{label: expression}` to compute several decays in a single network traversal; each label is appended to that variant's output column names (a plain string or `None` adds no suffix).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">nodes_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 The input `node_gdf` parameter is returned with additional columns populated with the calculated metrics. Two columns will be returned for each input landuse class and distance combination; a count of reachable locations, and the smallest distance to the nearest location.</div>
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

```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
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
# the default emits an unweighted (_nw) and a weighted (_wt) column;
# pass a single decay_fn (e.g. "1") to compute just one and save time
print(nodes_gdf["cc_c_400_nw"])
# nearest distance to landuse (decay-independent: one column)
print(nodes_gdf["cc_c_nearest_max_800"])
```

 For worked examples with real-world data, see the [OSM Accessibility](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/osm_accessibility.html) recipe.

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
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill</span>
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
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">barriers_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str | dict[str, str] | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pr">GeoDataFrame</span>
  <span class="pt">]</span>
</div>
</div>


 Compute landuse metrics. This function wraps the underlying `rust` optimised functions for aggregating and computing various mixed-use. These are computed simultaneously for any required combinations of measures (and distances). By default, hill measures will be computed, but the available flags e.g. `compute_hill` or `compute_shannon` can be used to configure which classes of measures should run.

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

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
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

 Distance thresholds in metres for the network traversal. Metrics are computed for each threshold independently. If not provided, the `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`. If not provided, the `distances` parameter must be provided instead.</div>
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
    <div class="name">barriers_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing barriers. These barriers will be considered during the assignment of data points to the network.</div>
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
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest candidates to consider when assigning respective data points to the nearest adjacent streets.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Walking speed in metres per second used to convert `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">decay_fn</div>
    <div class="type">str | dict[str, str]</div>
  </div>
  <div class="desc">

 An optional decay function expression using the variable `p`, where `p` is the normalised distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the Hill diversity weighting. Default is `&quot;1&quot;` (flat, no distance weighting). For distance-weighted metrics, provide an expression such as `&quot;exp(-4 * p)&quot;` for exponential decay, or use the `cityseer.decay` module helpers to generate expressions from absolute distance units; see [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of `{label: expression}` to compute several decays in a single network traversal; each label is appended to that variant's output column names (a plain string or `None` adds no suffix).</div>
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
| shannon | $$ -\sum_{i}^{S}\ p_{i}\ log\ p_{i}$$ | Shannon diversity (or_information entropy_) is one of the classic diversity indices. Note that it is preferable to use Hill Diversity with `q=1`, which is effectively a transformation of Shannon diversity into units of effective species.|
| gini | $$ 1 - \sum_{i}^{S} p_{i}^2$$ | Gini-Simpson is another classic diversity index. It can behave problematically because it does not adhere to the replication principle and places emphasis on the balance of species, which can be counter-productive for purposes of measuring mixed-uses. Note that where an emphasis on balance is desired, it is preferable to use Hill Diversity with `q=2`, which is effectively a transformation of Gini-Simpson diversity into units of effective species.|

:::note
`hill` at `q=0` is generally the best choice for granular landuse data, or else `q=1` or
`q=2` for increasingly crude landuse classifications schemas.
:::

 A worked example:
```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
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
# the default emits _nw and _wt; pass a single decay_fn to compute just one and save time
print(nodes_gdf["cc_hill_q0_800_nw"])
```

:::warning
Be cognisant that mixed-use and land-use accessibility measures are sensitive to the classification schema that
has been used. Meaningful comparisons from one location to another are only possible where the same schemas have
been applied.
:::

 For a worked example, see the [Mixed Uses](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/gpd_mixed_uses.html) recipe.

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
    <span class="pn">stats_column_labels</span>
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
    <span class="pa"> int = 100</span>
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
    <span class="pn">data_id_col</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">barriers_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
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
    <span class="pn">decay_fn</span>
    <span class="pc">:</span>
    <span class="pa"> str | dict[str, str] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">measures</span>
    <span class="pc">:</span>
    <span class="pa"> list[str] | None = None</span>
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
    <div class="name">stats_column_labels</div>
    <div class="type">list[str]</div>
  </div>
  <div class="desc">

 The column labels corresponding to the columns in `data_gdf` from which to take numerical information.</div>
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

 A [`rustalgos.graph.NetworkStructure`](/rustalgos/rustalgos#networkstructure). Best generated with the [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx) function.</div>
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

 Distance thresholds in metres for the network traversal. Metrics are computed for each threshold independently. If not provided, the `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking time thresholds in minutes. Converted to distance thresholds using `speed_m_s`. If not provided, the `distances` parameter must be provided instead.</div>
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
    <div class="name">barriers_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A [`GeoDataFrame`](https://geopandas.org/en/stable/docs/user_guide/data_structures.html#geodataframe) representing barriers. These barriers will be considered during the assignment of data points to the network.</div>
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
    <div class="name">n_nearest_candidates</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of nearest candidates to consider when assigning respective data points to the nearest adjacent streets.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Walking speed in metres per second used to convert `minutes` to distance thresholds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">decay_fn</div>
    <div class="type">str | dict[str, str]</div>
  </div>
  <div class="desc">

 An optional decay function expression using the variable `p`, where `p` is the normalised distance from 0 (source) to 1 (cutoff threshold). Controls how distance affects the statistical weighting. Default is `&quot;1&quot;` (flat, no distance weighting). For distance-weighted metrics, provide an expression such as `&quot;exp(-4 * p)&quot;` for exponential decay, or use the `cityseer.decay` module helpers. Values are clamped to [0, 1]. Supported functions include `exp`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `min`, `max`, and the `^` operator. When multiple distances are specified, `p` is normalised independently per threshold. See [`cityseer.decay`](/api/decay) for details and examples. Pass a dict of `{label: expression}` to compute several decays in a single network traversal; each label is appended to that variant's output column names (a plain string or `None` adds no suffix).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">measures</div>
    <div class="type">list[str]</div>
  </div>
  <div class="desc">

 An optional subset of statistical measures to compute, chosen from `&quot;sum&quot;`, `&quot;mean&quot;`, `&quot;count&quot;`, `&quot;var&quot;`, `&quot;median&quot;`, `&quot;mad&quot;`, `&quot;max&quot;`, and `&quot;min&quot;`. Defaults to `None`, which computes all of them. Restricting the set keeps the output `GeoDataFrame` smaller and skips the weighted median / MAD sort when neither `&quot;median&quot;` nor `&quot;mad&quot;` is requested.</div>
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

 Default exponential decay at multiple scales:

```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
print(nodes_gdf.head())
numerical_gdf = mock.mock_numerical_data(G, num_arrs=3)
print(numerical_gdf.head())
nodes_gdf, numerical_gdf = layers.compute_stats(
    data_gdf=numerical_gdf,
    stats_column_labels=["mock_numerical_1"],
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[200, 400, 800],
)
print(nodes_gdf.columns)
# mean at 400m; the default emits _nw and _wt. Pass a single decay_fn for just one,
# and measures=[...] to compute only the statistics you need — both save time
print(nodes_gdf["cc_mock_numerical_1_mean_400_nw"])
```

 Custom decay using the `p` variable directly (Gaussian peaking at 400m within a 1200m cutoff):

```python
nodes_gdf, numerical_gdf = layers.compute_stats(
    data_gdf=numerical_gdf,
    stats_column_labels=["mock_numerical_1"],
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[1200],
    decay_fn="exp(-(p - 0.333)^2 / (2 * 0.125^2))",  # Gaussian peaking at 400m
)
```

 Using the `cityseer.decay` helper module for the same Gaussian curve:

```python
from cityseer import decay

nodes_gdf, numerical_gdf = layers.compute_stats(
    data_gdf=numerical_gdf,
    stats_column_labels=["mock_numerical_1"],
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)
```

 Flat (unweighted) metrics:

```python
nodes_gdf, numerical_gdf = layers.compute_stats(
    data_gdf=numerical_gdf,
    stats_column_labels=["mock_numerical_1"],
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[800],
    decay_fn="1",
)
```


:::note
The following stat types will be available for each `stats_key` for each of the
computed distances:
- `max` and `min`
- `sum`
- `mean`
- `count`
- `median`
- `variance`
- `mad` (median absolute deviation)

The decay function (default exponential, or custom via `decay_fn`) controls how
distance affects the weighting. Use `decay_fn="1"` for flat (unweighted) metrics.
:::

 For a worked example, see the [Statistical Aggregations](https://benchmark-urbanism.github.io/cityseer-examples/recipes/stats/gpd_stats.html) recipe.

</div>



</section>
