---
layout: '@src/layouts/PageLayout.astro'
---

# How do I...?

Short answers to common questions, each pointing to the part of the API that handles it. In every case the measure is computed over the street network: distances follow the streets and aggregations run over network catchments, not straight-line buffers. The examples use the [CityNetwork](/api/network) class; every answer also works through the functional API, and most through the [QGIS plugin](/plugin) without code.

## Land use and accessibility

### Count the schools reachable within 800m of each street?

Use accessibility. Pass a GeoDataFrame of points or polygons with a category column and the categories you need:

```python
cn.compute_accessibilities(
    data_gdf,
    landuse_column_label="type",
    accessibility_keys=["school"],
    distances=[800],
)
```

Each street segment receives a count column (`cc_school_800`) and a `cc_school_nearest_max_800` distance to the closest feature. Pass a `decay_fn` to weight the count by distance. In QGIS, use the [Accessibility](/plugin/accessibility) algorithm.

### Find the distance to the nearest supermarket?

The same accessibility call computes it: the `cc_<category>_nearest_max_<distance>` column holds the network distance from each street to the nearest feature of that category. See [compute_accessibilities](/api/network).

### Measure how mixed the land uses are around each street?

Use mixed uses:

```python
cn.compute_mixed_uses(data_gdf, landuse_column_label="type", distances=[800])
```

Hill diversity at q=0 (`cc_hill_q0_800`) counts the distinct land uses reachable within the threshold. See the [land-use guide](/guide/land-use).

## Statistics

### Aggregate building areas around each street?

Use statistics. Give each building polygon an area column, then aggregate:

```python
gdf["area"] = gdf.geometry.area
cn.compute_stats(gdf, stats_column_labels=["area"], distances=[400, 800])
```

The `cc_area_sum_400` column is the total built area reachable within 400m of each street. In QGIS, use the [Statistics](/plugin/statistics) algorithm.

### Get the average income around each street?

Use statistics with the mean: attach the income value to census points or polygon centroids and read `cc_income_mean_<distance>`. Pass `measures=["mean"]` to compute only the statistics you need. See [compute_stats](/api/network).

### Aggregate several columns at once?

Pass them together:

```python
cn.compute_stats(gdf, stats_column_labels=["income", "age", "density"], distances=[800])
```

All three are computed in a single network pass, which is faster than three separate runs. The QGIS Statistics dialog also accepts multiple numeric fields.

## Centrality

### Find the streets likely to carry the most movement?

Betweenness centrality:

```python
cn.centrality_shortest(distances=[800, 4000])
```

This writes `cc_betweenness_<distance>` columns counting the shortest paths passing through each street. Shorter thresholds correspond to local pedestrian movement and longer thresholds to larger-scale through-movement. See the [centrality guide](/guide/centrality).

### Find the streets most central to their local area?

Closeness centrality comes from the same call: `cc_harmonic_<distance>` measures how easily each street reaches its surroundings. Closeness identifies local centres, whereas betweenness identifies through-routes. For other closeness forms, pass a `closeness` dict, for example `{"density": "1", "decay": "exp(-4 * p)"}`.

### Decide between metric and angular analysis?

Metric (shortest-path) analysis models distance-minimising movement. Angular (simplest-path) analysis models route choice by minimising turns, which suits legibility and wayfinding:

```python
cn.centrality_simplest(distances=[800])
```

See the [centrality guide](/guide/centrality) for how to choose.

## Flows

### Map the major routes between homes and shops?

Use demand betweenness. Trips are allocated by a spatial interaction model and routed along the network:

```python
cn.betweenness_demand(
    origins_gdf=homes,
    destinations_gdf=shops,
    origin_weight_col="population",
    destination_weight_col="floorspace",
    distances=[1600],
)
```

The `cc_demand_1600` value on a segment is the modelled trip volume passing through it. See the [flows guide](/guide/flows) or the QGIS [Demand Betweenness](/plugin/demand) algorithm.

### Route an observed commuting matrix onto streets?

If you have explicit origin-destination counts (for example census travel-to-work data), build a matrix and route it:

```python
od_matrix = cn.build_od_matrix(od_df, zones_gdf)
cn.betweenness_od(od_matrix, distances=[1600])
```

See the [flows guide](/guide/flows).

### See how flows would change if a street were added or removed?

Run the analysis twice, once on the current network and once on the edited network, then subtract the result columns. The difference shows which streets gain or lose flow under the change. Construction is fast and stateless, so build a fresh `CityNetwork` from the edited geometries for the second run.

## Networks and study area

### Get a network from OpenStreetMap?

`CityNetwork.from_osm(poly)` downloads, cleans, and builds a network for a boundary polygon in one step:

```python
cn = CityNetwork.from_osm(poly)
```

For finer control over the download and cleaning stages, see [tools.io](/tools/io).

### Clean a messy network?

Automated cleaning handles most OSM input. For configurable control over consolidation and simplification, see the [network cleaning guide](/guide/cleaning) and [tools.graphs](/tools/graphs).

### Avoid edge effects at the boundary of my study area?

Build the network over a buffered extent so catchments near the boundary stay complete, then restrict analysis sources to the study area:

```python
cn.set_boundary(polygon)
```

Streets inside the boundary are used as analysis sources, and the buffered surround completes their catchments. See [fundamentals](/guide/fundamentals#edge-rolloff) on edge rolloff.

## Distances and performance

### Use walking time instead of distance?

Every metric accepts `minutes` in place of `distances`, converted with a configurable walking speed (`speed_m_s`, default 1.33):

```python
cn.centrality_shortest(minutes=[5, 10, 20])
```

Output columns are keyed by the converted metre distances, so results stay comparable across runs. The QGIS dialogs expose minutes and walking speed under the advanced parameters.

### Speed up long-distance computations?

Pass `sample=True` to the centrality methods (experimental):

```python
cn.centrality_shortest(distances=[4000], sample=True)
```

A pilot poll measures each node's reach and samples only where that is predicted to be faster; inverse-probability weighting keeps estimates unbiased, and rankings are preserved at the default `epsilon=0.05`. See the [sampling module](/metrics/sampling).

## Output

### Export results to QGIS or a figure?

`cn.to_geopandas()` returns the street segments with all computed columns:

```python
result_gdf = cn.to_geopandas()
result_gdf.to_file("results.gpkg")
```

Write them to a GeoPackage for styling in QGIS, or plot with GeoPandas and matplotlib. The [From Results to Maps](/examples/networks/results-to-maps) recipe covers both.

### Do all of this without writing Python?

The [QGIS plugin](/plugin) exposes centrality, demand betweenness, accessibility, mixed uses, and statistics as Processing algorithms over ordinary QGIS layers.
