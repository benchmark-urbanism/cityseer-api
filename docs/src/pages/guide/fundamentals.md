---
layout: '@src/layouts/PageLayout.astro'
---

# Fundamentals

This guide walks through the core concepts and features of `cityseer`. It is aimed at researchers, urban planners, and developers who want to compute street-network centrality, land-use accessibility, or statistical aggregations at the pedestrian scale. Familiarity with Python and `geopandas` is assumed; for a gentler introduction, start with the [Python 101](/start) lessons. For the underlying research methods, see the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827).

For practical, end-to-end worked examples with real-world data, see the [examples](/examples) section.

This page covers how `cityseer` frames analysis and how to drive the library; [Networks](/guide/networks), [Centrality](/guide/centrality), and [Land-Use](/guide/land-use) cover the respective domains in depth.

## Localised analysis

`cityseer` computes metrics locally rather than globally. For each node in the network, the surrounding subgraph is isolated up to a specified distance threshold (for example, all streets within 800m walking distance), metrics are computed within that local subgraph, and the process repeats for every node. This avoids the boundary effects inherent in global network measures, where nodes near the edge of the study area receive artificially low scores.

Localised metrics are directly comparable across different locations and cities because the analysis window is defined by the distance threshold, not by the extent of the dataset. Shorter distance thresholds capture local neighbourhood structure while longer thresholds reveal city-wide patterns.

Nodes at the periphery of a study area can be marked as "dead" (non-live) using a boundary polygon. Dead nodes participate fully in network traversal but their own values are not reported: results are only written for live nodes. For closeness, dead nodes are skipped as sources in exact mode (a pure cost saving); for betweenness, every node serves as a source so that all routes passing through the study area, by design including routes that both start and end among dead nodes, credit the live nodes they traverse. This prevents artificially depressed values at the edges of the study area without discarding network connectivity. See the [Live Nodes](/examples/recipes/live-nodes) recipe for a worked example.

## Shortest-path and simplest-path heuristics

`cityseer` supports two routing heuristics:

- **Shortest path**: Routes minimise cumulative physical distance along the network. A 400m route is preferred over a 600m route, regardless of how many turns are involved.
- **Simplest path (angular)**: Routes minimise cumulative angular change, that is, the total amount of turning at each junction. A pedestrian following a simplest path prefers to continue straight ahead rather than turning, even if a shorter alternative exists.

When to use each:

- **Shortest path** is appropriate for accessibility analysis (how far is the nearest park?), walkability scoring, and situations where physical distance is the primary concern.
- **Simplest path** is appropriate for predicting pedestrian flows and commercial activity, because research shows that people tend to follow cognitively simple routes. It is also the basis for angular integration and choice measures used in space syntax research.

Both heuristics can be computed from a single `CityNetwork` instance at any combination of distance thresholds. When in doubt, compute both.

## Converting the network

`cityseer` converts the network into a `rust` data structure prior to computing derivative metrics. When using the [`CityNetwork`](/api/network) class described below, this conversion happens automatically during construction, including the dual graph representation required for angular analysis, and no manual steps are needed.

Note that all distances used when computing metrics are network distances (shortest paths along the network), not Euclidean distances.

:::note
For advanced users: the lower-level API exposes the conversion directly. [`network_structure_from_nx`](/tools/io#network-structure-from-nx) produces nodes and edges `GeoDataFrames` plus the internal `NetworkStructure`, and [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) converts a primal graph to the dual representation required for angular analysis. The [network preparation recipes](/examples/networks) teach this route step by step, with each recipe handing the prepared graph back to `CityNetwork.from_nx`.
:::

## Distance thresholds

Most `cityseer` functions accept a `distances` parameter specifying the network distance thresholds (in metres) at which to compute metrics. You can compute multiple thresholds simultaneously:

| Distance | Walking time | Typical use                   |
| -------- | ------------ | ----------------------------- |
| 200m     | ~2.5 min     | Immediate neighbourhood       |
| 400m     | ~5 min       | Local walkability             |
| 800m     | ~10 min      | Neighbourhood-scale access    |
| 1600m    | ~20 min      | District-scale patterns       |
| 5000m+   | ~60 min+     | City-wide structural analysis |

Computing metrics at multiple distances simultaneously reveals how urban structure varies across scales. For example, a street may be highly central at 800m (locally important) but not at 5000m (not a major through-route). The `minutes` parameter can also be used as an alternative to specifying distances directly.

## Edge rolloff

When calculating network or layer metrics, the network must be buffered by a distance equal to the maximum distance threshold used by the algorithms. This prevents distorted results arising from edge rolloff effects. For example, if running analysis at distances of 500, 1000, and 2000m, then the network must be buffered by at least 2000m. Data layers should cover these buffered extents as well.

The `live` node attribute controls this. Nodes within the original (non-buffered) extents are set to `live=True`, while nodes in the surrounding buffer are set to `live=False`. The shortest-path algorithms have access to both, preventing edge rolloff, but derivative metrics are only computed for `live=True` nodes. If boundary rolloff is not a concern, the default behaviour sets all nodes to `live=True`.

The [live nodes notebook](/examples/recipes/live-nodes) shows how to demarcate the study area with [`CityNetwork.set_boundary`](/api/network#set_boundary), which sets the node status based on whether each node falls inside the original boundary.

## CityNetwork API

The [`CityNetwork`](/api/network) class lets you build a network, compute centrality and land-use metrics, and export results without managing intermediate data structures. It builds dual graphs directly from input geometries and handles graph cleaning automatically.

## Constructors

| Constructor | Input format | Example |
| --- | --- | --- |
| [`from_geopandas`](/api/network#from_geopandas) | GeoDataFrame of LineStrings | [Network from Streets](/examples/networks/network-from-streets) |
| [`from_nx`](/api/network#from_nx) | NetworkX MultiGraph or MultiDiGraph | [OSMnx to Cityseer](/examples/networks/osmnx-to-cityseer) |
| [`from_osm`](/api/network#from_osm) | Shapely polygon (downloads from OSM) | [Create from BBox](/examples/networks/create-from-bbox) |
| [`from_wkts`](/api/network#from_wkts) | Dictionary of WKT strings or Shapely geometries | -- |
| [`load`](/api/network#load) | Previously saved parquet/pickle pair | [Save to File](/examples/networks/save-to-file) |

## Method chaining

Most methods return `self`, enabling fluent method chaining:

```python
cn = (
    CityNetwork.from_geopandas(edges_gdf, crs=32632)
    .set_boundary(boundary_polygon)
    .centrality_shortest(distances=[400, 800, 1600])
    .centrality_simplest(distances=[400, 800, 1600])
)
```

## Retrieving results

Because `CityNetwork` uses a dual graph internally, the `nodes_gdf` property exposes each street segment as a row with a Point geometry at the segment midpoint. To obtain results with the original LineString geometries (suitable for mapping and export), call [`to_geopandas()`](/api/network#to_geopandas):

```python
# Original LineString geometries with all computed columns
result_gdf = cn.to_geopandas()
result_gdf.to_file("results.gpkg")
```

## Automatic graph cleaning

Input geometries are automatically cleaned during construction: short self-loops, near-duplicate edges, and short danglers are removed. The [`feature_status`](/api/network#citynetwork) property returns a Series with values such as `"active"`, `"short_self_loop"`, `"duplicate"`, `"short_dangler"`, or `"invalid_geometry"`, indicating what happened to each input feature. When using the lower-level API, the [`tools.graphs`](/tools/graphs) module provides manual graph cleaning functions; see the [Network Simplification](/examples/networks/network-simplification) example.

## Column Naming Conventions

All computed metrics are written to columns on the `nodes_gdf` GeoDataFrame following a consistent pattern:

```text
cc_{metric}_{distance}            -- shortest-path metric
cc_{metric}_{distance}_ang        -- simplest-path (angular) metric
cc_{metric}_{label}_{distance}    -- land-use metric under a named decay label
```

The `cc_` prefix identifies columns generated by `cityseer`. The optional `{label}` segment appears only when a land-use method is called with a `{label: expression}` decay dict (see [Multiple decays in one traversal](/guide/land-use#multiple-decays-in-one-traversal)); a single decay expression or `None` produces no label segment. Examples:

```text
cc_harmonic_800         -- harmonic closeness at 800m
cc_betweenness_800_ang  -- angular betweenness at 800m
cc_hill_q0_400          -- Hill diversity q=0 at 400m
cc_retail_200               -- accessibility count for "retail" at 200m
cc_retail_nearest_max_800   -- nearest distance to "retail" at max threshold
cc_price_mean_1200          -- mean of "price" column at 1200m
cc_retail_grav_800          -- "retail" count at 800m under the "grav" decay label
cc_price_mean_grav_1200     -- mean of "price" at 1200m under the "grav" decay label
```

When analysing results programmatically, it is often useful to select subsets of the computed columns by pattern:

```python
# All cityseer columns
cc_cols = [c for c in cn.nodes_gdf.columns if c.startswith("cc_")]

# All columns for a specific distance
cols_800 = [c for c in cn.nodes_gdf.columns if c.endswith("_800")]

# All betweenness columns across distances
bt_cols = [c for c in cn.nodes_gdf.columns if "betweenness" in c]
```

## Additional Modules

### Visibility

The [`metrics.visibility`](/metrics/visibility) module computes line-of-sight visibility from street-level observer locations, accounting for building obstructions. See the [Visibility from OSM](/examples/visibility/vis-osm) example.

### Street continuity

The [`metrics.observe`](/metrics/observe) module identifies coherent street sequences based on name, route number, or highway classification. See the [Street Continuity from OSM](/examples/continuity/continuity-osm) example.

### Public transport (GTFS)

The [`add_gtfs`](/api/network#add_gtfs) method integrates public transport stops and routes from GTFS data, enabling centrality and accessibility analyses that account for transit connections. See the [Centrality with Metro](/examples/centrality/centrality-metro) and [Accessibility with Metro](/examples/accessibility/accessibility-metro) examples.
