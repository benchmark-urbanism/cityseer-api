---
layout: '@src/layouts/PageLayout.astro'
---

# Fundamentals

This guide covers the core concepts and features of `cityseer`. It is aimed at researchers, urban planners, and developers who want to compute street-network centrality, land-use accessibility, or statistical aggregations at the pedestrian scale. Familiarity with Python and `geopandas` is assumed; for a gentler introduction, start with the [Python 101](/start) lessons. For the underlying research methods, see the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827).

:::note
**Working with an LLM?** If you use an AI coding assistant, point it at [`llms.txt`](/llms.txt), a compact machine-readable index of this documentation, and at the [GitHub repository](https://github.com/benchmark-urbanism/cityseer-api), so its answers reflect the current API.
:::

For practical, end-to-end worked examples with real-world data, see the [examples](/examples) section.

This page covers how `cityseer` frames analysis and how to drive the library; [Networks](/guide/networks), [Centrality](/guide/centrality), and [Land-Use](/guide/land-use) cover the respective domains in depth.

## Localised analysis

`cityseer` computes network metrics locally rather than globally. For each node in the network, it isolates the surrounding subgraph out to a chosen distance radius, for example all streets within 800m walking distance. It computes the metrics within that local subgraph, then repeats the process for every node. These distances are network distances, measured along the streets, not straight-line (Euclidean) distances.

Because every location is measured to the same radius, locations can be compared like for like. In contrast, a global measure depends on the size of the whole network, so its values are not comparable in this way. Another advantage of localised methods is that the radius can be set by the analyst to any value (50m, 800m, and 20km are examples), so the same network can be analysed at a fine local scale or a broad structural one. A global measure, in contrast, is fixed to a single scale, the whole network.

A global measure is inherently subject to edge rolloff, the boundary effect at the limits of the data. Localised analysis avoids this: buffer the area of interest by the radius in use, so that every node within it has a full radius of network around it and is not affected by the boundary (see [Edge rolloff](#edge-rolloff)).

## Shortest and simplest heuristics

`cityseer` supports two routing heuristics:

- **Shortest path**: Routes minimise cumulative physical distance along the network. A 400m route is preferred over a 600m route, regardless of how many turns are involved.
- **Simplest path (angular)**: Routes minimise cumulative angular change, the total amount of turning along the route, counted both at junctions and along the curve of each street segment. A pedestrian following a simplest path prefers a route with fewer turns, even if a shorter alternative exists.

The choice is not either/or, and computing both is common. Which one fits depends on the network's cleanliness, the kind of movement being modelled, and how the results are read.

**Shortest path** depends only on distances, so it can be more robust on a messy network: unconsolidated junctions, spurious nodes, and roundabouts drawn as rings distort the turn counts that simplest path relies on, while distorting metric distance less. Over-represented geometry (a dual carriageway drawn as two parallel lines, say) still inflates distances, so it is not immune; even so, it tends to hold up better than simplest path in these cases. Use it for accessibility (how far is the nearest park?), walkability, and anywhere physical distance is the main concern; it is often the safer default when the network has not been cleaned to a high standard.

**Simplest path** tends to have an advantage on a clean, well-consolidated network when modelling pedestrian flows, commercial activity, or land-use behaviour, since people tend to follow cognitively simple routes. In space-syntax terms, its angular closeness corresponds to integration and its angular betweenness to choice (closeness and betweenness are covered under [Centrality](/guide/centrality#closeness-and-betweenness)).

Both heuristics can be computed from a single `CityNetwork` instance at any combination of distance thresholds, so computing both and comparing is often the most informative approach.

## Creating the network

When you create a network with the [`CityNetwork`](/api/network) class described below, `cityseer` internalises it in a `rust` data structure, so the analysis scales efficiently to large cities and regions.

Internally it uses a dual representation rather than a primal one. In the **primal** graph, junctions are nodes and streets are edges; the **dual** inverts this, so each street segment becomes a node at its midpoint and each junction becomes a connection between these midpoints. The geometry linking the midpoints stays faithful to the distances and angular changes along the network. The reason for the dual is that results are easier to reason about on the street segments themselves than on junctions, which dilute the influence of the streets entering them and which are harder to visualise intuitively. See [Networks](/guide/networks) for the fuller treatment.

:::note
For advanced users: the lower-level API builds these structures directly. [`network_structure_from_nx`](/tools/io#network-structure-from-nx) produces nodes and edges `GeoDataFrames` plus the internal `NetworkStructure`, and [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) produces the dual representation. The [network preparation recipes](/examples/networks) teach this route step by step, with each recipe handing the prepared graph back to `CityNetwork.from_nx`.
:::

## Distance thresholds

Most `cityseer` functions accept a `distances` parameter: the network distance thresholds, in metres, at which to compute metrics, with several computed at once. Instead of distance, the same thresholds can be expressed as walking times through the `minutes` parameter, converted with a walking speed set by the optional `speed_m_s` parameter (default 1.33 m/s). The table below relates the two:

| Distance | Walking time | Typical use                   |
| -------- | ------------ | ----------------------------- |
| 200m     | ~2.5 min     | Immediate neighbourhood       |
| 400m     | ~5 min       | Local walkability             |
| 800m     | ~10 min      | Neighbourhood-scale access    |
| 1600m    | ~20 min      | District-scale patterns       |
| 5000m+   | ~60 min+     | City-wide structural analysis |

Computing metrics at multiple distances reveals how urban structure varies across scales. For example, a street may be highly central at 800m (locally important) but not at 5000m (not a major through-route).

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

Input geometries are cleaned automatically during construction, in this order: short self-loops under 1m are removed, chains of segments meeting at degree-2 points are welded into single segments (`remove_fillers`, on by default), dead-end stubs up to 10m are removed (`remove_danglers=10.0`), and near-duplicate parallel edges within 2m are merged (`merge_parallel_dist=2.0`). Each step can be disabled through its constructor parameter. Directed networks skip filler welding and parallel merging, which do not preserve one-way semantics.

[`from_osm`](/api/network#from_osm) additionally runs an OSM-tuned simplification pipeline by default (`simplify=True`): junction consolidation, parallel carriageway merging by midline, and ironing. Pass `simplify=False` to skip it.

The [`feature_status`](/api/network#citynetwork) property returns a Series recording what happened to each input feature, with values such as `"active"`, `"merged"`, `"short_self_loop"`, `"short_dangler"`, `"duplicate"`, or `"invalid_geometry"`. For more extensive or custom simplification of non-OSM networks, the lower-level [`tools.graphs`](/tools/graphs) module provides manual cleaning functions; see the [Network Simplification](/examples/networks/network-simplification) example. The automated cleaning method and its trade-offs for centrality analysis are evaluated by [Abdeldayem et al. (2026)](https://journals.sagepub.com/doi/10.1177/23998083261433647); the [Network cleaning](/guide/cleaning) guide covers the pipeline and its parameters in full.

## Column naming conventions

All computed metrics are written to columns on the `nodes_gdf` GeoDataFrame following a consistent pattern:

```text
cc_{metric}_{distance}            -- shortest-path metric
cc_{metric}_{distance}_ang        -- simplest-path (angular) metric
cc_{metric}_{label}_{distance}    -- land-use / data metric under a named decay label
```

The `cc_` prefix identifies columns generated by `cityseer`. For the land-use and data methods (accessibility, mixed-use, statistics), the `CityNetwork` methods default to a single unweighted column per metric (`decay_fn="1"`). Pass a `decay_fn` expression to weight by distance, or a `{label: expression}` dict to embed each `{label}` in the metric name and compute several weightings in one pass (see [Multiple decays in one traversal](/guide/land-use#multiple-decays-in-one-traversal)). The lower-level `layers` functions instead default to two columns per metric, an unweighted `_nw` and a decay-weighted `_wt`. Examples:

```text
cc_harmonic_800         -- harmonic closeness at 800m
cc_betweenness_800_ang  -- angular betweenness at 800m
cc_hill_q0_400          -- Hill diversity q=0 at 400m
cc_retail_200           -- "retail" count at 200m
cc_retail_nearest_max_800   -- nearest distance to "retail" at max threshold
cc_price_mean_1200      -- mean of "price" at 1200m
cc_retail_grav_800      -- "retail" count at 800m under the "grav" decay label
cc_price_mean_grav_1200 -- mean of "price" at 1200m under the "grav" decay label
```

When analysing results programmatically, you can select subsets of the computed columns by pattern:

```python
# All cityseer columns
cc_cols = [c for c in cn.nodes_gdf.columns if c.startswith("cc_")]

# All columns for a specific distance
cols_800 = [c for c in cn.nodes_gdf.columns if c.endswith("_800")]

# All betweenness columns across distances
bt_cols = [c for c in cn.nodes_gdf.columns if "betweenness" in c]
```

## Additional modules

:::note
These modules are experimental. Feedback and bug reports are welcome on the [issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues) and in [discussions](https://github.com/benchmark-urbanism/cityseer-api/discussions).
:::

### Visibility

The [`metrics.visibility`](/metrics/visibility) module computes line-of-sight visibility from street-level observer locations, accounting for building obstructions. See the [Visibility from OSM](/examples/visibility/vis-osm) example.

### Street continuity

The [`metrics.observe`](/metrics/observe) module identifies coherent street sequences based on name, route number, or highway classification. See the [Street Continuity from OSM](/examples/continuity/continuity-osm) example.

### Public transport (GTFS)

The [`add_gtfs`](/api/network#add_gtfs) method integrates public transport stops and routes from GTFS data, enabling centrality and accessibility analyses that account for transit connections. See the [Centrality with Metro](/examples/centrality/centrality-metro) and [Accessibility with Metro](/examples/accessibility/accessibility-metro) examples.
