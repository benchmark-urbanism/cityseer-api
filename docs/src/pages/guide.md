---
layout: '@src/layouts/PageLayout.astro'
---

# Guide

This guide walks through the core concepts and features of `cityseer`. It is aimed at researchers, urban planners, and developers who want to compute street-network centrality, land-use accessibility, or statistical aggregations at the pedestrian scale. Familiarity with Python and `geopandas` is assumed; for a gentler introduction, start with the [Python 101](https://benchmark-urbanism.github.io/cityseer-examples/class/index.html) tutorials on the examples site. For the underlying research methods, see the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827).

For practical, end-to-end worked examples with real-world data (Madrid case study), see the [Cityseer Examples](https://benchmark-urbanism.github.io/cityseer-examples/) site.

## Installation

```bash
pip install --upgrade cityseer
```

`cityseer` requires Python 3.10 or later. The underlying algorithms are implemented in Rust and distributed as pre-compiled wheels, so no Rust toolchain is needed. A projected coordinate reference system (CRS) is required for all analyses; coordinates must be in metres, not degrees. Use [epsg.io](https://epsg.io/) to find the appropriate EPSG code for your study area (e.g. `EPSG:32630` for London, `EPSG:32632` for central Europe, `EPSG:2154` for France).

:::note
For users who prefer a GUI workflow, a [QGIS plugin](/plugin) is available for computing centrality metrics without writing Python code.
:::

## Quick Start

The [Quickstart](https://benchmark-urbanism.github.io/cityseer-examples/recipes/quickstart.html) notebook on the examples site provides a full walkthrough. The following minimal example downloads a street network from OpenStreetMap, computes centrality, and plots the result:

```python
from shapely.geometry import box
from cityseer.network import CityNetwork

# Define a bounding box in WGS84 (lon, lat)
polygon = box(-0.13, 51.51, -0.12, 51.52)

# Build the network (projected to UTM zone 30N)
cn = CityNetwork.from_osm(polygon, to_crs_code=32630)

# Compute shortest-path centrality at 400m and 800m walking distance
cn.centrality_shortest(distances=[400, 800])

# Export as a GeoDataFrame with original street geometries
result_gdf = cn.to_geopandas()

# Visualise betweenness at 800m (in a Jupyter notebook; for scripts, call plt.show())
result_gdf.plot(column="cc_betweenness_800", cmap="inferno", linewidth=0.5)
```

Distance thresholds can also be specified as walking times using the `minutes` parameter:

```python
cn.centrality_shortest(minutes=[5, 10, 20])  # assumes default walking speed of 1.33 m/s
```

### Saving and loading

Networks can be saved to disk and restored later, preserving all computed metrics:

```python
cn.save("my_network")
# Creates: my_network.nodes.parquet, my_network.state.pkl

cn_restored = CityNetwork.load("my_network")
```

:::note
The lower-level API (`cityseer.tools`, `cityseer.metrics`) offers step-by-step control over graph cleaning, network construction, and metric computation. Most users should start with `CityNetwork`; the lower-level API is useful when integrating cityseer into an existing NetworkX pipeline or when fine-grained control over processing steps is needed. See the [`tools`](/tools/io) and [`metrics`](/metrics/networks) module references for details.
:::

## Core Concepts

### Localised analysis

`cityseer` uses a moving-window methodology for network analysis. For each node, the surrounding subgraph is isolated up to a specified distance threshold, metrics are computed for that local context, and the process repeats for every other node. This avoids the edge roll-off effects inherent in global network measures, where nodes near the study area boundary receive artificially low scores.

Localised metrics are directly comparable across different locations and cities because the analysis window is defined by the distance threshold, not by the extent of the dataset. Common distance thresholds include:

- **400m** -- walkable neighbourhood (approximately 5 minutes)
- **800m** -- transit stop catchment (approximately 10 minutes)
- **1600m** -- local area (approximately 20 minutes)
- **5000m+** -- city-wide structure (consider using [adaptive sampling](#adaptive-sampling) at this scale)

Nodes at the periphery of a study area can be marked as "dead" (non-live) using a boundary polygon. Dead nodes are still reachable as targets during network traversal, but do not serve as sources for metric computation. This prevents artificially depressed values at the edges of the study area without discarding network connectivity. See the [Live Nodes](https://benchmark-urbanism.github.io/cityseer-examples/recipes/live_nodes.html) recipe for a worked example.

### Primal and dual graphs

Street networks can be represented in two complementary forms:

- **Primal graph**: Intersections are nodes; streets connecting them are edges. This is the conventional representation used by most network analysis tools.
- **Dual graph**: Streets (segments) become nodes; edges connect pairs of streets that meet at a common intersection. Each dual node is positioned at the midpoint of its corresponding street segment, so metrics are measured per street segment and can be visualised directly on street geometries.

The dual representation is essential for angular (simplest-path) analysis because each street segment forms an explicit routing state, enabling the accumulation of angular change at each turn. It also produces more intuitive outputs: a map coloured by betweenness shows which streets carry the most through-movement, rather than which intersections do.

The [`CityNetwork`](/api/network) class builds the dual graph automatically from input geometries; there is no need to call conversion functions manually. When using the lower-level API, convert a primal graph with [`graphs.nx_to_dual`](/tools/graphs#nx-to-dual) before computing angular centralities.

See the [Create Dual Graph](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/create_dual_graph.html) example for a visual comparison.

### Shortest-path and simplest-path heuristics

`cityseer` supports two routing heuristics:

- **Shortest path (metric)**: Routes minimise cumulative metric distance. Reflects physical walking distance.
- **Simplest path (angular)**: Routes minimise cumulative angular change along streets and at intersections. Reflects the cognitive simplicity of a route: a pedestrian following a simplest path prefers to continue straight ahead rather than turning, even if a shorter alternative exists.

When to use each:

- **Shortest path** is appropriate for accessibility analysis (how far is the nearest park?), walkability scoring, and situations where physical distance is the primary concern.
- **Simplest path** is appropriate for predicting pedestrian flows and commercial activity, because research shows that people tend to follow cognitively simple routes. It is also the basis for angular integration and choice measures used in space syntax research.

Both heuristics can be computed from a single `CityNetwork` instance at any combination of distance thresholds. When in doubt, compute both.

## CityNetwork API

The [`CityNetwork`](/api/network) class lets you build a network, compute centrality and land-use metrics, and export results without managing intermediate data structures. It builds dual graphs directly from input geometries and handles graph cleaning automatically.

### Constructors

| Constructor | Input format | Example |
| --- | --- | --- |
| [`from_geopandas`](/api/network#from-geopandas) | GeoDataFrame of LineStrings | [Network from Streets](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/network_from_streets.html) |
| [`from_nx`](/api/network#from-nx) | NetworkX MultiGraph or MultiDiGraph | [OSMnx to Cityseer](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/osmnx_to_cityseer.html) |
| [`from_osm`](/api/network#from-osm) | Shapely polygon (downloads from OSM) | [Create from BBox](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/create_from_bbox.html) |
| [`from_wkts`](/api/network#from-wkts) | Dictionary of WKT strings or Shapely geometries | -- |
| [`load`](/api/network#load) | Previously saved parquet/pickle pair | [Save to File](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/save_to_file.html) |

### Method chaining

Most methods return `self`, enabling fluent method chaining:

```python
cn = (
    CityNetwork.from_geopandas(edges_gdf, crs=32632)
    .set_boundary(boundary_polygon)
    .centrality_shortest(distances=[400, 800, 1600])
    .centrality_simplest(distances=[400, 800, 1600])
)
```

### Retrieving results

Because `CityNetwork` uses a dual graph internally, the `nodes_gdf` property exposes each street segment as a row with a Point geometry at the segment midpoint. To obtain results with the original LineString geometries (suitable for mapping and export), call [`to_geopandas()`](/api/network#to-geopandas):

```python
# Original LineString geometries with all computed columns
result_gdf = cn.to_geopandas()
result_gdf.to_file("results.gpkg")
```

### Automatic graph cleaning

Input geometries are automatically cleaned during construction: short self-loops, near-duplicate edges, and short danglers are removed. The [`feature_status`](/api/network#feature-status) property returns a Series with values such as `"active"`, `"short_self_loop"`, `"duplicate"`, `"short_dangler"`, or `"invalid_geometry"`, indicating what happened to each input feature. When using the lower-level API, the [`tools.graphs`](/tools/graphs) module provides manual graph cleaning functions; see the [Network Simplification](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/network_simplification.html) example.

## Centrality

Centrality metrics quantify the structural importance of each location in the street network. `cityseer` computes multiple centrality measures simultaneously for any combination of distance thresholds, leveraging Rust parallelism to minimise computation time.

### Shortest-path centrality

[`centrality_shortest`](/api/network#centrality-shortest) (or [`node_centrality_shortest`](/metrics/networks#node-centrality-shortest) in the lower-level API) computes the following metrics for each distance threshold `d`:

| Column | Description |
| --- | --- |
| `cc_density_{d}` | Count of reachable nodes within distance `d`. |
| `cc_harmonic_{d}` | Harmonic closeness: sum of inverse distances to reachable nodes. Higher values indicate better average proximity. |
| `cc_farness_{d}` | Sum of distances to all reachable nodes. Lower values indicate closer integration. |
| `cc_hillier_{d}` | Hillier normalisation (`density^2 / farness`), a space syntax closeness variant that normalises for the mathematical relationship between the count and cost of reachable destinations. |
| `cc_cycles_{d}` | Circuit rank of the locally reachable subgraph: the number of independent loops (city blocks). |
| `cc_decay_{d}` | Decay-weighted closeness using the [`decay_fn`](#decay-functions) expression. |
| `cc_betweenness_{d}` | Betweenness centrality: count of shortest paths passing through each node. High betweenness indicates through-movement potential. |
| `cc_betweenness_decay_{d}` | Decay-weighted betweenness using the `decay_fn` expression. |

### Simplest-path centrality

[`centrality_simplest`](/api/network#centrality-simplest) (or [`node_centrality_simplest`](/metrics/networks#node-centrality-simplest)) computes angular centrality metrics. Note the `_ang` suffix:

| Column | Description |
| --- | --- |
| `cc_density_{d}_ang` | Count of reachable nodes within distance `d` (angular routing). |
| `cc_harmonic_{d}_ang` | Harmonic closeness using cumulative angular change as impedance. |
| `cc_farness_{d}_ang` | Sum of cumulative angular changes to reachable nodes (angular integration in space syntax terminology). |
| `cc_hillier_{d}_ang` | Hillier normalisation: `density^2 / farness`. |
| `cc_betweenness_{d}_ang` | Betweenness using simplest angular paths (angular choice in space syntax terminology). |

Simplest-path centrality does not accept a `decay_fn` parameter because angular centralities use cumulative angular change as the routing cost rather than distance-based decay.

### Centrality recipes

- [Metric Centrality from GeoDataFrame](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_metric_centrality.html) -- shortest-path centrality workflow
- [Angular Centrality from GeoDataFrame](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_angular_centrality.html) -- simplest-path centrality workflow
- [OSM Centrality](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/osm_centrality.html) -- end-to-end from OpenStreetMap

## Decay Functions

Centrality, accessibility, mixed-use, and statistical aggregation methods accept an optional `decay_fn` parameter that controls how distance affects metric weighting. The decay function is a string expression using a single variable `p`, representing normalised progress from the source (`p = 0`) to the distance cutoff (`p = 1`), where `p = network_distance / max_distance`. Decay applies to shortest-path (metric distance) computations only; simplest-path (angular) centralities do not use decay because angular change is not a distance measure.

### When to use each preset

| Preset | Helper | When to use |
| --- | --- | --- |
| Exponential | `decay.exponential()` | Pedestrian catchments where nearby destinations matter far more than distant ones. Default for centrality. |
| Linear | `decay.linear()` | Uniform distance penalty with no abrupt boundary. |
| Flat | `decay.flat()` | Simple counts within a threshold, with no distance weighting. Default for accessibility and stats. |
| Gaussian | `decay.gaussian(peak, cutoff)` | Facilities with an optimal catchment distance (e.g. a park best accessed at 400m, not 0m). |
| Logistic | `decay.logistic(midpoint, cutoff)` | Hard catchment boundaries with a soft transition (e.g. a 600m service area that tapers at the edges). |

### Code examples

Decay functions apply to any method that accepts a `decay_fn` parameter:

```python
from cityseer import decay

# Exponential with custom steepness (default is 4)
cn.centrality_shortest(distances=[800], decay_fn=decay.exponential(steepness=6))

# Gaussian peaking at 400m within a 1200m cutoff
# (land-use methods return both the network and the assigned data GeoDataFrame)
cn, data_gdf = cn.compute_stats(
    data_gdf=prices_gdf,
    stats_column_labels=["price"],
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)

# Custom expression using the p variable directly
cn.centrality_shortest(distances=[800], decay_fn="max(0, 1 - p^2)")
```

### Default decay behaviour

- **Centrality** (`centrality_shortest`): defaults to `"exp(-4 * p)"` (exponential decay reaching ~1.8% at the cutoff).
- **Accessibility and stats** (`compute_accessibilities`, `compute_mixed_uses`, `compute_stats`): defaults to `"1"` (flat, no distance weighting). Pass a decay expression explicitly for distance-weighted aggregations.

### Expression syntax

Expressions support: `+`, `-`, `*`, `/`, `^`, and functions `exp`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `min`, `max`, `floor`, `ceil`, plus constants `pi` and `e`. Output values are automatically clamped to [0, 1]. See the [`cityseer.decay`](/api/decay) API reference for full details.

:::note
**Migrating from the old `betas` API:** The `betas` and `min_threshold_wt` parameters have been replaced by `decay_fn`. If you were using `distances=[800]` without specifying `betas`, your code continues to work unchanged. If you were passing explicit `betas` values for exponential decay, replace them with `decay_fn=decay.exponential()` (or a custom steepness). The old `_wt` and `_nw` column suffixes have been removed; all metrics now produce a single column controlled by `decay_fn`.
:::

## Land-Use Analysis

`cityseer` computes land-use and statistical aggregations over the street network from the same node locations used for centrality analysis, enabling direct correlation between structural and functional characteristics. All land-use methods accept an `angular=True` parameter for simplest-path routing (`CityNetwork` handles the required dual graph automatically).

### Accessibility

[`compute_accessibilities`](/metrics/layers#compute-accessibilities) counts the number of reachable instances of each specified land-use category and records the nearest distance. Counts may be optionally weighted by a decay function. The `angular=True` parameter enables simplest-path routing for accessibility analysis.

```python
cn, landuses_gdf = cn.compute_accessibilities(
    data_gdf=landuses_gdf,
    landuse_column_label="category",
    accessibility_keys=["retail", "cafe", "park"],
    distances=[400, 800],
)
print(cn.nodes_gdf["cc_retail_800"])           # count within 800m
print(cn.nodes_gdf["cc_park_nearest_max_800"]) # nearest distance to park
```

See the [OSM Accessibility](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/osm_accessibility.html) recipe.

### Mixed-use diversity

[`compute_mixed_uses`](/metrics/layers#compute-mixed-uses) measures the diversity of land-use categories reachable from each node:

- **Hill q=0** -- richness: count of distinct land-use categories. Best for granular classification schemas because evenness is irrelevant when categories are narrowly defined.
- **Hill q=1** -- balances richness and evenness. Use for moderate classification granularity where both the number and distribution of categories matter.
- **Hill q=2** -- emphasises the most frequent categories. Use for coarse schemas where the balance of dominant categories matters more than rare ones.

See the [Mixed Uses](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/gpd_mixed_uses.html) recipe.

### Statistical aggregations

[`compute_stats`](/metrics/layers#compute-stats) computes descriptive statistics (sum, mean, median, count, variance, mad, max, min) for numerical columns over the street network.

See the [Statistical Aggregations](https://benchmark-urbanism.github.io/cityseer-examples/recipes/stats/gpd_stats.html) recipe.

## Directed Graphs and One-Way Streets

By default, `cityseer` builds undirected networks where every street is traversable in both directions. This is correct for pedestrian analysis. For cycling or vehicular analysis where one-way streets matter, enable directed mode: one-way streets are restricted to their designated direction while two-way streets remain bidirectional.

### From a GeoDataFrame

Provide a boolean `oneway` column. Features with `oneway=True` are one-way in their LineString coordinate order:

```python
cn = CityNetwork.from_geopandas(gdf, directed=True)
```

### From a NetworkX MultiDiGraph

Passing a `MultiDiGraph` to [`from_nx`](/api/network#from-nx) automatically enables directed mode:

```python
cn = CityNetwork.from_nx(G_digraph)
assert cn.is_directed
```

### From OpenStreetMap (via OSMnx)

The built-in `from_osm` uses undirected simplification. For directed OSM data, fetch via [OSMnx](https://osmnx.readthedocs.io/) and convert:

```python
import osmnx as ox
from cityseer.tools import io

G_osmnx = ox.graph_from_polygon(polygon, network_type="drive")
G_osmnx = ox.projection.project_graph(G_osmnx, to_crs="EPSG:32630")
G_cityseer = io.nx_from_osm_nx(G_osmnx, directed=True)
cn = CityNetwork.from_nx(G_cityseer)
```

:::warning
Graph cleaning functions in the [`graphs`](/tools/graphs) module do not preserve edge directionality. Pass directed graphs directly to `CityNetwork.from_nx` or `CityNetwork.from_geopandas(directed=True)`.
:::

## Elevation and Slope

`cityseer` supports optional z (elevation) coordinates on network geometries. When present, [Tobler's hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) automatically adjusts traversal costs: uphill segments incur a penalty proportional to the grade, steep downhill segments are also penalised, and gentle downhill slopes (~3%) receive a slight bonus. The penalty is computed directionally (A to B differs from B to A) and is applied on top of the configured walking speed.

For angular analysis, the slope penalty affects only the reachability budget, not the angular routing metric itself.

When z coordinates are absent, all slope penalties default to 1.0 (no effect). To add elevation to a 2D network, drape it onto a digital elevation model (DEM) using a tool such as the [OSMnx elevation module](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.elevation) or [rasterio](https://rasterio.readthedocs.io/).

See the [3D Elevation](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/3d_elevation.html) example.

## Column Naming Conventions

All computed metrics are written to columns on the `nodes_gdf` GeoDataFrame following a consistent pattern:

```text
cc_{metric}_{distance}        -- shortest-path metric
cc_{metric}_{distance}_ang    -- simplest-path (angular) metric
```

The `cc_` prefix identifies columns generated by `cityseer`. Examples:

```text
cc_harmonic_800         -- harmonic closeness at 800m
cc_betweenness_800_ang  -- angular betweenness at 800m
cc_hill_q0_400          -- Hill diversity q=0 at 400m
cc_retail_200               -- accessibility count for "retail" at 200m
cc_retail_nearest_max_800   -- nearest distance to "retail" at max threshold
cc_price_mean_1200          -- mean of "price" column at 1200m
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

The [`metrics.visibility`](/metrics/visibility) module computes line-of-sight visibility from street-level observer locations, accounting for building obstructions. See the [Visibility from OSM](https://benchmark-urbanism.github.io/cityseer-examples/recipes/visibility/vis_osm.html) example.

### Street continuity

The [`metrics.observe`](/metrics/observe) module identifies coherent street sequences based on name, route number, or highway classification. See the [Street Continuity from OSM](https://benchmark-urbanism.github.io/cityseer-examples/recipes/continuity/continuity_osm.html) example.

### Public transport (GTFS)

The [`add_gtfs`](/api/network#add-gtfs) method integrates public transport stops and routes from GTFS data, enabling centrality and accessibility analyses that account for transit connections. See the [Centrality with Metro](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/centrality_metro.html) and [Accessibility with Metro](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/accessibility_metro.html) examples.

## Performance and Scale

The underlying algorithms are parallelised in Rust and scale to large networks. Typical performance on a modern laptop (8 cores):

- **10,000 edges** at 3 distance thresholds: seconds
- **50,000 edges** at 5 distance thresholds: under a minute
- **200,000+ edges** at long distances: minutes; consider [adaptive sampling](#adaptive-sampling) for 5 km+ thresholds

Computation time scales with the number of edges, the number of distance thresholds, and the reachability at each threshold. Simplest-path (angular) centrality is typically faster than shortest-path because angular routing produces sparser traversal trees.

## Adaptive Sampling

For large networks at long distance thresholds (5 km+), `cityseer` offers an experimental adaptive sampling feature. Rather than using every node as a source, a distance-dependent subset is sampled using the Hoeffding bound, with inverse-probability weighting to correct for non-uniform inclusion probabilities. Sampling is exact at short distances and progressively sparser at longer distances.

The `epsilon` parameter controls the error tolerance: `0.06` is a good starting point for most analyses, yielding Spearman rank correlations above 0.95 against exact computation. Use `0.03` for higher accuracy at the cost of more computation, or `0.1` for faster exploratory analysis:

```python
cn.centrality_shortest(
    distances=[800, 2000, 5000],
    sample=True,
    epsilon=0.06,
)
```

Both `centrality_shortest` and `centrality_simplest` support sampling.

:::warning
Adaptive sampling is experimental. When comparing centrality values across locations, use the same `epsilon` to ensure consistent error tolerances.
:::

For technical details, see the [`metrics.sampling`](/metrics/sampling) documentation.
