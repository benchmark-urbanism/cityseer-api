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

`cityseer` computes metrics locally rather than globally. For each node in the network, the surrounding subgraph is isolated up to a specified distance threshold (for example, all streets within 800m walking distance), metrics are computed within that local subgraph, and the process repeats for every node. This avoids the boundary effects inherent in global network measures, where nodes near the edge of the study area receive artificially low scores.

Localised metrics are directly comparable across different locations and cities because the analysis window is defined by the distance threshold, not by the extent of the dataset. Shorter distance thresholds capture local neighbourhood structure while longer thresholds reveal city-wide patterns.

Nodes at the periphery of a study area can be marked as "dead" (non-live) using a boundary polygon. Dead nodes participate fully in network traversal but their own values are not reported: results are only written for live nodes. For closeness, dead nodes are skipped as sources in exact mode (a pure cost saving); for betweenness, every node serves as a source so that routes passing through the study area — including those between dead nodes — correctly credit the live nodes they traverse. This prevents artificially depressed values at the edges of the study area without discarding network connectivity. See the [Live Nodes](https://benchmark-urbanism.github.io/cityseer-examples/recipes/live_nodes.html) recipe for a worked example.

### Primal and dual graphs

Street networks can be represented in two complementary forms:

- **Primal graph**: Intersections are nodes; streets connecting them are edges. This is the conventional representation used by most network analysis tools.
- **Dual graph**: Streets (segments) become nodes; edges connect pairs of streets that meet at a common intersection. Each dual node is positioned at the midpoint of its corresponding street segment, so metrics are measured per street segment and can be visualised directly on street geometries.

The dual representation is needed for angular (simplest-path) analysis because each street segment becomes an explicit node in the graph, allowing the cumulative turning angle (the sum of directional changes at each junction along a route) to be tracked explicitly. It also produces more intuitive outputs: a map coloured by betweenness shows which streets carry the most through-movement, rather than which intersections do.

The [`CityNetwork`](/api/network) class builds the dual graph automatically from input geometries; there is no need to call conversion functions manually. When using the lower-level API, convert a primal graph with [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) before computing angular centralities.

See the [Create Dual Graph](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/create_dual_graph.html) example for a visual comparison.

### Shortest-path and simplest-path heuristics

`cityseer` supports two routing heuristics:

- **Shortest path**: Routes minimise cumulative physical distance along the network. A 400m route is preferred over a 600m route, regardless of how many turns are involved.
- **Simplest path (angular)**: Routes minimise cumulative angular change — the total amount of turning at each junction. A pedestrian following a simplest path prefers to continue straight ahead rather than turning, even if a shorter alternative exists.

When to use each:

- **Shortest path** is appropriate for accessibility analysis (how far is the nearest park?), walkability scoring, and situations where physical distance is the primary concern.
- **Simplest path** is appropriate for predicting pedestrian flows and commercial activity, because research shows that people tend to follow cognitively simple routes. It is also the basis for angular integration and choice measures used in space syntax research.

Both heuristics can be computed from a single `CityNetwork` instance at any combination of distance thresholds. When in doubt, compute both.

## CityNetwork API

The [`CityNetwork`](/api/network) class lets you build a network, compute centrality and land-use metrics, and export results without managing intermediate data structures. It builds dual graphs directly from input geometries and handles graph cleaning automatically.

### Constructors

| Constructor | Input format | Example |
| --- | --- | --- |
| [`from_geopandas`](/api/network#from_geopandas) | GeoDataFrame of LineStrings | [Network from Streets](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/network_from_streets.html) |
| [`from_nx`](/api/network#from_nx) | NetworkX MultiGraph or MultiDiGraph | [OSMnx to Cityseer](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/osmnx_to_cityseer.html) |
| [`from_osm`](/api/network#from_osm) | Shapely polygon (downloads from OSM) | [Create from BBox](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/create_from_bbox.html) |
| [`from_wkts`](/api/network#from_wkts) | Dictionary of WKT strings or Shapely geometries | -- |
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

Because `CityNetwork` uses a dual graph internally, the `nodes_gdf` property exposes each street segment as a row with a Point geometry at the segment midpoint. To obtain results with the original LineString geometries (suitable for mapping and export), call [`to_geopandas()`](/api/network#to_geopandas):

```python
# Original LineString geometries with all computed columns
result_gdf = cn.to_geopandas()
result_gdf.to_file("results.gpkg")
```

### Automatic graph cleaning

Input geometries are automatically cleaned during construction: short self-loops, near-duplicate edges, and short danglers are removed. The [`feature_status`](/api/network#citynetwork) property returns a Series with values such as `"active"`, `"short_self_loop"`, `"duplicate"`, `"short_dangler"`, or `"invalid_geometry"`, indicating what happened to each input feature. When using the lower-level API, the [`tools.graphs`](/tools/graphs) module provides manual graph cleaning functions; see the [Network Simplification](https://benchmark-urbanism.github.io/cityseer-examples/recipes/networks/network_simplification.html) example.

## Centrality

Centrality metrics quantify the structural importance of each location in the street network. `cityseer` computes multiple centrality measures simultaneously for any combination of distance thresholds in a single pass.

### Expression-based metrics

Metrics are defined as `{name: expression}` dictionaries using two variables:

- **`c`** (cost): the raw routing cost to each reached node. For shortest paths, `c` is the metric distance in metres. For simplest paths, `c` is the cumulative angular change in degrees.
- **`p`** (progress): normalised progress from 0 at the source to 1 at the distance threshold. For shortest paths, `p = c / threshold`. For simplest paths, `p = elapsed_time / max_time`.

Metrics fall into four categories:

| Category | Role | Example |
| --- | --- | --- |
| **Closeness** | Expression is evaluated once per reached node and summed across all reachable nodes. | `{"harmonic": "1/c"}` accumulates $\sum_j 1/c_j$ |
| **Betweenness** | Expression weights the contribution of each destination during Brandes backpropagation. | `{"betweenness": "1"}` counts paths equally |
| **Cycles** | Boolean flag; computes the circuit rank of the locally reachable subgraph (shortest-path only). | `cycles=True` |
| **Postprocess** | Derives new columns from previously computed metrics using simple arithmetic (`+`, `-`, `*`, `/`, `**`). | `{"hillier": "density**2 / farness"}` |

Pass `None` to use the defaults for a category, or `{}` to skip it entirely.

### Shortest-path centrality

[`centrality_shortest`](/api/network#centrality_shortest) (or [`centrality_shortest`](/metrics/networks#centrality_shortest) in the lower-level API) computes the following default metrics for each distance threshold `d`. In the formulas below, the sum is over all nodes $j$ reachable within $d$, $c_j$ is the shortest-path distance in metres to node $j$, and $p_j = c_j / d$.

**Default closeness** (pass `closeness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_density_{d}` | `"1"` | $\sum_j 1$ | Count of nodes reachable within distance $d$. A simple measure of local connectivity. |
| `cc_farness_{d}` | `"c"` | $\sum_j c_j$ | Sum of metric distances to all reachable nodes. Lower values indicate better average proximity. |
| `cc_harmonic_{d}` | `"1/c"` | $\sum_j 1 / c_j$ | Harmonic closeness: sum of inverse distances. Higher values indicate better proximity. Unlike standard closeness, harmonic closeness handles distance-bounded analysis correctly because unreachable nodes contribute 0 rather than distorting the average. |
| `cc_decay_{d}` | `"exp(-4 * p)"` | $\sum_j e^{-4 p_j}$ | Exponential decay-weighted closeness. Nearby nodes contribute most; at the distance threshold ($p = 1$), weight drops to $e^{-4} \approx 1.8\%$. This is the continuous equivalent of the historical $\beta$-weighted metric. |

**Default betweenness** (pass `betweenness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_betweenness_{d}` | `"1"` | $\sum_{s,t} \sigma_{st}(v) / \sigma_{st}$ | Unweighted betweenness: for each origin–destination pair $(s, t)$, counts the fraction of shortest paths that pass through node $v$. High values indicate through-movement potential. |
| `cc_betweenness_decay_{d}` | `"exp(-4 * p)"` | $\sum_{s,t} e^{-4 p_t} \cdot \sigma_{st}(v) / \sigma_{st}$ | Decay-weighted betweenness: each pair's contribution is downweighted by the exponential decay applied to the destination's normalised distance $p_t$. Paths to distant destinations count less. |

**Default postprocess** (pass `postprocess=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_hillier_{d}` | `"density**2 / farness"` | $n^2 / \sum_j c_j$ | Hillier normalisation: rewards locations that are both well-connected (high density $n$) and proximate (low farness). This is the standard normalisation used in space syntax research. |

**Cycles** (`cycles=True` by default):

| Column | Description |
| --- | --- |
| `cc_cycles_{d}` | Circuit rank of the locally reachable subgraph: the number of independent loops ($e - n + 1$ where $e$ is edges and $n$ is nodes in the subgraph). In grid-like networks, each loop roughly corresponds to a city block; in less regular networks, it measures the redundancy of route choices. |

### Simplest-path centrality

[`centrality_simplest`](/api/network#centrality_simplest) (or [`centrality_simplest`](/metrics/networks#centrality_simplest) in the lower-level API) computes angular centrality metrics. For simplest paths, `c` is the cumulative angular change in degrees (the total turning at each junction along a route) rather than metric distance. The variable `p` is normalised elapsed time (not angular cost), so the distance reachability budget is still metric. Note the `_ang` suffix on all output columns.

**Default closeness** (pass `closeness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_density_{d}_ang` | `"1"` | $\sum_j 1$ | Count of nodes reachable within distance $d$ via angular routing. |
| `cc_farness_{d}_ang` | `"1 + c / 90"` | $\sum_j (1 + c_j / 90)$ | Angular farness. The $1 + c/90$ transform maps a straight-ahead path ($0°$) to 1 and a single $90°$ turn to 2, giving a meaningful scale that avoids division-by-zero at the source. Equivalent to angular integration in space syntax. |
| `cc_harmonic_{d}_ang` | `"1 / (1 + c / 90)"` | $\sum_j 1 / (1 + c_j / 90)$ | Angular harmonic closeness: the inverse of the farness expression. Higher values indicate better angular proximity (straighter routes to more destinations). |

**Default betweenness** (pass `betweenness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_betweenness_{d}_ang` | `"1"` | $\sum_{s,t} \sigma_{st}(v) / \sigma_{st}$ | Angular betweenness: fraction of simplest (minimum angular change) paths through each node. Equivalent to angular choice in space syntax. |

**Default postprocess** (pass `postprocess=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_hillier_{d}_ang` | `"density**2 / farness"` | $n^2 / \sum_j (1 + c_j / 90)$ | Hillier normalisation for angular metrics. |

Simplest-path centrality does not include decay-weighted metrics by default because angular cost is not a distance measure. Decay-weighted angular metrics can be added via custom expressions if needed.

### Custom metrics

To define custom metrics, pass a dictionary of `{name: expression}` pairs. Expressions can use `c`, `p`, and the mathematical functions `exp`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `min`, `max`, `floor`, `ceil`, plus constants `pi` and `e`.

```python
# Custom gravity model and linear-decay betweenness
cn.centrality_shortest(
    distances=[800],
    closeness={"gravity": "1 / c^2", "reach": "1"},
    betweenness={"bt_linear": "1 - p"},
    postprocess={},  # skip hillier
)

# Angular centrality with decay-weighted betweenness
cn.centrality_simplest(
    distances=[800],
    betweenness={"betweenness": "1", "bt_decay": "exp(-4 * p)"},
)
```

### Node weights

Every node carries a `weight` (default `1.0`). Set it on the nodes `GeoDataFrame`, or add a `weight` attribute to your NetworkX nodes before ingestion, to apply gravity-style weighting to centrality:

- **Closeness** weights each reachable node by its destination weight, so `density` becomes $\sum_j w_j$ (the sum of reachable node weights) rather than a plain count, and other closeness expressions are scaled accordingly. A node's own weight does **not** rescale its own score — weighting reflects the *opportunities it can reach*.
- **Betweenness** weights each origin–destination pair by the **product** of its endpoint weights $w_s \cdot w_t$, the standard gravity-flow form.

The same weighting is applied identically whether or not [adaptive sampling](#adaptive-sampling) is used. With the default weights of `1.0` the results are unchanged from an unweighted analysis.

:::note
Node weights affect **centrality only**. Land-use accessibility, mixed-use diversity, and statistical aggregations are intentionally *not* node-weighted — they weight reachable land-use data points (optionally by [distance decay](#decay-functions)), not network nodes.
:::

### Segment-weighted centrality

`segment_weighted=True` is a convenience preset over the node `weight` mechanism above: it temporarily sets each dual-graph node's weight to its corresponding street segment length, then restores the original weights afterwards. This means:

- **Closeness** metrics reflect total reachable street length rather than node counts (e.g. density becomes total metres of reachable street within the threshold).
- **Betweenness** weights each origin–destination pair by both endpoint segment lengths, so longer streets contribute more to betweenness flows.

This requires a dual graph representation (which `CityNetwork` builds automatically).

```python
cn.centrality_shortest(distances=[800], segment_weighted=True)
```

### Convenience wrappers

For cases where only closeness or only betweenness is needed, convenience functions skip the unused category:

- [`closeness_shortest`](/metrics/networks#closeness_shortest) / [`closeness_simplest`](/metrics/networks#closeness_simplest) — closeness only (betweenness disabled)
- [`betweenness_shortest`](/metrics/networks#betweenness_shortest) / [`betweenness_simplest`](/metrics/networks#betweenness_simplest) — betweenness only (closeness and cycles disabled)

### Origin–destination and demand betweenness

Standard betweenness treats every node pair equally. When you have real or modelled travel flows, two functions route those flows instead:

- [`betweenness_od`](/metrics/networks#betweenness_od) takes an **explicit** origin–destination matrix (build one from flow data and zone centroids with [`build_od_matrix`](/metrics/networks#build_od_matrix)) and accumulates each pair's trip weight along shortest paths.
- [`betweenness_demand`](/metrics/networks#betweenness_demand) takes weighted **origins** and **destinations** separately and *models* the matrix with a **singly (origin-)constrained spatial interaction model**: each origin distributes its full weight across reachable destinations in proportion to $W_d \cdot f(c_{od})$, where $f$ is a `decay_fn` expression. The classic gravity model is recovered with an exponential decay. The allocation is computed in the same traversal that routes the flows, so no explicit matrix is needed.

```python
from cityseer.metrics import networks

# population blocks -> retail attractors, exponential distance decay
nodes_gdf = networks.betweenness_demand(
    network_structure, nodes_gdf,
    origins_gdf=population_gdf, destinations_gdf=retail_gdf,
    origin_weight_col="population", destination_weight_col="floorspace",
    distances=[800], decay_fn="exp(-0.002 * c)",
)
print(nodes_gdf["cc_demand_800"])
```

Because both functions route flows through the same Brandes machinery, points that snap to the same network node have their weights **summed** (not overwritten), preserving total demand at each junction.

### Centrality recipes

- [Metric Centrality from GeoDataFrame](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_metric_centrality.html) -- shortest-path centrality workflow
- [Angular Centrality from GeoDataFrame](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/gpd_angular_centrality.html) -- simplest-path centrality workflow
- [OSM Centrality](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/osm_centrality.html) -- end-to-end from OpenStreetMap

## Decay Functions

Distance decay controls how feature importance or metric weighting decreases with distance from an analysis point. For **centrality**, decay is built into the metric expressions (e.g. the default `"exp(-4 * p)"` closeness and betweenness metrics described above). For **land-use methods** (`compute_accessibilities`, `compute_mixed_uses`, `compute_stats`), an optional `decay_fn` parameter accepts a string expression using a variable `p` that ranges from 0 at the source to 1 at the distance cutoff (`p = network_distance / max_distance`). The [`cityseer.decay`](/api/decay) module provides helper functions that return pre-built expression strings for common decay shapes.

### How decay weighting works

A decay function maps **normalised progress** `p` to a weight. `p = 0` at the analysis node (the source) and `p = 1` at the distance (or time) cutoff, so `p = network_distance / threshold`. The function is evaluated **once per reached element** — for every reachable node in a centrality calculation, or every reachable data point in a land-use calculation — and the resulting weight scales that element's contribution to the metric (a count, a numerical value, or a diversity contribution).

A few properties are worth understanding:

- **Per-threshold normalisation.** When several `distances` are requested, `p` is recomputed against each threshold independently. The same physical point therefore has a larger `p` (and so less weight under a decaying function) at a short threshold than at a long one, keeping every catchment internally consistent.
- **Clamping (land-use only).** Land-use decay output is clamped to `[0, 1]`, so an expression can never produce negative or amplifying weights. Centrality expressions are **not** clamped, because they are general metric formulas (e.g. `1/c`) rather than weights.
- **Flat by default for land-use.** With the default `"1"`, every reachable point contributes a weight of 1 — i.e. a plain unweighted count or sum within the threshold.
- **Decay vs. metric.** In centrality the decay is simply one possible metric expression (the default `"exp(-4 * p)"` `decay`/`betweenness_decay` columns). In the land-use methods the decay is a separate `decay_fn` that multiplies whatever is being aggregated.

### When to use each preset

| Preset | Helper | When to use |
| --- | --- | --- |
| Exponential | `decay.exponential()` | Pedestrian catchments where nearby destinations matter far more than distant ones. Default for centrality. |
| Linear | `decay.linear()` | Uniform distance penalty with no abrupt boundary. |
| Flat | `decay.flat()` | Simple counts within a threshold, with no distance weighting. Default for accessibility and stats. |
| Gaussian | `decay.gaussian(peak, cutoff)` | Use cases where peak relevance is at some distance from the source rather than immediately adjacent. |
| Logistic | `decay.logistic(midpoint, cutoff)` | Catchment boundaries with a gradual transition rather than a hard cutoff. |

### Code examples

**Centrality** metrics are specified as `{name: expression}` dicts using variables `c` (cost) and `p` (normalised progress). **Land-use methods** use `decay_fn` with the `p` variable:

```python
from cityseer import decay

# Centrality: custom closeness metric using c (cost in metres)
cn.centrality_shortest(
    distances=[800],
    closeness={"gravity": "exp(-0.005 * c)", "harmonic": "1/c"},
)

# Gaussian decay for land-use stats
cn, data_gdf = cn.compute_stats(
    data_gdf=prices_gdf,
    stats_column_labels=["price"],
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)
```

### Default decay behaviour

- **Centrality** (`centrality_shortest`): default closeness includes `"decay": "exp(-4 * p)"` and default betweenness includes `"betweenness_decay": "exp(-4 * p)"`.
- **Accessibility and stats** (`compute_accessibilities`, `compute_mixed_uses`, `compute_stats`): defaults to `"1"` (flat, no distance weighting). Pass a decay expression explicitly for distance-weighted aggregations.

### Multiple decays in one traversal

The expensive part of a land-use computation is the network traversal from every node; applying a decay weight to the reachable points is cheap by comparison. So instead of calling a method once per decay shape — repeating the traversal each time — the land-use methods (`compute_accessibilities`, `compute_mixed_uses`, `compute_stats`) let `decay_fn` be a `{label: expression}` dict and compute **every decay variant in a single shared traversal**.

- **Input.** `decay_fn` may be a single expression string, `None` (flat, the default), or a `{label: expression}` dict.
- **Output naming.** Each label is appended to that variant's output columns: `decay_fn={"grav": ..., "raw": ...}` yields `cc_retail_grav_800`, `cc_retail_raw_800`, and so on. A plain string or `None` adds **no** suffix, so existing column names — and their values — are unchanged. The dict form is therefore purely additive and backwards compatible.
- **When to use it.** Whenever you want the same features summarised under more than one distance weighting: a gravity-weighted *and* a plain count of the same amenity; or several catchment shapes (exponential, Gaussian, flat) for a sensitivity analysis. A pipeline that previously made *N* calls collapses to one.

```python
# gravity-weighted AND plain-count accessibility to retail, in one pass
cn, landuses_gdf = cn.compute_accessibilities(
    data_gdf=landuses_gdf,
    landuse_column_label="category",
    accessibility_keys=["retail"],
    distances=[800],
    decay_fn={"grav": decay.gaussian(peak=200, cutoff=800, std=150), "raw": decay.flat()},
)
print(cn.nodes_gdf[["cc_retail_grav_800", "cc_retail_raw_800"]])
```

This mirrors how `centrality_shortest` accepts a `{name: expression}` dict of metrics evaluated in a single traversal. Each labelled variant produces the method's full set of output columns. One caveat for `compute_mixed_uses`: only the Hill measures are distance-weighted (they use branch-distance weighting), so Shannon and Gini are computed from raw category counts and will be identical across labels.

### Expression syntax

Centrality expressions use two variables: `c` (raw cost) and `p` (normalised progress, `c / threshold`). Land-use decay expressions use `p` only. Both support: `+`, `-`, `*`, `/`, `^`, and functions `exp`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `min`, `max`, `floor`, `ceil`, plus constants `pi` and `e`. Land-use decay output is clamped to [0, 1]; centrality expressions are not clamped. See the [`cityseer.decay`](/api/decay) API reference for full details.

## Land-Use Analysis

`cityseer` computes land-use and statistical aggregations at the same node locations used for centrality. Because the results share a common spatial index, you can directly compare how well-connected a location is (centrality) with what amenities are reachable from it (accessibility). All land-use methods accept an `angular=True` parameter for simplest-path routing (`CityNetwork` handles the required dual graph automatically).

### Accessibility

[`compute_accessibilities`](/metrics/layers#compute_accessibilities) measures how many instances of each specified land-use category are reachable from every network node, and how far away the nearest instance is. For each category key and distance threshold it writes two kinds of column:

- `cc_{category}_{distance}` — the (optionally decay-weighted) **count** of reachable instances of that category within the threshold. With the default flat decay this is a plain count; with a decaying `decay_fn` it becomes a distance-weighted "gravity" accessibility.
- `cc_{category}_nearest_max_{distance}` — the network distance to the **nearest** instance of that category. This is written only at the largest threshold, since the nearest distance does not depend on the catchment size.

Pass `decay_fn` to weight counts by distance, including the `{label: expression}` dict form to produce several weightings at once (see [Multiple decays in one traversal](#multiple-decays-in-one-traversal)). The `angular=True` parameter enables simplest-path routing.

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

[`compute_mixed_uses`](/metrics/layers#compute_mixed_uses) measures the diversity of land-use categories reachable from each node. Hill numbers are computed by default (`compute_hill=True`); Shannon and Gini-Simpson indices are available via the `compute_shannon` and `compute_gini` flags. The three Hill orders differ in how strongly they weight common versus rare categories:

- **Hill q=0** (`cc_hill_q0_{d}`, equivalent to species richness) -- counts how many different land-use types are present. Best when using many fine-grained categories.
- **Hill q=1** (`cc_hill_q1_{d}`, equivalent to the exponential of Shannon entropy) -- accounts for both the number of land-use types and how evenly distributed they are.
- **Hill q=2** (`cc_hill_q2_{d}`, equivalent to the inverse Simpson concentration) -- focuses on the most common land-use types, downweighting rare ones. Best when using broad categories where the balance of dominant types matters most.

The Hill measures are distance-weighted through a branch-distance form, so a `decay_fn` shapes how strongly nearer instances count. Shannon (`cc_shannon_{d}`) and Gini (`cc_gini_{d}`) are computed from raw category counts and are not affected by `decay_fn`.

See the [Mixed Uses](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/gpd_mixed_uses.html) recipe.

### Statistical aggregations

[`compute_stats`](/metrics/layers#compute_stats) computes descriptive statistics for one or more numerical columns over the street network. For each input column and distance threshold it writes eight measures, named `cc_{column}_{measure}_{distance}`:

| Measure | Column suffix | Notes |
| --- | --- | --- |
| Sum | `_sum` | Decay-weighted sum of values. |
| Mean | `_mean` | Decay-weighted mean. |
| Count | `_count` | Sum of decay weights (a plain count under flat decay). |
| Variance | `_var` | Decay-weighted variance. |
| Median | `_median` | Weighted median. |
| MAD | `_mad` | Weighted median absolute deviation. |
| Max / Min | `_max` / `_min` | Extremes of reachable values (not affected by `decay_fn`). |

Pass a list of `stats_column_labels` to summarise several columns in one call, and a `decay_fn` to weight each value by distance — including the `{label: expression}` dict form for multiple weightings in a single traversal. By default all eight measures are produced; pass `measures=[...]` (any subset of the suffixes above) to compute only the ones you need. This keeps the output `GeoDataFrame` smaller and skips the weighted median/MAD sort when neither is requested.

```python
cn, prices_gdf = cn.compute_stats(
    data_gdf=prices_gdf,
    stats_column_labels=["price"],
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)
print(cn.nodes_gdf["cc_price_mean_1200"])
```

See the [Statistical Aggregations](https://benchmark-urbanism.github.io/cityseer-examples/recipes/stats/gpd_stats.html) recipe.

## Directed Graphs and One-Way Streets

By default, `cityseer` builds undirected networks where every street is traversable in both directions. This is correct for pedestrian analysis. For cycling or vehicular analysis where one-way streets matter, enable directed mode: one-way streets are restricted to their designated direction while two-way streets remain bidirectional.

### From a GeoDataFrame

Provide a boolean `oneway` column. Features with `oneway=True` are one-way in their LineString coordinate order:

```python
cn = CityNetwork.from_geopandas(gdf, directed=True)
```

### From a NetworkX MultiDiGraph

Passing a `MultiDiGraph` to [`from_nx`](/api/network#from_nx) automatically enables directed mode:

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

`cityseer` supports optional z (elevation) coordinates on network geometries. When present, elevation data is preserved through all processing steps (graph cleaning, decomposition, dual graph conversion, CRS reprojection, and serialisation). [Tobler's hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) automatically adjusts traversal costs based on the gradient: uphill segments incur a penalty proportional to the grade (for example, a 20% slope approximately doubles the effective distance), steep downhill segments are also penalised due to reduced walking speed, and gentle downhill slopes (~3%) receive a slight bonus matching the empirically observed optimal walking gradient. The penalty is a dimensionless multiplier on effective distance, computed directionally (A to B differs from B to A) and composing correctly with any configured walking speed.

For angular analysis, the slope penalty affects only the reachability budget (the distance a pedestrian can cover within the analysis threshold), not the angular routing metric itself. The cognitively simplest path is still selected, but steep terrain reduces how far the walker can reach.

When z coordinates are absent, all slope penalties default to 1.0 (no effect). To add elevation to a 2D network, drape it onto a digital elevation model (DEM) using a tool such as the [OSMnx elevation module](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.elevation) or [rasterio](https://rasterio.readthedocs.io/).

See the [3D Elevation](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/3d_elevation.html) example.

## Edge Impedance

Each network edge carries an `imp_factor` (default `1.0`) that multiplicatively scales its effective traversal cost — useful for representing road surface, road class, or any other static per-segment penalty. Impedance composes with the slope penalty above: the effective edge cost is `length × imp_factor × slope_pen`, so both factors apply together.

Set per-edge impedance by including an `imp_factor` column on the `GeoDataFrame` passed to `CityNetwork.from_geopandas`, an `imp_factor` attribute on edges of a `NetworkX` graph passed to `CityNetwork.from_nx`, or via the `impedances={fid: value}` keyword on `CityNetwork.from_wkts`. The value is propagated through dual graph construction: each dual edge — which traverses half of each adjacent primal segment — receives the **length-weighted mean** of its two primal impedances, so an all-`1.0` primal yields `1.0` on the dual (fully backwards compatible).

Impedance applies to **shortest-path** routing (and any equivalent time-converted budget), including the reachability budget used by simplest-path (angular) analysis. The angular cost itself is purely geometric (cumulative turning) and is **not** scaled by `imp_factor` — only how far an angular traversal can reach within the time budget.

## Column Naming Conventions

All computed metrics are written to columns on the `nodes_gdf` GeoDataFrame following a consistent pattern:

```text
cc_{metric}_{distance}            -- shortest-path metric
cc_{metric}_{distance}_ang        -- simplest-path (angular) metric
cc_{metric}_{label}_{distance}    -- land-use metric under a named decay label
```

The `cc_` prefix identifies columns generated by `cityseer`. The optional `{label}` segment appears only when a land-use method is called with a `{label: expression}` decay dict (see [Multiple decays in one traversal](#multiple-decays-in-one-traversal)); a single decay expression or `None` produces no label segment. Examples:

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

The [`metrics.visibility`](/metrics/visibility) module computes line-of-sight visibility from street-level observer locations, accounting for building obstructions. See the [Visibility from OSM](https://benchmark-urbanism.github.io/cityseer-examples/recipes/visibility/vis_osm.html) example.

### Street continuity

The [`metrics.observe`](/metrics/observe) module identifies coherent street sequences based on name, route number, or highway classification. See the [Street Continuity from OSM](https://benchmark-urbanism.github.io/cityseer-examples/recipes/continuity/continuity_osm.html) example.

### Public transport (GTFS)

The [`add_gtfs`](/api/network#add_gtfs) method integrates public transport stops and routes from GTFS data, enabling centrality and accessibility analyses that account for transit connections. See the [Centrality with Metro](https://benchmark-urbanism.github.io/cityseer-examples/recipes/centrality/centrality_metro.html) and [Accessibility with Metro](https://benchmark-urbanism.github.io/cityseer-examples/recipes/accessibility/accessibility_metro.html) examples.

## Performance and Scale

The underlying algorithms are parallelised in Rust and scale to large networks. Computation scales with the number of edges, the number of distance thresholds, and the reachability at each threshold. Simplest-path (angular) centrality is typically faster than shortest-path because angular routing explores fewer paths. For large networks at long distance thresholds, consider [adaptive sampling](#adaptive-sampling).

## Adaptive Sampling

For large networks at long distance thresholds, `cityseer` offers an experimental adaptive sampling feature. Rather than using every node as a source, a distance-dependent subset is selected using the Hoeffding inequality, a statistical bound that determines the minimum sample size needed to guarantee the approximation error stays within a specified tolerance. Results are corrected using inverse-probability weighting (IPW): if a node had a 25% chance of being selected as a source, its contribution is multiplied by 4 (the reciprocal of 0.25), so that the subset of sampled nodes approximates the result of using all nodes.

Sampling is applied per distance threshold. Short distances where few nodes are reachable run at full computation, since sampling offers no speedup. Longer distances automatically use sparser sampling as the number of reachable nodes increases. If the computed sampling rate offers no speedup for a given distance, that distance runs exactly.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is calibrated on real networks spanning the density range — from dense metropolitan grids to a sparse suburb — so that node rankings are preserved (Spearman ρ ≥ 0.95); denser networks clear the target comfortably. Both `centrality_shortest` and `centrality_simplest` support sampling. Pass `sample=True` to enable:

```python
cn.centrality_shortest(
    distances=[800, 2000, 5000],
    sample=True,
)
```

:::warning
Adaptive sampling is experimental: API and behaviour may change in future releases.
:::

For technical details, see the [`metrics.sampling`](/metrics/sampling) documentation.
