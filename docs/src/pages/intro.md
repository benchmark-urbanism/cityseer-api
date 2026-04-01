---
layout: '@src/layouts/PageLayout.astro'
---

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by network-based methods that have been developed from the ground-up for micro-morphological urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given street-front location. Importantly, `cityseer` computes metrics directly over the street network and offers distance-weighted variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian walking distances into account.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io), etc.), Open Street Map workflows with [`OSMnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/), [`geopandas`](https://geopandas.org/en/stable/) (and [`momepy`](http://docs.momepy.org)) stack of packages. The underlying algorithms are parallelised and implemented in `rust` so that the methods can be scaled to large networks. In-out convenience methods are provided for interfacing with `networkX` and graph cleaning tools aid the incorporation of complex network representations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api). Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package.

Code tests are run against Python versions `3.10` - `3.13`.

## CityNetwork API

As of v4.25, `cityseer` provides a new [`CityNetwork`](/api/network) class that wraps network construction, centrality computation, and land-use analysis into a single high-level interface. It builds dual graphs directly from LineString geometries (via GeoDataFrames, WKT dictionaries, NetworkX graphs, or OSM), bypassing the previous NetworkX-based construction pipeline for substantially faster builds. The existing lower-level API (individual functions in the `metrics`, `tools`, and `rustalgos` modules) remains fully available and unchanged for backwards compatibility.

## Decay Functions

Centrality, accessibility, mixed-use, and statistical aggregation methods accept an optional `decay_fn` parameter that controls how distance affects metric weighting. The decay function is expressed as a string using a single variable `p`, which represents normalised progress from the source node (`p = 0`) to the distance cutoff (`p = 1`), computed as `p = cost / max_cost`.

The [`cityseer.decay`](/api/decay) module provides helper functions for constructing common decay curves from absolute distance units (metres). Alternatively, expressions can be written directly using the `p` variable.

### Available presets

| Preset | Helper | Expression | Behaviour |
| --- | --- | --- | --- |
| Exponential | `decay.exponential()` | `"exp(-4 * p)"` | Steep initial decay, ~1.8% weight at cutoff. Default for centrality. |
| Linear | `decay.linear()` | `"max(0, 1 - p)"` | Uniform decay from 1 to 0. |
| Flat | `decay.flat()` | `"1"` | No decay: constant weight everywhere. Default for land-use and stats. |
| Gaussian | `decay.gaussian(peak, cutoff)` | bell curve | Peaks at a specified distance, useful for modelling facilities with an optimal catchment range. |
| Logistic | `decay.logistic(midpoint, cutoff)` | sigmoid | Sharp transition from full weight to zero, centred at a specified distance. |

### Using decay functions

```python
from cityseer import decay
from cityseer.metrics import layers

# Flat (unweighted) — count everything within the threshold equally
nodes_gdf, data_gdf = layers.compute_accessibilities(
    ..., distances=[800], decay_fn=decay.flat()
)

# Exponential — weight nearby locations more heavily (default for centrality)
nodes_gdf, data_gdf = layers.compute_stats(
    ..., distances=[800], decay_fn=decay.exponential()
)

# Gaussian — model a facility with an optimal 400m catchment within 1200m
nodes_gdf, data_gdf = layers.compute_stats(
    ..., distances=[1200], decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150)
)

# Custom expression — any valid math expression using the variable p
nodes_gdf, data_gdf = layers.compute_stats(
    ..., distances=[800], decay_fn="max(0, 1 - p^2)"  # quadratic decay
)
```

### Centrality decay

For [`node_centrality_shortest`](/metrics/networks#node-centrality-shortest), the `decay_fn` parameter defaults to `"exp(-4 * p)"` (exponential decay). This produces the `decay` and `betweenness_decay` output columns. Use `"1"` for flat (unweighted) decay metrics. The [`node_centrality_simplest`](/metrics/networks#node-centrality-simplest) function does not accept a `decay_fn` parameter because angular (simplest-path) centralities use angular cost rather than distance-based decay weighting.

### Supported expression syntax

Decay expressions are parsed by the `meval` crate and support the following:

- **Arithmetic**: `+`, `-`, `*`, `/`, `^` (power)
- **Functions**: `exp`, `ln`, `sqrt`, `abs`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `min`, `max`, `floor`, `ceil`
- **Constants**: `pi`, `e`
- **Clamping**: Output values are automatically clamped to the range [0, 1]

## Getting Started

A growing collection of recipes and examples is available via the [`Cityseer Examples`](https://benchmark-urbanism.github.io/cityseer-examples/) site. The example notebooks include workflows showing how to run graph cleaning, network centralities, and land-use accessibility analysis from data sources such as OSM or geospatial files (e.g. GeoPackages & Shapefiles).

## Local-Scale Analysis

`cityseer` is developed from the ground-up for pedestrian-scale urban analysis. It builds-on and further best-practices for urban analytics:

- It uses localised network analysis (as opposed to global forms of analysis) using a 'moving-window' methodology. A node is selected, the graph is then isolated at a selected distance threshold around the node, metrics are then computed, and then the process subsequently repeats for every other node in the network. `cityseer` exclusively uses localised methods for network analysis because they do not suffer from the same issues as global methods, which are inherently problematic because of edge roll-off effects. Localised methods have the distinct advantage of being comparable across different locations and cities, while also being capable of targeting both smaller and larger distance thresholds to reveal patterns at different scales of analysis.
- It is common to use either shortest-distance (metric) or simplest-path (shortest angular or geometric distance) heuristics for network analysis. In `cityseer`, simplest-path (angular) analysis is performed on dual graphs so that each segment forms an explicit routing state.
- `cityseer` supports analysis for both primal and dual graph representations, and contains methods for converting from primal (intersection-based) to dual (street-segment-based) representations. Shortest-path workflows support either topology. Angular workflows require the dual representation, which retains accurate street lengths and geometry (angles) while affording the opportunity to measure and visualise metrics relative to streets instead of intersections.
- `cityseer` supports customisable distance-decay weighting for centrality, accessibility, and mixed-use methods via the `decay_fn` parameter. See the [Decay Functions](#decay-functions) section for details.
- To support the evaluation of measures at finely-spaced intervals along street fronts, `cityseer` includes support for network decomposition.
- Granular evaluation of land-use accessibilities and mixed-uses requires that land uses be assigned to the street network in a contextually precise manner. `cityseer` assigns data-points to the nearest adjacent street segment and then allows access over the network from both sides, thereby allowing precise distances to be calculated dynamically based on the direction of approach.
- Centrality methods are susceptible to topological distortions arising from 'messy' graph representations as well as due to the conflation of topological and geometrical properties of street networks. `cityseer` addresses these through the inclusion of graph cleaning functions and procedures for splitting geometrical properties from topological representations.

## Directed Graphs and One-Way Streets

By default, `cityseer` builds undirected networks where every street can be traversed in both directions. This is appropriate for pedestrian analysis but does not reflect the constraints of cycling or vehicular traffic, where one-way streets restrict the direction of travel.

When directed mode is enabled, `cityseer` builds a directed graph where one-way streets are only traversable in their designated direction while two-way streets remain bidirectional. This affects all downstream computations: centrality metrics, accessibility, and mixed-use analyses all respect the directed topology.

:::note
The graph simplification and cleaning functions in the [`graphs`](/tools/graphs) module (e.g. `nx_remove_filler_nodes`, `nx_to_dual`) do not preserve edge directionality — do not pass a `MultiDiGraph` through them. The `from_osm` constructor also uses this undirected simplification pipeline internally. Directed graphs should be passed directly to [`CityNetwork.from_nx`](/api/network#from-nx), [`CityNetwork.from_geopandas`](/api/network#from-geopandas) (with `directed=True`), or [`io.network_structure_from_nx`](/tools/io#network-structure-from-nx). For directed OSM workflows, fetch a `MultiDiGraph` via [OSMnx](https://osmnx.readthedocs.io/) and pass it to `CityNetwork.from_nx`.
:::

### From a GeoDataFrame

Provide a boolean `oneway` column following the [OSMnx](https://osmnx.readthedocs.io/) and [momepy](https://docs.momepy.org/) convention. Features with `oneway=True` are one-way in their LineString coordinate order; features with `oneway=False` are two-way.

```python
import geopandas as gpd
from shapely.geometry import LineString

gdf = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString([(0, 0), (100, 0)]),   # one-way left to right
            LineString([(100, 0), (200, 0)]),  # two-way
        ],
        "oneway": [True, False],
    },
    crs="EPSG:32630",
)
cn = CityNetwork.from_geopandas(gdf, directed=True)
```

### From a NetworkX MultiDiGraph

Passing a `MultiDiGraph` to [`from_nx`](/api/network#from-nx) automatically enables directed mode. Two-way streets are represented as two reciprocal edges (A&#x2192;B and B&#x2192;A) while one-way streets have a single directed edge. This is the format produced by [OSMnx](https://osmnx.readthedocs.io/).

```python
import networkx as nx
from shapely.geometry import LineString

G = nx.MultiDiGraph(crs="EPSG:32630")
G.add_node("A", x=0.0, y=0.0)
G.add_node("B", x=100.0, y=0.0)
G.add_node("C", x=200.0, y=0.0)
# One-way: A -> B only
G.add_edge("A", "B", key=0, geom=LineString([(0, 0), (100, 0)]))
# Two-way: B <-> C (reciprocal edges)
G.add_edge("B", "C", key=0, geom=LineString([(100, 0), (200, 0)]))
G.add_edge("C", "B", key=0, geom=LineString([(200, 0), (100, 0)]))

cn = CityNetwork.from_nx(G)
assert cn.is_directed
```

### From OpenStreetMap (via OSMnx)

The built-in `from_osm` constructor uses an undirected simplification pipeline, so it always produces undirected networks. For directed routing with OSM data, fetch a directed `MultiDiGraph` via [OSMnx](https://osmnx.readthedocs.io/) and convert it with [`io.nx_from_osm_nx`](/tools/io#nx-from-osm-nx), then pass it to `from_nx`:

```python
import osmnx as ox
from cityseer.tools import io

# Fetch directed graph from OSMnx
G_osmnx = ox.graph_from_polygon(polygon, network_type="drive")
G_osmnx = ox.projection.project_graph(G_osmnx, to_crs="EPSG:32630")
# Convert to cityseer-compatible MultiDiGraph (preserving direction)
G_cityseer = io.nx_from_osm_nx(G_osmnx, directed=True)
# Build directed CityNetwork
cn = CityNetwork.from_nx(G_cityseer)
```

### Centrality semantics

On directed networks, centrality metrics take on directional interpretations:

- **Closeness** (harmonic, farness, etc.) becomes *out-closeness*: how efficiently each node can reach other nodes following the allowed traffic directions.
- **Betweenness** uses directed shortest paths, counting only paths that respect one-way constraints.
- **Accessibility and mixed-use** metrics respect direction when computing distances over the network. Data points are still assigned to the nearest street segment regardless of direction (the physical location of a shop does not depend on traffic flow).

## Elevation and Slope

`cityseer` supports optional z (elevation) coordinates on network nodes. When elevation data is available, it is preserved throughout the full processing chain: graph construction, decomposition, consolidation, merging, dual graph conversion, CRS reprojection, and round-trip serialisation between `networkX`, `GeoDataFrames`, and the Rust `NetworkStructure`.

When both endpoint nodes of an edge have z coordinates, `cityseer` automatically applies a slope-based walking impedance during shortest-path and simplest-path computations, using [Tobler's hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) (Tobler, 1993). This adjusts the effective traversal cost of each edge based on the gradient:

- **Uphill** segments incur a penalty proportional to the grade (e.g. a 20% slope approximately doubles the effective distance).
- **Steep downhill** segments are also penalised, reflecting the reduced walking speed on steep descents.
- **Gentle downhill** slopes (~3%) receive a slight bonus, matching the empirically observed optimal walking gradient.
- **Flat terrain** incurs no penalty (multiplier of 1.0).

The slope penalty is computed dynamically and directionally during graph traversal, so the cost of walking uphill from A to B differs from the cost of walking downhill from B to A. This operates independently of the configured walking speed: the penalty is a dimensionless multiplier on effective distance, meaning it composes correctly regardless of whether the walking speed is set to 1.4 m/s or any other value.

For simplest-path (angular) analysis, the slope penalty affects only the time budget (reachability cutoff), not the angular routing metric itself. This means the cognitively simplest path is still selected, but steep terrain reduces the distance a pedestrian can cover within the analysis threshold.

When z coordinates are not present, all slope penalties default to 1.0 (no effect), ensuring full backward compatibility with existing 2D workflows.

## Column Naming Conventions

All computed metrics are written to columns on the `nodes_gdf` GeoDataFrame. Column names follow a consistent pattern:

```text
cc_{metric}_{distance}        — shortest-path metric
cc_{metric}_{distance}_ang    — simplest-path (angular) metric
```

The `cc_` prefix identifies columns generated by `cityseer`. The `_ang` suffix is appended when `angular=True` (simplest-path heuristic). Examples:

```text
cc_harmonic_800         — harmonic closeness at 800m (shortest path)
cc_betweenness_800_ang  — betweenness at 800m (angular / simplest path)
cc_hill_q0_400          — Hill diversity q=0 at 400m
cc_shop_200             — accessibility count for "shop" at 200m
cc_price_mean_1200      — mean of "price" column at 1200m
```

### Centrality output columns

[`node_centrality_shortest`](/metrics/networks#node-centrality-shortest) produces the following columns for each distance threshold:

| Column | Description |
| --- | --- |
| `cc_density_{d}` | Count of reachable nodes within distance `d`. |
| `cc_harmonic_{d}` | Harmonic closeness: sum of inverse distances to reachable nodes. |
| `cc_farness_{d}` | Farness: sum of distances to reachable nodes. |
| `cc_hillier_{d}` | Hillier normalisation: `density² / farness`. |
| `cc_cycles_{d}` | Circuit rank of the locally reachable subgraph (meshedness). |
| `cc_decay_{d}` | Decay-weighted closeness using the `decay_fn` expression. |
| `cc_betweenness_{d}` | Betweenness centrality (shortest-path). |
| `cc_betweenness_decay_{d}` | Decay-weighted betweenness using the `decay_fn` expression. |

[`node_centrality_simplest`](/metrics/networks#node-centrality-simplest) produces the following columns (note the `_ang` suffix):

| Column | Description |
| --- | --- |
| `cc_density_{d}_ang` | Count of reachable nodes within distance `d` (angular routing). |
| `cc_harmonic_{d}_ang` | Harmonic closeness (cumulative angular change as impedance). |
| `cc_farness_{d}_ang` | Farness (sum of cumulative angular changes to reachable nodes). |
| `cc_hillier_{d}_ang` | Hillier normalisation: `density² / farness`. |
| `cc_betweenness_{d}_ang` | Betweenness centrality (simplest angular paths). |

### Accessibility output columns

[`compute_accessibilities`](/metrics/layers#compute-accessibilities) produces columns for each land-use key and distance:

| Column | Description |
| --- | --- |
| `cc_{key}_{d}` | Count of reachable instances of land-use `key` within distance `d`, weighted by `decay_fn`. |
| `cc_{key}_nearest_max_{d}` | Shortest network distance to the nearest instance (only at the maximum distance). |

### Mixed-use output columns

[`compute_mixed_uses`](/metrics/layers#compute-mixed-uses) produces columns for each distance:

| Column | Description |
| --- | --- |
| `cc_hill_q0_{d}` | Hill diversity at q=0 (richness: count of distinct land-uses). |
| `cc_hill_q1_{d}` | Hill diversity at q=1 (exponential Shannon entropy). |
| `cc_hill_q2_{d}` | Hill diversity at q=2 (inverse Simpson concentration). |
| `cc_shannon_{d}` | Shannon entropy (if `compute_shannon=True`). |
| `cc_gini_{d}` | Gini-Simpson index (if `compute_gini=True`). |

### Statistical aggregation output columns

[`compute_stats`](/metrics/layers#compute-stats) produces columns for each `stats_column_label` and distance:

| Column | Description |
| --- | --- |
| `cc_{label}_sum_{d}` | Sum of values, weighted by `decay_fn`. |
| `cc_{label}_mean_{d}` | Weighted mean. |
| `cc_{label}_count_{d}` | Weighted count. |
| `cc_{label}_median_{d}` | Weighted median. |
| `cc_{label}_var_{d}` | Weighted variance. |
| `cc_{label}_mad_{d}` | Weighted median absolute deviation. |
| `cc_{label}_max_{d}` | Maximum value within distance. |
| `cc_{label}_min_{d}` | Minimum value within distance. |

### Working with output columns

```python
from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, io

# Prepare network
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)

# Compute centrality at multiple distances
nodes_gdf = networks.node_centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[400, 800, 1600],
)

# Access individual columns
print(nodes_gdf["cc_harmonic_800"])
print(nodes_gdf["cc_betweenness_1600"])

# Select all cityseer columns
cc_cols = [c for c in nodes_gdf.columns if c.startswith("cc_")]

# Select all columns for a specific distance
cols_800 = [c for c in nodes_gdf.columns if c.endswith("_800")]

# Select all betweenness columns across distances
bt_cols = [c for c in nodes_gdf.columns if "betweenness" in c]
```

The broader emphasis on localised methods and how `cityseer` addresses these is broached in the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827). `cityseer` includes a variety of convenience methods for the general preparation of networks and their conversion into (and out of) the lower-level data structures used by the underlying algorithms. These graph utility methods are designed to work with `NetworkX` to facilitate ease of use. A complement of code tests has been developed to maintain the codebase's integrity through general package maintenance and upgrade cycles. Shortest-path algorithms, harmonic closeness, and betweenness algorithms are tested against `NetworkX`. Mock data and test plots have been used to visually confirm the intended behaviour for divergent simplest and shortest-path heuristics and for testing data assignment to network nodes given various scenarios.

The best way to get started is to see the [`Cityseer Examples`](https://benchmark-urbanism.github.io/cityseer-examples/) site, which contains a number of recipes for a variety of use-cases.

## QGIS Plugin

A [QGIS plugin](/plugin) is available for computing localised network centrality metrics directly within QGIS without writing code. See the [plugin page](/plugin) for installation and usage instructions.

## Support

Please report bugs to the [github issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues) and direct more general questions to [Github Discussions](https://github.com/benchmark-urbanism/cityseer-api/discussions).

Time permitting, for general help with workflows or feedback in support of research projects or papers, please start a new [discussion on Github](https://github.com/benchmark-urbanism/cityseer-api/discussions).

## Attribution

Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package for producing research.

`cityseer` is licensed as AGPLv3. Please [get in touch](mailto:info@benchmarkurbanism.com) if you need technical support developing related workflows, or if you wish to sponsor the development of additional or bespoke functionality.

If using the package to produce visual plots and outputs, please display the cityseer logo and a link to the documentation website.

<img src="/logos/cityseer_logo_white.png" alt="Cityseer white logo." width="350"></img>

<img src="/logos/cityseer_logo_light_red.png" alt="Cityseer red logo." width="350"></img>
