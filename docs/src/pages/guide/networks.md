---
layout: '@src/layouts/PageLayout.astro'
---

# Networks

How `cityseer` represents street networks: the primal and dual graph forms, one-way streets, elevation, and per-edge impedances. For building networks from data sources, see the [network preparation recipes](/examples/networks).

## Primal and dual graphs

Street networks can be represented in two complementary forms:

- **Primal graph**: Intersections are nodes; streets connecting them are edges. This is the conventional representation used by most network analysis tools.
- **Dual graph**: Streets (segments) become nodes; edges connect pairs of streets that meet at a common intersection. Each dual node is positioned at the midpoint of its corresponding street segment, so metrics are measured per street segment and can be visualised directly on street geometries.

The dual representation is needed for angular (simplest-path) analysis because each street segment becomes an explicit node in the graph, allowing the cumulative turning angle (the sum of directional changes at each junction along a route) to be tracked explicitly. It also produces more intuitive outputs: a map coloured by betweenness shows which streets carry the most through-movement, rather than which intersections do.

The [`CityNetwork`](/api/network) class builds the dual graph automatically from input geometries; there is no need to call conversion functions manually. When using the lower-level API, convert a primal graph with [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) before computing angular centralities.

See the [Create Dual Graph](/examples/networks/create-dual-graph) example for a visual comparison.

## Decomposition

Decomposition splits long street segments into shorter pieces of uniform maximum length using [`nx_decompose`](/tools/graphs#nx-decompose). Because metrics are computed at network nodes, node spacing determines spatial resolution: decomposing to, say, 50m means land-use accessibilities and statistical aggregations are sampled at regular intervals along street fronts rather than only at intersections, and it evens out node spacing so that long streets do not exert disproportionate influence on node-based centrality.

Decomposition multiplies the node count, and computation scales with it. A 50m decomposition of a large city network can grow it by an order of magnitude, so decompose district-scale extents rather than whole cities, and prefer coarser maxima (50-100m) unless the analysis genuinely needs finer sampling. Decomposition is part of the lower-level workflow (`tools.graphs.nx_decompose` before [`network_structure_from_nx`](/tools/io#network-structure-from-nx)); the high-level `CityNetwork` API does not decompose. For worked examples, see the [statistics recipes](/examples/stats), which use decomposition for street-front resolution.

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
Graph cleaning functions in the [`graphs`](/tools/graphs) module do not preserve edge directionality. Pass directed graphs directly to `CityNetwork.from_nx` or `CityNetwork.from_geopandas(directed=True)`. For a worked example, see the [Directed Networks recipe](/examples/networks/directed-networks).
:::

## Elevation and Slope

`cityseer` supports optional z (elevation) coordinates on network geometries. When present, elevation data is preserved through all processing steps (graph cleaning, decomposition, dual graph conversion, CRS reprojection, and serialisation). [Tobler's hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) automatically adjusts traversal costs based on the gradient: uphill segments incur a penalty proportional to the grade (for example, a 20% slope approximately doubles the effective distance), steep downhill segments are also penalised due to reduced walking speed, and gentle downhill slopes (~3%) receive a slight bonus matching the empirically observed optimal walking gradient. The penalty is a dimensionless multiplier on effective distance, computed directionally (A to B differs from B to A) and composing correctly with any configured walking speed.

For angular analysis, the slope penalty affects only the reachability budget (the distance a pedestrian can cover within the analysis threshold), not the angular routing metric itself. The cognitively simplest path is still selected, but steep terrain reduces how far the walker can reach.

When z coordinates are absent, all slope penalties default to 1.0 (no effect). To add elevation to a 2D network, drape it onto a digital elevation model (DEM) using a tool such as the [OSMnx elevation module](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.elevation) or [rasterio](https://rasterio.readthedocs.io/).

See the [3D Elevation](/examples/centrality/3d-elevation) example.

## Edge Impedance

Each network edge carries an `imp_factor` (default `1.0`) that multiplicatively scales its effective traversal cost, which is useful for representing road surface, road class, or any other static per-segment penalty. Impedance composes with the slope penalty above: the effective edge cost is `length × imp_factor × slope_pen`, so both factors apply together.

Set per-edge impedance by including an `imp_factor` column on the `GeoDataFrame` passed to `CityNetwork.from_geopandas`, an `imp_factor` attribute on edges of a `NetworkX` graph passed to `CityNetwork.from_nx`, or via the `impedances={fid: value}` keyword on `CityNetwork.from_wkts`. The value is propagated through dual graph construction: each dual edge, which traverses half of each adjacent primal segment, receives the **length-weighted mean** of its two primal impedances, so an all-`1.0` primal yields `1.0` on the dual (fully backwards compatible).

Impedance applies to **shortest-path** routing (and any equivalent time-converted budget), including the reachability budget used by simplest-path (angular) analysis. The angular cost itself is purely geometric (cumulative turning) and is **not** scaled by `imp_factor`; only how far an angular traversal can reach within the time budget.
