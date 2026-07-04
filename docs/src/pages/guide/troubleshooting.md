---
layout: '@src/layouts/PageLayout.astro'
---

# Troubleshooting

Common problems, organised by symptom. Each entry gives the cause, the fix, and a link to the guide section that explains the underlying behaviour.

## All my distances or results look wrong by orders of magnitude

Your data is in a geographic CRS, so coordinates are degrees rather than metres and every distance calculation is off by roughly five orders of magnitude. `cityseer` requires a projected CRS with metre units.

Check with `gdf.crs` (EPSG:4326 / WGS 84 is the usual culprit) or `gdf.crs.is_projected`, then reproject before passing data to `cityseer`:

```python
gdf = gdf.to_crs(epsg=32630)  # a UTM zone or regional projection with metre units
```

Search your city at [epsg.io](https://epsg.io) to find a suitable UTM zone, or use a regional projection such as EPSG:3035 (Europe), EPSG:27700 (Great Britain), or EPSG:2154 (France). When building from OSM with `CityNetwork.from_osm` and related `io` functions, `cityseer` selects an appropriate UTM zone automatically; the pitfall arises mainly when loading networks or data layers from files.

## Values fade toward the edge of my study area

Nodes near the boundary can only "see" part of their real catchment, so their values roll off artificially. This is edge rolloff, explained at [Edge Rolloff](/guide/fundamentals#edge-rolloff).

The fix is to buffer: download or build a network extending at least the maximum distance threshold beyond your study area (data layers should cover the buffered extent too), then restrict reporting to the study area itself. With `CityNetwork`, pass the unbuffered boundary to `set_boundary`; with the lower-level API, set buffer nodes to `live=False`:

```python
cn.set_boundary(study_area_polygon)  # buffer nodes participate in routing but are not reported
```

See the [live nodes recipe](/examples/recipes/live-nodes) for a worked example.

## Computation is slow

Runtime scales with the number of edges, the number of distance thresholds, and the reachability at each threshold, so the largest threshold usually dominates. Three things to check:

- **Distance thresholds.** Each threshold adds cost, and long thresholds cost disproportionately more because far more of the network is reachable from each node. Drop thresholds you do not need.
- **Decomposition.** Decomposing a network multiplies its node count, and computation scales with it; a fine decomposition of a large extent can grow the network by an order of magnitude. Decompose district-scale extents rather than whole cities, and prefer coarser maxima. See [Decomposition](/guide/networks#decomposition).
- **Sampling.** For large networks at long thresholds, pass `sample=True` to the centrality methods. Adaptive sampling estimates results from a subset of source nodes with controlled error, and automatically falls back to exact computation where sampling would not pay. See [Adaptive Sampling](/guide/centrality#adaptive-sampling).

As a rough calibration: the algorithms are parallelised in Rust, and a district-scale run such as the [quickstart](/examples/recipes/quickstart), a 1km-radius OSM network with thresholds up to 2000m, completes in seconds on an ordinary laptop. If a run of that size takes many minutes, one of the points above is usually the reason.

## OSM download fails or times out

`CityNetwork.from_osm` and `io.osm_graph_from_poly` fetch data from the Overpass API, a shared public service that rate-limits and sometimes rejects or delays requests. `cityseer` retries failed requests automatically (three attempts with a 300s timeout by default; both are configurable on the lower-level `io` functions), so transient failures often resolve themselves.

If requests are rejected outright, check your `cityseer` version: from v5, requests send an identifying `User-Agent`, which Overpass now requires; older versions receive `406 Not Acceptable` errors. Upgrade to fix.

Above all, avoid re-downloading on every run. Fetch once, compute, and save; subsequent runs load from disk with no network access:

```python
cn.save("my_network")             # writes my_network.nodes.parquet + my_network.state.pkl
cn = CityNetwork.load("my_network")
```

## My data points don't appear in results

Data points are assigned to the nearest street edge before aggregation, and points further than the maximum assignment distance (default 100m) from any edge are silently excluded. Points outside the network's extent entirely can never be reached and contribute nothing.

The land-use methods return the data `GeoDataFrame` with `nearest_assigned` and `next_nearest_assign` columns; rows with no assignment are the excluded points. If legitimate points are being dropped, for example building centroids set back from the street, raise the assignment distance:

```python
cn, data_gdf = cn.compute_accessibilities(data_gdf=data_gdf, ..., max_netw_assign_dist=400)
```

Also confirm the data layer covers the buffered network extent (see [Edge Rolloff](/guide/fundamentals#edge-rolloff)) and shares the network's projected CRS.

## My accessibility values don't reflect proximity

With `CityNetwork`, accessibility counts are plain (unweighted) by default: a shop 50m away counts the same as one 750m away. Pass a `decay_fn` expression so that nearer features contribute more; the lower-level `layers` functions instead default to emitting both variants, suffixed `_nw` (non-weighted) and `_wt` (decay-weighted). Use a decay when proximity matters (walkability studies); use a plain count when presence alone matters (counting all parks within reach). See [Decay Functions](/guide/land-use#decay-functions).

## Angular centrality raises an error

Simplest-path (angular) analysis requires a dual graph, where street segments are nodes and turning angles can be tracked between them. Calling an angular function on a primal graph raises an error stating that a dual graph is required.

`CityNetwork` builds the dual graph automatically, so this error only arises with the lower-level API: convert the primal graph before ingestion:

```python
G_dual = graphs.nx_to_dual(G)  # then network_structure_from_nx(G_dual, ...)
```

See [Primal and dual graphs](/guide/networks#primal-and-dual-graphs) and the [dual graph recipe](/examples/networks/create-dual-graph).
