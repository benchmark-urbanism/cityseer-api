# v4.24.0 Release Notes

## New Features

### Z-aware networks (elevation and slope)

Network nodes now support an optional `z` attribute for elevation. When both endpoints of an edge have z coordinates, a slope-based walking impedance using Tobler's hiking function is automatically applied during shortest-path and simplest-path computations. Uphill segments incur a penalty proportional to grade; steep downhill segments are also penalised; gentle downhill slopes receive a slight bonus. The penalty is directional (A→B differs from B→A) and composes with the configured walking speed.

Z coordinates are preserved through the full processing chain: graph construction, decomposition, consolidation, merging, dual graph conversion, CRS reprojection, and round-trip serialisation. When z is absent, behaviour is identical to previous versions.

Supported in all IO methods: `nx_from_osm_nx`, `nx_from_open_roads`, `nx_from_generic_geopandas`, `nx_from_cityseer_geopandas`, `network_structure_from_nx`, and `network_structure_from_gpd`.

### Adaptive sampling (experimental)

`node_centrality_shortest` and `node_centrality_simplest` accept `sample=True` to use distance-based Hoeffding/Eppstein-Wang sampling for approximate centrality, achieving 2-3x speedup while maintaining ρ ≥ 0.95. Sampling probability is derived deterministically from each distance threshold using a canonical grid network model.

### QGIS plugin updates

New accessibility and statistics processing algorithms. Expanded centrality algorithm with sampling support.

## Breaking Changes

### Angular (simplest-path) analysis now requires a dual graph

`node_centrality_simplest` (and the convenience wrappers `closeness_simplest`, `betweenness_simplest`) now raises `ValueError` if the input `NetworkStructure` was not ingested from a dual graph. Angular routing uses endpoint-aware dual-graph traversal instead of the previous bearing-based angular costs. Convert primal graphs with `graphs.nx_to_dual()` before calling `network_structure_from_nx()`.

### `tolerance` parameter semantics changed

The `tolerance` parameter on `node_centrality_shortest`, `node_centrality_simplest`, `betweenness_shortest`, `betweenness_simplest`, and `betweenness_od` now uses **relative percentage** semantics (e.g. `1.0` = 1%) instead of the previous absolute fraction. The default changed from `0.0` to `None`. A tiny internal epsilon is always enforced for floating-point stability. To migrate: multiply old values by 100 (e.g. old `0.05` → new `5.0`).

### `tolerance` parameter reordered in `node_centrality_simplest`

`tolerance` now appears before `angular_scaling_unit` and `farness_scaling_offset`. Code using positional arguments for these parameters will need updating.

### `betweenness_beta` removed from angular (simplest) results

`CentralitySimplestResult` no longer exposes `node_betweenness_beta`. The `node_centrality_simplest` function no longer writes `cc_betweenness_beta_*` columns. Only `cc_betweenness_*` columns are produced.

### `cycles` metric changed

The `cycles` output from `node_centrality_shortest` now measures the **circuit rank** of the locally reachable subgraph (m − n + c), providing a more stable measure of network meshedness than the older tree-cycle heuristic.

### Sampling functions moved from `config` to `sampling` module

`compute_distance_p`, `compute_hoeffding_p`, `HOEFFDING_EPSILON`, `HOEFFDING_DELTA`, and `GRID_SPACING` have moved from `cityseer.config` to `cityseer.sampling`. The `config` module is still importable via lazy-loading but no longer contains sampling functions. Update imports accordingly.

## Other Changes

- All result arrays (`CentralityShortestResult`, `CentralitySimplestResult`, `CentralitySegmentResult`, `Stats`, etc.) now return `np.float64` instead of `np.float32`.
- `betweenness_od` now accepts an optional `tolerance` parameter.
- `closeness_shortest` and `closeness_simplest` now accept an optional `tolerance` parameter.
- Bug fix: `is_dual` graph attribute was incorrectly cast via `CRS()` instead of `bool()` in `nx_remove_dangling_nodes` and `nx_merge_parallel_edges`.
- `NetworkStructure` now tracks `is_dual` explicitly and exposes `node_zs`, `node_xyzs`, and `coord_z` properties.
- Dual graph edges now pass `shared_primal_node_key` for endpoint-aware angular transitions.
- `measure_bearing` in `tools.util` now unpacks `x, y` in the correct order (was previously reversed but functionally equivalent due to symmetric usage).
