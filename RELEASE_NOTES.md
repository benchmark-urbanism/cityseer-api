# v5.4.1 Release Notes

A feature release adding Python 3.14 support. No breaking API changes. (5.4.0 was not published; it is superseded by 5.4.1.)

- **Python 3.14 support**: cityseer now builds and is tested on Python 3.10 through 3.14.
- **Fiona replaced by pyogrio**: the OS Open Roads reader (`io.nx_from_open_roads`) now reads through pyogrio, geopandas' default I/O engine, instead of Fiona. Fiona had stalled at 1.10.1 (September 2024) with no Python 3.14 wheels and is no longer a dependency.
- **rasterio**: the pinned `==1.4.3` (kept for older Intel Macs) is lifted to `>=1.4.4`, the first release with Python 3.14 wheels. Support for older Intel Macs is dropped.
- **Slimmer wheels**: PyPy and musllinux (Alpine) wheels are no longer built. They are near-unusable for cityseer because the GDAL-based dependencies rarely ship matching wheels. Install on those platforms from the source distribution instead.

# v5.3.1 Release Notes

A bugfix release. No breaking changes to the Python API.

- **Land-use aggregations skip uncategorised points**: `compute_accessibilities` and `compute_mixed_uses` now exclude data points whose land-use category is missing (`NaN`) instead of raising a `TypeError` at the Rust boundary. Uncategorised points belong to no land use, so they no longer count toward accessibility totals or register as a distinct class in mixed-use diversity, and the number excluded is logged. Resolves the freeze reported in [#146](https://github.com/benchmark-urbanism/cityseer-api/issues/146), which the 5.x rewrite had already turned from a hang into a hard error.

# v5.3.0 Release Notes

A QGIS plugin and documentation release. No breaking changes to the Python API.

- **QGIS plugin overhaul**: compatibility with the QGIS 4.2 processing GUI; two new algorithms, Demand Betweenness (OD Flow) and Mixed Uses (Hill, Shannon, Gini-Simpson); Statistics accepts multiple numerical fields in one pass and computes only the selected measures; advanced distance-decay expressions on Accessibility, Statistics, and Mixed Uses; advanced time thresholds (minutes with a walking speed) on all algorithms; adaptive per-node sampling replaces the retired deterministic distance schedule (off by default, with an epsilon control).
- **Documentation**: the QGIS plugin gains its own site section with a page per algorithm; a "How do I...?" question-and-answer guide; `llms.txt` for AI tools and a sitemap reference in `robots.txt`; lighter navigation typography.

# v5.2.0 Release Notes

A small feature release. No breaking changes.

- **`CityNetwork.betweenness_demand`**: demand-weighted (flow) betweenness is now a method on `CityNetwork`, matching `betweenness_od`. It runs the singly-constrained spatial interaction model on the network and writes a `cc_demand_{distance}` column, which `to_geopandas` projects onto the street segments. Previously this was reachable only through the lower-level functional API.
- **Flow recipes render on streets**: the demand-flow and directed-network examples now project their values onto the street segments (line width by intensity) instead of plotting node midpoints as points.

# v5.1.0 Release Notes

A documentation and usability release on top of v5.0.0. No breaking changes.

- **Origin-Destination Flows**: a new guide section, plus recipes covering explicit OD-matrix routing (`build_od_matrix` with `betweenness_od`) and demand-modelled flows (`betweenness_demand`).
- **Network cleaning guide**: a dedicated section on automated versus configurable cleaning, with references.
- **Configurable Overpass endpoint**: `fetch_osm_network` and `osm_graph_from_poly` accept an `overpass_url` argument, or the `CITYSEER_OVERPASS_URL` environment variable, plus an optional `cache_path` for offline and repeatable OSM builds.
- **Restyled example plots**: uniform colour with line width scaled by intensity, no colour bars.

# v5.0.0 Release Notes

## Headline

v5 is a major release centred on a new API: the high-level **`CityNetwork`** class becomes the primary interface, the fixed centrality metric set is replaced by an **expression-based API**, **betweenness** is redefined to count all routes (including routes that both start and end outside the boundary), and **sampling** becomes per-node adaptive, validated on four real networks. The documentation site has been rebuilt around the new API, with all examples and tutorials using `CityNetwork` throughout. Backwards compatibility is a first-class concern: the 4.x function names remain available as deprecated shims that produce the same default columns as before; see `COMPATIBILITY.md` for the full contract and migration table.

## New Features

### Expression-based centrality

`centrality_shortest` and `centrality_simplest` now accept arbitrary metric expressions instead of a fixed metric set. Closeness and betweenness are dicts of `{label: expression}` over the variables `c` (network cost) and `p` (normalised progress, `c / threshold`), and `decay_fn` supplies distance weighting:

```python
net.centrality_shortest(
    distances=[800, 2000],
    closeness={"harmonic": "1/c", "gravity": "exp(-0.005 * c)"},
    betweenness={"betweenness": "1"},
    decay_fn="exp(-4 * p)",
)
```

Each label becomes a `cc_{label}_{distance}` column. The default expressions reproduce the classic 4.x metrics. Land-use aggregations gain per-label decay functions on the same principle (#175).

### `CityNetwork` high-level API

A new `CityNetwork` class wraps graph preparation and metrics behind a lean interface. Its defaults are intentionally minimal (a single harmonic closeness, a single betweenness, cycles off) and any keyword can be overridden. The classic functional API is unchanged and remains the compatibility surface.

### Directed networks

One-way street routing via directed graphs (#173). Directed betweenness counts each ordered pair fully (see Breaking Changes).

### Demand-weighted betweenness

`betweenness_demand` computes spatial-interaction (origin–destination weighted) flow betweenness (#176).

### Sampling is now per-node adaptive (experimental)

`sample=True` now measures each node's local reach with a cheap pilot (a small number of
bounded shortest-path traversals polling the network) and assigns each node its own source
inclusion probability via the Hoeffding bound. Sparse areas
sample more heavily and dense areas less, so precision is uniform across the network, and
per-source inverse-probability weighting keeps estimates unbiased. A per-distance work test
selects exact computation wherever powered sampling would not be cheaper. The default
tolerance is `epsilon=0.05` (was 0.06), calibrated on real-world networks spanning the urban density
range, one held out from calibration; all pass Spearman rho >= 0.95 at 1–20 km under
this method. The previous
distance-only schedule remains available as a reference model in `cityseer.sampling`
(`compute_distance_p`) but is no longer used by the runtime. Sampling remains opt-in via
`sample=True` and experimental.

### Statistics measure selection

`compute_stats` accepts `measures=[...]` (e.g. `["mean", "count"]`) to compute only the statistics you need.

### Other improvements

- Impedances propagate through dual-graph construction.
- Improved NetworkX round-trip for momepy interoperability.
- Segment-length weighting via `segment_weighted=True` on the centrality functions.
- Expression evaluation uses `exmex` (replacing the unmaintained `meval`), with `ln`, `log10`, `abs`, and `round` available in expressions.

## Breaking Changes

### Betweenness counts all routes

Betweenness now sources from **every** node; the `live` designation only filters which nodes' values are reported. This is a deliberate change of definition, not a bug fix. Previously, routes that both started **and** ended outside the boundary were intentionally excluded, a pragmatic scoping choice, since such routes are relatively uncommon yet require every buffer node to run as a source, adding substantial computational weight. In v5 we opt for theoretical strictness: with a buffer at least as deep as the analysis distance, a route between two buffer nodes that passes through the study area is a real shortest path, so it is now counted and credits the live nodes it traverses. These buffer-to-buffer routes are the *only* difference. Consequences: on undirected, fully-live networks values are unchanged; on buffered networks, values near the boundary increase (the newly counted routes accrue there), and exact betweenness costs more to compute since all nodes source, which is part of the motivation for sampling. Separately, on directed networks each ordered origin–destination pair now contributes with weight 1, where previously the undirected pair-weighting of ½ was applied; directed values are therefore twice their 4.x magnitude.

### `segment_centrality` removed

The continuous-segment engine is gone. Calling `networks.segment_centrality` raises `NotImplementedError` with migration guidance; use node-based centrality with `segment_weighted=True` for segment-length weighting.

### Removed parameters

`betas=` and `spatial_tolerance=` are removed and raise `TypeError`. Replace `betas` with the equivalent expression: `decay_fn="exp(-beta * c)"`. `source_indices` is also removed. The full removed-parameter table with migrations is in `COMPATIBILITY.md`.

### Low-level API

The Rust-level result and function signatures changed with the expression API. The low-level surface (`rustalgos`) does not carry a compatibility guarantee; the high-level `cityseer.metrics` functions do.

## Backwards Compatibility

- `node_centrality_shortest` and `node_centrality_simplest` remain available as deprecated shims producing the **same default columns and values** as 4.x (`cc_beta_*`, `cc_harmonic_*`, `cc_betweenness_*`, `cc_betweenness_beta_*`, `cc_cycles_*`, ...). They emit `DeprecationWarning` and will be removed in a future major release.
- Land-use and statistics functions keep the classic paired `_nw`/`_wt` default columns.
- Columns prefixed `cc_` are managed by cityseer and are overwritten in place when a metric is recomputed for the same distance.
- See `COMPATIBILITY.md` for the two-surface policy (classic functional API vs lean `CityNetwork`), the removed-parameter table, and the deprecation timeline.

## Fixes

- OSM fetching sends an identifying `User-Agent`: the Overpass API now rejects generic python user agents with `406 Not Acceptable`, which broke `osm_graph_from_poly` and everything built on it.
- QGIS plugin: version check no longer reports a false mismatch between the plugin's `beta` spelling and pip's normalised `b` spelling.
- `nx_from_osm_nx`: fixed a key-lookup bug after node key stringification.
- Fixed node-weight semantics in centrality aggregation.
- Docs generator renders deprecation notices instead of crashing (CI fix).
- Release workflows serialised to avoid a tag race in `action-gh-release`.

# v4.24.0 Release Notes

## New Features

### Z-aware networks (elevation and slope)

Network nodes now support an optional `z` attribute for elevation. When both endpoints of an edge have z coordinates, a slope-based walking impedance using Tobler's hiking function is automatically applied during shortest-path and simplest-path computations. Uphill segments incur a penalty proportional to grade; steep downhill segments are also penalised; gentle downhill slopes receive a slight bonus. The penalty is directional (A→B differs from B→A) and composes with the configured walking speed.

Z coordinates are preserved through the full processing chain: graph construction, decomposition, consolidation, merging, dual graph conversion, CRS reprojection, and round-trip serialisation. When z is absent, behaviour is identical to previous versions.

Supported in all IO methods: `nx_from_osm_nx`, `nx_from_open_roads`, `nx_from_generic_geopandas`, `nx_from_cityseer_geopandas`, `network_structure_from_nx`, and `network_structure_from_gpd`.

### Adaptive sampling (experimental)

`centrality_shortest` and `centrality_simplest` accept `sample=True` to use distance-based Hoeffding/Eppstein-Wang sampling for approximate centrality, achieving 2-3x speedup while maintaining ρ ≥ 0.95. Sampling probability is derived deterministically from each distance threshold using a canonical grid network model.

### QGIS plugin updates

New accessibility and statistics processing algorithms. Expanded centrality algorithm with sampling support.

## Breaking Changes

### Angular (simplest-path) analysis now requires a dual graph

`centrality_simplest` (and the convenience wrappers `closeness_simplest`, `betweenness_simplest`) now raises `ValueError` if the input `NetworkStructure` was not ingested from a dual graph. Angular routing uses endpoint-aware dual-graph traversal instead of the previous bearing-based angular costs. Convert primal graphs with `graphs.nx_to_dual()` before calling `network_structure_from_nx()`.

### `tolerance` parameter semantics changed

The `tolerance` parameter on `centrality_shortest`, `centrality_simplest`, `betweenness_shortest`, `betweenness_simplest`, and `betweenness_od` now uses **relative percentage** semantics (e.g. `1.0` = 1%) instead of the previous absolute fraction. The default changed from `0.0` to `None`. A tiny internal epsilon is always enforced for floating-point stability. To migrate: multiply old values by 100 (e.g. old `0.05` → new `5.0`).

### `tolerance` parameter reordered in `centrality_simplest`

`tolerance` now appears before `angular_scaling_unit` and `farness_scaling_offset`. Code using positional arguments for these parameters will need updating.

### `betweenness_beta` removed from angular (simplest) results

`CentralitySimplestResult` no longer exposes `node_betweenness_beta`. The `centrality_simplest` function no longer writes `cc_betweenness_beta_*` columns. Only `cc_betweenness_*` columns are produced.

### `cycles` metric changed

The `cycles` output from `centrality_shortest` now measures the **circuit rank** of the locally reachable subgraph (m − n + c), providing a more stable measure of network meshedness than the older tree-cycle heuristic.

### Sampling functions moved from `config` to `sampling` module

`compute_distance_p`, `compute_hoeffding_p`, `HOEFFDING_EPSILON`, `HOEFFDING_DELTA`, and `GRID_SPACING` have moved from `cityseer.config` to `cityseer.sampling`. The `config` module is still importable via lazy-loading but no longer contains sampling functions. Update imports accordingly.

## Other Changes

- All result arrays (`CentralityShortestResult`, `CentralitySimplestResult`, `Stats`, etc.) now return `np.float64` instead of `np.float32`.
- `betweenness_od` now accepts an optional `tolerance` parameter.
- `closeness_shortest` and `closeness_simplest` now accept an optional `tolerance` parameter.
- Bug fix: `is_dual` graph attribute was incorrectly cast via `CRS()` instead of `bool()` in `nx_remove_dangling_nodes` and `nx_merge_parallel_edges`.
- `NetworkStructure` now tracks `is_dual` explicitly and exposes `node_zs`, `node_xyzs`, and `coord_z` properties.
- Dual graph edges now pass `shared_primal_node_key` for endpoint-aware angular transitions.
- `measure_bearing` in `tools.util` now unpacks `x, y` in the correct order (was previously reversed but functionally equivalent due to symmetric usage).
