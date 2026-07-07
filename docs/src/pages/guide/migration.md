---
layout: '@src/layouts/PageLayout.astro'
---

# Migrating from v4 to v5

v5 is a major release centred on a new API. The high-level [`CityNetwork`](/api/network) class becomes the primary interface, the fixed centrality metric set is replaced by [expression-based metrics](/guide/centrality#expression-based-metrics), and betweenness is redefined to count all routes. This page covers what breaks, what is preserved, and how to move existing code. The full compatibility contract lives in [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md).

## The short version

Existing scripts using the everyday arguments keep working: the 4.x function names remain available as deprecated shims that produce the same default columns and the same numbers as before. Removed parameters raise a `TypeError` that points to this guide; they are not silently reinterpreted to produce wrong numbers. New code should start with `CityNetwork`.

## Recommended: move to CityNetwork

The v4 workflow built a graph, converted it, and called metric functions on the pieces. The v5 workflow wraps all of that:

```python
# v4
from cityseer.tools import graphs, io
from cityseer.metrics import networks

G = io.osm_graph_from_poly(poly)
G_dual = graphs.nx_to_dual(G)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G_dual)
nodes_gdf = networks.node_centrality_shortest(network_structure, nodes_gdf, distances=[400, 800])
```

```python
# v5
from cityseer.network import CityNetwork

cn = CityNetwork.from_osm(poly, to_crs_code=32630)
cn.centrality_shortest(distances=[400, 800])
result_gdf = cn.to_geopandas()
```

`CityNetwork` builds the dual graph automatically, handles boundaries via `set_boundary` (replacing manual `live` flags), and exposes accessibility, mixed-use, statistics, GTFS, and save/load methods on the same object. Its defaults are lean: a single harmonic closeness and a single betweenness, with further metrics requested via expression dictionaries.

## Renamed functions

| v4 | v5 | Notes |
| --- | --- | --- |
| `networks.node_centrality_shortest` | `networks.centrality_shortest` | old name remains as a deprecated alias emitting `DeprecationWarning`; same default columns and values |
| `networks.node_centrality_simplest` | `networks.centrality_simplest` | as above; angular analysis requires the dual graph |
| `visibility.visibility_graph` | `visibility.visibility_from_gpd` | deprecated alias warns |
| `visibility.visibility_graph_from_osm` | `visibility.visibility_from_osm` | deprecated alias warns |
| `visibility.viewshed` | `visibility.viewshed_from_gpd` | deprecated alias warns |

## Removed parameters

These fail with a clear `TypeError` rather than being silently reinterpreted:

| Removed | Do this instead |
| --- | --- |
| `betas=[...]` | rely on `distances` (the default weighting is unchanged), or pass `decay_fn="exp(-beta * c)"` with an explicit beta |
| `min_threshold_wt=` | fold the scaling into the `decay_fn` expression |
| `spatial_tolerance=` | no direct equivalent |
| `compute_hill_weighted=` | pass a `decay_fn` to weight Hill diversity |
| `angular_scaling_unit=`, `farness_scaling_offset=` | bake into the simplest-path expression, e.g. `farness="1 + c / 90"` |
| `source_indices=` | the nearest equivalent is `sample=True` (adaptive source sampling) |

## Removed: segment_centrality

The continuous-segment engine is gone and its numbers cannot be reproduced. The nearest modern equivalent is `centrality_shortest(..., segment_weighted=True)`, which weights node centrality by street-segment length (a related but different calculation). Calling `segment_centrality` raises an error pointing here.

## Changed: betweenness counts all routes

Betweenness now sources from every node; the `live` designation only filters which nodes' values are reported. On undirected, fully-live networks, values are unchanged. On buffered networks, values near the boundary increase because routes between buffer nodes that pass through the study area are now counted. On directed networks, each ordered origin-destination pair contributes with weight 1, so directed values are twice their v4 magnitude. This is a deliberate change of definition; see the [release notes](https://github.com/benchmark-urbanism/cityseer-api/blob/master/RELEASE_NOTES.md) for the reasoning.

## Column naming

Legacy default columns (`cc_beta_*`, `cc_betweenness_beta_*`, land-use `_nw`/`_wt` pairs) are reproduced exactly when calling the functional API with everyday arguments. The expression engine is opt-in: passing `closeness`/`betweenness` dicts or a `decay_fn` switches to the new single-column naming (`cc_{label}_{distance}`). `CityNetwork` uses the new naming throughout. See [column naming conventions](/guide/fundamentals#column-naming-conventions).

## Deprecation timeline

The compatibility layer is active throughout v5: deprecated names and parameters warn but work. Documentation steers new code to `CityNetwork`. The compatibility layer is removed in v6.

## If something breaks

Work through [Troubleshooting](/guide/troubleshooting) first; if the issue is migration-specific, the [COMPATIBILITY.md](https://github.com/benchmark-urbanism/cityseer-api/blob/master/COMPATIBILITY.md) contract has the full parameter table, and the [issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues) is the right place for anything not covered.
