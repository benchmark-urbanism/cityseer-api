---
layout: '@src/layouts/PageLayout.astro'
---

# Network Cleaning

Raw street-network data is not ready for analysis. Sources such as OpenStreetMap, Ordnance Survey Open Roads, and US Census TIGER/Line represent the physical road surface, not the topological structure that centrality measures need. A single high street may appear as two parallel carriageways, a roundabout as a ring of segments, a junction as a cluster of slip roads and pedestrian crossings. Each of these adds nodes and edges that do not correspond to distinct decision points in the network.

Left uncleaned, this inflates node counts and distorts the metrics. Closeness and betweenness are pulled towards wherever the source data happens to be most finely subdivided. Angular centrality is affected most, because every spurious node introduces a spurious turn, and the measure is built from turn angles (see [choosing shortest or angular](/guide/centrality#choosing-shortest-or-angular)). Cleaning is a required step before the metrics can be trusted.

## The automated method

`cityseer` implements an automated network-cleaning and consolidation method that reduces a raw road surface to an analysis-ready topological graph: it merges parallel carriageways, consolidates complex junctions to single nodes, removes interstitial ("filler") nodes that only subdivide a continuous street, drops short dangling stubs, and irons out geometric noise. This automated approach is evaluated by [Abdeldayem et al. (2026), _Environment and Planning B: Urban Analytics and City Science_](https://journals.sagepub.com/doi/10.1177/23998083261433647), which compares fully automated modelling against hybrid (manually refined) models across four cities. It finds that automated workflows capture large-scale integration patterns well, while local, flow-sensitive measures are more sensitive to how the network is segmented, which is exactly why the pipeline below is configurable rather than fixed.

On the high-level path, [`CityNetwork.from_osm`](/api/network#from_osm) applies a sensible default pipeline when `simplify=True` (the default), so most users get a well-cleaned network without extra steps. Construction from a `GeoDataFrame` or NetworkX graph also cleans input geometries automatically (short self-loops and near-duplicate edges).

## It is configurable

The automated defaults suit most cases, but the right consolidation distance for a dense European core differs from a sprawling suburban grid, and some morphologies need more or less aggressive merging. The pipeline is fully exposed through the [`graphs`](/tools/graphs) module, so you can assemble and tune it step by step:

- [`nx_remove_filler_nodes`](/tools/graphs#nx_remove_filler_nodes) — remove degree-2 nodes that only subdivide a street.
- [`nx_consolidate_nodes`](/tools/graphs#nx_consolidate_nodes) — merge nearby nodes (e.g. the several nodes of one junction) within a buffer distance.
- [`nx_merge_parallel_edges`](/tools/graphs#nx_merge_parallel_edges) — collapse dual carriageways and other parallel edges.
- [`nx_remove_dangling_nodes`](/tools/graphs#nx_remove_dangling_nodes) — drop short dead-end stubs (the `despine` length) and disconnected fragments.
- [`nx_iron_edges`](/tools/graphs#nx_iron_edges) — straighten geometric noise below an angle threshold.

Tuning these parameters, and inspecting the result before computing centrality, is the reliable way to get a clean network for a specific dataset and morphology. The [Network Simplification recipe](/examples/networks/network-simplification) works through the configurable pipeline step by step, and the [Network Preparation recipes](/examples/networks) show the full range of construction routes.

:::note
Cleaning matters most when you intend to use angular (simplest-path) centrality, which is sensitive to network representation. Shortest-path (metric) centrality is more robust to residual noise, but a clean network improves every measure.
:::
