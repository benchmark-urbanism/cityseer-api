---
layout: '@src/layouts/PageLayout.astro'
---

# Network Cleaning

Raw street-network data is not ready for analysis. Sources such as OpenStreetMap, Ordnance Survey Open Roads, and US Census TIGER/Line represent the physical road surface, not the topological structure that centrality measures need. A single high street is often drawn as two parallel carriageways. Roundabouts become rings of segments, and complex junctions become clusters of slip roads and pedestrian crossings. These extra nodes and edges do not correspond to distinct decision points in the network.

Left uncleaned, this inflates node counts and distorts the metrics. Closeness and betweenness are pulled towards wherever the source data happens to be most finely subdivided. Angular centrality is affected most, because every spurious node introduces a spurious turn, and the measure is built from turn angles (see [choosing shortest or angular](/guide/centrality#choosing-shortest-or-angular)). Cleaning is a required step before the metrics can be trusted.

## The automated method

`cityseer` implements an automated network-cleaning and consolidation method that reduces a raw road surface to an analysis-ready topological graph: it merges parallel carriageways, consolidates complex junctions to single nodes, removes interstitial ("filler") nodes that only subdivide a continuous street, drops short dangling stubs, and irons out geometric noise. This automated approach is evaluated by [Abdeldayem et al. (2026), _Environment and Planning B: Urban Analytics and City Science_](https://journals.sagepub.com/doi/10.1177/23998083261433647), which compares fully automated modelling against hybrid (manually refined) models across four cities. It finds that automated workflows capture large-scale integration patterns well, while local, flow-sensitive measures are more sensitive to how the network is segmented, which is why the pipeline below is configurable.

On the high-level path, [`CityNetwork.from_osm`](/api/network#from_osm) applies a sensible default pipeline when `simplify=True` (the default), so most users get a well-cleaned network without extra steps. Every `CityNetwork` constructor also cleans input geometries during construction: short self-loops are removed, chains of segments meeting at filler (degree-2) endpoints are welded into single segments, short dead-end stubs are removed, and near-duplicate parallel edges are merged. Each step is controlled by a constructor parameter (`remove_fillers`, `remove_danglers`, `merge_parallel_dist`) and can be disabled to pass a network through as-is; directed networks skip the welding and merging steps automatically. This construction-time cleaning is deliberately conservative and does not replace the topological simplification described above: it does not consolidate complex junctions or merge separated dual carriageways.

## Configuring the pipeline

The automated defaults suit most cases, but the right consolidation distance for a dense European core differs from a sprawling suburban grid, and some morphologies need more or less aggressive merging. The pipeline is fully exposed through the [`graphs`](/tools/graphs) module, so you can assemble and tune it step by step:

- [`nx_remove_filler_nodes`](/tools/graphs#nx_remove_filler_nodes) removes degree-2 nodes that only subdivide a street.
- [`nx_consolidate_nodes`](/tools/graphs#nx_consolidate_nodes) merges nearby nodes (e.g. the several nodes of one junction) within a buffer distance.
- [`nx_merge_parallel_edges`](/tools/graphs#nx_merge_parallel_edges) collapses dual carriageways and other parallel edges.
- [`nx_remove_dangling_nodes`](/tools/graphs#nx_remove_dangling_nodes) drops short dead-end stubs (the `despine` length) and disconnected fragments.
- [`nx_iron_edges`](/tools/graphs#nx_iron_edges) straightens geometric noise below an angle threshold.

Tuning these parameters, and inspecting the result before computing centrality, is the reliable way to get a clean network for a specific dataset and morphology. The [Network Simplification recipe](/examples/networks/network-simplification) works through the configurable pipeline step by step, and the [Network Preparation recipes](/examples/networks) show the full range of construction routes.

:::note
Cleaning has the largest effect on angular (simplest-path) centrality, which is sensitive to how the network is represented. Shortest-path (metric) centrality is more robust to residual noise, though a clean network improves every measure.
:::
