---
layout: '@src/layouts/PageLayout.astro'
---

# Network Centrality

Centrality metrics measure how important or connected each node (or street segment) is within the network. They are widely used in urban analytics to identify busy corridors, understand movement patterns, and evaluate the structural role of streets. `cityseer` supports both **metric** (shortest-path) and **angular** (simplest-path) centrality measures, computed at multiple distance thresholds simultaneously.

:::note
Angular centrality measures require the **dual graph**. `CityNetwork` builds the dual graph automatically, so no conversion is needed on the recommended path. When using the lower-level API, convert a primal graph with [`graphs.nx_to_dual`](/tools/graphs#nx_to_dual) before computing angular centralities.
:::

| Notebook | Description |
| -------- | ----------- |
| [gpd_metric_centrality](/examples/centrality/gpd-metric-centrality) | Metric distance centralities (density, harmonic closeness, betweenness) from a `geopandas` `GeoDataFrame`. |
| [gpd_angular_centrality](/examples/centrality/gpd-angular-centrality) | Angular (simplest-path) centralities, weighting paths by cumulative turning angle rather than distance. |
| [osm_centrality](/examples/centrality/osm-centrality) | Metric distance centralities directly from OpenStreetMap data. |
| [3d_elevation](/examples/centrality/3d-elevation) | Elevation effects on centrality: with 3D geometries, Tobler's hiking function reshapes centrality in hilly terrain. |
| [centrality_metro](/examples/centrality/centrality-metro) | Adding GTFS transport data to centrality calculations (experimental). |
| [custom_expressions](/examples/centrality/custom-expressions) | Expression-based metrics: defining custom closeness and betweenness expressions, selecting only the metrics you need, derived metrics via postprocess, and statistic selection with `measures` and `decay_fn`. |
| [sampled_centrality](/examples/centrality/sampled-centrality) | Adaptive sampling for large networks at long distance thresholds: `sample=True`, the `epsilon` tolerance, and validating sampled against exact results. |
| [od_betweenness](/examples/centrality/od-betweenness) | Demand-weighted betweenness: modelling origin-destination flows with a singly constrained spatial interaction model (population to amenities), deterrence functions, and explicit OD matrices. |
