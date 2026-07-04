---
layout: '@src/layouts/PageLayout.astro'
---

# Statistics

Statistical metrics aggregate numeric properties of nearby features (such as building heights or areas) over the street network, computing summary statistics like `min`, `max`, `mean`, and `var` at each node within specified distance thresholds. The examples use the [`layers.compute_stats`](/metrics/layers#compute-stats) method, which requires a `geopandas` `GeoDataFrame` and a `stats_column_labels` parameter identifying the numeric columns.

| Notebook | Description |
| -------- | ----------- |
| [gpd_stats](/examples/stats/gpd-stats) | Building statistics from a `geopandas` `GeoDataFrame`. |
| [osm_stats](/examples/stats/osm-stats) | Building statistics from `osmnx` data. |
