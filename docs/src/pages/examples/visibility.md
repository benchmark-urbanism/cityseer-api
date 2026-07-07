---
layout: '@src/layouts/PageLayout.astro'
---

# Visibility

Visibility metrics assess how enclosed or open the street environment is at each point in the network, based on the surrounding built form. `cityseer` computes visibility by casting rays from each network node and measuring how far they travel before hitting a building or obstruction.

| Notebook | Description |
| -------- | ----------- |
| [vis_osm](/examples/visibility/vis-osm) | Visibility from OSM building footprints via bounding box. |
| [vis_gpd](/examples/visibility/vis-gpd) | Visibility from building footprints in a `geopandas` `GeoDataFrame`. |
| [vis_raster](/examples/visibility/vis-raster) | Visibility from raster data (GeoTIFF), blending footprints with DEM terrain. |
