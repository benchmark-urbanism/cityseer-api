---
layout: '@src/layouts/PageLayout.astro'
---

# Accessibility

Accessibility metrics quantify how easily different land uses (such as shops, parks, or restaurants) can be reached from each point in the street network. The examples use [`compute_accessibilities`](/metrics/layers#compute-accessibilities), which requires a `geopandas` `GeoDataFrame` of landuse data, a `landuse_column_label`, and the target `accessibility_keys`. For each land-use key and distance threshold, the output `GeoDataFrame` gains a **count** of reachable instances (`cc_{key}_{distance}`), optionally weighted by a distance decay via `decay_fn`, plus the **distance to the nearest** instance at the largest threshold (`cc_{key}_nearest_max_{distance}`). The `CityNetwork` method defaults to a plain count; the lower-level `layers` function, when `decay_fn` is omitted, retains the legacy default of both an unweighted (`_nw`) and a decay-weighted (`_wt`) count column. See the [Land-Use guide](/guide/land-use#accessibility) for details.

| Notebook | Description |
| -------- | ----------- |
| [gpd_accessibility](/examples/accessibility/gpd-accessibility) | Landuse accessibilities from a `geopandas` `GeoDataFrame`. |
| [osm_accessibility](/examples/accessibility/osm-accessibility) | Landuse accessibilities from OpenStreetMap. |
| [osm_accessibility_polys](/examples/accessibility/osm-accessibility-polys) | Park accessibilities from OSM polygon data. |
| [gpd_mixed_uses](/examples/accessibility/gpd-mixed-uses) | Mixed land-uses with [`layers.compute_mixed_uses`](/metrics/layers#compute-mixed-uses). |
| [accessibility_metro](/examples/accessibility/accessibility-metro) | Adding GTFS transport data (experimental). |
