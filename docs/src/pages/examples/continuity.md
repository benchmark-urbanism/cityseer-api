---
layout: '@src/layouts/PageLayout.astro'
---

# Continuity

Street continuity metrics assess how consistently named routes, road classifications, or route numbers extend through the network. `cityseer` computes continuity by tracing connected edges that share the same street name, route number, or highway classification. The results can be expressed by count (number of connected segments) or by length (total metres of the continuous route).

| Notebook | Description |
| -------- | ----------- |
| [continuity_osm](/examples/continuity/continuity-osm) | Street name, route, highway, and hybrid continuity metrics from OSM data. |
| [continuity_os_open](/examples/continuity/continuity-os-open) | Continuity metrics from Ordnance Survey OS Open Roads data. |
