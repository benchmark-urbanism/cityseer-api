---
layout: '@src/layouts/PageLayout.astro'
---

# Network Preparation

`cityseer` uses `networkx` [`MultiGraphs`](https://networkx.org/documentation/stable/reference/classes/multigraph.html) to represent pedestrian networks. Spatial information is embedded in the graph nodes and edges so that `cityseer` can accurately manipulate the graphs and calculate derivative metrics. The workflows in these notebooks automatically prepare the network and should be sufficient for the great majority of cases.

These recipes deliberately teach the lower-level `tools.io` and `tools.graphs` route, which offers step-by-step control over network construction and cleaning. If you do not need that control, the higher-level [`CityNetwork`](/guide/fundamentals#citynetwork-api) constructors (`from_osm`, `from_geopandas`, `from_nx`) wrap the same preparation behind a single call. Every recipe here ends by handing the prepared graph to [`CityNetwork.from_nx`](/api/network#from_nx), so whichever preparation route you take, the analysis workflow lands on the high-level API.

If you wish to prepare the network manually, note the following requirements:

- The network must be a `networkx` `MultiGraph` object.
- `cityseer` is focused on pedestrian networks, which do not ordinarily have directionality. Transportation networks are added at a later step in a way that retains directionality.
- The network must have a coordinate reference system associated with it, e.g. `G.graph['crs'] = 27700`. When converting from `osmnx` or `geopandas`, this is handled automatically.
- Nodes require `x` and `y` attributes storing coordinates in a projected CRS. Optional attributes: `live` (compute metrics for this node or use it for routing only) and `weight` (relative importance in centrality calculations).
- Edges store a `geom` attribute containing a `shapely` `LineString`. The graph topology should reflect the real-world topology of the street network: curvatures and turns within a street segment belong in the `LineString` geometry, not as additional nodes and segments, which would distort centrality measures.

| Notebook | Description |
| -------- | ----------- |
| [create_from_bbox](/examples/networks/create-from-bbox) | OSM network from a bounding box, including CRS handling. |
| [create_from_buffered_point](/examples/networks/create-from-buffered-point) | OSM network from a buffered coordinate. |
| [network_from_bounds](/examples/networks/network-from-bounds) | OSM network from a custom boundary file. |
| [network_from_relation_id](/examples/networks/network-from-relation-id) | OSM network from an OSM relation id. |
| [network_from_streets](/examples/networks/network-from-streets) | Custom network from a `geopandas` streets dataset. |
| [osmnx_to_cityseer](/examples/networks/osmnx-to-cityseer) | Convert a network from `osmnx`. |
| [momepy_to_cityseer](/examples/networks/momepy-to-cityseer) | Convert a network from `momepy`. |
| [save_to_file](/examples/networks/save-to-file) | Save a `networkx` graph to a file. |
| [create_dual_graph](/examples/networks/create-dual-graph) | Create a dual graph from a primal graph. |
| [network_simplification](/examples/networks/network-simplification) | Manually configured network simplification for OSM data. |
| [directed_networks](/examples/networks/directed-networks) | Directed networks and one-way streets: building directed graphs from OSMnx, `oneway` columns, and comparing directed against undirected betweenness. |
| [results_to_maps](/examples/networks/results-to-maps) | From results to maps: export computed metrics to a GeoPackage, style them in QGIS, and build a publication-quality matplotlib figure. |

On simplification: good sources of street network data, such as Ordnance Survey OS Open Roads, are already simplified to their essential structure with topology kept distinct from street geometry. When a high-quality source is available, it may be best not to attempt additional clean-up. Cleaning and simplification is recommended when working with OSM data; `cityseer` has an automated cleaning routine, and the simplification notebook shows manual configuration.
