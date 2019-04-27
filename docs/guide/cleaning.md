Graph cleaning
==============

Certain sources of street network data offer high quality representations that work well for network analysis algorithms. The Ordnance Survey's [OS Open Roads](https://www.ordnancesurvey.co.uk/business-and-government/products/os-open-roads.html) dataset is a great example of such a dataset:
- The network has been simplified to its essential topological structure: i.e. unnecessarily complex representations of intersections; on-ramps; split roadways; etc. have been reduced to a simpler representation concurring more readily with the core topological structure of street networks. Network representations focusing on completeness (e.g. for route way-finding, see [OS ITN Layer](https://www.ordnancesurvey.co.uk/business-and-government/help-and-support/products/itn-layer.html)) introduce an unnecessary level of complexity, serving to hinder rather than help network analysis algorithms: i.e. the extra extraneous detail can introduce unintuitive or irregular outcomes for shortest-path calculations;
- The topology of the network is kept distinct from the geometry of the streets. Oftentimes, as can be seen with [Open Street Map](https://www.openstreetmap.org), extra nodes have been added to streets for the purpose of representing geometric twists and turns along a roadway. These extra nodes do not represent the topological structure of the network (e.g. intersections) and can thus lead to substantial distortions in the derivation of network centrality measures;
- Bonus: It is open and free to use!

When a high-quality source is available, it may be best not to attempt additional cleanup unless there is a particular reason to do so. On the other-hand, many indispensable sources of network information, particularly Open Street Map data, can be messy (for the purposes of network analysis). This section describes how such sources can be cleaned and prepared for subsequent analysis.


Downloading data
----------------

This example will make use of OSM data downloaded from the [OSM API](https://wiki.openstreetmap.org/wiki/API). To keep things interesting, we'll pick Travalgar Square in London, which will be buffered and cleaned for a $1,600m$ radius.

::: warning Note
The following example makes use of two excellent python modules: [`utm`](https://github.com/Turbo87/utm) for converting between `WGS` geographic coordinates and `UTM` projected coordinates; and [`shapely`](https://github.com/Toblerity/Shapely) for generating and manipulating geometry.
:::

```python
import requests
import utm
from shapely import geometry

# let's download data within a 1,600m buffer around Travalgar Square in London:
lat, lng = (51.507999, -0.127970)
# cast the WGS coordinates to UTM prior to buffering
easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng) 
# create a point, and then buffer
pt = geometry.Point(easting, northing)
geom_utm = pt.buffer(1600)
# cast the geometry back to WGS for the OSM query
geom_wgs = [utm.to_latlon(e, n, utm_zone_number, utm_zone_letter) for e, n in geom_utm.exterior.coords]
# format for OSM query
geom_osm = str.join(' ', [f'{lat} {lng}' for lat, lng in geom_wgs])
# osm query
timeout = 10
filters = '["area"!~"yes"]' \
          '["highway"!~"path|footway|motor|proposed|construction|abandoned|platform|raceway|service"]' \
          '["foot"!~"no"]' \
          '["service"!~"private"]' \
          '["access"!~"private"]'
query = f'[out:json][timeout:{timeout}];(way["highway"]{filters}(poly:"{geom_osm}"); >;);out skel qt;'
try:
    response = requests.get('https://overpass-api.de/api/interpreter',
                            timeout=timeout,
                            params={
                                'data': query
                            })
except requests.exceptions.RequestException as e:
    raise e
    
# NOTE: this code block can be combined with the subsequent blocks for a continuous example.
```

::: tip Hint
You may want to experiment with the filtering applied to the OSM query. See the [OSM Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) for more information.
:::


Generating a graph
------------------

Now that the data has been downloaded, `cityseer.util` can be used to load the data into a NetworkX graph. The graph should be converted from WGS to UTM coordinates prior to any further processing.

```python
from cityseer.util import graphs, plot

# load the OSM response data into a networkX graph
G_wgs = graphs.nX_from_osm(osm_json=response.text)
# cast the graph to UTM coordinates prior to processing
G_utm = graphs.nX_wgs_to_utm(G_wgs)

plot.plot_nX(G_utm, figsize=(20, 20), dpi=150)
```

<ImageModal :path="require('../images/plots/guides/cleaning/graph_raw.png')" alt='Raw OSM graph' caption='The raw OSM graph after conversion to UTM coordinates. © OpenStreetMap contributors.'></ImageModal>


Deducing the network topology
-----------------------------

Now that raw OSM data has been loaded into a NetworkX graph, the `cityseer.util.graph` methods can be used to further clean and prepare the network prior to analysis.

At this stage, the raw OSM graph is going to look a bit messy. Note how that nodes have been used to represent the roadway geometry. These nodes need to be removed and will be abstracted into `shapely` `LineString` geometries assigned to the respective street edges. So doing, the geometric representation will be kept distinct from the network topology.

```python
# the raw osm nodes denote the road geometries by the placement of nodes
# the first step will generate explicit linestring geometries for each street edge
G = graphs.nX_simple_geoms(G_utm)
# the next step, will now strip these "filler-nodes" from the graph
# the associated geometries will be welded into continuous linestrings
# the new linestrings will be assigned to the newly consolidated topological links
G = graphs.nX_remove_filler_nodes(G)
# OSM graphs will often have "stubs", e.g. at entrances to buildings or parking lots
# these will now be removed, and can be fine-tuned with the despine parameter.
# The removed_disconnected flag will removed isolated network components
# i.e. disconnected portions of network that are not joined to the main street network
G = graphs.nX_remove_dangling_nodes(G, despine=25, remove_disconnected=True)

plot.plot_nX(G, figsize=(20, 20), dpi=150)
```

<ImageModal :path="require('../images/plots/guides/cleaning/graph_topo.png')" alt='OSM graph topology' caption='The OSM graph after conversion to a purer topological representation. © OpenStreetMap contributors.'></ImageModal>

::: warning Note
At this point it may initially appear that the roadway geometries have now gone missing. However, this information is still present in the `LineString` geometries assigned to each street edge. Put differently, the plotted representation is now topological, not geometric.
:::


Refining the network
--------------------

The emphasis now shifts to evening out the intensity of nodes across the network through the use of decomposition. This allows for a more granular representation of data along streetfronts, and reduces distortions in network centrality measures due to varied intensities of nodes. It is also beneficial in the context of small local distance thresholds, which may otherwise intersect longer street segments.
 
This step is here coupled with the consolidation of adjacent roadways, which may otherwise exaggerate the intensity or complexity of the network in certain situations.

```python
# decomposition of the network will even out the intensity of nodes
# set the decompose_max flag based on the level of granularity required
G = graphs.nX_decompose(G, decompose_max=50)
# simplify split roadways
# some experimentation may be required to find the optimal buffer distance
# setting it too large, will deteriorate the quality of the network
G = graphs.nX_consolidate_parallel(G, buffer_dist=15)

plot.plot_nX(G, figsize=(20, 20), dpi=150)
```

<ImageModal :path="require('../images/plots/guides/cleaning/graph_consolidated.png')" alt='OSM graph after decomposition and consolidation' caption='The OSM graph after decomposition and consolidation. © OpenStreetMap contributors.'></ImageModal>

The graph is now ready for analysis. Whereas by no means perfect, it is a substantial improvement (for the purpose of network analysis) against the original raw OSM data!
