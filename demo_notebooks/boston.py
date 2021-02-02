# %%
from cityseer.metrics import networks
from cityseer.util import graphs, plot
from matplotlib import colors
import numpy as np
import networkx as nx
import requests
from shapely import geometry
import utm

import importlib

importlib.reload(graphs)

# %%
# QUICK OPTION - 20km radius
lat, lng = (51.51342151135985, -0.1386875041450292)
# cast the WGS coordinates to UTM prior to buffering
easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)
# create a point, and then buffer
pt = geometry.Point(easting, northing)
poly_utm = pt.buffer(350)

# %%
'''
# FULL OPTION: USE SHAPEFILE
# import shapefile (it was exported to EPSG 4326 CRS)
with fiona.open('./demo_notebooks/boston/boston_4326.shp') as src:
    for i, shape in enumerate(src):
        print(f'processing item {i}')
        multipoly = geometry.shape(shape['geometry'])

# MultiPolygon
print(f'Geometry type: {multipoly.type}')
# number of geoms = 26
print(f'Number of geometries: {len(multipoly.geoms)}')
# largest poly is not valid...
print(f'Geometry validity: {multipoly.is_valid}')

# convert to UTM
geoms = []
utm_zone_number = utm_zone_letter = None
for geom in multipoly.geoms:
    if utm_zone_number is None:
        lng, lat = geom.exterior.coords[0]
        easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)
    geom_utm = [utm.from_latlon(lat, lng)[:2] for lng, lat in geom.exterior.coords]
    geom_utm = geometry.Polygon(geom_utm)
    geoms.append(geom_utm)
multipoly_utm = geometry.MultiPolygon(geoms)

# try fix validity
print(f'UTM geometry validity: {multipoly_utm.is_valid}')
# still not valid, so try buffer and erosion
poly_utm = multipoly_utm.buffer(10000).plot_buffer(-10000)
# is now polygon
print(f'UTM geometry type: {poly_utm.type}')
# now valid
print(f'UTM geometry validity: {poly_utm.is_valid}')
print(f'UTM geometry area: {poly_utm.area}')
print(f'UTM geometry width: {poly_utm.bounds[2] - poly_utm.bounds[0]}')
'''

# %%
# convert back to WGS
# the polygon is too big for the OSM server, so have to use convex hull then later prune
geom = [utm.to_latlon(east, north, utm_zone_number, utm_zone_letter) for east, north in
        poly_utm.convex_hull.exterior.coords]
poly_wgs = geometry.Polygon(geom)

# format for OSM query
geom_osm = str.join(' ', [f'{lat} {lng}' for lat, lng in poly_wgs.exterior.coords])

# osm query
timeout = 60 * 3  # 10 minutes

query = f'''
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
    */
    [out:json][timeout:{timeout}];
    /*
    build spatial_set from highways based on extent
    */
    way["highway"]
      ["area"!="yes"]
      ["highway"!~"motorway|motorway_link|bus_guideway|escape|raceway|proposed|abandoned|platform|construction"]
      ["service"!~"parking_aisle"]
      (if:
       /* don't fetch roads that don't have sidewalks */
       (t["sidewalk"] != "none" && t["sidewalk"] != "no")
       /* unless foot or bicycles permitted */
       || t["foot"]!="no"
       || (t["bicycle"]!="no" && t["bicycle"]!="unsuitable")
      )
      ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
      ["access"!~"private|customers"]
      ["indoor"!="yes"]
      (poly:"{geom_osm}") -> .spatial_set;
    /*
    build union_set from spatial_set
    */
    (
      way.spatial_set["highway"];
      way.spatial_set["foot"~"yes|designated"];
      way.spatial_set["bicycle"~"yes|designated"];
    ) -> .union_set;
    /*
    filter union_set
    */
    way.union_set -> .filtered_set;
    /*
    union filtered_set ways with nodes via recursion
    */
    (
      .filtered_set;
      >;
    );
    /*
    return only basic info
    */
    out skel qt;
    '''
try:
    response = requests.get('https://overpass-api.de/api/interpreter',
                            timeout=timeout,
                            params={
                                'data': query
                            })
except requests.exceptions.RequestException as e:
    raise e

# %%
# load the OSM response data into a networkX graph
G_wgs = graphs.nX_from_osm(osm_json=response.text)
print(nx.info(G_wgs))

'''
'["highway"!~"path|footway|motorway|proposed|construction|abandoned|platform|raceway|service"]'
Number of nodes: 1155459
Number of edges: 1195401
Average degree:   2.0691

'["highway"!~"path|footway|proposed|construction|abandoned|platform|raceway|service"]'
Number of nodes: 1193339
Number of edges: 1233961
Average degree:   2.0681

'["highway"!~"footway|proposed|construction|abandoned|platform|raceway|service"]'
Number of nodes: 1421452
Number of edges: 1469657
Average degree:   2.0678

'["highway"!~"proposed|construction|abandoned|platform|raceway|service"]'
Number of nodes: 1658411
Number of edges: 1739395
Average degree:   2.0977
'''

# %%
# 1m30s
# cast the graph to UTM coordinates prior to processing
G_utm = graphs.nX_wgs_to_utm(G_wgs)

# %%
# setup plotting function
# take centrepoint
# lat, lng = (51.510434954365735, -0.1297641580617722)
# convert to UTM using same UTM as before
easting, northing = utm.from_latlon(lat, lng, force_zone_letter=utm_zone_letter, force_zone_number=utm_zone_number)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(350)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds
print(f'min x: {min_x:.1f} min y: {min_y:.1f} max x: {max_x:.1f} max y: {max_y:.1f}')

# min x: 327449.4 min y: 4688809.9 max x: 333449.4 max y: 4694809.9
"""
def nx_plot_zoom(nx_graph,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 figsize=(20, 20),
                 dpi=200,
                 colour=None,
                 **kwargs):
    # cleanup old plots
    plt.ioff()
    plt.close('all')
    plt.cla()
    plt.clf()
    # create new plot
    plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    # extract x, y
    pos = {}
    for n, d in nx_graph.nodes(data=True):
        pos[n] = (d['x'], d['y'])
    # default node colour if needed
    if colour is None:
        colour = '#d32f2f'
    nx.draw(nx_graph,
            pos,
            with_labels=True,
            node_color=colour,
            node_size=30,
            node_shape='o',
            edge_color='b',
            width=1,
            alpha=0.75)
    # override face color if necessary
    plt.gcf().set_facecolor('#fff')
    # extract limits
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    # show
    plt.show()
"""
# %%
G_utm = graphs.nX_simple_geoms(G_utm)
plot.plot_nX(G_utm, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=False, figsize=(20, 20),
             dpi=200)
# small distance spatial consolidation to deal with problematic situations such as
# bridging overlapping walkways per lat, lng = (51.493, -0.06393)
# %%
import importlib

importlib.reload(graphs)
importlib.reload(plot)

# %%
G = graphs.nX_simple_geoms(G_utm)
G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=10, remove_disconnected=True)
G = graphs.nX_remove_filler_nodes(G)
plot.plot_nX(G, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
# %%
G1 = graphs.nX_consolidate_spatial(G,
                                   buffer_dist=25,
                                   min_node_threshold=6,
                                   min_node_degree=3,
                                   squash_by_highest_degree=False,
                                   merge_by_midline=True)
plot.plot_nX(G1, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
# %%
G2 = graphs.nX_consolidate_spatial(G1,
                                   buffer_dist=10,
                                   min_node_threshold=2,
                                   min_node_degree=3,
                                   min_cumulative_degree=7,
                                   max_cumulative_degree=16,
                                   squash_by_highest_degree=False,
                                   merge_by_midline=True)
plot.plot_nX(G2, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
# %%
G3 = graphs.nX_split_opposing_geoms(G2, buffer_dist=15, use_midline=True)
plot.plot_nX(G3, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
# %%
G4 = graphs.nX_consolidate_spatial(G3,
                                   buffer_dist=15,
                                   min_node_threshold=2,
                                   min_node_degree=2,
                                   max_cumulative_degree=9,
                                   neigh_policy='indirect',
                                   squash_by_highest_degree=False,
                                   merge_by_midline=True)
plot.plot_nX(G4, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
plot.plot_nX(G3, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
# create a Network layer from the networkX graph
N = networks.Network_Layer_From_nX(G_cons, distances=[1000, 5000, 10000])
# the underlying method allows the computation of various centralities simultaneously, e.g.
N.compute_centrality(measures=['node_harmonic', 'node_betweenness'])

# %%
G_metrics = N.to_networkX()

#  %%
# plot centrality
cent = []
for node, data in G_metrics.nodes(data=True):
    cent.append(data['metrics']['centrality']['node_harmonic'][10000])
# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
# mask outliers
cent = np.array(cent)
upper_threshold = np.percentile(cent, 99.9)
outlier_idx = cent > upper_threshold
cent[outlier_idx] = upper_threshold
# normalise the values
segment_harmonic_vals = colors.Normalize()(cent)
# cast against the colour map
segment_harmonic_cols = cmap(segment_harmonic_vals)
# plot
nx_plot_zoom(G_cons, colour=segment_harmonic_cols)

# plot centrality
cent = []
for node, data in G_metrics.nodes(data=True):
    cent.append(data['metrics']['centrality']['node_betweenness'][10000])
# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
# mask outliers
cent = np.array(cent)
upper_threshold = np.percentile(cent, 99.9)
outlier_idx = cent > upper_threshold
cent[outlier_idx] = upper_threshold
# normalise the values
segment_harmonic_vals = colors.Normalize()(cent)
# cast against the colour map
segment_harmonic_cols = cmap(segment_harmonic_vals)
# plot
nx_plot_zoom(G_cons, colour=segment_harmonic_cols)
