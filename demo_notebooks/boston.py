# %%
import requests
import fiona
from shapely import geometry
import utm
import networkx as nx
from cityseer.util import graphs, plot
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

#  %%
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
poly_utm = multipoly_utm.buffer(10000).buffer(-10000)
# is now polygon
print(f'UTM geometry type: {poly_utm.type}')
# now valid
print(f'UTM geometry validity: {poly_utm.is_valid}')
print(f'UTM geometry area: {poly_utm.area}')
print(f'UTM geometry width: {poly_utm.bounds[2] - poly_utm.bounds[0]}')

# convert back to WGS
# the polygon is too big for the OSM server, so have to use convex hull then later prune
geom = [utm.to_latlon(east, north, utm_zone_number, utm_zone_letter) for east, north in
        poly_utm.convex_hull.exterior.coords]
poly_wgs = geometry.Polygon(geom)

# format for OSM query
geom_osm = str.join(' ', [f'{lat} {lng}' for lat, lng in poly_wgs.exterior.coords])

#  %%
# osm query
timeout = 60 * 10  # 10 minutes
filters = '["area"!~"yes"]' \
          '["highway"!~"footway|proposed|construction|abandoned|platform|raceway|service"]' \
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

#  %%
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

#  %%
# 1m30s
# cast the graph to UTM coordinates prior to processing
G_utm = graphs.nX_wgs_to_utm(G_wgs)

#  %%
# take centrepoint
lng, lat = (-71.058884, 42.360081)
# convert to UTM using same UTM as before
easting, northing = utm.from_latlon(lat, lng, force_zone_letter=utm_zone_letter, force_zone_number=utm_zone_number)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(2000)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds
print(f'min x: {min_x:.1f} min y: {min_y:.1f} max x: {max_x:.1f} max y: {max_y:.1f}')


#  %%
# setup plotting function
# min x: 279932.7 min y: 4651857.9 max x: 369520.5 max y: 4733680.8
def nx_plot_zoom(nx_graph,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 figsize=(20, 20),
                 dpi=200,
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
    nx.draw(nx_graph,
            pos,
            with_labels=False,
            node_color='#d32f2f',
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


#  %%
nx_plot_zoom(G_utm)

#  %%
# 1hr
# optionally clean out portions of the network that aren't inside the original boundary
# might take a while so don't do until some free time
# can be optimised with strtree but probably not worth it for once-off
del_points = []
for n, data in tqdm(G_utm.nodes(data=True)):
    p = geometry.Point(data['x'], data['y'])
    if not multipoly_utm.contains(p):
        del_points.append(n)
G_utm.remove_nodes_from(del_points)
plot.plot_nX(G_utm)

#  %%
# 1m30s
G = graphs.nX_simple_geoms(G_utm)
G = graphs.nX_remove_dangling_nodes(G, despine=25, remove_disconnected=True)
nx_plot_zoom(G)

#  %%
# 3m
G = graphs.nX_remove_filler_nodes(G)
nx_plot_zoom(G)

#  %%
# 2m
G = graphs.nX_decompose(G, decompose_max=50)
nx_plot_zoom(G)

#  %%
# uses spatial indexing so is faster
G_spatial = graphs.nX_consolidate_spatial(G, buffer_dist=15)
nx_plot_zoom(G_spatial)

#  %%
# slower but better quality
# G_para = graphs.nX_consolidate_parallel(G, buffer_dist=15)
# nx_plot_zoom(G_para)
