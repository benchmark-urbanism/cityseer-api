'''
Plot guides

'''

# Cleaning
# graph cleanup examples
import requests
import utm
from shapely import geometry
import matplotlib.pyplot as plt

from cityseer.util import graphs, plot

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

G_wgs = graphs.nX_from_osm(osm_json=response.text)
G_utm = graphs.nX_wgs_to_utm(G_wgs)
plt.cla()
plt.clf()
plot.plot_nX(G_utm, 'guides/cleaning/graph_raw.png', dpi=150, figsize=(20, 20))

G_messy = graphs.nX_simple_geoms(G_utm)
G_messy = graphs.nX_remove_dangling_nodes(G_messy)
G_messy = graphs.nX_remove_filler_nodes(G_messy)
plt.cla()
plt.clf()
plot.plot_nX(G_messy, 'guides/cleaning/graph_topo.png', dpi=150, figsize=(20, 20))

G_messy = graphs.nX_decompose(G_messy, 50)
G_messy = graphs.nX_consolidate_parallel(G_messy)
plt.cla()
plt.clf()
plot.plot_nX(G_messy, 'guides/cleaning/graph_consolidated.png', dpi=150, figsize=(20, 20))