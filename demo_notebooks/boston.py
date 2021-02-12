from matplotlib import colors
import numpy as np
from shapely import geometry
import utm

from cityseer.metrics import networks
from cityseer.util import graphs, plot, mock


# %% create a test graph
lng, lat = -0.13435325883010577, 51.51197802142709
G_utm = mock.make_buffered_osm_graph(lng, lat, 1000)
# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(275)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds
# plot
plot.plot_nX(G_utm, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=False, figsize=(20, 20), dpi=200)


# %%
# basic initial cleanup
G = graphs.nX_simple_geoms(G_utm)
G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=10, remove_disconnected=True)
plot.plot_nX(G, labels=True, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
# initial pass of spatial consolidation to cleanup major intersections
G1 = graphs.nX_consolidate_spatial(G,
                                   buffer_dist=25,
                                   min_node_threshold=6,
                                   min_node_degree=3,
                                   squash_nodes_by_highest_degree=False,
                                   merge_edges_by_midline=True)
plot.plot_nX(G1, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

#%%
# second pass of spatial consolidation to cleanup smaller clusters
G2 = graphs.nX_consolidate_spatial(G1,
                                   buffer_dist=10,
                                   min_node_threshold=2,
                                   min_node_degree=3,
                                   min_cumulative_degree=7,
                                   max_cumulative_degree=16,
                                   squash_nodes_by_highest_degree=True,
                                   merge_edges_by_midline=True)
plot.plot_nX(G2, labels=False, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
# split opposing line geoms to facilitate parallel merging
G3 = graphs.nX_split_opposing_geoms(G2, buffer_dist=15, use_midline=True)
plot.plot_nX(G3, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
G4 = graphs.nX_consolidate_spatial(G3,
                                   buffer_dist=15,
                                   min_node_threshold=2,
                                   min_node_degree=2,
                                   max_cumulative_degree=9,
                                   neighbour_policy='indirect',
                                   squash_nodes_by_highest_degree=False,
                                   merge_edges_by_midline=True)
plot.plot_nX(G4, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)


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
