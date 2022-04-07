import os

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import utm
from matplotlib import colors
from shapely import geometry

from cityseer.metrics import networks, layers
from cityseer.tools import mock, graphs, plot

base_path = os.getcwd()
plt.style.use('matplotlibrc')

###
# INTRO PLOT
G = mock.mock_graph()
plot.plot_nX(G,
             labels=True,
             node_size=80,
             path='images/graph.png',
             dpi=150)

# INTRO EXAMPLE PLOTS
G = graphs.nX_simple_geoms(G)
G = graphs.nX_decompose(G, 20)

N = networks.NetworkLayerFromNX(G, distances=[400, 800])
N.segment_centrality(measures=['segment_harmonic'])

data_dict = mock.mock_data_dict(G, random_seed=25)
D = layers.DataLayerFromDict(data_dict)
D.assign_to_network(N, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
D.hill_branch_wt_diversity(landuse_labels, qs=[0])
G_metrics = N.to_networkX()

segment_harmonic_vals = []
mixed_uses_vals = []
for node, data in G_metrics.nodes(data=True):
    segment_harmonic_vals.append(data['metrics']['centrality']['segment_harmonic'][800])
    mixed_uses_vals.append(data['metrics']['mixed_uses']['hill_branch_wt'][0][400])

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
segment_harmonic_cols = cmap(segment_harmonic_vals)
plot.plot_nX(G_metrics,
             plot_geoms=True,
             node_colour=segment_harmonic_cols,
             path='images/intro_segment_harmonic.png',
             dpi=150)

# plot hill mixed uses
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)

plot.plot_assignment(N,
                     D,
                     node_colour=mixed_uses_cols,
                     data_labels=landuse_labels,
                     path='images/intro_mixed_uses.png',
                     dpi=150)

#
#
# MOCK MODULE
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
plot.plot_nX(G,
             plot_geoms=True,
             labels=True,
             node_size=80,
             path='images/graph_example.png',
             dpi=150)  # WITH LABELS

#
#
# GRAPH MODULE
plot.plot_nX(G,
             plot_geoms=True,
             path='images/graph_simple.png',
             dpi=150)  # NO LABELS

G_simple = graphs.nX_simple_geoms(G)
G_decomposed = graphs.nX_decompose(G_simple, 100)
plot.plot_nX(G_decomposed,
             plot_geoms=True,
             path='images/graph_decomposed.png',
             dpi=150)

G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=True,
                            path='images/graph_dual.png',
                            dpi=150)

# graph cleanup examples
lng, lat = -0.13396079424572427, 51.51371088849723
G_utm = mock.make_buffered_osm_graph(lng, lat, 1250)
easting, northing, _zone, _letter = utm.from_latlon(lat, lng)
buffer = 750
min_x = easting - buffer
max_x = easting + buffer
min_y = northing - buffer
max_y = northing + buffer


# reusable plot function
def simple_plot(_G, _path, plot_geoms=True):
    # plot using the selected extents
    plot.plot_nX(_G,
                 labels=False,
                 plot_geoms=plot_geoms,
                 node_size=10,
                 edge_width=1,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 dpi=200,
                 path=_path)


G = graphs.nX_simple_geoms(G_utm)
simple_plot(G, 'images/graph_cleaning_1.png', plot_geoms=False)

G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
G = graphs.nX_remove_filler_nodes(G)
simple_plot(G, 'images/graph_cleaning_2.png')

# first pass of consolidation
G1 = graphs.nX_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
simple_plot(G1, 'images/graph_cleaning_3.png')

# split opposing line geoms to facilitate parallel merging
G2 = graphs.nX_split_opposing_geoms(G1, buffer_dist=15)
simple_plot(G2, 'images/graph_cleaning_4.png')

# second pass of consolidation
G3 = graphs.nX_consolidate_nodes(G2,
                                 buffer_dist=15,
                                 crawl=False,
                                 min_node_degree=2,
                                 cent_min_degree=4)
simple_plot(G3, 'images/graph_cleaning_5.png')

#
#
# NETWORKS MODULE
# before and after plots
# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
plot.plot_nX(G,
             plot_geoms=True,
             labels=True,
             node_size=80,
             path='images/graph_before.png',
             dpi=150)

# generate the network layer and compute some metrics
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
# compute some-or-other metrics
N.segment_centrality(measures=['segment_harmonic'])
# convert back to networkX
G_post = N.to_networkX()
plot.plot_nX(G_post,
             plot_geoms=True,
             labels=True,
             node_size=80,
             path='images/graph_after.png',
             dpi=150)

#
#
# LAYERS MODULE
# show assignment to network
# random seed 25
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

data_dict = mock.mock_data_dict(G, random_seed=25)
L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(N, max_dist=500)
plot.plot_assignment(N,
                     L,
                     path='images/assignment.png',
                     dpi=150)

G_decomposed = graphs.nX_decompose(G, 50)
N_decomposed = networks.NetworkLayerFromNX(G_decomposed, distances=[200, 400, 800, 1600])

L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(N_decomposed, max_dist=500)
plot.plot_assignment(N_decomposed,
                     L,
                     path='images/assignment_decomposed.png',
                     dpi=150)

#
#
# PLOT MODULE
from cityseer.tools import mock, graphs, plot

G = mock.mock_graph()
G_simple = graphs.nX_simple_geoms(G)
G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=False,
                            path='images/graph_dual.png',
                            dpi=150)

# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)
G = graphs.nX_decompose(G, 50)
N = networks.NetworkLayerFromNX(G, distances=[800])
N.node_centrality(measures=['node_beta'])
G_after = N.to_networkX()
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data['metrics']['centrality']['node_beta'][800])
# let's create a custom colourmap using matplotlib
cmap = colors.LinearSegmentedColormap.from_list('cityseer',
                                                [(100 / 255, 193 / 255, 255 / 255, 255 / 255),
                                                 (211 / 255, 47 / 255, 47 / 255, 1 / 255)])
# normalise the values
vals = colors.Normalize()(vals)
# cast against the colour map
cols = cmap(vals)
plot.plot_nX(G_after,
             plot_geoms=True,
             path='images/graph_colour.png',
             node_colour=cols,
             dpi=150)

# assignment plot
data_dict = mock.mock_data_dict(G, random_seed=25)
D = layers.DataLayerFromDict(data_dict)
D.assign_to_network(N, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
plot.plot_assignment(N,
                     D,
                     path='images/assignment_plot.png',
                     data_labels=landuse_labels,
                     dpi=150)

#
#
# BETA DECAYS
plt.cla()
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

w_min = 0.01831563888873418
# set the betas
betas = []
for d_max in [200, 400, 800, 1600]:
    beta = np.log(w_min) / d_max
    distances_arr = []
    for d in range(0, d_max + 1):
        distances_arr.append(d)
    distances_arr = np.array(distances_arr)
    y_falloff = np.exp(beta * distances_arr)
    ax.plot(distances_arr, y_falloff, label=f'$\\beta={-round(beta, 4)}$')

# add w_min
plt.axhline(y=w_min, ls='--', lw=0.5)
ax.text(10, 0.035, '$w_{min}$')

ax.set_xticks([200, 400, 800, 1600])
ax.set_xticklabels(['$d_{max}=200$', '$d_{max}=400$', '$d_{max}=800$', '$d_{max}=1600$'])
ax.set_xlim([0, 1600])
ax.set_xlabel('Distance in Metres')
ax.set_ylim([0, 1.0])
ax.set_ylabel('Weighting')
ax.set_facecolor('#19181B')
leg = ax.legend(loc='upper right', title='$exp(-\\beta \\cdot d[i,j])$', fancybox=True, facecolor='#19181B')
leg.get_frame().set_linewidth(0.1)
plt.savefig('images/betas.png', dpi=300, facecolor='#19181B')

#
#
# OSMnx COMPARISON
# centre-point
lng, lat = -0.13396079424572427, 51.51371088849723

# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
buff = geometry.Point(easting, northing).buffer(1000)
min_x, min_y, max_x, max_y = buff.bounds


# reusable plot function
def simple_plot(_G, _path):
    # plot using the selected extents
    plot.plot_nX(_G,
                 labels=False,
                 plot_geoms=True,
                 node_size=10,
                 edge_width=1,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 dpi=200,
                 path=_path)


# Let's use OSMnx to fetch an OSM graph
# We'll use the same raw network for both workflows (hence simplify=False)
multi_di_graph_raw = ox.graph_from_point((lat, lng),
                                         dist=1250,
                                         simplify=False)

# Workflow 1: Using OSMnx for simplification
# ==========================================
# explicit simplification via OSMnx
multi_di_graph_utm = ox.project_graph(multi_di_graph_raw)
multi_di_graph_simpl = ox.simplify_graph(multi_di_graph_utm)
multi_di_graph_cons = ox.consolidate_intersections(multi_di_graph_simpl,
                                                   tolerance=10,
                                                   dead_ends=True)
# let's use the same plotting function for both scenarios to aid visual comparisons
multi_graph_cons = graphs.nX_from_OSMnx(multi_di_graph_cons, tolerance=50)
simple_plot(multi_graph_cons, 'images/osmnx_simplification.png')

# WORKFLOW 2: Using cityseer for simplification
# =============================================
# let's convert the OSMnx graph to cityseer compatible `multiGraph`
G_raw = graphs.nX_from_OSMnx(multi_di_graph_raw)
# convert to UTM
G = graphs.nX_wgs_to_utm(G_raw)
# infer geoms
G = graphs.nX_simple_geoms(G)
# remove degree=2 nodes
G = graphs.nX_remove_filler_nodes(G)
# remove dangling nodes
G = graphs.nX_remove_dangling_nodes(G, despine=10)
# repeat degree=2 removal to remove orphaned nodes due to despining
G = graphs.nX_remove_filler_nodes(G)
# let's consolidate the nodes
G1 = graphs.nX_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
# let's also remove as many parallel carriageways as possible
G2 = graphs.nX_split_opposing_geoms(G1, buffer_dist=15)
G3 = graphs.nX_consolidate_nodes(G2,
                                 buffer_dist=15,
                                 crawl=False,
                                 min_node_degree=2,
                                 cent_min_degree=4)
simple_plot(G3, 'images/osmnx_cityseer_simplification.png')
