import pathlib

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import utm
from matplotlib import colors
from shapely import geometry

from cityseer.metrics import layers, networks  # pylint: disable=import-error
from cityseer.tools import graphs, mock, plot  # pylint: disable=import-error

PLOT_RC_PATH = pathlib.Path(__file__).parent / "matplotlibrc"
print(f"matplotlibrc path: {PLOT_RC_PATH}")
plt.style.use(PLOT_RC_PATH)

IMAGES_PATH = pathlib.Path(__file__).parent.parent / "public/images"
print(f"images path: {IMAGES_PATH}")

FORMAT = "png"

###
# INTRO PLOT
G = mock.mock_graph()
plot.plot_nx(G, labels=True, node_size=80, path=f"{IMAGES_PATH}/graph.{FORMAT}", dpi=150)

# INTRO EXAMPLE PLOTS
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 20)

cc_netw = networks.NetworkLayerFromNX(G, distances=[400, 800])
cc_netw.segment_centrality(measures=["segment_harmonic"])

data_dict = mock.mock_data_dict(G, random_seed=25)
cc_data = layers.DataLayerFromDict(data_dict)
cc_data.assign_to_network(cc_netw, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
cc_data.hill_branch_wt_diversity(landuse_labels, qs=[0])
G_metrics = cc_netw.to_nx_multigraph()

segment_harmonic_vals = []
mixed_uses_vals = []
for node, data in G_metrics.nodes(data=True):
    segment_harmonic_vals.append(data["metrics"]["centrality"]["segment_harmonic"][800])
    mixed_uses_vals.append(data["metrics"]["mixed_uses"]["hill_branch_wt"][0][400])

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list("cityseer", ["#64c1ff", "#d32f2f"])
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
segment_harmonic_cols = cmap(segment_harmonic_vals)
plot.plot_nx(
    G_metrics,
    plot_geoms=True,
    node_colour=segment_harmonic_cols,
    path=f"{IMAGES_PATH}/intro_segment_harmonic.{FORMAT}",
    dpi=150,
)

# plot hill mixed uses
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)

plot.plot_assignment(
    cc_netw.network_structure,
    cc_netw.nx_multigraph,
    cc_data.data_map,
    path=f"{IMAGES_PATH}/intro_mixed_uses.{FORMAT}",
    node_colour=mixed_uses_cols,
    data_labels=landuse_labels,
    dpi=150,
)

#
#
# MOCK MODULE
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
plot.plot_nx(
    G,
    plot_geoms=True,
    labels=True,
    node_size=80,
    path=f"{IMAGES_PATH}/graph_example.{FORMAT}",
    dpi=150,
)  # WITH LABELS

#
#
# GRAPH MODULE
plot.plot_nx(G, plot_geoms=True, path=f"{IMAGES_PATH}/graph_simple.{FORMAT}", dpi=150)  # NO LABELS

G_simple = graphs.nx_simple_geoms(G)
G_decomposed = graphs.nx_decompose(G_simple, 100)
plot.plot_nx(G_decomposed, plot_geoms=True, path=f"{IMAGES_PATH}/graph_decomposed.{FORMAT}", dpi=150)

G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(G_simple, G_dual, plot_geoms=True, path=f"{IMAGES_PATH}/graph_dual.{FORMAT}", dpi=150)

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
    plot.plot_nx(
        _G,
        labels=False,
        plot_geoms=plot_geoms,
        node_size=10,
        edge_width=1,
        x_lim=(min_x, max_x),
        y_lim=(min_y, max_y),
        dpi=200,
        path=_path,
    )


G = graphs.nx_simple_geoms(G_utm)
simple_plot(G, f"{IMAGES_PATH}/graph_cleaning_1.{FORMAT}", plot_geoms=False)

G = graphs.nx_remove_filler_nodes(G)
G = graphs.nx_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
G = graphs.nx_remove_filler_nodes(G)
simple_plot(G, f"{IMAGES_PATH}/graph_cleaning_2.{FORMAT}")

# first pass of consolidation
G1 = graphs.nx_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
simple_plot(G1, f"{IMAGES_PATH}/graph_cleaning_3.{FORMAT}")

# split opposing line geoms to facilitate parallel merging
G2 = graphs.nx_split_opposing_geoms(G1, buffer_dist=15)
simple_plot(G2, f"{IMAGES_PATH}/graph_cleaning_4.{FORMAT}")

# second pass of consolidation
G3 = graphs.nx_consolidate_nodes(G2, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4)
simple_plot(G3, f"{IMAGES_PATH}/graph_cleaning_5.{FORMAT}")

#
#
# NETWORKS MODULE
# before and after plots
# prepare a mock graph
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
plot.plot_nx(
    G,
    plot_geoms=True,
    labels=True,
    node_size=80,
    path=f"{IMAGES_PATH}/graph_before.{FORMAT}",
    dpi=150,
)

# generate the network layer and compute some metrics
cc_netw = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
# compute some-or-other metrics
cc_netw.segment_centrality(measures=["segment_harmonic"])
# convert back to networkX
G_post = cc_netw.to_nx_multigraph()
plot.plot_nx(
    G_post,
    plot_geoms=True,
    labels=True,
    node_size=80,
    path=f"{IMAGES_PATH}/graph_after.{FORMAT}",
    dpi=150,
)

#
#
# LAYERS MODULE
# show assignment to network
# random seed 25
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
cc_netw = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])

data_dict = mock.mock_data_dict(G, random_seed=25)
L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(cc_netw, max_dist=500)
plot.plot_assignment(
    cc_netw.network_structure, cc_netw.nx_multigraph, L.data_map, path=f"{IMAGES_PATH}/assignment.{FORMAT}", dpi=150
)

G_decomposed = graphs.nx_decompose(G, 50)
N_decomposed = networks.NetworkLayerFromNX(G_decomposed, distances=[200, 400, 800, 1600])

L = layers.DataLayerFromDict(data_dict)
L.assign_to_network(N_decomposed, max_dist=500)
plot.plot_assignment(
    N_decomposed.network_structure,
    N_decomposed.nx_multigraph,
    L.data_map,
    path=f"{IMAGES_PATH}/assignment_decomposed.{FORMAT}",
    dpi=150,
)

#
#
# PLOT MODULE
from cityseer.tools import graphs, mock, plot

G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(G_simple, G_dual, plot_geoms=False, path=f"{IMAGES_PATH}/graph_dual.{FORMAT}", dpi=150)

# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 50)
cc_netw = networks.NetworkLayerFromNX(G, distances=[800])
cc_netw.node_centrality(measures=["node_beta"])
G_after = cc_netw.to_nx_multigraph()
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data["metrics"]["centrality"]["node_beta"][800])
# let's create a custom colourmap using matplotlib
cmap = colors.LinearSegmentedColormap.from_list(
    "cityseer",
    [
        (100 / 255, 193 / 255, 255 / 255, 255 / 255),
        (211 / 255, 47 / 255, 47 / 255, 1 / 255),
    ],
)
# normalise the values
vals = colors.Normalize()(vals)
# cast against the colour map
cols = cmap(vals)
plot.plot_nx(
    G_after,
    plot_geoms=True,
    path=f"{IMAGES_PATH}/graph_colour.{FORMAT}",
    node_colour=cols,
    dpi=150,
)

# assignment plot
data_dict = mock.mock_data_dict(G, random_seed=25)
cc_data = layers.DataLayerFromDict(data_dict)
cc_data.assign_to_network(cc_netw, max_dist=400)
landuse_labels = mock.mock_categorical_data(len(data_dict), random_seed=25)
plot.plot_assignment(
    cc_netw.network_structure,
    cc_netw.nx_multigraph,
    cc_data.data_map,
    path=f"{IMAGES_PATH}/assignment_plot.{FORMAT}",
    data_labels=landuse_labels,
    dpi=150,
)

#
#
# BETA DECAYS
plt.cla()
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

w_min = 0.01831563888873418
# set the betas
betas: list[float] = []
for d_max in [200, 400, 800, 1600]:
    beta = np.log(w_min) / d_max
    distances: list[float] = []
    for d in range(0, d_max + 1):
        distances.append(d)
    distances_arr = np.array(distances)
    y_falloff = np.exp(beta * distances_arr)
    ax.plot(distances_arr, y_falloff, label=f"$\\beta={-round(beta, 4)}$")

# add w_min
plt.axhline(y=w_min, ls="--", lw=0.5)
ax.text(10, 0.035, "$w_{min}$")

ax.set_xticks([200, 400, 800, 1600])
ax.set_xticklabels(["$d_{max}=200$", "$d_{max}=400$", "$d_{max}=800$", "$d_{max}=1600$"])
ax.set_xlim([0, 1600])
ax.set_xlabel("Distance in Metres")
ax.set_ylim([0, 1.0])
ax.set_ylabel("Weighting")
ax.set_facecolor("#19181B")
leg = ax.legend(
    loc="upper right",
    title="$exp(-\\beta \cdot d[i,j])$",
    fancybox=True,
    facecolor="#19181B",
)
leg.get_frame().set_linewidth(0.1)
plt.savefig(f"{IMAGES_PATH}/betas.{FORMAT}", dpi=300, facecolor="#19181B")

#
#
# OSMnx COMPARISON
# centre-point
lng, lat = -0.13396079424572427, 51.51371088849723

# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
buff = geometry.Point(easting, northing).buffer(1000)
min_x, min_y, max_x, max_y = buff.bounds

# Let's use OSMnx to fetch an OSM graph
# We'll use the same raw network for both workflows (hence simplify=False)
multi_di_graph_raw = ox.graph_from_point((lat, lng), dist=1250, simplify=False)

# Workflow 1: Using OSMnx for simplification
# ==========================================
# explicit simplification via OSMnx
multi_di_graph_utm = ox.project_graph(multi_di_graph_raw)
multi_di_graph_simpl = ox.simplify_graph(multi_di_graph_utm)
multi_di_graph_cons = ox.consolidate_intersections(multi_di_graph_simpl, tolerance=10, dead_ends=True)
# let's use the same plotting function for both scenarios to aid visual comparisons
multi_graph_cons = graphs.nx_from_osm_nx(multi_di_graph_cons, tolerance=50)
simple_plot(multi_graph_cons, f"{IMAGES_PATH}/osmnx_simplification.{FORMAT}")

# WORKFLOW 2: Using cityseer for simplification
# =============================================
# let's convert the OSMnx graph to cityseer compatible `multiGraph`
G_raw = graphs.nx_from_osm_nx(multi_di_graph_raw)
# convert to UTM
G = graphs.nx_wgs_to_utm(G_raw)
# infer geoms
G = graphs.nx_simple_geoms(G)
# remove degree=2 nodes
G = graphs.nx_remove_filler_nodes(G)
# remove dangling nodes
G = graphs.nx_remove_dangling_nodes(G, despine=10)
# repeat degree=2 removal to remove orphaned nodes due to despining
G = graphs.nx_remove_filler_nodes(G)
# let's consolidate the nodes
G1 = graphs.nx_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
# let's also remove as many parallel carriageways as possible
G2 = graphs.nx_split_opposing_geoms(G1, buffer_dist=15)
G3 = graphs.nx_consolidate_nodes(G2, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4)
simple_plot(G3, f"{IMAGES_PATH}/osmnx_cityseer_simplification.{FORMAT}")
