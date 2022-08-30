import pathlib

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import utm
from matplotlib import colors
from shapely import geometry

from cityseer.metrics import layers, networks  # pylint: disable=import-error
from cityseer.tools import graphs, io, mock, plot  # pylint: disable=import-error

PLOT_RC_PATH = pathlib.Path(__file__).parent / "matplotlibrc"
print(f"matplotlibrc path: {PLOT_RC_PATH}")
plt.style.use(PLOT_RC_PATH)

IMAGES_PATH = pathlib.Path(__file__).parent.parent / "public/images"
print(f"images path: {IMAGES_PATH}")

FORMAT = "png"

###
# INTRO PLOT
G = mock.mock_graph()
plot.plot_nx(G, labels=True, node_size=80, path=f"{IMAGES_PATH}/graph.{FORMAT}", dpi=200, figsize=(5, 5))

# INTRO EXAMPLE PLOTS
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 20)
nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
networks.segment_centrality(
    measures=["segment_harmonic"], network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[400, 800]
)
data_gdf = mock.mock_landuse_categorical_data(G, random_seed=25)
nodes_gdf, data_gdf = layers.hill_branch_wt_diversity(
    data_gdf,
    landuse_column_label="categorical_landuses",
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[400, 800],
    qs=[0],
)
# custom colourmap
segment_harmonic_vals = nodes_gdf["cc_metric_segment_harmonic_800"]
mixed_uses_vals = nodes_gdf["cc_metric_hill_branch_wt_q0_400"]
cmap = colors.LinearSegmentedColormap.from_list("cityseer", ["#64c1ff", "#d32f2f"])
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
segment_harmonic_cols = cmap(segment_harmonic_vals)
plot.plot_nx(
    G,
    plot_geoms=True,
    node_colour=segment_harmonic_cols,
    path=f"{IMAGES_PATH}/intro_segment_harmonic.{FORMAT}",
    dpi=200,
    figsize=(5, 5),
)
# plot hill mixed uses
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)
plot.plot_assignment(
    network_structure,
    G,
    data_gdf=data_gdf,
    path=f"{IMAGES_PATH}/intro_mixed_uses.{FORMAT}",
    node_colour=mixed_uses_cols,
    data_labels=data_gdf["categorical_landuses"],
    dpi=200,
    figsize=(5, 5),
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
    dpi=200,
    figsize=(5, 5),
)  # WITH LABELS

#
#
# GRAPH MODULE
plot.plot_nx(G, plot_geoms=True, path=f"{IMAGES_PATH}/graph_simple.{FORMAT}", dpi=200, figsize=(5, 5))  # NO LABELS

G_simple = graphs.nx_simple_geoms(G)
G_decomposed = graphs.nx_decompose(G_simple, 100)
plot.plot_nx(G_decomposed, plot_geoms=True, path=f"{IMAGES_PATH}/graph_decomposed.{FORMAT}", dpi=200, figsize=(5, 5))

G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(
    G_simple, G_dual, plot_geoms=True, path=f"{IMAGES_PATH}/graph_dual.{FORMAT}", dpi=200, figsize=(5, 5)
)

# graph cleanup examples
lng, lat = -0.13396079424572427, 51.51371088849723
buffer = 1250
poly_wgs, _poly_utm, _utm_zone_number, _utm_zone_letter = io.buffered_point_poly(lng, lat, buffer)
graph_raw = io.osm_graph_from_poly(poly_wgs, simplify=False)
graph_utm = io.osm_graph_from_poly(poly_wgs, simplify=True, remove_parallel=True, iron_edges=True)
# plot buffer
easting, northing = utm.from_latlon(lat, lng)[:2]
buff = geometry.Point(easting, northing).buffer(750)
min_x, min_y, max_x, max_y = buff.bounds


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


simple_plot(graph_raw, f"{IMAGES_PATH}/graph_cleaning_1a.{FORMAT}", plot_geoms=False)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_1b.{FORMAT}")

graph_utm = graphs.nx_simple_geoms(graph_raw)
graph_utm = graphs.nx_remove_filler_nodes(graph_utm)
graph_utm = graphs.nx_remove_dangling_nodes(graph_utm, despine=20, remove_disconnected=True)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_2.{FORMAT}")
# first pass of consolidation
graph_utm = graphs.nx_consolidate_nodes(
    graph_utm, buffer_dist=15, crawl=True, min_node_group=4, cent_min_degree=4, cent_min_names=4
)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_3.{FORMAT}")
# split opposing line geoms to facilitate parallel merging
graph_utm = graphs.nx_split_opposing_geoms(graph_utm, buffer_dist=15)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_4.{FORMAT}")
# second pass of consolidation
graph_utm = graphs.nx_consolidate_nodes(
    graph_utm, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4, cent_min_names=4
)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_5.{FORMAT}")
# iron edges
graph_utm = graphs.nx_iron_edges(graph_utm)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_6.{FORMAT}")

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
    dpi=200,
    figsize=(5, 5),
)

# generate the network layer and compute some metrics
nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
# compute some-or-other metrics
nodes_gdf = networks.segment_centrality(
    measures=["segment_harmonic"],
    network_structure=network_structure,
    nodes_gdf=nodes_gdf,
    distances=[200, 400, 800, 1600],
)
# convert back to networkX
G_post = graphs.nx_from_network_structure(nodes_gdf, network_structure, G)
plot.plot_nx(
    G_post,
    plot_geoms=True,
    labels=True,
    node_size=80,
    path=f"{IMAGES_PATH}/graph_after.{FORMAT}",
    dpi=200,
    figsize=(5, 5),
)

#
#
# LAYERS MODULE
# show assignment to network
# random seed 25
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
data_gdf = mock.mock_data_gdf(G, random_seed=25)
data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
plot.plot_assignment(
    network_structure,
    G,
    data_gdf,
    path=f"{IMAGES_PATH}/assignment.{FORMAT}",
    dpi=200,
    figsize=(5, 5),
)
G_decomposed = graphs.nx_decompose(G, 50)
nodes_gdf_decomp, network_structure_decomp = graphs.network_structure_from_nx(G_decomposed, crs=3395)
data_gdf = mock.mock_data_gdf(G, random_seed=25)
data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure_decomp, max_netw_assign_dist=400)
plot.plot_assignment(
    network_structure_decomp,
    G_decomposed,
    data_gdf,
    path=f"{IMAGES_PATH}/assignment_decomposed.{FORMAT}",
    dpi=200,
    figsize=(5, 5),
)

#
#
# PLOT MODULE
from cityseer.tools import graphs, mock, plot

G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(
    G_simple, G_dual, plot_geoms=False, path=f"{IMAGES_PATH}/graph_dual.{FORMAT}", dpi=200, figsize=(5, 5)
)

# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 50)
nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
networks.node_centrality(
    measures=["node_beta"], network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[800]
)
G_after = graphs.nx_from_network_structure(nodes_gdf, network_structure, G)
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data["cc_metric_node_beta_800"])
# let's create a custom colourmap using matplotlib
cmap = colors.LinearSegmentedColormap.from_list(
    "cityseer", [(100 / 255, 193 / 255, 255 / 255, 255 / 255), (211 / 255, 47 / 255, 47 / 255, 1 / 255)]
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
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G_decomp = graphs.nx_decompose(G, 50)
nodes_gdf, network_structure = graphs.network_structure_from_nx(G_decomp, crs=3395)
data_gdf = mock.mock_landuse_categorical_data(G_decomp, random_seed=25)
data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
plot.plot_assignment(
    network_structure,
    G_decomp,
    data_gdf,
    data_labels=data_gdf["categorical_landuses"].values,
    path=f"{IMAGES_PATH}/assignment_plot.{FORMAT}",
    dpi=200,
    figsize=(5, 5),
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
# centrepoint
lng, lat = -0.13396079424572427, 51.51371088849723

# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
buffer_dist = 1250
buffer_poly = geometry.Point(easting, northing).buffer(1000)
min_x, min_y, max_x, max_y = buffer_poly.bounds

# Let's use OSMnx to fetch an OSM graph
# We'll use the same raw network for both workflows (hence simplify=False)
multi_di_graph_raw = ox.graph_from_point((lat, lng), dist=buffer_dist, simplify=False)

# Workflow 1: Using OSMnx to prepare the graph
# ============================================
# explicit simplification and consolidation via OSMnx
multi_di_graph_utm = ox.project_graph(multi_di_graph_raw)
multi_di_graph_simpl = ox.simplify_graph(multi_di_graph_utm)
multi_di_graph_cons = ox.consolidate_intersections(multi_di_graph_simpl, tolerance=10, dead_ends=True)
# let's use the same plotting function for both scenarios to aid visual comparisons
multi_graph_cons = io.nx_from_osm_nx(multi_di_graph_cons, tolerance=50)
simple_plot(multi_graph_cons, f"{IMAGES_PATH}/osmnx_simplification.{FORMAT}")

# WORKFLOW 2: Using cityseer to manually clean an OSMnx graph
# ===========================================================
G_raw = io.nx_from_osm_nx(multi_di_graph_raw)
G = graphs.nx_wgs_to_utm(G_raw)
G = graphs.nx_simple_geoms(G)
G = graphs.nx_remove_filler_nodes(G)
G = graphs.nx_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
G1 = graphs.nx_consolidate_nodes(G, buffer_dist=15, crawl=True, min_node_group=4, cent_min_degree=4, cent_min_names=4)
G2 = graphs.nx_split_opposing_geoms(G1, buffer_dist=15)
G3 = graphs.nx_consolidate_nodes(
    G2, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4, cent_min_names=4
)
G4 = graphs.nx_iron_edges(G3)
simple_plot(G3, f"{IMAGES_PATH}/osmnx_cityseer_simplification.{FORMAT}")

# WORKFLOW 3: Using cityseer to download and automatically simplify the graph
# ===========================================================================
poly_wgs, _poly_utm, _utm_zone_number, _utm_zone_letter = io.buffered_point_poly(lng, lat, buffer_dist)
G_utm = io.osm_graph_from_poly(poly_wgs, simplify=True, remove_parallel=True, iron_edges=True)
simple_plot(G_utm, f"{IMAGES_PATH}/cityseer_only_simplification.{FORMAT}")
