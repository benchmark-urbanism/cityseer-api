import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io, mock, plot, util

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
nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
nodes_gdf = networks.segment_centrality(network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[400, 800])
data_gdf = mock.mock_landuse_categorical_data(G, random_seed=25)
nodes_gdf, data_gdf = layers.compute_mixed_uses(
    data_gdf,
    landuse_column_label="categorical_landuses",
    nodes_gdf=nodes_gdf,
    network_structure=network_structure,
    distances=[400, 800],
)
# custom colourmap
segment_harmonic_vals = nodes_gdf["cc_seg_harmonic_800"]
mixed_uses_vals = nodes_gdf["cc_hill_q0_800_wt"]
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
    G_simple, G_dual, plot_geoms=False, path=f"{IMAGES_PATH}/graph_dual.{FORMAT}", dpi=200, figsize=(5, 5)
)
# graph cleanup examples
lng, lat = -0.13396079424572427, 51.51371088849723
buffer = 1250
poly_wgs, _ = io.buffered_point_poly(lng, lat, buffer)
graph_raw = io.osm_graph_from_poly(poly_wgs, simplify=False)
graph_utm = io.osm_graph_from_poly(poly_wgs, simplify=True, iron_edges=True)
# plot buffer
buffered_point, _ = io.buffered_point_poly(lng, lat, 750, projected=True)
min_x, min_y, max_x, max_y = buffered_point.bounds


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


simple_plot(graph_raw, f"{IMAGES_PATH}/graph_cleaning_1.{FORMAT}", plot_geoms=False)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_1b.{FORMAT}")

graph_utm = graphs.nx_simple_geoms(graph_raw)
graph_utm = graphs.nx_remove_filler_nodes(graph_utm)
graph_utm = graphs.nx_remove_dangling_nodes(graph_utm, despine=15, remove_disconnected=10)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_2.{FORMAT}")
# first pass of consolidation
graph_utm = graphs.nx_consolidate_nodes(graph_utm, buffer_dist=15, crawl=True)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_3.{FORMAT}")
# split opposing line geoms to facilitate parallel merging
graph_utm = graphs.nx_split_opposing_geoms(graph_utm, buffer_dist=15)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_4.{FORMAT}")
# second pass of consolidation
graph_utm = graphs.nx_consolidate_nodes(graph_utm, buffer_dist=15, crawl=False, neighbour_policy="indirect")
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_5.{FORMAT}")
# iron edges
graph_utm = graphs.nx_iron_edges(graph_utm)
simple_plot(graph_utm, f"{IMAGES_PATH}/graph_cleaning_6.{FORMAT}")
# LAYERS MODULE
# show assignment to network
# random seed 25
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
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
nodes_gdf_decomp, _edges_gdf, network_structure_decomp = io.network_structure_from_nx(G_decomposed, crs=3395)
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

# PLOT MODULE
from cityseer.tools import graphs, mock, plot

G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 50)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
networks.node_centrality_shortest(network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[800])
G_after = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data["cc_beta_800"])
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
nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G_decomp, crs=3395)
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
