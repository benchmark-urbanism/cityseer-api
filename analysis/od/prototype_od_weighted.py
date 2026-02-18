# %%
"""
prototype_od_weighted.py - Prototype OD-weighted centrality using PCT London bicycle data.

Downloads PCT London OD data directly from GitHub, maps MSOA centroids to the GLA network,
and compares standard vs OD-weighted betweenness centrality.

Requires:
    - GLA GeoPackage at temp/os_open_roads/gla.gpkg
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer.metrics import networks
from cityseer.metrics.networks import build_od_matrix
from cityseer.tools import graphs, io
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from shapely.geometry import box

# %% Configuration

SCRIPT_DIR = Path(__file__).parent

PCT_OD_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/l.csv"
PCT_ZONES_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/z.geojson"

GLA_GPKG = SCRIPT_DIR.parent.parent.parent / "temp" / "os_open_roads" / "gla.gpkg"

DISTANCES = [5000, 10000]

# %% Download PCT data

od_df = pd.read_csv(PCT_OD_URL)
print(f"OD pairs: {len(od_df)}")

zones_gdf = gpd.read_file(PCT_ZONES_URL)
print(f"Zones: {len(zones_gdf)}")

# %% Derive London boundary from PCT zones

london_boundary_full = zones_gdf.to_crs(epsg=27700).union_all()
minx, miny, maxx, maxy = london_boundary_full.bounds
x_range = maxx - minx
y_range = maxy - miny
crop_box = box(
    minx + x_range / 3,
    miny + y_range / 3,
    maxx - x_range / 3,
    maxy - y_range / 3,
)
london_boundary = london_boundary_full.intersection(crop_box)
bbox = london_boundary.bounds  # (minx, miny, maxx, maxy)
print(f"London middle-third bbox: {bbox}")

# %% Load and clip GLA network to London boundary

if not GLA_GPKG.exists():
    raise FileNotFoundError(f"GLA GeoPackage not found: {GLA_GPKG}")

edges_gdf = gpd.read_file(GLA_GPKG, bbox=bbox)
edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty]
edges_gdf = edges_gdf[edges_gdf.intersects(london_boundary)]
edges_gdf = edges_gdf.explode(index_parts=False)
print(f"Edges within London: {len(edges_gdf)}")

G = io.nx_from_generic_geopandas(edges_gdf)
G = graphs.nx_remove_filler_nodes(G)
G = graphs.nx_remove_dangling_nodes(G)
G = graphs.nx_consolidate_nodes(G, buffer_dist=1)
G = graphs.nx_remove_dangling_nodes(G, despine=20)
print(f"GLA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %% Prepare network structure
G_dual = graphs.nx_to_dual(G)
nodes_gdf, _edges_gdf, net = io.network_structure_from_nx(G_dual)
print(f"Network structure: {net.street_node_count()} nodes")

# %% Build OD matrix

od = build_od_matrix(
    od_df,
    zones_gdf,
    net,
    origin_col="msoa1",
    destination_col="msoa2",
    weight_col="bicycle",
    zone_id_col="geo_code",
)
print(f"OdMatrix: {od.len()} pairs, {od.n_origins()} unique origins")

# %% Run standard centrality

nodes_std = networks.node_centrality_shortest(
    net,
    nodes_gdf.copy(),
    distances=DISTANCES,
    compute_closeness=False,
    compute_betweenness=True,
)

# %% Run OD-weighted centrality

nodes_od = networks.node_betweenness_od(
    net,
    nodes_gdf.copy(),
    od_matrix=od,
    distances=DISTANCES,
)

# %% Compare results

for dist in DISTANCES:
    betw_key = f"cc_betweenness_{dist}"

    std_betw = nodes_std[betw_key].values
    od_betw = nodes_od[betw_key].values

    mask = std_betw > 0
    if mask.sum() > 0:
        rho_betw, _ = spearmanr(std_betw[mask], od_betw[mask])
    else:
        rho_betw = float("nan")

    print(f"\nDistance: {dist}m")
    print(f"  Standard betweenness: mean={np.nanmean(std_betw):.1f}, max={np.nanmax(std_betw):.1f}")
    print(f"  OD-weighted betweenness: mean={np.nanmean(od_betw):.1f}, max={np.nanmax(od_betw):.1f}")
    print(f"  Spearman rho (betweenness): {rho_betw:.4f}")

# %% Compute crop bounds (middle third of London)

total_bounds = nodes_std.total_bounds  # (minx, miny, maxx, maxy)
x_range = total_bounds[2] - total_bounds[0]
y_range = total_bounds[3] - total_bounds[1]
crop_bounds = (
    total_bounds[0] + x_range / 3,
    total_bounds[1] + y_range / 3,
    total_bounds[2] - x_range / 3,
    total_bounds[3] - y_range / 3,
)
print(f"Crop bounds: {crop_bounds}")

MIN_LW = 0.1
MAX_LW = 5.0


def plot_lines(ax, gdf, values, norm, cmap_name):
    """Plot GeoDataFrame lines with per-segment colour and linewidth."""
    cmap = plt.get_cmap(cmap_name)
    normed = norm(values)
    colours = cmap(normed)
    lw_cap = np.nanquantile(values, 0.999)
    lw_frac = np.clip(values / lw_cap, 0, 1) if lw_cap > 0 else values * 0
    linewidths = MIN_LW + lw_frac**5 * (MAX_LW - MIN_LW)
    segments = [np.array(geom.coords) for geom in gdf.geometry]
    lc = LineCollection(segments, colors=colours, linewidths=linewidths)
    ax.add_collection(lc)
    ax.set_xlim(crop_bounds[0], crop_bounds[2])
    ax.set_ylim(crop_bounds[1], crop_bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


# %% Spatial plots: standard vs OD-weighted betweenness

for dist in DISTANCES:
    betw_key = f"cc_betweenness_{dist}"
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 16),
        facecolor="black",
    )

    std_vals = np.log1p(nodes_std[betw_key].values)
    od_vals = np.log1p(nodes_od[betw_key].values)

    axes[0].set_facecolor("black")
    std_norm = Normalize(vmin=0, vmax=np.nanquantile(std_vals, 0.99))
    plot_lines(axes[0], nodes_std, std_vals, std_norm, "inferno")
    axes[0].set_title(
        f"Standard betweenness ({dist}m)",
        color="white",
    )

    axes[1].set_facecolor("black")
    od_norm = Normalize(vmin=0, vmax=np.nanquantile(od_vals, 0.99))
    plot_lines(axes[1], nodes_od, od_vals, od_norm, "inferno")
    axes[1].set_title(
        f"OD-weighted betweenness ({dist}m)",
        color="white",
    )

    fig.tight_layout()
    plt.show()

# %% Scatter plot: standard vs OD-weighted betweenness

for dist in DISTANCES:
    betw_key = f"cc_betweenness_{dist}"

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = nodes_std[betw_key] > 0
    rho, _ = spearmanr(nodes_std[betw_key][mask], nodes_od[betw_key][mask])
    ax.scatter(
        nodes_std[betw_key][mask],
        nodes_od[betw_key][mask],
        s=0.5,
        alpha=0.3,
        c="#B2182B",
    )
    ax.set_xlabel("Standard betweenness")
    ax.set_ylabel("OD-weighted betweenness")
    ax.set_title(f"Betweenness {dist}m (rho={rho:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    plt.show()

# %%
