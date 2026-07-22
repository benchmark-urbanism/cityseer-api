# %%
"""
pct_london_flows.py - OD-weighted and demand-weighted cycle betweenness for central London.

Demonstrates the two OD workflows in cityseer on observed cycling data:

  - betweenness_od       : routes an *explicit* OD matrix (PCT observed bicycle commutes).
  - betweenness_demand   : *models* the matrix from weighted origins/destinations with a
                           singly-constrained gravity model, then routes it.

Data:
  - PCT London OD pairs + MSOA zones (auto-downloaded from github.com/Robinlovelace/pct-data).
    The `bicycle` column is the 2011 Census journey-to-work cycle count per MSOA pair (WU03EW).
  - OS Open Roads GB network (road_link layer), clipped to a central-London study area.
    Download: https://osdatahub.os.uk/downloads/open/OpenRoads  (set OSROADS below).

Calibration (the two decay levers do different jobs):
  - decay beta -> shapes destination choice; calibrated against the observed spatial pattern.
  - participation -> scales trip generation; with all-mode origins it is the cycle mode share.

The distance threshold is set well beyond the study extent (20 km) so it does not censor
trips: the decay, not a hard cutoff, governs destination choice.
"""

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from cityseer.network import CityNetwork
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from shapely.geometry import Point

# %% Configuration

SCRIPT_DIR = Path(__file__).parent
CACHE = SCRIPT_DIR.parent.parent / "temp" / "od_pct"
CACHE.mkdir(parents=True, exist_ok=True)
# OS Open Roads GB GeoPackage (road_link layer). Update this path to your local copy.
OSROADS = SCRIPT_DIR.parent.parent / "temp" / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"

PCT_L_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/l.csv"
PCT_Z_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/z.geojson"

CENTRE_LNG_LAT = (-0.12, 51.51)
STUDY_RADIUS = 4000  # study area radius (m); live-node analysis window
DISTANCE = 20000  # threshold (m): set well beyond the study extent so the decay, not the cutoff, governs
TOLERANCE = 10.0  # multipath tolerance (%), spreads flow across near-equal shortest paths

# %% Load PCT data and OS Open Roads network

pct_l = CACHE / "l.csv"
pct_z = CACHE / "z.geojson"
if not pct_l.exists():
    urllib.request.urlretrieve(PCT_L_URL, pct_l)
if not pct_z.exists():
    urllib.request.urlretrieve(PCT_Z_URL, pct_z)
l_df = pd.read_csv(pct_l)
zones = gpd.read_file(pct_z)

if not OSROADS.exists():
    raise FileNotFoundError(f"OS Open Roads GeoPackage not found: {OSROADS}")

centre = gpd.GeoSeries([Point(*CENTRE_LNG_LAT)], crs=4326).to_crs(27700).iloc[0]
study_poly = centre.buffer(STUDY_RADIUS)
edges = gpd.read_file(OSROADS, layer="road_link", bbox=study_poly.buffer(800).bounds)
edges = edges[edges.geometry.is_valid & ~edges.geometry.is_empty].explode(index_parts=False)
cn = CityNetwork.from_geopandas(edges, boundary=study_poly)
print(f"network: {len(edges)} OS Open Roads links")

# %% Zones and inputs

zones27 = zones.to_crs(27700)
zones27 = zones27[zones27.geometry.centroid.within(study_poly.buffer(500))].reset_index(drop=True)
keep = set(zones27["geo_code"])
sub = l_df[l_df["msoa1"].isin(keep) & l_df["msoa2"].isin(keep)]

# observed OD matrix (explicit bicycle counts)
matrix = cn.build_od_matrix(
    sub[["msoa1", "msoa2", "bicycle"]],
    zones27,
    origin_col="msoa1",
    destination_col="msoa2",
    weight_col="bicycle",
    zone_id_col="geo_code",
    max_netw_assign_dist=1500.0,
)

# independent demand inputs: all-mode commuter masses (origins = residents, destinations = jobs proxy)
orig_all = sub.groupby("msoa1")["all"].sum()
dest_all = sub.groupby("msoa2")["all"].sum()
cent = zones27.copy()
cent["geometry"] = cent.geometry.centroid
origins_gdf = cent.assign(w=cent["geo_code"].map(orig_all).fillna(0.0))
origins_gdf = origins_gdf[origins_gdf["w"] > 0][["w", "geometry"]]
dests_gdf = cent.assign(w=cent["geo_code"].map(dest_all).fillna(0.0))
dests_gdf = dests_gdf[dests_gdf["w"] > 0][["w", "geometry"]]
print(f"{len(zones27)} MSOA zones, {matrix.len()} OD pairs, {sub['bicycle'].sum():.0f} cycle trips")

# %% Standard and observed-OD betweenness

bkey, dkey = f"cc_betweenness_{DISTANCE}", f"cc_demand_{DISTANCE}"
cn.centrality_shortest(distances=[DISTANCE], closeness={}, betweenness=None, cycles=False)
std = cn.to_geopandas()
cn.betweenness_od(matrix, distances=[DISTANCE], tolerance=TOLERANCE)
odw = cn.to_geopandas()
live = odw.live.to_numpy()
obs_v = odw[bkey].to_numpy()


def rho(v):
    """Spearman rank correlation of `v` against the observed OD flow, on live streets."""
    return spearmanr(v[live], obs_v[live]).statistic


baseline = rho(std[bkey].to_numpy())


def demand(beta, participation):
    cn.betweenness_demand(
        origins_gdf=origins_gdf,
        destinations_gdf=dests_gdf,
        origin_weight_col="w",
        destination_weight_col="w",
        distances=[DISTANCE],
        decay_fn=f"exp(-{beta:.6f} * c)",
        participation=participation,
        tolerance=TOLERANCE,
        max_netw_assign_dist=1500.0,
    )
    return cn.to_geopandas()


# %% Calibrate beta (spatial fit) then participation (volume = mode share)

BETAS = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.002]
rho_beta = [rho(demand(b, 1.0)[dkey].to_numpy()) for b in BETAS]
beta_star = BETAS[int(np.argmax(rho_beta))]
mode_share = float(sub["bicycle"].sum()) / float(sub["all"].sum())
print(f"uniform baseline rho={baseline:.3f}")
for b, r in zip(BETAS, rho_beta, strict=True):
    print(f"  beta={b:<8} rho={r:.3f} lift={r - baseline:+.3f}")
print(f"beta* = {beta_star} (spatial), participation* = mode share = {mode_share:.3f}")

dem = demand(beta_star, max(mode_share, 0.01))
rho_final = rho(dem[dkey].to_numpy())
print(f"calibrated demand rho vs observed = {rho_final:.3f}")

# %% Figures

b = study_poly.buffer(-300).bounds


def draw(ax, gdf, col, title):
    liv = gdf[gdf.live].copy()
    lv = np.log1p(liv[col].to_numpy())
    lo, hi = np.quantile(lv, 0.20), np.quantile(lv, 0.99)
    t = np.clip((lv - lo) / (hi - lo + 1e-9), 0, 1) ** 1.1
    ax.set_facecolor("white")
    liv.plot(ax=ax, color="#d8d8d8", linewidth=0.2)
    order = np.argsort(t)
    liv.iloc[order].plot(
        ax=ax, color=plt.get_cmap("Reds")(0.10 + 0.90 * t[order]), linewidth=0.04 + 2.6 * t[order] ** 1.5
    )
    ax.set_title(title, color="black", fontsize=12, loc="left")
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="white")
draw(axes[0], std, bkey, "1 · Standard betweenness (uniform)")
draw(axes[1], odw, bkey, "2 · OD-weighted (observed PCT bicycle)")
draw(axes[2], dem, dkey, f"3 · Demand-weighted (modelled, calibrated · ρ={rho_final:.2f})")
fig.suptitle(
    f"cityseer · central London · OS Open Roads · PCT cycle commutes  —  "
    f"{DISTANCE // 1000} km threshold · β={beta_star:g}, participation={mode_share:.2f}",
    color="black",
    fontsize=11,
    y=0.05,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(SCRIPT_DIR / "pct_london_flows.png", dpi=150, facecolor="white", bbox_inches="tight")

fig2, ax = plt.subplots(figsize=(8, 5))
ax.axhline(baseline, ls="--", color="#888", label=f"uniform baseline ρ={baseline:.3f}")
ax.plot(BETAS, rho_beta, "o-", color="#B2182B")
ax.axvline(beta_star, ls=":", color="green", label=f"β* = {beta_star:g}")
ax.set_xscale("log")
ax.set_xlabel("decay β  (exp(-β·c))")
ax.set_ylabel("Spearman ρ vs observed bicycle OD")
ax.set_title(
    f"β calibration at {DISTANCE // 1000} km threshold (participation from mode share = {mode_share:.2f})",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(SCRIPT_DIR / "pct_london_calibration.png", dpi=150, bbox_inches="tight")
print("saved pct_london_flows.png, pct_london_calibration.png")

# %%
