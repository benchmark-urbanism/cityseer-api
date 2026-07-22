# %%
"""
pct_london_flows.py - OD-weighted and demand-weighted cycle betweenness for London.

Demonstrates the two OD workflows in cityseer on observed cycling data:

  - betweenness_od       : routes an *explicit* OD matrix (PCT observed bicycle commutes).
  - betweenness_demand   : *models* the matrix from weighted origins/destinations with a
                           singly-constrained gravity model, then routes it.

Two decay levers are calibrated separately:
  - decay beta -> shapes destination choice; fitted against the observed spatial pattern.
  - participation -> scales trip generation; with all-mode origins it is the cycle mode share.

Run scope (first CLI argument, default "central"):
  - central : 4 km study area around the centre; 3-panel figure (standard / OD / demand).
  - london  : whole Greater London (~264k OS Open Roads links); 2-panel figure (OD / demand).
              Standard (all-source) betweenness is skipped at this scale - it sources from every
              node rather than the ~900 origins, so it does not scale without sampling.

Data:
  - PCT London OD pairs + MSOA zones (auto-downloaded from github.com/Robinlovelace/pct-data).
    The `bicycle` / `all` columns are 2011 Census journey-to-work counts per MSOA pair (WU03EW).
  - OS Open Roads GB network (road_link layer). Download and set OSROADS below:
    https://osdatahub.os.uk/downloads/open/OpenRoads
"""

import sys
import time
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

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "central"  # "central" | "london"
SCRIPT_DIR = Path(__file__).parent
CACHE = SCRIPT_DIR.parent.parent / "temp" / "od_pct"
CACHE.mkdir(parents=True, exist_ok=True)
# OS Open Roads GB GeoPackage (road_link layer). Update this path to your local copy.
OSROADS = SCRIPT_DIR.parent.parent / "temp" / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"

PCT_L_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/l.csv"
PCT_Z_URL = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london/z.geojson"

CENTRE_LNG_LAT = (-0.12, 51.51)
CENTRAL_RADIUS = 4000  # central-scope study radius (m)
DISTANCE = 20000  # threshold (m): set beyond the study extent so the decay, not the cutoff, governs
TOLERANCE = 10.0  # multipath tolerance (%), spreads flow across near-equal shortest paths
BETAS = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.002]

timings: dict[str, float] = {}


def stamp(label, t0):
    """Record and print the elapsed seconds for a stage."""
    timings[label] = time.perf_counter() - t0
    print(f"[{label:20}] {timings[label]:7.1f} s", flush=True)


# %% Load PCT data and OS Open Roads network

pct_l, pct_z = CACHE / "l.csv", CACHE / "z.geojson"
if not pct_l.exists():
    urllib.request.urlretrieve(PCT_L_URL, pct_l)
if not pct_z.exists():
    urllib.request.urlretrieve(PCT_Z_URL, pct_z)
l_df = pd.read_csv(pct_l)
zones = gpd.read_file(pct_z).to_crs(27700)

if not OSROADS.exists():
    raise FileNotFoundError(f"OS Open Roads GeoPackage not found: {OSROADS}")

if SCOPE == "central":
    centre = gpd.GeoSeries([Point(*CENTRE_LNG_LAT)], crs=4326).to_crs(27700).iloc[0]
    study_poly = centre.buffer(CENTRAL_RADIUS)
    read_bbox = study_poly.buffer(800).bounds
    boundary = study_poly
else:
    study_poly = zones.union_all()
    read_bbox = tuple(zones.total_bounds)
    boundary = study_poly

t0 = time.perf_counter()
edges = gpd.read_file(OSROADS, layer="road_link", bbox=read_bbox)
edges = edges[edges.geometry.is_valid & ~edges.geometry.is_empty].explode(index_parts=False)
stamp("read_network", t0)
t0 = time.perf_counter()
cn = CityNetwork.from_geopandas(edges, boundary=boundary)
stamp("build_citynetwork", t0)
print(f"   scope={SCOPE}, {len(edges)} road links", flush=True)

# %% Zones and inputs

zones_in = zones[zones.geometry.centroid.within(study_poly.buffer(500))].reset_index(drop=True)
keep = set(zones_in["geo_code"])
sub = l_df[l_df["msoa1"].isin(keep) & l_df["msoa2"].isin(keep)]

t0 = time.perf_counter()
matrix = cn.build_od_matrix(
    sub[["msoa1", "msoa2", "bicycle"]],
    zones_in,
    origin_col="msoa1",
    destination_col="msoa2",
    weight_col="bicycle",
    zone_id_col="geo_code",
    max_netw_assign_dist=1500.0,
)
stamp("build_od_matrix", t0)
print(f"   {len(zones_in)} MSOAs, {matrix.len()} OD pairs, {matrix.n_origins()} origins", flush=True)

# independent demand inputs: all-mode commuter masses (residents as origins, jobs as destinations)
orig_all = sub.groupby("msoa1")["all"].sum()
dest_all = sub.groupby("msoa2")["all"].sum()
cent = zones_in.copy()
cent["geometry"] = cent.geometry.centroid
origins_gdf = cent.assign(w=cent["geo_code"].map(orig_all).fillna(0.0))
origins_gdf = origins_gdf[origins_gdf["w"] > 0][["w", "geometry"]]
dests_gdf = cent.assign(w=cent["geo_code"].map(dest_all).fillna(0.0))
dests_gdf = dests_gdf[dests_gdf["w"] > 0][["w", "geometry"]]

# %% Observed-OD betweenness (and standard betweenness for the central scope)

bkey, dkey = f"cc_betweenness_{DISTANCE}", f"cc_demand_{DISTANCE}"
std = None
if SCOPE == "central":
    cn.centrality_shortest(distances=[DISTANCE], closeness={}, betweenness=None, cycles=False)
    std = cn.to_geopandas()
t0 = time.perf_counter()
cn.betweenness_od(matrix, distances=[DISTANCE], tolerance=TOLERANCE)
odw = cn.to_geopandas()
stamp("betweenness_od", t0)
print(f"   {len(odw)} dual nodes (street segments)", flush=True)
live = odw.live.to_numpy()
obs_v = odw[bkey].to_numpy()


def rho(v):
    """Spearman rank correlation of `v` against the observed OD flow, on live streets."""
    return spearmanr(v[live], obs_v[live]).statistic


def demand(beta, participation):
    """Run betweenness_demand and return the resulting nodes GeoDataFrame."""
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

t0 = time.perf_counter()
rho_beta = [rho(demand(b, 1.0)[dkey].to_numpy()) for b in BETAS]
stamp("beta_sweep", t0)
beta_star = BETAS[int(np.argmax(rho_beta))]
mode_share = float(sub["bicycle"].sum()) / float(sub["all"].sum())
if std is not None:
    print(f"uniform baseline rho={rho(std[bkey].to_numpy()):.3f}", flush=True)
for b, r in zip(BETAS, rho_beta, strict=True):
    print(f"  beta={b:<8} rho={r:.3f}", flush=True)
print(f"beta* = {beta_star} (spatial), participation* = mode share = {mode_share:.3f}", flush=True)

dem = demand(beta_star, max(mode_share, 0.01))
rho_final = rho(dem[dkey].to_numpy())
print(f"calibrated demand rho vs observed = {rho_final:.3f}", flush=True)

# %% Figure

b = study_poly.bounds if SCOPE == "london" else study_poly.buffer(-300).bounds


def draw(ax, gdf, col, title):
    liv = gdf[gdf.live].copy()
    lv = np.log1p(liv[col].to_numpy())
    lo, hi = np.quantile(lv, 0.20), np.quantile(lv, 0.99)
    t = np.clip((lv - lo) / (hi - lo + 1e-9), 0, 1) ** 1.1
    lw = (0.03 if SCOPE == "london" else 0.04) + (1.6 if SCOPE == "london" else 2.6) * t**1.5
    ax.set_facecolor("white")
    liv.plot(ax=ax, color="#e0e0e0", linewidth=0.15)
    order = np.argsort(t)
    liv.iloc[order].plot(ax=ax, color=plt.get_cmap("Reds")(0.10 + 0.90 * t[order]), linewidth=lw[order])
    ax.set_title(title, color="black", fontsize=12, loc="left")
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


t0 = time.perf_counter()
panels = [
    (odw, bkey, "OD-weighted (observed PCT bicycle)"),
    (dem, dkey, f"Demand-weighted (modelled · ρ={rho_final:.2f})"),
]
if std is not None:
    panels = [(std, bkey, "Standard betweenness (uniform)"), *panels]
fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 8), facecolor="white")
for ax, (gdf, col, title) in zip(np.atleast_1d(axes), panels, strict=True):
    draw(ax, gdf, col, title)
scope_label = "central London" if SCOPE == "central" else "Greater London"
fig.suptitle(
    f"cityseer · {scope_label} · OS Open Roads ({len(edges)} links) · PCT cycle commutes  —  "
    f"{DISTANCE // 1000} km · β={beta_star:g}, participation={mode_share:.2f}",
    color="black",
    fontsize=11,
    y=0.06,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = SCRIPT_DIR / f"pct_london_flows_{SCOPE}.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
stamp("figure", t0)

print("\n=== TIMINGS ===", flush=True)
core = timings["build_citynetwork"] + timings["build_od_matrix"] + timings["betweenness_od"] + timings["beta_sweep"]
print(f"build + OD + beta sweep ({len(BETAS)} demand runs): {core:.1f} s", flush=True)
print(f"total: {sum(timings.values()):.1f} s", flush=True)
print(f"saved {out}", flush=True)

# %%
