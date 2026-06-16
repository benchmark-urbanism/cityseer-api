"""
Fetch TIGER/Line road data for the Cary, NC validation network (third city).

Cary is a low-connectivity planned suburb (Raleigh / Research Triangle) — the
real-world analog of the synthetic "tree" topology, and the sparsest of the three
validation networks (cf. dense Greater London and Greater Madrid).

Strategy (mirrors the GLA "sub-region inside larger data" pattern):
  inner boundary  = Cary municipal boundary (marks live nodes)
  road-load mask  = Cary boundary buffered outward 20 km (= d_max)
  data            = every TIGER county that intersects the 20 km buffer

We use the TIGER **EDGES** layer (topologically integrated: every edge is split
at nodes, with explicit TNIDF/TNIDT node IDs and a ROADFLG to select roads), NOT
the ROADS layer. EDGES is natively noded, so the graph connects without any
planar-noding workaround, and it is correct at grade-separated crossings
(overpasses share no node) where planar noding would wrongly fuse them.

Because US counties tile the plane, the union of all counties intersecting the
buffer necessarily contains the buffer — so the data covers the road-mask by
construction (asserted below). Downloads are cached in temp/tiger_cary/.
"""

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import requests
from shapely.ops import unary_union

CRS_M = "EPSG:32119"  # NAD83 / North Carolina (metres)
BUFFER_M = 20_000
TEMP = Path(__file__).resolve().parents[3] / "temp" / "tiger_cary"
TEMP.mkdir(parents=True, exist_ok=True)

COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
EDGES_URL = "https://www2.census.gov/geo/tiger/TIGER2023/EDGES/tl_2023_{fips}_edges.zip"


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  downloading: {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"    -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> int:
    print("1. Cary boundary + 20 km buffer")
    cary = ox.geocode_to_gdf("Cary, North Carolina, USA").to_crs(CRS_M)
    boundary = cary.geometry.iloc[0]
    buffered = boundary.buffer(BUFFER_M)
    print(f"   inner area: {boundary.area / 1e6:.0f} km^2 | buffered: {buffered.area / 1e6:.0f} km^2")

    print("2. Counties intersecting the buffer")
    cb_zip = download(COUNTY_URL, TEMP / "cb_2023_us_county_500k.zip")
    counties = gpd.read_file(cb_zip).to_crs(CRS_M)
    hit = counties[counties.intersects(buffered)].copy()
    hit = hit.sort_values("GEOID")
    for _, c in hit.iterrows():
        print(f"   {c['GEOID']}  {c['NAMELSAD']}, {c['STUSPS'] if 'STUSPS' in c else c['STATE_NAME']}")

    # Coverage guarantee: union of intersecting counties must contain the buffer.
    covered = unary_union(hit.geometry).buffer(0)
    frac = buffered.intersection(covered).area / buffered.area
    print(f"   buffer covered by downloaded counties: {frac * 100:.2f}%")
    assert frac > 0.999, f"County set does not cover the buffer ({frac * 100:.2f}%) — check CRS / county file."

    print("3. Downloading TIGER edges (topological road network) per county")
    road_files = []
    for fips in hit["GEOID"]:
        rf = download(EDGES_URL.format(fips=fips), TEMP / f"tl_2023_{fips}_edges.zip")
        # validate the zip opens (catch truncated downloads)
        with zipfile.ZipFile(rf) as z:
            assert any(n.endswith(".shp") for n in z.namelist()), f"{rf.name}: no .shp inside"
        road_files.append(rf)

    print(f"\nDone. {len(road_files)} county edge files in {TEMP}")
    print("FIPS:", ", ".join(hit["GEOID"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
