"""
Fetch TIGER/Line road data for a US suburban validation network.

Generalised fetcher for the US validation networks (Cary, NC — the calibration
network — and The Woodlands, TX — the held-out validation network). Pass a
geocodable place name, a projected CRS in metres, and an output directory:

    python fetch_tiger.py --place "Cary, North Carolina, USA" \
        --crs EPSG:32119 --out tiger_cary
    python fetch_tiger.py --place "The Woodlands, Texas, USA" \
        --crs EPSG:26915 --out tiger_woodlands

Strategy (mirrors the GLA "sub-region inside larger data" pattern):
  inner boundary  = municipal boundary geocoded from OSM (marks live nodes)
  road-load mask  = inner boundary buffered outward 20 km (= d_max)
  data            = every TIGER county that intersects the 20 km buffer

We use the TIGER **EDGES** layer (topologically integrated: every edge is split
at nodes, with explicit TNIDF/TNIDT node IDs and a ROADFLG to select roads), NOT
the ROADS layer. EDGES is natively noded, so the graph connects without any
planar-noding workaround, and it is correct at grade-separated crossings
(overpasses share no node) where planar noding would wrongly fuse them.

Because US counties tile the plane, the union of all counties intersecting the
buffer necessarily contains the buffer — so the data covers the road-mask by
construction (asserted below). Downloads are cached in temp/<out>/.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import requests
from shapely.ops import unary_union

BUFFER_M = 20_000
TEMP_ROOT = Path(__file__).resolve().parents[3] / "temp"

COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
EDGES_URL = "https://www2.census.gov/geo/tiger/TIGER2023/EDGES/tl_2023_{fips}_edges.zip"


def download(url: str, dest: Path, attempts: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  downloading: {url}")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
            break
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            if attempt == attempts:
                raise
            wait = 10 * attempt
            print(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    print(f"    -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def fetch(place: str, crs: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"1. {place}: boundary + {BUFFER_M / 1000:.0f} km buffer")
    gdf = ox.geocode_to_gdf(place).to_crs(crs)
    boundary = gdf.geometry.iloc[0]
    buffered = boundary.buffer(BUFFER_M)
    print(f"   inner area: {boundary.area / 1e6:.0f} km^2 | buffered: {buffered.area / 1e6:.0f} km^2")

    print("2. Counties intersecting the buffer")
    cb_zip = download(COUNTY_URL, out_dir / "cb_2023_us_county_500k.zip")
    counties = gpd.read_file(cb_zip).to_crs(crs)
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
        rf = download(EDGES_URL.format(fips=fips), out_dir / f"tl_2023_{fips}_edges.zip")
        # validate the zip opens (catch truncated downloads)
        with zipfile.ZipFile(rf) as z:
            assert any(n.endswith(".shp") for n in z.namelist()), f"{rf.name}: no .shp inside"
        road_files.append(rf)

    print(f"\nDone. {len(road_files)} county edge files in {out_dir}")
    print("FIPS:", ", ".join(hit["GEOID"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch TIGER EDGES for a buffered suburban study area")
    parser.add_argument("--place", required=True, help='Geocodable place, e.g. "Cary, North Carolina, USA"')
    parser.add_argument("--crs", required=True, help="Projected CRS in metres, e.g. EPSG:32119")
    parser.add_argument("--out", required=True, help="Directory name under temp/, e.g. tiger_cary")
    args = parser.parse_args()
    return fetch(args.place, args.crs, TEMP_ROOT / args.out)


if __name__ == "__main__":
    sys.exit(main())
