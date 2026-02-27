"""
Profile build_dual_network by mocking the minimal QGIS layer interface.

Usage:
    uv run python qgis_plugin/profile_build.py [--gpkg PATH] [--limit N] [--cprofile]

Loads line geometries from a geopackage, wraps them in a mock QGIS layer,
and times each phase of build_dual_network.
"""

from __future__ import annotations

import argparse
import time


# ---------------------------------------------------------------------------
# Minimal QGIS mocks — just enough for build_dual_network
# ---------------------------------------------------------------------------
class _MockCrs:
    def authid(self):
        return "EPSG:32630"

    def isValid(self):
        return True

    def isGeographic(self):
        return False


class _MockGeometry:
    def __init__(self, wkt: str):
        self._wkt = wkt

    def isEmpty(self):
        return not self._wkt

    def asWkt(self):
        return self._wkt


class _MockFeature:
    def __init__(self, fid: int, wkt: str):
        self._fid = fid
        self._wkt = wkt

    def id(self):
        return self._fid

    def geometry(self):
        return _MockGeometry(self._wkt)


class _MockLayer:
    def __init__(self, wkts: dict[int, str]):
        self._wkts = wkts

    def id(self):
        return "mock_layer"

    def crs(self):
        return _MockCrs()

    def wkbType(self):
        return 2  # QgsWkbTypes.Type.LineString

    def featureCount(self):
        return len(self._wkts)

    def getFeatures(self):
        for fid, wkt in self._wkts.items():
            yield _MockFeature(fid, wkt)


# ---------------------------------------------------------------------------
# Granular timing: instrument each phase
# ---------------------------------------------------------------------------
def profile_phases(wkts: dict[int, str]):
    """Time each phase of build_dual_network individually."""
    import collections
    import itertools

    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    from cityseer import rustalgos

    from cityseer_qgis.utils.converters import (
        _coords_to_wkt,
        _cumulative_lengths,
        _interpolate_at,
        _substring_coords,
    )

    def _parse_line(wkt_str):
        geom = shapely_wkt.loads(wkt_str)
        if geom.is_empty:
            return None
        if geom.geom_type == "MultiLineString":
            merged = linemerge(geom)
            if merged.geom_type == "MultiLineString":
                geom = max(merged.geoms, key=lambda g: g.length)
            else:
                geom = merged
        if geom.geom_type != "LineString" or len(geom.coords) < 2:
            return None
        if geom.has_z:
            geom = LineString([(c[0], c[1]) for c in geom.coords])
        return geom

    def _ep_key(pt):
        return (round(pt[0], 1), round(pt[1], 1))

    def _half_toward_coords(fid, endpoint):
        coords, cum = line_data[fid]
        ep = (round(endpoint[0], 1), round(endpoint[1], 1))
        end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        if ep == end:
            return _substring_coords(coords, cum, 0.5, 1.0)
        rev = coords[::-1]
        rev_cum = _cumulative_lengths(rev)
        return _substring_coords(rev, rev_cum, 0.5, 1.0)

    def _make_edge_wkt(fid_a, fid_b, endpoint_key):
        ha = _half_toward_coords(fid_a, endpoint_key)
        hb = _half_toward_coords(fid_b, endpoint_key)
        hb_rev = hb[::-1]
        merged = ha + hb_rev[1:]
        return _coords_to_wkt(merged)

    timings = {}

    # Phase 1: Parse WKT → shapely
    t0 = time.perf_counter()
    geoms = {}
    fid_list = []
    for fid, wkt_str in wkts.items():
        line = _parse_line(wkt_str)
        if line is None:
            continue
        geoms[fid] = line
        fid_list.append(fid)
    timings["1_parse_wkt_to_shapely"] = time.perf_counter() - t0
    print(f"  Parsed {len(geoms)} lines from {len(wkts)} WKTs")

    # Phase 2: Build endpoint adjacency + compute midpoints (pure-Python)
    t0 = time.perf_counter()
    endpoint_to_fids = collections.defaultdict(list)
    midpoints = {}
    line_data = {}
    for fid, line in geoms.items():
        coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(coords)
        line_data[fid] = (coords, cum)
        for pt in (coords[0], coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)
        mid = _interpolate_at(coords, cum, 0.5)
        midpoints[fid] = mid
    timings["2_endpoints_and_midpoints"] = time.perf_counter() - t0

    # Phase 3: Add nodes to NetworkStructure
    t0 = time.perf_counter()
    ns = rustalgos.graph.NetworkStructure()
    node_idx = {}
    for fid in fid_list:
        mx, my = midpoints[fid]
        idx = ns.add_street_node(node_key=fid, x=mx, y=my, live=True, weight=1.0)
        node_idx[fid] = idx
    timings["3_add_nodes"] = time.perf_counter() - t0

    # Phase 4a: Build edge WKTs (pure-Python coordinate arithmetic)
    t0 = time.perf_counter()
    edge_wkts = []
    seen = set()
    for endpoint, fids in endpoint_to_fids.items():
        for fid_a, fid_b in itertools.combinations(fids, 2):
            pair = frozenset({fid_a, fid_b})
            if pair in seen:
                continue
            seen.add(pair)
            wkt_fwd = _make_edge_wkt(fid_a, fid_b, endpoint)
            wkt_rev = _make_edge_wkt(fid_b, fid_a, endpoint)
            edge_wkts.append((fid_a, fid_b, wkt_fwd, wkt_rev))
    timings["4a_build_edge_wkts"] = time.perf_counter() - t0
    print(f"  Built {len(edge_wkts)} edge pairs ({len(edge_wkts)*2} directed)")

    # Phase 4b: Add edges to NetworkStructure (Rust WKT parsing)
    t0 = time.perf_counter()
    edge_counter = 0
    for fid_a, fid_b, wkt_fwd, wkt_rev in edge_wkts:
        ns.add_street_edge(
            node_idx[fid_a], node_idx[fid_b],
            edge_counter, fid_a, fid_b, wkt_fwd,
        )
        edge_counter += 1
        ns.add_street_edge(
            node_idx[fid_b], node_idx[fid_a],
            edge_counter, fid_b, fid_a, wkt_rev,
        )
        edge_counter += 1
    timings["4b_add_edges_rust"] = time.perf_counter() - t0

    # Phase 5: Validate + build R-tree
    t0 = time.perf_counter()
    ns.validate()
    ns.build_edge_rtree()
    timings["5_validate_and_rtree"] = time.perf_counter() - t0

    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile build_dual_network")
    parser.add_argument("--gpkg", default="temp/network.gpkg", help="Path to geopackage")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of features (0=all)")
    parser.add_argument("--cprofile", action="store_true", help="Also run cProfile on full build")
    args = parser.parse_args()

    # Load geometries via geopandas
    import geopandas as gpd

    print(f"Loading {args.gpkg}...")
    gdf = gpd.read_file(args.gpkg)
    if args.limit > 0:
        gdf = gdf.head(args.limit)
    print(f"Loaded {len(gdf)} features")

    # Extract WKTs (simulating what QGIS layer.getFeatures() + asWkt() produces)
    wkts = {}
    for i, row in enumerate(gdf.itertuples()):
        wkts[i] = row.geometry.wkt
    print(f"Extracted {len(wkts)} WKTs\n")

    # Phase-by-phase timing
    print("=== Phase-by-phase timing ===")
    timings = profile_phases(wkts)
    total = sum(timings.values())
    for phase, t in timings.items():
        pct = t / total * 100 if total > 0 else 0
        print(f"  {phase:35s}  {t:8.3f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':35s}  {total:8.3f}s")

    # Full build_dual_network via mock layer
    print("\n=== Full build_dual_network (end-to-end) ===")
    from cityseer_qgis.utils.converters import _inc_state, build_dual_network

    # Reset cache
    import cityseer_qgis.utils.converters as conv_mod
    conv_mod._inc_state = None

    layer = _MockLayer(wkts)
    t0 = time.perf_counter()
    ns, fid_list, midpoints, geoms = build_dual_network(layer)
    t_full = time.perf_counter() - t0
    print(f"  Full build: {t_full:.3f}s  ({ns.street_node_count()} nodes)")

    # Second run (cached, no changes)
    t0 = time.perf_counter()
    ns2, _, _, _ = build_dual_network(layer)
    t_cached = time.perf_counter() - t0
    print(f"  Cached run:  {t_cached:.3f}s")

    # cProfile
    if args.cprofile:
        import cProfile
        import pstats

        conv_mod._inc_state = None
        print("\n=== cProfile (top 30 by tottime) ===")
        profiler = cProfile.Profile()
        profiler.enable()
        build_dual_network(layer)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
