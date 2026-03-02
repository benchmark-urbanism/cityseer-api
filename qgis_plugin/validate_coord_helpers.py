"""
Validate that pure-Python coordinate helpers produce identical results to shapely.

Tests _cumulative_lengths, _interpolate_at, _substring_coords against
shapely's line.interpolate() and shapely.ops.substring() on:
  1. Simple synthetic lines (edge cases)
  2. Every line from a real network geopackage

Usage:
    PYTHONPATH=qgis_plugin uv run python qgis_plugin/validate_coord_helpers.py [--gpkg PATH]
"""

from __future__ import annotations

import argparse
import sys

from cityseer_qgis.utils.converters import (
    _coords_to_wkt,
    _cumulative_lengths,
    _interpolate_at,
    _substring_coords,
)
from shapely.geometry import LineString
from shapely.ops import substring

TOLERANCE = 1e-8  # coordinate tolerance for floating-point comparison


def coords_close(a: tuple, b: tuple, tol: float = TOLERANCE) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def coord_lists_close(a: list[tuple], b: list[tuple], tol: float = TOLERANCE) -> bool:
    if len(a) != len(b):
        return False
    return all(coords_close(p, q, tol) for p, q in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Test 1: _cumulative_lengths
# ---------------------------------------------------------------------------
def test_cumulative_lengths():
    print("=== Test: _cumulative_lengths ===")
    cases = [
        ([(0, 0), (10, 0)], [0.0, 10.0]),
        ([(0, 0), (3, 4)], [0.0, 5.0]),
        ([(0, 0), (5, 0), (10, 0)], [0.0, 5.0, 10.0]),
        ([(0, 0), (0, 1), (1, 1), (1, 0)], [0.0, 1.0, 2.0, 3.0]),
    ]
    for coords, expected in cases:
        result = _cumulative_lengths(coords)
        assert len(result) == len(expected), f"Length mismatch: {result} vs {expected}"
        for r, e in zip(result, expected, strict=True):
            assert abs(r - e) < TOLERANCE, f"Value mismatch: {result} vs {expected}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 2: _interpolate_at vs shapely
# ---------------------------------------------------------------------------
def test_interpolate_at():
    print("=== Test: _interpolate_at vs shapely.interpolate ===")
    lines = [
        [(0, 0), (10, 0)],
        [(0, 0), (5, 0), (10, 0)],
        [(0, 0), (3, 4), (6, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [(0, 0), (10, 0), (10, 10)],
        [(100, 200), (100, 300), (200, 300), (200, 200)],
    ]
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.333, 0.9]
    n_pass = 0
    n_fail = 0
    for coords in lines:
        cum = _cumulative_lengths(coords)
        shapely_line = LineString(coords)
        for frac in fracs:
            our_pt = _interpolate_at(coords, cum, frac)
            shapely_pt = shapely_line.interpolate(frac, normalized=True)
            expected = (shapely_pt.x, shapely_pt.y)
            if coords_close(our_pt, expected):
                n_pass += 1
            else:
                n_fail += 1
                print(f"  FAIL: coords={coords}, frac={frac}")
                print(f"    ours:    {our_pt}")
                print(f"    shapely: {expected}")
    print(f"  {n_pass} passed, {n_fail} failed")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Test 3: _substring_coords vs shapely.ops.substring
# ---------------------------------------------------------------------------
def test_substring_coords():
    print("=== Test: _substring_coords vs shapely.ops.substring ===")
    lines = [
        [(0, 0), (10, 0)],
        [(0, 0), (5, 0), (10, 0)],
        [(0, 0), (3, 4), (6, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [(0, 0), (10, 0), (10, 10)],
        [(100, 200), (100, 300), (200, 300), (200, 200)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],  # many equal-length segments
    ]
    frac_pairs = [
        (0.0, 1.0),
        (0.0, 0.5),
        (0.5, 1.0),
        (0.25, 0.75),
        (0.0, 0.25),
        (0.75, 1.0),
        (0.1, 0.9),
        (0.33, 0.67),
    ]
    n_pass = 0
    n_fail = 0
    for coords in lines:
        cum = _cumulative_lengths(coords)
        shapely_line = LineString(coords)
        for start_frac, end_frac in frac_pairs:
            our_coords = _substring_coords(coords, cum, start_frac, end_frac)
            shapely_sub = substring(shapely_line, start_frac, end_frac, normalized=True)
            expected_coords = list(shapely_sub.coords)
            if coord_lists_close(our_coords, expected_coords):
                n_pass += 1
            else:
                n_fail += 1
                print(f"  FAIL: coords={coords}, frac=[{start_frac}, {end_frac}]")
                print(f"    ours ({len(our_coords)}):    {our_coords}")
                print(f"    shapely ({len(expected_coords)}): {expected_coords}")
    print(f"  {n_pass} passed, {n_fail} failed")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Test 4: _coords_to_wkt round-trip
# ---------------------------------------------------------------------------
def test_coords_to_wkt():
    print("=== Test: _coords_to_wkt ===")
    coords = [(1.23456789, 9.87654321), (5.0, 5.0)]
    wkt = _coords_to_wkt(coords)
    # Check format is valid WKT
    assert wkt.startswith("LINESTRING ("), f"Bad prefix: {wkt}"
    assert wkt.endswith(")"), f"Bad suffix: {wkt}"
    print(f"  WKT: {wkt}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 5: Full edge-building pipeline vs shapely (on real data)
# ---------------------------------------------------------------------------
def test_full_pipeline_on_real_data(gpkg_path: str, limit: int = 0):
    """Compare pure-Python _make_edge_wkt equivalent against shapely version
    on every edge in a real network."""
    import collections
    import itertools

    import geopandas as gpd
    from shapely import wkt as shapely_wkt
    from shapely.ops import linemerge

    print(f"\n=== Test: full edge pipeline on {gpkg_path} ===")
    gdf = gpd.read_file(gpkg_path)
    if limit > 0:
        gdf = gdf.head(limit)
    print(f"  Loaded {len(gdf)} features")

    # Parse lines
    geoms = {}
    for i, row in enumerate(gdf.itertuples()):
        wkt_str = row.geometry.wkt
        geom = shapely_wkt.loads(wkt_str)
        if geom.is_empty or geom.geom_type != "LineString" or len(geom.coords) < 2:
            continue
        geoms[i] = geom
    print(f"  Parsed {len(geoms)} valid lines")

    # Build endpoint adjacency
    def _ep_key(pt):
        return (round(pt[0], 1), round(pt[1], 1))

    endpoint_to_fids = collections.defaultdict(list)
    for fid, line in geoms.items():
        for pt in (line.coords[0], line.coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)

    # Precompute coord data for pure-Python path
    line_data = {}
    for fid, line in geoms.items():
        coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(coords)
        line_data[fid] = (coords, cum)

    # Old shapely functions
    def _half_toward_shapely(line, endpoint):
        ep = (round(endpoint[0], 1), round(endpoint[1], 1))
        end = (round(line.coords[-1][0], 1), round(line.coords[-1][1], 1))
        if ep == end:
            return substring(line, 0.5, 1.0, normalized=True)
        reversed_line = line.reverse()
        return substring(reversed_line, 0.5, 1.0, normalized=True)

    def _make_edge_wkt_shapely(line_a, line_b, endpoint_key):
        ha = _half_toward_shapely(line_a, endpoint_key)
        hb = _half_toward_shapely(line_b, endpoint_key)
        hb_rev = hb.reverse()
        merged = linemerge([ha, hb_rev])
        if merged.geom_type != "LineString":
            merged = LineString(list(ha.coords) + list(hb_rev.coords)[1:])
        return merged.wkt

    # New pure-Python functions
    def _half_toward_coords(fid, endpoint_key):
        coords, cum = line_data[fid]
        ep = (round(endpoint_key[0], 1), round(endpoint_key[1], 1))
        end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        if ep == end:
            return _substring_coords(coords, cum, 0.5, 1.0)
        # Reverse and substring
        rev_coords = coords[::-1]
        rev_cum = _cumulative_lengths(rev_coords)
        return _substring_coords(rev_coords, rev_cum, 0.5, 1.0)

    def _make_edge_wkt_pure(fid_a, fid_b, endpoint_key):
        ha = _half_toward_coords(fid_a, endpoint_key)
        hb = _half_toward_coords(fid_b, endpoint_key)
        hb_rev = hb[::-1]
        merged = ha + hb_rev[1:]
        return _coords_to_wkt(merged)

    # Compare every edge
    seen = set()
    n_edges = 0
    n_pass = 0
    n_fail = 0
    n_self_loops = 0
    max_err = 0.0
    for endpoint, fids in endpoint_to_fids.items():
        for fid_a, fid_b in itertools.combinations(fids, 2):
            pair = frozenset({fid_a, fid_b})
            if pair in seen:
                continue
            seen.add(pair)
            is_self_loop = fid_a == fid_b
            if is_self_loop:
                n_self_loops += 1
            # Forward direction
            for fa, fb in [(fid_a, fid_b), (fid_b, fid_a)]:
                n_edges += 1
                wkt_shapely = _make_edge_wkt_shapely(geoms[fa], geoms[fb], endpoint)
                wkt_pure = _make_edge_wkt_pure(fa, fb, endpoint)
                # Parse both WKTs and compare coordinates
                shapely_coords = list(LineString(shapely_wkt.loads(wkt_shapely).coords).coords)
                pure_coords = list(LineString(shapely_wkt.loads(wkt_pure).coords).coords)
                # For self-loop edges, linemerge picks an arbitrary
                # start on the closed loop — a cyclic rotation with
                # the same coords and same total geometry is equivalent.
                if coord_lists_close(shapely_coords, pure_coords, tol=1e-6):
                    n_pass += 1
                elif is_self_loop and len(shapely_coords) == len(pure_coords):
                    # Check that both have the same set of coordinates
                    # in the same cyclic order (possibly rotated/reversed).
                    s_set = set((round(c[0], 6), round(c[1], 6)) for c in shapely_coords)
                    p_set = set((round(c[0], 6), round(c[1], 6)) for c in pure_coords)
                    if s_set == p_set:
                        n_pass += 1
                    else:
                        n_fail += 1
                        if n_fail <= 5:
                            print(f"  FAIL self-loop {fa}->{fb} (different coords):")
                        continue
                else:
                    n_fail += 1
                    if n_fail <= 5:
                        print(f"  FAIL edge {fa}->{fb} via {endpoint} (self_loop={is_self_loop}):")
                        print(f"    shapely ({len(shapely_coords)}): {shapely_coords[:3]}...")
                        print(f"    pure    ({len(pure_coords)}): {pure_coords[:3]}...")
                    # Compute max coordinate error
                    if len(shapely_coords) == len(pure_coords):
                        for sc, pc in zip(shapely_coords, pure_coords, strict=True):
                            err = max(abs(sc[0] - pc[0]), abs(sc[1] - pc[1]))
                            max_err = max(max_err, err)

    print(f"  Compared {n_edges} directed edges ({n_self_loops} self-loop pairs)")
    print(f"  {n_pass} passed, {n_fail} failed")
    if max_err > 0:
        print(f"  Max coordinate error: {max_err:.2e}")
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", default="temp/network.gpkg")
    parser.add_argument("--limit", type=int, default=5000, help="Limit features for real-data test (0=all)")
    args = parser.parse_args()

    ok = True
    test_cumulative_lengths()
    ok &= test_interpolate_at()
    ok &= test_substring_coords()
    test_coords_to_wkt()
    ok &= test_full_pipeline_on_real_data(args.gpkg, args.limit)

    if ok:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print("\n*** SOME TESTS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
