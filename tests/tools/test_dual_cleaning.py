# pyright: basic
"""Primal cleaning ahead of dual conversion: filler welding, danglers, parallel merging.

Covers the 5.5 cleaning parameters on `dual.build_dual` / the CityNetwork constructors, the
legacy parameter set pinned by the QGIS plugin, and segment-length weighting for CityNetwork.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from cityseer import CityNetwork
from cityseer.tools import dual
from cityseer.tools.dual import LEGACY_CLEAN_PARAMS, _clean_geometries
from shapely.geometry import LineString


def _plus_with_filler() -> gpd.GeoDataFrame:
    """A plus-junction at (50, 0); the west arm is drawn as two collinear pieces."""
    return gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (30, 0)]),
            LineString([(30, 0), (50, 0)]),  # filler endpoint at (30, 0)
            LineString([(50, 0), (100, 0)]),
            LineString([(50, 0), (50, 60)]),
            LineString([(50, 0), (50, -60)]),
        ],
        index=["w1", "w2", "e", "n", "s"],
        crs=32630,
    )


def test_weld_fillers():
    geoms = {
        "a": LineString([(0, 0), (30, 0)]),
        "b": LineString([(30, 0), (50, 0)]),
        "c": LineString([(50, 0), (100, 0)]),
        "n": LineString([(50, 0), (50, 60)]),
        "s": LineString([(50, 0), (50, -60)]),
    }
    cleaned, statuses, merges = _clean_geometries(dict(geoms))
    # a+b weld at the degree-2 endpoint; (50, 0) is degree >= 3 and stays a junction
    assert statuses["b"] == "merged" and merges["b"] == "a"  # a is longer, so a is kept
    assert "b" not in cleaned
    assert cleaned["a"].length == pytest.approx(50.0)
    assert set(cleaned) == {"a", "c", "n", "s"}


def test_weld_transitive_chain():
    # three collinear pieces chain-weld; merges resolve transitively to the single kept id
    geoms = {
        "p1": LineString([(0, 0), (40, 0)]),
        "p2": LineString([(40, 0), (70, 0)]),
        "p3": LineString([(70, 0), (90, 0)]),
        "x": LineString([(90, 0), (90, 50)]),
        "y": LineString([(90, 0), (90, -50)]),
    }
    cleaned, statuses, merges = _clean_geometries(dict(geoms))
    kept = [fid for fid in ("p1", "p2", "p3") if fid in cleaned]
    assert len(kept) == 1
    assert cleaned[kept[0]].length == pytest.approx(90.0)
    for absorbed in {"p1", "p2", "p3"} - set(kept):
        assert statuses[absorbed] == "merged"
        assert merges[absorbed] == kept[0]


def test_weld_ring_guard():
    # two segments forming a closed loop must not weld into a ring; lengths are distinct so
    # the loop halves are genuine alternatives (not near-duplicates)
    geoms = {
        "long_way": LineString([(0, 0), (50, 50), (100, 0)]),
        "short_way": LineString([(0, 0), (50, -12), (100, 0)]),
        "tail": LineString([(100, 0), (200, 0)]),
    }
    cleaned, _statuses, merges = _clean_geometries(dict(geoms))
    # (0, 0) joins exactly the two loop halves, but welding would create a ring: skipped
    assert "long_way" in cleaned and "short_way" in cleaned
    assert not merges


def test_danglers_judged_on_welded_length():
    # a 12 m stub drawn as two 6 m pieces: welded first, so it survives the 10 m dangler cut
    geoms = {
        "m1": LineString([(0, 0), (50, 0)]),
        "m2": LineString([(50, 0), (100, 0)]),
        "v": LineString([(50, 0), (50, 40)]),
        "d1": LineString([(50, 0), (50, -6)]),
        "d2": LineString([(50, -6), (50, -12)]),
    }
    cleaned, statuses, _merges = _clean_geometries(dict(geoms))
    stub = [fid for fid in ("d1", "d2") if fid in cleaned]
    assert len(stub) == 1
    assert cleaned[stub[0]].length == pytest.approx(12.0)
    # without welding, each 6 m piece iteratively dangles away
    cleaned_nf, statuses_nf, _ = _clean_geometries(dict(geoms), remove_fillers=False)
    assert "d1" not in cleaned_nf and "d2" not in cleaned_nf
    assert statuses_nf["d2"] == "short_dangler"


def test_merge_parallel_tolerance():
    geoms = {
        "p1": LineString([(0, 0), (100, 0)]),
        "p2": LineString([(0, 1.5), (100, 1.5)]),  # twin: endpoints 1.5 m off, same length
        "alt": LineString([(0, 0), (50, 40), (100, 0)]),  # distinctly longer alternative
        "c1": LineString([(0, 0), (0, -80)]),
        "c2": LineString([(100, 0), (100, -80)]),
    }
    cleaned, statuses, _ = _clean_geometries(dict(geoms), remove_fillers=False)
    assert statuses["p2"] == "duplicate" and "p2" not in cleaned  # merged at the 2 m default
    assert "alt" in cleaned  # fails the near-identical length test: preserved
    # legacy tolerance (exact endpoint keys) keeps the 1.5 m twin
    cleaned_legacy, _s, _m = _clean_geometries(dict(geoms), remove_fillers=False, merge_parallel_dist=0.1)
    assert "p2" in cleaned_legacy
    # disabled entirely
    cleaned_off, _s, _m = _clean_geometries(dict(geoms), remove_fillers=False, merge_parallel_dist=0)
    assert "p2" in cleaned_off


def test_directed_skips_weld_and_parallel():
    geoms = {
        "a": LineString([(0, 0), (30, 0)]),
        "b": LineString([(30, 0), (50, 0)]),
        "p1": LineString([(50, 0), (150, 0)]),
        "p2": LineString([(50, 1), (150, 1)]),
        "n": LineString([(50, 0), (50, 60)]),
    }
    cleaned, _statuses, merges = _clean_geometries(dict(geoms), directed=True)
    assert set(cleaned) == set(geoms)
    assert not merges


def test_legacy_params_preserve_pre55_behaviour():
    gdf = _plus_with_filler()
    cn = CityNetwork.from_geopandas(gdf, **LEGACY_CLEAN_PARAMS)
    assert cn.node_count == 5  # no welding
    assert not (cn.feature_status == "merged").any()


def test_city_network_welds_by_default():
    gdf = _plus_with_filler()
    cn = CityNetwork.from_geopandas(gdf)
    assert cn.node_count == 4
    assert (cn.feature_status == "merged").sum() == 1
    kept = "w1" if "w1" in cn.nodes_gdf.index else "w2"
    assert cn.nodes_gdf.loc[kept, "seg_length"] == pytest.approx(50.0)
    # cleaning fully disabled passes the network through beyond tiny self-loop removal
    cn_raw = CityNetwork.from_geopandas(gdf, remove_fillers=False, remove_danglers=0, merge_parallel_dist=0)
    assert cn_raw.node_count == 5


def test_segment_weighted_any_construction_path():
    gdf = _plus_with_filler()
    # per-call flag: density becomes total reachable street length (excluding self)
    cn = CityNetwork.from_geopandas(gdf)
    cn.centrality_shortest(distances=[5000], closeness={"density": "1"}, betweenness={}, segment_weighted=True)
    total = cn.nodes_gdf["seg_length"].sum()
    for fid in cn.nodes_gdf.index:
        expected = total - cn.nodes_gdf.loc[fid, "seg_length"]
        assert cn.nodes_gdf.loc[fid, "cc_density_5000"] == pytest.approx(expected)
    # construction-time default applies without the per-call flag and matches
    cn2 = CityNetwork.from_geopandas(gdf, segment_weighted=True)
    cn2.centrality_shortest(distances=[5000], closeness={"density": "1"}, betweenness={})
    assert cn2.nodes_gdf["cc_density_5000"].tolist() == pytest.approx(cn.nodes_gdf["cc_density_5000"].tolist())
    # per-call override wins over the construction default
    cn3 = CityNetwork.from_geopandas(gdf, segment_weighted=True)
    cn3.centrality_shortest(distances=[5000], closeness={"density": "1"}, betweenness={}, segment_weighted=False)
    assert cn3.nodes_gdf["cc_density_5000"].max() <= cn.node_count  # unit counts, not metres


def test_update_after_welds_full_rebuild():
    gdf = _plus_with_filler()
    cn = CityNetwork.from_geopandas(gdf)
    assert cn.node_count == 4
    gdf_upd = gdf.copy()
    gdf_upd.loc["n", "geometry"] = LineString([(50, 0), (50, 90)])
    cn.update(gdf_upd)
    assert cn.node_count == 4
    assert cn.nodes_gdf.loc["n", "seg_length"] == pytest.approx(90.0)  # not stale after rebuild
    assert (cn.feature_status == "merged").sum() == 1  # weld re-applied on rebuild


def test_welded_impedance_is_length_weighted_mean():
    # a welded chain inherits the length-weighted mean of its constituents' impedances,
    # observable on the dual edges joining the welded segment to the cross streets
    geoms = {
        "a": LineString([(0, 0), (100, 0)]),
        "b": LineString([(100, 0), (300, 0)]),  # welds with "a" at (100, 0)
        "n": LineString([(300, 0), (300, 200)]),
        "s": LineString([(300, 0), (300, -200)]),  # (300, 0) is degree 3: junction kept
    }
    impedances = {"a": 2.0, "b": 4.0, "n": 1.0, "s": 1.0}
    _ns, _nodes, state = dual.build_dual(geoms, crs=32630, impedances=impedances)
    welded_fid = next(iter(set(state["merges"].values())))
    welded_imp = (100.0 * 2.0 + 200.0 * 4.0) / 300.0  # length-weighted mean of a + b
    # a dual edge combines its two segments' impedances by length-weighted mean
    welded_len, n_len = 300.0, 200.0
    expected_edge_imp = (welded_len * welded_imp + n_len * 1.0) / (welded_len + n_len)
    edge_imps = [
        rec["imp_factor"]
        for rec in state["edge_records"].values()
        if welded_fid in (rec["start_key"], rec["end_key"]) and "n" in (rec["start_key"], rec["end_key"])
    ]
    assert edge_imps and all(imp == pytest.approx(expected_edge_imp) for imp in edge_imps)
    # the state keeps the caller's input impedances (no compounding across rebuild cycles)
    assert state["impedances"] == impedances


def test_build_dual_state_carries_clean_params():
    gdf = _plus_with_filler()
    _ns, _nodes, state = dual.build_dual(gdf, crs=32630)
    assert state["clean_params"] == {
        "remove_fillers": True,
        "remove_danglers": 10.0,
        "merge_parallel_dist": 2.0,
    }
    assert state["merges"]  # the filler weld is recorded
    _ns2, _nodes2, state2 = dual.build_dual(gdf, crs=32630, **LEGACY_CLEAN_PARAMS)
    assert state2["clean_params"] == LEGACY_CLEAN_PARAMS
    assert not state2["merges"]
