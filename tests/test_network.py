from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from cityseer import CityNetwork
from cityseer.tools import io
from pyproj import CRS
from shapely.geometry import LineString, Polygon


def _simple_streets_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (20, 0)]),
                LineString([(20, 0), (40, 0)]),
                LineString([(20, 0), (20, 20)]),
            ]
        },
        index=["a", "b", "c"],
        crs=CRS(32630),
    )


def test_citynetwork_lazy_import():
    from cityseer import CityNetwork as ImportedCityNetwork

    assert ImportedCityNetwork is CityNetwork


def test_from_nx_matches_from_geopandas(primal_graph):
    streets_gdf = io.geopandas_from_nx(primal_graph)
    streets_gdf = streets_gdf.copy()

    def _edge_key(row):
        a, b = str(row.start_nd_key), str(row.end_nd_key)
        return f"{min(a, b)}_{max(a, b)}_k{int(row.edge_idx)}"

    streets_gdf.index = [_edge_key(row) for _, row in streets_gdf.iterrows()]
    from_gdf = CityNetwork.from_geopandas(streets_gdf).centrality_simplest(distances=[400])
    from_nx = CityNetwork.from_nx(primal_graph).centrality_simplest(distances=[400])

    for column in [
        "cc_density_400_ang",
        "cc_harmonic_400_ang",
        "cc_farness_400_ang",
        "cc_hillier_400_ang",
        "cc_betweenness_400_ang",
    ]:
        np.testing.assert_allclose(
            from_gdf.nodes_gdf.sort_index()[column],
            from_nx.nodes_gdf.sort_index()[column],
        )


def test_set_boundary_updates_live_flags():
    streets_gdf = _simple_streets_gdf()
    city_network = CityNetwork.from_geopandas(streets_gdf)
    boundary = Polygon([(-1, -1), (19, -1), (19, 1), (-1, 1)])

    city_network.set_boundary(boundary)

    assert city_network.nodes_gdf["live"].to_dict() == {"a": True, "b": False, "c": False}
    city_network.set_all_live()
    assert city_network.nodes_gdf["live"].to_dict() == {"a": True, "b": True, "c": True}


def test_incremental_update_preserves_unchanged_indices():
    streets_gdf = _simple_streets_gdf()
    city_network = CityNetwork.from_geopandas(streets_gdf)
    before = city_network.nodes_gdf["ns_node_idx"].to_dict()

    updated_gdf = streets_gdf.copy()
    updated_gdf.at["b", "geometry"] = LineString([(20, 0), (60, 0)])
    city_network.update(updated_gdf)

    after = city_network.nodes_gdf["ns_node_idx"].to_dict()
    assert after["a"] == before["a"]
    assert after["c"] == before["c"]
    assert city_network.nodes_gdf.at["b", "x"] == 40.0


def test_save_load_roundtrip_preserves_metrics_and_fast_state(tmp_path):
    streets_gdf = _simple_streets_gdf()
    path = tmp_path / "city_network"
    city_network = CityNetwork.from_geopandas(streets_gdf).centrality_simplest(distances=[50])
    city_network.save(path)

    loaded = CityNetwork.load(path)

    np.testing.assert_allclose(
        loaded.nodes_gdf["cc_density_50_ang"].sort_index(),
        city_network.nodes_gdf["cc_density_50_ang"].sort_index(),
    )
    updated_gdf = streets_gdf.copy()
    updated_gdf.at["c", "geometry"] = LineString([(20, 0), (20, 40)])
    loaded.update(updated_gdf)
    assert loaded.nodes_gdf.at["c", "y"] == 20.0


def test_to_nx_exports_primal_graph():
    city_network = CityNetwork.from_geopandas(_simple_streets_gdf())

    exported = city_network.to_nx()

    assert "is_dual" not in exported.graph
    assert exported.number_of_nodes() == 4
    assert exported.number_of_edges() == 3
    edge_statuses = [data["feature_status"] for _, _, data in exported.edges(data=True)]
    assert edge_statuses == ["active", "active", "active"]


def test_cleanup_thresholds_use_min_self_loop_and_narrow_duplicate_ratio():
    wkts = {
        "near_base": LineString([(0, 0), (100, 0)]).wkt,
        "near_bent": LineString([(0, 0), (50, 10), (100, 0)]).wkt,
        "wide_base": LineString([(0, 20), (100, 20)]).wkt,
        "wide_bent": LineString([(0, 20), (50, 45), (100, 20)]).wkt,
        "short_loop": LineString([(0, 40), (0.4, 40), (0, 40)]).wkt,
        "long_loop": LineString([(0, 60), (1, 60), (1, 61), (0, 61), (0, 60)]).wkt,
    }
    city_network = CityNetwork.from_wkts(wkts, crs=CRS(32630))

    assert city_network.feature_status["short_loop"] == "short_self_loop"
    assert city_network.feature_status["long_loop"] == "active"
    assert city_network.feature_status["near_base"] == "duplicate"
    assert city_network.feature_status["near_bent"] == "active"
    assert city_network.feature_status["wide_base"] == "active"
    assert city_network.feature_status["wide_bent"] == "active"


def test_cleaned_and_deleted_features_are_tagged():
    wkts = {
        "valid": LineString([(0, 0), (20, 0)]).wkt,
        "invalid": "LINESTRING EMPTY",
    }
    city_network = CityNetwork.from_wkts(wkts, crs=CRS(32630))

    assert city_network.feature_status["valid"] == "active"
    assert city_network.feature_status["invalid"] == "invalid_geometry"

    city_network.update({"invalid": LineString([(0, 0), (20, 0)]).wkt})

    assert city_network.feature_status["valid"] == "deleted"
    assert city_network.feature_status["invalid"] == "active"


# --- Directed graph tests ---


def _directed_streets_gdf() -> gpd.GeoDataFrame:
    """A simple T-junction with one-way on the horizontal segment 'a'.

    Layout:
        (0,0) --a--> (20,0) --b-- (40,0)
                        |
                        c
                        |
                      (20,20)

    'a' is one-way left-to-right. 'b' and 'c' are two-way.
    """
    return gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (20, 0)]),
                LineString([(20, 0), (40, 0)]),
                LineString([(20, 0), (20, 20)]),
            ],
            "oneway": [True, False, False],
        },
        index=["a", "b", "c"],
        crs=CRS(32630),
    )


def test_directed_from_geopandas():
    """Directed network has fewer dual edges than undirected for the same topology."""
    gdf = _directed_streets_gdf()
    undirected = CityNetwork.from_geopandas(gdf)
    directed = CityNetwork.from_geopandas(gdf, directed=True)

    assert directed.is_directed is True
    assert undirected.is_directed is False
    # Directed should have fewer edges due to one-way constraint on 'a'
    assert directed.network_structure.edge_count < undirected.network_structure.edge_count


def test_directed_from_geopandas_missing_column():
    """ValueError when directed=True but no 'oneway' column."""
    gdf = _simple_streets_gdf()
    import pytest

    with pytest.raises(ValueError, match="oneway"):
        CityNetwork.from_geopandas(gdf, directed=True)


def test_directed_from_geopandas_bad_oneway_dtype():
    """TypeError when oneway column has non-boolean values."""
    import pytest

    gdf = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (20, 0)])], "oneway": ["yes"]},
        index=["a"],
        crs=CRS(32630),
    )
    with pytest.raises(TypeError, match="boolean"):
        CityNetwork.from_geopandas(gdf, directed=True)


def test_directed_from_wkts():
    """Directed network from WKTs with oneway_fids."""
    wkts = {
        "a": LineString([(0, 0), (20, 0)]).wkt,
        "b": LineString([(20, 0), (40, 0)]).wkt,
        "c": LineString([(20, 0), (20, 20)]).wkt,
    }
    undirected = CityNetwork.from_wkts(wkts, crs=CRS(32630))
    directed = CityNetwork.from_wkts(wkts, crs=CRS(32630), directed=True, oneway_fids={"a"})

    assert directed.is_directed is True
    assert directed.network_structure.edge_count < undirected.network_structure.edge_count


def test_directed_from_wkts_missing_oneway_fids():
    """ValueError when directed=True but oneway_fids not provided."""
    import pytest

    wkts = {"a": LineString([(0, 0), (20, 0)]).wkt}
    with pytest.raises(ValueError, match="oneway_fids"):
        CityNetwork.from_wkts(wkts, crs=CRS(32630), directed=True)


def test_directed_repr():
    """Directed flag appears in repr."""
    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    assert "is_directed=True" in repr(cn)


def test_directed_backward_compat():
    """Default directed=False produces identical results to undirected construction."""
    gdf = _simple_streets_gdf()
    cn_default = CityNetwork.from_geopandas(gdf)
    cn_explicit = CityNetwork.from_geopandas(gdf, directed=False)

    assert cn_default.is_directed is False
    assert cn_explicit.is_directed is False
    assert cn_default.network_structure.edge_count == cn_explicit.network_structure.edge_count


def test_directed_from_nx_multidigraph():
    """Auto-detect directed mode from MultiDiGraph."""
    import networkx as nx

    # A -> B (one-way), B <-> C (two-way = two directed edges)
    G = nx.MultiDiGraph()
    G.graph["crs"] = CRS(32630)
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=20.0, y=0.0)
    G.add_node("C", x=40.0, y=0.0)
    G.add_edge("A", "B", key=0, geom=LineString([(0, 0), (20, 0)]))
    G.add_edge("B", "C", key=0, geom=LineString([(20, 0), (40, 0)]))
    G.add_edge("C", "B", key=0, geom=LineString([(40, 0), (20, 0)]))

    cn = CityNetwork.from_nx(G)
    assert cn.is_directed is True
    # 3 directed edges -> 3 dual nodes, each one-way
    assert cn.node_count == 3


def test_directed_from_nx_preserves_distinct_edges():
    """Opposite-direction edges with the same key are kept as separate dual nodes."""
    import networkx as nx

    G = nx.MultiDiGraph()
    G.graph["crs"] = CRS(32630)
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=100.0, y=0.0)
    # Two distinct edges sharing key=0 but with different attributes
    G.add_edge("A", "B", key=0, geom=LineString([(0, 0), (100, 0)]), name="forward_road")
    G.add_edge("B", "A", key=0, geom=LineString([(100, 0), (50, 10), (0, 0)]), name="reverse_road")

    cn = CityNetwork.from_nx(G)
    assert cn.is_directed is True
    # Both edges must be preserved as separate dual nodes
    assert cn.node_count == 2


def test_directed_update_new_feature_with_oneway():
    """Adding a one-way feature via GeoDataFrame update respects oneway column."""
    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)

    # Add a new one-way street 'd'
    updated_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (20, 0)]),
                LineString([(20, 0), (40, 0)]),
                LineString([(20, 0), (20, 20)]),
                LineString([(40, 0), (60, 0)]),
            ],
            "oneway": [True, False, False, True],
        },
        index=["a", "b", "c", "d"],
        crs=CRS(32630),
    )
    cn.update(updated_gdf)

    assert cn.is_directed is True
    assert cn.node_count == 4


def test_directed_update_flip_oneway():
    """Changing oneway status without geometry change triggers rebuild."""
    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    edge_count_oneway = cn.network_structure.edge_count

    # Flip 'a' from one-way to two-way (same geometry)
    updated_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (20, 0)]),
                LineString([(20, 0), (40, 0)]),
                LineString([(20, 0), (20, 20)]),
            ],
            "oneway": [False, False, False],  # 'a' now two-way
        },
        index=["a", "b", "c"],
        crs=CRS(32630),
    )
    cn.update(updated_gdf)

    assert cn.is_directed is True
    # More edges now that 'a' is two-way
    assert cn.network_structure.edge_count > edge_count_oneway


def test_directed_update_missing_oneway_column():
    """ValueError when updating a directed network with a GeoDataFrame missing oneway."""
    import pytest

    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    bad_gdf = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (20, 0)])]},
        index=["a"],
        crs=CRS(32630),
    )
    with pytest.raises(ValueError, match="oneway"):
        cn.update(bad_gdf)


def test_directed_update_bad_oneway_dtype():
    """TypeError when updating a directed network with non-boolean oneway values."""
    import pytest

    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    bad_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (20, 0)]),
                LineString([(20, 0), (40, 0)]),
                LineString([(20, 0), (20, 20)]),
            ],
            "oneway": ["False", "False", "False"],
        },
        index=["a", "b", "c"],
        crs=CRS(32630),
    )
    with pytest.raises(TypeError, match="boolean"):
        cn.update(bad_gdf)


def test_directed_to_nx_raises_without_source_graph():
    """to_nx() raises NotImplementedError for directed networks without a source graph."""
    import pytest

    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    with pytest.raises(NotImplementedError, match="to_nx"):
        cn.to_nx()


def test_directed_to_nx_roundtrip_from_nx():
    """to_nx() returns the original MultiDiGraph when built via from_nx()."""
    import networkx as nx

    G = nx.MultiDiGraph()
    G.graph["crs"] = CRS(32630)
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=20.0, y=0.0)
    G.add_edge("A", "B", key=0, geom=LineString([(0, 0), (20, 0)]))

    cn = CityNetwork.from_nx(G)
    exported = cn.to_nx()
    assert isinstance(exported, nx.MultiDiGraph)


def test_directed_save_load_roundtrip(tmp_path):
    """Directed flag and directions survive save/load."""
    gdf = _directed_streets_gdf()
    cn = CityNetwork.from_geopandas(gdf, directed=True)
    edge_count_before = cn.network_structure.edge_count

    cn.save(tmp_path / "directed_net")
    cn_loaded = CityNetwork.load(tmp_path / "directed_net")

    assert cn_loaded.is_directed is True
    assert cn_loaded.network_structure.edge_count == edge_count_before


def _dual_edge_imp_factors(cn: CityNetwork) -> list[float]:
    """Collect the imp_factor of every edge in a CityNetwork's underlying dual structure."""
    ns = cn.network_structure
    return [ns.get_edge_payload_py(s, e, idx).imp_factor for s, e, idx in ns.edge_references()]


def test_citynetwork_imp_factor_propagates_from_geopandas(tmp_path):
    """A primal `imp_factor` column on the input GeoDataFrame flows through to each dual edge
    as the length-weighted mean of the two adjacent primal segments' impedances, and survives
    a save/load round-trip.
    """
    # Two primal segments joined at (100, 0): lengths 100 and 200, impedances 2.0 and 4.0.
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (300, 0)]),
            ],
            "imp_factor": [2.0, 4.0],
        },
        crs="EPSG:32630",
    )
    cn = CityNetwork.from_geopandas(gdf)
    # both dual edges (one per direction in this undirected graph) should carry the
    # length-weighted mean of the two primal impedances.
    expected = (100.0 * 2.0 + 200.0 * 4.0) / (100.0 + 200.0)
    imps = _dual_edge_imp_factors(cn)
    assert imps and all(abs(imp - expected) < 1e-4 for imp in imps)

    # round-trip preserves the same dual impedances.
    cn.save(tmp_path / "imp_net")
    cn_loaded = CityNetwork.load(tmp_path / "imp_net")
    imps_loaded = _dual_edge_imp_factors(cn_loaded)
    assert sorted(imps) == pytest.approx(sorted(imps_loaded), rel=1e-4)

    # back-compat: omitting the column leaves every dual edge at the 1.0 default.
    gdf_default = gpd.GeoDataFrame({"geometry": gdf.geometry.tolist()}, crs="EPSG:32630")
    cn_default = CityNetwork.from_geopandas(gdf_default)
    assert all(abs(imp - 1.0) < 1e-6 for imp in _dual_edge_imp_factors(cn_default))


def test_citynetwork_imp_factor_propagates_from_nx():
    """A primal `imp_factor` edge attribute flows through `CityNetwork.from_nx` to the dual edges."""
    import networkx as nx_

    G = nx_.MultiGraph(crs="EPSG:32630")
    G.add_node("a", x=0.0, y=0.0)
    G.add_node("b", x=100.0, y=0.0)
    G.add_node("c", x=300.0, y=0.0)
    G.add_edge("a", "b", geom=LineString([(0, 0), (100, 0)]), imp_factor=2.0)
    G.add_edge("b", "c", geom=LineString([(100, 0), (300, 0)]), imp_factor=4.0)
    cn = CityNetwork.from_nx(G)
    expected = (100.0 * 2.0 + 200.0 * 4.0) / (100.0 + 200.0)
    imps = _dual_edge_imp_factors(cn)
    assert imps and all(abs(imp - expected) < 1e-4 for imp in imps)
