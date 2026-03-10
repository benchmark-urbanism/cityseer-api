from __future__ import annotations

import geopandas as gpd
import numpy as np
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
    streets_gdf.index = [
        f"{min(str(row.start_nd_key), str(row.end_nd_key))}"
        f"_{max(str(row.start_nd_key), str(row.end_nd_key))}"
        f"_k{int(row.edge_idx)}"
        for _, row in streets_gdf.iterrows()
    ]
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
