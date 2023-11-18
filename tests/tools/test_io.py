import networkx as nx
import numpy as np
import pytest
from pyproj import CRS, Transformer
from shapely import geometry, wkt

from cityseer import config
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io, mock

LNG = -0.1270
LAT = 51.5195
BUFFER = 200

BUFF_POLY_WGS = "POLYGON ((-0.1271130917990788 51.51770397813231, -0.1273947196842271 51.51771954212104, -0.1276725466454932 51.517752252774045, -0.1279438973528948 51.51780179511064, -0.1282061588163137 51.51786769206874, -0.1284568055399938 51.517949309096984, -0.1286934238356681 51.51804586026305, -0.128913735060938 51.51815641581937, -0.1291156175592228 51.518279911153485, -0.1292971270900654 51.51841515703722, -0.1294565155533125 51.518560851075584, -0.1295922478267924 51.51871559024599, -0.1297030165551059 51.518877884406606, -0.1297877547470147 51.51904617064404, -0.1298456460600064 51.51921882832255, -0.1298761326723199 51.51939419468955, -0.129878920666531 51.51957058088751, -0.1298539828721531 51.519746288217924, -0.1298015591397003 51.5199196245008, -0.1297221540428503 51.520088920372224, -0.1296165320304397 51.52025254536267, -0.1294857100746502 51.52040892360169, -0.1293309478856884 51.52055654899707, -0.1291537357868499 51.52069399974257, -0.1289557803665534 51.52081995201437, -0.1287389880453566 51.52093319272387, -0.1285054467159878 51.52103263120425, -0.1282574056332582 51.52111730971796, -0.1279972537478946 51.52118641268364, -0.1277274966926313 51.52123927453383, -0.1274507326428171 51.521275386127336, -0.1271696272843819 51.5212943996545, -0.1268868881303738 51.52129613198823, -0.1266052384339993 51.52128056644823, -0.1263273909498861 51.5212478529615, -0.1260560217967686 51.52119830661759, -0.1257937446735707 51.52113240463262, -0.1255430856778857 51.52105078175109, -0.1253064589695679 51.52095422413028, -0.1250861435142875 51.52084366176584, -0.1248842611312151 51.520720159531805, -0.1247027560563805 51.52058490692158, -0.1245433762186705 51.520439206588584, -0.1244076564088176 51.52028446179731, -0.1242969035032037 51.52012216290564, -0.1242121838850473 51.51995387300876, -0.1241543131830677 51.51978121288301, -0.1241238484269407 51.51960584537507, -0.1241210826942207 51.519429459386295, -0.1241460422998502 51.51925375360715, -0.1241984865550993 51.51908042015799, -0.1242779100976907 51.51891112829389, -0.1243835477698803 51.51874750833033, -0.1245143819976746 51.51859113594458, -0.1246691525992798 51.51844351700395, -0.1248463689281289 51.518306073066576, -0.1250443242331875 51.51818012769466, -0.1252611120983763 51.5180668937117, -0.1254946448022335 51.517967461525934, -0.1257426734213362 51.517882788633095, -0.126002809484 51.51781369039846, -0.1262725479655365 51.51776083220771, -0.1265492914043386 51.51772472306144, -0.1268303749065618 51.51770571067522, -0.1271130917990788 51.51770397813231))"

BUFF_POLY_UTM = "POLYGON ((699518.8923773962 5711511.242083298, 699517.9293227306 5711491.638655232, 699515.0494334769 5711472.224018895, 699510.2804445426 5711453.185147847, 699503.6682838985 5711434.705396825, 699495.2766302659 5711416.962735933, 699485.1862998568 5711400.128036694, 699473.4944680688 5711384.363426466, 699460.3137336335 5711369.820727061, 699445.771034229 5711356.639992625, 699430.0064240001 5711344.948160837, 699413.1717247615 5711334.8578304285, 699395.4290638693 5711326.466176796, 699376.9493128471 5711319.854016151, 699357.9104417994 5711315.085027217, 699338.4958054621 5711312.205137963, 699318.8923773962 5711311.242083298, 699299.2889493303 5711312.205137963, 699279.874312993 5711315.085027217, 699260.8354419454 5711319.854016151, 699242.3556909232 5711326.466176796, 699224.613030031 5711334.8578304285, 699207.7783307923 5711344.948160837, 699192.0137205635 5711356.639992625, 699177.471021159 5711369.820727061, 699164.2902867236 5711384.363426466, 699152.5984549357 5711400.128036694, 699142.5081245266 5711416.962735933, 699134.116470894 5711434.705396825, 699127.5043102498 5711453.185147847, 699122.7353213156 5711472.224018895, 699119.8554320618 5711491.638655232, 699118.8923773962 5711511.242083298, 699119.8554320618 5711530.845511364, 699122.7353213156 5711550.260147701, 699127.5043102498 5711569.299018749, 699134.116470894 5711587.778769771, 699142.5081245266 5711605.521430663, 699152.5984549357 5711622.356129902, 699164.2902867236 5711638.120740131, 699177.471021159 5711652.663439536, 699192.0137205635 5711665.844173971, 699207.7783307923 5711677.536005759, 699224.613030031 5711687.626336168, 699242.3556909232 5711696.0179898, 699260.8354419454 5711702.630150445, 699279.874312993 5711707.399139379, 699299.2889493303 5711710.279028633, 699318.8923773962 5711711.242083298, 699338.4958054621 5711710.279028633, 699357.9104417994 5711707.399139379, 699376.9493128471 5711702.630150445, 699395.4290638693 5711696.0179898, 699413.1717247615 5711687.626336168, 699430.0064240001 5711677.536005759, 699445.771034229 5711665.844173971, 699460.3137336335 5711652.663439536, 699473.4944680688 5711638.120740131, 699485.1862998568 5711622.356129902, 699495.2766302659 5711605.521430663, 699503.6682838985 5711587.778769771, 699510.2804445426 5711569.299018749, 699515.0494334769 5711550.260147701, 699517.9293227306 5711530.845511364, 699518.8923773962 5711511.242083298))"


# TODO: currently tested via test_nx_wgs_to_utm which calls nx_epsg_conversion internally
def nx_epsg_conversion():
    pass


def test_nx_wgs_to_utm():
    # check that node coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_converted = io.nx_wgs_to_utm(G_wgs)
    for n, d in G_utm.nodes(data=True):
        # rounding can be tricky
        assert np.allclose(d["x"], G_converted.nodes[n]["x"], atol=0.1, rtol=0)
        assert np.allclose(d["y"], G_converted.nodes[n]["y"], atol=0.1, rtol=0)

    # check that edge coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_utm = graphs.nx_simple_geoms(G_utm)

    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_wgs = graphs.nx_simple_geoms(G_wgs)

    G_converted = io.nx_wgs_to_utm(G_wgs)
    for s, e, k, d in G_utm.edges(data=True, keys=True):
        assert round(d["geom"].length, 1) == round(G_converted[s][e][k]["geom"].length, 1)

    # check that non-LineString geoms throw an error
    G_wgs = mock.mock_graph(wgs84_coords=True)
    for s, e, k in G_wgs.edges(keys=True):
        G_wgs[s][e][k]["geom"] = geometry.Point([G_wgs.nodes[s]["x"], G_wgs.nodes[s]["y"]])
    with pytest.raises(TypeError):
        io.nx_wgs_to_utm(G_wgs)

    # check that missing node keys throw an error
    for k in ["x", "y"]:
        G_wgs = mock.mock_graph(wgs84_coords=True)
        for n in G_wgs.nodes():
            # delete key from first node and break
            del G_wgs.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            io.nx_wgs_to_utm(G_wgs)

    # check that non WGS coordinates throw error
    G_utm = mock.mock_graph()
    with pytest.raises(ValueError):
        io.nx_wgs_to_utm(G_utm)

    # check that non-matching UTM zones are coerced to the same zone
    # this scenario spans two UTM zones
    G_wgs_b = nx.MultiGraph()
    nodes = [
        (1, {"x": -0.0005, "y": 51.572}),
        (2, {"x": -0.0005, "y": 51.571}),
        (3, {"x": 0.0005, "y": 51.570}),
        (4, {"x": -0.0005, "y": 51.569}),
        (5, {"x": -0.0015, "y": 51.570}),
    ]
    G_wgs_b.add_nodes_from(nodes)
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 2)]
    G_wgs_b.add_edges_from(edges)
    G_utm_30 = io.nx_wgs_to_utm(G_wgs_b)
    G_utm_30 = graphs.nx_simple_geoms(G_utm_30)

    # if not consistently coerced to UTM zone, the distances from 2-3 and 3-4 will be over 400km
    for s, e, d in G_utm_30.edges(data=True):
        assert d["geom"].length < 200

    # check that explicit zones are respectively coerced
    G_utm_31 = io.nx_wgs_to_utm(G_wgs_b, force_zone_number=31)
    G_utm_31 = graphs.nx_simple_geoms(G_utm_31)
    for n, d in G_utm_31.nodes(data=True):
        assert d["x"] != G_utm_30.nodes[n]["x"]

    # from cityseer.tools import plot
    # plot.plot_nx(G_wgs_b, labels=True, node_size=80)
    # plot.plot_nx(G_utm_b, labels=True, node_size=80)


def test_buffered_point_poly():
    """ """
    poly_wgs, poly_utm, utm_zone_number, utm_zone_letter = io.buffered_point_poly(LNG, LAT, BUFFER)
    test_wgs_poly = wkt.loads(BUFF_POLY_WGS)
    assert utm_zone_number == 30
    assert utm_zone_letter == "U"
    # check WGS
    assert np.allclose(
        poly_wgs.exterior.coords.xy[0], test_wgs_poly.exterior.coords.xy[0], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        poly_wgs.exterior.coords.xy[1], test_wgs_poly.exterior.coords.xy[1], atol=config.ATOL, rtol=config.RTOL
    )
    test_utm_poly = wkt.loads(BUFF_POLY_UTM)
    # check UTM
    assert np.allclose(
        poly_utm.exterior.coords.xy[0], test_utm_poly.exterior.coords.xy[0], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        poly_utm.exterior.coords.xy[1], test_utm_poly.exterior.coords.xy[1], atol=config.ATOL, rtol=config.RTOL
    )


# tested from next method
def test_fetch_osm_network():
    """ """
    pass


def test_osm_graph_from_poly():
    """ """
    # scaffold
    poly_wgs, poly_utm, utm_zone_number, _utm_zone_letter = io.buffered_point_poly(LNG, LAT, BUFFER)
    # check that default 4326 works - this will convert to UTM internally
    network_from_wgs = io.osm_graph_from_poly(poly_wgs, simplify=False)
    # visual check for debugging
    # plot.plot_nx(network_from_wgs)
    assert isinstance(network_from_wgs, nx.MultiGraph)
    assert len(network_from_wgs.nodes) > 0
    assert len(network_from_wgs.edges) > 0
    # check that from CRS conversions are working
    # 32630 corresponds to UTM 30N
    # this will first convert to WGS for OSM query - then back to UTM
    crs = CRS.from_dict({"proj": "utm", "zone": utm_zone_number, "north": True, "south": False})
    utm_epsg = crs.to_epsg()
    assert utm_epsg == 32630
    network_from_utm = io.osm_graph_from_poly(poly_utm, poly_epsg_code=utm_epsg, simplify=False)
    # visual check for debugging
    # plot.plot_nx(network_from_utm)
    assert isinstance(network_from_utm, nx.MultiGraph)
    assert len(network_from_utm.nodes) > 0
    assert len(network_from_utm.edges) > 0
    # check that networks match
    assert list(network_from_utm.nodes) == list(network_from_wgs.nodes)
    assert list(network_from_utm.edges) == list(network_from_wgs.edges)
    # check that to CRS conversions are working
    # this will convert out graph to BNG - EPSG 27700
    network_to_bng = io.osm_graph_from_poly(poly_wgs, to_epsg_code=27700, simplify=False)
    # networks should still match
    assert list(network_to_bng.nodes) == list(network_from_wgs.nodes)
    assert list(network_to_bng.edges) == list(network_from_wgs.edges)
    # but coordinates should be different
    transformer = Transformer.from_crs(utm_epsg, 27700, always_xy=True)
    for nd_idx, bng_nd_data in network_to_bng.nodes(data=True):
        utm_nd_data = network_from_utm.nodes[nd_idx]
        bng_easting, bng_northing = transformer.transform(utm_nd_data["x"], utm_nd_data["y"])
        assert np.isclose(bng_nd_data["x"], bng_easting)
        assert np.isclose(bng_nd_data["y"], bng_northing)
    # check that non-integer EPSG codes are caught
    with pytest.raises(TypeError):
        network_to_bng = io.osm_graph_from_poly(poly_wgs, to_epsg_code="27700")
    with pytest.raises(TypeError):
        network_to_bng = io.osm_graph_from_poly(poly_wgs, poly_epsg_code="27700")


def test_nx_from_osm():
    """Tested through usage - see demo notebooks"""
    pass


def test_nx_from_osm_nx():
    """Tested through usage - see demo notebooks"""
    pass


def test_nx_from_open_roads():
    """Tested through usage - see demo notebooks"""
    pass


def test_network_structure_from_nx(diamond_graph):
    # test maps vs. networkX
    G_test: nx.MultiGraph = diamond_graph.copy()
    # set some random 'live' statuses
    G_test.nodes["0"]["live"] = True
    G_test.nodes["1"]["live"] = True
    G_test.nodes["2"]["live"] = True
    G_test.nodes["3"]["live"] = False
    G_test_dual = graphs.nx_to_dual(G_test)
    # set some random 'live' statuses
    G_test_dual.nodes["0_1_k0"]["live"] = True
    G_test_dual.nodes["0_2_k0"]["live"] = True
    G_test_dual.nodes["1_2_k0"]["live"] = True
    G_test_dual.nodes["1_3_k0"]["live"] = True
    G_test_dual.nodes["2_3_k0"]["live"] = False
    for G, is_dual in zip((G_test, G_test_dual), (False, True)):
        # generate test maps
        nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, 3395)
        network_structure.validate()
        # debug plot
        # plot.plot_graphs(primal=G)
        # plot.plot_network_structure(nodes_gdf, node_data, edge_data)
        # check lengths
        assert len(nodes_gdf) == (network_structure.node_count()) == G.number_of_nodes()
        # edges = x2
        assert network_structure.edge_count == G.number_of_edges() * 2
        # check node maps (idx and label match in this case...)
        node_idxs = network_structure.node_indices()
        for node_idx in node_idxs:
            node_payload = network_structure.get_node_payload(node_idx)
            assert node_payload.coord.x - nodes_gdf.loc[node_payload.node_key].x < config.ATOL
            assert node_payload.coord.y - nodes_gdf.loc[node_payload.node_key].y < config.ATOL
            assert node_payload.live == nodes_gdf.loc[node_payload.node_key].live
        # check edge maps (idx and label match in this case...)
        for start_ns_node_idx, end_ns_node_idx, edge_idx in network_structure.edge_references():
            edge_payload = network_structure.get_edge_payload(start_ns_node_idx, end_ns_node_idx, edge_idx)
            start_nd_key = edge_payload.start_nd_key
            end_nd_key = edge_payload.end_nd_key
            edge_idx = edge_payload.edge_idx
            length = edge_payload.length
            angle_sum = edge_payload.angle_sum
            imp_factor = edge_payload.imp_factor
            in_bearing = edge_payload.in_bearing
            out_bearing = edge_payload.out_bearing
            # check against edges_gdf
            gdf_edge_key = f"{start_nd_key}-{end_nd_key}"
            assert edges_gdf.loc[gdf_edge_key, "start_ns_node_idx"] == start_ns_node_idx
            assert edges_gdf.loc[gdf_edge_key, "end_ns_node_idx"] == end_ns_node_idx
            assert edges_gdf.loc[gdf_edge_key, "edge_idx"] == edge_idx
            assert edges_gdf.loc[gdf_edge_key, "nx_start_node_key"] == start_nd_key
            assert edges_gdf.loc[gdf_edge_key, "nx_end_node_key"] == end_nd_key
            assert edges_gdf.loc[gdf_edge_key, "length"] - length < config.ATOL
            assert edges_gdf.loc[gdf_edge_key, "angle_sum"] - angle_sum < config.ATOL
            assert edges_gdf.loc[gdf_edge_key, "imp_factor"] - imp_factor < config.ATOL
            assert edges_gdf.loc[gdf_edge_key, "in_bearing"] - in_bearing < config.ATOL
            assert edges_gdf.loc[gdf_edge_key, "out_bearing"] - out_bearing < config.ATOL
            # manual checks
            if not is_dual:
                if (start_nd_key, end_nd_key) == ("0", "1"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            120.0,
                            120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0", "2"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            60.0,
                            60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1", "0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            -60.0,
                            -60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1", "2"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1", "3"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            60.0,
                            60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("2", "0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            -120.0,
                            -120.0,
                        ),
                    )
                elif (start_nd_key, end_nd_key) == ("2", "1"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            180.0,
                            180.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("2", "3"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            120.0,
                            120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("3", "1"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            -120.0,
                            -120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("3", "2"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            0.0,
                            1.0,
                            -60.0,
                            -60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                else:
                    raise KeyError("Unmatched edge.")
            else:
                if (start_nd_key, end_nd_key) == ("0_1_k0", "0_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            -60.0,
                            60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0_1_k0", "1_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            120.0,
                            0.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0_1_k0", "1_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            60.0,
                            1.0,
                            120.0,
                            60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0_2_k0", "0_1_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            -120.0,
                            120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0_2_k0", "1_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            60.0,
                            180.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("0_2_k0", "2_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            60.0,
                            1.0,
                            60.0,
                            120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_2_k0", "0_1_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            180.0,
                            -60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_2_k0", "0_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            0.0,
                            -120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_2_k0", "1_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            180.0,
                            60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_2_k0", "2_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            0.0,
                            120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_3_k0", "0_1_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            60.0,
                            1.0,
                            -120.0,
                            -60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_3_k0", "1_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            -120.0,
                            0.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("1_3_k0", "2_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            60.0,
                            -60.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("2_3_k0", "0_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            60.0,
                            1.0,
                            -60.0,
                            -120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("2_3_k0", "1_2_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            -60.0,
                            180.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                elif (start_nd_key, end_nd_key) == ("2_3_k0", "1_3_k0"):
                    assert np.allclose(
                        (length, angle_sum, imp_factor, in_bearing, out_bearing),
                        (
                            100.0,
                            120.0,
                            1.0,
                            120.0,
                            -120.0,
                        ),
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                else:
                    raise KeyError("Unmatched edge.")
    # check that non string indices throw an error
    G_test = diamond_graph.copy()
    G_test.add_node(0)
    G_test.add_node(1)
    G_test.add_edge(0, 1)
    with pytest.raises(TypeError):
        io.network_structure_from_nx(G_test, 3395)
    # check that missing geoms throw an error
    G_test = diamond_graph.copy()
    for start_nd, end_nd, edge_key in G_test.edges(keys=True):
        # delete key from first node and break
        del G_test[start_nd][end_nd][edge_key]["geom"]
        break
    with pytest.raises(KeyError):
        io.network_structure_from_nx(G_test, 3395)
    # check that non-LineString geoms throw an error
    G_test = diamond_graph.copy()
    for start_nd, end_nd, edge_key in G_test.edges(keys=True):
        G_test[start_nd][end_nd][edge_key]["geom"] = geometry.Point(
            [G_test.nodes[start_nd]["x"], G_test.nodes[start_nd]["y"]]
        )
    with pytest.raises(TypeError):
        io.network_structure_from_nx(G_test, 3395)
    # check that missing node keys throw an error
    G_test = diamond_graph.copy()
    for edge_key in ["x", "y"]:
        for nd_idx in G_test.nodes():
            # delete key from first node and break
            del G_test.nodes[nd_idx][edge_key]
            break
        with pytest.raises(KeyError):
            io.network_structure_from_nx(G_test, 3395)
    # check that invalid imp_factors are caught
    G_test = diamond_graph.copy()
    # corrupt imp_factor value and break
    for corrupt_val in [-1, -np.inf, np.nan]:
        for start_nd, end_nd, edge_key in G_test.edges(keys=True):
            G_test[start_nd][end_nd][edge_key]["imp_factor"] = corrupt_val
            break
        with pytest.raises(ValueError):
            io.network_structure_from_nx(G_test, 3395)


def test_nx_from_cityseer_geopandas(primal_graph):
    # also see test_networks.test_to_nx_multigraph for tests on implementation via Network layer
    # check round trip to and from graph maps results in same graph
    # explicitly set live params for equality checks
    # network_structure_from_networkX generates these implicitly if missing
    for node_key in primal_graph.nodes():
        primal_graph.nodes[node_key]["live"] = bool(np.random.randint(0, 2))
    # test directly from and to graph maps
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    G_round_trip = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
    assert list(G_round_trip.nodes) == list(primal_graph.nodes)
    assert list(G_round_trip.edges) == list(primal_graph.edges)
    # check with  missings weights and live columns
    for col in ["live", "weights"]:
        nodes_gdf_miss = nodes_gdf.copy()
        if col in nodes_gdf_miss.columns:
            nodes_gdf_miss.drop(columns=[col], inplace=True)
        G_round_trip_miss = io.nx_from_cityseer_geopandas(nodes_gdf_miss, edges_gdf)
        assert list(G_round_trip_miss.nodes) == list(primal_graph.nodes)
        assert list(G_round_trip_miss.edges) == list(primal_graph.edges)
        assert "live" in G_round_trip_miss.nodes["0"]
        assert "weight" in G_round_trip_miss.nodes["0"]
    # check with metrics
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    nodes_gdf = networks.node_centrality_shortest(
        network_structure=network_structure, nodes_gdf=nodes_gdf, compute_closeness=True, distances=[500, 1000]
    )
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, length=50)
    nodes_gdf, data_gdf = layers.compute_accessibilities(
        data_gdf,
        landuse_column_label="categorical_landuses",
        accessibility_keys=["a", "c"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=[500, 1000],
    )
    column_labels: list[str] = [
        "cc_metric_a_500_non_weighted",
        "cc_metric_a_1000_non_weighted",
        "cc_metric_a_500_weighted",
        "cc_metric_a_1000_weighted",
        "cc_metric_c_500_non_weighted",
        "cc_metric_c_1000_non_weighted",
        "cc_metric_c_500_weighted",
        "cc_metric_c_1000_weighted",
    ]
    # without backbone
    G_round_trip_nx = io.nx_from_cityseer_geopandas(
        nodes_gdf,
        edges_gdf,
    )
    for node_key, node_row in nodes_gdf.iterrows():  # type: ignore
        for col_label in column_labels:
            assert G_round_trip_nx.nodes[node_key][col_label] == node_row[col_label]
    # check that arbitrary geom column name doesn't raise
    edges_gdf = edges_gdf.rename_geometry("geommoeg")
    G_round_trip_nx = io.nx_from_cityseer_geopandas(
        nodes_gdf,
        edges_gdf,
    )
    # test with decomposed
    G_decomposed = graphs.nx_decompose(primal_graph, decompose_max=20)
    # set live explicitly
    for node_key in G_decomposed.nodes():
        G_decomposed.nodes[node_key]["live"] = bool(np.random.randint(0, 2))
    nodes_gdf_decomp, edges_gdf_decomp, network_structure_decomp = io.network_structure_from_nx(G_decomposed, 3395)
    G_round_trip_decomp = io.nx_from_cityseer_geopandas(nodes_gdf_decomp, edges_gdf_decomp)
    assert list(G_round_trip_decomp.nodes) == list(G_decomposed.nodes)
    for node_key, iter_node_data in G_round_trip_decomp.nodes(data=True):
        assert node_key in G_decomposed
        assert iter_node_data["live"] == G_decomposed.nodes[node_key]["live"]
        assert iter_node_data["x"] - G_decomposed.nodes[node_key]["x"] < config.ATOL
        assert iter_node_data["y"] - G_decomposed.nodes[node_key]["y"] < config.ATOL
    assert G_round_trip_decomp.edges == G_decomposed.edges


def test_generic_edges_geopandas_from_nx(primal_graph):
    """ """
    edges_gdf = io.generic_edges_geopandas_from_nx(primal_graph, 3395)
    assert len(edges_gdf) == len(primal_graph.edges)
    for _idx, row_data in edges_gdf.iterrows():
        assert (
            row_data["geom"]
            == primal_graph[row_data["start_nd_key"]][row_data["end_nd_key"]][row_data["edge_idx"]]["geom"]
        )


def test_nx_from_generic_geopandas(primal_graph):
    """ """
    # generate a GDF for testing with
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    generic_gdf = edges_gdf[["geom"]]
    # generic_gdf has directed edges but nx_from_generic_geopandas will deduplicate
    nx_from_generic = io.nx_from_generic_geopandas(generic_gdf)
    assert len(nx_from_generic.edges) == len(primal_graph.edges)
    # the single dangling node will be lost, so expect 56 instead of 57
    assert len(nx_from_generic) == 56
    total_lens_input = 0
    for s, e, d in primal_graph.edges(data=True):
        total_lens_input += d["geom"].length
    total_lens_generic = 0
    for s, e, d in nx_from_generic.edges(data=True):
        total_lens_generic += d["geom"].length
    assert total_lens_input - total_lens_generic < config.ATOL
