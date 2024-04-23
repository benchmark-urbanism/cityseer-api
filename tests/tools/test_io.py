import networkx as nx
import numpy as np
import pytest
from pyproj import CRS, Transformer
from shapely import geometry, wkt

from cityseer import config
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io, mock, util

LNG = -0.1270
LAT = 51.5195
BUFFER = 200

BUFF_POLY_WGS = "POLYGON ((-0.124121086898852 51.519429403644374, -0.1241460465042167 51.51925369786772, -0.1241984907589688 51.51908036442407, -0.1242779143007734 51.51891107256845, -0.1243835519719313 51.51874745261625, -0.1245143861984709 51.518591080244676, -0.1246691567986036 51.51844346132084, -0.1248463731257467 51.51830601740275, -0.12504432842893 51.518180072052424, -0.1252611162920526 51.51806683809309, -0.125494648993675 51.51796740593284, -0.1257426776104289 51.51788273306712, -0.1260028136706174 51.51781363486092, -0.1262725521495951 51.51776077669969, -0.1265492955857784 51.5177246675837, -0.1268303790853467 51.51770565522827, -0.1271130959752183 51.51770392271632, -0.1273947238577024 51.51771948673591, -0.1276725508163672 51.51775219741937, -0.1279439015212038 51.51780173978573, -0.1282061629821696 51.5178676367726, -0.1284568097035021 51.51794925382837, -0.1286934279969655 51.51804580502043, -0.1289137392201867 51.51815636060095, -0.1291156217165839 51.518279855957296, -0.1292971312457285 51.51841510186104, -0.1294565197075041 51.51856079591701, -0.1295922519797332 51.51871553510245, -0.1297030207070127 51.51887782927539, -0.1297877588981471 51.5190461155223, -0.1298456502106143 51.51921877320739, -0.1298761368226687 51.519394139577976, -0.1298789248168654 51.51957052577651, -0.129853987022744 51.51974623310446, -0.1298015632908124 51.51991956938189, -0.1297221581947249 51.52008886524491, -0.1296165361833233 51.52025249022409, -0.129485714228783 51.520408868449074, -0.1293309520412871 51.52055649382778, -0.1291537399441231 51.52069394455416, -0.1289557845257095 51.52081989680455, -0.1287389922065713 51.520933137490545, -0.1285054508793919 51.52103257594557, -0.1282574097990151 51.52111725443229, -0.1279972579161086 51.52118635736964, -0.1277275008633753 51.521239219190434, -0.1274507368161756 51.521275330753696, -0.127169631460404 51.52129434425011, -0.1268868923090621 51.521296076552886, -0.1266052426153342 51.521280510981995, -0.1263273951338489 51.521247797464746, -0.1260560259832953 51.521198251091, -0.1257937488625735 51.52113234907714, -0.1255430898692405 51.52105072616796, -0.1253064631631432 51.520954168521044, -0.1250861477099421 51.52084360613221, -0.1248842653287756 51.52072010387581, -0.1247027602556458 51.5205848512454, -0.1245433804194271 51.52043915089466, -0.1244076606108201 51.52028440608823, -0.1242969077062594 51.520122107184136, -0.1242121880888822 51.51995381727766, -0.1241543173874287 51.51978115714529, -0.1241238526315784 51.51960578963372, -0.124121086898852 51.519429403644374))"

BUFF_POLY_UTM = "POLYGON ((699518.8923665844 5711511.241661096, 699517.9293119188 5711491.63823303, 699515.049422665 5711472.223596693, 699510.2804337308 5711453.184725645, 699503.6682730867 5711434.704974623, 699495.276619454 5711416.962313731, 699485.1862890449 5711400.127614492, 699473.494457257 5711384.3630042635, 699460.3137228217 5711369.8203048585, 699445.7710234171 5711356.639570423, 699430.0064131883 5711344.947738635, 699413.1717139496 5711334.8574082265, 699395.4290530575 5711326.465754594, 699376.9493020353 5711319.853593949, 699357.9104309876 5711315.084605015, 699338.4957946503 5711312.204715761, 699318.8923665844 5711311.241661096, 699299.2889385185 5711312.204715761, 699279.8743021812 5711315.084605015, 699260.8354311335 5711319.853593949, 699242.3556801113 5711326.465754594, 699224.6130192191 5711334.8574082265, 699207.7783199805 5711344.947738635, 699192.0137097517 5711356.639570423, 699177.4710103471 5711369.8203048585, 699164.2902759118 5711384.3630042635, 699152.5984441239 5711400.127614492, 699142.5081137147 5711416.962313731, 699134.1164600821 5711434.704974623, 699127.504299438 5711453.184725645, 699122.7353105037 5711472.223596693, 699119.85542125 5711491.63823303, 699118.8923665844 5711511.241661096, 699119.85542125 5711530.845089162, 699122.7353105037 5711550.259725499, 699127.504299438 5711569.298596547, 699134.1164600821 5711587.778347569, 699142.5081137147 5711605.521008461, 699152.5984441239 5711622.3557077, 699164.2902759118 5711638.1203179285, 699177.4710103471 5711652.6630173335, 699192.0137097517 5711665.843751769, 699207.7783199805 5711677.535583557, 699224.6130192191 5711687.6259139655, 699242.3556801113 5711696.017567598, 699260.8354311335 5711702.629728243, 699279.8743021812 5711707.398717177, 699299.2889385185 5711710.278606431, 699318.8923665844 5711711.241661096, 699338.4957946503 5711710.278606431, 699357.9104309876 5711707.398717177, 699376.9493020353 5711702.629728243, 699395.4290530575 5711696.017567598, 699413.1717139496 5711687.6259139655, 699430.0064131883 5711677.535583557, 699445.7710234171 5711665.843751769, 699460.3137228217 5711652.6630173335, 699473.494457257 5711638.1203179285, 699485.1862890449 5711622.3557077, 699495.276619454 5711605.521008461, 699503.6682730867 5711587.778347569, 699510.2804337308 5711569.298596547, 699515.049422665 5711550.259725499, 699517.9293119188 5711530.845089162, 699518.8923665844 5711511.241661096))"


def test_nx_epsg_conversion():
    G_wgs = mock.mock_graph(wgs84_coords=True)
    nd_data = G_wgs.nodes["0"]
    utm_code = util.extract_utm_epsg_code(nd_data["x"], nd_data["y"])
    G_utm = mock.mock_graph()
    G_27700 = io.nx_epsg_conversion(G_utm, from_crs_code=utm_code, to_crs_code=27700)
    G_27700_2 = io.nx_epsg_conversion(G_utm, from_crs_code=utm_code, to_crs_code="27700")
    G_27700_3 = io.nx_epsg_conversion(G_utm, from_crs_code=utm_code, to_crs_code="EPSG:27700")
    for node_key in G_27700.nodes():
        nd_1 = G_27700.nodes[node_key]
        nd_2 = G_27700_2.nodes[node_key]
        nd_3 = G_27700_3.nodes[node_key]
        assert nd_1["x"] == nd_2["x"] == nd_3["x"]
        assert nd_1["y"] == nd_2["y"] == nd_3["y"]
    G_moll = io.nx_epsg_conversion(G_utm, from_crs_code=utm_code, to_crs_code="ESRI:54009")
    G_round = io.nx_epsg_conversion(G_moll, from_crs_code="ESRI:54009", to_crs_code=utm_code)
    for node_key in G_utm.nodes():
        nd_1 = G_utm.nodes[node_key]
        nd_2 = G_round.nodes[node_key]
        assert nd_1["x"] - nd_1["x"] <= config.ATOL
        assert nd_1["y"] - nd_1["y"] <= config.ATOL


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

    # from cityseer.tools import plot
    # plot.plot_nx(G_wgs_b, labels=True, node_size=80)
    # plot.plot_nx(G_utm_b, labels=True, node_size=80)


def test_buffered_point_poly():
    """ """
    wgs_poly, wgs_epsg_code = io.buffered_point_poly(LNG, LAT, BUFFER)
    test_wgs_poly = wkt.loads(BUFF_POLY_WGS)
    assert wgs_epsg_code == 4326
    # check WGS
    assert np.allclose(
        wgs_poly.exterior.coords.xy[0], test_wgs_poly.exterior.coords.xy[0], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        wgs_poly.exterior.coords.xy[1], test_wgs_poly.exterior.coords.xy[1], atol=config.ATOL, rtol=config.RTOL
    )
    # check UTM
    utm_poly, utm_epsg_code = io.buffered_point_poly(LNG, LAT, BUFFER, projected=True)
    assert utm_epsg_code == 32630
    test_utm_poly = wkt.loads(BUFF_POLY_UTM)
    assert np.allclose(
        utm_poly.exterior.coords.xy[0], test_utm_poly.exterior.coords.xy[0], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        utm_poly.exterior.coords.xy[1], test_utm_poly.exterior.coords.xy[1], atol=config.ATOL, rtol=config.RTOL
    )


# tested from next method
def test_fetch_osm_network():
    """ """
    pass


def test_osm_graph_from_poly():
    """ """
    # scaffold
    poly_wgs, _ = io.buffered_point_poly(LNG, LAT, BUFFER)
    # check that default 4326 works - this will convert to UTM internally
    network_from_wgs = io.osm_graph_from_poly(poly_wgs, simplify=False)
    # visual check for debugging
    # from cityseer.tools import plot
    # plot.plot_nx(network_from_wgs)
    assert isinstance(network_from_wgs, nx.MultiGraph)
    assert len(network_from_wgs.nodes) > 0
    assert len(network_from_wgs.edges) > 0
    # check that from CRS conversions are working
    # 32630 corresponds to UTM 30N
    poly_utm, utm_epsg = io.buffered_point_poly(LNG, LAT, BUFFER, projected=True)
    assert utm_epsg == 32630
    network_from_utm = io.osm_graph_from_poly(poly_utm, poly_crs_code=utm_epsg, simplify=False)
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
    network_to_bng = io.osm_graph_from_poly(poly_wgs, to_crs_code=27700, simplify=False)
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
        # CRS check
        assert nodes_gdf.crs.to_epsg() == 3395
        # dual specific checks
        if is_dual is True:
            # check that primal geom is copied across
            for _row_idx, row in nodes_gdf.iterrows():  # type: ignore
                assert (
                    row["primal_edge"]
                    == G_test[row["primal_edge_node_a"]][row["primal_edge_node_b"]][row["primal_edge_idx"]]["geom"]
                )
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


def test_network_structure_from_gpd(primal_graph):
    G: nx.MultiGraph = primal_graph.copy()
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, 3395)
    # round trip network structure
    network_structure_round = io.network_structure_from_gpd(nodes_gdf, edges_gdf)
    # node indices
    assert network_structure.node_indices() == network_structure_round.node_indices()
    # check node data consistency
    for start_nd_idx in network_structure.node_indices():
        node_data = network_structure.get_node_payload(start_nd_idx)
        node_data_round = network_structure_round.get_node_payload(start_nd_idx)
        assert node_data.coord.x == node_data_round.coord.x
        assert node_data.coord.y == node_data_round.coord.y
        assert node_data.node_key == node_data_round.node_key
        assert node_data.live == node_data_round.live
        assert node_data.weight == node_data_round.weight
    # edge indices
    assert network_structure.edge_references() == network_structure_round.edge_references()
    # check edge data consistency
    for start_nd_idx, end_nd_idx, edge_idx in network_structure.edge_references():
        edge_data = network_structure.get_edge_payload(start_nd_idx, end_nd_idx, edge_idx)
        edge_data_round = network_structure_round.get_edge_payload(start_nd_idx, end_nd_idx, edge_idx)
        assert edge_data.edge_idx == edge_data_round.edge_idx
        assert edge_data.start_nd_key == edge_data_round.start_nd_key
        assert edge_data.end_nd_key == edge_data_round.end_nd_key
        assert edge_data.length == edge_data_round.length
        assert edge_data.angle_sum == edge_data_round.angle_sum
        assert edge_data.imp_factor == edge_data_round.imp_factor
        assert edge_data.in_bearing == edge_data_round.in_bearing
        assert edge_data.out_bearing == edge_data_round.out_bearing


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
        "cc_a_500_nw",
        "cc_a_1000_nw",
        "cc_a_500_wt",
        "cc_a_1000_wt",
        "cc_c_500_nw",
        "cc_c_1000_nw",
        "cc_c_500_wt",
        "cc_c_1000_wt",
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


def test_geopandas_from_nx(primal_graph):
    """ """
    edges_gdf = io.geopandas_from_nx(primal_graph, 3395)
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
