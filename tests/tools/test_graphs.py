# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely import geometry, ops

from cityseer import config
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_nx_simple_geoms():
    # generate a mock graph
    g_raw = mock.mock_graph()
    g_copy = graphs.nx_simple_geoms(g_raw)
    # test that geoms have been inferred correctly
    for s, e, k in g_copy.edges(keys=True):
        line_geom = geometry.LineString(
            [
                [g_raw.nodes[s]["x"], g_raw.nodes[s]["y"]],
                [g_raw.nodes[e]["x"], g_raw.nodes[e]["y"]],
            ]
        )
        assert line_geom == g_copy[s][e][k]["geom"]
    # check that missing node keys throw an error
    g_copy = g_raw.copy()
    for k in ["x", "y"]:
        for n in g_copy.nodes():
            # delete key from first node and break
            del g_copy.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nx_simple_geoms(g_copy)
    # check that zero length self-loops are caught and removed
    g_copy = g_raw.copy()
    g_copy.add_edge(0, 0)  # simple geom from self edge = length of zero
    g_simple = graphs.nx_simple_geoms(g_copy)
    assert not g_simple.has_edge(0, 0)


def test_add_node(diamond_graph):
    new_name, is_dupe = graphs._add_node(diamond_graph, [0, 1], 50, 50)
    assert is_dupe is False
    assert new_name == "0±1"
    assert list(diamond_graph.nodes) == [0, 1, 2, 3, "0±1"]
    assert diamond_graph.nodes["0±1"] == {"x": 50, "y": 50}

    # same name and coordinates should return None
    response, is_dupe = graphs._add_node(diamond_graph, [0, 1], 50, 50)
    assert is_dupe is True

    # same name and different coordinates should return v2
    new_name, is_dupe = graphs._add_node(diamond_graph, [0, 1], 40, 50)
    assert is_dupe is False
    assert new_name == "0±1§v2"
    assert list(diamond_graph.nodes) == [0, 1, 2, 3, "0±1", "0±1§v2"]
    assert diamond_graph.nodes["0±1§v2"] == {"x": 40, "y": 50}

    # likewise v3
    new_name, is_dupe = graphs._add_node(diamond_graph, [0, 1], 30, 50)
    assert is_dupe is False
    assert new_name == "0±1§v3"
    assert list(diamond_graph.nodes) == [0, 1, 2, 3, "0±1", "0±1§v2", "0±1§v3"]
    assert diamond_graph.nodes["0±1§v3"] == {"x": 30, "y": 50}

    # and names should concatenate over old merges
    new_name, is_dupe = graphs._add_node(diamond_graph, [0, "0±1"], 60, 30)
    assert is_dupe is False
    assert new_name == "0±0±1"
    assert list(diamond_graph.nodes) == [0, 1, 2, 3, "0±1", "0±1§v2", "0±1§v3", "0±0±1"]
    assert diamond_graph.nodes["0±0±1"] == {"x": 60, "y": 30}


# TODO:
def test_nx_from_osm():
    pass


# TODO: currently tested via test_nx_wgs_to_utm which calls nx_epsg_conversion internally
def nx_epsg_conversion():
    pass


def test_nx_wgs_to_utm():
    # check that node coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_converted = graphs.nx_wgs_to_utm(G_wgs)
    for n, d in G_utm.nodes(data=True):
        # rounding can be tricky
        assert np.allclose(d["x"], G_converted.nodes[n]["x"], atol=0.1, rtol=0)
        assert np.allclose(d["y"], G_converted.nodes[n]["y"], atol=0.1, rtol=0)

    # check that edge coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_utm = graphs.nx_simple_geoms(G_utm)

    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_wgs = graphs.nx_simple_geoms(G_wgs)

    G_converted = graphs.nx_wgs_to_utm(G_wgs)
    for s, e, k, d in G_utm.edges(data=True, keys=True):
        assert round(d["geom"].length, 1) == round(G_converted[s][e][k]["geom"].length, 1)

    # check that non-LineString geoms throw an error
    G_wgs = mock.mock_graph(wgs84_coords=True)
    for s, e, k in G_wgs.edges(keys=True):
        G_wgs[s][e][k]["geom"] = geometry.Point([G_wgs.nodes[s]["x"], G_wgs.nodes[s]["y"]])
    with pytest.raises(TypeError):
        graphs.nx_wgs_to_utm(G_wgs)

    # check that missing node keys throw an error
    for k in ["x", "y"]:
        G_wgs = mock.mock_graph(wgs84_coords=True)
        for n in G_wgs.nodes():
            # delete key from first node and break
            del G_wgs.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nx_wgs_to_utm(G_wgs)

    # check that non WGS coordinates throw error
    G_utm = mock.mock_graph()
    with pytest.raises(ValueError):
        graphs.nx_wgs_to_utm(G_utm)

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
    G_utm_30 = graphs.nx_wgs_to_utm(G_wgs_b)
    G_utm_30 = graphs.nx_simple_geoms(G_utm_30)

    # if not consistently coerced to UTM zone, the distances from 2-3 and 3-4 will be over 400km
    for s, e, d in G_utm_30.edges(data=True):
        assert d["geom"].length < 200

    # check that explicit zones are respectively coerced
    G_utm_31 = graphs.nx_wgs_to_utm(G_wgs_b, force_zone_number=31)
    G_utm_31 = graphs.nx_simple_geoms(G_utm_31)
    for n, d in G_utm_31.nodes(data=True):
        assert d["x"] != G_utm_30.nodes[n]["x"]

    # from cityseer.tools import plot
    # plot.plot_nx(G_wgs_b, labels=True, node_size=80)
    # plot.plot_nx(G_utm_b, labels=True, node_size=80)


def make_messy_graph(G):
    # test that redundant (sraight) intersections are removed
    G_messy = G.copy(G)

    # complexify the graph - write changes to new graph to avoid in-place iteration errors
    for i, (s, e, k, d) in enumerate(G.edges(data=True, keys=True)):
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d["geom"].coords.xy)
            G_messy[s][e][k]["geom"] = geometry.LineString(
                [[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])]
            )
        # split each second geom
        if i % 2 == 0:
            line_geom = G[s][e][k]["geom"]
            # check geom coordinates directionality - flip if facing backwards direction
            if not (G.nodes[s]["x"], G.nodes[s]["y"]) == line_geom.coords[0][:2]:
                flipped_coords = np.fliplr(line_geom.coords.xy)
                line_geom = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
            # remove old edge
            G_messy.remove_edge(s, e)
            # new midpoint 'x' and 'y' coordinates
            s_geom = ops.substring(line_geom, 0, 0.5, normalized=True)
            e_geom = ops.substring(line_geom, 0.5, 1, normalized=True)
            # looking for the non-matching coordinates
            mid_x, mid_y = s_geom.coords[-1][:2]
            # add new edges
            G_messy.add_edge(s, f"{s}-{e}", geom=s_geom)
            G_messy.add_edge(e, f"{s}-{e}", geom=e_geom)
            G_messy.nodes[f"{s}-{e}"]["x"] = mid_x
            G_messy.nodes[f"{s}-{e}"]["y"] = mid_y

    # test recursive weld by manually adding a chained series of orphan nodes
    geom = G[10][43][0]["geom"]
    geom_a = ops.substring(geom, 0, 0.25, normalized=True)
    G_messy.add_edge(10, "t_1", geom=geom_a)
    a_x, a_y = geom_a.coords[-1][:2]
    G_messy.nodes["t_1"]["x"] = a_x
    G_messy.nodes["t_1"]["y"] = a_y
    geom_b = ops.substring(geom, 0.25, 0.5, normalized=True)
    G_messy.add_edge("t_1", "t_2", geom=geom_b)
    b_x, b_y = geom_b.coords[-1][:2]
    G_messy.nodes["t_2"]["x"] = b_x
    G_messy.nodes["t_2"]["y"] = b_y
    geom_c = ops.substring(geom, 0.5, 0.75, normalized=True)
    G_messy.add_edge("t_2", "t_3", geom=geom_c)
    c_x, c_y = geom_c.coords[-1][:2]
    G_messy.nodes["t_3"]["x"] = c_x
    G_messy.nodes["t_3"]["y"] = c_y
    geom_d = ops.substring(geom, 0.75, 1.0, normalized=True)
    G_messy.add_edge("t_3", 43, geom=geom_d)
    # remove original geom
    G_messy.remove_edge(10, 43)

    return G_messy


def test_nx_remove_dangling_nodes(primal_graph):
    G_messy = make_messy_graph(primal_graph)
    # no despining or disconnected components removal
    G_post = graphs.nx_remove_dangling_nodes(G_messy, despine=0, remove_disconnected=False, cleanup_filler_nodes=False)
    assert G_post.nodes == G_messy.nodes
    assert G_post.edges == G_messy.edges
    # check that all single neighbour nodes have been removed if geom less than despine distance
    G_post = graphs.nx_remove_dangling_nodes(G_messy, despine=100, remove_disconnected=False)
    for n in G_messy.nodes():
        if nx.degree(G_messy, n) == 1:
            nb = list(nx.neighbors(G_messy, n))[0]
            if G_messy[n][nb][0]["geom"].length <= 100:
                assert (n, nb) not in G_post.edges
            else:
                assert (n, nb) in G_post.edges
    # check that disconnected components are removed
    # this behaviour changed in networkx 2.4
    G_post = graphs.nx_remove_dangling_nodes(G_messy, despine=0, remove_disconnected=True, cleanup_filler_nodes=False)
    pre_components = list(nx.algorithms.components.connected_components(G_messy))
    post_components = list(nx.algorithms.components.connected_components(G_post))
    assert len(pre_components) != 1
    assert len(post_components) == 1
    # check that components match
    biggest_component = sorted(pre_components, key=len, reverse=True)[0]
    # index to 0 because post_components is still in list form
    assert biggest_component == post_components[0]
    # check that actual graphs are equivalent
    G_biggest_component = nx.MultiGraph(G_messy.subgraph(biggest_component))
    assert G_biggest_component.nodes == G_post.nodes
    assert G_biggest_component.edges == G_post.edges


def test_nx_remove_filler_nodes(primal_graph):
    # test that redundant intersections are removed, i.e. where degree == 2
    G_messy = make_messy_graph(primal_graph)
    # from cityseer.tools import plot
    # plot.plot_nx(G_messy, labels=True, plot_geoms=True, node_size=80)
    # simplify and test
    G_simplified = graphs.nx_remove_filler_nodes(G_messy)
    # plot.plot_nx(G_simplified, labels=True, plot_geoms=True, node_size=80)
    # check that the simplified version matches the original un-messified version
    # but note the simplified version will have the disconnected loop of 52-53-54-55 now condensed to only #52
    g_nodes = set(primal_graph.nodes)
    g_nodes = g_nodes.difference([53, 54, 55])
    assert list(g_nodes).sort() == list(G_simplified.nodes).sort()
    g_edges = set(primal_graph.edges)
    g_edges = g_edges.difference([(52, 53), (53, 54), (54, 55), (52, 55)])  # condensed edges
    g_edges = g_edges.union([(52, 52)])  # the new self loop
    assert list(g_edges).sort() == list(G_simplified.edges).sort()
    # plot.plot_nx(G_simplified, labels=True, node_size=80, plot_geoms=True)
    # check the integrity of the edges
    for s, e, k, d in G_simplified.edges(data=True, keys=True):
        # ignore the new self-looping disconnected edge
        if s == 52 and e == 52:
            continue
        # and the parallel edge
        if s in [45, 30] and e in [45, 30]:
            continue
        assert G_simplified[s][e][k]["geom"].length == primal_graph[s][e][k]["geom"].length
    # manually check that the new self-looping edge is equal in length to its original segments
    l = 0
    for s, e in [(52, 53), (53, 54), (54, 55), (52, 55)]:
        l += primal_graph[s][e][0]["geom"].length
    assert l == G_simplified[52][52][0]["geom"].length
    # and that the new parallel edge is correct
    l = 0
    for s, e in [(45, 56), (56, 30)]:
        l += primal_graph[s][e][0]["geom"].length
    assert l == G_simplified[45][30][0]["geom"].length
    # check that all nodes still have 'x' and 'y' keys
    for n, d in G_simplified.nodes(data=True):
        assert "x" in d
        assert "y" in d
    # lollipop test - where a looping component (all nodes == degree 2) suspends off a node with degree > 2
    G_lollipop = nx.MultiGraph()
    nodes = [
        (1, {"x": 400, "y": 750}),
        (2, {"x": 400, "y": 650}),
        (3, {"x": 500, "y": 550}),
        (4, {"x": 400, "y": 450}),
        (5, {"x": 300, "y": 550}),
    ]
    G_lollipop.add_nodes_from(nodes)
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 2)]
    G_lollipop.add_edges_from(edges)
    # add edge geoms
    G_lollipop = graphs.nx_simple_geoms(G_lollipop)
    # flip some geometry
    G_lollipop[2][5][0]["geom"] = geometry.LineString(G_lollipop[2][5][0]["geom"].coords[::-1])
    # plot.plot_nx(G_lollipop, labels=True, plot_geoms=True, node_size=80)
    # simplify
    G_lollipop_simpl = graphs.nx_remove_filler_nodes(G_lollipop)
    # plot.plot_nx(G_lollipop_simpl, labels=True, plot_geoms=True, node_size=80)
    # check integrity of graph
    assert nx.number_of_nodes(G_lollipop_simpl) == 2
    assert nx.number_of_edges(G_lollipop_simpl) == 2
    # geoms should still be same cumulative length
    before_len = 0
    for s, e, d in G_lollipop.edges(data=True):
        before_len += d["geom"].length
    after_len = 0
    for s, e, d in G_lollipop_simpl.edges(data=True):
        after_len += d["geom"].length
    assert before_len == after_len
    # end point of stick should match start / end point of lollipop
    assert G_lollipop_simpl[1][2][0]["geom"].coords[-1] == G_lollipop_simpl[2][2][0]["geom"].coords[0]
    # start and end point of lollipop should match
    assert G_lollipop_simpl[2][2][0]["geom"].coords[0] == G_lollipop_simpl[2][2][0]["geom"].coords[-1]
    # manually check welded geom - either direction of winding is OK
    assert G_lollipop_simpl[2][2][0]["geom"].wkt in [
        "LINESTRING (400 650, 500 550, 400 450, 300 550, 400 650)",
        "LINESTRING (400 650, 300 550, 400 450, 500 550, 400 650)",
    ]
    # stairway test - where overlapping edges (all nodes == degree 2) have overlapping coordinates in 2D space
    G_stairway = nx.MultiGraph()
    nodes = [
        ("1-down", {"x": 400, "y": 750}),
        ("2-down", {"x": 400, "y": 650}),
        ("3-down", {"x": 500, "y": 550}),
        ("4-down", {"x": 400, "y": 450}),
        ("5-down", {"x": 300, "y": 550}),
        ("2-mid", {"x": 400, "y": 650}),
        ("3-mid", {"x": 500, "y": 550}),
        ("4-mid", {"x": 400, "y": 450}),
        ("5-mid", {"x": 300, "y": 550}),
        ("2-up", {"x": 400, "y": 650}),
        ("1-up", {"x": 400, "y": 750}),
    ]
    G_stairway.add_nodes_from(nodes)
    G_stairway.add_nodes_from(nodes)
    edges = [
        ("1-down", "2-down"),
        ("2-down", "3-down"),
        ("3-down", "4-down"),
        ("4-down", "5-down"),
        ("5-down", "2-mid"),
        ("2-mid", "3-mid"),
        ("3-mid", "4-mid"),
        ("4-mid", "5-mid"),
        ("5-mid", "2-up"),
        ("2-up", "1-up"),
    ]
    G_stairway.add_edges_from(edges)
    # add edge geoms
    G_stairway = graphs.nx_simple_geoms(G_stairway)
    # flip some geometry
    G_stairway["5-down"]["2-mid"][0]["geom"] = geometry.LineString(
        G_stairway["5-down"]["2-mid"][0]["geom"].coords[::-1]
    )
    # plot.plot_nx(G_stairway, labels=True, plot_geoms=True, node_size=80)
    # simplify
    G_stairway_simpl = graphs.nx_remove_filler_nodes(G_stairway)
    # plot.plot_nx(G_stairway_simpl, labels=True, plot_geoms=True, node_size=80)
    # check integrity of graph
    assert nx.number_of_nodes(G_stairway_simpl) == 2
    assert nx.number_of_edges(G_stairway_simpl) == 1
    # geoms should still be same cumulative length
    before_len = 0
    for s, e, d in G_stairway.edges(data=True):
        before_len += d["geom"].length
    after_len = 0
    for s, e, d in G_stairway_simpl.edges(data=True):
        after_len += d["geom"].length
    assert before_len == after_len
    # either direction is OK
    assert G_stairway_simpl["1-down"]["1-up"][0]["geom"].wkt in [
        "LINESTRING (400 750, 400 650, 500 550, 400 450, 300 550, 400 650, 500 550, 400 450, 300 550, 400 650, 400 750)",
        "LINESTRING (400 750, 400 650, 300 550, 400 450, 500 550, 400 650, 300 550, 400 450, 500 550, 400 650, 400 750)",
    ]
    # check that missing geoms throw an error
    G_k = G_messy.copy()
    for i, (s, e, k) in enumerate(G_k.edges(keys=True)):
        if i % 2 == 0:
            del G_k[s][e][k]["geom"]
    with pytest.raises(KeyError):
        graphs.nx_remove_filler_nodes(G_k)
    # check that non-LineString geoms throw an error
    G_k = G_messy.copy()
    for s, e, k in G_k.edges(keys=True):
        G_k[s][e][k]["geom"] = geometry.Point([G_k.nodes[s]["x"], G_k.nodes[s]["y"]])
    with pytest.raises(TypeError):
        graphs.nx_remove_filler_nodes(G_k)
    # catch non-touching LineStrings
    G_corr = G_messy.copy()
    for s, e, k in G_corr.edges(keys=True):
        geom = G_corr[s][e][k]["geom"]
        start = list(geom.coords[0])
        end = list(geom.coords[1])
        # corrupt a point
        start[0] = start[0] - 1
        G_corr[s][e][k]["geom"] = geometry.LineString([start, end])
    with pytest.raises(ValueError):
        graphs.nx_remove_filler_nodes(G_corr)


# this method tests both nx_consolidate_spatial and nx_consolidate_parallel
def test_nx_consolidate():
    # create a test graph
    G = nx.MultiGraph()
    nodes = [
        (0, {"x": 620, "y": 720}),
        (1, {"x": 620, "y": 700}),
        (2, {"x": 660, "y": 700}),
        (3, {"x": 660, "y": 660}),
        (4, {"x": 700, "y": 800}),
        (5, {"x": 720, "y": 800}),
        (6, {"x": 700, "y": 720}),
        (7, {"x": 720, "y": 720}),
        (8, {"x": 700, "y": 700}),
        (9, {"x": 700, "y": 620}),
        (10, {"x": 720, "y": 620}),
        (11, {"x": 760, "y": 760}),
        (12, {"x": 800, "y": 760}),
        (13, {"x": 780, "y": 720}),
        (14, {"x": 840, "y": 720}),
        (15, {"x": 840, "y": 700}),
    ]
    edges = [
        (0, 6),
        (1, 2),
        (2, 3),
        (2, 8),
        (4, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (7, 10),
        (7, 13),
        (8, 9),
        (8, 15),
        (11, 12),
        (11, 13),
        (12, 13),
        (13, 14),
    ]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G = graphs.nx_simple_geoms(G)
    # behaviour confirmed visually
    # from cityseer.tools import plot
    # plot.plot_nx(G, labels=True, node_size=80, plot_geoms=True)
    G_merged_spatial = graphs.nx_consolidate_nodes(G, buffer_dist=25, crawl=True, merge_edges_by_midline=True)
    # plot.plot_nx(G_merged_spatial, labels=True, node_size=80, plot_geoms=True)
    # this time, start with same origin graph but split opposing geoms first
    G_split_opps = graphs.nx_split_opposing_geoms(G, buffer_dist=25, merge_edges_by_midline=True)
    # plot.plot_nx(G_split_opps, labels=True, node_size=80, plot_geoms=True)
    G_merged_spatial = graphs.nx_consolidate_nodes(
        G_split_opps, buffer_dist=25, merge_edges_by_midline=True, cent_min_degree=2
    )
    # plot.plot_nx(G_merged_spatial, labels=True, node_size=80, plot_geoms=True)

    assert G_merged_spatial.number_of_nodes() == 8
    assert G_merged_spatial.number_of_edges() == 8

    node_coords = []
    for n, d in G_merged_spatial.nodes(data=True):
        node_coords.append((d["x"], d["y"]))
    assert node_coords == [
        (660, 660),
        (620.0, 710.0),
        (660.0, 710.0),
        (710.0, 800.0),
        (710.0, 710.0),
        (710.0, 620.0),
        (780.0, 710.0),
        (840.0, 710.0),
    ]

    edge_lens = []
    for s, e, d in G_merged_spatial.edges(data=True):
        edge_lens.append(d["geom"].length)
    assert edge_lens == [50.0, 40.0, 50.0, 90.0, 90.0, 70.0, 60.0, 147.70329614269008]


def test_nx_decompose(primal_graph):
    # check that missing geoms throw an error
    G = primal_graph.copy()
    del G[0][1][0]["geom"]
    with pytest.raises(KeyError):
        graphs.nx_decompose(G, 20)

    # check that non-LineString geoms throw an error
    G = primal_graph.copy()
    for s, e, k in G.edges(keys=True):
        G[s][e][k]["geom"] = geometry.Point([G.nodes[s]["x"], G.nodes[s]["y"]])
        break
    with pytest.raises(TypeError):
        graphs.nx_decompose(G, 20)

    # test decomposition
    G = primal_graph.copy()
    # first clean the graph to strip disconnected looping component
    # this gives a start == end node situation for testing
    G_simple = graphs.nx_remove_filler_nodes(G)
    G_decompose = graphs.nx_decompose(G_simple, 50)

    # from cityseer.tools import plot
    # plot.plot_nx(G_simple, labels=True, node_size=80, plot_geoms=True)
    # plot.plot_nx(G_decompose, plot_geoms=True)
    assert nx.number_of_nodes(G_decompose) == 292
    assert nx.number_of_edges(G_decompose) == 314
    for s, e in G_decompose.edges():
        assert G_decompose.number_of_edges(s, e) == 1

    # check that total lengths are the same
    G_lens = 0
    for s, e, e_data in G_simple.edges(data=True):
        G_lens += e_data["geom"].length
    G_d_lens = 0
    for s, e, e_data in G_decompose.edges(data=True):
        G_d_lens += e_data["geom"].length
    assert np.allclose(G_lens, G_d_lens, atol=config.ATOL, rtol=config.RTOL)

    # check that geoms are correctly flipped
    G_forward = primal_graph.copy()
    G_forward_decompose = graphs.nx_decompose(G_forward, 20)

    G_backward = primal_graph.copy()
    for i, (s, e, k, d) in enumerate(G_backward.edges(data=True, keys=True)):
        # flip each third geom
        if i % 3 == 0:
            G[s][e][k]["geom"] = geometry.LineString(d["geom"].coords[::-1])
    G_backward_decompose = graphs.nx_decompose(G_backward, 20)

    for n, d in G_forward_decompose.nodes(data=True):
        assert d["x"] == G_backward_decompose.nodes[n]["x"]
        assert d["y"] == G_backward_decompose.nodes[n]["y"]

    # test that geom coordinate mismatch throws an error
    G = primal_graph.copy()
    for k in ["x", "y"]:
        for n in G.nodes():
            G.nodes[n][k] = G.nodes[n][k] + 1
            break
        with pytest.raises(ValueError):
            graphs.nx_decompose(G, 20)


def test_nx_to_dual(primal_graph, diamond_graph):
    # check that missing geoms throw an error
    G = diamond_graph.copy()
    del G[0][1][0]["geom"]
    with pytest.raises(KeyError):
        graphs.nx_to_dual(G)

    # check that non-LineString geoms throw an error
    G = diamond_graph.copy()
    for s, e, k in G.edges(keys=True):
        G[s][e][k]["geom"] = geometry.Point([G.nodes[s]["x"], G.nodes[s]["y"]])
    with pytest.raises(TypeError):
        graphs.nx_to_dual(G)

    # check that missing node keys throw an error
    for k in ["x", "y"]:
        G = diamond_graph.copy()
        for n in G.nodes():
            # delete key from first node and break
            del G.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nx_to_dual(G)

    # test dual
    G = diamond_graph.copy()
    G_dual = graphs.nx_to_dual(G)
    # from cityseer.tools import plot
    # plot.plot_nx_primal_or_dual(primal_graph=G, dual_graph=G_dual, plot_geoms=True, labels=True, node_size=80)

    assert G_dual.number_of_nodes() == 5
    assert G_dual.number_of_edges() == 8
    # the new dual nodes have three edges each, except for the midspan which now has four redges
    for n in G_dual.nodes():
        if n == "1_2":
            assert nx.degree(G_dual, n) == 4
        else:
            assert nx.degree(G_dual, n) == 3
    for start, end, d in G_dual.edges(data=True):
        # the new geoms should also be 100m length (split 50m x 2)
        assert round(d["geom"].length) == 100
        # check the starting and ending bearings per diamond graph
        if (G_dual.nodes[start]["x"], G_dual.nodes[start]["y"]) == d["geom"].coords[0]:
            s_x, s_y = d["geom"].coords[0]
            m_x, m_y = d["geom"].coords[1]
            e_x, e_y = d["geom"].coords[-1]
        else:
            s_x, s_y = d["geom"].coords[-1]
            m_x, m_y = d["geom"].coords[1]
            e_x, e_y = d["geom"].coords[0]
        in_bearing = np.rad2deg(np.arctan2(m_y - s_y, m_x - s_x)).round()
        out_bearing = np.rad2deg(np.arctan2(e_y - m_y, e_x - m_x)).round()
        if (start, end) == ("0_1", "0_2"):
            assert (in_bearing, out_bearing) == (-60, 60)
        elif (start, end) == ("0_1", "1_2"):
            assert (in_bearing, out_bearing) == (120, 0)
        elif (start, end) == ("0_1", "1_3"):
            assert (in_bearing, out_bearing) == (120, 60)
        elif (start, end) == ("0_2", "1_2"):
            assert (in_bearing, out_bearing) == (60, 180)
        elif (start, end) == ("0_2", "2_3"):
            assert (in_bearing, out_bearing) == (60, 120)
        elif (start, end) == ("1_2", "1_3"):
            assert (in_bearing, out_bearing) == (180, 60)
        elif (start, end) == ("1_2", "2_3"):
            assert (in_bearing, out_bearing) == (0, 120)
        elif (start, end) == ("1_3", "2_3"):
            assert (in_bearing, out_bearing) == (60, -60)

    # complexify the geoms to check with and without kinks, and in mixed forward and reverse directions
    # see if any issues arise
    G = primal_graph.copy()
    for i, (s, e, k, d) in enumerate(G.edges(data=True, keys=True)):
        # add a kink to each second geom
        if i % 2 == 0:
            geom = d["geom"]
            start = geom.coords[0]
            end = geom.coords[-1]
            # bump the new midpoint coordinates
            mid = list(geom.centroid.coords[0])
            mid[0] += 10
            mid[1] -= 10
            # append 3d coord to check behaviour on 3d data
            kinked_3d_geom = []
            for n in [start, mid, end]:
                n = list(n)
                n.append(10)
                kinked_3d_geom.append(n)
            G[s][e][k]["geom"] = geometry.LineString(kinked_3d_geom)
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d["geom"].coords.xy)
            G[s][e][k]["geom"] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
    G_dual = graphs.nx_to_dual(G)
    # from cityseer.tools import plot
    # plot.plot_nx_primal_or_dual(primal_graph=G, dual_graph=G_dual, plot_geoms=True, labels=True, node_size=80)
    # 3 + 4 + 1 + 3 + 3 + (9 + 12) + (9 + 12) + (9 + 12) = 77
    assert G_dual.number_of_nodes() == 79
    assert G_dual.number_of_edges() == 155
    for s, e in G_dual.edges():
        assert G_dual.number_of_edges(s, e) == 1


def test_network_structure_from_nx(diamond_graph):
    # test maps vs. networkX
    G_test: nx.MultiGraph = diamond_graph.copy()
    # set some random 'live' statuses
    G_test.nodes[0]["live"] = True
    G_test.nodes[1]["live"] = True
    G_test.nodes[2]["live"] = True
    G_test.nodes[3]["live"] = False
    G_test_dual = graphs.nx_to_dual(G_test)
    # set some random 'live' statuses
    G_test_dual.nodes["0_1"]["live"] = True
    G_test_dual.nodes["0_2"]["live"] = True
    G_test_dual.nodes["1_2"]["live"] = True
    G_test_dual.nodes["1_3"]["live"] = True
    G_test_dual.nodes["2_3"]["live"] = False
    for G, is_dual in zip((G_test, G_test_dual), (False, True)):
        # generate test maps
        nodes_gdf, network_structure = graphs.network_structure_from_nx(G, 3395)
        network_structure.validate()
        # debug plot
        # plot.plot_graphs(primal=G)
        # plot.plot_network_structure(nodes_gdf, node_data, edge_data)

        # check lengths
        assert len(nodes_gdf) == network_structure.nodes.count == G.number_of_nodes()
        # edges = x2
        assert network_structure.edges.count == G.number_of_edges() * 2

        # check node maps (idx and label match in this case...)
        np.allclose(network_structure.nodes.xs, nodes_gdf.geometry.x, atol=config.ATOL, rtol=config.RTOL)
        np.allclose(network_structure.nodes.ys, nodes_gdf.geometry.y, atol=config.ATOL, rtol=config.RTOL)
        np.allclose(network_structure.nodes.live, nodes_gdf.live, atol=config.ATOL, rtol=config.RTOL)

        # check edge maps (idx and label match in this case...)
        for edge_idx in range(network_structure.edges.count):
            start_nd = network_structure.edges.start[edge_idx]
            end_nd = network_structure.edges.end[edge_idx]
            length = network_structure.edges.length[edge_idx]
            angle_sum = network_structure.edges.angle_sum[edge_idx]
            imp_factor = network_structure.edges.imp_factor[edge_idx]
            in_bearing = network_structure.edges.in_bearing[edge_idx]
            out_bearing = network_structure.edges.out_bearing[edge_idx]
            # checks
            if not is_dual:
                if (start_nd, end_nd) == (0.0, 1.0):
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
                elif (start_nd, end_nd) == (0.0, 2.0):
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
                elif (start_nd, end_nd) == (1.0, 0.0):
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
                elif (start_nd, end_nd) == (1.0, 2.0):
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
                elif (start_nd, end_nd) == (1.0, 3.0):
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
                elif (start_nd, end_nd) == (2.0, 0.0):
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
                elif (start_nd, end_nd) == (2.0, 1.0):
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
                elif (start_nd, end_nd) == (2.0, 3.0):
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
                elif (start_nd, end_nd) == (3.0, 1.0):
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
                elif (start_nd, end_nd) == (3.0, 2.0):
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
                if (start_nd, end_nd) == (0.0, 1.0):  # 0_1 0_2
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
                elif (start_nd, end_nd) == (0.0, 2.0):  # 0_1 1_2
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
                elif (start_nd, end_nd) == (0.0, 3.0):  # 0_1 1_3
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
                elif (start_nd, end_nd) == (1.0, 0.0):  # 0_2 0_1
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
                elif (start_nd, end_nd) == (1.0, 2.0):  # 0_2 1_2
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
                elif (start_nd, end_nd) == (1.0, 4.0):  # 0_2 2_3
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
                elif (start_nd, end_nd) == (2.0, 0.0):  # 1_2 0_1
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
                elif (start_nd, end_nd) == (2.0, 1.0):  # 1_2 0_2
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
                elif (start_nd, end_nd) == (2.0, 3.0):  # 1_2 1_3
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
                elif (start_nd, end_nd) == (2.0, 4.0):  # 1_2 2_3
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
                elif (start_nd, end_nd) == (3.0, 0.0):  # 1_3 0_1
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
                elif (start_nd, end_nd) == (3.0, 2.0):  # 1_3 1_2
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
                elif (start_nd, end_nd) == (3.0, 4.0):  # 1_3 2_3
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
                elif (start_nd, end_nd) == (4.0, 1.0):  # 2_3 0_2
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
                elif (start_nd, end_nd) == (4.0, 2.0):  # 2_3 1_2
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
                elif (start_nd, end_nd) == (4.0, 3.0):  # 2_3 1_3
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
    # check that missing geoms throw an error
    G_test = diamond_graph.copy()
    for start_nd, end_nd, edge_key in G_test.edges(keys=True):
        # delete key from first node and break
        del G_test[start_nd][end_nd][edge_key]["geom"]
        break
    with pytest.raises(KeyError):
        graphs.network_structure_from_nx(G_test, 3395)

    # check that non-LineString geoms throw an error
    G_test = diamond_graph.copy()
    for start_nd, end_nd, edge_key in G_test.edges(keys=True):
        G_test[start_nd][end_nd][edge_key]["geom"] = geometry.Point(
            [G_test.nodes[start_nd]["x"], G_test.nodes[start_nd]["y"]]
        )
    with pytest.raises(TypeError):
        graphs.network_structure_from_nx(G_test, 3395)

    # check that missing node keys throw an error
    G_test = diamond_graph.copy()
    for edge_key in ["x", "y"]:
        for nd_idx in G_test.nodes():
            # delete key from first node and break
            del G_test.nodes[nd_idx][edge_key]
            break
        with pytest.raises(KeyError):
            graphs.network_structure_from_nx(G_test, 3395)

    # check that invalid imp_factors are caught
    G_test = diamond_graph.copy()
    # corrupt imp_factor value and break
    for corrupt_val in [-1, -np.inf, np.nan]:
        for start_nd, end_nd, edge_key in G_test.edges(keys=True):
            G_test[start_nd][end_nd][edge_key]["imp_factor"] = corrupt_val
            break
        with pytest.raises(ValueError):
            graphs.network_structure_from_nx(G_test, 3395)


def test_nx_from_network_structure(primal_graph):
    # also see test_networks.test_to_nx_multigraph for tests on implementation via Network layer
    # check round trip to and from graph maps results in same graph
    # explicitly set live params for equality checks
    # network_structure_from_networkX generates these implicitly if missing
    for node_key in primal_graph.nodes():
        primal_graph.nodes[node_key]["live"] = bool(np.random.randint(0, 2))
    # test directly from and to graph maps
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    G_round_trip = graphs.nx_from_network_structure(nodes_gdf, network_structure)
    assert list(G_round_trip.nodes) == list(primal_graph.nodes)
    assert list(G_round_trip.edges) == list(primal_graph.edges)
    # check with metrics
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    nodes_gdf = networks.node_centrality(
        measures=["node_harmonic"], network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[500, 1000]
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
    G_round_trip_nx = graphs.nx_from_network_structure(
        nodes_gdf,
        network_structure,
    )
    for node_key, node_row in nodes_gdf.iterrows():  # type: ignore
        for col_label in column_labels:
            assert G_round_trip_nx.nodes[node_key][col_label] == node_row[col_label]
    # with backbone
    G_round_trip_nx = graphs.nx_from_network_structure(
        nodes_gdf,
        network_structure,
        nx_multigraph=primal_graph,
    )
    for node_key, node_row in nodes_gdf.iterrows():  # type: ignore
        for col_label in column_labels:
            assert G_round_trip_nx.nodes[node_key][col_label] == node_row[col_label]
    # test with decomposed
    G_decomposed = graphs.nx_decompose(primal_graph, decompose_max=20)
    # set live explicitly
    for node_key in G_decomposed.nodes():
        G_decomposed.nodes[node_key]["live"] = bool(np.random.randint(0, 2))
    nodes_gdf_decomp, network_structure_decomp = graphs.network_structure_from_nx(G_decomposed, 3395)
    G_round_trip_decomp = graphs.nx_from_network_structure(nodes_gdf_decomp, network_structure_decomp, G_decomposed)
    assert list(G_round_trip_decomp.nodes) == list(G_decomposed.nodes)
    for node_key, iter_node_data in G_round_trip_decomp.nodes(data=True):
        assert node_key in G_decomposed
        assert iter_node_data["live"] == G_decomposed.nodes[node_key]["live"]
        assert np.isclose(iter_node_data["x"], G_decomposed.nodes[node_key]["x"], rtol=config.RTOL, atol=config.ATOL)
        assert np.isclose(iter_node_data["y"], G_decomposed.nodes[node_key]["y"], rtol=config.RTOL, atol=config.ATOL)
    assert G_round_trip_decomp.edges == G_decomposed.edges
    # error checks for when using backbone graph:
    # mismatching numbers of nodes
    corrupt_G = primal_graph.copy()
    corrupt_G.remove_node(0)
    with pytest.raises(ValueError):
        graphs.nx_from_network_structure(
            nodes_gdf,
            network_structure,
            nx_multigraph=corrupt_G,
        )
    # mismatching node_key
    with pytest.raises(KeyError):
        corrupt_nodes_gdf = nodes_gdf.copy(deep=True)
        corrupt_index = corrupt_nodes_gdf.index.values
        corrupt_index[0] = -1
        corrupt_nodes_gdf.set_index(corrupt_index)
        graphs.nx_from_network_structure(
            corrupt_nodes_gdf,
            network_structure,
            nx_multigraph=primal_graph,
        )
    # missing edge
    with pytest.raises(KeyError):
        corrupt_primal_graph = primal_graph.copy()
        corrupt_primal_graph.remove_edge(0, 1)
        graphs.nx_from_network_structure(
            nodes_gdf,
            network_structure,
            nx_multigraph=corrupt_primal_graph,
        )
