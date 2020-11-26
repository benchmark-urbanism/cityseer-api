import networkx as nx
import numpy as np
import pytest
from shapely import geometry, ops

from cityseer.algos import checks
from cityseer.metrics import networks, layers
from cityseer.util import mock, graphs
from cityseer.util.mock import primal_graph, diamond_graph


def test_nX_simple_geoms(primal_graph):
    for s, e in primal_graph.edges():
        line_geom = geometry.LineString([
            [primal_graph.nodes[s]['x'], primal_graph.nodes[s]['y']],
            [primal_graph.nodes[e]['x'], primal_graph.nodes[e]['y']]
        ])
        assert line_geom == primal_graph[s][e]['geom']

    # check that missing node keys throw an error
    for k in ['x', 'y']:
        for n in primal_graph.nodes():
            # delete key from first node and break
            del primal_graph.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nX_simple_geoms(primal_graph)


# TODO:
def test_nX_from_osm():
    pass


def test_nX_wgs_to_utm():
    # check that node coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_converted = graphs.nX_wgs_to_utm(G_wgs)
    for n, d in G_utm.nodes(data=True):
        # rounding can be tricky
        assert np.allclose(d['x'], G_converted.nodes[n]['x'], atol=0.1, rtol=0)
        assert np.allclose(d['y'], G_converted.nodes[n]['y'], atol=0.1, rtol=0)

    # check that edge coordinates are correctly converted
    G_utm = mock.mock_graph()
    G_utm = graphs.nX_simple_geoms(G_utm)

    G_wgs = mock.mock_graph(wgs84_coords=True)
    G_wgs = graphs.nX_simple_geoms(G_wgs)

    G_converted = graphs.nX_wgs_to_utm(G_wgs)
    for s, e, d in G_utm.edges(data=True):
        assert round(d['geom'].length, 1) == round(G_converted[s][e]['geom'].length, 1)

    # check that non-LineString geoms throw an error
    G_wgs = mock.mock_graph(wgs84_coords=True)
    for s, e in G_wgs.edges():
        G_wgs[s][e]['geom'] = geometry.Point([G_wgs.nodes[s]['x'], G_wgs.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.nX_wgs_to_utm(G_wgs)

    # check that missing node keys throw an error
    for k in ['x', 'y']:
        G_wgs = mock.mock_graph(wgs84_coords=True)
        for n in G_wgs.nodes():
            # delete key from first node and break
            del G_wgs.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nX_wgs_to_utm(G_wgs)

    # check that non WGS coordinates throw error
    G_utm = mock.mock_graph()
    with pytest.raises(ValueError):
        graphs.nX_wgs_to_utm(G_utm)

    # check that non-matching UTM zones are coerced to the same zone
    # this scenario spans two UTM zones
    G_wgs_b = nx.Graph()
    nodes = [
        (1, {'x': -0.0005, 'y': 51.572}),
        (2, {'x': -0.0005, 'y': 51.571}),
        (3, {'x': 0.0005, 'y': 51.570}),
        (4, {'x': -0.0005, 'y': 51.569}),
        (5, {'x': -0.0015, 'y': 51.570})
    ]
    G_wgs_b.add_nodes_from(nodes)
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 2)
    ]
    G_wgs_b.add_edges_from(edges)
    G_utm_30 = graphs.nX_wgs_to_utm(G_wgs_b)
    G_utm_30 = graphs.nX_simple_geoms(G_utm_30)

    # if not consistently coerced to UTM zone, the distances from 2-3 and 3-4 will be over 400km
    for s, e, d in G_utm_30.edges(data=True):
        assert d['geom'].length < 200

    # check that explicit zones are respectively coerced
    G_utm_31 = graphs.nX_wgs_to_utm(G_wgs_b, force_zone_number=31)
    G_utm_31 = graphs.nX_simple_geoms(G_utm_31)
    for n, d in G_utm_31.nodes(data=True):
        assert d['x'] != G_utm_30.nodes[n]['x']

    # from cityseer.util import plot
    # plot.plot_nX(G_wgs_b, labels=True)
    # plot.plot_nX(G_utm_b, labels=True)


def make_messy_graph(G):
    # test that redundant (sraight) intersections are removed
    G_messy = G.copy(G)

    # complexify the graph - write changes to new graph to avoid in-place iteration errors
    for i, (s, e, d) in enumerate(G.edges(data=True)):
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d['geom'].coords.xy)
            G_messy[s][e]['geom'] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
        # split each second geom
        if i % 2 == 0:
            line_geom = G[s][e]['geom']
            # check geom coordinates directionality - flip if facing backwards direction
            if not (G.nodes[s]['x'], G.nodes[s]['y']) == line_geom.coords[0][:2]:
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
            G_messy.add_edge(s, f'{s}-{e}', geom=s_geom)
            G_messy.add_edge(e, f'{s}-{e}', geom=e_geom)
            G_messy.nodes[f'{s}-{e}']['x'] = mid_x
            G_messy.nodes[f'{s}-{e}']['y'] = mid_y

    # test recursive weld by manually adding a chained series of orphan nodes
    geom = G[10][43]['geom']
    geom_a = ops.substring(geom, 0, 0.25, normalized=True)
    G_messy.add_edge(10, 't_1', geom=geom_a)
    a_x, a_y = geom_a.coords[-1][:2]
    G_messy.nodes['t_1']['x'] = a_x
    G_messy.nodes['t_1']['y'] = a_y
    geom_b = ops.substring(geom, 0.25, 0.5, normalized=True)
    G_messy.add_edge('t_1', 't_2', geom=geom_b)
    b_x, b_y = geom_b.coords[-1][:2]
    G_messy.nodes['t_2']['x'] = b_x
    G_messy.nodes['t_2']['y'] = b_y
    geom_c = ops.substring(geom, 0.5, 0.75, normalized=True)
    G_messy.add_edge('t_2', 't_3', geom=geom_c)
    c_x, c_y = geom_c.coords[-1][:2]
    G_messy.nodes['t_3']['x'] = c_x
    G_messy.nodes['t_3']['y'] = c_y
    geom_d = ops.substring(geom, 0.75, 1.0, normalized=True)
    G_messy.add_edge('t_3', 43, geom=geom_d)
    # remove original geom
    G_messy.remove_edge(10, 43)

    return G_messy


def test_nX_remove_dangling_nodes(primal_graph):
    G_messy = make_messy_graph(primal_graph)

    # no despining or disconnected components removal
    G_post = graphs.nX_remove_dangling_nodes(G_messy, despine=0, remove_disconnected=False)
    assert G_post.nodes == G_messy.nodes
    assert G_post.edges == G_messy.edges

    # check that all single neighbour nodes have been removed if geom less than despine distance
    G_post = graphs.nX_remove_dangling_nodes(G_messy, despine=100, remove_disconnected=False)
    for n in G_messy.nodes():
        if nx.degree(G_messy, n) == 1:
            nb = list(nx.neighbors(G_messy, n))[0]
            if G_messy[n][nb]['geom'].length <= 100:
                assert (n, nb) not in G_post.edges
            else:
                assert (n, nb) in G_post.edges

    # check that disconnected components are removed
    # this behaviour changed in networkx 2.4
    G_post = graphs.nX_remove_dangling_nodes(G_messy, despine=0, remove_disconnected=True)
    pre_components = list(nx.algorithms.components.connected_components(G_messy))
    post_components = list(nx.algorithms.components.connected_components(G_post))
    assert len(pre_components) != 1
    assert len(post_components) == 1
    # check that components match
    biggest_component = sorted(pre_components, key=len, reverse=True)[0]
    # index to 0 because post_components is still in list form
    assert biggest_component == post_components[0]
    # check that actual graphs are equivalent
    G_biggest_component = nx.Graph(G_messy.subgraph(biggest_component))
    assert G_biggest_component.nodes == G_post.nodes
    assert G_biggest_component.edges == G_post.edges


def test_nX_remove_filler_nodes(primal_graph):
    # test that redundant intersections are removed, i.e. where degree == 2
    G_messy = make_messy_graph(primal_graph)

    # from cityseer.util import plot
    # plot.plot_nX(G_messy, labels=True)

    # simplify and test
    G_simplified = graphs.nX_remove_filler_nodes(G_messy)
    # plot.plot_nX(G_simplified, labels=True)

    # check that the simplified version matches the original un-messified version
    # but note the simplified version will have the disconnected loop of 52-53-54-55 now condensed to only #52
    g_nodes = set(primal_graph.nodes)
    g_nodes = g_nodes.difference([53, 54, 55])
    assert list(g_nodes).sort() == list(G_simplified.nodes).sort()
    g_edges = set(primal_graph.edges)
    g_edges = g_edges.difference([(52, 53), (53, 54), (54, 55), (52, 55)])  # condensed edges
    g_edges = g_edges.union([(52, 52)])  # the new self loop
    assert list(g_edges).sort() == list(G_simplified.edges).sort()

    # check the integrity of the edges
    for s, e, d in G_simplified.edges(data=True):
        # ignore the new self-looping disconnected edge
        if s == 52 and e == 52:
            continue
        assert G_simplified[s][e]['geom'].length == primal_graph[s][e]['geom'].length
    # manually check that the new self-looping edge is equal in length to its original segments
    l = 0
    for s, e in [(52, 53), (53, 54), (54, 55), (52, 55)]:
        l += primal_graph[s][e]['geom'].length
    assert l == G_simplified[52][52]['geom'].length

    # check that all nodes still have 'x' and 'y' keys
    for n, d in G_simplified.nodes(data=True):
        assert 'x' in d
        assert 'y' in d

    # lollipop test - where looping component (all nodes == degree 2) suspend off a node with degree > 2
    # lollipops are handled slightly differently from isolated looping components (all nodes == degree 2)
    # there are no lollipops in the mock graph, so create one here

    # generate graph
    G_lollipop = nx.Graph()
    nodes = [
        (1, {'x': 700400, 'y': 5719750}),
        (2, {'x': 700400, 'y': 5719650}),
        (3, {'x': 700500, 'y': 5719550}),
        (4, {'x': 700400, 'y': 5719450}),
        (5, {'x': 700300, 'y': 5719550})
    ]
    G_lollipop.add_nodes_from(nodes)
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 2)
    ]
    G_lollipop.add_edges_from(edges)

    # add edge geoms
    G_lollipop = graphs.nX_simple_geoms(G_lollipop)

    # flip some geometry
    G_lollipop[2][5]['geom'] = geometry.LineString(G_lollipop[2][5]['geom'].coords[::-1])
    # simplify
    G_lollipop_simpl = graphs.nX_remove_filler_nodes(G_lollipop)

    # check integrity of graph
    assert nx.number_of_nodes(G_lollipop_simpl) == 2
    assert nx.number_of_edges(G_lollipop_simpl) == 2

    # geoms should still be same cumulative length
    before_len = 0
    for s, e, d in G_lollipop.edges(data=True):
        before_len += d['geom'].length
    after_len = 0
    for s, e, d in G_lollipop_simpl.edges(data=True):
        after_len += d['geom'].length
    assert before_len == after_len
    # end point of stick should match start / end point of lollipop
    assert G_lollipop_simpl[1][2]['geom'].coords[-1] == G_lollipop_simpl[2][2]['geom'].coords[0]
    # start and end point of lollipop should match
    assert G_lollipop_simpl[2][2]['geom'].coords[0] == G_lollipop_simpl[2][2]['geom'].coords[-1]

    # check that missing geoms throw an error
    G_k = G_messy.copy()
    for i, (s, e) in enumerate(G_k.edges()):
        if i % 2 == 0:
            del G_k[s][e]['geom']
    with pytest.raises(KeyError):
        graphs.nX_remove_filler_nodes(G_k)

    # check that non-LineString geoms throw an error
    G_k = G_messy.copy()
    for s, e in G_k.edges():
        G_k[s][e]['geom'] = geometry.Point([G_k.nodes[s]['x'], G_k.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.nX_remove_filler_nodes(G_k)

    # catch non-touching Linestrings
    G_corr = G_messy.copy()
    for s, e in G_corr.edges():
        geom = G_corr[s][e]['geom']
        start = list(geom.coords[0])
        end = list(geom.coords[1])
        # corrupt a point
        start[0] = start[0] - 1
        G_corr[s][e]['geom'] = geometry.LineString([start, end])
    with pytest.raises(TypeError):
        graphs.nX_remove_filler_nodes(G_corr)


# this method tests both nX_consolidate_spatial and nX_consolidate_parallel
def test_nX_consolidate():
    # create a test graph
    G = nx.Graph()
    nodes = [
        (0, {'x': 700620, 'y': 5719720}),
        (1, {'x': 700620, 'y': 5719700}),
        (2, {'x': 700660, 'y': 5719720}),
        (3, {'x': 700660, 'y': 5719700}),
        (4, {'x': 700660, 'y': 5719660}),
        (5, {'x': 700700, 'y': 5719800}),
        (6, {'x': 700720, 'y': 5719800}),
        (7, {'x': 700700, 'y': 5719720}),
        (8, {'x': 700720, 'y': 5719720}),
        (9, {'x': 700700, 'y': 5719700}),
        (10, {'x': 700720, 'y': 5719700}),
        (11, {'x': 700700, 'y': 5719620}),
        (12, {'x': 700720, 'y': 5719620}),
        (13, {'x': 700760, 'y': 5719760}),
        (14, {'x': 700800, 'y': 5719760}),
        (15, {'x': 700780, 'y': 5719720}),
        (16, {'x': 700780, 'y': 5719700}),
        (17, {'x': 700840, 'y': 5719720}),
        (18, {'x': 700840, 'y': 5719700})]
    edges = [
        (0, 2),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 7),
        (3, 4),
        (3, 9),
        (5, 7),
        (6, 8),
        (7, 8),
        (7, 9),
        (8, 10),
        (8, 15),
        (9, 11),
        (9, 10),
        (10, 12),
        (10, 16),
        (13, 14),
        (13, 15),
        (14, 15),
        (15, 16),
        (15, 17),
        (16, 18)
    ]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    G = graphs.nX_simple_geoms(G)
    # behaviour confirmed visually
    # from cityseer.util import plot
    # plot.plot_nX(G, labels=True)

    # simplify first to test lollipop self-loop from node 15
    G = graphs.nX_remove_filler_nodes(G)
    # plot.plot_nX(G, labels=True, figsize=(10, 10), dpi=150)
    G_merged_parallel = graphs.nX_consolidate_parallel(G, buffer_dist=25)
    # plot.plot_nX(G_merged_parallel, labels=True, figsize=(10, 10), dpi=150)

    assert G_merged_parallel.number_of_nodes() == 8
    assert G_merged_parallel.number_of_edges() == 8

    node_coords = []
    for n, d in G_merged_parallel.nodes(data=True):
        node_coords.append((d['x'], d['y']))
    assert node_coords == [
        (700660, 5719660),
        (700620.0, 5719710.0),
        (700660.0, 5719710.0),
        (700710.0, 5719800.0),
        (700710.0, 5719710.0),
        (700710.0, 5719620.0),
        (700780.0, 5719710.0),
        (700840.0, 5719710.0)]

    edge_lens = []
    for s, e, d in G_merged_parallel.edges(data=True):
        edge_lens.append(d['geom'].length)
    assert edge_lens == [50.0, 40.0, 50.0, 90.0, 90.0, 70.0, 147.70329614269008, 60.0]

    G_merged_spatial = graphs.nX_consolidate_spatial(G, buffer_dist=25)
    # plot.plot_nX(G_merged_spatial, labels=True)

    assert G_merged_spatial.number_of_nodes() == 8
    assert G_merged_spatial.number_of_edges() == 8

    node_coords = []
    for n, d in G_merged_spatial.nodes(data=True):
        node_coords.append((d['x'], d['y']))
    assert node_coords == [
        (700660, 5719660),
        (700620.0, 5719710.0),
        (700660.0, 5719700.0),
        (700710.0, 5719800.0),
        (700710.0, 5719710.0),
        (700710.0, 5719620.0),
        (700780.0, 5719720.0),
        (700840.0, 5719710.0)]

    edge_lens = []
    for s, e, d in G_merged_spatial.edges(data=True):
        edge_lens.append(d['geom'].length)
    assert edge_lens == [40.0, 41.23105625617661, 50.99019513592785, 90.0, 90.0, 70.71067811865476, 129.4427190999916,
                         60.8276253029822]

    # visual tests on OSM data
    # TODO: can furnish more extensive tests, e.g. to verify veracity of new geoms
    osm_json = mock.mock_osm_data()
    g_1 = graphs.nX_from_osm(osm_json=osm_json)

    osm_json_alt = mock.mock_osm_data(alt=True)
    g_2 = graphs.nX_from_osm(osm_json=osm_json_alt)

    for g in [g_1, g_2]:
        G_utm = graphs.nX_wgs_to_utm(g)
        G = graphs.nX_simple_geoms(G_utm)
        G = graphs.nX_remove_filler_nodes(G)
        G = graphs.nX_remove_dangling_nodes(G)
        # G_decomp = graphs.nX_decompose(G, 25)
        # from cityseer.util import plot
        # plot.plot_nX(G, figsize=(10, 10), dpi=150)
        G_spatial = graphs.nX_consolidate_spatial(G, buffer_dist=15)
        # plot.plot_nX(G_spatial, figsize=(10, 10), dpi=150)
        G_parallel = graphs.nX_consolidate_parallel(G, buffer_dist=14)
        # plot.plot_nX(G_parallel, figsize=(10, 10), dpi=150)


def test_nX_decompose(primal_graph):
    # check that missing geoms throw an error
    G = primal_graph.copy()
    del G[0][1]['geom']
    with pytest.raises(KeyError):
        graphs.nX_decompose(G, 20)

    # check that non-LineString geoms throw an error
    G = primal_graph.copy()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.nX_decompose(G, 20)

    # test decomposition
    G = primal_graph.copy()
    # first clean the graph to strip disconnected looping component
    # this gives a start == end node situation for testing
    G_simple = graphs.nX_remove_filler_nodes(G)
    G_decompose = graphs.nX_decompose(G_simple, 20)

    # from cityseer.util import plot
    # plot.plot_nX(G_simple, labels=True)
    # plot.plot_nX(G_decompose)
    assert nx.number_of_nodes(G_decompose) == 661
    assert nx.number_of_edges(G_decompose) == 682

    # check that total lengths are the same
    G_lens = 0
    for s, e, e_data in G_simple.edges(data=True):
        G_lens += e_data['geom'].length
    G_d_lens = 0
    for s, e, e_data in G_decompose.edges(data=True):
        G_d_lens += e_data['geom'].length
    assert np.allclose(G_lens, G_d_lens, atol=0.001, rtol=0)

    # check that all ghosted edges have one or two edges
    for n, n_data in G_decompose.nodes(data=True):
        if 'ghosted' in n_data and n_data['ghosted']:
            nbs = list(G_decompose.neighbors(n))
            assert len(nbs) == 1 or len(nbs) == 2

    # check that geoms are correctly flipped
    G_forward = primal_graph.copy()
    G_forward_decompose = graphs.nX_decompose(G_forward, 20)

    G_backward = primal_graph.copy()
    for i, (s, e, d) in enumerate(G_backward.edges(data=True)):
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d['geom'].coords.xy)
            G[s][e]['geom'] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
    G_backward_decompose = graphs.nX_decompose(G_backward, 20)

    for n, d in G_forward_decompose.nodes(data=True):
        assert d['x'] == G_backward_decompose.nodes[n]['x']
        assert d['y'] == G_backward_decompose.nodes[n]['y']

    # test that geom coordinate mismatch throws an error
    G = primal_graph.copy()
    for k in ['x', 'y']:
        for n in G.nodes():
            G.nodes[n][k] = G.nodes[n][k] + 1
            break
        with pytest.raises(ValueError):
            graphs.nX_decompose(G, 20)


def test_nX_to_dual(primal_graph, diamond_graph):
    # check that missing geoms throw an error
    G = diamond_graph.copy()
    del G[0][1]['geom']
    with pytest.raises(KeyError):
        graphs.nX_to_dual(G)

    # check that non-LineString geoms throw an error
    G = diamond_graph.copy()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.nX_to_dual(G)

    # check that missing node keys throw an error
    for k in ['x', 'y']:
        G = diamond_graph.copy()
        for n in G.nodes():
            # delete key from first node and break
            del G.nodes[n][k]
            break
        # check that missing key throws an error
        with pytest.raises(KeyError):
            graphs.nX_to_dual(G)

    # test dual
    G = diamond_graph.copy()
    G_dual = graphs.nX_to_dual(G)

    # from cityseer.util import plot
    # plot.plot_nX_primal_or_dual(primal=G, dual=G_dual)

    assert G_dual.number_of_nodes() == G.number_of_edges()
    # the new dual nodes have three edges each, except for the midspan which now has four redges
    for n in G_dual.nodes():
        if n == '1_2':
            assert nx.degree(G_dual, n) == 4
        else:
            assert nx.degree(G_dual, n) == 3
    for start, end, d in G_dual.edges(data=True):
        # the new geoms should also be 100m length (split 50m x 2)
        assert round(d['geom'].length) == 100
        # check the starting and ending bearings per diamond graph
        if (G_dual.nodes[start]['x'], G_dual.nodes[start]['y']) == d['geom'].coords[0]:
            s_x, s_y = d['geom'].coords[0]
            m_x, m_y = d['geom'].coords[1]
            e_x, e_y = d['geom'].coords[-1]
        else:
            s_x, s_y = d['geom'].coords[-1]
            m_x, m_y = d['geom'].coords[1]
            e_x, e_y = d['geom'].coords[0]
        start_bearing = np.rad2deg(np.arctan2(m_y - s_y, m_x - s_x)).round()
        end_bearing = np.rad2deg(np.arctan2(e_y - m_y, e_x - m_x)).round()
        if (start, end) == ('0_1', '0_2'):
            assert (start_bearing, end_bearing) == (-60, 60)
        elif (start, end) == ('0_1', '1_2'):
            assert (start_bearing, end_bearing) == (120, 0)
        elif (start, end) == ('0_1', '1_3'):
            assert (start_bearing, end_bearing) == (120, 60)
        elif (start, end) == ('0_2', '1_2'):
            assert (start_bearing, end_bearing) == (60, 180)
        elif (start, end) == ('0_2', '2_3'):
            assert (start_bearing, end_bearing) == (60, 120)
        elif (start, end) == ('1_2', '1_3'):
            assert (start_bearing, end_bearing) == (180, 60)
        elif (start, end) == ('1_2', '2_3'):
            assert (start_bearing, end_bearing) == (0, 120)
        elif (start, end) == ('1_3', '2_3'):
            assert (start_bearing, end_bearing) == (60, -60)

    # complexify the geoms to check with and without kinks, and in mixed forward and reverse directions
    # see if any issues arise
    G = primal_graph.copy()
    for i, (s, e, d) in enumerate(G.edges(data=True)):
        # add a kink to each second geom
        if i % 2 == 0:
            geom = d['geom']
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
            G[s][e]['geom'] = geometry.LineString(kinked_3d_geom)
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d['geom'].coords.xy)
            G[s][e]['geom'] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
    G_dual = graphs.nX_to_dual(G)


def test_graph_maps_from_nX(diamond_graph):
    # test maps vs. networkX
    G_test = diamond_graph.copy()
    G_test_dual = graphs.nX_to_dual(G_test)
    for G in (G_test, G_test_dual):
        # set some random 'live' statuses
        for n in G.nodes():
            G.nodes[n]['live'] = bool(np.random.randint(0, 1))

        # generate test maps
        node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)
        # debug plot
        # plot.plot_graphs(primal=G)
        # plot.plot_graph_maps(node_uids, node_data, edge_data)

        # run check
        checks.check_network_maps(node_data, edge_data, node_edge_map)

        # check lengths
        assert len(node_uids) == len(node_data) == G.number_of_nodes()
        # edges = x2
        assert len(edge_data) == G.number_of_edges() * 2

        # check node maps (idx and label match in this case...)
        for n_label in node_uids:
            assert node_data[n_label][0] == G.nodes[n_label]['x']
            assert node_data[n_label][1] == G.nodes[n_label]['y']
            assert node_data[n_label][2] == G.nodes[n_label]['live']

        # check edge maps (idx and label match in this case...)
        for start, end, length, angle_sum, imp_factor, start_bearing, end_bearing in edge_data:
            assert np.allclose(length, G[start][end]['geom'].length, atol=0.001, rtol=0)
            assert angle_sum == 0
            assert imp_factor == 1
            s_x, s_y = node_data[int(start)][:2]
            e_x, e_y = node_data[int(end)][:2]
            assert np.allclose(start_bearing, np.rad2deg(np.arctan2(e_y - s_y, e_x - s_x)), atol=0.001, rtol=0)
            assert np.allclose(end_bearing, np.rad2deg(np.arctan2(e_y - s_y, e_x - s_x)), atol=0.001, rtol=0)

        # TODO: rework this to map and compare ...
        for start, end, d in G_dual.edges(data=True):
            # the new geoms should also be 100m length (split 50m x 2)
            assert round(d['geom'].length) == 100
            # check the starting and ending bearings per diamond graph
            if (G_dual.nodes[start]['x'], G_dual.nodes[start]['y']) == d['geom'].coords[0]:
                s_x, s_y = d['geom'].coords[0]
                m_x, m_y = d['geom'].coords[1]
                e_x, e_y = d['geom'].coords[-1]
            else:
                s_x, s_y = d['geom'].coords[-1]
                m_x, m_y = d['geom'].coords[1]
                e_x, e_y = d['geom'].coords[0]
            start_bearing = np.rad2deg(np.arctan2(m_y - s_y, m_x - s_x)).round()
            end_bearing = np.rad2deg(np.arctan2(e_y - m_y, e_x - m_x)).round()
            if (start, end) == ('0_1', '0_2'):
                assert (start_bearing, end_bearing) == (-60, 60)
            elif (start, end) == ('0_1', '1_2'):
                assert (start_bearing, end_bearing) == (120, 0)
            elif (start, end) == ('0_1', '1_3'):
                assert (start_bearing, end_bearing) == (120, 60)
            elif (start, end) == ('0_2', '1_2'):
                assert (start_bearing, end_bearing) == (60, 180)
            elif (start, end) == ('0_2', '2_3'):
                assert (start_bearing, end_bearing) == (60, 120)
            elif (start, end) == ('1_2', '1_3'):
                assert (start_bearing, end_bearing) == (180, 60)
            elif (start, end) == ('1_2', '2_3'):
                assert (start_bearing, end_bearing) == (0, 120)
            elif (start, end) == ('1_3', '2_3'):
                assert (start_bearing, end_bearing) == (60, -60)

    # check that missing geoms throw an error
    G_test = diamond_graph.copy()
    for s, e in G_test.edges():
        # delete key from first node and break
        del G_test[s][e]['geom']
        break
    with pytest.raises(KeyError):
        graphs.graph_maps_from_nX(G_test)

    # check that non-LineString geoms throw an error
    G_test = diamond_graph.copy()
    for s, e in G_test.edges():
        G_test[s][e]['geom'] = geometry.Point([G_test.nodes[s]['x'], G_test.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.graph_maps_from_nX(G_test)

    # check that missing node keys throw an error
    G_test = diamond_graph.copy()
    for k in ['x', 'y']:
        for n in G_test.nodes():
            # delete key from first node and break
            del G_test.nodes[n][k]
            break
        with pytest.raises(KeyError):
            graphs.graph_maps_from_nX(G_test)

    # check that invalid imp_factors are caught
    G_test = diamond_graph.copy()
    # corrupt imp_factor value and break
    for corrupt_val in [-1, -np.inf, np.nan]:
        for s, e in G_test.edges():
            G_test[s][e]['imp_factor'] = corrupt_val
            break
        with pytest.raises(ValueError):
            graphs.graph_maps_from_nX(G_test)


def test_nX_from_graph_maps(primal_graph):
    # also see test_networks.test_to_networkX for tests on implementation via Network layer

    # check round trip to and from graph maps results in same graph
    # explicitly set live params for equality checks
    # graph_maps_from_networkX generates these implicitly if missing
    for n in primal_graph.nodes():
        primal_graph.nodes[n]['live'] = bool(np.random.randint(0, 1))

    # test directly from and to graph maps
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(primal_graph)
    G_round_trip = graphs.nX_from_graph_maps(node_uids, node_data, edge_data, node_edge_map)
    assert list(G_round_trip.nodes) == list(primal_graph.nodes)
    assert list(G_round_trip.edges) == list(primal_graph.edges)

    # check with metrics dictionary
    N = networks.Network_Layer_From_nX(primal_graph, distances=[500, 1000])

    N.compute_node_centrality(measures=['node_harmonic'])
    data_dict = mock.mock_data_dict(primal_graph)
    landuse_labels = mock.mock_categorical_data(len(data_dict))
    D = layers.Data_Layer_From_Dict(data_dict)
    D.assign_to_network(N, max_dist=400)
    D.compute_aggregated(landuse_labels,
                         mixed_use_keys=['hill', 'shannon'],
                         accessibility_keys=['a', 'c'],
                         qs=[0, 1])
    metrics_dict = N.metrics_to_dict()
    # without backbone
    G_round_trip_data = graphs.nX_from_graph_maps(node_uids,
                                                  node_data,
                                                  edge_data,
                                                  node_edge_map,
                                                  metrics_dict=metrics_dict)
    for uid, metrics in metrics_dict.items():
        assert G_round_trip_data.nodes[uid]['metrics'] == metrics
    # with backbone
    G_round_trip_data = graphs.nX_from_graph_maps(node_uids,
                                                  node_data,
                                                  edge_data,
                                                  node_edge_map,
                                                  networkX_graph=primal_graph,
                                                  metrics_dict=metrics_dict)
    for uid, metrics in metrics_dict.items():
        assert G_round_trip_data.nodes[uid]['metrics'] == metrics

    # test with decomposed
    G_decomposed = graphs.nX_decompose(primal_graph, decompose_max=20)
    # set live explicitly
    for n in G_decomposed.nodes():
        G_decomposed.nodes[n]['live'] = bool(np.random.randint(0, 1))
    node_uids_d, node_data_d, edge_data_d, node_edge_map_d = graphs.graph_maps_from_nX(G_decomposed)

    G_round_trip_d = graphs.nX_from_graph_maps(node_uids_d, node_data_d, edge_data_d, node_edge_map_d)
    assert list(G_round_trip_d.nodes) == list(G_decomposed.nodes)
    for n, node_data in G_round_trip.nodes(data=True):
        assert n in G_decomposed
        assert node_data['live'] == G_decomposed.nodes[n]['live']
        assert node_data['x'] == G_decomposed.nodes[n]['x']
        assert node_data['y'] == G_decomposed.nodes[n]['y']
    assert G_round_trip_d.edges == G_decomposed.edges

    # error checks for when using backbone graph:
    # mismatching numbers of nodes
    corrupt_G = primal_graph.copy()
    corrupt_G.remove_node(0)
    with pytest.raises(ValueError):
        graphs.nX_from_graph_maps(node_uids, node_data, edge_data, node_edge_map, networkX_graph=corrupt_G)
    # mismatching node uid
    with pytest.raises(ValueError):
        corrupt_node_uids = list(node_uids)
        corrupt_node_uids[0] = 'boo'
        graphs.nX_from_graph_maps(corrupt_node_uids, node_data, edge_data, node_edge_map, networkX_graph=primal_graph)
