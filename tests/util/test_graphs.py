import pytest
import numpy as np
import networkx as nx
from shapely import geometry
from cityseer.util import mock, graphs, plot
from cityseer.metrics import centrality


def test_networkX_simple_geoms():

    G, pos = mock.mock_graph()
    G_geoms = graphs.networkX_simple_geoms(G)

    for s, e in G.edges():
        line_geom = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
        assert line_geom == G_geoms[s][e]['geom']

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G, pos_wgs = mock.mock_graph(wgs84_coords=True)
        for n in G.nodes():
            # delete attribute from first node and break
            del G.nodes[n][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            graphs.networkX_simple_geoms(G)


def test_networkX_wgs_to_utm():

    # check that node coordinates are correctly converted
    G_utm, pos = mock.mock_graph()
    G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
    G_converted = graphs.networkX_wgs_to_utm(G_wgs)
    for n, d in G_utm.nodes(data=True):
        # rounding can be tricky
        assert abs(d['x'] - G_converted.nodes[n]['x']) < 0.01
        assert abs(d['y'] - G_converted.nodes[n]['y']) < 0.01

    # check that edge coordinates are correctly converted
    G_utm, pos = mock.mock_graph()
    G_utm = graphs.networkX_simple_geoms(G_utm)

    G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
    G_wgs = graphs.networkX_simple_geoms(G_wgs)

    G_converted = graphs.networkX_wgs_to_utm(G_wgs)
    for s, e, d in G_utm.edges(data=True):
        assert round(d['geom'].length, 1) == round(G_converted[s][e]['geom'].length, 1)

    # check that non-LineString geoms throw an error
    G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
    for s, e in G_wgs.edges():
        G_wgs[s][e]['geom'] = geometry.Point([G_wgs.nodes[s]['x'], G_wgs.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.networkX_wgs_to_utm(G_wgs)

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
        for n in G_wgs.nodes():
            # delete attribute from first node and break
            del G_wgs.nodes[n][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            graphs.networkX_wgs_to_utm(G_wgs)

    # check that non WGS coordinates throw error
    G_utm, pos = mock.mock_graph()
    with pytest.raises(AttributeError):
        graphs.networkX_wgs_to_utm(G_utm)


def test_networkX_remove_straight_intersections():

    # test that redundant (straight) intersections are removed
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G_messy = G.copy()

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
            # add new edges
            # TODO: change to ops.substring once shapely 1.7 released (bug fix)
            G_messy.add_edge(s, f'{s}-{e}', geom=graphs.substring(line_geom, 0, 0.5, normalized=True))
            G_messy.add_edge(e, f'{s}-{e}', geom=graphs.substring(line_geom, 0.5, 1, normalized=True))

    # simplify and test
    G_simplified = graphs.networkX_remove_straight_intersections(G_messy)
    assert G_simplified.nodes == G.nodes
    assert G_simplified.edges == G.edges
    for s, e, d in G_simplified.edges(data=True):
        assert G_simplified[s][e]['geom'].length == G[s][e]['geom'].length

    # check that missing geoms throw an error
    G_attr = G_messy.copy()
    for i, (s, e) in enumerate(G_attr.edges()):
        if i % 2 == 0:
            del G_attr[s][e]['geom']
    with pytest.raises(AttributeError):
        graphs.networkX_remove_straight_intersections(G_attr)

    # check that non-LineString geoms throw an error
    G_attr = G_messy.copy()
    for s, e in G_attr.edges():
        G_attr[s][e]['geom'] = geometry.Point([G_attr.nodes[s]['x'], G_attr.nodes[s]['y']])
    with pytest.raises(AttributeError):
        graphs.networkX_remove_straight_intersections(G_attr)


def test_networkX_decompose():

    # check that missing geoms throw an error
    G, pos = mock.mock_graph()
    with pytest.raises(AttributeError):
        graphs.networkX_decompose(G, 20)

    # check that non-LineString geoms throw an error
    G, pos = mock.mock_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.networkX_decompose(G, 20)

    # test decomposition
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)

    G_decompose = graphs.networkX_decompose(G, 20)
    assert nx.number_of_nodes(G_decompose) == 602
    assert nx.number_of_edges(G_decompose) == 625

    # check that geoms are correctly flipped
    G_forward, pos = mock.mock_graph()
    G_forward = graphs.networkX_simple_geoms(G_forward)
    G_forward_decompose = graphs.networkX_decompose(G_forward, 20)

    G_backward, pos = mock.mock_graph()
    G_backward = graphs.networkX_simple_geoms(G_backward)
    for i, (s, e, d) in enumerate(G_backward.edges(data=True)):
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d['geom'].coords.xy)
            G[s][e]['geom'] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
    G_backward_decompose = graphs.networkX_decompose(G_backward, 20)

    for n, d in G_forward_decompose.nodes(data=True):
        assert d['x'] == G_backward_decompose.nodes[n]['x']
        assert d['y'] == G_backward_decompose.nodes[n]['y']

    # test that geom coordinate mismatch throws an error
    G, pos = mock.mock_graph()
    for attr in ['x', 'y']:
        for n in G.nodes():
            G.nodes[n][attr] = G.nodes[n][attr] + 1
            break
        with pytest.raises(AttributeError):
            graphs.networkX_decompose(G, 20)


def test_networkX_to_dual():

    # check that missing geoms throw an error
    G, pos = mock.mock_graph()
    with pytest.raises(AttributeError):
        graphs.networkX_to_dual(G)

    # check that non-LineString geoms throw an error
    G, pos = mock.mock_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.networkX_to_dual(G)

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G, pos = mock.mock_graph()
        for n in G.nodes():
            # delete attribute from first node and break
            del G.nodes[n][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            graphs.networkX_to_dual(G)

    # test dual
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    # complexify the geoms to check with and without kinks, and in mixed forward and reverse directions
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
            for n in [start, mid, end]:
                n = list(n)
                n.append(10)
            G[s][e]['geom'] = geometry.LineString([start, mid, end])
        # flip each third geom
        if i % 3 == 0:
            flipped_coords = np.fliplr(d['geom'].coords.xy)
            G[s][e]['geom'] = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
    G_dual = graphs.networkX_to_dual(G)
    # dual nodes should equal primal edges
    assert G_dual.number_of_nodes() == G.number_of_edges()
    # all new nodes should have in-out-degrees of 4
    for n in G_dual.nodes():
        assert nx.degree(G_dual, n) == 4

    # for debugging
    # plot.plot_networkX_graphs(primal=G, dual=G_dual)


def test_networkX_edge_defaults():

    # check that missing geoms throw an error
    G, pos = mock.mock_graph()
    with pytest.raises(AttributeError):
        graphs.networkX_edge_defaults(G)

    # check that non-LineString geoms throw an error
    G, pos = mock.mock_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(TypeError):
        graphs.networkX_edge_defaults(G)

    # test edge defaults
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G_edge_defaults = graphs.networkX_edge_defaults(G)
    for s, e, d in G.edges(data=True):
        assert d['geom'].length == G_edge_defaults[s][e]['length']
        assert d['geom'].length == G_edge_defaults[s][e]['impedance']


def test_networkX_m_weighted_nodes():

    # check that missing length attribute throws error
    G, pos = mock.mock_graph()
    with pytest.raises(AttributeError):
        graphs.networkX_m_weighted_nodes(G)

    # test length weighted nodes
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    G = graphs.networkX_m_weighted_nodes(G)
    for n, d in G.nodes(data=True):
        agg_length = 0
        for nb in G.neighbors(n):
            agg_length += G[n][nb]['length'] / 2
        assert d['weight'] == agg_length


def test_graph_maps_from_networkX():

    # template graph
    G_template, pos = mock.mock_graph()
    G_template = graphs.networkX_simple_geoms(G_template)

    # test maps vs. networkX
    G_test = G_template.copy()
    G_test = graphs.networkX_edge_defaults(G_test)
    # set some random 'live' statuses
    for n in G_test.nodes():
        G_test.nodes[n]['live'] = bool(np.random.randint(0, 1))
    # randomise the impedances
    for s, e in G_test.edges():
        G_test[s][e]['impedance'] = G_test[s][e]['impedance'] * np.random.random() * 2000
    # generate length weighted nodes
    G_test = graphs.networkX_m_weighted_nodes(G_test)
    # generate test maps
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G_test)
    # debug plot
    # plot.plot_graphs(primal=G_test)
    # plot.plot_graph_maps(node_labels, node_map, edge_map)

    # check lengths
    assert len(node_labels) == len(node_map) == G_test.number_of_nodes()
    assert len(edge_map) == G_test.number_of_edges() * 2
    # check node maps (idx and label match in this case...)
    for n_label in node_labels:
        assert node_map[n_label][0] == G_test.nodes[n_label]['x']
        assert node_map[n_label][1] == G_test.nodes[n_label]['y']
        assert node_map[n_label][2] == G_test.nodes[n_label]['live']
        assert node_map[n_label][4] == G_test.nodes[n_label]['weight']
    # check edge maps (idx and label match in this case...)
    for start, end, length, impedance in edge_map:
        assert length == G_test[start][end]['length']
        assert impedance == G_test[start][end]['impedance']

    # check that missing node attributes throw an error
    G_test = G_template.copy()
    for attr in ['x', 'y']:
        G_test = graphs.networkX_edge_defaults(G_test)
        for n in G_test.nodes():
            # delete attribute from first node and break
            del G_test.nodes[n][attr]
            break
        with pytest.raises(AttributeError):
            graphs.graph_maps_from_networkX(G_test)

    # check that missing edge attributes throw an error
    G_test = G_template.copy()
    for attr in ['length', 'impedance']:
        G_test = graphs.networkX_edge_defaults(G_test)
        for s, e in G_test.edges():
            # delete attribute from first edge and break
            del G_test[s][e][attr]
            break
        with pytest.raises(AttributeError):
            graphs.graph_maps_from_networkX(G_test)

    # check that invalid lengths are caught
    G_test = G_template.copy()
    G_test = graphs.networkX_edge_defaults(G_test)
    # corrupt length attribute and break
    for corrupt_val in [0, -1, -np.inf, np.nan]:
        for s, e in G_test.edges():
            G_test[s][e]['length'] = corrupt_val
            break
        with pytest.raises(AttributeError):
            graphs.graph_maps_from_networkX(G_test)

    # check that invalid impedances are caught
    G_test = G_template.copy()
    G_test = graphs.networkX_edge_defaults(G_test)
    # corrupt impedance attribute and break
    for corrupt_val in [-1, -np.inf, np.nan]:
        for s, e in G_test.edges():
            G_test[s][e]['length'] = corrupt_val
            break
        with pytest.raises(AttributeError):
            graphs.graph_maps_from_networkX(G_test)


def test_networkX_from_graph_maps():

    # check round trip to and from graph maps results in same graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    # explicitly set live and weight params for equality checks
    # graph_maps_from_networkX generates these implicitly if missing
    for n in G.nodes():
        G.nodes[n]['live'] = bool(np.random.randint(0, 1))
        G.nodes[n]['weight'] = np.random.random() * 2000
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    G_round_trip = graphs.networkX_from_graph_maps(node_labels, node_map, edge_map)
    assert G_round_trip.nodes == G.nodes
    assert G_round_trip.edges == G.edges

    # check that mismatching node_labels length triggers error
    with pytest.raises(ValueError):
        graphs.networkX_from_graph_maps(node_labels[:-1], node_map, edge_map)

    # check that malformed node or edge maps trigger errors
    with pytest.raises(ValueError):
        graphs.networkX_from_graph_maps(node_labels, node_map[:, :4], edge_map)

    with pytest.raises(ValueError):
        graphs.networkX_from_graph_maps(node_labels, node_map, edge_map[:, :3])

    # check with data tuples
    harmonic = centrality.harmonic_closeness(node_map, edge_map, [200])
    data_tuples = [('harmonic_200', harmonic)]
    G_round_trip = graphs.networkX_from_graph_maps(node_labels, node_map, edge_map, node_data=data_tuples)
    for n, d in G_round_trip.nodes(data=True):
        node_idx = node_labels.index(n)
        assert d['harmonic_200'] == harmonic[node_idx]

    # check that malformed tuples raise errors
    for bad_tuple in [(4, harmonic), ('boo', 'baa'), (harmonic, 'boo'), ('boo'), ('boo', harmonic, 'boo')]:
        with pytest.raises(TypeError):
            graphs.networkX_from_graph_maps(node_labels, node_map, edge_map, node_data=[bad_tuple])

    # check that incorrect data tuple enumerable lengths flags error
    with pytest.raises(ValueError):
        graphs.networkX_from_graph_maps(node_labels, node_map, edge_map, node_data=[('boo', harmonic[:-1])])
