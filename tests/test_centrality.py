import pytest
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from itertools import permutations
from shapely import geometry
from cityseer import centrality, networks
from cityseer.util import graph_util


def test_generate_graph():

    G, pos = graph_util.tutte_graph()

    #nx.draw(G, pos=pos, with_labels=True)
    #plt.show()

    assert G.number_of_nodes() == 46
    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435


def test_distance_from_beta():

    assert centrality.distance_from_beta(0.04) == (np.array([100.]), 0.01831563888873418)

    assert centrality.distance_from_beta(0.0025) == (np.array([1600.]), 0.01831563888873418)

    arr, t_w = centrality.distance_from_beta(0.04, min_threshold_wt=0.001)
    assert np.array_equal(arr.round(8), np.array([172.69388197]).round(8))
    assert t_w == 0.001

    arr, t_w = centrality.distance_from_beta([0.04, 0.0025])
    assert np.array_equal(arr, np.array([100, 1600]))
    assert t_w == 0.01831563888873418

    with pytest.raises(ValueError):
        centrality.distance_from_beta(-0.04)


def test_networkX_wgs_to_utm():

    # check that node coordinates are correctly converted
    G_utm, pos = graph_util.tutte_graph()
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)
    G_converted = centrality.networkX_wgs_to_utm(G_wgs)
    for n, d in G_utm.nodes(data=True):
        assert d['x'] == round(G_converted.nodes[n]['x'], 1)
        assert d['y'] == round(G_converted.nodes[n]['y'], 1)

    # check that edge coordinates are correctly converted
    G_utm, pos = graph_util.tutte_graph()
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)
    for g in [G_utm, G_wgs]:
        for s, e in g.edges():
            g[s][e]['geom'] = geometry.LineString([
                [g.nodes[s]['x'], g.nodes[s]['y']],
                [g.nodes[e]['x'], g.nodes[e]['y']]
            ])
    G_converted = centrality.networkX_wgs_to_utm(G_wgs)
    for s, e, d in G_utm.edges(data=True):
        assert round(d['geom'].length, 1) == round(G_converted[s][e]['geom'].length, 1)

    # check that non-LineString geoms throw an error
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)
    for s, e in G_wgs.edges():
        G_wgs[s][e]['geom'] = geometry.Point([G_wgs.nodes[s]['x'], G_wgs.nodes[s]['y']])
    with pytest.raises(ValueError):
        centrality.networkX_wgs_to_utm(G_wgs)

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)
        for n in G_wgs.nodes():
            # delete attribute from first node and break
            del G_wgs.nodes[n][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(ValueError):
            centrality.networkX_wgs_to_utm(G_wgs)

    # check that non WGS coordinates throw error
    G_utm, pos = graph_util.tutte_graph()
    with pytest.raises(ValueError):
        centrality.networkX_wgs_to_utm(G_utm)


def test_networkX_decompose():

    # check that missing geoms throw an error
    G, pos = graph_util.tutte_graph()
    with pytest.raises(ValueError):
        centrality.networkX_decompose(G, 20)

    # check that non-LineString geoms throw an error
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(ValueError):
        centrality.networkX_decompose(G, 20)

    # test decomposition
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
                [G.nodes[s]['x'], G.nodes[s]['y']],
                [G.nodes[e]['x'], G.nodes[e]['y']]
            ])
    G_decompose = centrality.networkX_decompose(G, 20)
    assert nx.number_of_nodes(G_decompose) == 602
    assert nx.number_of_edges(G_decompose) == 625

    # check that geoms are correctly flipped
    G_forward, pos = graph_util.tutte_graph()
    for s, e in G_forward.edges():
        G_forward[s][e]['geom'] = geometry.LineString([
                [G_forward.nodes[s]['x'], G_forward.nodes[s]['y']],  # start
                [G_forward.nodes[e]['x'], G_forward.nodes[e]['y']]  # end
            ])
    G_forward_decompose = centrality.networkX_decompose(G_forward, 20)

    G_backward, pos = graph_util.tutte_graph()
    for s, e in G_backward.edges():
        G_backward[s][e]['geom'] = geometry.LineString([
                [G_backward.nodes[e]['x'], G_backward.nodes[e]['y']],  # end
                [G_backward.nodes[s]['x'], G_backward.nodes[s]['y']]  # start
            ])
    G_backward_decompose = centrality.networkX_decompose(G_backward, 20)

    for n, d in G_forward_decompose.nodes(data=True):
        assert d['x'] == G_backward_decompose.nodes[n]['x']
        assert d['y'] == G_backward_decompose.nodes[n]['y']

    # test that geom coordinate mismatch throws an error
    G, pos = graph_util.tutte_graph()
    for n in G.nodes():
        G.nodes[n]['x'] = G.nodes[n]['x'] + 1
        break
    with pytest.raises(ValueError):
        centrality.networkX_decompose(G, 20)


def test_networkX_edge_defaults():

    # check that missing geoms throw an error
    G, pos = graph_util.tutte_graph()
    with pytest.raises(ValueError):
        centrality.networkX_edge_defaults(G)

    # check that non-LineString geoms throw an error
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.Point([G.nodes[s]['x'], G.nodes[s]['y']])
    with pytest.raises(ValueError):
        centrality.networkX_edge_defaults(G)

    # test edge defaults
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
    G_edge_defaults = centrality.networkX_edge_defaults(G)
    for s, e, d in G.edges(data=True):
        assert d['geom'].length == G_edge_defaults[s][e]['length']
        assert d['geom'].length == G_edge_defaults[s][e]['impedance']


def test_networkX_length_weighted_nodes():

    # check that missing length attribute throws error
    G, pos = graph_util.tutte_graph()
    with pytest.raises(ValueError):
        centrality.networkX_length_weighted_nodes(G)

    # test length weighted nodes
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
    G = centrality.networkX_edge_defaults(G)  # generate edge defaults (includes length attribute)
    G = centrality.networkX_length_weighted_nodes(G)  # generate length weighted nodes
    for n, d in G.nodes(data=True):
        agg_length = 0
        for nb in G.neighbors(n):
            agg_length += G[n][nb]['length'] / 2
        assert d['weight'] == agg_length


def test_graph_maps_from_networkX():

    # template graph
    G_template, pos = graph_util.tutte_graph()
    # add geoms
    for s, e in G_template.edges():
        G_template[s][e]['geom'] = geometry.LineString([
            [G_template.nodes[s]['x'], G_template.nodes[s]['y']],
            [G_template.nodes[e]['x'], G_template.nodes[e]['y']]
        ])

    # test maps vs. networkX
    G_test = G_template.copy()
    G_test = centrality.networkX_edge_defaults(G_test)
    # set some random 'live' statuses
    for n in G_test.nodes():
        G_test.nodes[n]['live'] = bool(random.getrandbits(1))
    # randomise the impedances
    for s, e in G_test.edges():
        G_test[s][e]['impedance'] = G_test[s][e]['impedance'] * random.uniform(0, 2)
    # generate length weighted nodes
    G_test = centrality.networkX_length_weighted_nodes(G_test)
    # generate test maps
    node_labels, node_map, edge_map = centrality.graph_maps_from_networkX(G_test)
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
        G_test = centrality.networkX_edge_defaults(G_test)
        for n in G_test.nodes():
            # delete attribute from first node and break
            del G_test.nodes[n][attr]
            break
        with pytest.raises(ValueError):
            centrality.graph_maps_from_networkX(G_test)

    # check that missing edge attributes throw an error
    G_test = G_template.copy()
    for attr in ['length', 'impedance']:
        G_test = centrality.networkX_edge_defaults(G_test)
        for s, e in G_test.edges():
            # delete attribute from first edge and break
            del G_test[s][e][attr]
            break
        with pytest.raises(ValueError):
            centrality.graph_maps_from_networkX(G_test)

    # check that invalid lengths are caught
    G_test = G_template.copy()
    G_test = centrality.networkX_edge_defaults(G_test)
    # corrupt length attribute and break
    for corrupt_val in [0, -1, -np.inf, np.nan]:
        for s, e in G_test.edges():
            G_test[s][e]['length'] = corrupt_val
            break
        with pytest.raises(ValueError):
            centrality.graph_maps_from_networkX(G_test)

    # check that invalid impedances are caught
    G_test = G_template.copy()
    G_test = centrality.networkX_edge_defaults(G_test)
    # corrupt impedance attribute and break
    for corrupt_val in [-1, -np.inf, np.nan]:
        for s, e in G_test.edges():
            G_test[s][e]['length'] = corrupt_val
            break
        with pytest.raises(ValueError):
            centrality.graph_maps_from_networkX(G_test)


def test_shortest_path_tree():

    # for extracting paths from predecessor map
    def find_path(pred_map, src):
        s_path = []
        pred = src
        while True:
            s_path.append(int(pred))
            pred = pred_map[int(pred)]
            if np.isnan(pred):
                break
        return list(reversed(s_path))

    # load the test graph
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
    G = centrality.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = centrality.graph_maps_from_networkX(G)  # generate node and edge maps

    # assume all nodes are reachable
    # i.e. prepare pseudo trim_to_full_idx_map and full_to_trim_idx_map maps consisting of the full range of indices
    pseudo_maps = np.array(list(range(len(n_map))))

    # test shortest path algorithm against networkX
    for max_d in [200, 500, 2000]:
        for i in range(len(G)):
            # check shortest path maps
            map_impedance, map_distance, map_pred, cycles = \
                networks.shortest_path_tree(n_map, e_map, i, pseudo_maps, pseudo_maps, max_dist=max_d)
            nx_dist, nx_path = nx.single_source_dijkstra(G, i, weight='impedance', cutoff=max_d)
            for j in range(len(G)):
                if j in nx_path:
                    assert find_path(map_pred, j) == nx_path[j]
                    assert map_impedance[j] == map_distance[j] == nx_dist[j]

    #TODO: test shortest path on angular metric


def test_compute_centrality():

    # load the test graph
    G, pos = graph_util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
    G = centrality.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = centrality.graph_maps_from_networkX(G)  # generate node and edge maps

    dist = [100]

    # check that malformed signatures trigger errors
    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map[:,:4], e_map, dist, close_metrics=['node_density'])

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map[:,:3], dist, close_metrics=['node_density'])

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map, dist)

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map, dist, close_metrics=['node_density_spelling_typo'])

    # check the number of returned types
    closeness_types = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity', 'cycles']
    betweenness_types = ['betweenness', 'betweenness_gravity']

    centrality.compute_centrality(n_map, e_map, dist, close_metrics=['node_density'])

    for cl_types in permutations(closeness_types):
        for bt_types in permutations(betweenness_types):
            args_ret = centrality.compute_centrality(n_map, e_map, dist,
                                                     close_metrics=list(cl_types), between_metrics=list(bt_types))
            assert len(args_ret) == len(cl_types) + len(bt_types) + 1  # beta list is also added

    # test centrality methods
    # networkx doesn't have a maximum distance cutoff, so have to run on the whole graph
    dist = [2000]
    node_density, harmonic, betweenness, betas = \
        centrality.compute_centrality(n_map, e_map, dist,
                                      close_metrics=['node_density', 'harmonic'], between_metrics=['betweenness'])

    # check betas
    for b, d in zip(betas, dist):
        assert np.exp(b * d) == 0.01831563888873418

    # test node density
    # node density count doesn't include self-node
    for n in node_density[0]:
        assert n + 1 == len(G)

    # NOTE: modified improved closeness is not comparable to networkx version
    # test harmonic closeness
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.array_equal(nx_harm_cl.round(8), harmonic[0].round(8))

    # test betweenness
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    assert np.array_equal(nx_betw, betweenness[0])

    # test easy wrapper version of harmonic closeness
    harmonic_easy, betas = centrality.compute_harmonic_closeness(n_map, e_map, dist)
    assert np.array_equal(nx_harm_cl.round(8), harmonic_easy[0].round(8))

    # test easy wrapper version of betweenness
    betweenness_easy, betas = centrality.compute_betweenness(n_map, e_map, dist)
    assert np.array_equal(nx_betw, betweenness_easy[0])
