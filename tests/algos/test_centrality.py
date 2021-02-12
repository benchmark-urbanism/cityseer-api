import os
import networkx as nx
import numpy as np
import pytest
import timeit

from cityseer.algos import centrality
from cityseer.metrics import networks
from cityseer.util import graphs, plot
from cityseer.util.mock import primal_graph, dual_graph, diamond_graph


def find_path(start_idx, target_idx, tree_preds):
    '''
    for extracting paths from predecessor map
    '''
    s_path = []
    pred = start_idx
    while True:
        s_path.append(pred)
        if pred == target_idx:
            break
        pred = tree_preds[pred].astype(int)
    return list(reversed(s_path))


def test_shortest_path_tree(primal_graph, dual_graph):
    node_uids_p, node_data_p, edge_data_p, node_edge_map_p = graphs.graph_maps_from_nX(primal_graph)
    # prepare round-trip graph for checks
    G_round_trip = graphs.nX_from_graph_maps(node_uids_p, node_data_p, edge_data_p, node_edge_map_p)
    # prepare dual graph
    node_uids_d, node_data_d, edge_data_d, node_edge_map_d = graphs.graph_maps_from_nX(dual_graph)
    assert len(node_uids_d) > len(node_uids_p)
    # test all shortest paths against networkX version of dijkstra
    for max_dist in [0, 500, 2000, np.inf]:
        for src_idx in range(len(primal_graph)):
            # check shortest path maps
            tree_map, tree_edges = centrality.shortest_path_tree(edge_data_p,
                                                                 node_edge_map_p,
                                                                 src_idx,
                                                                 max_dist=max_dist,
                                                                 angular=False)
            tree_preds_p = tree_map[:, 1]
            tree_dists_p = tree_map[:, 2]
            tree_imps_p = tree_map[:, 3]
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G_round_trip, src_idx, weight='length', cutoff=max_dist)
            for j in range(len(primal_graph)):
                if j in nx_path:
                    assert find_path(j, src_idx, tree_preds_p) == nx_path[j]
                    assert np.allclose(tree_imps_p[j], tree_dists_p[j], atol=0.001, rtol=0)
                    assert np.allclose(tree_imps_p[j], nx_dist[j], atol=0.001, rtol=0)
    # compare angular impedances and paths for a selection of targets on primal vs. dual
    # remember, this is angular change not distance travelled
    # can be compared from primal to dual in this instance because edge segments are straight
    # i.e. same amount of angular change whether primal or dual graph
    # plot.plot_nX_primal_or_dual(primal=primal_graph, dual=dual_graph, labels=True)
    p_source_idx = node_uids_p.index(0)
    primal_targets = (15, 20, 37)
    dual_sources = ('0_1', '0_16', '0_31')
    dual_targets = ('13_15', '17_20', '36_37')
    for p_target, d_source, d_target in zip(primal_targets, dual_sources, dual_targets):
        p_target_idx = node_uids_p.index(p_target)
        d_source_idx = node_uids_d.index(d_source)  # dual source index changes depending on direction
        d_target_idx = node_uids_d.index(d_target)
        tree_map_p, tree_edges_p = centrality.shortest_path_tree(edge_data_p,
                                                                 node_edge_map_p,
                                                                 p_source_idx,
                                                                 max_dist=max_dist,
                                                                 angular=True)
        tree_imps_p = tree_map_p[:, 3]
        tree_map_d, tree_edges_d = centrality.shortest_path_tree(edge_data_d,
                                                                 node_edge_map_d,
                                                                 d_source_idx,
                                                                 max_dist=max_dist,
                                                                 angular=True)
        tree_imps_d = tree_map_d[:, 3]
        assert np.allclose(tree_imps_p[p_target_idx], tree_imps_d[d_target_idx], atol=0.001, rtol=0)
    # angular impedance should take a simpler but longer path - test basic case on dual
    # source and target are the same for either
    src_idx = node_uids_d.index('11_6')
    target = node_uids_d.index('39_40')
    # SIMPLEST PATH: get simplest path tree using angular impedance
    tree_map, tree_edges = centrality.shortest_path_tree(edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=True)  # ANGULAR = TRUE
    # find path
    tree_preds = tree_map[:, 1]
    path = find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    # takes 1597m route via long outside segment
    # tree_dists[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '11_14', '10_14', '10_43', '43_44', '40_44', '39_40']
    # SHORTEST PATH:
    # get shortest path tree using non angular impedance
    tree_map, tree_edges = centrality.shortest_path_tree(edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=False)  # ANGULAR = FALSE
    # find path
    tree_preds = tree_map[:, 1]
    path = find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    # takes 1345m shorter route
    # tree_dists[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '6_7', '3_7', '3_4', '1_4', '0_1', '0_31', '31_32', '32_34', '34_37', '37_39',
                              '39_40']
    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src_idx = node_uids_d.index('10_43')
    target = node_uids_d.index('10_5')
    tree_map, tree_edges = centrality.shortest_path_tree(edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=True)
    # find path
    tree_preds = tree_map[:, 1]
    path = find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    # print(path_transpose)
    assert path_transpose == ['10_43', '10_5']
    # WITH SIDESTEPS - set angular flag to False
    # manually overwrite distance impedances with angular for this test
    # (angular has to be false otherwise shortest-path sidestepping avoided)
    edge_data_d_temp = edge_data_d.copy()
    # angular impedances at index 3 copied to distance impedances at distance 2
    edge_data_d_temp[:, 2] = edge_data_d_temp[:, 3]
    tree_map, tree_edges = centrality.shortest_path_tree(edge_data_d_temp,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=False)
    # find path
    tree_preds = tree_map[:, 1]
    path = find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    assert path_transpose == ['10_43', '10_14', '10_5']


def test_local_node_centrality(primal_graph):
    '''
    Also tested indirectly via test_networks.test_compute_centrality

    Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    Note that NetworkX improved closeness is not the same as derivation used in this package
    NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)
    '''
    # generate node and edge maps
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(primal_graph)
    G_round_trip = graphs.nX_from_graph_maps(node_uids, node_data, edge_data, node_edge_map)
    # needs a large enough beta so that distance thresholds aren't encountered
    betas = np.array([-0.02, -0.01, -0.005, -0.0008, -0.0])
    distances = networks.distance_from_beta(betas)
    # set the keys - add shuffling to be sure various orders work
    measure_keys = [
        'node_density',
        'node_farness',
        'node_cycles',
        'node_harmonic',
        'node_beta',
        'node_betweenness',
        'node_betweenness_beta'
    ]
    np.random.shuffle(measure_keys)  # in place
    measure_keys = tuple(measure_keys)
    # generate the measures
    measures_data = centrality.local_node_centrality(node_data,
                                                     edge_data,
                                                     node_edge_map,
                                                     distances,
                                                     betas,
                                                     measure_keys)
    node_density = measures_data[measure_keys.index('node_density')]
    node_farness = measures_data[measure_keys.index('node_farness')]
    node_cycles = measures_data[measure_keys.index('node_cycles')]
    node_harmonic = measures_data[measure_keys.index('node_harmonic')]
    node_beta = measures_data[measure_keys.index('node_beta')]
    node_betweenness = measures_data[measure_keys.index('node_betweenness')]
    node_betweenness_beta = measures_data[measure_keys.index('node_betweenness_beta')]
    # improved closeness is derived after the fact
    improved_closness = node_density / node_farness / node_density

    # test node density
    # node density count doesn't include self-node
    # connected component == 48 == len(G) - 1
    # isolated looping component == 3
    # isolated edge == 1
    # isolated node == 0
    for n in node_density[4]:  # infinite distance - exceeds cutoff clashes
        assert n in [48, 3, 1, 0]

    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G_round_trip, distance='length')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.allclose(nx_harm_cl, node_harmonic[4], atol=0.001, rtol=0)

    # test betweenness vs NetworkX
    # set endpoint counting to false and do not normalise
    # nx node centrality not implemented for MultiGraph
    G_non_multi = nx.Graph()
    G_non_multi.add_nodes_from(G_round_trip.nodes())
    for s, e, k, d in G_round_trip.edges(keys=True, data=True):
        assert k == 0
        G_non_multi.add_edge(s, e, **d)
    nx_betw = nx.betweenness_centrality(G_non_multi, weight='length', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    # nx betweenness gives 0.5 instead of 1 for all disconnected looping component nodes
    # nx presumably takes equidistant routes into account, in which case only the fraction is aggregated
    assert np.allclose(nx_betw[:52], node_betweenness[4][:52], atol=0.001, rtol=0)

    # do the comparisons array-wise so that betweenness can be aggregated
    d_n = len(distances)
    betw = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    betw_wt = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    dens = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    far_imp = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    far_dist = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    harmonic_cl = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    grav = np.full((d_n, primal_graph.number_of_nodes()), 0.0)
    cyc = np.full((d_n, primal_graph.number_of_nodes()), 0.0)

    for src_idx in range(len(primal_graph)):
        # get shortest path maps
        tree_map, tree_edges = centrality.shortest_path_tree(edge_data,
                                                             node_edge_map,
                                                             src_idx,
                                                             max(distances),
                                                             angular=False)
        tree_nodes = np.where(tree_map[:, 0])[0]
        tree_preds = tree_map[:, 1]
        tree_dists = tree_map[:, 2]
        tree_imps = tree_map[:, 3]
        tree_cycles = tree_map[:, 4]
        for to_idx in tree_nodes:
            # skip self nodes
            if to_idx == src_idx:
                continue
            # get distance and impedance
            to_imp = tree_imps[to_idx]
            to_dist = tree_dists[to_idx]
            cycles = tree_cycles[to_idx]
            # continue if exceeds max
            if np.isinf(to_dist):
                continue
            for d_idx in range(len(distances)):
                dist_cutoff = distances[d_idx]
                beta = betas[d_idx]
                if to_dist <= dist_cutoff:
                    # don't exceed threshold
                    # if to_dist <= dist_cutoff:
                    # aggregate values
                    dens[d_idx][src_idx] += 1
                    far_imp[d_idx][src_idx] += to_imp
                    far_dist[d_idx][src_idx] += to_dist
                    harmonic_cl[d_idx][src_idx] += 1 / to_imp
                    grav[d_idx][src_idx] += np.exp(beta * to_dist)
                    # cycles
                    cyc[d_idx][src_idx] += cycles
                    # only process betweenness in one direction
                    if to_idx < src_idx:
                        continue
                    # betweenness - only counting truly between vertices, not starting and ending verts
                    inter_idx = tree_preds[to_idx]
                    # isolated nodes will have no predecessors
                    if np.isnan(inter_idx):
                        continue
                    inter_idx = np.int(inter_idx)
                    while True:
                        # break out of while loop if the intermediary has reached the source node
                        if inter_idx == src_idx:
                            break
                        betw[d_idx][inter_idx] += 1
                        betw_wt[d_idx][inter_idx] += np.exp(beta * to_dist)
                        # follow
                        inter_idx = np.int(tree_preds[inter_idx])
    improved_cl = dens / far_dist / dens

    assert np.allclose(node_density, dens, atol=0.001, rtol=0)
    assert np.allclose(node_farness, far_dist, atol=0.01, rtol=0)  # relax precision
    assert np.allclose(node_cycles, cyc, atol=0.001, rtol=0)
    assert np.allclose(node_harmonic, harmonic_cl, atol=0.001, rtol=0)
    assert np.allclose(node_beta, grav, atol=0.001, rtol=0)
    assert np.allclose(improved_closness, improved_cl, equal_nan=True, atol=0.001, rtol=0)
    assert np.allclose(node_betweenness, betw, atol=0.001, rtol=0)
    assert np.allclose(node_betweenness_beta, betw_wt, atol=0.001, rtol=0)

    # catch typos
    with pytest.raises(ValueError):
        centrality.local_node_centrality(node_data,
                                         edge_data,
                                         node_edge_map,
                                         distances,
                                         betas,
                                         ('typo_key',))


def test_local_centrality(diamond_graph):
    '''
    manual checks for all methods against diamond graph
    measures_data is multidimensional in the form of measure_keys x distances x nodes
    '''
    # generate node and edge maps
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(diamond_graph)
    # generate dual
    diamond_graph_dual = graphs.nX_to_dual(diamond_graph)
    node_uids_dual, node_data_dual, edge_data_dual, node_edge_map_dual = graphs.graph_maps_from_nX(diamond_graph_dual)
    # setup distances and betas
    distances = np.array([50, 150, 250])
    betas = networks.beta_from_distance(distances)

    # NODE SHORTEST
    # set the keys - add shuffling to be sure various orders work
    node_keys = [
        'node_density',
        'node_farness',
        'node_cycles',
        'node_harmonic',
        'node_beta',
        'node_betweenness',
        'node_betweenness_beta'
    ]
    np.random.shuffle(node_keys)  # in place
    measure_keys = tuple(node_keys)
    measures_data = centrality.local_node_centrality(node_data,
                                                     edge_data,
                                                     node_edge_map,
                                                     distances,
                                                     betas,
                                                     measure_keys)
    # node density
    # additive nodes
    m_idx = node_keys.index('node_density')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [2, 3, 3, 2], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [3, 3, 3, 3], atol=0.001, rtol=0)
    # node farness
    # additive distances
    m_idx = node_keys.index('node_farness')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [200, 300, 300, 200], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [400, 300, 300, 400], atol=0.001, rtol=0)
    # node cycles
    # additive cycles
    m_idx = node_keys.index('node_cycles')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [1, 2, 2, 1], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [2, 2, 2, 2], atol=0.001, rtol=0)
    # node harmonic
    # additive 1 / distances
    m_idx = node_keys.index('node_harmonic')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0.02, 0.03, 0.03, 0.02], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [0.025, 0.03, 0.03, 0.025], atol=0.001, rtol=0)
    # node beta
    # additive exp(beta * dist)
    m_idx = node_keys.index('node_beta')
    # beta = -0.0
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    # beta = -0.02666667
    assert np.allclose(measures_data[m_idx][1], [0.1389669, 0.20845035, 0.20845035, 0.1389669], atol=0.001, rtol=0)
    # beta = -0.016
    assert np.allclose(measures_data[m_idx][2], [0.44455525, 0.6056895, 0.6056895, 0.44455522], atol=0.001, rtol=0)
    # node betweenness
    # additive 1 per node en route
    m_idx = node_keys.index('node_betweenness')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=0.001, rtol=0)
    # takes first out of multiple equidistant routes
    assert np.allclose(measures_data[m_idx][2], [0, 1, 0, 0], atol=0.001, rtol=0)
    # node betweenness beta
    # additive exp(beta * dist) en route
    m_idx = node_keys.index('node_betweenness_beta')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)  # beta = -0.08
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=0.001, rtol=0)  # beta = -0.02666667
    # takes first out of multiple equidistant routes
    # beta evaluated over 200m distance from 3 to 0 via node 1
    assert np.allclose(measures_data[m_idx][2], [0, 0.0407622, 0, 0])  # beta = -0.016

    # NODE SIMPLEST
    node_keys_angular = [
        'node_harmonic_angular',
        'node_betweenness_angular'
    ]
    np.random.shuffle(node_keys_angular)  # in place
    measure_keys = tuple(node_keys_angular)
    measures_data = centrality.local_node_centrality(node_data,
                                                     edge_data,
                                                     node_edge_map,
                                                     distances,
                                                     betas,
                                                     measure_keys,
                                                     angular=True)
    # node harmonic angular
    # additive 1 / (1 + (to_imp / 180))
    m_idx = node_keys_angular.index('node_harmonic_angular')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [2, 3, 3, 2], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [2.75, 3, 3, 2.75], atol=0.001, rtol=0)
    # node betweenness angular
    # additive 1 per node en simplest route
    m_idx = node_keys_angular.index('node_betweenness_angular')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [0, 1, 0, 0], atol=0.001, rtol=0)

    # NODE SIMPLEST ON DUAL
    node_keys_angular = [
        'node_harmonic_angular',
        'node_betweenness_angular'
    ]
    np.random.shuffle(node_keys_angular)  # in place
    measure_keys = tuple(node_keys_angular)
    measures_data = centrality.local_node_centrality(node_data_dual,
                                                     edge_data_dual,
                                                     node_edge_map_dual,
                                                     distances,
                                                     betas,
                                                     measure_keys,
                                                     angular=True)
    # node_uids_dual = ('0_1', '0_2', '1_2', '1_3', '2_3')
    # node harmonic angular
    # additive 1 / (1 + (to_imp / 180))
    m_idx = node_keys_angular.index('node_harmonic_angular')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [1.95, 1.95, 2.4, 1.95, 1.95], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [2.45, 2.45, 2.4, 2.45, 2.45], atol=0.001, rtol=0)
    # node betweenness angular
    # additive 1 per node en simplest route
    m_idx = node_keys_angular.index('node_betweenness_angular')
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [0, 0, 0, 1, 1], atol=0.001, rtol=0)

    # SEGMENT SHORTEST
    segment_keys = [
        'segment_density',
        'segment_harmonic',
        'segment_beta',
        'segment_betweenness'
    ]
    np.random.shuffle(segment_keys)  # in place
    measure_keys = tuple(segment_keys)
    measures_data = centrality.local_segment_centrality(node_data,
                                                        edge_data,
                                                        node_edge_map,
                                                        distances,
                                                        betas,
                                                        measure_keys,
                                                        angular=False)
    # segment density
    # additive segment lengths
    m_idx = segment_keys.index('segment_density')
    assert np.allclose(measures_data[m_idx][0], [100, 150, 150, 100], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [400, 500, 500, 400], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [500, 500, 500, 500], atol=0.001, rtol=0)
    # segment harmonic
    # segments are potentially approached from two directions
    # i.e. along respective shortest paths to intersection of shortest routes
    # i.e. in this case, the midpoint of the middle segment is apportioned in either direction
    # additive log(b) - log(a) + log(d) - log(c)
    # nearer distance capped at 1m to avert negative numbers
    m_idx = segment_keys.index('segment_harmonic')
    assert np.allclose(measures_data[m_idx][0], [7.824046, 11.736069, 11.736069, 7.824046], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [10.832201, 15.437371, 15.437371, 10.832201], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [11.407564, 15.437371, 15.437371, 11.407565], atol=0.001, rtol=0)
    # segment beta
    # additive (np.exp(beta * b) - np.exp(beta * a)) / beta + (np.exp(beta * d) - np.exp(beta * c)) / beta
    # beta = 0 resolves to b - a and avoids division through zero
    m_idx = segment_keys.index('segment_beta')
    assert np.allclose(measures_data[m_idx][0], [24.542109, 36.813164, 36.813164, 24.542109], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [77.46391, 112.358284, 112.358284, 77.46391], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [133.80205, 177.43903, 177.43904, 133.80205], atol=0.001, rtol=0)
    # segment betweenness
    # similar formulation to segment beta: start and end segment of each betweenness pair assigned to intervening nodes
    # distance thresholds are computed using the inside edges of the segments
    # so if the segments are touching, they will count up to the threshold distance...
    m_idx = segment_keys.index('segment_betweenness')
    assert np.allclose(measures_data[m_idx][0], [0, 24.542109, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0, 69.78874, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [0, 99.76293, 0, 0], atol=0.001, rtol=0)

    # SEGMENT SIMPLEST ON PRIMAL!!! ( NO DOUBLE COUNTING )
    segment_keys_angular = [
        'segment_harmonic_hybrid',
        'segment_betweeness_hybrid'
    ]
    np.random.shuffle(segment_keys_angular)  # in place
    measure_keys = tuple(segment_keys_angular)
    measures_data = centrality.local_segment_centrality(node_data,
                                                        edge_data,
                                                        node_edge_map,
                                                        distances,
                                                        betas,
                                                        measure_keys,
                                                        angular=True)
    # segment density
    # additive segment lengths divided through angular impedance
    # (f - e) / (1 + (ang / 180))
    m_idx = segment_keys_angular.index('segment_harmonic_hybrid')
    assert np.allclose(measures_data[m_idx][0], [100, 150, 150, 100], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [305, 360, 360, 305], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [410, 420, 420, 410], atol=0.001, rtol=0)
    # segment harmonic
    # additive segment lengths / (1 + (ang / 180))
    m_idx = segment_keys_angular.index('segment_betweeness_hybrid')
    assert np.allclose(measures_data[m_idx][0], [0, 75, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][1], [0, 150, 0, 0], atol=0.001, rtol=0)
    assert np.allclose(measures_data[m_idx][2], [0, 150, 0, 0], atol=0.001, rtol=0)

    # SEGMENT SIMPLEST IS DISCOURAGED FOR DUAL
    # this is because it leads to double counting where segments overlap
    # e.g. 6 segments replace a single four-way intersection
    # it also causes issuse with sidestepping vs. discovering all necessary edges...


def test_decomposed_local_centrality(primal_graph):
    # centralities on the original nodes within the decomposed network should equal non-decomposed workflow
    betas = np.array([-0.02, -0.01, -0.005, -0.0008, -0.0])
    distances = networks.distance_from_beta(betas)
    node_measure_keys = ('node_density',
                         'node_farness',
                         'node_cycles',
                         'node_harmonic',
                         'node_beta',
                         'node_betweenness',
                         'node_betweenness_beta')
    segment_measure_keys = (
        'segment_density',
        'segment_harmonic',
        'segment_beta',
        'segment_betweenness')
    # test a decomposed graph
    G_decomposed = graphs.nX_decompose(primal_graph, 20)
    # graph maps
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(
        primal_graph)  # generate node and edge maps
    node_uids_decomp, node_data_decomp, edge_data_decomp, node_edge_map_decomp = graphs.graph_maps_from_nX(G_decomposed)
    # non-decomposed case
    node_measures_data = centrality.local_node_centrality(node_data,
                                                          edge_data,
                                                          node_edge_map,
                                                          distances,
                                                          betas,
                                                          node_measure_keys,
                                                          angular=False)
    # decomposed case
    node_measures_data_decomposed = centrality.local_node_centrality(node_data_decomp,
                                                                     edge_data_decomp,
                                                                     node_edge_map_decomp,
                                                                     distances,
                                                                     betas,
                                                                     node_measure_keys,
                                                                     angular=False)
    # node
    d_range = len(distances)
    m_range = len(node_measure_keys)
    assert node_measures_data.shape == (m_range, d_range, len(primal_graph))
    assert node_measures_data_decomposed.shape == (m_range, d_range, len(G_decomposed))
    # with increasing decomposition:
    # - node based measures will not match
    # - closeness segment measures will match - these measure to the cut endpoints per thresholds
    # - betweenness segment measures won't match - don't measure to cut endpoints
    # segment versions
    segment_measures_data = centrality.local_segment_centrality(node_data,
                                                                edge_data,
                                                                node_edge_map,
                                                                distances,
                                                                betas,
                                                                segment_measure_keys,
                                                                angular=False)
    segment_measures_data_decomposed = centrality.local_segment_centrality(node_data_decomp,
                                                                           edge_data_decomp,
                                                                           node_edge_map_decomp,
                                                                           distances,
                                                                           betas,
                                                                           segment_measure_keys,
                                                                           angular=False)
    m_range = len(segment_measure_keys)
    assert segment_measures_data.shape == (m_range, d_range, len(primal_graph))
    assert segment_measures_data_decomposed.shape == (m_range, d_range, len(G_decomposed))
    for m_idx in range(m_range):
        for d_idx in range(d_range):
            match = np.allclose(segment_measures_data[m_idx][d_idx],
                                # compare against the original 56 elements (prior to adding decomposed)
                                segment_measures_data_decomposed[m_idx][d_idx][:56],
                                atol=0.1,
                                rtol=0)  # relax precision
            if m_range in (0, 1, 2):
                assert match


def test_local_centrality_time(primal_graph):
    '''
    originally based on node_harmonic and node_betweenness:
    OLD VERSION with trim maps:
    Timing: 10.490865555 for 10000 iterations
    NEW VERSION with numba typed list - faster and removes arcane full vs. trim maps workflow
    8.24 for 10000 iterations
    VERSION with node_edge_map Dict - tad slower but worthwhile for cleaner and more intuitive code
    8.88 for 10000 iterations
    VERSION with shortest path tree algo simplified to nodes and non-angular only
    8.19 for 10000 iterations

    float64 - 17.881911942000002
    float32 - 13.612861239
    segments of unreachable code used to add to timing regardless...
    this seems to have been fixed in more recent versions of numba

    separating the logic into functions results in ever so slightly slower times...
    though this may be due to function setup at invocation (x10000) which wouldn't be incurred in real scenarios...?

    tests on using a List(Dict('x', 'y', etc.) structure proved almost four times slower, so sticking with arrays

    thoughts of using golang proved too complex re: bindings...
    '''
    # load the test graph
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(
        primal_graph)  # generate node and edge maps
    # needs a large enough beta so that distance thresholds aren't encountered
    distances = np.array([np.inf])
    betas = networks.beta_from_distance(distances)

    # setup timing wrapper
    def wrapper_func():
        '''
        node density invokes aggregative workflow
        betweenness node invokes betweenness workflow
        segment density invokes segments workflow
        '''
        return centrality.local_node_centrality(node_data,
                                                edge_data,
                                                node_edge_map,
                                                distances,
                                                betas,
                                                (
                                                    'node_density',
                                                    # 8.73 / strip out everything except node density = 8.5s
                                                    # 'node_betweenness',  # 9.34 / 9.56 stacked w. above
                                                    # 'segment_density',  # 11.19 / 12.14 stacked w. above
                                                    # 'segment_betweenness',  # 9.96 / 13.44 stacked w. above
                                                ),
                                                angular=False,
                                                suppress_progress=True)

    # prime the function
    wrapper_func()
    iters = 10000
    # time and report
    func_time = timeit.timeit(wrapper_func, number=iters)
    print(f'Timing: {func_time} for {iters} iterations')
    if 'GITHUB_ACTIONS' not in os.environ:
        assert func_time < 20
