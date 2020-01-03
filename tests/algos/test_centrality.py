import os
import networkx as nx
import numpy as np
import pytest
import timeit

from cityseer.algos import data, centrality, checks
from cityseer.metrics import networks
from cityseer.util import mock, graphs


# for extracting paths from predecessor map
def _find_path(start_idx, target_idx, tree_preds):
    s_path = []
    pred = start_idx
    while True:
        s_path.append(pred)
        if pred == target_idx:
            break
        pred = tree_preds[pred].astype(int)
    return list(reversed(s_path))


def test_shortest_path_tree():
    # prepare primal graph
    G_primal = mock.mock_graph()
    G_primal = graphs.nX_simple_geoms(G_primal)
    node_uids_p, node_data_p, edge_data_p, node_edge_map_p, edge_ghost_map_p = graphs.graph_maps_from_nX(G_primal)
    import numba
    print(numba.typeof(node_edge_map_p))
    # prepare round-trip graph for checks
    G_round_trip = graphs.nX_from_graph_maps(node_uids_p, node_data_p, edge_data_p, node_edge_map_p, edge_ghost_map_p)
    # prepare dual graph
    G_dual = mock.mock_graph()
    G_dual = graphs.nX_simple_geoms(G_dual)
    G_dual = graphs.nX_to_dual(G_dual)
    node_uids_d, node_data_d, edge_data_d, node_edge_map_d, edge_ghost_map_d = graphs.graph_maps_from_nX(G_dual)
    assert len(node_uids_d) > len(node_uids_p)
    # test all shortest paths against networkX version of dijkstra
    for max_dist in [0, 500, 2000, np.inf]:
        for src_idx in range(len(G_primal)):
            # check shortest path maps
            tree_map, tree_edges = centrality.shortest_path_tree(node_data_p,
                                                                 edge_data_p,
                                                                 node_edge_map_p,
                                                                 src_idx,
                                                                 max_dist=max_dist,
                                                                 angular=False)
            tree_preds_p = tree_map[:, 1]
            tree_dists_p = tree_map[:, 2]
            tree_imps_p = tree_map[:, 3]
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G_round_trip, src_idx, weight='length', cutoff=max_dist)
            for j in range(len(G_primal)):
                if j in nx_path:
                    assert _find_path(j, src_idx, tree_preds_p) == nx_path[j]
                    assert np.allclose(tree_imps_p[j], tree_dists_p[j])
                    assert np.allclose(tree_imps_p[j], nx_dist[j])
    # compare angular impedances and paths for a selection of targets on primal vs. dual
    # this works for this graph because edge segments are straight
    p_source_idx = node_uids_p.index(0)
    primal_targets = (15, 20, 37)
    dual_sources = ('0_1', '0_16', '0_31')
    dual_targets = ('13_15', '17_20', '36_37')
    for p_target, d_source, d_target in zip(primal_targets, dual_sources, dual_targets):
        p_target_idx = node_uids_p.index(p_target)
        d_source_idx = node_uids_d.index(d_source)  # dual source index changes depending on direction
        d_target_idx = node_uids_d.index(d_target)
        tree_map_p, tree_edges_p = centrality.shortest_path_tree(node_data_p,
                                                                 edge_data_p,
                                                                 node_edge_map_p,
                                                                 p_source_idx,
                                                                 max_dist=max_dist,
                                                                 angular=True)
        tree_imps_p = tree_map_p[:, 3]
        tree_map_d, tree_edges_d = centrality.shortest_path_tree(node_data_d,
                                                                 edge_data_d,
                                                                 node_edge_map_d,
                                                                 d_source_idx,
                                                                 max_dist=max_dist,
                                                                 angular=True)
        tree_imps_d = tree_map_d[:, 3]
        assert np.allclose(tree_imps_p[p_target_idx], tree_imps_d[d_target_idx])
    # angular impedance should take a simpler but longer path - test basic case on dual
    # for debugging
    # from cityseer.util import plot
    # plot.plot_nX_primal_or_dual(primal=G_primal, dual=G_dual, labels=True)
    # source and target are the same for either
    src_idx = node_uids_d.index('11_6')
    target = node_uids_d.index('39_40')
    # SIMPLEST PATH: get simplest path tree using angular impedance
    tree_map, tree_edges = centrality.shortest_path_tree(node_data_d,
                                                         edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=True)
    # find path
    tree_preds = tree_map[:, 1]
    path = _find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    # takes 1597m route via long outside segment
    # tree_dists[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '11_14', '10_14', '10_43', '43_44', '40_44', '39_40']
    # SHORTEST PATH:
    # get shortest path tree using non angular impedance
    tree_map, tree_edges = centrality.shortest_path_tree(node_data_d,
                                                         edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=False)
    # find path
    tree_preds = tree_map[:, 1]
    path = _find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    # takes 1345m shorter route
    # tree_dists[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '6_7', '3_7', '3_4', '1_4', '0_1', '0_31', '31_32', '32_34', '34_37', '37_39',
                              '39_40']
    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src_idx = node_uids_d.index('10_43')
    target = node_uids_d.index('10_5')
    tree_map, tree_edges = centrality.shortest_path_tree(node_data_d,
                                                         edge_data_d,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=True)
    # find path
    tree_preds = tree_map[:, 1]
    path = _find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    assert path_transpose == ['10_43', '10_5']
    # WITH SIDESTEPS - set angular flag to False
    # manually overwrite distance impedances with angular for this test
    # (angular has to be false otherwise shortest-path sidestepping avoided)
    edge_data_d_temp = edge_data_d.copy()
    # angular impedances at index 3 copied to distance impedances at distance 2
    edge_data_d_temp[:, 2] = edge_data_d_temp[:, 3]
    tree_map, tree_edges = centrality.shortest_path_tree(node_data_d,
                                                         edge_data_d_temp,
                                                         node_edge_map_d,
                                                         src_idx,
                                                         max_dist=np.inf,
                                                         angular=False)
    # find path
    tree_preds = tree_map[:, 1]
    path = _find_path(target, src_idx, tree_preds)
    path_transpose = [node_uids_d[n] for n in path]
    assert path_transpose == ['10_43', '10_14', '10_5']


def test_decomposed_local_centrality():
    # centralities on the original nodes within the decomposed network should equal non-decomposed workflow
    betas = np.array([-0.02, -0.01, -0.005, -0.0008, -0.0])
    distances = networks.distance_from_beta(betas)
    measure_keys = ('node_harmonic', 'node_betweenness')
    # test a decomposed graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map, edge_ghost_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    measures_data = centrality.local_centrality(node_data,
                                                edge_data,
                                                node_edge_map,
                                                edge_ghost_map,
                                                distances,
                                                betas,
                                                measure_keys,
                                                angular=False)
    G_decomposed = graphs.nX_decompose(G, 20)
    # from cityseer.util import plot
    # plot.plot_nX(G_decomposed, labels=True)
    # generate node and edge maps
    node_uids, node_data, edge_data, node_edge_map, edge_ghost_map = graphs.graph_maps_from_nX(G_decomposed)
    checks.check_network_maps(node_data, edge_data, node_edge_map, edge_ghost_map)
    measures_data_decomposed = centrality.local_centrality(node_data,
                                                           edge_data,
                                                           node_edge_map,
                                                           edge_ghost_map,
                                                           distances,
                                                           betas,
                                                           measure_keys,
                                                           angular=False)
    # test harmonic closeness on original nodes for non-decomposed vs decomposed
    d_range = len(distances)
    m_range = len(measure_keys)
    assert measures_data.shape == (m_range, d_range, len(G))
    assert measures_data_decomposed.shape == (m_range, d_range, len(G_decomposed))
    original_node_idx = np.where(node_data[:, 3] == 0)
    for m_idx in range(m_range):
        for d_idx in range(d_range):
            assert np.allclose(measures_data[m_idx][d_idx], measures_data_decomposed[m_idx][d_idx][original_node_idx])


def test_local_centrality():
    '''
    Also tested indirectly via test_networks.test_compute_centrality

    Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    Note that NetworkX improved closeness is not the same as derivation used in this package
    NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)
    '''
    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map, edge_ghost_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    G_round_trip = graphs.nX_from_graph_maps(node_uids, node_data, edge_data, node_edge_map, edge_ghost_map)
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
        'segment_density',
        'segment_harmonic',
        'segment_beta',
        'node_betweenness',
        'node_betweenness_beta',
        'segment_betweenness'
    ]
    np.random.shuffle(measure_keys)  # in place
    measure_keys = tuple(measure_keys)
    # generate the measures
    measures_data = centrality.local_centrality(node_data,
                                                edge_data,
                                                node_edge_map,
                                                edge_ghost_map,
                                                distances,
                                                betas,
                                                measure_keys,
                                                angular=False)
    node_density = measures_data[measure_keys.index('node_density')]
    node_farness = measures_data[measure_keys.index('node_farness')]
    node_cycles = measures_data[measure_keys.index('node_cycles')]
    node_harmonic = measures_data[measure_keys.index('node_harmonic')]
    node_beta = measures_data[measure_keys.index('node_beta')]
    segment_density = measures_data[measure_keys.index('segment_density')]
    segment_harmonic = measures_data[measure_keys.index('segment_harmonic')]
    segment_beta = measures_data[measure_keys.index('segment_beta')]
    node_betweenness = measures_data[measure_keys.index('node_betweenness')]
    node_betweenness_beta = measures_data[measure_keys.index('node_betweenness_beta')]
    segment_betweenness = measures_data[measure_keys.index('segment_betweenness')]
    # post compute improved
    improved_closness = node_density / node_farness / node_density
    # angular keys
    measure_keys_angular = [
        'node_harmonic_angular',
        'segment_harmonic_hybrid',
        'node_betweenness_angular',
        'segment_betweeness_hybrid'
    ]
    np.random.shuffle(measure_keys_angular)  # in place
    measure_keys_angular = tuple(measure_keys_angular)
    # generate the angular measures
    measures_data_angular = centrality.local_centrality(node_data,
                                                        edge_data,
                                                        node_edge_map,
                                                        edge_ghost_map,
                                                        distances,
                                                        betas,
                                                        measure_keys_angular,
                                                        angular=True)
    node_harmonic_angular = measures_data_angular[measure_keys_angular.index('node_harmonic_angular')]
    segment_harmonic_hybrid = measures_data_angular[measure_keys_angular.index('segment_harmonic_hybrid')]
    node_betweenness_angular = measures_data_angular[measure_keys_angular.index('node_betweenness_angular')]
    segment_betweeness_hybrid = measures_data_angular[measure_keys_angular.index('segment_betweeness_hybrid')]

    # test node density
    # node density count doesn't include self-node
    # connected component == 48 == len(G) - 4
    # isolated looping component == 3
    # isolated edge == 1
    # isolated node == 0
    for n in node_density[4]:  # infinite distance - exceeds cutoff clashes
        assert n in [48, 3, 1, 0]
    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G_round_trip, distance='length')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.allclose(nx_harm_cl, node_harmonic[4])

    # test betweenness vs NetworkX
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G_round_trip, weight='length', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    # for some reason nx betweenness gives 0.5 instead of 1 for disconnected looping component (should be 1)
    # maybe two equidistant routes being divided through 2
    # nx betweenness gives 0.5 instead of 1 for all disconnected looping component nodes
    # nx presumably takes equidistant routes into account, in which case only the fraction is aggregated
    assert np.array_equal(nx_betw[:52], node_betweenness[4][:52])

    # test against various distances
    for d_idx in range(len(distances)):
        dist_cutoff = distances[d_idx]
        beta = betas[d_idx]

        # do the comparisons array-wise so that betweenness can be aggregated
        betw = np.full(G.number_of_nodes(), 0.0)
        betw_wt = np.full(G.number_of_nodes(), 0.0)
        dens = np.full(G.number_of_nodes(), 0.0)
        far_imp = np.full(G.number_of_nodes(), 0.0)
        far_dist = np.full(G.number_of_nodes(), 0.0)
        harmonic_cl = np.full(G.number_of_nodes(), 0.0)
        grav = np.full(G.number_of_nodes(), 0.0)
        cyc = np.full(G.number_of_nodes(), 0.0)

        for src_idx in range(len(G)):
            # get shortest path maps
            tree_map, tree_edges = centrality.shortest_path_tree(node_data,
                                                      edge_data,
                                                      node_edge_map,
                                                      src_idx,
                                                      dist_cutoff,
                                                      angular=False)
            tree_preds = tree_map[:, 1]
            tree_dists = tree_map[:, 2]
            tree_imps = tree_map[:, 3]
            tree_cycles = tree_map[:, 4]
            for n_idx in G.nodes():
                # skip self nodes
                if n_idx == src_idx:
                    continue
                # get distance and impedance
                dist = tree_dists[n_idx]
                imp = tree_imps[n_idx]
                # continue if exceeds max
                if np.isinf(dist) or dist > dist_cutoff:
                    continue
                # aggregate values
                dens[src_idx] += 1
                far_imp[src_idx] += imp
                far_dist[src_idx] += dist
                harmonic_cl[src_idx] += 1 / imp
                grav[src_idx] += np.exp(beta * dist)
                # cycles
                if tree_cycles[n_idx]:
                    cyc[src_idx] += 1
                # BETWEENNESS
                # only process betweenness in one direction
                if n_idx < src_idx:
                    continue
                # betweenness - only counting truly between vertices, not starting and ending verts
                inter_idx = tree_preds[n_idx]
                # isolated nodes will have no predecessors
                if np.isnan(inter_idx):
                    continue
                inter_idx = np.int(inter_idx)
                while True:
                    # break out of while loop if the intermediary has reached the source node
                    if inter_idx == src_idx:
                        break
                    betw[inter_idx] += 1
                    betw_wt[inter_idx] += np.exp(beta * dist)
                    # follow
                    inter_idx = np.int(tree_preds[inter_idx])
        improved_cl = dens / far_dist / dens

        # check betweenness
        print(distances[d_idx])
        assert np.allclose(node_density[d_idx], dens)
        assert np.allclose(node_farness[d_idx], far_dist)
        assert np.allclose(node_cycles[d_idx], cyc)
        assert np.allclose(node_harmonic[d_idx], harmonic_cl)
        assert np.allclose(node_beta[d_idx], grav)
        assert np.allclose(improved_closness[d_idx], improved_cl, equal_nan=True)
        assert np.allclose(node_betweenness[d_idx], betw)
        assert np.allclose(node_betweenness_beta[d_idx], betw_wt)

        #TODO: how to test segment_density, harmonic_segment, segment_beta, segment_betweenness

    # check that problematic keys are caught
    for angular, k in zip([False, True], ['node_harmonic', 'node_harmonic_angular']):
        # catch typos
        with pytest.raises(ValueError):
            centrality.local_centrality(node_data,
                                        edge_data,
                                        node_edge_map,
                                        edge_ghost_map,
                                        distances,
                                        betas,
                                        ('typo_key',),
                                        angular=False)
        # catch duplicates
        with pytest.raises(ValueError):
            centrality.local_centrality(node_data,
                                        edge_data,
                                        node_edge_map,
                                        edge_ghost_map,
                                        distances,
                                        betas,
                                        (k, k),
                                        angular=False)
        # catch mixed angular and non-angular keys
        with pytest.raises(ValueError):
            centrality.local_centrality(node_data,
                                        edge_data,
                                        node_edge_map,
                                        edge_ghost_map,
                                        distances,
                                        betas,
                                        ('node_density', 'node_harmonic_angular'),
                                        angular=False)


def test_local_centrality_time():
    '''
    originally based on node_harmonic and node_betweenness:
    OLD VERSION with trim maps:
    Timing: 10.490865555 for 10000 iterations
    NEW VERSION with numba typed list - faster and removes arcane full vs. trim maps workflow
    8.242256040000001 for 10000 iterations
    VERSION with node_edge_map Dict - tad slower but worthwhile for cleaner and more intuitive code
    8.882408618 for 10000 iterations

    float64 - 17.881911942000002
    float32 - 13.612861239
    segments of unreachable code add to timing regardless...
    possibly because of high number of iters vs. function prep and teardown...?

    14.976446869
    '''
    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map, edge_ghost_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
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
        return centrality.local_centrality(node_data,
                                           edge_data,
                                           node_edge_map,
                                           edge_ghost_map,
                                           distances,
                                           betas,
                                           ('node_density',  # 7.16s
                                            'node_betweenness',  # 8.08s - adds around 1s
                                            'segment_density',  # 11.2s - adds around 3s
                                            'segment_betweenness'
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