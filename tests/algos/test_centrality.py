import networkx as nx
import numpy as np
import pytest
import timeit

from cityseer.algos import data, centrality, checks
from cityseer.metrics import networks
from cityseer.util import mock, graphs


def test_shortest_path_tree():
    # for extracting paths from predecessor map
    def find_path(start_idx, target_idx, map_pred):
        s_path = []
        pred = start_idx
        while True:
            s_path.append(pred)
            if pred == target_idx:
                break
            pred = map_pred[pred].astype(int)
        return list(reversed(s_path))

    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_nX(G)
    G_round_trip = graphs.nX_from_graph_maps(node_uids, node_map, edge_map)

    # for debugging
    # from cityseer.util import plot
    # plot.plot_nX_primal_or_dual(primal=G, labels=True)

    # test all shortest paths against networkX version of dijkstra
    for max_dist in [0, 500, 2000]:
        for src_idx in range(len(G)):
            # check shortest path maps
            map_pred, map_impedance, map_distance, cycles = centrality.shortest_path_tree(node_map,
                                                                                          edge_map,
                                                                                          src_idx,
                                                                                          max_dist=max_dist,
                                                                                          angular=False)
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G_round_trip, src_idx, weight='length', cutoff=max_dist)
            for j in range(len(G)):
                if j in nx_path:
                    assert find_path(j, src_idx, map_pred) == nx_path[j]
                    assert map_impedance[j] == map_distance[j] == nx_dist[j]

    # angular impedance should take a simpler but longer path - test basic case on dual
    G_dual = mock.mock_graph()
    G_dual = graphs.nX_simple_geoms(G_dual)
    G_dual = graphs.nX_to_dual(G_dual)

    node_uids_dual, node_map_dual, edge_map_dual = graphs.graph_maps_from_nX(G_dual)

    # source and target are the same for either
    src_idx = node_uids_dual.index('11_6')
    target = node_uids_dual.index('39_40')

    # SIMPLEST PATH: get simplest path tree using angular impedance
    map_pred, map_impedance, map_distance, cycles = centrality.shortest_path_tree(node_map_dual,
                                                                                  edge_map_dual,
                                                                                  src_idx,
                                                                                  max_dist=np.inf,
                                                                                  angular=True)
    # find path
    path = find_path(target, src_idx, map_pred)
    path_transpose = [node_uids_dual[n] for n in path]
    # takes 1597m route via long outside segment
    # map_distance[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '11_14', '10_14', '10_43', '43_44', '40_44', '39_40']

    # SHORTEST PATH:
    # get shortest path tree using non angular impedance
    map_pred, map_impedance, map_distance, cycles = centrality.shortest_path_tree(node_map_dual,
                                                                                  edge_map_dual,
                                                                                  src_idx,
                                                                                  max_dist=np.inf,
                                                                                  angular=False)
    # find path
    path = find_path(target, src_idx, map_pred)
    path_transpose = [node_uids_dual[n] for n in path]
    # takes 1345m shorter route
    # map_distance[int(full_to_trim_idx_map[node_labels.index('39_40')])]
    assert path_transpose == ['11_6', '6_7', '3_7', '3_4', '1_4', '0_1', '0_31', '31_32', '32_34', '34_37', '37_39',
                              '39_40']

    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src_idx = node_uids_dual.index('10_43')
    target = node_uids_dual.index('10_5')
    map_pred, map_impedance, map_distance, cycles = centrality.shortest_path_tree(node_map_dual,
                                                                                  edge_map_dual,
                                                                                  src_idx,
                                                                                  max_dist=np.inf,
                                                                                  angular=True)
    # find path
    path = find_path(target, src_idx, map_pred)
    path_transpose = [node_uids_dual[n] for n in path]
    assert path_transpose == ['10_43', '10_5']

    # WITH SIDESTEPS - set angular flag to False
    # manually overwrite distance impedances with angular for this test
    # (angular has to be false otherwise shortest-path sidestepping avoided)
    edge_map_dual_temp = edge_map_dual.copy()
    # angular impedances at index 3 copied to distance impedances at distance 2
    edge_map_dual_temp[:, 2] = edge_map_dual_temp[:, 3]
    map_pred, map_impedance, map_distance, cycles = centrality.shortest_path_tree(node_map_dual,
                                                                                  edge_map_dual_temp,
                                                                                  src_idx,
                                                                                  max_dist=np.inf,
                                                                                  angular=False)
    # find path
    path = find_path(target, src_idx, map_pred)
    path_transpose = [node_uids_dual[n] for n in path]
    assert path_transpose == ['10_43', '10_14', '10_5']


def test_decomposed_local_centrality():
    # centralities on the original nodes within the decomposed network should equal non-decomposed workflow
    betas = np.array([-0.02, -0.01, -0.005, -0.0008])
    distances = networks.distance_from_beta(betas)
    measure_keys = ('harmonic_node', 'betweenness_node')

    # test a decomposed graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    measures_data = centrality.local_centrality(node_map,
                                                edge_map,
                                                distances,
                                                betas,
                                                measure_keys,
                                                angular=False)

    G_decomposed = graphs.nX_decompose(G, 20)
    # from cityseer.util import plot
    # plot.plot_nX(G_decomposed, labels=True)
    node_uids, node_map, edge_map = graphs.graph_maps_from_nX(G_decomposed)  # generate node and edge maps
    measures_data_decomposed = centrality.local_centrality(node_map,
                                                           edge_map,
                                                           distances,
                                                           betas,
                                                           measure_keys,
                                                           angular=False)

    # test harmonic closeness on original nodes for non-decomposed vs decomposed
    d_range = len(distances)
    m_range = len(measure_keys)
    assert measures_data.shape == (m_range, d_range, len(G))
    assert measures_data_decomposed.shape == (m_range, d_range, len(G_decomposed))
    original_node_idx = np.where(node_map[:, 4] == 0)
    for m_idx in range(m_range):
        for d_idx in range(d_range):
            assert np.allclose(measures_data[m_idx][d_idx], measures_data_decomposed[m_idx][d_idx][original_node_idx])


def test_local_centrality_time():
    # OLD VERSION with trim maps:
    # Timing: 10.490865555 for 10000 iterations
    # NEW VERSION with numba typed list:
    # VS. 8.242256040000001 for 10000 iterations

    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    # needs a large enough beta so that distance thresholds aren't encountered
    distances = np.array([np.inf])
    betas = networks.beta_from_distance(distances)

    # setup timing wrapper
    def wrapper_func():
        return centrality.local_centrality(node_map,
                                           edge_map,
                                           distances,
                                           betas,
                                           ('harmonic_node', 'betweenness_node'),
                                           angular=False,
                                           suppress_progress=True)

    # prime
    wrapper_func()
    iters = 10000
    # report
    func_time = timeit.timeit(wrapper_func, number=iters)
    print(f'Timing: {func_time} for {iters} iterations')


def test_local_centrality():
    '''
    Also tested indirectly via test_networks.test_compute_centrality
    '''

    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    G_round_trip = graphs.nX_from_graph_maps(node_uids, node_map, edge_map)

    # Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    # Note that NetworkX improved closeness is not the same as derivation used in this package
    # Note that NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)

    # needs a large enough beta so that distance thresholds aren't encountered
    betas = np.array([-0.02, -0.01, -0.005, -0.0008])
    distances = networks.distance_from_beta(betas)

    # set the keys - add shuffling to be sure various orders work
    measure_keys = [
        'node_density',
        # 'segment_density',
        'farness',
        'cycles',
        'harmonic_node',
        # 'harmonic_segment',
        'beta_node',
        # 'beta_segment'
        'betweenness_node',
        # 'betweenness_node_wt',
        # 'betweenness_segment'
    ]
    np.random.shuffle(measure_keys)
    measure_keys = tuple(measure_keys)

    measure_keys_angular = [
        # 'harmonic_node_angle',
        # 'harmonic_segment_hybrid',
        # 'betweenness_node_angle',
        # 'betweeness_segment_hybrid'
    ]
    np.random.shuffle(measure_keys_angular)
    measure_keys_angular = tuple(measure_keys_angular)

    measures_data = centrality.local_centrality(node_map,
                                                edge_map,
                                                distances,
                                                betas,
                                                measure_keys,
                                                angular=False)

    node_density = measures_data[measure_keys.index('node_density')]
    harmonic_node = measures_data[measure_keys.index('harmonic_node')]

    betweenness = measures_data[measure_keys.index('betweenness_node')]

    # test node density
    # node density count doesn't include self-node
    # connected component == 48 == len(G) - 4
    # isolated looping component == 3
    # isolated edge == 1
    # isolated node == 0
    for n in node_density[3]:  # only largest distance - which exceeds cutoff clashes
        assert n in [48, 3, 1, 0]

    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G_round_trip, distance='length')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.allclose(nx_harm_cl, harmonic_node[3])

    # test betweenness vs NetworkX
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G_round_trip, weight='length', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    # for some reason nx betweenness gives 0.5 instead of 1 for disconnected looping component (should be 1)
    # maybe two equidistant routes being divided through 2
    # nx betweenness gives 0.5 instead of 1 for all disconnected looping component nodes
    # nx presumably takes equidistant routes into account, in which case only the fraction is aggregated
    assert np.array_equal(nx_betw[:52], betweenness[3][:52])


def test_temp():
    ###
    # test manual metrics against all nodes
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
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
        improved_cl = np.full(G.number_of_nodes(), 0.0)

        for src_idx in range(len(G)):

            # generate trim and full index maps
            trim_to_full_idx_map, full_to_trim_idx_map = data.radial_filter(x_arr[src_idx],
                                                                            y_arr[src_idx],
                                                                            x_arr,
                                                                            y_arr,
                                                                            dist_cutoff)

            # get shortest path maps
            map_impedance_trim, map_distance_trim, map_pred_trim, cycles_trim = \
                centrality.shortest_path_tree(node_map,
                                              edge_map,
                                              src_idx,
                                              trim_to_full_idx_map,
                                              full_to_trim_idx_map,
                                              max_dist=dist_cutoff,
                                              angular=False)

            for n_idx in G.nodes():
                # skip self nodes
                if n_idx == src_idx:
                    continue
                # get trim idx
                trim_idx = full_to_trim_idx_map[n_idx]
                if np.isnan(trim_idx):
                    continue
                trim_idx = int(trim_idx)
                # get distance and impedance
                dist = map_distance_trim[trim_idx]
                imp = map_impedance_trim[trim_idx]
                # continue if exceeds max
                if dist > dist_cutoff:
                    continue

                # aggregate values
                dens[src_idx] += 1
                far_imp[src_idx] += imp
                far_dist[src_idx] += dist
                harmonic_cl[src_idx] += 1 / imp
                grav[src_idx] += np.exp(beta * dist)
                # cycles
                if cycles_trim[trim_idx]:
                    cyc[src_idx] += np.exp(beta * dist)

                # BETWEENNESS
                # only process betweenness in one direction
                if n_idx < src_idx:
                    continue

                # betweenness - only counting truly between vertices, not starting and ending verts
                inter_idx_trim = np.int(map_pred_trim[trim_idx])
                inter_idx_full = np.int(trim_to_full_idx_map[inter_idx_trim])

                while True:
                    # break out of while loop if the intermediary has reached the source node
                    if inter_idx_trim == full_to_trim_idx_map[src_idx]:
                        break

                    betw[inter_idx_full] += 1
                    betw_wt[inter_idx_full] += np.exp(beta * dist)

                    # follow
                    inter_idx_trim = np.int(map_pred_trim[inter_idx_trim])
                    inter_idx_full = np.int(trim_to_full_idx_map[inter_idx_trim])

                '''
                # for debugging:
                if src_idx == 6 and dist_cutoff == 400:
    
                    # for debugging
                    from cityseer.util import plot
                    from shapely import geometry
                    geom = geometry.Point(x_arr[src_idx], y_arr[src_idx]).buffer(dist_cutoff)
                    # override live designation to distinguish reachable nodes...
                    temp_node_map = node_map.copy()
                    temp_node_map[:, 2] = 0
                    for live_idx in trim_to_full_idx_map[np.isfinite(map_distance_trim)]:
                        temp_node_map[:, 2][int(live_idx)] = 1
                    plot.plot_graph_maps(node_uids, temp_node_map, edge_map, poly=geom)
                '''

        # check betweenness
        assert np.allclose(node_density[d_idx], dens)
        assert np.allclose(far_impedance[d_idx], far_imp)
        assert np.allclose(far_distance[d_idx], far_dist)
        assert np.allclose(harmonic[d_idx], harmonic_cl)
        assert np.allclose(improved[d_idx], improved_cl)
        assert np.allclose(gravity_index[d_idx], grav)
        assert np.allclose(cycles[d_idx], cyc)
        assert np.allclose(betweenness[d_idx], betw)
        assert np.allclose(betweenness_decay[d_idx], betw_wt)

    # check behaviour of weights
    node_map_w = node_map.copy()
    node_map_w[:, 4] = 2
    # unshuffle the keys
    closeness_keys.sort()
    betweenness_keys.sort()
    # compute
    closeness_data_w, betweenness_data_w = \
        centrality.local_centrality(node_map_w,
                                    edge_map,
                                    distances,
                                    betas,
                                    closeness_keys,
                                    betweenness_keys,
                                    angular=False)
    # unpack
    node_density_w, far_impedance_w, far_distance_w, harmonic_w, improved_w, gravity_index_w, cycles_w = \
        closeness_data_w[closeness_keys]
    betweenness_w, betweenness_decay_w = betweenness_data_w[betweenness_keys]

    # closenesss
    assert np.allclose(node_density, node_density_w / 2)  # should double
    assert np.allclose(far_impedance, far_impedance_w * 2)  # should half
    assert np.allclose(far_distance, far_distance_w)  # should be no change
    assert np.allclose(harmonic, harmonic_w / 2)  # should double
    assert np.allclose(improved, improved_w / 4)  # should quadruple due to square of weighted node density
    assert np.allclose(gravity_index, gravity_index_w / 2)  # should double
    assert np.allclose(cycles, cycles_w)  # should be no change
    # betweenness
    assert np.allclose(betweenness, betweenness_w / 2)  # should double
    assert np.allclose(betweenness_decay, betweenness_decay_w / 2)  # should double

    # check that angular is passed-through
    # actual angular tests happen in test_shortest_path_tree()
    # here the emphasis is simply on checking that the angular instruction gets chained through

    # setup dual data
    G_dual = graphs.nX_to_dual(G)
    node_labels_dual, node_map_dual, edge_map_dual = graphs.graph_maps_from_nX(G_dual)

    cl_dual, bt_dual = \
        centrality.local_centrality(node_map_dual,
                                    edge_map_dual,
                                    distances,
                                    betas,
                                    closeness_keys,
                                    betweenness_keys,
                                    angular=True)

    cl_dual_sidestep, bt_dual_sidestep = \
        centrality.local_centrality(node_map_dual,
                                    edge_map_dual,
                                    distances,
                                    betas,
                                    closeness_keys,
                                    betweenness_keys,
                                    angular=False)

    assert not np.allclose(cl_dual, cl_dual_sidestep)
    assert not np.allclose(bt_dual, bt_dual_sidestep)

    # check that problematic keys are caught
    for cl_key, bt_key in [([], []),  # missing
                           ([-1], [1]),  # negative
                           ([1], [-1]),
                           ([7], [1]),  # out of range
                           ([1], [2]),
                           ([1, 1], [1]),  # duplicate
                           ([1], [1, 1])]:
        with pytest.raises(ValueError):
            centrality.local_centrality(node_map,
                                        edge_map,
                                        distances,
                                        betas,
                                        np.array(cl_key),
                                        np.array(bt_key),
                                        angular=False)
