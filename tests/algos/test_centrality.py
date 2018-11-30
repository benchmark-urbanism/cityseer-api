import networkx as nx
import numpy as np
import pytest

from cityseer.algos import data, centrality
from cityseer.util import mock, graphs


def test_shortest_path_tree():
    # for extracting paths from predecessor map
    def find_path(map_pred, target, trim_to_full_idx_map, full_to_trim_idx_map):
        s_path = []
        pred = full_to_trim_idx_map[target]  # trim indices
        while True:
            s_path.append(int(trim_to_full_idx_map[int(pred)]))  # full indices
            pred = map_pred[int(pred)]
            if np.isnan(pred):
                break
        return list(reversed(s_path))

    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)

    # test all shortest paths against networkX version of dijkstra
    for max_dist in [200, 500, 2000]:
        for src in range(len(G)):
            # generate trim and full index maps
            x_arr = n_map[:, 0]
            y_arr = n_map[:, 1]
            src = 0
            index_map = data.generate_index(x_arr, y_arr)
            trim_to_full_idx_map, full_to_trim_idx_map = data.distance_filter(index_map, x_arr[src], y_arr[src],
                                                                              max_dist, radial=True)
            # check shortest path maps
            map_impedance, map_distance, map_pred, cycles = centrality.shortest_path_tree(n_map,
                                                                                          e_map,
                                                                                          src,
                                                                                          trim_to_full_idx_map,
                                                                                          full_to_trim_idx_map,
                                                                                          max_dist=max_dist,
                                                                                          angular=False)
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G, src, weight='impedance', cutoff=max_dist)
            for j in range(len(G)):
                if j in nx_path:
                    assert find_path(map_pred, j, trim_to_full_idx_map, full_to_trim_idx_map) == nx_path[j]
                    j_trim = int(full_to_trim_idx_map[j])
                    assert map_impedance[j_trim] == map_distance[j_trim] == nx_dist[j]

    # angular impedance should take a simpler but longer path
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G_dual = graphs.networkX_to_dual(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G_dual)
    # source and target are the same for either
    src = n_labels.index('6_11')
    target = n_labels.index('39_40')
    # generate trim and full index maps
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    index_map = data.generate_index(x_arr, y_arr)
    trim_to_full_idx_map, full_to_trim_idx_map = data.distance_filter(index_map, x_arr[src], y_arr[src], np.inf,
                                                                      radial=True)

    # SIMPLEST PATH: get simplest path tree using angular impedance
    map_impedance_a, map_distance_a, map_pred_a, cycles_a = centrality.shortest_path_tree(n_map,
                                                                                          e_map,
                                                                                          src,
                                                                                          trim_to_full_idx_map,
                                                                                          full_to_trim_idx_map,
                                                                                          max_dist=np.inf,
                                                                                          angular=True)
    # find path
    path_a = find_path(map_pred_a, target, trim_to_full_idx_map, full_to_trim_idx_map)
    path_transpose_a = [n_labels[n] for n in path_a]
    # takes 1597m route via long outside segment
    # map_distance_a[int(full_to_trim_idx_map[n_labels.index('39_40')])]
    assert path_transpose_a == ['6_11', '11_14', '10_14', '10_43', '43_44', '40_44', '39_40']

    # SHORTEST PATH: override angular impedances with true distances
    e_map_m = e_map.copy()
    e_map_m[:, 3] = e_map_m[:, 2]
    # get shortest path tree using distance impedance
    map_impedance_m, map_distance_m, map_pred_m, cycles_m = centrality.shortest_path_tree(n_map,
                                                                                          e_map_m,
                                                                                          src,
                                                                                          trim_to_full_idx_map,
                                                                                          full_to_trim_idx_map,
                                                                                          max_dist=np.inf,
                                                                                          angular=True)
    # find path
    path_m = find_path(map_pred_m, target, trim_to_full_idx_map, full_to_trim_idx_map)
    path_transpose_m = [n_labels[n] for n in path_m]
    # takes 1345m route
    # map_distance_m[int(full_to_trim_idx_map[n_labels.index('39_40')])]
    assert path_transpose_m == ['6_11', '6_7', '3_7', '3_4', '1_4', '0_1', '0_31', '31_32', '32_34', '34_37', '37_39',
                                '39_40']

    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src = n_labels.index('10_43')
    target = n_labels.index('5_10')
    map_impedance_ns, map_distance_ns, map_pred_ns, cycles_ns = centrality.shortest_path_tree(n_map,
                                                                                              e_map,
                                                                                              src,
                                                                                              trim_to_full_idx_map,
                                                                                              full_to_trim_idx_map,
                                                                                              max_dist=np.inf,
                                                                                              angular=True)
    # find path
    path_ns = find_path(map_pred_ns, target, trim_to_full_idx_map, full_to_trim_idx_map)
    path_transpose_ns = [n_labels[n] for n in path_ns]
    assert path_transpose_ns == ['10_43', '5_10']

    # WITH SIDESTEPS - set angular flag to False
    map_impedance_s, map_distance_s, map_pred_s, cycles_s = centrality.shortest_path_tree(n_map,
                                                                                          e_map,
                                                                                          src,
                                                                                          trim_to_full_idx_map,
                                                                                          full_to_trim_idx_map,
                                                                                          max_dist=np.inf,
                                                                                          angular=False)
    # find path
    path_s = find_path(map_pred_s, target, trim_to_full_idx_map, full_to_trim_idx_map)
    path_transpose_s = [n_labels[n] for n in path_s]
    assert path_transpose_s == ['10_43', '10_14', '5_10']

    # check that out of range src index raises error
    with pytest.raises(ValueError):
        centrality.shortest_path_tree(n_map,
                                      e_map,
                                      len(n_map),
                                      trim_to_full_idx_map,
                                      full_to_trim_idx_map,
                                      max_dist=np.inf,
                                      angular=False)
    # test that mismatching full_to_trim length raises error
    with pytest.raises(ValueError):
        centrality.shortest_path_tree(n_map,
                                      e_map,
                                      0,
                                      trim_to_full_idx_map,
                                      full_to_trim_idx_map[:-1],
                                      max_dist=np.inf,
                                      angular=False)


def test_network_centralities():
    '''
    Also tested indirectly via test_centrality.test_compute_centrality
    '''

    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    # test centrality methods
    # networkx doesn't have a maximum distance cutoff, so have to run on the whole graph

    dist = 2000
    min_threshold_wt = 0.01831563888873418
    distances = np.array([dist])
    betas = np.array([np.log(min_threshold_wt) / dist])
    closeness_keys = np.array([0, 3])
    betweenness_keys = np.array([0])
    print(betas)
    # compute closeness and betweenness
    closeness_data, betweenness_data = \
        centrality.local_centrality(n_map,
                                    e_map,
                                    distances,
                                    betas,
                                    closeness_keys,
                                    betweenness_keys,
                                    angular=False)

    node_density, harmonic = closeness_data[closeness_keys]
    betweenness = betweenness_data[betweenness_keys]

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
    assert np.array_equal(nx_betw, betweenness[0][0])
