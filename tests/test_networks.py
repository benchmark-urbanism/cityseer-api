import numpy as np
import networkx as nx
from cityseer import networks, graphs, util
import matplotlib.pyplot as plt


def test_crow_flies():

    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)

    max_dist = 200
    x_arr = n_map[:,0]
    y_arr = n_map[:,1]

    # generate trim and full index maps
    trim_to_full_idx_map, full_to_trim_idx_map = networks.crow_flies(0, max_dist, x_arr, y_arr)

    #nx.draw(G, pos, with_labels=True)
    #plt.show()

    # manually confirmed for 200m max distance:
    assert np.array_equal(trim_to_full_idx_map, np.array([0, 1, 16, 31]))
    # check that the full to trim is still the same length
    assert len(full_to_trim_idx_map) == G.number_of_nodes()
    # check that all non NaN full_to_trim_idx_map indices are reflected in the either direction
    for idx, n in enumerate(full_to_trim_idx_map):
        if not np.isnan(n):
            assert trim_to_full_idx_map[int(n)] == idx

    # check that aggregate distances are less than max
    map_impedance, map_distance, map_pred, cycles = \
        networks.shortest_path_tree(n_map, e_map, 0, trim_to_full_idx_map, full_to_trim_idx_map, max_dist=max_dist)
    for n in full_to_trim_idx_map:
        if not np.isnan(n):
            assert map_distance[int(n)] < max_dist


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
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)

    # test all shortest paths against networkX version of dijkstra
    for max_dist in [200, 500, 2000]:
        for src in range(len(G)):
            # generate trim and full index maps
            x_arr = n_map[:, 0]
            y_arr = n_map[:, 1]
            trim_to_full_idx_map, full_to_trim_idx_map = networks.crow_flies(src, max_dist, x_arr, y_arr)
            # check shortest path maps
            map_impedance, map_distance, map_pred, cycles = networks.shortest_path_tree(n_map, e_map, src,
                                        trim_to_full_idx_map, full_to_trim_idx_map, max_dist=max_dist, angular=False)
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G, src, weight='impedance', cutoff=max_dist)
            for j in range(len(G)):
                if j in nx_path:
                    assert find_path(map_pred, j, trim_to_full_idx_map, full_to_trim_idx_map) == nx_path[j]
                    j_trim = int(full_to_trim_idx_map[j])
                    assert map_impedance[j_trim] == map_distance[j_trim] == nx_dist[j]

    # test that special cases using angular metric will take longer distance routes for shorter angular impedance
    # ordinary distance impedance
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)
    # generate trim and full index maps
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    src = 6
    trim_to_full_idx_map, full_to_trim_idx_map = networks.crow_flies(src, np.inf, x_arr, y_arr)
    # get shortest path tree
    map_impedance, map_distance, map_pred, cycles = networks.shortest_path_tree(n_map, e_map, src,
                                        trim_to_full_idx_map, full_to_trim_idx_map, max_dist=np.inf, angular=False)
    # find path
    path_dist = find_path(map_pred, 40, trim_to_full_idx_map, full_to_trim_idx_map)
    # takes 1310m route through middle of network
    # map_distance[int(full_to_trim_idx_map[40])]
    assert path_dist == [6, 7, 3, 4, 1, 0, 31, 32, 35, 36, 41, 40]

    # angular impedance
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G_dual = graphs.networkX_to_dual(G)
    n_labels_ang, n_map_ang, e_map_ang = graphs.graph_maps_from_networkX(G_dual)
    # generate trim and full index maps
    x_arr_ang = n_map_ang[:, 0]
    y_arr_ang = n_map_ang[:, 1]
    src_ang = n_labels_ang.index('6_11')
    trim_to_full_idx_map_ang, full_to_trim_idx_map_ang = networks.crow_flies(src_ang, np.inf, x_arr_ang, y_arr_ang)
    # get shortest path tree
    map_impedance_ang, map_distance_ang, map_pred_ang, cycles_ang = networks.shortest_path_tree(n_map_ang, e_map_ang,
                            src_ang, trim_to_full_idx_map_ang, full_to_trim_idx_map_ang, max_dist=np.inf, angular=True)
    # find path
    target = n_labels_ang.index('39_40')
    path_ang = find_path(map_pred_ang, target, trim_to_full_idx_map_ang, full_to_trim_idx_map_ang)
    path_transpose = [n_labels_ang[n] for n in path_ang]
    # takes 1597m route via long outside segment
    # map_distance_ang[int(full_to_trim_idx_map_ang[n_labels_ang.index('39_40')])]
    assert path_transpose == ['6_11', '11_14', '10_14', '10_43', '43_44', '40_44', '39_40']

    print(path_dist)
    print(path_transpose)


def test_network_centralities():
    '''
    Also tested indirectly via test_centrality.test_compute_centrality
    '''

    # load the test graph
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    # test centrality methods
    # networkx doesn't have a maximum distance cutoff, so have to run on the whole graph

    dist = 2000
    min_threshold_wt = 0.01831563888873418
    distances = [dist]
    betas = [np.log(min_threshold_wt) / dist]
    closeness_map = [0, 3]
    betweenness_map = [0]
    closeness_data, betweenness_data = \
        networks.network_centralities(n_map, e_map, np.array(distances), np.array(betas),
                                      closeness_map=np.array(closeness_map), betweenness_map=np.array(betweenness_map), angular=False)

    node_density, harmonic = closeness_data[closeness_map]
    betweenness = betweenness_data[betweenness_map]

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
