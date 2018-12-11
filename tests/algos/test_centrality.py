import networkx as nx
import numpy as np
import pytest

from cityseer.algos import data, centrality
from cityseer.metrics import networks
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
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    for max_dist in [200, 500, 2000]:
        for src in range(len(G)):
            # generate trim and full index maps
            trim_to_full_idx_map, full_to_trim_idx_map = data.radial_filter(x_arr[src],
                                                                            y_arr[src],
                                                                            x_arr,
                                                                            y_arr,
                                                                            max_dist)
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

    # for debugging
    # from cityseer.util import plot
    # plot.plot_networkX_primal_or_dual(primal=G, dual=G_dual)

    # source and target are the same for either
    src = n_labels.index('6_11')
    target = n_labels.index('39_40')
    # generate trim and full index maps
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    trim_to_full_idx_map, full_to_trim_idx_map = data.radial_filter(x_arr[src],
                                                                    y_arr[src],
                                                                    x_arr,
                                                                    y_arr,
                                                                    np.inf)

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
                                                                                          angular=False)
    # find path
    path_m = find_path(map_pred_m, target, trim_to_full_idx_map, full_to_trim_idx_map)
    path_transpose_m = [n_labels[n] for n in path_m]
    # takes 1345m route shorter route
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
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    # Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    # Note that NetworkX improved closeness is not the same as derivation used in this package
    # Note that NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)

    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances = networks.distance_from_beta(betas)

    closeness_keys = np.array([0, 1, 2, 3, 4, 5, 6])
    betweenness_keys = np.array([0, 1])

    # compute closeness and betweenness
    closeness_data, betweenness_data = \
        centrality.local_centrality(node_map,
                                    edge_map,
                                    distances,  # use only last (biggest) distance to avoid cutoffs
                                    betas,
                                    closeness_keys,
                                    betweenness_keys,
                                    angular=False)

    node_density, far_impedance, far_distance, harmonic, improved, gravity, cycles = closeness_data[closeness_keys]
    betweenness, betweenness_gravity = betweenness_data[betweenness_keys]

    # test node density
    # node density count doesn't include self-node
    # else == 48 == len(G) - 4
    # isolated edge == 1
    # isolated node == 0
    for n in node_density[3]:  # only largest distance - which exceeds cutoff clashes
        assert n in [len(G) - 4, 1, 0]

    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.allclose(nx_harm_cl, harmonic[3])

    # test betweenness vs NetworkX
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    assert np.array_equal(nx_betw, betweenness[3])

    # test a variety of distances and metrics for all nodes against manual versions
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    for d_idx in range(len(distances)):
        dist_cutoff = distances[d_idx]
        beta = betas[d_idx]

        betw = np.full(G.number_of_nodes(), 0.0)
        betw_wt = np.full(G.number_of_nodes(), 0.0)

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

            dens = 0
            far_imp = 0
            far_dist = 0
            harmonic_cl = 0
            grav = 0
            cyc = 0

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
                dens += 1
                far_imp += imp
                far_dist += dist
                harmonic_cl += 1 / imp
                grav += np.exp(beta * dist)
                # cycles
                if cycles_trim[trim_idx]:
                    cyc += np.exp(beta * dist)

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

            if far_dist == 0:
                improved_cl = 0
            else:
                improved_cl = dens ** 2 / far_dist

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

            # check closeness
            assert node_density[d_idx][src_idx] == dens
            assert far_impedance[d_idx][src_idx] == far_imp
            assert far_distance[d_idx][src_idx] == far_dist
            assert harmonic[d_idx][src_idx] == harmonic_cl
            assert improved[d_idx][src_idx] == improved_cl
            assert gravity[d_idx][src_idx] == grav
            assert cycles[d_idx][src_idx] == cyc

        # check betweenness
        assert np.array_equal(betweenness[d_idx], betw)
        assert np.array_equal(betweenness_gravity[d_idx], betw_wt)

    # check that problematic keys are caught
    for cl_key, bt_key in [(np.array([]), np.array([])),  # missing
                           (np.array([-1]), np.array([1])),  # negative
                           (np.array([1]), np.array([-1])),
                           (np.array([7]), np.array([1])),  # out of range
                           (np.array([1]), np.array([2])),
                           (np.array([1, 1]), np.array([1])),  # duplicate
                           (np.array([1]), np.array([1, 1]))]:
        with pytest.raises(ValueError):
            centrality.local_centrality(node_map, edge_map, distances, betas, cl_key, bt_key, angular=False)
