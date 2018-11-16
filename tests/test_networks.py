import numpy as np
import networkx as nx
from shapely import geometry
from cityseer import networks, graphs, util


# TODO: add test_crow_flies - though implicitly tested?


def test_shortest_path_tree():

    # TODO: test shortest path on angular metric

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
    G, pos = util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

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


#TODO: network_centralities is tested via centrality.compute_centrality wrapper - whether to split out?
def test_network_centralities():
    '''
    Also tested indirectly via test_centrality.test_compute_centrality
    '''

    # load the test graph
    G, pos = util.tutte_graph()
    for s, e in G.edges():
        G[s][e]['geom'] = geometry.LineString([
            [G.nodes[s]['x'], G.nodes[s]['y']],
            [G.nodes[e]['x'], G.nodes[e]['y']]
        ])
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
        networks.network_centralities(n_map, e_map, np.array(distances), np.array(betas), closeness_map=np.array(closeness_map), betweenness_map=np.array(betweenness_map), angular_wt=False)

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
