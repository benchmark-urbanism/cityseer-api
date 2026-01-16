# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
from cityseer import config, rustalgos
from cityseer.tools import graphs, io


def find_path(start_idx, target_idx, tree_map):
    """
    for extracting paths from predecessor map
    """
    s_path: list[int] = []
    pred_idx: int = start_idx
    while True:
        s_path.append(pred_idx)
        if pred_idx == target_idx:
            break
        pred_idx = tree_map[pred_idx].pred

    return list(reversed(s_path))


def test_shortest_path_trees(primal_graph, dual_graph):
    nodes_gdf_p, edges_gdf_p, network_structure_p = io.network_structure_from_nx(primal_graph)
    # prepare round-trip graph for checks
    G_round_trip = io.nx_from_cityseer_geopandas(nodes_gdf_p, edges_gdf_p)
    for start_nd_key, end_nd_key, edge_idx in G_round_trip.edges(keys=True):
        geom = G_round_trip[start_nd_key][end_nd_key][edge_idx]["geom"]
        G_round_trip[start_nd_key][end_nd_key][edge_idx]["length"] = geom.length
    # from cityseer.tools import plot
    # plot.plot_nx_primal_or_dual(primal_graph=primal_graph, dual_graph=dual_graph, labels=True, primal_node_size=80)
    # test all shortest path routes against networkX version of dijkstra
    for max_dist in [0, 500, 2000, 5000]:
        max_seconds = max_dist / config.SPEED_M_S
        for src_idx in range(len(primal_graph)):
            # check shortest path maps
            _visited_nodes, tree_map = network_structure_p.dijkstra_tree_shortest(
                src_idx,
                int(max_seconds),
                config.SPEED_M_S,
            )
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G_round_trip, str(src_idx), weight="length", cutoff=max_dist)
            for j_node_key, j_nx_path in nx_path.items():
                assert find_path(int(j_node_key), src_idx, tree_map) == [int(j) for j in j_nx_path]
                assert tree_map[int(j_node_key)].short_dist - nx_dist[j_node_key] < config.ATOL
    # check with jitter
    nodes_gdf_j, edges_gdf_j, network_structure_j = io.network_structure_from_nx(primal_graph)
    for max_dist in [2000]:
        max_seconds = max_dist / config.SPEED_M_S
        # use aggressive jitter and check that at least one shortest path is different to non-jitter
        for jitter in [2]:
            diffs = False
            for src_idx in range(len(primal_graph)):
                # don't calculate for isolated nodes
                if src_idx >= 49:
                    continue
                # no jitter
                _visited_nodes, tree_map = network_structure_p.dijkstra_tree_shortest(
                    src_idx,
                    int(max_seconds),
                    config.SPEED_M_S,
                )
                # with jitter
                _visited_nodes_j, tree_map_j = network_structure_j.dijkstra_tree_shortest(
                    src_idx, int(max_seconds), config.SPEED_M_S, jitter_scale=jitter
                )
                for to_idx in range(len(primal_graph)):
                    if to_idx >= 49:
                        continue
                    if find_path(int(to_idx), src_idx, tree_map) != find_path(int(to_idx), src_idx, tree_map_j):
                        diffs = True
                        break
                if diffs is True:
                    break
            assert diffs is True
    # test all shortest distance calculations against networkX
    max_seconds_5000 = 5000 / config.SPEED_M_S
    for src_idx in range(len(G_round_trip)):
        shortest_dists = nx.shortest_path_length(G_round_trip, str(src_idx), weight="length")
        _visted_nodes, tree_map = network_structure_p.dijkstra_tree_shortest(
            src_idx, int(max_seconds_5000), config.SPEED_M_S, jitter_scale=0.0
        )
        for target_idx in range(len(G_round_trip)):
            if str(target_idx) not in shortest_dists:
                continue
            assert shortest_dists[str(target_idx)] - tree_map[target_idx].short_dist <= config.ATOL
    # prepare dual graph
    nodes_gdf_d, edges_gdf_d, network_structure_d = io.network_structure_from_nx(dual_graph)
    assert len(nodes_gdf_d) > len(nodes_gdf_p)
    # compare angular simplest paths for a selection of targets on primal vs. dual
    # remember, this is angular change not distance travelled
    # can be compared from primal to dual in this instance because edge segments are straight
    # i.e. same amount of angular change whether primal or dual graph
    # from cityseer.tools import plot
    # plot.plot_nx_primal_or_dual(primal_graph, dual_graph, labels=True, primal_node_size=80, dpi=300)
    p_source_idx = nodes_gdf_p.index.tolist().index("0")
    primal_targets = ("15", "20", "37")
    dual_sources = ("0_1_k0", "0_16_k0", "0_31_k0")
    dual_targets = ("13_15_k0", "17_20_k0", "36_37_k0")
    for p_target, d_source, d_target in zip(primal_targets, dual_sources, dual_targets, strict=True):
        p_target_idx = nodes_gdf_p.index.tolist().index(p_target)
        d_source_idx = nodes_gdf_d.index.tolist().index(d_source)  # dual source index changes depending on direction
        d_target_idx = nodes_gdf_d.index.tolist().index(d_target)
        _visited_nodes_p, tree_map_p = network_structure_p.dijkstra_tree_simplest(
            p_source_idx,
            int(max_seconds_5000),
            config.SPEED_M_S,
        )
        _visited_nodes_d, tree_map_d = network_structure_d.dijkstra_tree_simplest(
            d_source_idx,
            int(max_seconds_5000),
            config.SPEED_M_S,
        )
        assert tree_map_p[p_target_idx].simpl_dist - tree_map_d[d_target_idx].simpl_dist < config.ATOL
    # angular impedance should take a simpler but longer path - test basic case on dual
    # source and target are the same for either
    src_idx = nodes_gdf_d.index.tolist().index("11_6_k0")
    target = nodes_gdf_d.index.tolist().index("39_40_k0")
    # SIMPLEST PATH: get simplest path tree using angular impedance
    _visited_nodes_d2, tree_map_d2 = network_structure_d.dijkstra_tree_simplest(
        src_idx,
        int(max_seconds_5000),
        config.SPEED_M_S,
    )
    # find path
    path = find_path(target, src_idx, tree_map_d2)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # takes 1597m route via long outside segment
    # this should follow the simplest path periphery route instead of cutting through the shortest route central node
    # tree_dists[int(full_to_trim_idx_map[node_keys.index('39_40')])]
    assert path_transpose == [
        "11_6_k0",
        "11_14_k0",
        "10_14_k0",
        "10_43_k0",
        "43_44_k0",
        "40_44_k0",
        "39_40_k0",
    ]
    # SHORTEST PATH:
    # get shortest path tree using non angular impedance
    # this should cut through central node
    # would otherwise have used outside periphery route if using simplest path
    _visited_nodes_d3, tree_map_d3 = network_structure_d.dijkstra_tree_shortest(
        src_idx,
        int(max_seconds_5000),
        config.SPEED_M_S,
    )
    # find path
    path = find_path(target, src_idx, tree_map_d3)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # takes 1345m shorter route
    # tree_dists[int(full_to_trim_idx_map[node_keys.index('39_40')])]
    assert path_transpose == [
        "11_6_k0",
        "6_7_k0",
        "3_7_k0",
        "3_4_k0",
        "1_4_k0",
        "0_1_k0",
        "0_31_k0",
        "31_32_k0",
        "32_34_k0",
        "34_37_k0",
        "37_39_k0",
        "39_40_k0",
    ]
    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src_idx = nodes_gdf_d.index.tolist().index("10_43_k0")
    target = nodes_gdf_d.index.tolist().index("10_5_k0")
    _visited_nodes_d4, tree_map_d4 = network_structure_d.dijkstra_tree_simplest(
        src_idx,
        int(max_seconds_5000),
        config.SPEED_M_S,
    )
    # find path
    path = find_path(target, src_idx, tree_map_d4)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # print(path_transpose)
    assert path_transpose == ["10_43_k0", "10_5_k0"]


def test_local_node_centrality_shortest(primal_graph):
    """
    Also tested indirectly via test_networks.test_compute_centrality

    Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    Note that NetworkX improved closeness is not the same as derivation used in this package
    NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)
    """
    # generate node and edge maps
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    G_round_trip = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
    for start_nd_key, end_nd_key, edge_idx in G_round_trip.edges(keys=True):
        geom = G_round_trip[start_nd_key][end_nd_key][edge_idx]["geom"]
        G_round_trip[start_nd_key][end_nd_key][edge_idx]["length"] = geom.length
    # needs a large enough beta so that distance thresholds aren't encountered
    betas = [0.02, 0.01, 0.005, 0.0008]
    distances = rustalgos.distances_from_betas(betas)
    # generate the measures
    node_result_short = network_structure.local_node_centrality_shortest(
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    # test node density
    # node density count doesn't include self-node
    # connected component == 49 == len(G) - 1
    # isolated looping component == 3
    # isolated edge == 1
    # isolated node == 0
    for n in node_result_short.node_density[5000]:  # large distance - exceeds cutoff clashes
        assert n in [49, 3, 1, 0]
    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G_round_trip, distance="length")
    for src_idx in range(len(G_round_trip)):
        assert nx_harm_cl[str(src_idx)] - node_result_short.node_harmonic[5000][src_idx] < config.ATOL
    # test betweenness vs NetworkX
    # set endpoint counting to false and do not normalise
    # nx node centrality NOT implemented for MultiGraph
    G_non_multi = nx.Graph()  # don't change to MultiGraph!!!
    G_non_multi.add_nodes_from(G_round_trip.nodes())
    for s, e, k, d in G_round_trip.edges(keys=True, data=True):
        assert k == 0
        G_non_multi.add_edge(s, e, **d)
    nx_betw = nx.betweenness_centrality(G_non_multi, weight="length", endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    # nx betweenness gives 0.5 instead of 1 for all disconnected looping component nodes
    # nx presumably takes equidistant routes into account, in which case only the fraction is aggregated
    np.allclose(nx_betw[:52], node_result_short.node_betweenness[5000][:52], atol=config.ATOL, rtol=config.RTOL)
    # do the comparisons array-wise so that betweenness can be aggregated
    d_n = len(distances)
    n_nodes: int = primal_graph.number_of_nodes()
    betw: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    betw_wt: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    dens: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    far_short_dist: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    harmonic_cl: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    grav: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    cyc_to: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    cyc_src: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    #
    max_seconds_5000 = 5000 / config.SPEED_M_S
    for src_idx in range(n_nodes):
        # get shortest path maps
        visited_nodes, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, int(max_seconds_5000), speed_m_s=config.SPEED_M_S
        )
        for to_idx in visited_nodes:
            # skip self nodes
            if to_idx == src_idx:
                continue
            # get shortest / simplest distances
            to_short_dist = tree_map[to_idx].short_dist
            n_cycles = tree_map[to_idx].cycles
            # continue if exceeds max
            if np.isinf(to_short_dist):
                continue
            for d_idx, _ in enumerate(distances):
                dist_cutoff = distances[d_idx]
                beta = betas[d_idx]
                if to_short_dist <= dist_cutoff:
                    # don't exceed threshold
                    # if to_dist <= dist_cutoff:
                    # aggregate values
                    dens[d_idx][src_idx] += 1
                    far_short_dist[d_idx][src_idx] += to_short_dist
                    harmonic_cl[d_idx][src_idx] += 1 / to_short_dist
                    grav[d_idx][src_idx] += np.exp(-beta * to_short_dist)
                    # cycles - compute both source and target aggregation
                    # Rust uses target aggregation; source aggregation verifies totals match
                    cyc_to[d_idx][to_idx] += n_cycles
                    cyc_src[d_idx][src_idx] += n_cycles
                    # only process betweenness in one direction
                    if to_idx < src_idx:
                        continue
                    # betweenness - only counting truly between vertices, not starting and ending verts
                    inter_idx = tree_map[to_idx].pred
                    # isolated nodes will have no predecessors
                    if np.isnan(inter_idx):
                        continue
                    inter_idx = int(inter_idx)
                    while True:
                        # break out of while loop if the intermediary has reached the source node
                        if inter_idx == src_idx:
                            break
                        betw[d_idx][inter_idx] += 1
                        betw_wt[d_idx][inter_idx] += np.exp(-beta * to_short_dist)
                        # follow
                        inter_idx = int(tree_map[inter_idx].pred)
    for d_idx, dist in enumerate(distances):
        assert np.allclose(node_result_short.node_density[dist], dens[d_idx], atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(
            node_result_short.node_farness[dist], far_short_dist[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
        # Cycles: Rust uses target aggregation (cyc_to), check exact match
        assert np.allclose(node_result_short.node_cycles[dist], cyc_to[d_idx], atol=config.ATOL, rtol=config.RTOL)
        # Source aggregation (cyc_src) has different distribution but same total - verifies correctness
        assert np.allclose(
            node_result_short.node_cycles[dist].sum(), cyc_src[d_idx].sum(), atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            node_result_short.node_harmonic[dist], harmonic_cl[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(node_result_short.node_beta[dist], grav[d_idx], atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(node_result_short.node_betweenness[dist], betw[d_idx], atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(
            node_result_short.node_betweenness_beta[dist], betw_wt[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
    # check weights
    for wt in [0.5, 2]:
        # create a weighted version fo the graph
        primal_graph_wt = primal_graph.copy()
        for nd_idx in primal_graph_wt:
            primal_graph_wt.nodes[nd_idx]["weight"] = wt
        # compute weighted measures
        nodes_gdf_wt, edges_gdf_wt, network_structure_wt = io.network_structure_from_nx(primal_graph_wt)
        # weights should persists to the nodes GDF
        assert np.all(nodes_gdf_wt.weight == wt)
        node_result_short_wt = network_structure_wt.local_node_centrality_shortest(
            distances=distances,
            compute_closeness=True,
            compute_betweenness=True,
        )
        # check that weighted versions behave as anticipated
        for dist in distances:
            assert np.allclose(
                node_result_short.node_beta[dist] * wt,
                node_result_short_wt.node_beta[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_betweenness[dist] * wt,
                node_result_short_wt.node_betweenness[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_betweenness_beta[dist] * wt,
                node_result_short_wt.node_betweenness_beta[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_cycles[dist] * wt,
                node_result_short_wt.node_cycles[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_density[dist] * wt,
                node_result_short_wt.node_density[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_farness[dist] * wt,
                node_result_short_wt.node_farness[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_short.node_harmonic[dist] * wt,
                node_result_short_wt.node_harmonic[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )


def test_local_centrality_all(diamond_graph):
    """
    manual checks for all methods against diamond graph
    measures_data is multidimensional in the form of measure_keys x distances x nodes
    """
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(diamond_graph)
    # generate dual
    diamond_graph_dual = graphs.nx_to_dual(diamond_graph)
    _nodes_gdf_d, _edges_gdf_d, network_structure_dual = io.network_structure_from_nx(diamond_graph_dual)
    # setup distances and betas
    distances = [50, 150, 250]
    rustalgos.betas_from_distances(distances)
    # NODE SHORTEST
    node_result_short = network_structure.local_node_centrality_shortest(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    # node density
    # additive nodes
    assert np.allclose(node_result_short.node_density[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_density[150], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_density[250], [3, 3, 3, 3], atol=config.ATOL, rtol=config.RTOL)
    # node farness
    # additive distances
    assert np.allclose(node_result_short.node_farness[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_farness[150], [200, 300, 300, 200], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_farness[250], [400, 300, 300, 400], atol=config.ATOL, rtol=config.RTOL)
    # node cycles
    # additive cycles
    assert np.allclose(node_result_short.node_cycles[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_cycles[150], [1, 2, 2, 1], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_cycles[250], [2, 2, 2, 2], atol=config.ATOL, rtol=config.RTOL)
    # node harmonic
    # additive 1 / distances
    assert np.allclose(node_result_short.node_harmonic[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        node_result_short.node_harmonic[150], [0.02, 0.03, 0.03, 0.02], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        node_result_short.node_harmonic[250], [0.025, 0.03, 0.03, 0.025], atol=config.ATOL, rtol=config.RTOL
    )
    # node beta
    # additive exp(-beta * dist)
    # beta = 0.0
    assert np.allclose(node_result_short.node_beta[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # beta = 0.02666667
    np.allclose(
        node_result_short.node_beta[150],
        [0.1389669, 0.20845035, 0.20845035, 0.1389669],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # beta = 0.016
    np.allclose(
        node_result_short.node_beta[250],
        [0.44455525, 0.6056895, 0.6056895, 0.44455522],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # node betweenness
    # additive 1 per node en route
    assert np.allclose(node_result_short.node_betweenness[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_betweenness[150], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # takes first out of multiple options, so either of following is correct
    assert np.allclose(
        node_result_short.node_betweenness[250], [0, 0, 1, 0], atol=config.ATOL, rtol=config.RTOL
    ) or np.allclose(node_result_short.node_betweenness[250], [0, 1, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # node betweenness beta
    # additive exp(-beta * dist) en route
    assert np.allclose(
        node_result_short.node_betweenness_beta[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL
    )  # beta = 0.08
    assert np.allclose(
        node_result_short.node_betweenness_beta[150], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL
    )  # beta = 0.02666667
    # takes first out of multiple options, so either of following is correct
    # beta evaluated over 200m distance from 3 to 0
    # beta = 0.016
    assert np.allclose(node_result_short.node_betweenness_beta[250], [0, 0.0407622, 0, 0]) or np.allclose(
        node_result_short.node_betweenness_beta[250], [0, 0, 0.0407622, 0]
    )
    # node shortest weights tested in previous function

    # NODE SIMPLEST
    node_result_simplest = network_structure.local_node_centrality_simplest(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    node_result_simplest_0_180 = network_structure.local_node_centrality_simplest(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
        farness_scaling_offset=0,
        angular_scaling_unit=180,
    )
    # node density
    # additive nodes
    assert np.allclose(node_result_simplest.node_density[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_density[150], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_density[250], [3, 3, 3, 3], atol=config.ATOL, rtol=config.RTOL)
    # node farness
    # additive angular distances
    # additive 1 + (angle / 180)
    assert np.allclose(node_result_simplest.node_farness[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_farness[150], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_farness[250], [3.333, 3, 3, 3.333], atol=config.ATOL, rtol=config.RTOL)
    # custom scaling
    assert np.allclose(node_result_simplest_0_180.node_farness[150], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        node_result_simplest_0_180.node_farness[250], [0.333, 0, 0, 0.333], atol=config.ATOL, rtol=config.RTOL
    )
    # node harmonic angular
    # additive 1 / (1 + (to_imp / 180))
    assert np.allclose(node_result_simplest.node_harmonic[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_harmonic[150], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_harmonic[250], [2.75, 3, 3, 2.75], atol=config.ATOL, rtol=config.RTOL)
    # should be the same
    assert np.allclose(node_result_simplest_0_180.node_harmonic[150], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        node_result_simplest_0_180.node_harmonic[250], [2.75, 3, 3, 2.75], atol=config.ATOL, rtol=config.RTOL
    )
    # node betweenness angular
    # additive 1 per node en simplest route
    assert np.allclose(node_result_simplest.node_betweenness[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_betweenness[150], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        node_result_simplest.node_betweenness[250], [0, 1, 0, 0], atol=config.ATOL, rtol=config.RTOL
    ) or np.allclose(node_result_simplest.node_betweenness[250], [0, 0, 1, 0], atol=config.ATOL, rtol=config.RTOL)
    # check weights
    for wt in [0.5, 2]:
        # for weighted checks
        diamond_graph_wt = diamond_graph.copy()
        for nd_idx in diamond_graph_wt.nodes():
            diamond_graph_wt.nodes[nd_idx]["weight"] = wt
        _nodes_gdf_wt, _edges_gdf_wt, network_structure_wt = io.network_structure_from_nx(diamond_graph_wt)
        node_result_simplest_wt = network_structure_wt.local_node_centrality_simplest(
            distances,
            compute_closeness=True,
            compute_betweenness=True,
        )
        # check that weighted versions behave as anticipated
        for dist in distances:
            assert np.allclose(
                node_result_simplest.node_betweenness[dist] * wt,
                node_result_simplest_wt.node_betweenness[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.allclose(
                node_result_simplest.node_harmonic[dist] * wt,
                node_result_simplest_wt.node_harmonic[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
            )
    # NODE SIMPLEST ON DUAL network_structure_dual
    node_result_simplest = network_structure_dual.local_node_centrality_simplest(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    # node_keys_dual = ('0_1', '0_2', '1_2', '1_3', '2_3')
    # node harmonic angular
    # make sure the angle is at least 1 to avoid infinity for 0 angular distance summation
    # additive 1 / (1 + to_imp / 180)
    assert np.allclose(node_result_simplest.node_harmonic[50], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        node_result_simplest.node_harmonic[150], [1.95, 1.95, 2.4, 1.95, 1.95], atol=config.ATOL, rtol=config.RTOL
    )
    assert np.allclose(
        node_result_simplest.node_harmonic[250], [2.45, 2.45, 2.4, 2.45, 2.45], atol=config.ATOL, rtol=config.RTOL
    )
    # node betweenness angular
    # additive 1 per node en simplest route
    assert np.allclose(node_result_simplest.node_betweenness[50], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_betweenness[150], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_simplest.node_betweenness[250], [0, 0, 0, 1, 1], atol=config.ATOL, rtol=config.RTOL)
    # SEGMENT SHORTEST
    segment_result = network_structure.local_segment_centrality(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    # segment density
    # additive segment lengths
    assert np.allclose(segment_result.segment_density[50], [100, 150, 150, 100], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(segment_result.segment_density[150], [400, 500, 500, 400], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(segment_result.segment_density[250], [500, 500, 500, 500], atol=config.ATOL, rtol=config.RTOL)
    # segment harmonic
    # segments are potentially approached from two directions
    # i.e. along respective shortest paths to intersection of shortest routes
    # i.e. in this case, the midpoint of the middle segment is apportioned in either direction
    # additive log(b) - log(a) + log(d) - log(c)
    # nearer distance capped at 1m to avert negative numbers
    assert np.allclose(
        segment_result.segment_harmonic[50],
        [7.824046, 11.736069, 11.736069, 7.824046],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        segment_result.segment_harmonic[150],
        [10.832201, 15.437371, 15.437371, 10.832201],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        segment_result.segment_harmonic[250],
        [11.407564, 15.437371, 15.437371, 11.407565],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # segment beta
    # additive (np.exp(-beta * b) - np.exp(-beta * a)) / -beta + (np.exp(-beta * d) - np.exp(-beta * c)) / -beta
    # beta = 0 resolves to b - a and avoids division through zero
    assert np.allclose(
        segment_result.segment_beta[50],
        [24.54211, 36.813164, 36.813164, 24.54211],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        segment_result.segment_beta[150],
        [77.45388, 112.34476, 112.34476, 77.45388],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        segment_result.segment_beta[250],
        [133.80203, 177.439, 177.439, 133.80203],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # segment betweenness
    # similar formulation to segment beta: start and end segment of each betweenness pair assigned to intervening nodes
    # distance thresholds are computed using the inside edges of the segments
    # so if the segments are touching, they will count up to the threshold distance...
    assert np.allclose(segment_result.segment_betweenness[50], [0, 0, 24.542109, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(segment_result.segment_betweenness[150], [0, 0, 69.78874, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(segment_result.segment_betweenness[250], [0, 0, 99.76293, 0], atol=config.ATOL, rtol=config.RTOL)
    """
    NOTE: segment simplest has been removed since v4
    # SEGMENT SIMPLEST ON PRIMAL::: ( NO DOUBLE COUNTING )
    # segment density
    # additive segment lengths divided through angular impedance
    # (f - e) / (1 + (ang / 180))
    m_idx = segment_keys_angular.index("segment_harmonic_hybrid")
    assert np.allclose(measures_data[m_idx][0], [100, 150, 150, 100], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [305, 360, 360, 305], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [410, 420, 420, 410], atol=config.ATOL, rtol=config.RTOL)
    # segment harmonic
    # additive segment lengths / (1 + (ang / 180))
    m_idx = segment_keys_angular.index("segment_betweeness_hybrid")
    assert np.allclose(measures_data[m_idx][0], [0, 75, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0, 150, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [0, 150, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # SEGMENT SIMPLEST IS DISCOURAGED FOR DUAL
    # this is because it leads to double counting where segments overlap
    # e.g. 6 segments replace a single four-way intersection
    # it also causes issuse with sidestepping vs. discovering all necessary edges...
    """


def test_decomposed_local_centrality(primal_graph):
    # centralities on the original nodes within the decomposed network should equal non-decomposed workflow
    distances = [200, 400, 800, 5000]
    # test a decomposed graph
    G_decomposed = graphs.nx_decompose(primal_graph, 20)
    # graph maps
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)  # generate node and edge maps
    _node_keys_decomp, _edges_gdf_decomp, network_structure_decomp = io.network_structure_from_nx(G_decomposed)
    # with increasing decomposition:
    # - node based measures will not match
    # - closeness segment measures will match - these measure to the cut endpoints per thresholds
    # - betweenness segment measures won't match - don't measure to cut endpoints
    segment_result = network_structure.local_segment_centrality(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    segment_result_decomp = network_structure_decomp.local_segment_centrality(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    # compare against the original 56 elements (decomposed adds new nodes)
    assert np.allclose(segment_result.segment_density[400].sum(), segment_result_decomp.segment_density[400][:57].sum())
    assert np.allclose(segment_result.segment_beta[400].sum(), segment_result_decomp.segment_beta[400][:57].sum())
    assert np.allclose(
        segment_result.segment_harmonic[400].sum(), segment_result_decomp.segment_harmonic[400][:57].sum()
    )
