# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config
from cityseer.algos import centrality
from cityseer.metrics import networks
from cityseer.tools import graphs


def find_path(start_idx, target_idx, tree_preds):
    """
    for extracting paths from predecessor map
    """
    s_path: list[int] = []
    pred: int = start_idx
    while True:
        s_path.append(pred)
        if pred == target_idx:
            break
        pred = tree_preds[pred].astype(int)

    return list(reversed(s_path))


def test_shortest_path_tree(primal_graph, dual_graph):
    nodes_gdf_p, network_structure_p = graphs.network_structure_from_nx(primal_graph, 3395)
    # prepare round-trip graph for checks
    G_round_trip = graphs.nx_from_network_structure(nodes_gdf_p, network_structure_p)
    # prepare dual graph
    nodes_gdf_d, network_structure_d = graphs.network_structure_from_nx(dual_graph, 3395)
    assert len(nodes_gdf_d) > len(nodes_gdf_p)
    # test all shortest paths against networkX version of dijkstra
    for max_dist in [0, 500, 2000, np.inf]:
        for src_idx in range(len(primal_graph)):
            # check shortest path maps
            (
                _visited_nodes,
                preds_p0,
                short_dist_p0,
                _simpl_dist,
                _cycles,
                _origin_seg,
                _last_seg,
                _out_bearings,
                _visited_edges,
            ) = centrality.shortest_path_tree(
                network_structure_p.edges.start,
                network_structure_p.edges.end,
                network_structure_p.edges.length,
                network_structure_p.edges.angle_sum,
                network_structure_p.edges.imp_factor,
                network_structure_p.edges.in_bearing,
                network_structure_p.edges.out_bearing,
                network_structure_p.node_edge_map,
                src_idx,
                np.float32(max_dist),
                angular=False,
            )
            # compare against networkx dijkstra
            nx_dist, nx_path = nx.single_source_dijkstra(G_round_trip, src_idx, weight="length", cutoff=max_dist)
            for j in range(len(primal_graph)):
                if j in nx_path:
                    assert find_path(j, src_idx, preds_p0) == nx_path[j]
                    assert np.allclose(short_dist_p0[j], nx_dist[j], atol=config.ATOL, rtol=config.RTOL)
    # compare angular simplest paths for a selection of targets on primal vs. dual
    # remember, this is angular change not distance travelled
    # can be compared from primal to dual in this instance because edge segments are straight
    # i.e. same amount of angular change whether primal or dual graph
    # plot.plot_nx_primal_or_dual(primal=primal_graph, dual=dual_graph, labels=True, node_size=80)
    p_source_idx = nodes_gdf_p.index.tolist().index(0)
    primal_targets = (15, 20, 37)
    dual_sources = ("0_1", "0_16", "0_31")
    dual_targets = ("13_15", "17_20", "36_37")
    for p_target, d_source, d_target in zip(primal_targets, dual_sources, dual_targets):
        p_target_idx = nodes_gdf_p.index.tolist().index(p_target)
        d_source_idx = nodes_gdf_d.index.tolist().index(d_source)  # dual source index changes depending on direction
        d_target_idx = nodes_gdf_d.index.tolist().index(d_target)
        (
            _visited_nodes,
            _preds,
            _short_dist,
            simpl_dist_p1,
            _cycles,
            _origin_seg,
            _last_seg,
            _out_bearings,
            _visited_edges,
        ) = centrality.shortest_path_tree(
            network_structure_p.edges.start,
            network_structure_p.edges.end,
            network_structure_p.edges.length,
            network_structure_p.edges.angle_sum,
            network_structure_p.edges.imp_factor,
            network_structure_p.edges.in_bearing,
            network_structure_p.edges.out_bearing,
            network_structure_p.node_edge_map,
            p_source_idx,
            np.float32(np.inf),
            angular=True,
        )
        (
            _visited_nodes,
            _preds,
            _short_dist,
            simpl_dist_d1,
            _cycles,
            _origin_seg,
            _last_seg,
            _out_bearings,
            _visited_edges,
        ) = centrality.shortest_path_tree(
            network_structure_d.edges.start,
            network_structure_d.edges.end,
            network_structure_d.edges.length,
            network_structure_d.edges.angle_sum,
            network_structure_d.edges.imp_factor,
            network_structure_d.edges.in_bearing,
            network_structure_d.edges.out_bearing,
            network_structure_d.node_edge_map,
            d_source_idx,
            np.float32(np.inf),
            angular=True,
        )
        assert np.allclose(
            simpl_dist_p1[p_target_idx],
            simpl_dist_d1[d_target_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # angular impedance should take a simpler but longer path - test basic case on dual
    # source and target are the same for either
    src_idx = nodes_gdf_d.index.tolist().index("11_6")
    target = nodes_gdf_d.index.tolist().index("39_40")
    # SIMPLEST PATH: get simplest path tree using angular impedance
    (
        _visited_nodes,
        preds_d2,
        _short_dist,
        _simpl_dist,
        _cycles,
        _origin_seg,
        _last_seg,
        _out_bearings,
        _visited_edges,
    ) = centrality.shortest_path_tree(
        network_structure_d.edges.start,
        network_structure_d.edges.end,
        network_structure_d.edges.length,
        network_structure_d.edges.angle_sum,
        network_structure_d.edges.imp_factor,
        network_structure_d.edges.in_bearing,
        network_structure_d.edges.out_bearing,
        network_structure_d.node_edge_map,
        src_idx,
        np.float32(np.inf),
        angular=True,
    )  # ANGULAR = TRUE
    # find path
    path = find_path(target, src_idx, preds_d2)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # takes 1597m route via long outside segment
    # tree_dists[int(full_to_trim_idx_map[node_keys.index('39_40')])]
    assert path_transpose == [
        "11_6",
        "11_14",
        "10_14",
        "10_43",
        "43_44",
        "40_44",
        "39_40",
    ]
    # SHORTEST PATH:
    # get shortest path tree using non angular impedance
    (
        _visited_nodes,
        preds_d3,
        _short_dist,
        _simpl_dist,
        _cycles,
        _origin_seg,
        _last_seg,
        _out_bearings,
        _visited_edges,
    ) = centrality.shortest_path_tree(
        network_structure_d.edges.start,
        network_structure_d.edges.end,
        network_structure_d.edges.length,
        network_structure_d.edges.angle_sum,
        network_structure_d.edges.imp_factor,
        network_structure_d.edges.in_bearing,
        network_structure_d.edges.out_bearing,
        network_structure_d.node_edge_map,
        src_idx,
        max_dist=np.float32(np.inf),
        angular=False,
    )  # ANGULAR = FALSE
    # find path
    path = find_path(target, src_idx, preds_d3)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # takes 1345m shorter route
    # tree_dists[int(full_to_trim_idx_map[node_keys.index('39_40')])]
    assert path_transpose == [
        "11_6",
        "6_7",
        "3_7",
        "3_4",
        "1_4",
        "0_1",
        "0_31",
        "31_32",
        "32_34",
        "34_37",
        "37_39",
        "39_40",
    ]
    # NO SIDESTEPS - explicit check that sidesteps are prevented
    src_idx = nodes_gdf_d.index.tolist().index("10_43")
    target = nodes_gdf_d.index.tolist().index("10_5")
    (
        _visited_nodes,
        preds_d4,
        _short_dist,
        _simpl_dist,
        _cycles,
        _origin_seg,
        _last_seg,
        _out_bearings,
        _visited_edges,
    ) = centrality.shortest_path_tree(
        network_structure_d.edges.start,
        network_structure_d.edges.end,
        network_structure_d.edges.length,
        network_structure_d.edges.angle_sum,
        network_structure_d.edges.imp_factor,
        network_structure_d.edges.in_bearing,
        network_structure_d.edges.out_bearing,
        network_structure_d.node_edge_map,
        src_idx,
        max_dist=np.float32(np.inf),
        angular=True,
    )
    # find path
    path = find_path(target, src_idx, preds_d4)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    # print(path_transpose)
    assert path_transpose == ["10_43", "10_5"]
    # WITH SIDESTEPS - set angular flag to False
    # manually overwrite distance impedances with angular for this test
    # (angular has to be false otherwise shortest-path sidestepping avoided)
    nodes_gdf_d, network_structure_d = graphs.network_structure_from_nx(dual_graph, 3395)
    network_structure_d.edges.length = network_structure_d.edges.angle_sum
    (
        _visited_nodes,
        preds_d5,
        _short_dist,
        _simpl_dist,
        _cycles,
        _origin_seg,
        _last_seg,
        _out_bearings,
        _visited_edges,
    ) = centrality.shortest_path_tree(
        network_structure_d.edges.start,
        network_structure_d.edges.end,
        network_structure_d.edges.length,
        network_structure_d.edges.angle_sum,
        network_structure_d.edges.imp_factor,
        network_structure_d.edges.in_bearing,
        network_structure_d.edges.out_bearing,
        network_structure_d.node_edge_map,
        src_idx,
        max_dist=np.float32(np.inf),
        angular=False,
    )
    # find path
    path = find_path(target, src_idx, preds_d5)
    path_transpose = [nodes_gdf_d.index[n] for n in path]
    assert path_transpose == ["10_43", "10_14", "10_5"]


def test_local_node_centrality(primal_graph):
    """
    Also tested indirectly via test_networks.test_compute_centrality

    Test centrality methods where possible against NetworkX - i.e. harmonic closeness and betweenness
    Note that NetworkX improved closeness is not the same as derivation used in this package
    NetworkX doesn't have a maximum distance cutoff, so run on the whole graph (low beta / high distance)
    """
    # generate node and edge maps
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    G_round_trip = graphs.nx_from_network_structure(nodes_gdf, network_structure)
    # needs a large enough beta so that distance thresholds aren't encountered
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0008], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    # set the keys - add shuffling to be sure various orders work
    measure_keys = [
        "node_density",
        "node_farness",
        "node_cycles",
        "node_harmonic",
        "node_beta",
        "node_betweenness",
        "node_betweenness_beta",
    ]
    np.random.shuffle(measure_keys)  # in place
    measure_keys = tuple(measure_keys)
    # generate the measures
    measures_data = centrality.local_node_centrality(
        distances,
        betas,
        measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
    )
    node_density = measures_data[measure_keys.index("node_density")]
    node_farness = measures_data[measure_keys.index("node_farness")]
    node_cycles = measures_data[measure_keys.index("node_cycles")]
    node_harmonic = measures_data[measure_keys.index("node_harmonic")]
    node_beta = measures_data[measure_keys.index("node_beta")]
    node_betweenness = measures_data[measure_keys.index("node_betweenness")]
    node_betweenness_beta = measures_data[measure_keys.index("node_betweenness_beta")]
    # improved closeness is derived after the fact
    improved_closness = node_density / node_farness / node_density
    # test node density
    # node density count doesn't include self-node
    # connected component == 49 == len(G) - 1
    # isolated looping component == 3
    # isolated edge == 1
    # isolated node == 0
    for n in node_density[3]:  # large distance - exceeds cutoff clashes
        assert n in [49, 3, 1, 0]
    # test harmonic closeness vs NetworkX
    nx_harm_cl = nx.harmonic_centrality(G_round_trip, distance="length")
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.allclose(nx_harm_cl, node_harmonic[3], atol=config.ATOL, rtol=config.RTOL)
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
    assert np.allclose(nx_betw[:52], node_betweenness[3][:52], atol=config.ATOL, rtol=config.RTOL)
    # do the comparisons array-wise so that betweenness can be aggregated
    d_n = len(distances)
    n_nodes: int = primal_graph.number_of_nodes()
    betw: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    betw_wt: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    dens: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    far_short_dist: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    far_simpl_dist: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    harmonic_cl: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    grav: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    cyc: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    for src_idx in range(n_nodes):
        # get shortest path maps
        (
            visited_nodes,
            preds,
            short_dist,
            simpl_dist,
            cycles,
            _origin_seg,
            _last_seg,
            _out_bearings,
            _visited_edges,
        ) = centrality.shortest_path_tree(
            network_structure.edges.start,
            network_structure.edges.end,
            network_structure.edges.length,
            network_structure.edges.angle_sum,
            network_structure.edges.imp_factor,
            network_structure.edges.in_bearing,
            network_structure.edges.out_bearing,
            network_structure.node_edge_map,
            src_idx,
            max(distances),
            angular=False,
        )
        tree_nodes: list[int | str] = np.where(visited_nodes)[0]
        for to_idx in tree_nodes:
            # skip self nodes
            if to_idx == src_idx:
                continue
            # get shortest / simplest distances
            to_short_dist = short_dist[to_idx]
            to_simpl_dist = simpl_dist[to_idx]
            n_cycles = cycles[to_idx]
            # continue if exceeds max
            if np.isinf(to_short_dist):
                continue
            for d_idx in range(len(distances)):
                dist_cutoff = distances[d_idx]
                beta = betas[d_idx]
                if to_short_dist <= dist_cutoff:
                    # don't exceed threshold
                    # if to_dist <= dist_cutoff:
                    # aggregate values
                    dens[d_idx][src_idx] += 1
                    far_short_dist[d_idx][src_idx] += to_short_dist
                    far_simpl_dist[d_idx][src_idx] += to_simpl_dist
                    harmonic_cl[d_idx][src_idx] += 1 / to_short_dist
                    grav[d_idx][src_idx] += np.exp(-beta * to_short_dist)
                    # cycles
                    cyc[d_idx][src_idx] += n_cycles
                    # only process betweenness in one direction
                    if to_idx < src_idx:
                        continue
                    # betweenness - only counting truly between vertices, not starting and ending verts
                    inter_idx = preds[to_idx]
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
                        inter_idx = int(preds[inter_idx])
    improved_cl = dens / far_short_dist / dens
    assert np.allclose(node_density, dens, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_farness, far_short_dist, atol=config.ATOL, rtol=config.RTOL)  # relax precision
    assert np.allclose(node_cycles, cyc, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_harmonic, harmonic_cl, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_beta, grav, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(improved_closness, improved_cl, equal_nan=True, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_betweenness, betw, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_betweenness_beta, betw_wt, atol=config.ATOL, rtol=config.RTOL)
    # catch typos
    with pytest.raises(ValueError):
        centrality.local_node_centrality(
            distances,
            betas,
            ("typo_key",),
            network_structure.nodes.live,
            network_structure.edges.start,
            network_structure.edges.end,
            network_structure.edges.length,
            network_structure.edges.angle_sum,
            network_structure.edges.imp_factor,
            network_structure.edges.in_bearing,
            network_structure.edges.out_bearing,
            network_structure.node_edge_map,
        )


def test_local_centrality(diamond_graph):
    """
    manual checks for all methods against diamond graph
    measures_data is multidimensional in the form of measure_keys x distances x nodes
    """
    # generate node and edge maps
    _nodes_gdf, network_structure = graphs.network_structure_from_nx(diamond_graph, 3395)
    # generate dual
    diamond_graph_dual = graphs.nx_to_dual(diamond_graph)
    _node_keys_dual, network_structure_dual = graphs.network_structure_from_nx(diamond_graph_dual, 3395)
    # setup distances and betas
    distances: npt.NDArray[np.int_] = np.array([50, 150, 250], dtype=np.float32)
    betas = networks.beta_from_distance(distances)
    # NODE SHORTEST
    # set the keys - add shuffling to be sure various orders work
    node_keys = [
        "node_density",
        "node_farness",
        "node_cycles",
        "node_harmonic",
        "node_beta",
        "node_betweenness",
        "node_betweenness_beta",
    ]
    np.random.shuffle(node_keys)  # in place
    measure_keys = tuple(node_keys)
    measures_data = centrality.local_node_centrality(
        distances,
        betas,
        measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
    )
    # node density
    # additive nodes
    m_idx = node_keys.index("node_density")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [3, 3, 3, 3], atol=config.ATOL, rtol=config.RTOL)
    # node farness
    # additive distances
    m_idx = node_keys.index("node_farness")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [200, 300, 300, 200], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [400, 300, 300, 400], atol=config.ATOL, rtol=config.RTOL)
    # node cycles
    # additive cycles
    m_idx = node_keys.index("node_cycles")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [1, 2, 2, 1], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [2, 2, 2, 2], atol=config.ATOL, rtol=config.RTOL)
    # node harmonic
    # additive 1 / distances
    m_idx = node_keys.index("node_harmonic")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0.02, 0.03, 0.03, 0.02], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [0.025, 0.03, 0.03, 0.025], atol=config.ATOL, rtol=config.RTOL)
    # node beta
    # additive exp(-beta * dist)
    m_idx = node_keys.index("node_beta")
    # beta = 0.0
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # beta = 0.02666667
    assert np.allclose(
        measures_data[m_idx][1],
        [0.1389669, 0.20845035, 0.20845035, 0.1389669],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # beta = 0.016
    assert np.allclose(
        measures_data[m_idx][2],
        [0.44455525, 0.6056895, 0.6056895, 0.44455522],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # node betweenness
    # additive 1 per node en route
    m_idx = node_keys.index("node_betweenness")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # takes first out of multiple equidistant routes
    assert np.allclose(measures_data[m_idx][2], [0, 1, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # node betweenness beta
    # additive exp(-beta * dist) en route
    m_idx = node_keys.index("node_betweenness_beta")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)  # beta = 0.08
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)  # beta = 0.02666667
    # takes first out of multiple equidistant routes
    # beta evaluated over 200m distance from 3 to 0 via node 1
    assert np.allclose(measures_data[m_idx][2], [0, 0.0407622, 0, 0])  # beta = 0.016

    # NODE SIMPLEST
    node_keys_angular = ["node_harmonic_angular", "node_betweenness_angular"]
    np.random.shuffle(node_keys_angular)  # in place
    measure_keys = tuple(node_keys_angular)
    measures_data = centrality.local_node_centrality(
        distances,
        betas,
        measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        angular=True,
    )
    # node harmonic angular
    # additive 1 / (1 + (to_imp / 180))
    m_idx = node_keys_angular.index("node_harmonic_angular")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [2, 3, 3, 2], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [2.75, 3, 3, 2.75], atol=config.ATOL, rtol=config.RTOL)
    # node betweenness angular
    # additive 1 per node en simplest route
    m_idx = node_keys_angular.index("node_betweenness_angular")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [0, 1, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # NODE SIMPLEST ON DUAL
    node_keys_angular = ["node_harmonic_angular", "node_betweenness_angular"]
    np.random.shuffle(node_keys_angular)  # in place
    measure_keys = tuple(node_keys_angular)
    measures_data = centrality.local_node_centrality(
        distances,
        betas,
        measure_keys,
        network_structure_dual.nodes.live,
        network_structure_dual.edges.start,
        network_structure_dual.edges.end,
        network_structure_dual.edges.length,
        network_structure_dual.edges.angle_sum,
        network_structure_dual.edges.imp_factor,
        network_structure_dual.edges.in_bearing,
        network_structure_dual.edges.out_bearing,
        network_structure_dual.node_edge_map,
        angular=True,
    )
    # node_keys_dual = ('0_1', '0_2', '1_2', '1_3', '2_3')
    # node harmonic angular
    # additive 1 / (1 + (to_imp / 180))
    m_idx = node_keys_angular.index("node_harmonic_angular")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [1.95, 1.95, 2.4, 1.95, 1.95], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [2.45, 2.45, 2.4, 2.45, 2.45], atol=config.ATOL, rtol=config.RTOL)
    # node betweenness angular
    # additive 1 per node en simplest route
    m_idx = node_keys_angular.index("node_betweenness_angular")
    assert np.allclose(measures_data[m_idx][0], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0, 0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [0, 0, 0, 1, 1], atol=config.ATOL, rtol=config.RTOL)
    # SEGMENT SHORTEST
    segment_keys = [
        "segment_density",
        "segment_harmonic",
        "segment_beta",
        "segment_betweenness",
    ]
    np.random.shuffle(segment_keys)  # in place
    measure_keys = tuple(segment_keys)
    measures_data = centrality.local_segment_centrality(
        distances,
        betas,
        measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        angular=False,
    )
    # segment density
    # additive segment lengths
    m_idx = segment_keys.index("segment_density")
    assert np.allclose(measures_data[m_idx][0], [100, 150, 150, 100], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [400, 500, 500, 400], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [500, 500, 500, 500], atol=config.ATOL, rtol=config.RTOL)
    # segment harmonic
    # segments are potentially approached from two directions
    # i.e. along respective shortest paths to intersection of shortest routes
    # i.e. in this case, the midpoint of the middle segment is apportioned in either direction
    # additive log(b) - log(a) + log(d) - log(c)
    # nearer distance capped at 1m to avert negative numbers
    m_idx = segment_keys.index("segment_harmonic")
    assert np.allclose(
        measures_data[m_idx][0],
        [7.824046, 11.736069, 11.736069, 7.824046],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        measures_data[m_idx][1],
        [10.832201, 15.437371, 15.437371, 10.832201],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        measures_data[m_idx][2],
        [11.407564, 15.437371, 15.437371, 11.407565],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # segment beta
    # additive (np.exp(-beta * b) - np.exp(-beta * a)) / -beta + (np.exp(-beta * d) - np.exp(-beta * c)) / -beta
    # beta = 0 resolves to b - a and avoids division through zero
    m_idx = segment_keys.index("segment_beta")
    assert np.allclose(
        measures_data[m_idx][0],
        [24.542109, 36.813164, 36.813164, 24.542109],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        measures_data[m_idx][1],
        [77.46391, 112.358284, 112.358284, 77.46391],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        measures_data[m_idx][2],
        [133.80205, 177.43903, 177.43904, 133.80205],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # segment betweenness
    # similar formulation to segment beta: start and end segment of each betweenness pair assigned to intervening nodes
    # distance thresholds are computed using the inside edges of the segments
    # so if the segments are touching, they will count up to the threshold distance...
    m_idx = segment_keys.index("segment_betweenness")
    assert np.allclose(measures_data[m_idx][0], [0, 24.542109, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][1], [0, 69.78874, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(measures_data[m_idx][2], [0, 99.76293, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    # SEGMENT SIMPLEST ON PRIMAL!!! ( NO DOUBLE COUNTING )
    segment_keys_angular = ["segment_harmonic_hybrid", "segment_betweeness_hybrid"]
    np.random.shuffle(segment_keys_angular)  # in place
    measure_keys = tuple(segment_keys_angular)
    measures_data = centrality.local_segment_centrality(
        distances,
        betas,
        measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        angular=True,
    )
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


def test_decomposed_local_centrality(primal_graph):
    # centralities on the original nodes within the decomposed network should equal non-decomposed workflow
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0008], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    node_measure_keys = (
        "node_density",
        "node_farness",
        "node_cycles",
        "node_harmonic",
        "node_beta",
        "node_betweenness",
        "node_betweenness_beta",
    )
    segment_measure_keys = (
        "segment_density",
        "segment_harmonic",
        "segment_beta",
        "segment_betweenness",
    )
    # test a decomposed graph
    G_decomposed = graphs.nx_decompose(primal_graph, 20)
    # graph maps
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)  # generate node and edge maps
    _node_keys_decomp, network_structure_decomp = graphs.network_structure_from_nx(G_decomposed, 3395)
    # non-decomposed case
    node_measures_data = centrality.local_node_centrality(
        distances,
        betas,
        node_measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        angular=False,
    )
    # decomposed case
    node_measures_data_decomposed = centrality.local_node_centrality(
        distances,
        betas,
        node_measure_keys,
        network_structure_decomp.nodes.live,
        network_structure_decomp.edges.start,
        network_structure_decomp.edges.end,
        network_structure_decomp.edges.length,
        network_structure_decomp.edges.angle_sum,
        network_structure_decomp.edges.imp_factor,
        network_structure_decomp.edges.in_bearing,
        network_structure_decomp.edges.out_bearing,
        network_structure_decomp.node_edge_map,
        angular=False,
    )
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
    segment_measures_data = centrality.local_segment_centrality(
        distances,
        betas,
        segment_measure_keys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        angular=False,
    )
    segment_measures_data_decomposed = centrality.local_segment_centrality(
        distances,
        betas,
        segment_measure_keys,
        network_structure_decomp.nodes.live,
        network_structure_decomp.edges.start,
        network_structure_decomp.edges.end,
        network_structure_decomp.edges.length,
        network_structure_decomp.edges.angle_sum,
        network_structure_decomp.edges.imp_factor,
        network_structure_decomp.edges.in_bearing,
        network_structure_decomp.edges.out_bearing,
        network_structure_decomp.node_edge_map,
        angular=False,
    )
    m_range = len(segment_measure_keys)
    assert segment_measures_data.shape == (m_range, d_range, len(primal_graph))
    assert segment_measures_data_decomposed.shape == (
        m_range,
        d_range,
        len(G_decomposed),
    )
    for m_idx in range(m_range):
        for d_idx in range(d_range):
            match = np.allclose(
                segment_measures_data[m_idx][d_idx],
                # compare against the original 56 elements (prior to adding decomposed)
                segment_measures_data_decomposed[m_idx][d_idx][:57],
                atol=config.ATOL,
                rtol=config.RTOL,
            )  # relax precision
            if m_range in (0, 1, 2):
                assert match
