# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
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


def make_single_edge_graph(z0: float | None = None, z1: float | None = None) -> nx.MultiGraph:
    from pyproj import CRS

    G = nx.MultiGraph()
    G.graph["crs"] = CRS(32630)
    node_0 = {"x": 0.0, "y": 0.0}
    node_1 = {"x": 100.0, "y": 0.0}
    if z0 is not None:
        node_0["z"] = z0
    if z1 is not None:
        node_1["z"] = z1
    G.add_node("0", **node_0)
    G.add_node("1", **node_1)
    G.add_edge("0", "1")
    return graphs.nx_simple_geoms(G)


def make_two_segment_line_graph(
    z0: float | None = None,
    z1: float | None = None,
    z2: float | None = None,
) -> nx.MultiGraph:
    from pyproj import CRS

    G = nx.MultiGraph()
    G.graph["crs"] = CRS(32630)
    node_0 = {"x": 0.0, "y": 0.0}
    node_1 = {"x": 100.0, "y": 0.0}
    node_2 = {"x": 200.0, "y": 0.0}
    if z0 is not None:
        node_0["z"] = z0
    if z1 is not None:
        node_1["z"] = z1
    if z2 is not None:
        node_2["z"] = z2
    G.add_node("0", **node_0)
    G.add_node("1", **node_1)
    G.add_node("2", **node_2)
    G.add_edge("0", "1")
    G.add_edge("1", "2")
    return graphs.nx_simple_geoms(G)


def make_angular_plateau_graph() -> nx.MultiGraph:
    from pyproj import CRS

    G = nx.MultiGraph()
    G.graph["crs"] = CRS(32630)
    coords = {
        "A": (0.0, 0.0),
        "B": (100.0, 0.0),
        "C": (200.0, 0.0),
        "D": (300.0, 0.0),
        "E": (400.0, 0.0),
        "BU": (100.0, 100.0),
        "BD": (100.0, -100.0),
        "CU": (200.0, 100.0),
    }
    for node_key, (x, y) in coords.items():
        G.add_node(node_key, x=x, y=y)
    for start, end in [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("B", "BU"),
        ("B", "BD"),
        ("C", "CU"),
    ]:
        G.add_edge(start, end)
    return graphs.nx_simple_geoms(G)


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
    # test all shortest distance calculations against networkX
    max_seconds_5000 = 5000 / config.SPEED_M_S
    for src_idx in range(len(G_round_trip)):
        shortest_dists = nx.shortest_path_length(G_round_trip, str(src_idx), weight="length")
        _visted_nodes, tree_map = network_structure_p.dijkstra_tree_shortest(
            src_idx, int(max_seconds_5000), config.SPEED_M_S
        )
        for target_idx in range(len(G_round_trip)):
            if str(target_idx) not in shortest_dists:
                continue
            assert shortest_dists[str(target_idx)] - tree_map[target_idx].short_dist <= config.ATOL
    with pytest.raises(ValueError, match="dual graph"):
        network_structure_p.dijkstra_tree_simplest(0, int(max_seconds_5000), config.SPEED_M_S)
    # prepare dual graph
    nodes_gdf_d, edges_gdf_d, network_structure_d = io.network_structure_from_nx(dual_graph)
    assert len(nodes_gdf_d) > len(nodes_gdf_p)
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


def test_simplest_requires_dual_graph(primal_graph):
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    with pytest.raises(ValueError, match="dual graph"):
        network_structure.centrality_simplest(
            compute_closeness=True,
            compute_betweenness=False,
            distances=[500],
        )


def test_closeness_shortest(primal_graph):
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
    node_result_short = network_structure.centrality_shortest(
        compute_closeness=True,
        compute_betweenness=False,
        distances=distances,
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
    # do the comparisons array-wise so that closeness metrics can be verified
    d_n = len(distances)
    n_nodes: int = primal_graph.number_of_nodes()
    dens: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    far_short_dist: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    cycles_surplus: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    harmonic_cl: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    grav: npt.NDArray[np.float32] = np.full((d_n, n_nodes), 0.0, dtype=np.float32)
    #
    max_seconds_5000 = 5000 / config.SPEED_M_S
    for src_idx in range(n_nodes):
        preds, dists = nx.dijkstra_predecessor_and_distance(G_round_trip, str(src_idx), weight="length")
        for to_idx in range(n_nodes):
            if to_idx == src_idx:
                continue
            to_key = str(to_idx)
            if to_key not in dists:
                continue
            to_short_dist = dists[to_key]
            for d_idx, dist_cutoff in enumerate(distances):
                if to_short_dist <= dist_cutoff:
                    cycles_surplus[d_idx][to_idx] += max(0, len(preds[to_key]) - 1)
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
    for d_idx, dist in enumerate(distances):
        assert np.allclose(node_result_short.node_density[dist], dens[d_idx], atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(
            node_result_short.node_farness[dist], far_short_dist[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            node_result_short.node_cycles[dist], cycles_surplus[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            node_result_short.node_harmonic[dist], harmonic_cl[d_idx], atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(node_result_short.node_beta[dist], grav[d_idx], atol=config.ATOL, rtol=config.RTOL)
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
        node_result_short_wt = network_structure_wt.centrality_shortest(
            compute_closeness=True,
            compute_betweenness=False,
            distances=distances,
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


def test_directional_slope_penalty_shortest_and_simplest():
    """Slope penalties should be directional while missing z falls back to flat cost."""
    max_seconds = 1000

    # Rising 5 m over 100 m: source=0 aggregates target 1 -> 0 (downhill),
    # source=1 aggregates target 0 -> 1 (uphill).
    elevated_graph = make_single_edge_graph(z0=0.0, z1=5.0)
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(elevated_graph)
    idx_0 = nodes_gdf.index.tolist().index("0")
    idx_1 = nodes_gdf.index.tolist().index("1")

    _visited_short_0, tree_short_0 = network_structure.dijkstra_tree_shortest(idx_0, max_seconds, config.SPEED_M_S)
    _visited_short_1, tree_short_1 = network_structure.dijkstra_tree_shortest(idx_1, max_seconds, config.SPEED_M_S)
    downhill_short_seconds = tree_short_0[idx_1].agg_seconds
    uphill_short_seconds = tree_short_1[idx_0].agg_seconds

    flat_seconds = 100.0 / config.SPEED_M_S

    assert downhill_short_seconds < flat_seconds < uphill_short_seconds

    with pytest.raises(ValueError, match="dual graph"):
        network_structure.dijkstra_tree_simplest(idx_0, max_seconds, config.SPEED_M_S)

    # On a dual graph, simplest routing should remain angular-only while slope
    # still affects the metric/time cutoff accumulator.
    elevated_line_graph = make_two_segment_line_graph(z0=0.0, z1=5.0, z2=10.0)
    elevated_line_graph_dual = graphs.nx_to_dual(elevated_line_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(elevated_line_graph_dual)
    dual_idx_01 = nodes_gdf_dual.index.tolist().index("0_1_k0")
    dual_idx_12 = nodes_gdf_dual.index.tolist().index("1_2_k0")
    _visited_simpl_01, tree_simpl_01 = network_structure_dual.dijkstra_tree_simplest(
        dual_idx_01, max_seconds, config.SPEED_M_S
    )
    _visited_simpl_12, tree_simpl_12 = network_structure_dual.dijkstra_tree_simplest(
        dual_idx_12, max_seconds, config.SPEED_M_S
    )
    simpl_seconds_01_to_12 = tree_simpl_01[dual_idx_12].agg_seconds
    simpl_seconds_12_to_01 = tree_simpl_12[dual_idx_01].agg_seconds
    assert not np.isclose(simpl_seconds_01_to_12, simpl_seconds_12_to_01, atol=config.ATOL)
    assert min(simpl_seconds_01_to_12, simpl_seconds_12_to_01) < flat_seconds
    assert max(simpl_seconds_01_to_12, simpl_seconds_12_to_01) > flat_seconds
    assert np.isclose(tree_simpl_01[dual_idx_12].simpl_dist, 0.0, atol=config.ATOL)
    assert np.isclose(tree_simpl_12[dual_idx_01].simpl_dist, 0.0, atol=config.ATOL)

    # Missing z on either endpoint should disable slope penalties entirely.
    partial_z_graph = make_single_edge_graph(z0=0.0, z1=None)
    nodes_gdf_partial, _edges_gdf_partial, network_structure_partial = io.network_structure_from_nx(partial_z_graph)
    partial_idx_0 = nodes_gdf_partial.index.tolist().index("0")
    partial_idx_1 = nodes_gdf_partial.index.tolist().index("1")
    _visited_partial_0, tree_partial_0 = network_structure_partial.dijkstra_tree_shortest(
        partial_idx_0, max_seconds, config.SPEED_M_S
    )
    _visited_partial_1, tree_partial_1 = network_structure_partial.dijkstra_tree_shortest(
        partial_idx_1, max_seconds, config.SPEED_M_S
    )
    assert np.isclose(tree_partial_0[partial_idx_1].agg_seconds, flat_seconds, atol=config.ATOL)
    assert np.isclose(tree_partial_1[partial_idx_0].agg_seconds, flat_seconds, atol=config.ATOL)


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
    node_result_short = network_structure.centrality_shortest(
        distances=distances,
        compute_closeness=True,
        compute_betweenness=False,
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
    # simplified cycle score from surplus shortest-path predecessors
    assert np.allclose(node_result_short.node_cycles[50], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_cycles[150], [0, 0, 0, 0], atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(node_result_short.node_cycles[250], [1, 0, 0, 1], atol=config.ATOL, rtol=config.RTOL)
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
    # node shortest weights tested in previous function

    with pytest.raises(ValueError, match="dual graph"):
        network_structure.centrality_simplest(
            distances=distances,
            compute_closeness=True,
            compute_betweenness=False,
        )
    # NODE SIMPLEST ON DUAL network_structure_dual
    node_result_simplest = network_structure_dual.centrality_simplest(
        distances=distances,
        compute_closeness=True,
        compute_betweenness=False,
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
    # SEGMENT SHORTEST
    segment_result = network_structure.segment_centrality(
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
    segment_result = network_structure.segment_centrality(
        distances,
        compute_closeness=True,
        compute_betweenness=True,
    )
    segment_result_decomp = network_structure_decomp.segment_centrality(
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


def test_betweenness_vs_networkx(primal_graph):
    """Compare cityseer betweenness against NetworkX betweenness_centrality.

    NetworkX betweenness_centrality(normalized=False) uses the standard Brandes algorithm
    and divides by 2 for undirected graphs — matching cityseer's convention.
    At a large distance cutoff (5000m) that exceeds the mock graph extent,
    the two should agree exactly.
    """
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    G_round_trip = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
    for start_nd_key, end_nd_key, edge_idx in G_round_trip.edges(keys=True):
        geom = G_round_trip[start_nd_key][end_nd_key][edge_idx]["geom"]
        G_round_trip[start_nd_key][end_nd_key][edge_idx]["length"] = geom.length
    # Use a large distance so no cutoff interferes
    betw_result = network_structure.centrality_shortest(
        compute_closeness=False, compute_betweenness=True, distances=[5000]
    )
    nx_betw = nx.betweenness_centrality(G_round_trip, normalized=False, weight="length")
    for src_idx in range(len(G_round_trip)):
        assert abs(nx_betw[str(src_idx)] - betw_result.node_betweenness[5000][src_idx]) < config.ATOL, (
            f"Betweenness mismatch at node {src_idx}: "
            f"NX={nx_betw[str(src_idx)]:.6f}, cityseer={betw_result.node_betweenness[5000][src_idx]:.6f}"
        )


def test_simplest_closeness_differs_from_shortest(dual_graph):
    """Simplest (angular) closeness produces different values from shortest closeness.

    On a non-trivial graph, angular impedance routes through geometrically simpler
    (straighter) paths, which differ from metrically shortest paths. This verifies
    that the simplest path algorithm is not accidentally using shortest-path routing.
    """
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    # Use large distance to avoid cutoff differences between path types
    distances = [5000]
    res_shortest = network_structure.centrality_shortest(
        compute_closeness=True, compute_betweenness=False, distances=distances
    )
    res_simplest = network_structure.centrality_simplest(
        compute_closeness=True, compute_betweenness=False, distances=distances
    )
    # At large distance, density should match (all nodes reachable either way)
    for d in distances:
        assert np.allclose(
            res_shortest.node_density[d], res_simplest.node_density[d], atol=config.ATOL, rtol=config.RTOL
        ), f"Density should match at {d}m when no cutoff applies"
    # Farness and harmonic should differ — angular impedance uses different cost metric
    for d in distances:
        if np.sum(res_shortest.node_farness[d]) > 0:
            assert not np.allclose(res_shortest.node_farness[d], res_simplest.node_farness[d], atol=config.ATOL), (
                f"Farness should differ at {d}m (angular vs metric impedance)"
            )
        if np.sum(res_shortest.node_harmonic[d]) > 0:
            assert not np.allclose(res_shortest.node_harmonic[d], res_simplest.node_harmonic[d], atol=config.ATOL), (
                f"Harmonic should differ at {d}m (angular vs metric impedance)"
            )


def test_simplest_betweenness_invariant_to_node_order():
    """Angular betweenness must not depend on node insertion order.

    Builds an asymmetric graph (T-junction with an angled branch) using three
    different node-label orderings.  All three must produce identical betweenness.
    """
    from pyproj import CRS

    # T-junction: straight road 0--1--2 with angled branch 1--3
    #
    #   3
    #    \
    #     1---2
    #     |
    #     0
    base_coords = {
        "A": (500000.0, 0.0),
        "B": (500000.0, 100.0),
        "C": (500100.0, 100.0),
        "D": (500050.0, 170.0),
    }
    base_edges = [("A", "B"), ("B", "C"), ("B", "D")]
    # Three different label→index mappings
    orderings = [
        ["A", "B", "C", "D"],
        ["D", "C", "B", "A"],
        ["C", "A", "D", "B"],
    ]
    results = []
    for ordering in orderings:
        label_to_idx = {label: str(i) for i, label in enumerate(ordering)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        G = nx.MultiGraph()
        G.graph["crs"] = CRS(32630)
        for label in ordering:
            idx = label_to_idx[label]
            x, y = base_coords[label]
            G.add_node(idx, x=x, y=y)
        for a, b in base_edges:
            G.add_edge(label_to_idx[a], label_to_idx[b])
        G = graphs.nx_simple_geoms(G)
        G_dual = graphs.nx_to_dual(G)
        nodes_gdf, _edges_gdf, net = io.network_structure_from_nx(G_dual)
        res = net.centrality_simplest(
            compute_closeness=False,
            compute_betweenness=True,
            distances=[500],
        )
        # Map dual nodes back to canonical primal edges so we can compare across orderings
        betw_by_edge = {}
        for node_pos, node_key in enumerate(nodes_gdf.index):
            row = nodes_gdf.loc[node_key]
            edge_label = tuple(sorted((idx_to_label[row["primal_edge_node_a"]], idx_to_label[row["primal_edge_node_b"]])))
            betw_by_edge[edge_label] = res.node_betweenness[500][node_pos]
        results.append(betw_by_edge)
    # All orderings must agree
    for edge_label in [("A", "B"), ("B", "C"), ("B", "D")]:
        vals = [r[edge_label] for r in results]
        assert all(abs(v - vals[0]) < config.ATOL for v in vals), (
            f"Edge {edge_label}: betweenness varies with insertion order: {vals}"
        )


def test_betweenness_mixed_live_non_live_invariant_to_node_order():
    """Betweenness must be invariant when non-live nodes are interleaved by index."""
    from pyproj import CRS

    # Linear corridor A-B-C-D with D non-live (boundary/context node).
    base_coords = {
        "A": (500000.0, 0.0),
        "B": (500000.0, 100.0),
        "C": (500000.0, 200.0),
        "D": (500000.0, 300.0),
    }
    base_edges = [("A", "B"), ("B", "C"), ("C", "D")]
    live_by_label = {"A": True, "B": True, "C": True, "D": False}
    # Keep topology fixed but permute label->index assignment, including
    # orderings where the non-live node has a smaller index than live nodes.
    orderings = [
        ["A", "B", "C", "D"],
        ["D", "A", "B", "C"],
        ["B", "D", "A", "C"],
    ]

    shortest_results = []
    simplest_results = []
    for ordering in orderings:
        label_to_idx = {label: str(i) for i, label in enumerate(ordering)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        G = nx.MultiGraph()
        G.graph["crs"] = CRS(32630)
        for label in ordering:
            idx = label_to_idx[label]
            x, y = base_coords[label]
            G.add_node(idx, x=x, y=y, live=live_by_label[label])
        for a, b in base_edges:
            G.add_edge(label_to_idx[a], label_to_idx[b])
        G = graphs.nx_simple_geoms(G)
        _nodes_gdf, _edges_gdf, net = io.network_structure_from_nx(G)
        G_dual = graphs.nx_to_dual(G)
        nodes_gdf_dual, _edges_gdf_dual, net_dual = io.network_structure_from_nx(G_dual)

        res_shortest = net.centrality_shortest(
            compute_closeness=False,
            compute_betweenness=True,
            distances=[1000],
        )
        res_simplest = net_dual.centrality_simplest(
            compute_closeness=False,
            compute_betweenness=True,
            distances=[1000],
        )

        shortest_results.append(
            {label: res_shortest.node_betweenness[1000][int(label_to_idx[label])] for label in ordering}
        )
        simplest_by_edge = {}
        for node_pos, node_key in enumerate(nodes_gdf_dual.index):
            row = nodes_gdf_dual.loc[node_key]
            edge_label = tuple(sorted((idx_to_label[row["primal_edge_node_a"]], idx_to_label[row["primal_edge_node_b"]])))
            simplest_by_edge[edge_label] = res_simplest.node_betweenness[1000][node_pos]
        simplest_results.append(simplest_by_edge)

    for label in ["A", "B", "C", "D"]:
        vals = [r[label] for r in shortest_results]
        assert all(abs(v - vals[0]) < config.ATOL for v in vals), (
            f"shortest node {label}: betweenness varies with insertion order: {vals}"
        )
    for edge_label in [("A", "B"), ("B", "C"), ("C", "D")]:
        vals = [r[edge_label] for r in simplest_results]
        assert all(abs(v - vals[0]) < config.ATOL for v in vals), (
            f"simplest edge {edge_label}: betweenness varies with insertion order: {vals}"
        )


def test_simplest_betweenness_differs_from_shortest(dual_graph):
    """Simplest (angular) betweenness produces different values from shortest betweenness.

    Verifies that the angular betweenness is not accidentally falling back to
    shortest-path routing, which would be a bug.
    """
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    distances = [500, 2000]
    res_shortest = network_structure.centrality_shortest(
        compute_closeness=False, compute_betweenness=True, distances=distances
    )
    res_simplest = network_structure.centrality_simplest(
        compute_closeness=False, compute_betweenness=True, distances=distances
    )
    for d in distances:
        betw_short = np.array(res_shortest.node_betweenness[d])
        betw_simpl = np.array(res_simplest.node_betweenness[d])
        if np.sum(betw_short) > 0 and np.sum(betw_simpl) > 0:
            assert not np.allclose(betw_short, betw_simpl, atol=config.ATOL), (
                f"Betweenness should differ at {d}m (angular vs metric path choice)"
            )


def test_simplest_brandes_handles_zero_angle_plateaus():
    """Angular Brandes should stay smooth across straight zero-angle runs.

    This graph reproduces the old discontinuity where the upstream corridor
    segment received a large spike while the next straight segment collapsed.
    """
    dual_graph = graphs.nx_to_dual(make_angular_plateau_graph())
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    res_simplest = network_structure.centrality_simplest(
        compute_closeness=False,
        compute_betweenness=True,
        distances=[1000],
    )
    betw = {node_key: res_simplest.node_betweenness[1000][idx] for idx, node_key in enumerate(nodes_gdf.index)}
    ratio = betw["B_C_k0"] / betw["C_D_k0"]
    # On this topology the through-segments carry 18 and 10 ordered source-target
    # pairs respectively, so the stable corridor ratio should be 1.8 rather than
    # the old discontinuous spike.
    assert np.isclose(ratio, 1.8, atol=1e-6)
