# pyright: basic
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config, structures
from cityseer.algos import centrality
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_distance_from_beta():
    # some basic checks using float form
    for b, d in zip([0.04, 0.0025, 0.0], [100, 1600, np.inf]):
        # simple straight check against corresponding distance
        assert networks.distance_from_beta(b) == np.array([d])
        # circular check
        assert networks.beta_from_distance(networks.distance_from_beta(b)) == b
        # array form check
        assert networks.distance_from_beta(np.array([b])) == np.array([d])
    # check that custom min_threshold_wt works
    arr = networks.distance_from_beta(0.04, min_threshold_wt=0.001)
    assert np.allclose(arr, np.array([172.69388197455342]), atol=config.ATOL, rtol=config.RTOL)
    # check on array form
    arr = networks.distance_from_beta([0.04, 0.0025, 0.0])
    assert np.allclose(arr, np.array([100, 1600, np.inf]), atol=config.ATOL, rtol=config.RTOL)
    # check for type error
    with pytest.raises(TypeError):
        networks.distance_from_beta("boo")
    # check that invalid beta values raise an error
    # positive integer of zero should raise, but not positive float
    assert networks.distance_from_beta(0.0) == np.inf  # this should not raise
    for b in [None]:
        with pytest.raises(TypeError):
            networks.distance_from_beta(b)
    for b in [-0.04, 0, -0, -0.0, []]:
        with pytest.raises(ValueError):
            networks.distance_from_beta(b)


def test_beta_from_distance():
    # some basic checks
    for dist, b in zip([100, 1600, np.inf], [0.04, 0.0025, 0.0]):
        # simple straight check against corresponding distance
        assert np.allclose(networks.beta_from_distance(dist), np.array([b]), atol=config.ATOL, rtol=config.RTOL)
        # circular check
        assert networks.distance_from_beta(networks.beta_from_distance(dist)) == dist
        # array form check
        assert np.allclose(
            networks.beta_from_distance(np.array([dist])),
            np.array([b]),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # check that custom min_threshold_wt works
    arr = networks.beta_from_distance(172.69388197455342, min_threshold_wt=0.001)
    assert np.allclose(arr, np.array([0.04]), atol=config.ATOL, rtol=config.RTOL)
    # check on array form
    arr = networks.beta_from_distance([100, 1600, np.inf])
    assert np.allclose(arr, np.array([0.04, 0.0025, 0.0]), atol=config.ATOL, rtol=config.RTOL)
    # check for type error
    with pytest.raises(TypeError):
        networks.beta_from_distance("boo")
    # check that invalid distance values raise an error
    for dist in [-100, 0, []]:
        with pytest.raises(ValueError):
            networks.beta_from_distance(dist)
    for dist in [None]:
        with pytest.raises(TypeError):
            networks.beta_from_distance(dist)


def test_Network_Layer(primal_graph):
    # manual graph maps for comparison
    node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    betas = [0.02, 0.005]
    distances = networks.distance_from_beta(betas)

    # test NetworkLayer's class
    for d, b in zip([distances, None], [None, betas]):
        cc_netw = networks.NetworkLayer(node_keys, network_structure, distances=d, betas=b)
        assert np.allclose(cc_netw.node_keys, node_keys, atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(
            cc_netw.distances, distances, atol=config.ATOL, rtol=config.RTOL
        )  # inferred automatically when only betas provided
        assert np.allclose(
            cc_netw.betas, betas, atol=config.ATOL, rtol=config.RTOL
        )  # inferred automatically when only distances provided
        assert cc_netw._min_threshold_wt == config.MIN_THRESH_WT

    # test round-trip graph to and from NetworkLayer
    cc_netw = networks.NetworkLayer(node_keys, network_structure, distances=distances)
    G_round_trip = cc_netw.to_nx_multigraph()
    # network_structure_from_networkX generates implicit live (all True) and weight (all 1) attributes if missing
    # i.e. can't simply check that all nodes equal, so check properties manually
    for node_key, node_data in primal_graph.nodes(data=True):
        assert node_key in G_round_trip
        assert G_round_trip.nodes[node_key]["x"] == node_data["x"]
        assert G_round_trip.nodes[node_key]["y"] == node_data["y"]
    # edges can be checked en masse
    assert G_round_trip.edges == primal_graph.edges
    # check alternate min_threshold_wt gets passed through successfully
    alt_min = 0.02
    alt_distances = networks.distance_from_beta(betas, min_threshold_wt=alt_min)
    cc_netw = networks.NetworkLayer(
        node_keys,
        network_structure,
        betas=betas,
        min_threshold_wt=alt_min,
    )
    assert np.allclose(cc_netw.distances, alt_distances, atol=config.ATOL, rtol=config.RTOL)
    # check for malformed signatures
    with pytest.raises(ValueError):
        networks.NetworkLayer(node_keys[:-1], network_structure, distances)
    with pytest.raises(AttributeError):
        networks.NetworkLayer(node_keys, None, distances)  # type: ignore
    with pytest.raises(ValueError):
        networks.NetworkLayer(node_keys, network_structure)  # no betas or distances
    with pytest.raises(ValueError):
        networks.NetworkLayer(node_keys, network_structure, distances=None, betas=None)
    with pytest.raises(ValueError):
        networks.NetworkLayer(node_keys, network_structure, distances=[])
    with pytest.raises(ValueError):
        networks.NetworkLayer(node_keys, network_structure, betas=[])


def test_Network_Layer_From_nx(primal_graph):
    node_keys, outer_network_structure = graphs.network_structure_from_nx(primal_graph)
    betas: npt.NDArray[np.float32] = np.array([0.04, 0.02], dtype=np.float32)
    distances = networks.distance_from_beta(betas)

    # test Network_Layer_From_NetworkX's class
    for dist, beta in zip([distances, None], [None, betas]):
        cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=dist, betas=beta)
        assert np.allclose(cc_netw.node_keys, node_keys, atol=config.ATOL, rtol=config.RTOL)
        assert np.allclose(
            cc_netw.network_structure.nodes.xs, outer_network_structure.nodes.xs, atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            cc_netw.network_structure.nodes.ys, outer_network_structure.nodes.ys, atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            cc_netw.network_structure.edges.start,
            outer_network_structure.edges.start,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.network_structure.edges.end, outer_network_structure.edges.end, atol=config.ATOL, rtol=config.RTOL
        )
        assert np.allclose(
            cc_netw.network_structure.edges.length,
            outer_network_structure.edges.length,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.network_structure.edges.angle_sum,
            outer_network_structure.edges.angle_sum,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.network_structure.edges.imp_factor,
            outer_network_structure.edges.imp_factor,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.network_structure.edges.in_bearing,
            outer_network_structure.edges.in_bearing,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.network_structure.edges.out_bearing,
            outer_network_structure.edges.out_bearing,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.distances, distances, atol=config.ATOL, rtol=config.RTOL
        )  # inferred automatically when only betas provided
        assert np.allclose(
            cc_netw.betas, betas, atol=config.ATOL, rtol=config.RTOL
        )  # inferred automatically when only distances provided
        assert networks.MIN_THRESH_WT == config.MIN_THRESH_WT

    # check alternate min_threshold_wt gets passed through successfully
    alt_min = 0.02
    alt_distances = networks.distance_from_beta(betas, min_threshold_wt=alt_min)
    cc_netw = networks.NetworkLayerFromNX(primal_graph, betas=betas, min_threshold_wt=alt_min)
    assert np.allclose(cc_netw.distances, alt_distances, atol=config.ATOL, rtol=config.RTOL)

    # check for malformed signatures
    with pytest.raises(TypeError):
        networks.NetworkLayerFromNX("boo", distances=distances)
    with pytest.raises(ValueError):
        networks.NetworkLayerFromNX(primal_graph)  # no betas or distances
    with pytest.raises(ValueError):
        networks.NetworkLayerFromNX(primal_graph, distances=None, betas=None)
    with pytest.raises(ValueError):
        networks.NetworkLayerFromNX(primal_graph, distances=[])
    with pytest.raises(ValueError):
        networks.NetworkLayerFromNX(primal_graph, betas=[])


def dict_check(dict_node_metrics: structures.DictNodeMetrics, Network: networks.NetworkLayer):
    for i, node_key in enumerate(Network.node_keys):
        # general
        assert dict_node_metrics[node_key]["x"] == Network.network_structure.nodes.xs[i]
        assert dict_node_metrics[node_key]["y"] == Network.network_structure.nodes.ys[i]
        assert dict_node_metrics[node_key]["live"] == Network.network_structure.nodes.live[i]
        # centrality
        for c_key, c_val in Network.metrics_state.centrality.items():
            for d_key, d_val in c_val.items():
                assert d_val[i] == dict_node_metrics[node_key]["centrality"][c_key][d_key]
        # mixed uses
        for mu_key, mu_val in Network.metrics_state.mixed_uses.items():
            if "hill" in mu_key:
                for q_key, q_val in mu_val.items():
                    for d_key, d_val in q_val.items():
                        assert d_val[i] == dict_node_metrics[node_key]["mixed_uses"][mu_key][q_key][d_key]
            else:
                for d_key, d_val in mu_val.items():
                    assert d_val[i] == dict_node_metrics[node_key]["mixed_uses"][mu_key][d_key]
        # accessibility
        for cl_key, cl_val in Network.metrics_state.accessibility.weighted.items():
            for d_key, d_val in cl_val.items():
                assert d_val[i] == dict_node_metrics[node_key]["accessibility"]["weighted"][cl_key][d_key]
        for cl_key, cl_val in Network.metrics_state.accessibility.non_weighted.items():
            for d_key, d_val in cl_val.items():
                assert d_val[i] == dict_node_metrics[node_key]["accessibility"]["non_weighted"][cl_key][d_key]
        # stats
        for th_key, th_val in Network.metrics_state.stats.items():
            for stat_key, stat_val in th_val.items():
                for d_key, d_val in stat_val.items():
                    # some NaN so use np.allclose
                    assert np.allclose(
                        d_val[i],
                        dict_node_metrics[node_key]["stats"][th_key][stat_key][d_key],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )


def test_metrics_to_dict(primal_graph):
    # create a network layer and run some metrics
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=[500, 1000])
    # check with no metrics
    dict_metrics = cc_netw.metrics_to_dict()
    dict_check(dict_metrics, cc_netw)
    # check with data
    cc_netw.node_centrality(measures=["node_harmonic"])
    data_dict = mock.mock_data_dict(primal_graph)
    cat_vals = mock.mock_categorical_data(len(data_dict))
    cc_data = layers.DataLayerFromDict(data_dict)
    cc_data.assign_to_network(cc_netw, max_dist=500)
    cc_data.compute_landuses(
        landuse_labels=cat_vals, mixed_use_keys=["hill", "shannon"], qs=[0], accessibility_keys=tuple(set(cat_vals[:4]))
    )
    num_vals = mock.mock_numerical_data(len(data_dict))
    cc_data.compute_stats(stats_keys="boo", stats_data=num_vals)
    dict_metrics = cc_netw.metrics_to_dict()
    dict_check(dict_metrics, cc_netw)


def test_to_nx_multigraph(primal_graph):
    # also see test_graphs.test_networkX_from_network_structure for underlying graph maps version
    # check round trip to and from graph maps results in same graph
    # explicitly set live and weight params for equality checks
    # network_structure_from_networkX generates these implicitly if missing
    G = graphs.nx_decompose(primal_graph, decompose_max=20)
    for node_key in G.nodes():
        G.nodes[node_key]["live"] = bool(np.random.randint(0, 2))
    for start_node_key, end_node_key, edge_key in G.edges(keys=True):
        G[start_node_key][end_node_key][edge_key]["imp_factor"] = np.random.randint(0, 2)
    # add random data to check persistence at other end
    baa_node = None
    for node_key in G.nodes():
        baa_node = node_key
        G.nodes[node_key]["boo"] = "baa"
        break
    boo_edge = None
    for start_node_key, end_node_key, edge_key in G.edges(keys=True):
        boo_edge = (start_node_key, end_node_key)
        G[start_node_key][end_node_key][edge_key]["baa"] = "boo"
        break
    # test with metrics
    cc_netw = networks.NetworkLayerFromNX(G, distances=[500])
    cc_netw.node_centrality(measures=["node_harmonic"])
    metrics_dict = cc_netw.metrics_to_dict()
    G_round_trip = cc_netw.to_nx_multigraph()
    for node_key, node_data in G.nodes(data=True):
        assert np.isclose(G_round_trip.nodes[node_key]["x"], node_data["x"], atol=config.ATOL, rtol=config.RTOL)
        assert np.isclose(G_round_trip.nodes[node_key]["y"], node_data["y"], atol=config.ATOL, rtol=config.RTOL)
        assert np.isclose(G_round_trip.nodes[node_key]["live"], node_data["live"], atol=config.ATOL, rtol=config.RTOL)
    for start_node_key, end_node_key, edge_key, node_data in G.edges(keys=True, data=True):
        assert G_round_trip[start_node_key][end_node_key][edge_key]["geom"] == node_data["geom"]
        assert np.isclose(
            G_round_trip[start_node_key][end_node_key][edge_key]["imp_factor"],
            node_data["imp_factor"],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # check that metrics came through
    for node_key, metrics in metrics_dict.items():
        assert G_round_trip.nodes[node_key]["metrics"] == metrics
    # check data persistence
    assert G_round_trip.nodes[baa_node]["boo"] == "baa"
    assert G_round_trip[boo_edge[0]][boo_edge[1]][0]["baa"] == "boo"


def test_compute_centrality(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    # generate data structures
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    # CHECK NODE BASED
    node_measures = [
        "node_density",
        "node_farness",
        "node_cycles",
        "node_harmonic",
        "node_beta",
        "node_betweenness",
        "node_betweenness_beta",
    ]
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    # check measures against underlying method
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    cc_netw.node_centrality(measures=["node_density"])
    # test against underlying method
    measures_data = centrality.local_node_centrality(
        cc_netw.network_structure,
        distances,
        betas,
        measure_keys=("node_density",),
    )
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(cc_netw.metrics_state.centrality["node_density"][d_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(node_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys: npt.NDArray[np.int_] = np.array(node_measures[min_idx:])
        cc_netw: networks.NetworkLayer = networks.NetworkLayerFromNX(primal_graph, distances=distances)
        cc_netw.node_centrality(measures=node_measures)
        # test against underlying method
        measures_data = centrality.local_node_centrality(
            cc_netw.network_structure,
            distances,
            betas,
            measure_keys=tuple(measure_keys),
        )
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                assert np.allclose(
                    cc_netw.metrics_state.centrality[measure_name][d_key],
                    measures_data[m_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
    # check that angular gets passed through
    cc_netw_ang = networks.NetworkLayerFromNX(primal_graph, distances=[2000])
    cc_netw_ang.node_centrality(measures=["node_harmonic_angular"], angular=True)
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=[2000])
    cc_netw.node_centrality(measures=["node_harmonic"], angular=False)
    assert not np.allclose(
        cc_netw_ang.metrics_state.centrality["node_harmonic_angular"][2000],
        cc_netw.metrics_state.centrality["node_harmonic"][2000],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        cc_netw.node_centrality(measures=["spelling_typo"])
    with pytest.raises(ValueError):
        cc_netw.node_centrality(measures=["node_density", "node_density"])
    with pytest.raises(ValueError):
        cc_netw.node_centrality(measures=["node_density", "node_harmonic_angular"])

    # CHECK SEGMENTISED
    segment_measures = [
        "segment_density",
        "segment_harmonic",
        "segment_beta",
        "segment_betweenness",
    ]
    # segment_measures_ang = ["segment_harmonic_hybrid", "segment_betweeness_hybrid"]

    # check measures against underlying method
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    cc_netw.segment_centrality(measures=["segment_density"])
    # test against underlying method
    measures_data = centrality.local_segment_centrality(
        cc_netw.network_structure,
        distances,
        betas,
        measure_keys=("segment_density",),
    )
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(cc_netw.metrics_state.centrality["segment_density"][d_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(segment_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys = np.array(segment_measures[min_idx:])
        cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=distances)
        cc_netw.segment_centrality(measures=segment_measures)
        # test against underlying method
        measures_data = centrality.local_segment_centrality(
            cc_netw.network_structure,
            distances,
            betas,
            measure_keys=tuple(measure_keys),
        )
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                assert np.allclose(
                    cc_netw.metrics_state.centrality[measure_name][d_key],
                    measures_data[m_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
    # check that angular gets passed through
    cc_netw_ang = networks.NetworkLayerFromNX(primal_graph, distances=[2000])
    cc_netw_ang.segment_centrality(measures=["segment_harmonic_hybrid"], angular=True)
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=[2000])
    cc_netw.segment_centrality(measures=["segment_harmonic"], angular=False)
    assert not np.allclose(
        cc_netw_ang.metrics_state.centrality["segment_harmonic_hybrid"][2000],
        cc_netw.metrics_state.centrality["segment_harmonic"][2000],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        cc_netw.segment_centrality(measures=["spelling_typo"])
    with pytest.raises(ValueError):
        cc_netw.segment_centrality(measures=["segment_density", "segment_density"])
    with pytest.raises(ValueError):
        cc_netw.segment_centrality(measures=["segment_density", "segment_harmonic_hybrid"])
