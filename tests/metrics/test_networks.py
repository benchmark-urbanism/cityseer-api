# pyright: basic
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config
from cityseer.algos import centrality
from cityseer.metrics import networks
from cityseer.tools import graphs


def test_distance_from_beta():
    # some basic checks using float form
    for b, d in zip([0.04, 0.0025], [100, 1600]):
        # simple straight check against corresponding distance
        assert networks.distance_from_beta(b) == np.array([d])
        # circular check
        assert networks.beta_from_distance(networks.distance_from_beta(b)) == b
        # array form check
        assert networks.distance_from_beta(np.array([b])) == np.array([d])
    # check that custom min_threshold_wt works
    arr = networks.distance_from_beta(0.04, min_threshold_wt=0.001)
    assert np.allclose(arr, 172, atol=config.ATOL, rtol=config.RTOL)
    # check on array form
    arr = networks.distance_from_beta([0.04, 0.0025])
    assert np.allclose(arr, [100, 1600], atol=config.ATOL, rtol=config.RTOL)
    # check for type error
    with pytest.raises(TypeError):
        networks.distance_from_beta("boo")
    # check that invalid beta values raise an error
    for b in [None]:
        with pytest.raises(TypeError):
            networks.distance_from_beta(b)
    for b in [-0.04, 0, -0, -0.0, 0.0, []]:
        with pytest.raises(ValueError):
            networks.distance_from_beta(b)


def test_beta_from_distance():
    # some basic checks
    for dist, b in zip([100, 1600], [0.04, 0.0025]):
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
    arr = networks.beta_from_distance([100, 1600])
    assert np.allclose(arr, np.array([0.04, 0.0025]), atol=config.ATOL, rtol=config.RTOL)
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


def test_avg_distance_for_beta():
    betas = networks.beta_from_distance([100, 200, 400, 800, 1600])
    assert np.allclose(
        networks.avg_distance_for_beta(betas),
        [35.11949, 70.23898, 140.47797, 280.95593, 561.91187],
        atol=config.ATOL,
        rtol=config.RTOL,
    )


def test_pair_distances_betas():
    raw_distances = [100, 200, 400, 800, 1600]
    _distances, _betas = networks.pair_distances_betas(distances=raw_distances)
    assert np.allclose(_betas, networks.beta_from_distance(_distances), atol=config.ATOL, rtol=config.RTOL)
    betas = networks.beta_from_distance([100, 200, 400, 800, 1600])
    _distances, _betas = networks.pair_distances_betas(betas=betas)
    assert np.allclose(_distances, raw_distances, atol=config.ATOL, rtol=config.RTOL)


def test_clip_weights_curve():
    raw_distances = [400, 800, 1600]
    distances, betas = networks.pair_distances_betas(distances=raw_distances)
    # should be 1 if dist buffer is zero
    max_curve_wts = networks.clip_weights_curve(distances, betas, 0)
    assert np.allclose([1, 1, 1], max_curve_wts, atol=config.ATOL, rtol=config.RTOL)
    # check for a random distance
    max_curve_wts = networks.clip_weights_curve(distances, betas, 50)
    assert np.allclose([0.60653067, 0.7788008, 0.8824969], max_curve_wts, atol=config.ATOL, rtol=config.RTOL)
    # should raise if buffer_distance is less than zero
    with pytest.raises(ValueError):
        max_curve_wts = networks.clip_weights_curve(distances, betas, -1)
    # should raise if buffer_distance is greater than distances
    with pytest.raises(ValueError):
        max_curve_wts = networks.clip_weights_curve(distances, betas, 401)


def test_node_centrality(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
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
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    nodes_gdf = networks.node_centrality(
        measures=["node_density"], network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
    )
    # test against underlying method
    measures_data = centrality.local_node_centrality(
        distances,
        betas,
        ("node_density",),
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
    for d_idx, d_key in enumerate(distances):
        data_key = config.prep_gdf_key(f"node_density_{d_key}")
        assert np.allclose(nodes_gdf[data_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(node_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys: npt.NDArray[np.int_] = np.array(node_measures[min_idx:])
        networks.node_centrality(
            measures=node_measures, network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
        )
        # test against underlying method
        measures_data = centrality.local_node_centrality(
            distances,
            betas,
            tuple(measure_keys),
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
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                data_key = config.prep_gdf_key(f"{measure_name}_{d_key}")
                assert np.allclose(
                    nodes_gdf[data_key],
                    measures_data[m_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
    # check that angular gets passed through
    nodes_gdf_angular = networks.node_centrality(
        measures=["node_harmonic_angular"],
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        betas=betas,
        angular=True,
    )
    nodes_gdf = networks.node_centrality(
        measures=["node_harmonic"],
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        betas=betas,
        angular=False,
    )
    data_key_angular = config.prep_gdf_key(f"node_harmonic_angular_800")
    data_key = config.prep_gdf_key(f"node_harmonic_800")
    assert not np.allclose(
        nodes_gdf_angular[data_key_angular],
        nodes_gdf[data_key],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        nodes_gdf = networks.node_centrality(
            measures=["spelling_typo"], network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
        )
    with pytest.raises(ValueError):
        nodes_gdf = networks.node_centrality(
            measures=["node_density", "node_density"],
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            betas=betas,
        )
    with pytest.raises(ValueError):
        nodes_gdf = networks.node_centrality(
            measures=["node_density", "node_harmonic_angular"],
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            betas=betas,
        )


def test_segment_centrality(primal_graph):
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    segment_measures = [
        "segment_density",
        "segment_harmonic",
        "segment_beta",
        "segment_betweenness",
    ]
    # segment_measures_ang = ["segment_harmonic_hybrid", "segment_betweeness_hybrid"]
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    nodes_gdf = networks.segment_centrality(
        measures=["segment_density"], network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
    )
    # test against underlying method
    measures_data = centrality.local_segment_centrality(
        distances,
        betas,
        ("segment_density",),
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
    for d_idx, d_key in enumerate(distances):
        data_key = config.prep_gdf_key(f"segment_density_{d_key}")
        assert np.allclose(nodes_gdf[data_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(segment_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys: npt.NDArray[np.int_] = np.array(segment_measures[min_idx:])
        networks.segment_centrality(
            measures=segment_measures, network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
        )
        # test against underlying method
        measures_data = centrality.local_segment_centrality(
            distances,
            betas,
            tuple(measure_keys),
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
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                data_key = config.prep_gdf_key(f"{measure_name}_{d_key}")
                assert np.allclose(
                    nodes_gdf[data_key],
                    measures_data[m_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
    # check that angular gets passed through
    nodes_gdf_angular = networks.segment_centrality(
        measures=["segment_harmonic_hybrid"],
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        betas=betas,
        angular=True,
    )
    nodes_gdf = networks.segment_centrality(
        measures=["segment_harmonic"],
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        betas=betas,
        angular=False,
    )
    data_key_angular = config.prep_gdf_key(f"segment_harmonic_hybrid_800")
    data_key = config.prep_gdf_key(f"segment_harmonic_800")
    assert not np.allclose(
        nodes_gdf_angular[data_key_angular],
        nodes_gdf[data_key],
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        nodes_gdf = networks.segment_centrality(
            measures=["spelling_typo"], network_structure=network_structure, nodes_gdf=nodes_gdf, betas=betas
        )
    with pytest.raises(ValueError):
        nodes_gdf = networks.segment_centrality(
            measures=["segment_harmonic", "segment_harmonic"],
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            betas=betas,
        )
    with pytest.raises(ValueError):
        nodes_gdf = networks.segment_centrality(
            measures=["segment_harmonic", "segment_harmonic_hybrid"],
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            betas=betas,
        )
