# pyright: basic
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config, rustalgos
from cityseer.algos import centrality
from cityseer.metrics import networks
from cityseer.tools import graphs


def test_node_centrality(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = rustalgos.distances_from_betas(betas)
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
    distances = rustalgos.distances_from_betas(betas)
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
