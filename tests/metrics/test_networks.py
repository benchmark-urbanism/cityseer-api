# pyright: basic
from __future__ import annotations

import numpy as np
import pytest

from cityseer import config, rustalgos
from cityseer.metrics import networks
from cityseer.tools import io


def test_node_centrality_shortest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.node_centrality_shortest(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                compute_closeness=_closeness,
                compute_betweenness=_betweenness,
            )
            # test against underlying method
            node_result_short = network_structure.local_node_centrality_shortest(
                betas=betas, compute_closeness=_closeness, compute_betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    for measure_name in ["node_beta", "node_cycles", "node_density", "node_farness", "node_harmonic"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(
                            nodes_gdf[data_key],
                            getattr(node_result_short, measure_name)[distance],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_hillier_{distance}")],
                        node_result_short.node_density[distance] ** 2 / node_result_short.node_farness[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                if _betweenness is True:
                    for measure_name in ["node_betweenness", "node_betweenness_beta"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(
                            nodes_gdf[data_key],
                            getattr(node_result_short, measure_name)[distance],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )


def test_node_centrality_simplest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.node_centrality_simplest(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                compute_closeness=_closeness,
                compute_betweenness=_betweenness,
            )
            # test against underlying method
            node_result_simplest = network_structure.local_node_centrality_simplest(
                betas=betas, compute_closeness=_closeness, compute_betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_density_simplest_{distance}")],
                        node_result_simplest.node_density[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_farness_simplest_{distance}")],
                        node_result_simplest.node_farness[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_hillier_simplest_{distance}")],
                        node_result_simplest.node_density[distance] ** 2 / node_result_simplest.node_farness[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_harmonic_simplest_{distance}")],
                        node_result_simplest.node_harmonic[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                if _betweenness is True:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"node_betweenness_simplest_{distance}")],
                        node_result_simplest.node_betweenness[distance],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )


def test_segment_centrality(primal_graph):
    """
    Tests segment centrality. As of v4 this is only implemented for shortest path.
    """
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.segment_centrality(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                compute_closeness=_closeness,
                compute_betweenness=_betweenness,
            )
            # test against underlying method
            segment_result = network_structure.local_segment_centrality(
                betas=betas, compute_closeness=_closeness, compute_betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    for measure_name in ["segment_density", "segment_harmonic", "segment_beta"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(segment_result, measure_name)[distance])
                if _betweenness is True:
                    for measure_name in ["segment_betweenness"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(segment_result, measure_name)[distance])
