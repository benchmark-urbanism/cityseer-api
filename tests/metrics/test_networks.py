# pyright: basic
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config, rustalgos
from cityseer.metrics import networks
from cityseer.tools import graphs


def test_node_centrality_shortest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = rustalgos.distances_from_betas(betas)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.node_centrality_shortest(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                closeness=_closeness,
                betweenness=_betweenness,
            )
            # test against underlying method
            close_result, betw_result = network_structure.local_node_centrality_shortest(
                betas=betas, closeness=_closeness, betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    for measure_name in ["node_beta", "node_cycles", "node_density", "node_farness", "node_harmonic"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(close_result, measure_name)[distance])
                if _betweenness is True:
                    for measure_name in ["node_betweenness", "node_betweenness_beta"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(betw_result, measure_name)[distance])


def test_node_centrality_simplest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = rustalgos.distances_from_betas(betas)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.node_centrality_simplest(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                closeness=_closeness,
                betweenness=_betweenness,
            )
            # test against underlying method
            close_result, betw_result = network_structure.local_node_centrality_simplest(
                betas=betas, closeness=_closeness, betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    for measure_name in ["node_harmonic"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(close_result, measure_name)[distance])
                if _betweenness is True:
                    for measure_name in ["node_betweenness"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(betw_result, measure_name)[distance])


def test_segment_centrality(primal_graph):
    """
    Tests segment centrality. As of v4 this is only implemented for shortest path.
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = rustalgos.distances_from_betas(betas)
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            # test shortest
            nodes_gdf = networks.segment_centrality(
                network_structure=network_structure,
                nodes_gdf=nodes_gdf,
                distances=_distances,
                betas=_betas,
                closeness=_closeness,
                betweenness=_betweenness,
            )
            # test against underlying method
            close_result, betw_result = network_structure.local_segment_centrality_shortest(
                betas=betas, closeness=_closeness, betweenness=_betweenness
            )
            for distance in distances:
                if _closeness is True:
                    for measure_name in ["segment_density", "segment_harmonic", "segment_beta"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(close_result, measure_name)[distance])
                if _betweenness is True:
                    for measure_name in ["segment_betweenness"]:
                        data_key = config.prep_gdf_key(f"{measure_name}_{distance}")
                        assert np.allclose(nodes_gdf[data_key], getattr(betw_result, measure_name)[distance])
