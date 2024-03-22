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
            for dist_key in distances:
                if _closeness is True:
                    for measure_key, attr_key in [
                        ("beta", "node_beta"),
                        ("cycles", "node_cycles"),
                        ("density", "node_density"),
                        ("farness", "node_farness"),
                        ("harmonic", "node_harmonic"),
                    ]:
                        data_key = config.prep_gdf_key(measure_key, dist_key)
                        assert np.allclose(
                            nodes_gdf[data_key],
                            getattr(node_result_short, attr_key)[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key("hillier", dist_key)],
                        node_result_short.node_density[dist_key] ** 2 / node_result_short.node_farness[dist_key],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                if _betweenness is True:
                    for measure_key, attr_key in [
                        ("betweenness", "node_betweenness"),
                        ("betweenness_beta", "node_betweenness_beta"),
                    ]:
                        data_key = config.prep_gdf_key(measure_key, dist_key)
                        assert np.allclose(
                            nodes_gdf[data_key],
                            getattr(node_result_short, attr_key)[dist_key],
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
            for dist_key in distances:
                if _closeness is True:
                    for measure_key, attr_key in [
                        ("density", "node_density"),
                        ("farness", "node_farness"),
                        ("harmonic", "node_harmonic"),
                    ]:
                        assert np.allclose(
                            nodes_gdf[config.prep_gdf_key(measure_key, dist_key, angular=True)],
                            getattr(node_result_simplest, attr_key)[dist_key],
                            equal_nan=True,
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key("hillier", dist_key, angular=True)],
                        node_result_simplest.node_density[dist_key] ** 2 / node_result_simplest.node_farness[dist_key],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                if _betweenness is True:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key("betweenness", dist_key, angular=True)],
                        node_result_simplest.node_betweenness[dist_key],
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
            for dist_key in distances:
                if _closeness is True:
                    for measure_key, attr_key in [
                        ("seg_density", "segment_density"),
                        ("seg_harmonic", "segment_harmonic"),
                        ("seg_beta", "segment_beta"),
                    ]:
                        data_key = config.prep_gdf_key(measure_key, dist_key)
                        assert np.allclose(nodes_gdf[data_key], getattr(segment_result, attr_key)[dist_key])
                if _betweenness is True:
                    data_key = config.prep_gdf_key("seg_betweenness", dist_key)
                    assert np.allclose(nodes_gdf[data_key], getattr(segment_result, "segment_betweenness")[dist_key])
