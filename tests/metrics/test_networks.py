# pyright: basic
from __future__ import annotations

import numpy as np
from cityseer import config, rustalgos
from cityseer.metrics import networks
from cityseer.tools import io


def test_node_centrality_shortest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    seed = 42
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
                random_seed=seed,
            )
            for dist_key in distances:
                if _closeness is True:
                    # test closeness against underlying source-sampling method
                    node_result_short = network_structure.closeness_shortest(
                        betas=betas,
                    )
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
                    # test betweenness against underlying R-K path sampling method
                    betweenness_result = network_structure.betweenness_shortest(
                        betas=betas, random_seed=seed
                    )
                    for measure_key, attr_key in [
                        ("betweenness", "node_betweenness"),
                        ("betweenness_beta", "node_betweenness_beta"),
                    ]:
                        data_key = config.prep_gdf_key(measure_key, dist_key)
                        assert np.allclose(
                            nodes_gdf[data_key],
                            getattr(betweenness_result, attr_key)[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )


def test_node_centrality_simplest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    seed = 42
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            for _far_scale_off, _ang_scale_unit in [(0, 180), (0, 90), (1, 180)]:
                nodes_gdf = networks.node_centrality_simplest(
                    network_structure=network_structure,
                    nodes_gdf=nodes_gdf,
                    distances=_distances,
                    betas=_betas,
                    compute_closeness=_closeness,
                    compute_betweenness=_betweenness,
                    farness_scaling_offset=_far_scale_off,
                    angular_scaling_unit=_ang_scale_unit,
                    random_seed=seed,
                )
                for dist_key in distances:
                    if _closeness is True:
                        # test closeness against underlying source-sampling method
                        node_result_simplest = network_structure.closeness_simplest(
                            betas=betas,
                            farness_scaling_offset=_far_scale_off,
                            angular_scaling_unit=_ang_scale_unit,
                        )
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
                            node_result_simplest.node_density[dist_key] ** 2
                            / node_result_simplest.node_farness[dist_key],
                            equal_nan=True,
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                    if _betweenness is True:
                        # test betweenness against underlying R-K path sampling method
                        betweenness_result = network_structure.betweenness_simplest(
                            betas=betas, random_seed=seed
                        )
                        assert np.allclose(
                            nodes_gdf[config.prep_gdf_key("betweenness", dist_key, angular=True)],
                            betweenness_result.node_betweenness[dist_key],
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
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
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
            segment_result = network_structure.segment_centrality(
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
                    assert np.allclose(nodes_gdf[data_key], segment_result.segment_betweenness[dist_key])


def test_closeness_shortest(primal_graph):
    """Test standalone closeness_shortest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.closeness_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
        probe_density=20.0,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("beta", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("cycles", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist) in nodes_gdf_result.columns


def test_closeness_simplest(primal_graph):
    """Test standalone closeness_simplest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.closeness_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
        probe_density=20.0,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist, angular=True) in nodes_gdf_result.columns


def test_betweenness_shortest(primal_graph):
    """Test standalone betweenness_shortest with R-K sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.betweenness_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("betweenness", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("betweenness_beta", dist) in nodes_gdf_result.columns


def test_betweenness_simplest(primal_graph):
    """Test standalone betweenness_simplest with R-K sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.betweenness_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("betweenness", dist, angular=True) in nodes_gdf_result.columns
