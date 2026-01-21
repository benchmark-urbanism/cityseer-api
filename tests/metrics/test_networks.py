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
    # node_measures_ang = ["node_harmonic_angular", "node_betweenness_angular"]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
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
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        for _distances, _betas in [(distances, None), (None, betas)]:
            for _far_scale_off, _ang_scale_unit in [(0, 180), (0, 90), (1, 180)]:
                # test shortest
                nodes_gdf = networks.node_centrality_simplest(
                    network_structure=network_structure,
                    nodes_gdf=nodes_gdf,
                    distances=_distances,
                    betas=_betas,
                    compute_closeness=_closeness,
                    compute_betweenness=_betweenness,
                    farness_scaling_offset=_far_scale_off,
                    angular_scaling_unit=_ang_scale_unit,
                )
                # test against underlying method
                node_result_simplest = network_structure.local_node_centrality_simplest(
                    betas=betas,
                    compute_closeness=_closeness,
                    compute_betweenness=_betweenness,
                    farness_scaling_offset=_far_scale_off,
                    angular_scaling_unit=_ang_scale_unit,
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
                            node_result_simplest.node_density[dist_key] ** 2
                            / node_result_simplest.node_farness[dist_key],
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
                    assert np.allclose(nodes_gdf[data_key], segment_result.segment_betweenness[dist_key])


def test_node_centrality_shortest_adaptive(primal_graph):
    """
    Test adaptive shortest-path centrality with per-distance sampling.

    The adaptive function should produce results that correlate highly with
    full computation, while potentially using sampling at larger distances.
    """
    from scipy.stats import spearmanr

    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)

    # Run adaptive version with high target accuracy
    nodes_gdf_adaptive = networks.node_centrality_shortest_adaptive(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        target_rho=0.95,
        compute_closeness=True,
        compute_betweenness=True,
        random_seed=42,
        n_probes=20,  # Smaller for test speed
    )

    # Run full computation for comparison
    nodes_gdf_full = networks.node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
    )

    # Check that columns were created
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist) in nodes_gdf_adaptive.columns
        assert config.prep_gdf_key("betweenness", dist) in nodes_gdf_adaptive.columns

    # Check correlation with full computation
    # For this small test graph, adaptive should match very closely
    for dist in distances:
        harmonic_key = config.prep_gdf_key("harmonic", dist)
        full_vals = nodes_gdf_full[harmonic_key].values
        adaptive_vals = nodes_gdf_adaptive[harmonic_key].values

        # Filter out zeros/nans for correlation
        mask = (full_vals > 0) & np.isfinite(full_vals) & np.isfinite(adaptive_vals)
        if mask.sum() > 5:
            rho, _ = spearmanr(full_vals[mask], adaptive_vals[mask])
            # Should achieve at least the target accuracy
            assert rho >= 0.85, f"Correlation too low at {dist}m: {rho:.3f}"


def test_node_centrality_simplest_adaptive(primal_graph):
    """
    Test adaptive simplest-path (angular) centrality with per-distance sampling.
    """
    from scipy.stats import spearmanr

    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)

    # Run adaptive version
    nodes_gdf_adaptive = networks.node_centrality_simplest_adaptive(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        target_rho=0.95,
        compute_closeness=True,
        compute_betweenness=True,
        random_seed=42,
        n_probes=20,
    )

    # Run full computation for comparison
    nodes_gdf_full = networks.node_centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
    )

    # Check that columns were created
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist, angular=True) in nodes_gdf_adaptive.columns
        assert config.prep_gdf_key("betweenness", dist, angular=True) in nodes_gdf_adaptive.columns

    # Check correlation with full computation
    for dist in distances:
        harmonic_key = config.prep_gdf_key("harmonic", dist, angular=True)
        full_vals = nodes_gdf_full[harmonic_key].values
        adaptive_vals = nodes_gdf_adaptive[harmonic_key].values

        mask = (full_vals > 0) & np.isfinite(full_vals) & np.isfinite(adaptive_vals)
        if mask.sum() > 5:
            rho, _ = spearmanr(full_vals[mask], adaptive_vals[mask])
            assert rho >= 0.85, f"Correlation too low at {dist}m: {rho:.3f}"
