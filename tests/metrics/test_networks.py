# pyright: basic
from __future__ import annotations

import numpy as np
import pytest
from cityseer import config
from cityseer.metrics import networks
from cityseer.tools import io


def test_centrality_shortest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        nodes_gdf = networks.centrality_shortest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            distances=distances,
            compute_closeness=_closeness,
            compute_betweenness=_betweenness,
        )
        for dist_key in distances:
            if _closeness is True:
                # test closeness against underlying source-sampling method
                node_result_short = network_structure.centrality_shortest(
                    compute_closeness=True,
                    compute_betweenness=False,
                    distances=distances,
                )
                for measure_key, attr_key in [
                    ("decay", "node_decay"),
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
                with np.errstate(divide="ignore", invalid="ignore"):
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key("hillier", dist_key)],
                        node_result_short.node_density[dist_key] ** 2 / node_result_short.node_farness[dist_key],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
            if _betweenness is True:
                # test betweenness against underlying exact Brandes method
                betweenness_result = network_structure.centrality_shortest(
                    compute_closeness=False,
                    compute_betweenness=True,
                    distances=[dist_key],
                )
                for measure_key, attr_key in [
                    ("betweenness", "node_betweenness"),
                    ("betweenness_decay", "node_betweenness_decay"),
                ]:
                    data_key = config.prep_gdf_key(measure_key, dist_key)
                    assert np.allclose(
                        nodes_gdf[data_key],
                        getattr(betweenness_result, attr_key)[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )


def test_centrality_simplest(dual_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    for _far_scale_off, _ang_scale_unit in [(0, 180), (0, 90), (1, 180)]:
        nodes_gdf = networks.centrality_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            distances=distances,
            compute_closeness=True,
            compute_betweenness=True,
            farness_scaling_offset=_far_scale_off,
            angular_scaling_unit=_ang_scale_unit,
        )
        for dist_key in distances:
            # test closeness against underlying method
            node_result_simplest = network_structure.centrality_simplest(
                compute_closeness=True,
                compute_betweenness=False,
                distances=distances,
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
            with np.errstate(divide="ignore", invalid="ignore"):
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key("hillier", dist_key, angular=True)],
                    node_result_simplest.node_density[dist_key] ** 2 / node_result_simplest.node_farness[dist_key],
                    equal_nan=True,
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
            # test betweenness against underlying method
            betw_result = network_structure.centrality_simplest(
                compute_closeness=False, compute_betweenness=True, distances=distances
            )
            assert np.allclose(
                nodes_gdf[config.prep_gdf_key("betweenness", dist_key, angular=True)],
                betw_result.node_betweenness[dist_key],
                equal_nan=True,
                atol=config.ATOL,
                rtol=config.RTOL,
            )


def test_segment_weighted(dual_graph):
    """Test segment_weighted=True on a dual graph."""
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    # Verify it requires a dual graph
    assert network_structure.is_dual
    assert "primal_edge" in nodes_gdf.columns
    # Compute with segment_weighted
    nodes_gdf_sw = networks.centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=True,
    )
    # Compute without segment_weighted
    nodes_gdf_plain = networks.centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=False,
    )
    for d in distances:
        density_key = config.prep_gdf_key("density", d)
        harmonic_key = config.prep_gdf_key("harmonic", d)
        betw_key = config.prep_gdf_key("betweenness", d)
        # Segment-weighted density should be total reachable street length (> node count)
        assert nodes_gdf_sw[density_key].sum() > nodes_gdf_plain[density_key].sum()
        # All expected columns should be present and non-null for live nodes
        assert not nodes_gdf_sw[density_key].isna().all()
        assert not nodes_gdf_sw[harmonic_key].isna().all()
        assert not nodes_gdf_sw[betw_key].isna().all()
    # Verify weights are restored after computation
    for nd_idx in network_structure.node_indices():
        assert network_structure.get_node_weight(nd_idx) == 1.0
    # Test simplest path variant too
    nodes_gdf_sw_ang = networks.centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=True,
    )
    for d in distances:
        density_key = config.prep_gdf_key("density", d, angular=True)
        assert not nodes_gdf_sw_ang[density_key].isna().all()


def test_closeness_shortest(primal_graph):
    """Test standalone closeness_shortest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.closeness_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("decay", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("cycles", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist) in nodes_gdf_result.columns


def test_closeness_shortest_seeded_determinism(primal_graph):
    """Same seed produces identical adaptive closeness_shortest results."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    kwargs = dict(
        network_structure=network_structure,
        distances=distances,
        random_seed=42,
    )
    r1 = networks.closeness_shortest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    r2 = networks.closeness_shortest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    for dist in distances:
        key = config.prep_gdf_key("density", dist)
        assert np.allclose(r1[key].values, r2[key].values), f"Non-deterministic at {dist}m"


def test_closeness_simplest(dual_graph):
    """Test standalone closeness_simplest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    nodes_gdf_result = networks.closeness_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist, angular=True) in nodes_gdf_result.columns


def test_closeness_simplest_seeded_determinism(dual_graph):
    """Same seed produces identical adaptive closeness_simplest results."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    kwargs = dict(
        network_structure=network_structure,
        distances=distances,
        random_seed=42,
    )
    r1 = networks.closeness_simplest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    r2 = networks.closeness_simplest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    for dist in distances:
        key = config.prep_gdf_key("density", dist, angular=True)
        assert np.allclose(r1[key].values, r2[key].values), f"Non-deterministic at {dist}m"


def test_simplest_wrappers_require_dual_graph(primal_graph):
    distances = [200, 400]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    with pytest.raises(ValueError, match="dual graph"):
        networks.closeness_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )
    with pytest.raises(ValueError, match="dual graph"):
        networks.betweenness_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )
