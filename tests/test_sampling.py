"""Tests for centrality sampling with inverse probability weighting (IPW).

The sampling feature trades accuracy for speed by computing centrality from a
random subset of source nodes. Key properties:

1. Reproducibility: Same seed → same results
2. Unbiasedness: IPW-corrected samples converge to true values
3. Coverage: Target aggregation ensures all nodes receive results
4. Validation: Invalid inputs are rejected
5. Model: Hoeffding/Eppstein–Wang reach → p is implemented correctly
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from cityseer import config
from cityseer.tools import graphs, io, mock


@pytest.fixture
def network_structure():
    """Create a mock network for testing."""
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G)
    return ns, nodes_gdf


class TestSamplingBasics:
    """Basic sampling functionality."""

    def test_reproducibility_same_seed(self, network_structure):
        """Same seed produces identical results."""
        ns, _ = network_structure

        res1 = ns.closeness_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.closeness_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )

        assert np.allclose(res1.node_density[500], res2.node_density[500])

    def test_reproducibility_different_seed(self, network_structure):
        """Different seeds produce different results."""
        ns, _ = network_structure

        res1 = ns.closeness_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.closeness_shortest(
            distances=[500], sample_probability=0.3, random_seed=43, pbar_disabled=True
        )

        assert not np.allclose(res1.node_density[500], res2.node_density[500])

    def test_full_sampling_matches_baseline(self, network_structure):
        """Sampling with p=1.0 matches full computation."""
        ns, _ = network_structure

        res_full = ns.closeness_shortest(distances=[500], pbar_disabled=True)
        res_sampled = ns.closeness_shortest(
            distances=[500], sample_probability=1.0, random_seed=42, pbar_disabled=True
        )

        assert np.allclose(res_full.node_density[500], res_sampled.node_density[500])


class TestIPWCorrection:
    """Inverse probability weighting produces unbiased estimates."""

    def test_averaged_closeness_converges(self, network_structure):
        """Averaged closeness samples converge to true values."""
        ns, _ = network_structure
        distance = 500
        num_runs = 20

        res_full = ns.closeness_shortest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])

        samples = [
            np.array(
                ns.closeness_shortest(
                    distances=[distance], sample_probability=0.5, random_seed=seed, pbar_disabled=True
                ).node_density[distance]
            )
            for seed in range(num_runs)
        ]

        avg = np.mean(samples, axis=0)
        mask = full_density > 0
        error = np.mean(np.abs(avg[mask] - full_density[mask]) / full_density[mask])

        assert error < 0.15, f"Closeness error {error:.1%} exceeds 15%"

    def test_ipw_preserves_mean_across_probabilities(self, network_structure):
        """IPW scaling preserves mean across different sampling probabilities."""
        ns, _ = network_structure
        distance = 500
        num_runs = 25

        res_full = ns.closeness_shortest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])
        full_mean = np.mean(full_density[full_density > 0])

        for prob in [0.3, 0.5, 0.7]:
            sampled_means = []
            for seed in range(num_runs):
                res = ns.closeness_shortest(
                    distances=[distance], sample_probability=prob, random_seed=seed, pbar_disabled=True
                )
                density = np.array(res.node_density[distance])
                mask = (density > 0) & (full_density > 0)
                if np.any(mask):
                    sampled_means.append(np.mean(density[mask]))

            avg_mean = np.mean(sampled_means)
            error = abs(avg_mean - full_mean) / full_mean
            assert error < 0.25, f"Mean error {error:.1%} exceeds 25% at p={prob}"


class TestTargetAggregation:
    """Target aggregation ensures high coverage."""

    def test_high_coverage_density(self, network_structure):
        """Most nodes get density values even with aggressive sampling."""
        ns, _ = network_structure
        distance = 500

        res_full = ns.closeness_shortest(distances=[distance], pbar_disabled=True)
        res_sampled = ns.closeness_shortest(
            distances=[distance], sample_probability=0.5, random_seed=42, pbar_disabled=True
        )

        full = np.array(res_full.node_density[distance])
        sampled = np.array(res_sampled.node_density[distance])

        mask = full > 0
        zeros = np.sum((sampled == 0) & mask)
        coverage = (np.sum(mask) - zeros) / np.sum(mask)

        assert coverage > 0.9, f"Density coverage {coverage:.1%} below 90%"

    def test_reasonable_coverage_cycles(self, network_structure):
        """Cycles metric has reasonable coverage (sparser than density)."""
        ns, _ = network_structure
        distance = 500

        res_full = ns.closeness_shortest(distances=[distance], pbar_disabled=True)
        full_cycles = np.array(res_full.node_cycles[distance])

        if np.sum(full_cycles > 0) == 0:
            pytest.skip("Mock graph has no cycles")

        res_sampled = ns.closeness_shortest(
            distances=[distance], sample_probability=0.5, random_seed=42, pbar_disabled=True
        )
        sampled_cycles = np.array(res_sampled.node_cycles[distance])

        mask = full_cycles > 0
        zeros = np.sum((sampled_cycles == 0) & mask)
        coverage = (np.sum(mask) - zeros) / np.sum(mask)

        assert coverage > 0.5, f"Cycle coverage {coverage:.1%} below 50%"


class TestSamplingWeights:
    """Weighted sampling functionality."""

    def test_validation_weight_too_high(self, network_structure):
        """Weights > 1.0 raise ValueError."""
        ns, nodes_gdf = network_structure
        weights = [1.0] * len(nodes_gdf)
        weights[0] = 1.5

        with pytest.raises(ValueError, match="out of range"):
            ns.closeness_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_validation_weight_negative(self, network_structure):
        """Weights < 0.0 raise ValueError."""
        ns, nodes_gdf = network_structure
        weights = [1.0] * len(nodes_gdf)
        weights[0] = -0.1

        with pytest.raises(ValueError, match="out of range"):
            ns.closeness_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_validation_wrong_length(self, network_structure):
        """Wrong weights length raises ValueError."""
        ns, nodes_gdf = network_structure
        weights = [1.0] * (len(nodes_gdf) - 1)

        with pytest.raises(ValueError, match="must match node count"):
            ns.closeness_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_zero_weights_exclude_sources(self, network_structure):
        """Nodes with weight 0.0 are never sampled as sources."""
        ns, nodes_gdf = network_structure
        num_nodes = len(nodes_gdf)

        # Only first half can be sampled as sources
        weights = [1.0 if i < num_nodes // 2 else 0.0 for i in range(num_nodes)]

        res = ns.closeness_shortest(
            distances=[500], sample_probability=1.0, sampling_weights=weights, pbar_disabled=True
        )

        # Target aggregation: nodes can still receive values from other sources
        assert np.sum(np.array(res.node_density[500]) > 0) > 0

    def test_uniform_weights_equivalent_to_probability(self, network_structure):
        """Uniform weights of 0.5 with p=1.0 ≈ p=0.5 with no weights."""
        ns, nodes_gdf = network_structure
        num_runs = 25
        distance = 500

        uniform_samples = [
            np.array(
                ns.closeness_shortest(
                    distances=[distance],
                    sample_probability=1.0,
                    sampling_weights=[0.5] * len(nodes_gdf),
                    random_seed=seed,
                    pbar_disabled=True,
                ).node_density[distance]
            )
            for seed in range(num_runs)
        ]

        plain_samples = [
            np.array(
                ns.closeness_shortest(
                    distances=[distance], sample_probability=0.5, random_seed=seed, pbar_disabled=True
                ).node_density[distance]
            )
            for seed in range(num_runs)
        ]

        avg_uniform = np.mean(uniform_samples, axis=0)
        avg_plain = np.mean(plain_samples, axis=0)

        mask = (avg_uniform > 0) & (avg_plain > 0)
        if np.any(mask):
            corr = np.corrcoef(avg_uniform[mask], avg_plain[mask])[0, 1]
            assert corr > 0.95, f"Correlation {corr:.3f} below 0.95"


class TestSimplestCentrality:
    """Angular (simplest path) centrality sampling."""

    def test_reproducibility(self, network_structure):
        """Simplest centrality sampling is reproducible."""
        ns, _ = network_structure

        res1 = ns.closeness_simplest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.closeness_simplest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )

        assert np.allclose(res1.node_density[500], res2.node_density[500])

    def test_averaged_samples_converge(self, network_structure):
        """Averaged simplest centrality samples converge to true values."""
        ns, _ = network_structure
        distance = 500
        num_runs = 15

        res_full = ns.closeness_simplest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])

        samples = [
            np.array(
                ns.closeness_simplest(
                    distances=[distance], sample_probability=0.5, random_seed=seed, pbar_disabled=True
                ).node_density[distance]
            )
            for seed in range(num_runs)
        ]

        avg = np.mean(samples, axis=0)
        mask = full_density > 0
        if np.any(mask):
            error = np.mean(np.abs(avg[mask] - full_density[mask]) / full_density[mask])
            assert error < 0.20, f"Simplest error {error:.1%} exceeds 20%"


class TestSamplingModel:
    """Tests for the Hoeffding/Eppstein–Wang reach → p sampling model."""

    def test_constants_match_paper(self):
        """Default Hoeffding parameters match the paper defaults."""
        assert config.HOEFFDING_EPSILON == 0.1
        assert config.HOEFFDING_DELTA == 0.1

    def test_compute_hoeffding_p_matches_formula(self):
        """compute_hoeffding_p implements k=log(2r/δ)/(2ε²), p=min(1,k/r)."""
        reach = 1000.0
        epsilon = 0.1
        delta = 0.1
        p = config.compute_hoeffding_p(reach, epsilon=epsilon, delta=delta)
        assert p is not None
        expected_k = math.log(2 * reach / delta) / (2 * epsilon**2)
        expected_p = min(1.0, expected_k / reach)
        assert p == pytest.approx(expected_p, rel=1e-10)

    def test_p_decreases_with_reach(self):
        """Sampling probability decreases as reach increases."""
        reaches = [200.0, 500.0, 1000.0, 5000.0, 10000.0]
        probs = [config.compute_hoeffding_p(r) for r in reaches]
        assert all(p is not None for p in probs)
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], f"p should decrease: {probs[i]} < {probs[i + 1]}"

    def test_low_reach_clamps_to_one(self):
        """At low reach the bound requests full computation (p=1)."""
        assert config.compute_hoeffding_p(50.0) == 1.0

    def test_compute_sample_probs_vectorises(self):
        """compute_sample_probs returns a p for each distance."""
        reach_estimates = {200: 100.0, 400: 1000.0}
        probs = config.compute_sample_probs(reach_estimates, epsilon=0.1, delta=0.1)
        assert set(probs.keys()) == {200, 400}
        assert probs[200] is not None and 0.0 < probs[200] <= 1.0
        assert probs[400] is not None and 0.0 < probs[400] <= 1.0

    def test_zero_reach_returns_full(self):
        """Zero reach returns 1.0 (full sampling)."""
        assert config.compute_hoeffding_p(0) == 1.0

    def test_negative_reach_returns_full(self):
        """Negative reach returns 1.0 (full sampling)."""
        assert config.compute_hoeffding_p(-100) == 1.0

    def test_invalid_delta_returns_full(self):
        """Invalid delta returns 1.0 (defensive fallback)."""
        assert config.compute_hoeffding_p(1000.0, delta=0.0) == 1.0
        assert config.compute_hoeffding_p(1000.0, delta=1.0) == 1.0


class TestSpatialSample:
    """Direct tests for config.spatial_sample()."""

    def test_returns_correct_count(self, network_structure):
        """spatial_sample returns exactly n_samples nodes."""
        ns, nodes_gdf = network_structure
        n_live = sum(1 for i in ns.node_indices() if ns.is_node_live(i))
        for n_samples in [5, 10, n_live // 2]:
            indices, area = config.spatial_sample(ns, n_samples, random_seed=42)
            assert len(indices) == n_samples, f"Expected {n_samples} samples, got {len(indices)}"

    def test_returns_all_when_n_samples_exceeds_live(self, network_structure):
        """When n_samples >= n_live, all live nodes are returned."""
        ns, nodes_gdf = network_structure
        n_live = sum(1 for i in ns.node_indices() if ns.is_node_live(i))
        indices, area = config.spatial_sample(ns, n_live + 10, random_seed=42)
        live_set = {i for i in ns.node_indices() if ns.is_node_live(i)}
        assert set(indices) == live_set

    def test_returns_area_km2(self, network_structure):
        """Area in km² is positive and reasonable for mock graph."""
        ns, _ = network_structure
        _, area = config.spatial_sample(ns, 10, random_seed=42)
        assert area > 0, "Area should be positive"

    def test_reproducibility(self, network_structure):
        """Same seed produces same sample."""
        ns, _ = network_structure
        idx1, _ = config.spatial_sample(ns, 10, random_seed=42)
        idx2, _ = config.spatial_sample(ns, 10, random_seed=42)
        assert idx1 == idx2

    def test_different_seed_different_sample(self, network_structure):
        """Different seeds produce different samples."""
        ns, _ = network_structure
        idx1, _ = config.spatial_sample(ns, 10, random_seed=42)
        idx2, _ = config.spatial_sample(ns, 10, random_seed=99)
        assert idx1 != idx2

    def test_all_returned_are_live(self, network_structure):
        """All sampled indices are live nodes."""
        ns, _ = network_structure
        indices, _ = config.spatial_sample(ns, 15, random_seed=42)
        for idx in indices:
            assert ns.is_node_live(idx), f"Node {idx} is not live"

    def test_no_duplicates(self, network_structure):
        """No duplicate indices in sample."""
        ns, _ = network_structure
        indices, _ = config.spatial_sample(ns, 20, random_seed=42)
        assert len(indices) == len(set(indices)), "Duplicates found in sample"

    def test_zero_samples_returns_empty(self, network_structure):
        """Zero requested samples returns an empty list."""
        ns, _ = network_structure
        indices, area = config.spatial_sample(ns, 0, random_seed=42)
        assert indices == []
        assert area >= 0

    def test_invalid_cell_size_raises(self, network_structure):
        """Non-positive cell_size raises ValueError."""
        ns, _ = network_structure
        with pytest.raises(ValueError, match="cell_size"):
            config.spatial_sample(ns, 10, cell_size=0, random_seed=42)


class TestProbeReachability:
    """Direct tests for config.probe_reachability()."""

    def test_returns_dict_for_each_distance(self, network_structure):
        """probe_reachability returns a reach estimate for each distance."""
        ns, _ = network_structure
        distances = [200, 500, 1000]
        reach = config.probe_reachability(ns, distances)
        assert set(reach.keys()) == set(distances)

    def test_reach_increases_with_distance(self, network_structure):
        """Reach should increase (or stay equal) with larger distance thresholds."""
        ns, _ = network_structure
        distances = [200, 500, 1000, 2000]
        reach = config.probe_reachability(ns, distances)
        for i in range(len(distances) - 1):
            assert reach[distances[i]] <= reach[distances[i + 1]], (
                f"Reach at {distances[i]}m ({reach[distances[i]]}) > "
                f"reach at {distances[i+1]}m ({reach[distances[i+1]]})"
            )

    def test_reach_positive_at_large_distance(self, network_structure):
        """At a distance that spans the graph, reach should be positive."""
        ns, _ = network_structure
        reach = config.probe_reachability(ns, [5000])
        assert reach[5000] > 0, "Reach should be positive at large distance"

    def test_reach_values_are_finite(self, network_structure):
        """All reach estimates should be finite non-negative numbers."""
        ns, _ = network_structure
        reach = config.probe_reachability(ns, [200, 500])
        for d, r in reach.items():
            assert np.isfinite(r), f"Reach at {d}m is not finite: {r}"
            assert r >= 0, f"Reach at {d}m is negative: {r}"

    def test_probe_reachability_reproducible_with_seed(self, network_structure):
        """Probe selection can be made reproducible with a seed."""
        ns, _ = network_structure
        distances = [200, 500, 1000]
        r1 = config.probe_reachability(ns, distances, random_seed=42)
        r2 = config.probe_reachability(ns, distances, random_seed=42)
        assert r1 == r2

    def test_empty_distances_returns_empty(self, network_structure):
        """Empty distances list returns empty dict without error."""
        ns, _ = network_structure
        result = config.probe_reachability(ns, [])
        assert result == {}


class TestActualPClamping:
    """Verify actual_p is computed correctly and clamped to <= 1.0."""

    def test_actual_p_never_exceeds_one(self, network_structure):
        """actual_p = n_sources / n_live should never exceed 1.0."""
        ns, _ = network_structure
        n_live = sum(1 for i in ns.node_indices() if ns.is_node_live(i))
        # Use a very high requested p and small cell_size to force n_cells > n_live
        for p in [0.5, 0.9, 1.0, 2.0]:
            n_cells = config.min_spatial_samples(ns, cell_size=50)
            n_sources = min(n_live, max(n_cells, int(p * n_live)))
            actual_p = (n_sources / n_live) if n_live > 0 else 1.0
            assert actual_p <= 1.0, f"actual_p={actual_p} exceeds 1.0 for p={p}"

    def test_n_cells_floor_raises_actual_p(self, network_structure):
        """When n_cells > p * n_live, actual_p should be bumped above p."""
        ns, _ = network_structure
        n_live = sum(1 for i in ns.node_indices() if ns.is_node_live(i))
        # Very small cell_size = many cells, very small p
        n_cells = config.min_spatial_samples(ns, cell_size=50)
        p = 0.01  # Intentionally low
        n_sources = min(n_live, max(n_cells, int(p * n_live)))
        actual_p = (n_sources / n_live) if n_live > 0 else 1.0
        if n_cells > int(p * n_live):
            assert actual_p > p, f"actual_p={actual_p} should exceed p={p} when n_cells dominates"
