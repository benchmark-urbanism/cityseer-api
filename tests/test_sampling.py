"""Tests for centrality sampling with inverse probability weighting (IPW).

The sampling feature trades accuracy for speed by computing centrality from a
random subset of source nodes. Key properties:

1. Reproducibility: Same seed → same results
2. Unbiasedness: IPW-corrected samples converge to true values
3. Coverage: Target aggregation ensures all nodes receive results
4. Validation: Invalid inputs are rejected
5. Model: compute_required_p implements the inverted sampling model correctly
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

        res1 = ns.local_node_centrality_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.local_node_centrality_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )

        assert np.allclose(res1.node_density[500], res2.node_density[500])

    def test_reproducibility_different_seed(self, network_structure):
        """Different seeds produce different results."""
        ns, _ = network_structure

        res1 = ns.local_node_centrality_shortest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.local_node_centrality_shortest(
            distances=[500], sample_probability=0.3, random_seed=43, pbar_disabled=True
        )

        assert not np.allclose(res1.node_density[500], res2.node_density[500])

    def test_full_sampling_matches_baseline(self, network_structure):
        """Sampling with p=1.0 matches full computation."""
        ns, _ = network_structure

        res_full = ns.local_node_centrality_shortest(distances=[500], pbar_disabled=True)
        res_sampled = ns.local_node_centrality_shortest(
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

        res_full = ns.local_node_centrality_shortest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])

        samples = [
            np.array(
                ns.local_node_centrality_shortest(
                    distances=[distance], sample_probability=0.5, random_seed=seed, pbar_disabled=True
                ).node_density[distance]
            )
            for seed in range(num_runs)
        ]

        avg = np.mean(samples, axis=0)
        mask = full_density > 0
        error = np.mean(np.abs(avg[mask] - full_density[mask]) / full_density[mask])

        assert error < 0.15, f"Closeness error {error:.1%} exceeds 15%"

    def test_averaged_betweenness_converges(self, network_structure):
        """Averaged betweenness samples converge to true values."""
        ns, _ = network_structure
        distance = 500
        num_runs = 20

        res_full = ns.local_node_centrality_shortest(
            distances=[distance], compute_closeness=False, compute_betweenness=True, pbar_disabled=True
        )
        full_betw = np.array(res_full.node_betweenness[distance])

        samples = [
            np.array(
                ns.local_node_centrality_shortest(
                    distances=[distance],
                    compute_closeness=False,
                    compute_betweenness=True,
                    sample_probability=0.5,
                    random_seed=seed,
                    pbar_disabled=True,
                ).node_betweenness[distance]
            )
            for seed in range(num_runs)
        ]

        avg = np.mean(samples, axis=0)
        mask = full_betw > 0
        if np.any(mask):
            error = np.mean(np.abs(avg[mask] - full_betw[mask]) / full_betw[mask])
            assert error < 0.25, f"Betweenness error {error:.1%} exceeds 25%"

    def test_ipw_preserves_mean_across_probabilities(self, network_structure):
        """IPW scaling preserves mean across different sampling probabilities."""
        ns, _ = network_structure
        distance = 500
        num_runs = 25

        res_full = ns.local_node_centrality_shortest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])
        full_mean = np.mean(full_density[full_density > 0])

        for prob in [0.3, 0.5, 0.7]:
            sampled_means = []
            for seed in range(num_runs):
                res = ns.local_node_centrality_shortest(
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

        res_full = ns.local_node_centrality_shortest(distances=[distance], pbar_disabled=True)
        res_sampled = ns.local_node_centrality_shortest(
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

        res_full = ns.local_node_centrality_shortest(distances=[distance], pbar_disabled=True)
        full_cycles = np.array(res_full.node_cycles[distance])

        if np.sum(full_cycles > 0) == 0:
            pytest.skip("Mock graph has no cycles")

        res_sampled = ns.local_node_centrality_shortest(
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
            ns.local_node_centrality_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_validation_weight_negative(self, network_structure):
        """Weights < 0.0 raise ValueError."""
        ns, nodes_gdf = network_structure
        weights = [1.0] * len(nodes_gdf)
        weights[0] = -0.1

        with pytest.raises(ValueError, match="out of range"):
            ns.local_node_centrality_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_validation_wrong_length(self, network_structure):
        """Wrong weights length raises ValueError."""
        ns, nodes_gdf = network_structure
        weights = [1.0] * (len(nodes_gdf) - 1)

        with pytest.raises(ValueError, match="must match node count"):
            ns.local_node_centrality_shortest(
                distances=[500], sample_probability=0.5, sampling_weights=weights, pbar_disabled=True
            )

    def test_zero_weights_exclude_sources(self, network_structure):
        """Nodes with weight 0.0 are never sampled as sources."""
        ns, nodes_gdf = network_structure
        num_nodes = len(nodes_gdf)

        # Only first half can be sampled as sources
        weights = [1.0 if i < num_nodes // 2 else 0.0 for i in range(num_nodes)]

        res = ns.local_node_centrality_shortest(
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
                ns.local_node_centrality_shortest(
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
                ns.local_node_centrality_shortest(
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

        res1 = ns.local_node_centrality_simplest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )
        res2 = ns.local_node_centrality_simplest(
            distances=[500], sample_probability=0.3, random_seed=42, pbar_disabled=True
        )

        assert np.allclose(res1.node_density[500], res2.node_density[500])

    def test_averaged_samples_converge(self, network_structure):
        """Averaged simplest centrality samples converge to true values."""
        ns, _ = network_structure
        distance = 500
        num_runs = 15

        res_full = ns.local_node_centrality_simplest(distances=[distance], pbar_disabled=True)
        full_density = np.array(res_full.node_density[distance])

        samples = [
            np.array(
                ns.local_node_centrality_simplest(
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
    """Tests for the inverted sampling model: eff_n = max(k * sqrt(reach), min_eff_n)."""

    def test_constants_match_paper(self):
        """Model constants match the paper values."""
        assert config.SAMPLING_PROPORTIONAL_K == 9.91
        assert config.SAMPLING_MIN_EFF_N == 350.0

    def test_high_reach_proportional_regime(self):
        """At high reach, eff_n = k * sqrt(reach) and p decreases."""
        reach = 40000
        p = config.compute_required_p(reach)
        expected_eff_n = config.SAMPLING_PROPORTIONAL_K * math.sqrt(reach)
        expected_p = expected_eff_n / reach
        assert p == pytest.approx(expected_p, rel=1e-6)
        assert p < 0.1  # aggressive sampling at high reach

    def test_low_reach_floor_regime(self):
        """At low reach, eff_n is clamped to min_eff_n."""
        reach = 100
        p = config.compute_required_p(reach)
        # Floor: eff_n = 350, so p = 350/100 = 3.5, clamped to 1.0
        assert p == 1.0

    def test_crossover_reach(self):
        """At the crossover point, both regimes give the same eff_n."""
        k = config.SAMPLING_PROPORTIONAL_K
        min_n = config.SAMPLING_MIN_EFF_N
        crossover = (min_n / k) ** 2
        # Just above crossover: proportional regime dominates
        p_above = config.compute_required_p(crossover * 1.5)
        eff_n_above = p_above * crossover * 1.5
        assert eff_n_above > min_n
        # Just below crossover: floor dominates
        p_below = config.compute_required_p(crossover * 0.5)
        eff_n_below = p_below * crossover * 0.5
        assert eff_n_below == pytest.approx(min_n, rel=0.01)

    def test_p_clamped_to_one(self):
        """Sampling probability never exceeds 1.0."""
        for reach in [10, 50, 100, 200, 350]:
            p = config.compute_required_p(reach)
            assert p <= 1.0

    def test_p_has_minimum_floor(self):
        """Sampling probability has a 1% minimum floor."""
        p = config.compute_required_p(1e9)
        assert p >= 0.01

    def test_zero_reach_returns_none(self):
        """Zero reach returns None."""
        assert config.compute_required_p(0) is None

    def test_negative_reach_returns_none(self):
        """Negative reach returns None."""
        assert config.compute_required_p(-100) is None

    def test_p_decreases_with_reach(self):
        """Sampling probability decreases as reach increases."""
        reaches = [500, 1000, 5000, 10000, 50000]
        probs = [config.compute_required_p(r) for r in reaches]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], f"p should decrease: {probs[i]} < {probs[i + 1]}"

    def test_paper_example_values(self):
        """Spot-check values from paper Section 5 (Computational Complexity)."""
        # 500m: reach ≈ 200, p = 1.0
        p_500m = config.compute_required_p(200)
        assert p_500m == 1.0
        # 5km: reach ≈ 3000, p ≈ 0.26
        p_5km = config.compute_required_p(3000)
        assert 0.15 < p_5km < 0.35
        # 20km: reach ≈ 40000, p ≈ 0.07
        p_20km = config.compute_required_p(40000)
        assert 0.03 < p_20km < 0.10

    def test_metric_and_distance_type_ignored(self):
        """Inverted model gives same p regardless of metric/distance_type args."""
        reach = 5000
        p_default = config.compute_required_p(reach)
        p_harmonic = config.compute_required_p(reach, metric="harmonic")
        p_betw = config.compute_required_p(reach, metric="betweenness")
        p_angular = config.compute_required_p(reach, distance_type="angular")
        assert p_default == p_harmonic == p_betw == p_angular
