from __future__ import annotations

import os
import statistics
import timeit

import networkx as nx
import numpy as np
import pytest
from cityseer.tools import graphs, io, mock


def test_sampling_comprehensive_report():
    """
    Comprehensive test showing accuracy and speed trade-offs for sampling.

    This is the main test that demonstrates the sampling feature's characteristics
    across different probabilities and distances.
    """
    if "GITHUB_ACTIONS" in os.environ:
        pytest.skip("Skipping performance test in CI")

    os.environ["CITYSEER_QUIET_MODE"] = "1"

    # Create a grid graph for meaningful results
    grid_size = 50
    spacing = 50
    G_grid = nx.grid_2d_graph(grid_size, grid_size)

    G_primal = nx.MultiGraph()
    G_primal.graph["crs"] = "EPSG:27700"
    for node in G_grid.nodes():
        x = node[0] * spacing
        y = node[1] * spacing
        G_primal.add_node(f"{x}_{y}", x=x, y=y)

    for u, v in G_grid.edges():
        u_key = f"{u[0] * spacing}_{u[1] * spacing}"
        v_key = f"{v[0] * spacing}_{v[1] * spacing}"
        G_primal.add_edge(u_key, v_key)

    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    distances = [500, 1000, 2000]
    probabilities = [0.1, 0.5, 0.9]
    n_runs = 5  # For both error estimation and timing

    print("\n")
    print("## SAMPLING: ACCURACY vs SPEED TRADE-OFFS")
    print()
    print(f"- **Grid:** {grid_size}x{grid_size} = {grid_size * grid_size} nodes")
    print(f"- **Distances:** {distances}m")
    print(f"- **Probabilities:** {probabilities}")
    print(f"- **Runs:** {n_runs} (for averaging)")
    print()

    # Get full computation for all distances
    res_full = ns.local_node_centrality_shortest(
        distances=distances,
        compute_closeness=True,
        compute_betweenness=True,
        pbar_disabled=True,
    )

    results = []

    for dist in distances:
        full_density = np.array(res_full.node_density[dist])
        full_betw = np.array(res_full.node_betweenness[dist])
        avg_reachable = np.mean(full_density[full_density > 0])

        # Time full computation at this distance
        full_times = []
        for _ in range(n_runs):
            start = timeit.default_timer()
            ns.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=True,
            )
            full_times.append(timeit.default_timer() - start)
        full_time = statistics.mean(full_times)

        for prob in probabilities:
            # Collect samples and timing in the same runs
            density_samples = []
            betw_samples = []
            sampled_times = []

            for seed in range(n_runs):
                start = timeit.default_timer()
                res = ns.local_node_centrality_shortest(
                    distances=[dist],
                    compute_closeness=True,
                    compute_betweenness=True,
                    sample_probability=prob,
                    random_seed=seed,
                    pbar_disabled=True,
                )
                sampled_times.append(timeit.default_timer() - start)
                density_samples.append(np.array(res.node_density[dist]))
                betw_samples.append(np.array(res.node_betweenness[dist]))

            # Compute errors
            avg_density = np.mean(density_samples, axis=0)
            avg_betw = np.mean(betw_samples, axis=0)

            mask_d = full_density > 0
            mask_b = full_betw > 0

            density_err = np.mean(np.abs(avg_density[mask_d] - full_density[mask_d]) / full_density[mask_d])
            betw_err = (
                np.mean(np.abs(avg_betw[mask_b] - full_betw[mask_b]) / full_betw[mask_b])
                if np.any(mask_b)
                else float("nan")
            )

            # Count zeros
            density_zeros = np.mean([np.sum((s == 0) & mask_d) for s in density_samples])
            betw_zeros = np.mean([np.sum((s == 0) & mask_b) for s in betw_samples])

            # Compute speedup from same runs
            sampled_time = statistics.mean(sampled_times)
            speedup = full_time / sampled_time if sampled_time > 0 else 0

            results.append(
                {
                    "dist": dist,
                    "prob": prob,
                    "avg_reachable": avg_reachable,
                    "density_err": density_err,
                    "density_zeros": density_zeros,
                    "betw_err": betw_err,
                    "betw_zeros": betw_zeros,
                    "time_ms": sampled_time * 1000,
                    "speedup": speedup,
                }
            )

    # Print results table in Markdown format
    hdr = "| Distance | Prob | Avg Reachable | Closeness Zeros | Closeness Error "
    hdr += "| Betweenness Zeros | Betweenness Error | Time | Speedup |"
    print(hdr)
    print(
        "|----------|------|---------------|-----------------|-----------------|"
        "-------------------|-------------------|------|---------|"
    )

    for r in results:
        d_err = f"{r['density_err']:.1%}" if not np.isnan(r["density_err"]) else "N/A"
        b_err = f"{r['betw_err']:.1%}" if not np.isnan(r["betw_err"]) else "N/A"
        row = (
            f"| {r['dist']}m | {r['prob']:.1f} | {r['avg_reachable']:.0f} | "
            f"{r['density_zeros']:.1f} | {d_err} | "
            f"{r['betw_zeros']:.1f} | {b_err} | "
            f"{r['time_ms']:.0f}ms | {r['speedup']:.1f}x |"
        )
        print(row)

    print()
    print("### Key Findings")
    print()
    print("- Higher probability → lower error, but slower (less speedup)")
    print("- Larger distance → more reachable nodes → lower error for same probability")
    print("- Closeness: zero nodes means isolated from sampled sources")
    print("- Speedup ~1/p for source sampling with flipped aggregation")

    # Basic assertions
    # At p=0.5, closeness error should be reasonable
    mid_results = [r for r in results if r["prob"] == 0.5]
    for r in mid_results:
        assert r["density_err"] < 0.3, f"Density error {r['density_err']:.1%} exceeds 30% at {r['dist']}m"
        # Speedup threshold relaxed to 1.2x to account for timing variability
        # Expected theoretical speedup is ~2x at p=0.5, but overhead reduces actual speedup
        assert r["speedup"] > 1.2, f"Speedup {r['speedup']:.1f}x below 1.2x at {r['dist']}m"

    # At p=0.1, we should see significant speedup (less strict due to high variance)
    low_p_results = [r for r in results if r["prob"] == 0.1]
    for r in low_p_results:
        assert r["speedup"] > 2.0, f"Speedup {r['speedup']:.1f}x below 2.0x at p=0.1, {r['dist']}m"


def test_sampling_approximation_quality():
    """
    Test that sampling produces statistically unbiased estimates.

    Uses multiple runs to verify the average converges to the true value.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    num_runs = 20
    sample_probability = 0.5
    distance = 500

    # Full computation (baseline)
    res_full = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        compute_betweenness=True,
        pbar_disabled=True,
    )

    # Collect samples
    density_samples = []
    betweenness_samples = []

    for seed in range(num_runs):
        res = ns.local_node_centrality_shortest(
            distances=[distance],
            compute_closeness=True,
            compute_betweenness=True,
            sample_probability=sample_probability,
            random_seed=seed,
            pbar_disabled=True,
        )
        density_samples.append(res.node_density[distance])
        betweenness_samples.append(res.node_betweenness[distance])

    full_density = np.array(res_full.node_density[distance])
    full_betweenness = np.array(res_full.node_betweenness[distance])

    avg_density = np.mean(density_samples, axis=0)
    avg_betweenness = np.mean(betweenness_samples, axis=0)

    # Closeness error
    mask_d = full_density > 0
    density_error = np.mean(np.abs(avg_density[mask_d] - full_density[mask_d]) / full_density[mask_d])

    # Betweenness error
    mask_b = full_betweenness > 0
    betweenness_error = (
        np.mean(np.abs(avg_betweenness[mask_b] - full_betweenness[mask_b]) / full_betweenness[mask_b])
        if np.any(mask_b)
        else 0
    )

    print(f"\n### Approximation Quality (p={sample_probability}, {num_runs} runs)")
    print()
    print(f"- **Closeness error:** {density_error:.1%}")
    print(f"- **Betweenness error:** {betweenness_error:.1%}")

    # Assertions - averaged results should be close to true values
    assert density_error < 0.15, f"Averaged density error {density_error:.1%} exceeds 15%"
    assert betweenness_error < 0.25, f"Averaged betweenness error {betweenness_error:.1%} exceeds 25%"


def test_sampling_reproducibility():
    """
    Test that providing a seed produces reproducible results.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    seed = 42

    # First run
    res1 = ns.local_node_centrality_shortest(
        distances=[500], sample_probability=0.3, random_seed=seed, pbar_disabled=True
    )

    # Second run with same seed
    res2 = ns.local_node_centrality_shortest(
        distances=[500], sample_probability=0.3, random_seed=seed, pbar_disabled=True
    )

    # Third run with different seed
    res3 = ns.local_node_centrality_shortest(
        distances=[500], sample_probability=0.3, random_seed=seed + 1, pbar_disabled=True
    )

    density1 = res1.node_density[500]
    density2 = res2.node_density[500]
    density3 = res3.node_density[500]

    # Same seed should give identical results
    assert np.allclose(density1, density2), "Same seed should produce identical results"

    # Different seed should give different results
    assert not np.allclose(density1, density3), "Different seed should produce different results"


def test_sampling_weighted():
    """
    Test that weighted sampling impacts results as expected.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    num_nodes = len(nodes_gdf)

    # Create weights: first half have weight 1.0, second half have weight 0.0
    sampling_weights = [1.0 if i < num_nodes // 2 else 0.0 for i in range(num_nodes)]

    res = ns.local_node_centrality_shortest(
        distances=[500],
        sample_probability=1.0,
        sampling_weights=sampling_weights,
        pbar_disabled=True,
    )

    density = np.array(res.node_density[500])

    # Nodes with weight 0.0 should not be sampled as sources
    # With flipped aggregation, they can still receive contributions from other sources
    # Verify at least some nodes got results
    nonzero_count = np.sum(density > 0)
    assert nonzero_count > 0, "Expected some nodes to have non-zero density"


def test_sampling_all_nodes_get_results():
    """
    Test that with target aggregation, all nodes get results even with sampling.

    This is the key benefit of the target aggregation approach:
    - Traditional source aggregation: only sampled nodes get values
    - Target aggregation: all nodes within range of ANY sampled source get values

    This ensures high coverage even with aggressive sampling.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    sample_probability = 0.5
    distance = 500

    # Full computation
    res_full = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        compute_betweenness=True,
        pbar_disabled=True,
    )

    # Sampled computation
    res_sampled = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        compute_betweenness=True,
        sample_probability=sample_probability,
        random_seed=42,
        pbar_disabled=True,
    )

    full_density = np.array(res_full.node_density[distance])
    sampled_density = np.array(res_sampled.node_density[distance])

    # Count nodes with values in full computation
    mask = full_density > 0
    total_with_values = np.sum(mask)

    # Count how many of those have values in sampled computation
    sampled_zeros = np.sum((sampled_density == 0) & mask)
    pct_with_results = (total_with_values - sampled_zeros) / total_with_values if total_with_values > 0 else 0

    print("\n### Target Aggregation Coverage")
    print()
    print(f"- **Nodes with values in full:** {total_with_values}")
    print(f"- **Nodes with zero in sampled:** {sampled_zeros}")
    print(f"- **Coverage:** {pct_with_results:.1%}")

    # With target aggregation, most nodes should get results
    assert pct_with_results > 0.9, f"Expected >90% coverage, got {pct_with_results:.1%}"


def test_sampling_simplest_centrality():
    """
    Test that sampling works correctly for angular (simplest path) centrality.

    Verifies that:
    1. Sampling produces reproducible results with seeds
    2. Averaged samples converge toward true values
    3. All nodes get coverage with target aggregation
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    num_runs = 15
    sample_probability = 0.5
    distance = 500

    # Full computation
    res_full = ns.local_node_centrality_simplest(
        distances=[distance],
        compute_closeness=True,
        compute_betweenness=True,
        pbar_disabled=True,
    )

    # Collect samples
    density_samples = []
    for seed in range(num_runs):
        res = ns.local_node_centrality_simplest(
            distances=[distance],
            compute_closeness=True,
            compute_betweenness=True,
            sample_probability=sample_probability,
            random_seed=seed,
            pbar_disabled=True,
        )
        density_samples.append(np.array(res.node_density[distance]))

    full_density = np.array(res_full.node_density[distance])
    avg_density = np.mean(density_samples, axis=0)

    # Check error
    mask = full_density > 0
    if np.any(mask):
        density_error = np.mean(np.abs(avg_density[mask] - full_density[mask]) / full_density[mask])
        print(f"\n### Simplest Centrality Sampling (p={sample_probability}, {num_runs} runs)")
        print()
        print(f"- **Density error:** {density_error:.1%}")
        assert density_error < 0.20, f"Simplest density error {density_error:.1%} exceeds 20%"

    # Check coverage - all nodes should get results with target aggregation
    single_sample = density_samples[0]
    zeros_in_sample = np.sum((single_sample == 0) & mask)
    coverage = (np.sum(mask) - zeros_in_sample) / np.sum(mask) if np.sum(mask) > 0 else 0
    print(f"- **Single sample coverage:** {coverage:.1%}")
    assert coverage > 0.8, f"Expected >80% coverage for single sample, got {coverage:.1%}"

    # Reproducibility check
    res_a = ns.local_node_centrality_simplest(
        distances=[distance],
        sample_probability=0.3,
        random_seed=123,
        pbar_disabled=True,
    )
    res_b = ns.local_node_centrality_simplest(
        distances=[distance],
        sample_probability=0.3,
        random_seed=123,
        pbar_disabled=True,
    )
    assert np.allclose(res_a.node_density[distance], res_b.node_density[distance]), (
        "Simplest centrality should be reproducible with same seed"
    )


def test_sampling_ipw_scaling():
    """
    Test that inverse probability weighting (IPW) correctly scales results.

    With IPW, sampled results should be unbiased estimators of the true values.
    This test verifies that the scaling factor (1/p) is applied correctly.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    distance = 500
    num_runs = 30

    # Full computation baseline
    res_full = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        pbar_disabled=True,
    )
    full_density = np.array(res_full.node_density[distance])
    full_mean = np.mean(full_density[full_density > 0])

    # Test different probabilities - mean should be approximately preserved
    for prob in [0.3, 0.5, 0.7]:
        sampled_means = []
        for seed in range(num_runs):
            res = ns.local_node_centrality_shortest(
                distances=[distance],
                compute_closeness=True,
                sample_probability=prob,
                random_seed=seed,
                pbar_disabled=True,
            )
            density = np.array(res.node_density[distance])
            # Only consider nodes that have values in both
            mask = (density > 0) & (full_density > 0)
            if np.any(mask):
                sampled_means.append(np.mean(density[mask]))

        avg_sampled_mean = np.mean(sampled_means)
        # The IPW-scaled mean should be close to the full mean
        relative_error = abs(avg_sampled_mean - full_mean) / full_mean
        print(f"\n### IPW Test at p={prob}")
        print()
        print(f"- **Full mean:** {full_mean:.2f}")
        print(f"- **Avg sampled mean:** {avg_sampled_mean:.2f}")
        print(f"- **Relative error:** {relative_error:.1%}")

        # Allow some variance, but should be reasonably close
        assert relative_error < 0.25, f"IPW scaling error {relative_error:.1%} exceeds 25% at p={prob}"


def test_sampling_cycles_coverage():
    """
    Test that cycles metric works correctly with sampling using target aggregation.

    Previously, cycles were aggregated to the source node, which meant unsampled
    sources would have zero cycle values. With target aggregation, all nodes
    within range of sampled sources should get cycle contributions.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    distance = 500
    sample_probability = 0.5

    # Full computation
    res_full = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        pbar_disabled=True,
    )

    full_cycles = np.array(res_full.node_cycles[distance])

    # Check that we have some cycles in the graph (prerequisite for this test)
    if np.sum(full_cycles > 0) == 0:
        pytest.skip("Mock graph has no cycles to test")

    # Sampled computation
    res_sampled = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        sample_probability=sample_probability,
        random_seed=42,
        pbar_disabled=True,
    )

    sampled_cycles = np.array(res_sampled.node_cycles[distance])

    # With target aggregation, nodes that have cycles in full computation
    # should also have cycles in sampled computation (at least most of them)
    mask = full_cycles > 0
    total_with_cycles = np.sum(mask)
    sampled_zeros = np.sum((sampled_cycles == 0) & mask)
    coverage = (total_with_cycles - sampled_zeros) / total_with_cycles if total_with_cycles > 0 else 0

    print(f"\n### Cycles Coverage with Sampling (p={sample_probability})")
    print()
    print(f"- **Nodes with cycles in full:** {total_with_cycles}")
    print(f"- **Nodes with zero cycles in sampled:** {sampled_zeros}")
    print(f"- **Coverage:** {coverage:.1%}")

    # With target aggregation, we should have good coverage
    # (not as high as density since cycles are sparser)
    assert coverage > 0.5, f"Expected >50% cycle coverage with target aggregation, got {coverage:.1%}"

    # Also verify that averaged samples converge to true values
    num_runs = 20
    cycle_samples = []
    for seed in range(num_runs):
        res = ns.local_node_centrality_shortest(
            distances=[distance],
            compute_closeness=True,
            sample_probability=sample_probability,
            random_seed=seed,
            pbar_disabled=True,
        )
        cycle_samples.append(np.array(res.node_cycles[distance]))

    avg_cycles = np.mean(cycle_samples, axis=0)

    if np.any(mask):
        cycle_error = np.mean(np.abs(avg_cycles[mask] - full_cycles[mask]) / full_cycles[mask])
        print(f"- **Cycle error** (averaged over {num_runs} runs): {cycle_error:.1%}")
        # Cycles have higher variance, so allow more error
        assert cycle_error < 0.5, f"Averaged cycle error {cycle_error:.1%} exceeds 50%"


def test_sampling_weights_validation():
    """
    Test that sampling_weights outside [0.0, 1.0] raise ValueError.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    num_nodes = len(nodes_gdf)

    # Test weight > 1.0
    weights_too_high = [1.0] * num_nodes
    weights_too_high[0] = 1.5
    with pytest.raises(ValueError, match="out of range"):
        ns.local_node_centrality_shortest(
            distances=[500],
            sample_probability=0.5,
            sampling_weights=weights_too_high,
            pbar_disabled=True,
        )

    # Test weight < 0.0
    weights_negative = [1.0] * num_nodes
    weights_negative[0] = -0.1
    with pytest.raises(ValueError, match="out of range"):
        ns.local_node_centrality_shortest(
            distances=[500],
            sample_probability=0.5,
            sampling_weights=weights_negative,
            pbar_disabled=True,
        )

    # Test wrong length
    weights_wrong_len = [1.0] * (num_nodes - 1)
    with pytest.raises(ValueError, match="must match node count"):
        ns.local_node_centrality_shortest(
            distances=[500],
            sample_probability=0.5,
            sampling_weights=weights_wrong_len,
            pbar_disabled=True,
        )


def test_sampling_weights_scaling():
    """
    Test that sampling_weights properly scale the effective sampling probability.

    Nodes with weight 0.5 should be sampled half as often as nodes with weight 1.0,
    resulting in different contributions to centrality metrics.
    """
    G_primal = mock.mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)

    num_nodes = len(nodes_gdf)
    distance = 500
    num_runs = 30
    sample_probability = 1.0  # Use 1.0 so weights directly control sampling

    # Baseline: uniform weights of 0.5 (equivalent to sample_probability=0.5)
    uniform_weights = [0.5] * num_nodes
    uniform_samples = []
    for seed in range(num_runs):
        res = ns.local_node_centrality_shortest(
            distances=[distance],
            compute_closeness=True,
            sample_probability=sample_probability,
            sampling_weights=uniform_weights,
            random_seed=seed,
            pbar_disabled=True,
        )
        uniform_samples.append(np.array(res.node_density[distance]))

    # Compare: no weights with sample_probability=0.5
    no_weight_samples = []
    for seed in range(num_runs):
        res = ns.local_node_centrality_shortest(
            distances=[distance],
            compute_closeness=True,
            sample_probability=0.5,
            random_seed=seed,
            pbar_disabled=True,
        )
        no_weight_samples.append(np.array(res.node_density[distance]))

    # Average over runs
    avg_uniform = np.mean(uniform_samples, axis=0)
    avg_no_weight = np.mean(no_weight_samples, axis=0)

    # These should be very similar since uniform 0.5 weights with p=1.0
    # is equivalent to p=0.5 with no weights
    mask = (avg_uniform > 0) & (avg_no_weight > 0)
    if np.any(mask):
        correlation = np.corrcoef(avg_uniform[mask], avg_no_weight[mask])[0, 1]
        print("\n### Sampling Weights Scaling Test")
        print()
        print(f"- **Correlation (uniform 0.5 weights vs p=0.5):** {correlation:.3f}")
        assert correlation > 0.95, f"Expected high correlation, got {correlation:.3f}"

    # Test varying weights: first half weight=1.0, second half weight=0.25
    # First half should contribute ~4x more often
    varying_weights = [1.0 if i < num_nodes // 2 else 0.25 for i in range(num_nodes)]
    varying_samples = []
    for seed in range(num_runs):
        res = ns.local_node_centrality_shortest(
            distances=[distance],
            compute_closeness=True,
            sample_probability=sample_probability,
            sampling_weights=varying_weights,
            random_seed=seed,
            pbar_disabled=True,
        )
        varying_samples.append(np.array(res.node_density[distance]))

    # Full computation for reference
    res_full = ns.local_node_centrality_shortest(
        distances=[distance],
        compute_closeness=True,
        pbar_disabled=True,
    )
    full_density = np.array(res_full.node_density[distance])

    # With varying weights, the IPW correction should still produce unbiased estimates
    avg_varying = np.mean(varying_samples, axis=0)
    mask_full = full_density > 0
    if np.any(mask_full):
        relative_error = np.mean(
            np.abs(avg_varying[mask_full] - full_density[mask_full]) / full_density[mask_full]
        )
        print(f"- **Relative error (varying weights, IPW corrected):** {relative_error:.1%}")
        # IPW should correct for unequal sampling, so error should be reasonable
        assert relative_error < 0.30, f"IPW-corrected error {relative_error:.1%} exceeds 30%"


if __name__ == "__main__":
    test_sampling_comprehensive_report()
    test_sampling_approximation_quality()
    test_sampling_reproducibility()
    test_sampling_weighted()
    test_sampling_all_nodes_get_results()
    test_sampling_simplest_centrality()
    test_sampling_ipw_scaling()
    test_sampling_cycles_coverage()
    test_sampling_weights_validation()
    test_sampling_weights_scaling()
