---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# Adaptive Sampling for Network Centrality

Computing network centrality metrics across multiple distance thresholds (e.g., 500m to 20km) can be computationally expensive. `cityseer` provides **adaptive per-distance sampling** that automatically calibrates sampling probability for each distance threshold based on network reachability.

## The Problem

When using uniform sampling across all distance thresholds, a fundamental tension arises:

| Distance | Typical Reach | With Uniform 20% Sampling |
|----------|---------------|---------------------------|
| 500m     | ~100 nodes    | eff_n = 20, poor accuracy |
| 1000m    | ~300 nodes    | eff_n = 60, marginal accuracy |
| 5000m    | ~2000 nodes   | eff_n = 400, good accuracy |
| 20000m   | ~10000 nodes  | eff_n = 2000, excellent accuracy |

- **Short distances** (low reach) have insufficient effective sample size, resulting in poor ranking accuracy
- **Long distances** (high reach) are over-sampled, wasting computation that could be reduced

## The Solution: Adaptive Per-Distance Sampling

The adaptive approach works in three steps:

1. **Probe reachability** at each distance threshold using a small sample of nodes
2. **Compute required sampling probability** for each distance to achieve a target accuracy (Spearman ρ ≥ 0.95)
3. **Run separate Dijkstra computations** for each distance with calibrated sampling

This means:

- Short distances use **full or near-full computation** (where reach is low, sampling doesn't help)
- Long distances use **aggressive sampling** (where high reach provides statistical power)

## Usage

```python
from cityseer.metrics import networks

# Standard approach (uniform sampling)
nodes_gdf = networks.node_centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[500, 2000, 5000, 20000],
    sample_probability=0.2,  # Same p for all distances
)

# Adaptive approach (per-distance calibration)
nodes_gdf = networks.node_centrality_shortest_adaptive(
    network_structure,
    nodes_gdf,
    distances=[500, 2000, 5000, 20000],
    target_rho=0.95,  # Target accuracy level
)
```

## Empirical Models

The expected Spearman ρ is predicted using fitted hyperbolic models of the form `ρ = 1 - A / (B + effective_n)`, where `effective_n = mean_reachability × sample_probability`.

Separate models are fitted for harmonic (closeness) and betweenness centrality based on 10th percentile estimates across network topologies. Betweenness exhibits higher variance than harmonic, so its model is more conservative. When computing both metrics together, the betweenness model is used to ensure both achieve the target accuracy.

## Performance Results

Tests on synthetic network topologies show:

| Topology | Full (s) | Adaptive (s) | Speedup | Harmonic ρ | Betweenness ρ |
|----------|----------|--------------|---------|------------|---------------|
| trellis  |     3.00 |         1.44 |    2.1x |       0.96 |          1.00 |
| tree     |     1.39 |         0.76 |    1.8x |       1.00 |          0.99 |
| linear   |     3.09 |         1.55 |    2.0x |       1.00 |          1.00 |

Adaptive sampling achieves **consistent accuracy (ρ ≥ 0.95) across all distances** while providing **1.8-2.1x speedup**.

## Recommendations

1. **Use adaptive sampling for multi-scale analyses** spanning short to long distances
2. **Set `target_rho=0.95`** for general use, or `target_rho=0.97+` if betweenness accuracy is critical
3. **For single-distance computations**, standard uniform sampling remains appropriate
4. **For very large networks** (>50,000 nodes), adaptive sampling provides substantial speedups while maintaining accuracy guarantees

## API Reference

- [`node_centrality_shortest_adaptive`](/metrics/networks#node-centrality-shortest-adaptive) - Adaptive shortest-path centrality
- [`node_centrality_simplest_adaptive`](/metrics/networks#node-centrality-simplest-adaptive) - Adaptive simplest-path (angular) centrality

## Technical Details

For the full methodology, empirical model fitting, and test results, see the [analysis documentation on GitHub](https://github.com/benchmark-urbanism/cityseer-api/blob/master/analysis/adaptive_sampling_results.md).

</section>
