---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# Adaptive Sampling for Network Centrality

> **Experimental.** Adaptive sampling is under active development and its API or behaviour may change in future releases.

Computing network centrality at long distance thresholds can be slow because each source node reaches a large portion of the network. Adaptive sampling reduces this cost by using only a subset of nodes as sources, then correcting the results using inverse-probability weighting (IPW) to produce unbiased estimates. The sample size for each distance threshold is determined by the Hoeffding inequality, a concentration bound that guarantees the approximation error stays within a specified tolerance `epsilon`.

## How It Works

Sampling is applied per distance threshold:

1. **Measure reach.** A pilot stage polls the network with a small number of bounded shortest-path traversals (sampled sources, one Dijkstra each) to estimate each node's reach at every requested distance, respecting barriers and dead ends that a straight-line count would miss. Probabilities derive from the lower confidence bound on reach, so estimation error lands on the oversampling side.
2. **Assign per-node probabilities.** The Hoeffding bound converts each node's reach estimate into its own inclusion probability: sparse areas receive high probabilities and dense areas low ones, so every catchment accumulates approximately the required number of effective samples and precision is uniform across the network.
3. **Decide whether sampling pays.** For each distance, the estimated sampled work is compared against exact work; if properly powered sampling would not be cheaper, that distance runs exactly instead of being sampled under-powered.
4. **Correct results.** Each sampled source's contribution is reweighted by the reciprocal of its own inclusion probability (inverse-probability weighting), producing an unbiased estimate of the full computation regardless of how rough the pilot estimate is.

![Per-node reach-based sampling schematic.](/images/sampling_method_schematic.svg) _The pilot measures each node's catchment (A), converts the measurements into per-node inclusion probabilities (B, marker size proportional to probability), and draws sources under those probabilities so each catchment collects the samples it needs (C); a fixed rate spending the same budget oversamples the dense core while missing most of the sparse catchment it should count in full (D)._

![Work test schematic.](/images/sampling_work_test.png) _The per-distance work test: sampling engages only when its predicted cost falls clearly below exact computation (left); where exact computation is already cheap, as for closeness on networks with small live areas, the distance runs exactly (right)._

## Accuracy

A single set of per-node probabilities serves both closeness and betweenness: each sampled source traversal computes both metrics at once, so sharing the sample halves the work relative to sampling each metric separately. Probabilities are derived from measured per-node reach, so they adapt to the network under analysis; pass `random_seed` for reproducible draws.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is calibrated empirically on real street networks spanning the urban density range such that node _rankings_ are preserved: Spearman ρ ≥ 0.95 against exact computation at 1–20 km. Because probabilities are set from measured per-node reach, precision holds in sparse districts and on sparse networks as well as in dense cores; a fourth, very sparse network held out from calibration meets the target at all distances under this method. Loosen `epsilon` (for example 0.08 to 0.1) for exploratory work where approximate rankings suffice; halving `epsilon` roughly quadruples the number of samples. Speedups are largest on dense networks at long distances; on sparse networks with small live areas the work test will often select exact computation, which is the correct outcome rather than a missed optimisation.

The full methodology and validation are documented in the sampling study under [`analysis/sampling/`](https://github.com/benchmark-urbanism/cityseer-api/tree/master/analysis/sampling) in the repository.

## Usage

With the high-level API:

```python
cn.centrality_shortest(
    distances=[500, 2000, 5000, 20000],
    sample=True,  # epsilon=0.05 default; pass epsilon=... to override
)
```

Or with the lower-level functional API:

```python
from cityseer.metrics import networks

nodes_gdf = networks.centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[500, 2000, 5000, 20000],
    sample=True,
)
```

## API Reference

- [`centrality_shortest`](/metrics/networks#centrality_shortest)
- [`centrality_simplest`](/metrics/networks#centrality_simplest)

</section>
