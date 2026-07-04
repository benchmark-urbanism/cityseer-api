---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# Adaptive Sampling for Network Centrality

> **Experimental.** Adaptive sampling is under active development and its API or behaviour may change in future releases.

Computing network centrality at long distance thresholds can be slow because each source node reaches a large portion of the network. Adaptive sampling reduces this cost by using only a subset of nodes as sources, then correcting the results using inverse-probability weighting (IPW) to produce unbiased estimates. The sample size for each distance threshold is determined by the Hoeffding inequality, a concentration bound that guarantees the approximation error stays within a specified tolerance `epsilon`.

## How It Works

Sampling is applied per distance threshold:

1. **Measure reach.** A pilot stage counts, for each node, the nodes within the straight-line radius (a KD-tree query, no network traversal) and converts the count into a conservative estimate of that node's network catchment.
2. **Assign per-node probabilities.** The Hoeffding bound converts each node's reach estimate into its own inclusion probability: sparse areas receive high probabilities and dense areas low ones, so every catchment accumulates approximately the required number of effective samples and precision is uniform across the network.
3. **Decide whether sampling pays.** For each distance, the estimated sampled work is compared against exact work; if properly powered sampling would not be cheaper, that distance runs exactly instead of being sampled under-powered.
4. **Correct results.** Each sampled source's contribution is reweighted by the reciprocal of its own inclusion probability (inverse-probability weighting), producing an unbiased estimate of the full computation regardless of how rough the pilot estimate is.

## Accuracy

A single schedule serves both closeness and betweenness: each sampled source traversal computes both metrics at once, so sharing the schedule halves the work relative to sampling each metric separately. The schedule is deterministic and network-agnostic; the same distance always yields the same sampling probability regardless of the network, which keeps sampled results directly comparable across studies and cities.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is calibrated empirically on real street networks spanning the urban density range (Greater London, Madrid, and a low-density US suburb, Cary NC) such that node *rankings* are preserved: Spearman ρ ≥ 0.95 against exact computation at 1–20 km. Because probabilities are set from measured per-node reach, precision holds in sparse districts and on sparse networks as well as in dense cores; a fourth network held out from calibration (The Woodlands, TX, a very sparse dendritic suburb) meets the target at all distances under this method. Loosen `epsilon` (for example 0.08 to 0.1) for exploratory work where approximate rankings suffice; halving `epsilon` roughly quadruples the number of samples. Speedups are largest on dense networks at long distances; on sparse networks with small live areas the work test will often select exact computation, which is the correct outcome rather than a missed optimisation.

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
