---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# Adaptive Sampling for Network Centrality

> **Experimental.** Adaptive sampling is under active development and its API or behaviour may change in future releases.

Computing network centrality at long distance thresholds can be slow because each source node reaches a large portion of the network. Adaptive sampling reduces this cost by using only a subset of nodes as sources, then correcting the results using inverse-probability weighting (IPW) to produce unbiased estimates. The sample size for each distance threshold is determined by the Hoeffding inequality, a concentration bound that guarantees the approximation error stays within a specified tolerance `epsilon`.

## How It Works

Sampling is applied per distance threshold:

1. **Determine sampling probability.** For each distance threshold, estimate how many nodes are reachable using a regular grid model with a default spacing of 175m between intersections (`r = pi * d^2 / s^2`). The Hoeffding bound then converts this reachability estimate into the minimum sampling probability needed to stay within `epsilon` error.
2. **Select sources.** Each node is included as a source independently with the computed probability. Short distances where few nodes are reachable run at full computation, since sampling cannot reduce work. Longer distances use sparser sampling as reachability increases.
3. **Correct results.** Each sampled source's contribution is reweighted by the reciprocal of its inclusion probability (inverse-probability weighting), producing an unbiased estimate of the full computation.

If the computed sampling rate offers no speedup for a given distance, that distance runs at full (exact) computation.

## Accuracy

A single schedule serves both closeness and betweenness: each sampled source traversal computes both metrics at once, so sharing the schedule halves the work relative to sampling each metric separately. The schedule is deterministic and network-agnostic — the same distance always yields the same sampling probability, regardless of the network — which keeps sampled results directly comparable across studies and cities.

The `epsilon` parameter is the single tuning knob, controlling the error tolerance. The default of 0.05 is calibrated empirically on three real street networks spanning the urban density range — Greater London, Madrid, and a low-density US suburb (Cary, NC) — such that node *rankings* are preserved (Spearman ρ ≥ 0.95 against exact computation at 1–20 km) even on the sparsest network; denser networks clear the target comfortably. Tighten `epsilon` for networks sparser than a typical suburb; loosen it (e.g. 0.08–0.1) for exploratory work where approximate rankings suffice. Halving `epsilon` roughly quadruples the number of samples.

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
