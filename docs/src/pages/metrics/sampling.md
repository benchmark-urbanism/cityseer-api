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

Separate accuracy models are fitted for closeness and betweenness centrality. Betweenness centrality is more sensitive to which nodes are included in the sample (its sampling error has higher variance than closeness), so the betweenness model requires more samples at the same distance threshold. When computing both metrics together, the betweenness model is used to ensure both achieve the target accuracy.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is suitable for most analyses.

## Usage

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

- [`centrality_shortest`](/metrics/networks#centrality-shortest)
- [`centrality_simplest`](/metrics/networks#centrality-simplest)

</section>
