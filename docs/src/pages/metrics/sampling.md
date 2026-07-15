---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# Adaptive Sampling

> **Experimental.** Adaptive sampling is under active development and its API or behaviour may change in future releases.

Centrality runs a shortest-path traversal from every node in the network. At long distance thresholds each traversal spans a large part of the network, so run time grows steeply with the threshold: a city-wide analysis at 10 or 20 km can take hours where an 800 m analysis takes seconds.

Adaptive sampling reduces this cost by running the traversals from a subset of nodes. Each node is given its own probability of being used as a source, and every sampled source's contribution is multiplied by the reciprocal of that probability (inverse-probability weighting), so the estimates remain unbiased. The probabilities are set with the Hoeffding inequality, a statistical bound on how far a sample can stray from the true value, so that the approximation error stays within a chosen tolerance `epsilon`.

## How it works

Sampling is decided separately for each distance threshold:

1. **Measure reach.** A pilot runs one bounded traversal from each of a small sample of source nodes. Each node's reach is then estimated from how many of those pilot sources reach it: a node reached by many of them has a large catchment, a node reached by few has a small one. Because the pilot travels the network itself, barriers, dead ends, and disconnected fringes that a straight-line count would miss are respected. Reach is taken from the cautious (lower) end of its confidence interval, so estimation error leads to slightly more sampling rather than less.
2. **Set per-node probabilities.** The Hoeffding bound converts each reach estimate into a sampling probability. Dense areas can be sampled sparsely because each catchment holds many candidate sources; sparse areas are sampled at high probability, often 1. Every catchment then collects roughly the number of samples it needs, and precision is uniform across the network.
3. **Check that sampling pays.** The predicted cost of sampling (pilot plus sampled traversals) is compared with the cost of exact computation. If sampling would not be clearly cheaper, that threshold runs exactly.
4. **Compute and correct.** Sources are drawn under the assigned probabilities and each source's contributions are reweighted by the reciprocal of its probability. The result is unbiased even where the pilot's reach estimates are rough.

![Per-node reach-based sampling schematic.](/images/sampling_method_schematic.svg) _A pilot measures each node's catchment (A) and converts the measurement into a sampling probability (B). Drawing sources under those probabilities supplies every catchment with roughly the samples it needs (C). A single fixed rate spending the same budget oversamples the dense core and starves the sparse catchment (D)._

## When sampling engages

The check in step 3 runs per distance threshold, so a single call can mix modes: long thresholds on a large network are sampled while short thresholds run exactly. Networks with small live areas often run exactly at every threshold because exact computation is already cheap there. This is the intended behaviour, not a missed optimisation.

![Work test schematic.](/images/sampling_work_test.png) _The per-distance decision: sampling engages only when its predicted cost falls clearly below exact computation (left). Where exact computation is already cheap, the threshold runs exactly (right)._

## Accuracy

The `epsilon` parameter sets the error tolerance. The default of 0.05 is calibrated on real street networks spanning the urban density range so that node rankings are preserved: Spearman ρ ≥ 0.95 against exact computation at thresholds from 1 to 20 km. Because probabilities are derived from measured reach, precision holds in sparse districts and on sparse networks as well as in dense cores.

Loosen `epsilon` (for example 0.08 to 0.1) for exploratory work where approximate rankings suffice. Tightening it is expensive: halving `epsilon` roughly quadruples the number of samples.

One set of probabilities serves both closeness and betweenness, so computing both from the same call shares the sampled traversals. Pass `random_seed` for reproducible draws.

## Usage

With the recommended [`CityNetwork`](/api/network) interface:

```python
cn.centrality_shortest(
    distances=[500, 2000, 5000, 20000],
    sample=True,  # epsilon=0.05 default; pass epsilon=... to override
)
```

The lower-level [`metrics.networks`](/metrics/networks) functions accept the same `sample` and `epsilon` arguments:

```python
from cityseer.metrics import networks

nodes_gdf = networks.centrality_shortest(
    network_structure,
    nodes_gdf,
    distances=[500, 2000, 5000, 20000],
    sample=True,
)
```

## Further reading

- [`centrality_shortest`](/metrics/networks#centrality_shortest) and [`centrality_simplest`](/metrics/networks#centrality_simplest) API reference
- The full methodology and validation are documented under [`analysis/sampling/`](https://github.com/benchmark-urbanism/cityseer-api/tree/master/analysis/sampling) in the repository

</section>
