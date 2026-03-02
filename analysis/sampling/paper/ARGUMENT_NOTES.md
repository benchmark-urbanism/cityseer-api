# Paper Argument: One-Page Summary

## The Problem

Localised network centrality (closeness, betweenness within distance thresholds) is fundamental to comparative urban morphological analysis. At metropolitan scales (10--20 km), exact computation becomes prohibitively expensive because each source traversal must explore tens of thousands of reachable nodes. Source sampling can reduce cost, but introduces a design question: how much to sample?

## The Core Requirement: Deterministic Comparability

The central constraint --- and the paper's motivating insight --- is that urban analysts routinely compare centrality patterns **within and between cities**. This means:

- Sampling must be **deterministic**: the same analysis distance must produce the same sampling probability regardless of which network is being analysed.
- Sampling must be **network-agnostic**: no per-network calibration, no dependence on local graph density or node count.
- The schedule must be **conservative enough** to preserve rank ordering across heterogeneous morphologies, but **not so conservative** that it eliminates the computational benefit of sampling.

Without deterministic comparability, sampled centrality values from different locations would reflect different noise levels, making cross-city or cross-neighbourhood comparison meaningless.

## The Solution: Distance-Only Schedule via Canonical Grid

We construct a single function p(d) that converts analysis distance to sampling probability:

1. **Canonical grid model**: Estimate reach as r = pi \* d^2 / s^2 using a fixed grid spacing s = 175 m (representative of sparse street networks). This is intentionally conservative: real urban networks are typically denser, so actual reach exceeds canonical reach, meaning the schedule oversamples relative to what the network requires.

2. **Hoeffding bound**: Given canonical reach r, compute the required sample count k = log(2r/delta) / (2 \* epsilon^2), then p = min(1, k/r).

3. **Unified for both metrics**: The same p(d) applies to closeness and betweenness. This is not just for simplicity: a single Brandes-style Dijkstra traversal from each sampled source produces both closeness accumulation and betweenness backpropagation simultaneously. Using the same sampling schedule for both metrics means each source traversal is shared, halving computation time compared to running separate schedules. Although betweenness is noisier in principle, the practical benefit of shared traversals outweighs the marginal gain from metric-specific tuning.

The user-facing parameter is epsilon (default 0.06), which controls the error--speed trade-off. Lower epsilon = more conservative = more samples = slower but more accurate.

## What We Validate

We are **not** trying to prove that sampled centrality preserves absolute values. We are showing that it preserves **rank ordering** (Spearman rho), which is what matters for the comparative analyses that motivate this work: identifying the most central streets, comparing centrality profiles across neighbourhoods, tracking morphological change.

Specifically:

1. **Epsilon sweep on synthetic networks** (Fig 1): At epsilon = 0.06, rho >= 0.95 across trellis, tree, and linear topologies --- covering the structural range of real street networks.

2. **Practical guide** (Fig 3): Shows the deterministic schedule across epsilon values so practitioners can choose their operating point. At epsilon = 0.06, sampling kicks in beyond ~5 km and reaches ~20x speedup at 20 km.

3. **Real-world validation** (Figs 4--6, Tables 2, 4): Greater London (~295k nodes) and Greater Madrid confirm rho >= 0.95 at distances from 1--20 km for both closeness and betweenness.

4. **Spatial residuals** (Fig 7): No systematic spatial bias --- the sampling error is spatially uniform, not concentrated in particular areas of the network.

5. **Precision scales with importance**: High-centrality nodes have high reach, hence high effective sample size, hence the best precision. This is a desirable property: the nodes analysts care about most are estimated most accurately.

## The Narrative Arc

1. **Problem**: Exact multi-scale centrality is O(n \* r) per distance, prohibitive at metropolitan scales.
2. **Requirement**: Comparative urban analysis demands a deterministic, network-agnostic sampling schedule so that results are directly comparable within and between cities.
3. **Theoretical grounding**: The Hoeffding/Eppstein-Wang bound, applied to a canonical grid reach model, yields a conservative distance-only schedule p(d).
4. **Practical calibration**: We sweep epsilon on synthetic networks to identify a regime (epsilon = 0.06) that reliably achieves rho >= 0.95 across diverse topologies without excessive oversampling.
5. **Validation**: Two large real-world networks confirm rank preservation and demonstrate meaningful speedups.
6. **Implementation**: Released in the open-source cityseer package with user-configurable epsilon.

## Key Messages

- **Rank preservation, not absolute accuracy**: The goal is rho >= 0.95, not matching exact centrality values. This is appropriate because urban morphological analysis uses centrality for relative comparison.
- **Deterministic = comparable**: The same epsilon and distance always produce the same p, regardless of the network. This is what makes cross-city analysis valid.
- **Conservative by design**: The canonical grid spacing of 175 m underestimates the reach of most real networks, so the schedule oversamples. This is the right trade-off: slightly less speedup in exchange for robust rank preservation.
- **User control**: Practitioners can adjust epsilon to suit their needs. Tighter tolerance (lower epsilon) for publication-quality analysis; looser tolerance (higher epsilon) for exploratory work.
