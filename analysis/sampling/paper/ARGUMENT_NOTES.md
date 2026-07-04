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

1. **Canonical grid model (fixed)**: Estimate reach as r = pi \* d^2 / s^2 using a fixed grid spacing s = 175 m. This is held constant, not fitted. For dense networks, actual reach exceeds canonical reach, so the schedule oversamples (conservative). For the sparsest networks (e.g. a low-density US suburb), actual reach falls *below* canonical reach, so the grid model is no longer conservative there — that case is covered by the tolerance epsilon, which is calibrated against it.

2. **Hoeffding bound**: Given canonical reach r, compute the required sample count k = log(2r/delta) / (2 \* epsilon^2), then p = min(1, k/r).

3. **Unified for both metrics**: The same p(d) applies to closeness and betweenness. This is not just for simplicity: a single Brandes-style Dijkstra traversal from each sampled source produces both closeness accumulation and betweenness backpropagation simultaneously. Using the same sampling schedule for both metrics means each source traversal is shared, halving computation time compared to running separate schedules. Although betweenness is noisier in principle, the practical benefit of shared traversals outweighs the marginal gain from metric-specific tuning.

The single calibrated parameter is epsilon (default 0.05): with s fixed, epsilon is the one knob, tuned once so the sparsest calibration network (Cary, NC) preserves rank. Lower epsilon = more samples = slower but more accurate.

4. **Per-node adaptive method (the method proper)**: the canonical schedule is the zero-knowledge baseline. The runtime instead measures per-node reach with a KD-tree pilot (Euclidean counts deflated by 2.5, calibrated against measured Euclidean-to-network ratios on the four networks), assigns per-node inclusion probabilities q = min(1, k(r)/r), reweights per source by 1/q (Horvitz-Thompson, unbiased regardless of pilot quality), and falls back to exact computation per distance whenever powered sampling cannot undercut exact cost. Precision is thereby uniform across dense and sparse areas, which is the comparability property the fixed schedule claimed but could not deliver.

## What We Validate

We are **not** trying to prove that sampled centrality preserves absolute values. We are showing that it preserves **rank ordering** (Spearman rho), which is what matters for the comparative analyses that motivate this work: identifying the most central streets, comparing centrality profiles across neighbourhoods, tracking morphological change.

Specifically:

1. **Epsilon calibration on the binding network** (Cary): epsilon is tuned to the smallest value at which the *sparsest* validated network clears rho >= 0.95 at every distance. That value (0.05) is then fixed and applied to all networks.

2. **Practical guide** (Fig 3): Shows the deterministic schedule across epsilon values so practitioners can choose their operating point. At epsilon = 0.05, sampling engages beyond ~5 km and reaches ~15x speedup at 20 km.

3. **Real-world validation** (Figs 4--6, Tables 2, 4, 5, 6, 7): four networks spanning the density range. Under the canonical baseline: London, Madrid, and Cary confirm rho >= 0.95 at 1--20 km; the held-out Woodlands fails closeness at 20 km (rho = 0.94) for a mechanical reason (reach at 38% of canonical vs 51% for Cary). Under the per-node method (Table 7): all four networks pass, because betweenness sampling is properly powered by measured reach and closeness on low-live-fraction suburbs is correctly routed to exact computation by the work test. The baseline failure is reported, not hidden: it is the evidence that measuring reach matters.

4. **Spatial residuals** (Fig 7): No systematic spatial bias --- the sampling error is spatially uniform, not concentrated in particular areas of the network.

5. **Precision scales with importance**: High-centrality nodes have high reach, hence high effective sample size, hence the best precision. This is a desirable property: the nodes analysts care about most are estimated most accurately.

## The Narrative Arc

1. **Problem**: Exact multi-scale centrality is O(n \* r) per distance, prohibitive at metropolitan scales.
2. **Requirement**: Comparative urban analysis demands a deterministic, network-agnostic sampling schedule so that results are directly comparable within and between cities.
3. **Theoretical grounding**: The Hoeffding/Eppstein-Wang bound, applied to a canonical grid reach model, yields a conservative distance-only schedule p(d).
4. **Practical calibration**: With the grid spacing s fixed, epsilon is the single free parameter. We calibrate it on the sparsest real network (Cary, NC) --- the binding case --- to the smallest value (epsilon = 0.05) that holds rho >= 0.95 at every distance.
5. **Validation**: Three real-world networks spanning the density range confirm rank preservation and demonstrate meaningful speedups.
6. **Implementation**: Released in the open-source cityseer package with user-configurable epsilon.

## Buffer Nodes and the Break-Even Threshold

> Updated for the 4.25 betweenness redesign: betweenness now counts routes from **every** node
> (`live` is an output filter, not a source restriction), and the old per-pair compensation
> (dead-to-dead = 0, live-live = 0.5, live->dead = 1.0) has been removed.

Networks include boundary buffer ("dead") nodes surrounding the analysis area to mitigate edge
effects. The two metrics now treat them differently:

1. **Closeness — live sources in exact mode, all sources under sampling.** Exact mode aggregates
   at the source, so only live nodes need to run. Under sampling the estimator target-aggregates
   with IPW, so buffer nodes must be eligible sources; otherwise boundary live nodes underestimate
   closeness ("edge roll-off").

2. **Betweenness — every node is a source, in both modes.** A shortest path between two buffer
   nodes that passes through the inner area legitimately credits the live intermediate, so all
   nodes source; `live` only filters which nodes' values are reported. Each ordered pair is counted
   with `pair_count = 0.5` (undirected) or `1.0` (directed) — the per-pair 0.5 is the global `/2`
   for the two symmetric orderings. Buffer-to-buffer routes through the interior are counted (with
   buffer >= d_max they are real shortest paths, not truncation artifacts).

3. **Break-even differs by metric.** Closeness exact mode runs only `n_live` traversals, so
   sampling helps when `p < phi` (`phi = n_live / n_total`), with effective speedup `phi / p`.
   Betweenness exact mode now runs all `n_total` sources, so sampling helps whenever `p < 1`, with
   speedup `1/p`. (A closeness-only run skips buffer sources in exact mode; requesting betweenness
   forces every node to traverse.)

4. **Aggregation direction (closeness):**
   - Exact mode: source-based aggregation (accumulates to the source node)
   - Sampling mode: target-based aggregation with IPW (accumulates to the target node, 1/p scaling)

## Key Messages

- **Rank preservation, not absolute accuracy**: The goal is rho >= 0.95, not matching exact centrality values. This is appropriate because urban morphological analysis uses centrality for relative comparison.
- **Deterministic = comparable**: The same epsilon and distance always produce the same p, regardless of the network. This is what makes cross-city analysis valid.
- **Conservative by design**: The canonical grid spacing of 175 m underestimates the reach of most real networks, so the schedule oversamples. This is the right trade-off: slightly less speedup in exchange for robust rank preservation.
- **User control**: Practitioners can adjust epsilon to suit their needs. Tighter tolerance (lower epsilon) for publication-quality analysis; looser tolerance (higher epsilon) for exploratory work.
