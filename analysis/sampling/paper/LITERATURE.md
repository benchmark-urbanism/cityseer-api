# Literature Survey: Sampling for Centrality Approximation

How existing methods work, what they guarantee, and how they apply to localised (distance-bounded) centrality.

---

## The computational constraint

Localised centrality runs a distance-bounded Dijkstra from each source node. Each source's Dijkstra discovers all nodes within distance d, contributing to their centrality estimates. Sampling means running Dijkstra from a fraction p of sources. The question: how small can p be while preserving accuracy?

This is **source-sampling**: the unit of computation is a full Dijkstra tree, not a single shortest path. This distinction matters because the existing literature splits into two families with fundamentally different computational primitives.

---

## Source-sampling methods

### Eppstein & Wang (2004)

**What it does.** Sample k nodes uniformly at random, run SSSP from each, estimate closeness of every node from the k samples.

**The bound.** Each source contributes a bounded random variable (distance in [0, Δ_G] where Δ_G is the diameter). By Hoeffding's inequality applied to each target node v, then union-bounded over all n nodes:

```
k = O( log(n/δ) / ε² )
```

This guarantees |farness_hat(v) - farness(v)| ≤ ε · Δ_G for ALL v simultaneously, with probability ≥ 1 − δ.

**Localisation.** Replacing n with r and Δ_G with d_max:

```
k_local = O( log(r/δ) / ε² )
```

This is valid because: (a) Hoeffding applies to any bounded variable, (b) only r nodes receive contributions from a given source, (c) distances are bounded by d_max. The log(r) grows very slowly, so the bound is dominated by 1/ε².

**Why it's too conservative.** At ε = 0.1, δ = 0.05:

| Reach r | k (EW) | p = k/r | Speedup |
| ------- | ------ | ------- | ------- |
| 100     | ~760   | 1.0     | none    |
| 1,000   | ~990   | 0.99    | none    |
| 10,000  | ~1,220 | 0.12    | ~8×     |
| 50,000  | ~1,380 | 0.028   | ~36×    |

At ε = 0.05, sample counts roughly quadruple. The bound guarantees uniform additive error at every node, paying for worst-case peripheral nodes whose exact values don't matter. Speedups only emerge at very high reach.

**What it guarantees vs what we want.** EW guarantees additive ε at every node. We want Spearman ρ ≥ 0.95 (rank preservation). These are structurally different: Spearman is tolerant of large additive errors on low-centrality nodes (rank swaps among near-zero values contribute little) and concentrates sensitivity on high-centrality nodes (where estimates are most precise under source-sampling). The EW guarantee is sufficient for rank preservation but massively overpays for it.

### Brandes & Pich (2007)

**What it does.** Source-sampling with various pivot selection strategies (random, maximum degree, maximum closeness). No formal guarantee — "use k sources and hope for the best."

**Relevance.** Establishes source-sampling as practical. Shows that random selection works surprisingly well. Does not provide sample-size calibration. Our work fills this gap with the √r model.

### Bader, Kintali, Madduri & Mihail (2007)

**What it does.** Adaptive source-sampling for estimating the betweenness of a SINGLE vertex v. Sample sources, accumulate dependency scores, stop when the accumulated sum exceeds c · n.

**The bound.** Multiplicative: |b_hat(v) − b(v)| ≤ ε · b(v) with probability ≥ 1 − 1/n. Sample complexity: O(1/(ε² · b(v))), independent of n for fixed b(v). High-centrality nodes need few samples; low-centrality nodes need many.

**Localisation.** Adapts straightforwardly: replace n with r in the stopping criterion. The 1/b(v) dependence means low-centrality nodes are hard — but these are precisely the nodes Spearman doesn't care about. This is conceptually aligned with our approach.

**Key insight for us.** The adaptive principle — fewer samples for important nodes — is the same structural property our model exploits. Bader makes it explicit for one vertex; our model achieves it implicitly across all vertices via the reach-dependent sampling rate.

### Geisberger, Sanders & Schultes (2008)

**What it does.** Source-sampling with Brandes backpropagation, like Brandes-Pich, but with bias correction. Addresses the overestimation problem near the sampled source (nodes close to the source tend to lie on many of that source's shortest paths, inflating their betweenness estimate).

**Key innovations:**
- **Linear scaling:** Weights dependency contributions by distance from the source to reduce overestimation of near-source nodes.
- **Bisection method:** Only counts betweenness contributions for a vertex if it is at least half the distance between the source and target on the shortest path, eliminating trivially close contributions.
- **Unbiased estimator:** The paper proves their method provides an unbiased estimator of betweenness centrality, whereas naive n/k scaling (Brandes-Pich) can be biased in practice.

**Result:** Euclidean error reduced by a factor of 4 at the same runtime, or 16× faster for the same error.

**Relevance.** Shows that the naive n/k scaling used in Brandes-Pich has systematic bias, and that unbiased estimation requires care. Our approach uses Horvitz-Thompson (1/p) weighting, which is the standard unbiased estimator for inclusion-probability sampling — a different but principled approach to the same problem.

---

## Path-sampling methods

### Riondato & Kornaropoulos (2016)

**What it does.** Sample random (s, t) pairs uniformly, compute one shortest path per pair, increment betweenness estimates for internal vertices.

**The VC-dimension bound.** Define the set system: for each vertex w, the function f_w(s,t) = σ_st(w)/σ_st (fraction of shortest paths through w). The VC-dimension of this family is:

```
VCdim ≤ ⌊log₂(VD − 2)⌋ + 1
```

where VD is the vertex diameter (max vertices on any shortest path). By the ε-sample theorem:

```
τ = O( (log₂(VD) + log(1/δ)) / ε² )
```

**Why it doesn't adapt to localised analysis.**

1. **Different primitive.** Path-sampling computes ONE path per sample. Source-sampling computes a full Dijkstra tree. These are not interchangeable — a single path tells you about one (s,t) pair; a Dijkstra tree tells you about one source and ALL its reachable targets.

2. **VD is global.** The vertex diameter is a global graph property. For localised analysis, you'd need VD(d_max) — the max vertices on any shortest path of metric length ≤ d_max. This is a different quantity and requires separate analysis.

3. **Changed distribution.** Localised path-sampling would sample (s,t) pairs with d(s,t) ≤ d_max, not from V × V. The VC analysis would need to be re-derived for this restricted domain.

**Could you adapt it?** In principle, yes. Define localised path-sampling: sample (s,t) with d(s,t) ≤ d_max. On street networks with d_max = 5km, VD(d_max) might be 50–200, giving log₂(VD) ~ 6–8. Sample complexity: O(8/ε²) ~ 800 at ε = 0.1. But each sample requires finding a random pair within distance d_max and computing a shortest path — operationally different from source-sampling and potentially slower per sample.

**The deeper issue.** Even if you could adapt RK to localised analysis, it still targets additive ε for betweenness. For closeness, there is no VC-dimension result at all (RK is betweenness-specific). Our setting computes both metrics from the same Dijkstra trees, so source-sampling is the natural primitive.

### Borassi & Natale (2019) — KADABRA

**What it does.** Path-sampling with adaptive per-vertex stopping. Uses martingale concentration (Freedman's inequality) to maintain running confidence intervals for each vertex's betweenness. Stops when all intervals are narrow enough.

**Per-vertex guarantee.** |b_hat(v) − b(v)| ≤ λ for ALL v with probability ≥ 1 − δ. Non-uniform error budget allocation: more δ budget to low-centrality vertices, tightening high-centrality estimates.

**Balanced bidirectional BFS.** Key efficiency: to sample one shortest path, BFS runs from both s and t simultaneously, expanding whichever frontier has smaller total degree. On power-law-like graphs, this achieves O(|E|^{1/2}) per path.

**Adaptation to source-sampling.** The confidence intervals depend on path-sampling independence (each sample is an independent Bernoulli for each vertex). In source-sampling, a single Dijkstra contributes to ALL reachable nodes simultaneously, creating correlations. The martingale analysis breaks.

However, the PRINCIPLE of monitoring accuracy online is valuable. One could imagine: run Dijkstra from k sources, compute running Spearman ρ against a reference (e.g., first k/2 vs second k/2), stop when the rank order stabilises. This is not what we do (we pre-compute p), but it's a direction worth noting.

### Pellegrina & Vandin (2023) — SILVAN

**What it does.** Path-sampling with progressive (doubling) sample sizes and data-dependent bounds using empirical Rademacher complexity.

**Rademacher bounds.** The Rademacher complexity of function class F on sample S measures how well F can fit random noise:

```
R(F, S) = E_σ [ sup_{f∈F} (1/m) Σ σᵢ f(cᵢ) ]
```

The generalisation bound: with probability ≥ 1 − δ, the maximum deviation of any function's sample mean from its true mean is at most 2R(F,S) + O(√(log(1/δ)/m)).

**Why Rademacher is tighter than VC.** VC-dimension is a worst-case combinatorial property. Rademacher averages are computed from the actual data, capturing the specific graph's structure. On well-structured graphs (like street networks), actual shattering capacity is far below the VC bound.

**Non-uniform bounds via empirical peeling.** SILVAN partitions vertices by estimated variance, computing separate Rademacher bounds per group. Low-variance vertices (low betweenness) get much tighter bounds.

**Relevance.** SILVAN's philosophy directly supports our empirical finding: street networks have far less variance than worst-case theory predicts. But the machinery is for path-sampling, not source-sampling. The variance-dependent philosophy could inspire source-sampling analogues.

---

## Target-sampling / hybrid methods

### Cohen, Delling, Pajor & Werneck (2014)

**What it does.** Three approaches for global closeness:

1. **Sampling** (Eppstein-Wang style): high variance, no per-node control.
2. **Pivoting**: for each non-sampled node v, find closest sampled node c(v). Estimate v's closeness using c(v)'s distances. By triangle inequality: |d(v,u) − d(c(v),u)| ≤ d(v,c(v)). Error bounded by distance-to-pivot (a structural, not statistical, guarantee).
3. **Hybrid**: exact for nearby nodes, pivot for distant nodes.

**Per-node guarantees.** Pivoting provides bounded relative error for any instance — a property sampling cannot match. But it only works for closeness (not betweenness) and requires the triangle inequality on distances.

**Relevance to localised analysis.** Harmonic closeness (sum of 1/d) does not satisfy the triangle inequality in the same way, so pivoting bounds don't transfer. More fundamentally, Cohen's method targets global closeness on a single graph, not multi-scale localised centrality.

---

## What the literature tells us about localised source-sampling

### What's known

1. **EW adapts directly.** Replace n with r, Δ_G with d_max. The bound is valid but conservative. This is the only formal guarantee available for source-sampling of localised centrality.

2. **Path-sampling methods don't adapt.** RK, KADABRA, SILVAN all use a different computational primitive (single path vs. Dijkstra tree) and rely on global graph properties (vertex diameter, VC-dimension). Converting between the two primitives is non-trivial.

3. **No formal link exists between additive ε and Spearman ρ.** The relationship depends on the distribution of centrality values. On skewed distributions (typical of street networks), small additive errors preserve ranks well.

### What's not known (and where the gap is)

1. **No rank-preservation bounds for source-sampling.** Nobody has derived the sample complexity for achieving Spearman ρ ≥ ρ₀. This is theoretically hard because ρ depends on the entire joint distribution of true and estimated values, not just pointwise errors.

2. **No variance-dependent source-sampling bounds.** SILVAN shows that data-dependent bounds are much tighter than worst-case on structured graphs. An analogous result for source-sampling — using the actual variance of source contributions rather than worst-case Hoeffding — does not exist.

3. **No localised path-sampling analysis.** The VC-dimension of the localised set system (restricted to paths within d_max) has not been derived. This could yield tighter bounds than localised EW.

---

## Could there be something better than the empirical √r model?

### Why √r arises

The Hájek variance of the estimated centrality of target v under uniform source-sampling with probability p is:

```
Var_v ≈ (1 − p) · σ²_v / (p · r)
```

where σ²_v is the population variance of source contributions to v. If σ²_v grows linearly in r (plausible: more sources means more variation in their contributions, and the total grows as O(r)), then:

```
Var_v ≈ (1 − p) · c / p
```

For rank preservation, we need Var_v / μ²_v (the squared CV) to be below some threshold. If μ_v also grows with r (for closeness: μ_v ∝ r; for betweenness: μ_v ∝ r²/n or similar), then the required p decreases with r. The exact rate depends on how σ²_v and μ_v scale with r.

For closeness (μ_v ∝ r, σ²_v ∝ r): CV² ∝ 1/(p · r), so p ∝ 1/r for fixed CV. This is FASTER than √r — suggesting closeness needs fewer samples than our model prescribes.

For betweenness (μ_v ∝ r · betweenness_fraction, σ²_v harder to characterise): the variance depends on path structure, not just reachability. This is why betweenness consistently needs more samples.

### Theoretical alternatives to consider

**1. Variance-dependent EW (tightest available formal bound)**

Instead of using worst-case Hoeffding (which assumes bounded variables), use Bernstein's inequality, which accounts for variance:

```
Pr[ |X̄ − μ| ≥ t ] ≤ 2 exp( −k·t² / (2σ² + 2bt/3) )
```

where b is the max absolute value and σ² is the variance. For source contributions to closeness, σ² is typically much smaller than b² (most contributions are modest; a few are large). This could give a tighter bound than Hoeffding without requiring empirical calibration.

For localised analysis: estimate σ²_v from a pilot sample, then use Bernstein to set the total sample count. This would be a formal variance-dependent bound, tighter than EW, applied to source-sampling. It wouldn't target Spearman ρ directly, but it would size samples more efficiently.

**2. Localised path-sampling (new theoretical direction)**

Define the localised path-sampling problem: sample (s,t) pairs with d(s,t) ≤ d_max, compute one shortest path, increment betweenness estimates. The localised vertex diameter VD(d_max) is much smaller than the global VD. On a street network with d_max = 5km:

- Typical path might traverse 50–200 nodes
- VD(d_max) ~ 200
- log₂(VD(d_max)) ~ 8
- Sample complexity: O(8/ε²) for additive ε

This is competitive with our model at moderate reach and has a formal guarantee. But:

- Only for betweenness (no closeness)
- Requires a way to sample random (s,t) pairs within d_max efficiently
- Each sample computes one path, not a full Dijkstra tree — so closeness cannot piggyback
- The per-sample cost is lower than source-sampling, but you need to run this separately from closeness computation

Since localised analysis typically computes BOTH closeness and betweenness from the same Dijkstra trees, path-sampling would only help betweenness and would require a separate computation. This is a poor fit for the implementation architecture.

**3. Per-node confidence intervals via concentration (new formal approach)**

Instead of targeting aggregate Spearman ρ, derive per-node confidence intervals for the centrality estimate under source-sampling. For target v with reach r_v:

- The centrality estimate is a sum of p · r_v independent contributions (in expectation)
- By CLT: the estimate is approximately normal for p · r_v ≥ 30
- The confidence interval width scales as σ_v / √(p · r_v)

Nodes with high r_v automatically get narrow intervals. Nodes with low r_v get wide intervals but also have low centrality (on street networks), so rank swaps among them are inconsequential.

This provides a _distribution_ of per-node accuracies rather than a single aggregate ρ. The connection to Spearman ρ: if the top-k nodes all have narrow intervals (relative to the gap between their true values), then ρ ≈ 1. This is formalizable but requires knowing the centrality distribution, which is unknown a priori.

**4. Online adaptive stopping (practical improvement)**

Rather than pre-computing p from the model, run sources one at a time, maintaining a running estimate of rank stability. Stop when adding more sources doesn't change the ranking (measured by Spearman ρ between estimates from first half and second half of samples, or between successive batches).

This avoids needing a model entirely — the data tells you when to stop. Downsides:

- Cannot parallelise as easily (need to check convergence)
- No formal guarantee on the final ρ
- Our current model is simple and works; adaptive stopping is complex

### Assessment

The √r model is a reasonable empirical choice that sits between:

- **Too conservative**: EW bound (O(log r / ε²), no practical speedup at moderate reach)
- **No guarantee**: Brandes-Pich (pick k, hope for the best)
- **Wrong primitive**: RK/KADABRA/SILVAN (path-sampling, doesn't compute closeness)

The most promising theoretical improvement would be **variance-dependent source-sampling** (Bernstein rather than Hoeffding), which would formally tighten the EW bound by accounting for the actual variance structure of street networks. This would not target Spearman ρ directly, but it would narrow the gap between the formal bound and empirical practice.

The most promising practical improvement would be **online adaptive stopping**, which avoids pre-specifying p at all.

Neither of these invalidates the current √r model — they represent refinements. The √r scaling has theoretical motivation (Hájek variance), empirical support (α̂ = 0.544, p = 0.584 for H0: α = 0.5), and practical validation (ρ ≥ 0.985 on two real networks). The floor (n_min = 300) addresses the finite-sample regime where Spearman ρ itself becomes degenerate. Together, they provide a simple, conservative, and validated calibration.

---

## Per-metric variance scaling: why one model for two metrics is a compromise

The √r model uses a single exponent for both harmonic closeness and betweenness. But the variance structure is fundamentally different:

### Closeness (harmonic)

Each source s contributes 1/d(s,v) to target v's closeness. Under sampling with probability p:

- μ*v = Σ*{s reachable} 1/d(s,v) ∝ r (grows linearly with reach)
- σ²_v = Var of source contributions ∝ r (more sources → more variance, but each term bounded by 1/d_min)
- CV² = σ²_v / μ²_v ∝ 1/r (decreases with reach)
- Required p for fixed CV: p ∝ 1/r

This suggests closeness scaling is FASTER than √r — closeness might need eff_n ∝ r^α with α < 0.5, or equivalently p ∝ r^{α-1} with α < 0.5. The empirical data supports this: closeness consistently achieves higher ρ than betweenness at the same p.

### Betweenness

Each source s contributes dependency δ_s(v) to target v's betweenness. The dependency is 0 for most sources and can be large for sources whose shortest paths pass through v:

- μ*v = Σ*{s reachable} δ_s(v) — depends on path structure, not just reachability
- σ²_v = Var of dependencies — heavy-tailed because a few sources contribute large dependencies
- The variance is NOT simply proportional to r; it depends on the topological role of v

Bridge nodes (high betweenness, moderate reachability) have high-variance estimates because a few sources contribute large dependencies while most contribute zero. This is why betweenness needs more samples and why the Hájek argument is less clean for betweenness.

### Implication

The single √r model is a conservative compromise. It's calibrated at the 75th percentile across both metrics, which in practice means it's sized for betweenness (the harder metric). Closeness gets oversampled, which explains why closeness ρ is consistently higher than betweenness ρ in validation.

A per-metric model would be more efficient: lower k for closeness, higher k for betweenness. The paper currently uses a single k = 10.16 for both. Whether the complexity of two models is worth the efficiency gain is a practical question — the speedup improvement would be modest because betweenness is the binding constraint at every distance.

---

## The EW bound applied to our setting: worked example

To make the gap between EW and our model concrete, here is a worked example at 10km on GLA (reach = 11,590):

**EW bound (additive ε):**

- Target: |farness_hat(v) − farness(v)| ≤ ε · d_max for all v, with probability ≥ 0.95
- At ε = 0.1: k = log(11,590 / 0.05) / (2 × 0.01) = log(231,800) / 0.02 ≈ 12.35 / 0.02 ≈ 618
- p = 618 / 11,590 = 0.053, speedup ≈ 19×
- At ε = 0.05: k ≈ 2,470, p = 0.213, speedup ≈ 5×

**Our model (Spearman ρ ≥ 0.95):**

- eff_n = max(10.16 × √11,590, 300) = max(1,094, 300) = 1,094
- p = 1,094 / 11,590 = 0.094, speedup ≈ 10.6×
- Observed ρ: 0.996 (closeness), 0.989 (betweenness) — well above 0.95

At ε = 0.1, the EW bound actually prescribes FEWER samples than our model (618 vs 1,094), but it guarantees something weaker in practice (uniform additive error scaled by d_max, which is loose for closeness where values are sums of 1/d). At ε = 0.05 it prescribes more (2,470 vs 1,094).

The key insight: the EW bound and our model are targeting different things. EW sizes for worst-case additive error at every node; our model sizes for aggregate rank preservation. These are incommensurable — you can't directly compare sample counts because the guarantees differ. What you CAN say is that our model delivers ρ ≥ 0.985 empirically, while EW at the same p would guarantee additive ε ≈ 0.09 — which doesn't translate to any particular ρ without knowing the centrality distribution.

---

## Bibliography inventory: what's cited vs what's available

The references.bib file contains 47 entries. The paper cites 23 of them. The 24 uncited entries fall into categories:

**Foundational network science (not directly relevant):** Bavelas1950, Sabidussi1966, Dijkstra1959, Watts1998, Barabasi1999, Newman2010, Brandes2008

**Urban network context (background but not cited in argument):** Hillier1984, Hillier1993, Turner2007, Porta2006, Crucitti2006, Strano2013, Boeing2021, Cooper2015, Cooper2018, Sevtsuk2012, Hansen1959, Vale2017

**Centrality variants (not used):** Marchiori2000, Latora2001, Boldi2014

**Earlier versions / theses (superseded):** Riondato2014, Riondato2014thesis, Borassi2016

**Statistics (referenced indirectly):** Moran1950, Koenker1978

The key uncited entries that COULD strengthen the paper:

- **Bergamini2019** (survey comparing approximation approaches) — useful for positioning
- **Cooper2015/2018** (sDNA, localised centrality in practice) — validates the problem's importance
- **Brandes2008** (variants of betweenness) — could support discussion of betweenness variance

---

## Summary table: what each method contributes to our understanding

| Source                 | Key insight for localised source-sampling                                         |
| ---------------------- | --------------------------------------------------------------------------------- |
| Eppstein-Wang          | Formal bound adapts directly (n→r); establishes the baseline we improve on        |
| Brandes-Pich           | Source-sampling works empirically for both closeness and betweenness; no calibration provided |
| Bader et al.           | Adaptive 1/b(v) scaling = our "precision scales with importance"                  |
| Geisberger et al.      | Bias correction for source-sampling betweenness; unbiased estimator via distance weighting |
| Riondato-Kornaropoulos | VC-dimension tighter than EW for betweenness, but path-sampling only              |
| KADABRA                | Adaptive stopping + per-vertex budgets; principle applicable, machinery not       |
| SILVAN                 | Variance-dependent bounds much tighter on structured graphs; supports our finding |
| Cohen et al.           | Per-node structural guarantees (pivoting); closeness-specific, not localised      |
| Hájek (1971)           | Variance framework motivating √r; the theory behind the hypothesis                |
| Hoeffding (1963)       | Concentration inequality underlying EW; worst-case, no variance dependence        |
| Bernstein              | Variance-dependent concentration; unexploited for source-sampling                 |

---

## Formal analysis: Bernstein vs Hoeffding for source-sampling

**Script**: `10_bernstein_and_stopping.py`
**Generated**: 2026-02-09

### Setup

Both Hoeffding and Bernstein's inequality provide additive-ε guarantees for the mean of bounded random variables. In our setting, the random variables are per-source contributions to each target node's centrality. We union-bound over r target nodes (each within reach), giving:

- **Hoeffding**: k = log(2r/δ) / (2ε²)
- **Bernstein**: k = (2σ² log(2r/δ))/ε² + (2b log(2r/δ))/(3ε)

where σ² is the per-source contribution variance, b is the bound on a single contribution, ε is the additive error tolerance, and δ is the failure probability.

The key parameter is **σ²/b²**: when this ratio is small (most contributions are modest relative to the worst case), Bernstein requires far fewer samples.

### Results: Bernstein tightening factor

At ε = 0.1, δ = 0.1:

|  Reach | Hoeffding k | Bernstein k (σ²/b²=0.1) | Bernstein k (σ²/b²=0.03) | √r model k |
| -----: | ----------: | ----------------------: | -----------------------: | ---------: |
|    100 |         380 |                     203 |                       96 |        300 |
|  1,000 |         495 |                     264 |                      125 |        321 |
|  5,000 |         576 |                     307 |                      146 |        718 |
| 10,000 |         610 |                     326 |                      155 |      1,016 |
| 50,000 |         691 |                     368 |                      175 |      2,272 |

At σ²/b² = 0.1 (betweenness), Bernstein needs **53%** of Hoeffding's samples.
At σ²/b² = 0.03 (closeness), Bernstein needs **25%** of Hoeffding's samples.

### Key observation: O(log r) vs O(√r)

Both Hoeffding and Bernstein scale as **O(log r)** — the log(2r/δ) factor grows very slowly. The √r model scales as **O(√r)** — faster. This creates a crossover:

- **At low reach (r < 2,000)**: the formal bounds prescribe fewer samples than the √r model. The model's floor (n_min = 300) is _above_ what the formal bounds require.
- **At high reach (r > 5,000)**: the √r model prescribes _more_ samples than either formal bound. At r = 50,000, the model prescribes 2,272 while Bernstein needs only 368 (betweenness) or 175 (closeness).

This means the √r model is **more conservative than Bernstein** at large networks, not less. The model's conservatism comes from its faster growth rate, not from pessimistic constants.

### Empirical variance ratios from synthetic data

From the 3×7 synthetic network analysis (trellis, tree, linear topologies, 250m–4km):

| Metric      | σ²/b² median | σ²/b² range    | CV median | CV range       |
| ----------- | ------------ | -------------- | --------- | -------------- |
| Harmonic    | 0.016        | [0.002, 0.032] | 0.187     | [0.056, 0.272] |
| Betweenness | 0.015        | [0.007, 0.023] | 0.459     | [0.194, 1.510] |

The σ²/b² ratios are very low for both metrics (0.015 median), even lower than the theoretical estimates (0.03–0.1) used in the comparison table above. This is because the per-source contribution is much smaller than its worst-case bound for the vast majority of sources.

**Caveat**: these σ²/b² values are estimated from the cross-node variation of mean per-source contributions, which is Var_v[E_s[X_sv]], not the true E_v[Var_s[X_sv]]. The former measures inter-node heterogeneity; the latter measures per-source variability. On spatially autocorrelated networks these are correlated but not identical. True per-source variance likely exceeds the estimated values.

Despite the low σ²/b² for betweenness (suggesting Bernstein would prescribe very few samples), the CV for betweenness is 2.5× that of closeness (0.46 vs 0.19). This confirms the heavy-tailed contribution structure: most sources contribute zero to betweenness (pass no shortest paths through the target), while a few contribute large values. The CV captures this tail heaviness better than σ²/b².

### Why Bernstein doesn't close the gap

Bernstein tightens the formal bound by 2–4× depending on the metric. But the gap between formal bounds and what works empirically is much larger than 4×. The √r model at r = 10,000 prescribes k = 1,016 — and achieves ρ = 0.989 for betweenness, well above the target of 0.95. The true minimum k for ρ ≥ 0.95 is somewhere around 100–300 (based on the GLA first-crossing analysis). Bernstein at the same reach prescribes 326, which is closer but still formally targets additive ε, not rank preservation.

The fundamental issue: **no formal bound targets Spearman ρ directly**. Hoeffding and Bernstein guarantee additive ε at every node. The empirical model targets ρ ≥ 0.95 (rank preservation). These are incommensurable — rank preservation is structurally easier because it tolerates large errors on low-centrality nodes, which are precisely the nodes where source-sampling is least precise. The Bernstein bound is tighter than Hoeffding but still fundamentally overpays for rank preservation.

---

## Adaptive stopping: empirical rho trajectories

**Script**: `10_bernstein_and_stopping.py`
**Generated**: 2026-02-09

### Method

Using the GLA validation data (p = 0.01 to 0.9, at distances 1km–20km), track the Spearman ρ trajectory as a function of effective_n (= reach × p). Identify the first effective_n at which ρ ≥ 0.95, and compare to the model's prescribed effective_n.

### Results: first-crossing analysis

| Distance | Metric      |  Reach | First ρ≥0.95 eff_n | Model eff_n | Oversample |
| -------: | ----------- | -----: | -----------------: | ----------: | ---------: |
|     1 km | Closeness   |    100 |                 50 |         100 |       2.0× |
|     2 km | Closeness   |    411 |                123 |         300 |       2.4× |
|     5 km | Closeness   |  2,783 |                139 |         536 |       3.9× |
|    10 km | Closeness   | 11,590 |                116 |       1,094 |       9.4× |
|    20 km | Closeness   | 43,165 |                432 |       2,111 |       4.9× |
|     1 km | Betweenness |    100 |                 30 |         100 |       3.3× |
|     2 km | Betweenness |    411 |                 41 |         300 |       7.3× |
|     5 km | Betweenness |  2,783 |                 28 |         536 |      19.3× |
|    10 km | Betweenness | 11,590 |                116 |       1,094 |       9.4× |
|    20 km | Betweenness | 43,165 |                432 |       2,111 |       4.9× |

Model oversample ratios: closeness median 2.4×, betweenness median 4.9×.

### Interpretation

The model oversamples by 2–19× depending on distance and metric. The apparent oversample is largest at 5km betweenness (19.3×) because the lowest tested p (0.01) already achieves ρ = 0.99 — the true minimum eff_n is well below 28.

**Important caveat**: the first-crossing eff_n is the _observed_ minimum across the test grid (p = 0.01, 0.025, 0.05, ...). The true minimum lies between the first-crossing p and the next-lower p value. At moderate-to-high reach, the lowest tested p already exceeds the target, so the true minimum is unconstrained from below.

### Why adaptive stopping is harder than it looks

Adaptive stopping works by running sources one at a time, monitoring a convergence criterion, and stopping when the ranking stabilises. In principle, this avoids needing a model entirely. In practice:

1. **No ground truth at runtime.** The ρ trajectories above use the full (p=1.0) ground truth as reference. Without it, the convergence criterion must be self-referential: comparing the current estimate to a previous estimate (e.g., split-half correlation). This requires at least 2× the minimum eff_n before the split halves are individually reliable.

2. **Non-monotonicity risk.** ρ can dip below the target after crossing it, especially for betweenness at moderate eff_n. A stopping criterion that triggers at first crossing would sometimes stop too early.

3. **Overhead.** Each convergence check requires computing Spearman ρ between two partial estimates — an O(n log n) operation. If checked every k_step sources, the total overhead is O(n log n × total_sources / k_step). On large networks (n = 300k), this is non-negligible.

4. **Loss of parallelism.** The current implementation runs all sources in parallel (the sample_probability parameter in the Rust backend). Adaptive stopping requires sequential execution with periodic convergence checks, fundamentally changing the computation model.

5. **The model works.** The √r model is simple, pre-computable, and validated on two real networks. Its conservatism is a feature: it provides safety margin at minimal cost (the computation saved by sampling already gives 5–10× speedup; an extra 2× oversample reduces this to 3–7×, still a substantial gain).

### When adaptive stopping WOULD help

Adaptive stopping would be most valuable in scenarios where:

- Only closeness is needed (model oversamples by 2.4× median for closeness alone)
- The network is very large (r > 20,000) and computation is expensive
- The user needs a tighter accuracy guarantee than the model provides
- The user wants to specify a target ρ different from 0.95

For the default use case (both metrics, ρ ≥ 0.95), the √r model's conservatism is modest relative to the overall speedup.

---

## Synthesis: what formal bounds tell us about the empirical model

The √r model sits in a space that no formal bound occupies:

| Approach          | Targets           | Growth         | Formal?   | Practical?                            |
| ----------------- | ----------------- | -------------- | --------- | ------------------------------------- |
| Hoeffding (EW)    | Additive ε        | O(log r)       | Yes       | Too conservative at moderate reach    |
| Bernstein         | Additive ε        | O(log r)       | Yes       | Better, but still targets wrong thing |
| √r model          | Spearman ρ ≥ 0.95 | O(√r)          | No        | Yes — validated on real networks      |
| Adaptive stopping | Rank convergence  | Data-dependent | Partially | Complex implementation                |

The Bernstein analysis explains **why the √r model works**: per-source contribution variance on street networks is much lower than the worst-case bound (σ²/b² ≈ 0.015). This means the EW/Hoeffding bound massively overpays, and even Bernstein still overpays because it targets additive ε rather than rank preservation.

The √r model captures the right empirical relationship: on street networks, the effective sample size needed for rank preservation grows as √r, not as log r. This is faster than formal bounds require, but the additional conservatism provides safety margin and is justified by the lack of formal guarantees for rank preservation.

The O(log r) vs O(√r) crossover means the model is:

- **Less conservative than formal bounds at low reach** (r < 2,000): the floor n_min = 300 is the binding constraint, and it's above what Bernstein would prescribe.
- **More conservative than formal bounds at high reach** (r > 5,000): the √r growth outpaces O(log r), providing increasing safety margin.

---

## Hoeffding as a practical model: validation on GLA and Madrid

**Generated**: 2026-02-09

### The question

Rather than treating the EW/Hoeffding bound as a theoretical reference and fitting an empirical √r model against a Spearman ρ target, what if we use the Hoeffding bound _directly_ as the sampling model? The bound has formal guarantees (additive ε at every node with probability ≥ 1 − δ). If it also delivers sufficient rank preservation (ρ ≥ 0.95), then the empirical √r model becomes unnecessary — the formal bound is the model.

The formula: k = log(2r/δ) / (2ε²), then p = min(1, k/r).

The free parameters are ε and δ. At ε = 0.1, δ = 0.1:

### GLA validation

Hoeffding-prescribed probabilities were compared against the GLA validation sweep data (nearest tested p to Hoeffding's prescription):

| Distance |  Reach | Hoeffding p | √r model p | ρ_close (Hoeff) | ρ_betw (Hoeff) | ρ_close (√r) | ρ_betw (√r) |
| -------: | -----: | ----------: | ---------: | :-------------: | :------------: | :----------: | :---------: |
|      5km |  2,783 |       0.200 |      0.192 |     0.9970      |     0.9963     |    0.9968    |   0.9960    |
|     10km | 11,590 |       0.053 |      0.094 |     0.9963      |     0.9876     |    0.9964    |   0.9893    |
|     20km | 43,165 |       0.016 |      0.049 |     0.9914      |     0.9845     |    0.9986    |   0.9864    |

All configurations pass ρ ≥ 0.95. At 5km, Hoeffding and √r prescribe similar p (crossover region). At 10–20km, Hoeffding prescribes substantially less sampling. The tightest margin is 20km betweenness at ρ = 0.9845.

### Madrid validation (direct experiment)

Script `04_validate_madrid.py` was modified to test both √r and Hoeffding (ε = 0.1, δ = 0.1) prescribed probabilities at each distance. Unlike GLA (which used nearest-p from existing sweep data), Madrid runs the exact Hoeffding-prescribed probability.

| Distance |  Reach | Model        |     p | ρ_close | ρ_betw | Speedup |
| -------: | -----: | ------------ | ----: | :-----: | :----: | ------: |
|      1km |     77 | both (p=1.0) | 1.000 |  1.000  | 1.000  |    1.0× |
|      2km |    269 | both (p=1.0) | 1.000 |  1.000  | 1.000  |    1.0× |
|      5km |  1,451 | √r           | 0.267 | 0.9956  | 0.9927 |    3.5× |
|      5km |  1,451 | Hoeffding    | 0.354 | 0.9969  | 0.9949 |    2.7× |
|     10km |  5,219 | √r           | 0.141 | 0.9980  | 0.9892 |    6.9× |
|     10km |  5,219 | Hoeffding    | 0.111 | 0.9974  | 0.9872 |    8.6× |
|     20km | 15,300 | √r           | 0.082 | 0.9986  | 0.9855 |   12.5× |
|     20km | 15,300 | Hoeffding    | 0.041 | 0.9976  | 0.9794 |   25.3× |

**All Hoeffding-prescribed probabilities pass ρ ≥ 0.95**, including the critical 20km betweenness test (ρ = 0.9794).

### The crossover in practice

The O(log r) vs O(√r) crossover is clearly visible:

- **5km (r = 1,451)**: Hoeffding prescribes _more_ sampling (p = 0.354 vs 0.267). Hoeffding is the more conservative model here.
- **10km (r = 5,219)**: Hoeffding prescribes _less_ sampling (p = 0.111 vs 0.141). Crossover has occurred.
- **20km (r = 15,300)**: Hoeffding prescribes _half_ the sampling (p = 0.041 vs 0.082). The √r model is 2× more conservative. Hoeffding delivers 25× speedup vs √r's 12.5×.

### Epsilon calibration from synthetic data

To determine the tightest ε that still delivers ρ ≥ 0.95, the synthetic validation data (3 topologies × 7 distances × multiple probabilities) was analysed. For each (topology, distance) configuration, the Hoeffding-prescribed p was computed at various ε values, and ρ at that p was checked against the 0.95 target.

At ε = 0.125: 95% success rate (95/100 Hoeffding-approved configurations pass ρ ≥ 0.95). The 7 failures are ALL trellis harmonic at 1,500–2,000m — pathological configurations where closeness values are so tightly clustered on regular grids that ranking is inherently unstable even at p = 0.9.

Betweenness alone achieves 100% success up to ε = 0.130.

The standard choice ε = 0.1 is therefore well within the safe range.

### Implications for the paper's argument

This analysis supports a fundamental reframing: **use the Hoeffding bound as the model, with Spearman ρ as a reported consequence rather than a calibration target.**

The argument becomes:

1. Source-sampling of localised centrality produces bounded random variables.
2. The Eppstein–Wang adaptation of Hoeffding's inequality gives k = log(2r/δ) / (2ε²), yielding additive-ε guarantees at every node.
3. At ε = 0.1, δ = 0.1, this also delivers ρ ≥ 0.97 on both real networks tested.
4. The formal guarantee comes for free; rank preservation is a bonus.

Advantages over the √r model:

- **Formal guarantees**: additive ε at every node, not just empirical ρ
- **No empirical calibration of k or n_min**: ε and δ are standard parameters with interpretable meanings
- **Better speedup at high reach**: O(log r) scaling gives 25× at 20km vs √r's 12.5×
- **The floor emerges naturally**: at low reach, Hoeffding prescribes k > r (or close to it), so p → 1.0; no separate n_min needed

Remaining concern:

- **Betweenness margin at 20km**: ρ = 0.9794 under Hoeffding vs 0.9855 under √r. Still comfortably above 0.95, but the margin is thinner (0.029 vs 0.035). On a network with higher reach than Madrid, the Hoeffding model's O(log r) scaling means this margin would thin further. The √r model's O(√r) scaling provides increasing safety margin with reach — a feature, not a bug, for betweenness where variance grows with path complexity.
- **ε is still a free parameter**: choosing ε = 0.1 rather than 0.05 or 0.15 is itself an empirical decision. But it's a more interpretable one ("10% additive error tolerance") than the √r model's k = 10.16.

### Assessment

The Hoeffding model works as a practical sampling prescription on both tested networks. The key decision for the paper is whether to:

**(A) Hoeffding as the model.** Present the adapted EW bound as the sampling formula. Report ε = 0.1 as a recommended default. Show that rank preservation (ρ ≥ 0.97) follows as a consequence. The contribution is the adaptation to localised centrality + validation showing it works.

**(B) Hoeffding as departure point, √r as refinement.** Present the Hoeffding bound first, then show that the empirical √r model provides additional safety margin for betweenness at high reach. The contribution is both the formal bound and the empirical refinement.

**(C) √r as the model, Hoeffding as theoretical context.** (Current paper structure.) Present the √r model as the primary contribution, with the Hoeffding bound as the baseline it improves upon. Risk: the Hoeffding bound is simpler, formally grounded, and delivers sufficient accuracy — a reviewer may ask why the √r model is needed.

Option (A) is the cleanest. The √r model can be discussed as a more conservative alternative for users who want wider safety margins on betweenness, but the default recommendation would be Hoeffding with ε = 0.1.
