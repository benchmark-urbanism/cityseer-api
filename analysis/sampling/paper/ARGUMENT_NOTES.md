# Argument Development Notes

Working notes on the paper's central argument, developed through discussion and expert review.

## Argument Progression Outline

### The arc in one sentence

Localised centrality sampling works because accuracy depends on _reach_ not network size, and reach governs both estimation precision and computational cost — so sampling helps most where it works best.

### Step-by-step argument

**1. Problem: multi-scale centrality is expensive (§1 Introduction)**
Localised centrality at multiple distance thresholds is effectively quadratic. Existing sampling bounds target global centrality on general graphs; they do not calibrate per distance or exploit the localised structure.

**2. Background: what exists and what's missing (§2 Background)**
Source-sampling (Eppstein–Wang), path-sampling (Riondato–Kornaropoulos, KADABRA, SILVAN), and target-sampling (Cohen) each provide additive ε bounds. None addresses localised centrality, distance-dependent calibration, or rank-preservation targets.

**3. Methods: setup and accuracy metric (§3 Methods)**
Define harmonic closeness, betweenness, reach. Justify Spearman ρ as accuracy metric (rank preservation, not uniform additive error). Define normalised error for Hoeffding comparison. Describe synthetic topologies, boundary buffering, validation networks (GLA, Madrid).

**4. The opportunity (§4.1)**
Sampling works: accuracy saturates well below p = 1.0, and speedup grows with distance.
→ **Fig 1**: headline opportunity (accuracy vs p; speedup vs distance)

**5. Effective sample size and √r scaling (§4.2)**
Define eff_n = r × p. Motivate √r scaling from Hájek variance: if σ²_v grows linearly in r and CV is roughly constant, then eff_n ∝ √r keeps rank stability. Fit the general power model: α̂ ≈ 0.5, cannot reject α = 0.5. Fit k at 75th percentile → k = 10.16.
→ **Fig 2**: model derivation (eff_n vs ρ; required eff_n vs reach)
→ **Fig S1** (appendix): bootstrap diagnostics for α

**6. The floor: reliability at intermediate reach (§4.3)**
Proportional model alone fails at intermediate reach (100 < r < crossover). Logistic regression identifies n_min = 300 as the effective sample size below which success is unreliable.
→ **Fig 3**: floor justification (success rate by eff_n bin; proportional deficit)

**7. The complete model (§4.4)**
eff_n = max(k√r, n_min). Two regimes: floor-dominated (low reach) and proportional (high reach), crossover at r ≈ 872.
→ **Fig 4**: combined model (eff_n vs reach; p vs reach)
→ **Tab 1**: fitted model parameters

**8. Error structure across reachability — "precision scales with importance" (§4.5)**
Aggregate Spearman ρ is valid (full value range). Within-quartile ρ is _not_ valid (range restriction attenuation, Thorndike 1949). Instead, use error-based diagnostics: MAE and NRMSE by reachability quartile. The crossover: absolute MAE increases with reach (larger sums accumulate larger errors), but normalised MAE _decreases_ (high-reach nodes enjoy high eff_n, so relative error shrinks). High-importance nodes get highest precision.
→ **Fig 5**: error crossover (2×2: betweenness absolute/normalised, harmonic absolute/normalised)

**9. Model selection (§4.6)**
Five candidate models compared by AIC and leave-one-topology-out Brier score. √r + floor retained: AIC difference is modest, cross-validated prediction is equivalent, and √r connects to sampling theory.
→ **Tab model_comparison**: model selection table

**10. Relationship to theoretical bounds (§4.7)**
Adapt Eppstein–Wang to localised setting: replace n with r. Three reasons our model needs fewer samples: local subgraph (not full network), rank target (not uniform ε), planar street networks (bounded degree). At high reach, the two converge: implied ε ≈ 0.1. Hoeffding bound satisfied in 93%+ of synthetic configs; violations at low reach where sampling is moot anyway.
→ **Tab bounds_comparison**: method comparison table (in-text)

**11. Algorithm (§4.8)**
Two-phase: probe reach, then sample per distance. Scaling by 1/p.
→ **Algorithm 1**: pseudocode

**12. GLA validation (§5.1–5.2)**
Model-recommended p achieves ρ > target at 5/10/20 km. Observed ρ exceeds predictions (conservatism confirmed). EW bound holds.
→ **Fig 6**: GLA validation (accuracy vs p; model conservatism)
→ **Tab 2**: GLA validation results

**13. Madrid external validation (§5.3)**
Identical protocol, different city. ρ > target, EW bound holds.
→ **Fig 7**: Madrid validation (accuracy vs distance; speedup)

**14. Discussion (§6)**
Practical guidelines, assumptions and limitations (uniform sampling, homogeneous model, betweenness-conservative, shortest-path only), future directions.
→ **Fig 8**: practical guide (lookup charts)
→ **Tab 3**: practical lookup table

**15. Appendix**
Synthetic generation, GLA preprocessing, NetworkX verification, Hájek estimator details, cityseer implementation, power diagnostics, detailed EW bound evaluation.
→ **Fig S1**: power exponent bootstrap
→ **Fig EW**: predicted vs observed normalised error + sampling comparison
→ **Tab EW**: EW comparison table

### Figure–argument mapping

| Figure                  | Section | Argument role                                                                 |
| ----------------------- | ------- | ----------------------------------------------------------------------------- |
| Fig 1 (headline)        | §4.1    | Motivates the paper: sampling works, speedup grows with distance              |
| Fig 2 (derivation)      | §4.2    | Derives the √r relationship empirically                                       |
| Fig 3 (floor)           | §4.3    | Justifies the n_min floor for intermediate reach                              |
| Fig 4 (combined)        | §4.4    | Presents the complete two-regime model                                        |
| Fig 5 (error crossover) | §4.5    | Demonstrates "precision scales with importance" — the key qualitative insight |
| Fig 6 (GLA)             | §5.2    | Primary validation on real-world network                                      |
| Fig 7 (Madrid)          | §5.3    | External validation — generalisability                                        |
| Fig 8 (practical)       | §6.1    | Practitioner guidance                                                         |
| Fig S1 (diagnostics)    | App A   | Confirms α ≈ 0.5 via bootstrap                                                |
| Fig EW (synthetic)      | App A   | Detailed Hoeffding bound validation                                           |

### Table–argument mapping

| Table                 | Section | Argument role                                              |
| --------------------- | ------- | ---------------------------------------------------------- |
| Tab 1 (parameters)    | §4.4    | Reports fitted k, n_min, crossover reach                   |
| Tab model_comparison  | §4.6    | Justifies √r + floor over alternatives                     |
| Tab bounds_comparison | §4.7    | Positions our approach vs theoretical literature (in-text) |
| Tab 2 (validation)    | §5.2    | Quantitative GLA results                                   |
| Tab 3 (lookup)        | §6.1    | Practitioner reference                                     |
| Tab EW (comparison)   | App A   | Detailed bound satisfaction rates                          |

### What needs further clarification

**In the analysis/figures:**

1. **Top-k precision**: ~~Already computed in cache but never reported in any figure or table.~~ **Resolved.** Computed top-10% set overlap at model-recommended p. Results: betweenness median 0.822 (min 0.727), harmonic median 0.654 (min 0.189 on trellis). The low harmonic scores reflect value compression on regular grids — the top 10% are barely distinguishable from the next 10%, so tiny errors shuffle nodes across an arbitrary cutoff. This is _not_ a precision failure; the error crossover (Fig 5) already shows that normalised errors are negligible for closeness at all reach levels. Top-k overlap is an overly strict binary metric for the "precision scales with importance" claim; the error-based analysis is more informative. **Decision: mention briefly in discussion/limitations as a transparency note, not in main figures.** Keep diagnostic code in script 01.

2. **Betweenness variance gap**: **Resolved — paragraph added to §4.2.** Both metrics scale with reach (more terms in the sum), but per-source contributions differ in variance. Closeness: each source contributes 1/d(s,v), smooth and deterministic — every reachable source contributes something, values are bounded and well-behaved. Betweenness: each source contributes Σ_t σ_st(v)/σ_st, sparse and heavy-tailed — most sources route zero paths through v, a few contribute heavily (e.g. sources on the other side of a bridge). Higher per-source variance means more samples needed for the same CV, explaining the ~1.5× factor. Bridge nodes on street networks are the canonical example: sources on one side contribute all cross-river paths (large contribution), while sources on the same side contribute little or nothing. These are also the nodes where betweenness accuracy matters most. The model handles this implicitly by fitting to betweenness (the harder metric), making it conservative for closeness. ~~Action: add a paragraph in §4.2 after the Hájek motivation explaining this.~~ **Done** (2026-02-08).

3. **Fig 5 normalisation and Panel D flatness**: **Resolved — caption updated.** The normalisation (reach for closeness, r(r-1) for betweenness) matches the EW bound framework in §4.7. It is conservative for betweenness (actual betweenness ≪ r(r-1) on street networks), which means the NRMSE decrease in Panel B is partly an artifact of the denominator growing faster than actual centrality. Per-quartile normalisation by median centrality was considered but rejected — it breaks cross-quartile comparability, which is the figure's purpose. The "precision scales with importance" claim rests on the _mechanism_ (high-reach nodes have higher eff_n, therefore lower estimation variance) rather than on the normalised figure being quantitatively precise. Panel D's near-constant normalised MAE (~2–4 × 10⁻⁵) follows because closeness contributions scale as 1/d, so centrality magnitude grows proportionally with reach; the normaliser tracks the same scale. ~~Action: keep current normalisation; note in §4.5 text that it matches the EW bound framework; add a clause to the Fig 5 caption explaining Panel D flatness.~~ **Done** (2026-02-08): added normaliser explanation and EW bound cross-reference to Fig 5 caption.

4. **Thorndike 1949 citation**: **Resolved.** Was missing from references.bib. Added under "Psychometrics / Range Restriction" section.

5. **Re-run downstream scripts**: Scripts 03–05 need re-running with updated model parameters (k=10.16, n_min=300) and renamed figure outputs. GLA/Madrid ground truth caches may need regeneration with node_reach included.

6. **Paper macros drift**: The paper sources pull model parameters via `paper/tables/model_macros.tex`, but Table 1 currently hard-codes newer values (k=10.16, n*min=300, crossover=872). Until `07_generate_macros.py` is re-run, the \_text + algorithm + practical guidelines* will continue to use the older macro values (e.g. k=9.91, n_min=350, crossover=1247). **Action:** re-run `07_generate_macros.py` after scripts 03/04 complete, and consider rewriting Table 1 to use macros (
   `\kProp`, `\minEffN`, `\crossoverReach`) to prevent silent divergence.

**In the theoretical framing:**

1. **"Hypothesis" language for √r**: **Resolved.** Re-reading §4.2, the framing is already honest: "variance-scaling heuristic rather than a universal theorem" (line 25), "theoretically motivated hypothesis" (line 32). The one sentence that was slightly strong — "confirming the theoretical prediction" — has been changed to "consistent with the theoretical prediction". ~~Action: change "confirming" to "consistent with" in §4.2 line 34.~~ **Done** (2026-02-08).

2. **Mean reach as network-level summary**: The most vulnerable point (expert review §Gaps #3). Our quartile analysis (Fig 5) addresses this for synthetic networks — error structure is favourable by reachability quartile. But GLA/Madrid caches don't yet have node_reach, so we can't show this on validation networks. **Tied to item 5**: re-running scripts 03–04 with `--force` will store node_reach, enabling per-quartile error analysis on real networks. That would be the strongest response. For now, synthetic quartile analysis + aggregate ρ exceeding target on real networks is the best available evidence.

3. **Aggregate Spearman vs per-quartile error**: **Resolved — already handled.** §4.5 opens with "Although the model is calibrated on aggregate Spearman ρ (which is valid over the full value range)" and explains the Thorndike argument. The bridge is implicit: §4.2 fits on aggregate ρ, §4.5 (immediately after the complete model) explains why that's the right choice. No forward reference needed.

4. **Cohen 2014 engagement**: **Resolved — already handled.** §2 line 29 explicitly explains: "standard centrality algorithms, including localised implementations with distance cutoffs, are inherently source-based... Source-sampling bounds therefore analyse the same computational primitive that localised centrality uses, whereas path-sampling bounds analyse a different primitive." Cohen's target-sampling is classified as a third variant. The reasoning for preferring source-sampling is already clear.

---

## Core Argument Paragraphs (latest version)

### Paragraph 1: The mechanism and its structural property

When centrality is estimated from a subset of sampled source nodes, each target node's estimate is built only from those sampled sources that can reach it within the distance threshold. The precision of a target's estimate therefore depends on how many sampled sources contribute to it, which is proportional to the target's reachability. At low reachability, few sampled sources contribute and estimates are noisy; but once reachability exceeds a modest threshold, the estimate stabilises and further increases in reach yield diminishing returns. The critical question is where this threshold falls relative to the computational cost of exact computation. On street networks, the two are structurally linked: low reachability implies small local subgraphs and cheap exact computation, while high reachability implies large subgraphs and expensive computation. Sampling is therefore needed precisely in the regime where it works well, and dispensable where it does not. This structural alignment also explains why rank correlation (Spearman rho) is a more natural accuracy metric than uniform additive error for the localised setting: high-reachability targets, which receive the most precise estimates, are also the high-centrality nodes whose ranking matters most, while low-reachability targets with noisier estimates are peripheral nodes whose exact ranking is less consequential.

### Paragraph 2: Why empirical validation is load-bearing

This target-centric view reveals a limitation of any network-level sampling calibration: targets with low reachability receive contributions from fewer sampled sources and therefore have higher-variance estimates, yet the adaptive model uses mean reach as a single summary per distance threshold. The sqrt(r) scaling is therefore a hypothesis rather than a formal guarantee; it assumes that the reachability distribution within each distance threshold is regular enough for a mean-based calibration to preserve rank stability across nodes. The model is deliberately conservative — calibrated at the 75th percentile of required effective sample size rather than the median, and incorporating a minimum floor below which near-exact computation is prescribed — precisely because the theory cannot guarantee uniform accuracy across all nodes. This is why empirical validation on synthetic networks spanning distinct topologies with different reachability distributions, and on real street networks where boundary effects and spatial density variation introduce further heterogeneity, is not optional but essential to the argument.

---

## Why sampling changes the computational flow

In the full (unsampled) computation, every node runs its own Dijkstra and aggregates to itself — each node has complete self-knowledge of its own centrality. Sampling breaks this symmetry: most nodes no longer run their own Dijkstra, so they can only receive centrality estimates by being _reached_ by the sampled sources that did run. This is not a source-vs-target distinction (on undirected graphs, reachability is symmetric), but a structural consequence of sampling: it shifts from a regime where every node has complete information about itself to one where nodes depend on being visited by others' Dijkstras. Reachability becomes the governing variable because it determines how many sampled sources contribute to each node's estimate. In the full computation, this variable is irrelevant — every node gets contributions from all nodes that can reach it, by definition.

---

## Key Logical Chain

1. Localised centrality has no global guarantees — each target's estimate depends on how many sampled sources happen to reach it.
2. The accuracy problem is fundamentally heterogeneous — different nodes at the same distance threshold have different effective sample sizes.
3. The question becomes: at what point does the reachability of _most_ nodes become sufficient for reliable estimates?
4. This is why we use the 75th percentile (not median) to set k — asking "at what sampling rate do most nodes have enough contributing sources?"
5. Two failure boundaries: (a) per-node breakdown (drives the n_min floor), (b) per-distance breakdown where reach is too low for sampling to help (drives p = 1.0).
6. The model is explicitly conservative, designed for the majority of nodes across the majority of topologies.
7. Empirical validation isn't decoration — it's how we verify the conservative calibration holds on real heterogeneous networks.

## Why Spearman (not additive error)

- Additive error treats all nodes equally, paying the cost of guaranteeing accuracy for the hardest-to-estimate nodes (low reachability / low centrality), even though those are the nodes whose exact values matter least.
- Spearman is naturally tolerant of large absolute errors on low-centrality nodes (rank swaps among near-zero values contribute little) and sensitive to errors on high-centrality nodes (where rank swaps matter).
- Spearman is _structurally matched_ to the heterogeneous accuracy regime that localised sampling creates.
- This isn't just "we care about ranks in urban analysis" — it's that Spearman aligns with the accuracy structure of the method itself.

---

## Expert Review Feedback

### Strengths identified

- Target-centric framing is logically valid (especially for closeness)
- Structural alignment (sampling works where needed) is not circular — derives from two independent facts
- Eppstein-Wang adaptation is well-executed
- Dual-track validation (rank + error bound) is effective
- Epistemic humility in calling sqrt(r) a hypothesis is appreciated

### Gaps and vulnerabilities

#### 1. Betweenness is the weak link

The Hajek variance argument works cleanly for closeness (simple sum over source contributions) but betweenness has a fundamentally different covariance structure — per-source contributions depend on path structure, not just reachability. The paper treats both as "summation-based" without confronting why betweenness needs ~1.5x more samples. The "weakly dependent, comparable scale" assumption is less defensible for betweenness.

#### 2. sqrt(r) is a chain of assumptions, not a derivation

Three "assume" steps compound: (a) sigma^2 grows linearly in r, (b) CV is roughly constant across distances, (c) Hajek applies cleanly. Each is plausible but none guaranteed. The honest framing is: "sqrt(r) is an empirical regularity consistent with variance-scaling theory." The paper comes close but still presents the Hajek argument as providing more predictive power than it does.

#### 3. Mean reach as network-level summary is the most vulnerable point

A reviewer could ask: "What is Spearman rho computed separately for the bottom quartile of nodes by reachability?" If it's substantially below 0.95, the aggregate metric masks spatial heterogeneity. The conservatism helps but doesn't formally address this.

#### 4. Spearman is not a top-k metric

High rho could hide misranking at the top if compensated by good preservation elsewhere. A top-k analysis (overlap at k, Kendall tau for top 10%) would strengthen the "precision scales with importance" claim. For betweenness specifically, bridge nodes in sparse regions can have high betweenness but modest reachability — these are where rank accuracy matters most but sampling accuracy is poorest.

#### 5. Cohen 2014 is underutilised

Cohen's target-sampling approach could provide per-node accuracy guarantees rather than aggregate rho. The paper should discuss why source-sampling is preferred beyond "standard algorithms are source-based." Note: Cohen's method is actually a hybrid pivot-based interpolation for global closeness, not a direct competitor, but the related work discussion should be more substantive.

#### 6. Betweenness normalisation makes the 93% bound very loose

Dividing by r(r-1) assumes worst-case (star graph), vastly overestimating max betweenness on street networks. The bound being satisfied in 93% of cases is less informative than it sounds.

### Recommended improvements

- **Spatial decomposition of accuracy**: report rho by reachability quartile, not just aggregate
- **Top-k rank analysis**: verify "precision scales with importance" directly
- **More candid about sqrt(r)**: present as empirical finding with theoretical motivation, not derivation
- **Engage with Cohen 2014**: explain why source-sampling is preferred for localised analysis
- **Acknowledge betweenness variance gap**: Hajek argument is cleaner for closeness than betweenness
- **Small synthetic training set**: only three topologies — leave-one-out CV is extremely noisy with N=3
- **No angular distance validation**: noted as future work but limits the "urban network analysis" claim

---

## Integrity notes (implementation vs narrative)

- **"NRMSE" naming**: the current analysis pipeline uses _normalised MAE_ (MAE divided by a metric-appropriate normaliser), not RMSE. The code also computes **median** absolute error within reach quartiles (robust to outliers) even if the paper uses the phrase "mean absolute error". **Action:** standardise wording to "(median) absolute error" and "normalised MAE" (avoid the NRMSE acronym unless RMSE is actually computed).

### Expert bottom line

> "The paper makes a genuine contribution. The target-centric insight that accuracy depends on reachability rather than network size is valid and practically important. The adaptive algorithm is simple, implementable, and empirically effective."

Main risks: reviewer demanding per-node accuracy analysis, theorist finding Hajek motivation insufficient, limited topology diversity in training data.

---

## Script Review Findings

### What the code does correctly

- Model fitted on betweenness only (conservative: betweenness is harder)
- Top-k precision (top 10% overlap) already computed in `compute_accuracy_metrics()` and cached
- Sampling implementation correct: source-side random inclusion, global Hajek scaling, contributions aggregated to targets

### What was missing (now addressed)

- **Per-node reachability not saved**: `node_reach` array was discarded, only `mean_reach` scalar kept. **Fixed**: GLA ground truth cache now stores `node_reach` array.
- **No per-reachability-quartile analysis**: accuracy was always aggregate Spearman rho. **Fixed**: `compute_quartile_accuracy()` added to utilities.py, called in GLA validation loop. Results saved as `spearman_q1`-`q4` and `reach_q1`-`q4` columns.
- **Top-k precision computed but never reported**: already in cached results. Should now be reported in paper.

### Things to note in the paper

- Zero-valued nodes excluded from Spearman rho (`mask = true_vals > 0`) — defensible but should be stated
- Hajek correction is a global scalar (N_live/n_sampled), not per-node — consistent with the model's mean-reach approach
- GLA has N_RUNS=1 (probability sweep, ground truth is deterministic reference); Madrid has N_RUNS=5 (model-prescribed test)
- Model fitted at 75th percentile of implied k — pragmatic conservative choice, not statistically derived

### Script changes made

- `utilities.py`: added `compute_quartile_accuracy()` function
- `00_generate_cache.py`: now computes per-quartile Spearman rho for both metrics, stores `spearman_q1`-`q4` and `reach_q1`-`q4` in cached results
- `03_validate_gla.py`: now saves `node_reach` in ground truth cache, computes and saves per-quartile Spearman rho for both metrics
- `04_validate_madrid.py`: now saves `node_reach` in ground truth cache, computes and saves per-quartile Spearman rho (prefixed `h_`/`b_` for closeness/betweenness)

### Note on existing ground truth caches

Existing GLA and Madrid ground truth caches don't have `node_reach`. The code handles this gracefully (`gt_data.get("node_reach", None)`). Re-running with `--force` will regenerate them with the full array, but the ground truth computation is expensive. Alternatively, reach can be recomputed from a separate pass without recomputing centrality.

---

## Open Questions

- [x] Should we run a spatial decomposition analysis (rho by reachability quartile)? **YES — implemented**
- [x] Should we add a top-k rank metric? **Already computed; decision: diagnostic only, not in main figures**
- [x] How to handle the betweenness variance gap in framing? **Paragraph added to §4.2** (2026-02-08)
- [ ] Where exactly do these paragraphs go in the introduction?
- [ ] How much of this feeds into the discussion/limitations section?
- [ ] Does the abstract need further revision to match?
- [ ] Re-generate GLA ground truth caches with node_reach included (or add a reach-only recomputation step)
- [ ] Re-run script 07 (macros) after scripts 03/04 complete — paper currently renders with stale values (k=9.91, n_min=350, crossover=1247)
- [ ] Re-run scripts 08 (power exponent) and 09 (model comparison) — their output JSONs are missing
