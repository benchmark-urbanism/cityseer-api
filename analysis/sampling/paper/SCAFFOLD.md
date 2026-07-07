# Paper Scaffold

This is the logical skeleton of the paper. `main.tex` must trace to it section by
section; anything in the prose that has no home here is a candidate for deletion, and
any change of argument is made here first, then in the prose. Keep the two in sync in
the same commit.

**The single claim:** in localised analysis, precision must follow the catchment.
Every section sets up that claim, delivers it, or defends it.

**The intuition:** the method is a survey. Each street has its own population (its
catchment), so each needs its own sample size. The population sizes are unknown, so a
cheap poll measures them first; the survey proper is then sized street by street. The
work test is "when a population is small, a census is cheaper than a survey"; the
confidence bound is "when the poll is unsure how small a population is, survey it in
full"; inverse-probability weighting is standard survey practice.

**The argument at a glance** (each line is the claim its section must land):

1. Street importance is measured within catchments, and exact computation does not scale.
2. Sampling makes it fast, but one fixed rate makes results silently unequal.
3. Poll the network first to measure every catchment, then give every street a
   sampling rate sized to its catchment.
   ![method schematic](figures/fig1_method_schematic.svg)
4. Ablation ladder: reach can be assumed from a formula, counted on a map, or
   measured on the network. Each rung fails where the one above catches it: the
   held-out network fails under assumed reach exactly where the reach gap predicts
   (fig6), and map counts are blind to the circuity that separates disc from reach (example figure).
5. Measure reach, and all four networks pass, the held-out one included. Against the
   fixed rate the gain is control: the held-out failure disappears, and speedups
   remain where sampling pays (6.8--8.3x on London at 20 km; Madrid near break-even
   at 1.1--1.5x; suburban betweenness from break-even on Cary, 1.0x, to 2.6x on the
   held-out network). The fixed rate records larger speedups on the shared cells
   because it prices in no pilot and no confidence bounds; its unmanaged rate fails
   on the held-out suburb, and its per-cell accuracy cannot be known in advance.
   ![baseline vs method](figures/fig12_baseline_vs_adaptive.svg)
6. What the user gets: one parameter, unbiased estimates, honest speedups; where
   sampling cannot pay, the method declines to sample.
7. Scope: rank-based evaluation, betweenness caveat, pilot tail, angular untested;
   the main validation ends at 20 km with the buffered network extent (by
   construction). The extended-distance test reruns the held-out network on a 50 km
   buffer (nine counties, Houston-scale) across the FULL 1-50 km range
   (11_frontier_woodlands.py --distances 1000..50000; output/woodlands_frontier.csv),
   so fig13 panel C is one continuous series from one build, no divider or diamond
   special case. Closeness routes exact throughout on that build; betweenness plateaus
   just above the target line rather than crossing it (speedups 6-14x in the timing
   session of record; frontier sampled timings swing between sessions, per the Setup
   caveat).

**Reader:** urban analysts deciding whether to trust sampled centrality, plus a
methods referee. Their questions, in order: can I trust it, when does it apply, what
do I set, what does it cost.

**Voice rule:** one method. The canonical schedule is an ablation, confined to its own
subsection. No "two ways to supply reach" framing anywhere else.

**Intuition rule:** one analogy, the survey, paper-wide. Every technical component is
introduced in survey terms before its mathematics. No second metaphor.

**Register rule (2026-07-06):** formal academic tone throughout, explanatory but never
chatty. No staccato gloss fragments, no conversational vocabulary ("crowd", "doing its
job", "in plain terms"), no double introductions of one concept. Accessibility is
achieved with precise subordinate clauses, concrete nouns, worked numbers, tables, and
figures, inside the formal register.

**Framing rule (2026-07-06):** do not imply analysts already sample; few do. The arc,
stated in the abstract and mirrored by the introduction: localised centrality is the
goal; exact computation is expensive; sampling is therefore appealing; sampling raises
a design problem (comparable uncertainty across unequal catchments); this design
answers it; these are the results.

---

## 1. Introduction

- Localised centrality at metropolitan scale is expensive; analysts compare across
  places and cities.
- Comparison requires uniform, known uncertainty. A fixed sampling rate silently gives
  dense places good estimates and sparse places bad ones.
- Insight: under localisation the relevant population is the catchment; catchments
  vary; therefore the sampling rate must be per node, derived from measured reach.
- Contributions: (i) the localised sample-size bound (population = reach, not network
  size); (ii) the per-node design (pilot, q, Horvitz-Thompson, work test, one
  tolerance parameter); (iii) validation on four networks, one held out from
  accuracy calibration (the work-test margin was set from timings across all runs
  and affects engagement, not estimates), with an ablation showing assumed reach
  fails; (iv) open-source implementation.

## 2. Preliminaries

- Harmonic closeness, with the justification (disconnected, variably sized catchments;
  additive form suits per-source estimation).
- Betweenness semantics (all routes; live filters reporting).
- Reach; buffers and live nodes.
- Rank fidelity (Spearman rho >= 0.95) as the evaluation target, and why.

## 3. Method (stated once, completely)

- 3.1 How many to ask: k(r) = ln(2r/delta) / (2 eps^2); the failure budget is divided
  across the catchment, not the network. The bound is stated for an idealised
  fixed-size survey of the catchment; under locally homogeneous reach the allocation
  keeps the expected number of failing nodes per catchment at or below delta. The
  deployed design (independent per-node inclusions, heterogeneous q) targets the same
  expected reached-source count and is judged empirically. Single tolerance eps.
- 3.2 How big is each population: the pilot poll. m sampled sources (2.5% of nodes,
  floored at 400; counted in parallel in the Rust layer), one bounded Dijkstra each;
  a node's hit count measures its reach (hypergeometric under without-replacement
  draws; binomial bounds remain conservative), so one traversal set measures every
  catchment at every distance, barriers included. Error costs are asymmetric: overestimated reach undersamples (an accuracy
  risk), underestimated reach only oversamples (a time cost), so q_u = min(1,
  k(r_u)/r_u) derives from a lower confidence bound on reach. Saturation at q = 1 is a
  census of sparse catchments; per-source 1/q weighting is unbiased regardless of
  pilot quality; probability floor bounds the weights.
- 3.3 When a census is cheaper: closeness live-exact vs sampled-IPW source pools; the
  per-distance work test, priced from the upper confidence bound on reach (a node the
  poll never hit is censused, not free); sampling engages only below an overhead
  margin (0.75) of exact work, the one empirically chosen constant besides epsilon;
  where sampling engages (dense networks, long distances) and where it declines (low
  live-fraction closeness). Buffer handling. Defines "mode" (per-distance exact vs
  sampled; the test runs once per call, priced by whether betweenness is requested,
  so per-metric modes come from single-metric calls).
- 3.4 Algorithm box + worked-example schematic (fig1).
- 3.5 Guaranteed vs validated: additive-in-expectation is guaranteed; rank fidelity is
  validated; betweenness has no per-source bound and is judged empirically.

Figures owned: fig1 (schematic). Tables owned: none.

## 4. Ablation: why measurement is necessary

The ONLY home of the canonical schedule. Function: prove the load-bearing component.
Structured as a ladder: three ways to know a population's size, each failing where the
next one catches it.

- Rung 1, assume it: the canonical reach (pi d^2 / s^2, s = 175 m). Reach-vs-assumed
  figure with the gap annotated (fig6): suburbs sit below the curve. Result: the
  held-out sparse suburb fails 20 km closeness (rho = 0.94) with a mechanistic
  explanation (reach at 38% of assumed vs 51% for Cary); dense networks are
  unaffected. A fixed rate cannot know this; measurement does.
- Rung 2, count it on a map: the Euclidean neighbour count, deflated by 2.5. The
  deflation sits between the measured per-node disc-to-reach medians (1.3--1.8) and
  99th percentiles (up to 3.4); measured ratios recorded in
  output/euclidean_reach_ratios.csv (measure_disc_reach_ratio.py). Correct on
  average, blind to circuity and barriers alike: a node counts disc neighbours the
  network cannot deliver within the limit. Example figure: one high-ratio node, the
  Euclidean disc beside the actually-reached set, the exclusion driven by ordinary
  circuity concentrating disc population near the rim.
- Rung 3, measure it: the pilot poll (the method, Section 3.2). Nothing to ablate;
  the rung the others are judged against.
- Epsilon history lives here if needed (the Cary sweep), briefly.

Figures owned: fig6, the disc-vs-reach example figure, plus the baseline curves in the
comparison figure (fig12). Tables owned: one condensed ablation table (or appendix
record of per-network canonical results).

## 5. Validation (canonical schedule vs method throughout)

Every claim in this section is a comparison: the canonical schedule beside the
method, on the same networks, seeds, and ground truth. Three comparisons, in order.

- Setup: four networks spanning the density range (London, Madrid, Cary, The
  Woodlands), the last held out from calibrating the accuracy parameters; three
  seeds; exact ground truth. Each metric is validated through its single-metric
  entry point, so modes are recorded per metric.
- Comparison 1, accuracy and its geography: rho vs distance for both metrics
  (fig13: calibration networks in panels A and B; the held-out network in panel C,
  validated 1-50 km on the 50 km build); per-reach-quartile rho (fig12
  panel C: the upper three quartiles improve slightly, the lowest shifts by a
  similar small amount the other way);
  paired rank-shift statistics on the held-out network at 20 km (baseline closeness:
  neighbour-error correlation 0.55 and mean rank displacement 0.068 of the rank
  range; the method routes the cell to exact and the residuals are zero; betweenness
  mean rank displacement is similar under both designs, 0.048 vs 0.051, while the
  neighbour-error correlation falls from 0.25 to 0.18). Section 5.3 is a single
  summary paragraph; all four error-structure figures (method-only rank-shift maps
  fig7, error-vs-reach fig8, decile matrices fig11, paired closeness rank-shift map
  fig15) live in the appendix and the paragraph points at them. Betweenness spatial
  claims stay in the text because their improvement is de-clustering, not magnitude,
  which a median hexbin cannot show. Residual betweenness gradient acknowledged
  (tie-heavy low-reach values).
- Comparison 2, cost at the same target: consolidated table of rho and realised
  speedups, pilot included; sampling pays on London at 20 km (6.8x/8.3x) and on
  held-out betweenness at 20 km (2.6x); Madrid's 20 km cells sit near break-even
  (1.1x/1.5x), Cary betweenness breaks even (1.0x), and every other sampled cell
  at 5--10 km except London's 10 km betweenness (1.2x) runs below 1x, where the
  pilot overhead cancels the small gain. The fixed schedule records larger
  speedups on the shared cells (8.2x/11.2x on London at 20 km; 9.7x/16.2x for
  suburban betweenness) because it prices in no pilot and no confidence bounds;
  its unmanaged rate is what fails on the held-out suburb (Section 4), and its
  per-cell accuracy is unknown in advance.
- Comparison 3, behaviour: per-metric mode matrix (exact vs sampled by network and
  distance); the method declines to sample short distances and low-live-fraction
  closeness, and the held-out network passes everywhere. Error structure also
  reports within-quartile statistics for the sampled metro closeness cells (rho
  depressed by range restriction; per-quartile median rank shift stays small).

Figures owned: adaptive accuracy (fig13) and comparison (fig12). Appendix: all four
error-structure figures (fig7, fig8, fig11, fig15), plus the paired rank-shift
statistics (spatial_macros.tex). Tables owned: the consolidated adaptive results
table (with per-metric modes).

## 6. Discussion

- Practical guidance: one knob (eps); expected speedups by density class and live
  fraction; the work test selects exact computation where sampling would cost more.
- Limitations: rank-based evaluation has no formal bridge from the additive bound;
  betweenness bound looseness, with the measured mechanism: rank error concentrates
  among high-traffic nodes (route-pilot diagnostic, Woodlands 20 km: nodes crossed by
  100+ of 400 pilot trees err median 1.7 / p90 9.6 percentile points while the
  low-route periphery holds stable), so the noise is the ordering of important
  streets, driven by contribution concentration (a few origins dominate an
  arterial's value); a concentration-aware pilot is the identified future work; pilot tail (a node whose reach the poll overestimates
  is under-sampled with probability bounded by alpha); suburban sampled closeness
  sits outside the main validation, but a forced-sampling check at 20 km meets the
  target on both suburbs (forced_closeness_check.py; output/forced_closeness.csv);
  residual quartile gradient; angular (simplest-path) untested.
- Related work: compact; KADABRA et al. adapt to global targets, not per-catchment
  precision in the distance-bounded setting.

## 7. Conclusion

Claim, mechanism, evidence, scope. Nothing new.

---

## Exclusion list (things the paper must NOT contain)

- The canonical schedule outside Section 4.
- The fixed p(d) practical-guide figure and distance lookup table (guidance for a
  schedule the software no longer runs).
- "Two ways to supply reach" framing.
- Per-network canonical results as headline tables (appendix at most).
- Claims that quartile precision is fully uniform for betweenness (the upper
  quartiles improve; the lowest-reach quartile is limited by tie-heavy values and
  the gradient does not vanish).

## Figure/table inventory (target state)

| Item                                | Section  | Content                                  | Status                                     |
| ----------------------------------- | -------- | ---------------------------------------- | ------------------------------------------ |
| fig1 schematic                      | 3        | worked example, 2x2                      | done                                       |
| fig6 reach gap                      | 4        | assumed vs measured reach, gap annotated | done (ablation caption)                    |
| disc-vs-reach fig (fig14)           | 4        | one node: Euclidean disc vs reached set  | done (Cary 10 km; barrier_macros.tex)      |
| ablation table (tab8)               | 4        | canonical per-network minima, condensed  | done (minima sentence now data-driven)     |
| adaptive accuracy fig (fig13)       | 5        | rho vs distance, method                  | done (regenerated 2026-07-04, final CSVs)  |
| adaptive results table (tab7)       | 5        | rho + speedups, all networks/distances   | done (regenerated 2026-07-04, final CSVs)  |
| fig12 comparison                    | 4/5      | baseline vs method bars + quartiles      | done (regenerated 2026-07-04, final CSVs)  |
| rank-shift, deciles, error-vs-reach | appendix | fig7, fig8, fig11; stats in 5.3 text     | done (all moved to appendix 2026-07-06)    |
| fig15 paired rank-shift map         | appendix | Woodlands 20 km closeness, paired        | done (moved to appendix 2026-07-06)        |
| canonical tables 2/4/5/6            | appendix | ablation record                          | done                                       |
| fig3 + distance lookup              | none     | canonical guidance                       | removed (generators deleted from 06)       |
| fig2/fig4/fig5 canonical figures    | none     | canonical err/accuracy/speedup           | removed (generators deleted from 05)       |

**Status note (2026-07-04, cycle 6):** the polled-pilot validation re-run is complete;
the final GLA CSV landed at 17:31. Scripts 06, 07, 08, and 09 were re-run at ~17:45
against the final CSVs, so model_macros.tex, tab7, tab8, fig1, fig12, fig13, the
spatial figures (fig7/fig8/fig11/fig15), and spatial_macros.tex all trace to the
current data era. 06 now also emits min/max speedup macros over the sampled 20 km
cells (1.0--8.3x) and asserts n_seeds == 3 for every sampled cell. Speedups divide
exact baselines cached at ground-truth build time (Jun 16 / Jul 3) by sampled runtimes
from the 16:30--17:31 re-run; Setup discloses the machine and configuration. A
same-session timing re-measurement on an idle machine remains open.

**Status note (2026-07-06, shortening + register pass):** the paper was cut for length
and the main text lightened. All four error-structure figures now sit in the appendix
and Section 5.3 is one summary paragraph. Per-cell numbers (speedups, quartile rhos,
canonical comparisons, paired rank-shift statistics) live in tab7/tab8, the appendix
tables, and the appendix error-structure paragraph; the text repeats only load-bearing
values (held-out failure rho, minimum sampled rho, 20 km speedup range). The survey
analogy is stated in the Section 3 opener and no longer repeated; fig1 and appendix
captions were halved; Related work is one paragraph leaning on tab:bounds_comparison.
Register rule now in force: main-text sentences stay short and plain; technical
machinery lives in the appendix. New appendix paragraph "Pilot reach estimator and
confidence bounds" (sec:appendix_pilot) holds the hit-count estimator, Clopper-Pearson
construction, and without-replacement note; the C_max per-term bounds moved into the
IPW appendix paragraph. Body source tokens 9097 -> 6721 (-26%); main text ends p14
(references from p15); 25 pp total.
