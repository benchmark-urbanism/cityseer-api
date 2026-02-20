# Paper Revision Plan: Hoeffding-First Reframing

**Created**: 2026-02-09

**Note (2026-02-09):** This document is a historical planning note. The repository now implements the Hoeffding-first framing with an 00--07 analysis pipeline (see `CONTENT_MANIFEST.md`), and the library implementation uses Horvitz--Thompson-style inverse-probability weighting (i.e., $1/p$ scaling) rather than $N/n$ post-scaling.

## Overview

The current paper presents an empirical √r model as the primary contribution and treats the Eppstein–Wang (EW) bound as a secondary cross-check. The revision inverts this: the adapted EW/Hoeffding bound **is** the model; ε = 0.1 is calibrated empirically; rank preservation (ρ ≥ 0.97) is reported as a consequence.

Zero fitted parameters. Three conventional choices: Hoeffding's inequality (textbook), ε = 0.1 (standard in the sampling literature), δ = 0.1 (standard).

---

## Phase 1: New/Modified Analysis Scripts

These produce the figures and tables the revised paper needs.

### 1a. Hoeffding model figure

**Purpose**: Generate the Hoeffding model derivation figure.

- **Panel A**: Hoeffding-prescribed k = log(2r/δ)/(2ε²) vs reach r, for ε = 0.1, δ = 0.1. Show the resulting p = min(1, k/r). Overlay the empirical √r model for comparison.
- **Panel B**: Effective sample size (eff_n = k when p < 1) vs reach. Show O(log r) growth of Hoeffding vs O(√r) growth of empirical model. Annotate the crossover at r ≈ 2,000–5,000.
- **Implemented as**: `scripts/07_hoeffding_model_figure.py` → `paper/figures/fig2_hoeffding_model.pdf`

### 1b. Epsilon calibration figure (optional)

**Purpose**: (Optional) generate an explicit ε calibration figure on synthetic data.

- Read synthetic cache data. For each ε ∈ {0.05, 0.075, 0.1, 0.125, 0.15, 0.2}, compute the Hoeffding-prescribed p at each (topology, distance) and check whether ρ ≥ 0.95.
- **Panel A**: Success rate (fraction of configs with ρ ≥ 0.95) vs ε. Mark ε = 0.1 (100% non-pathological), ε = 0.125 (95%).
- **Panel B**: Failure analysis at ε = 0.125 — show which configs fail (all trellis harmonic at 1,500–2,000m).
- **Status**: Not currently part of the committed paper figures.

### 1c. Modify `01_fit_rank_model.py`

- Fig 1 (headline): **keep as-is** — still motivates the paper.
- Fig 5 (error crossover): **keep as-is** — still demonstrates "precision scales with importance" (becomes Fig 4 in new numbering).
- Figs 2, 3, 4 (current √r derivation, floor, combined): **demote to appendix** or remove. The √r model becomes appendix material. Rename outputs to `fig_s*`.
- Tab 1 (parameters): **demote to appendix**. The √r parameters (k, n_min) are no longer the main result.

### 1d. Modify `03_validate_gla.py`

- Currently produces `fig6_gla_validation.pdf` showing accuracy curves with √r model-recommended p marked.
- **Change**: Mark the **Hoeffding-prescribed p** as the primary marker. Optionally show √r as secondary. The figure becomes Fig 5.

### 1e. Modify `04_validate_madrid.py`

- Already produces both √r and Hoeffding results in `madrid_validation.csv`.
- **Change figure**: Show Hoeffding as primary, √r as comparison side-by-side. Becomes Fig 6.

### 1f. Modify `05_practical_guide.py`

- **Change**: ε becomes the user-facing parameter. Show lookup curves for ε = 0.05, 0.1, 0.15. The current k/n_min-based guide is replaced.
- Becomes Fig 7.

### 1g. Generate LaTeX macros

- Add Hoeffding-specific macros: `\hoeffdingEpsilon`, `\hoeffdingDelta`.
- Add Madrid Hoeffding-specific results as macros.
- Keep legacy / comparison macros only if still referenced.

---

## Phase 2: Revised Paper Structure

### New section numbering and content

| #     | Section                   | Content                                                                                                                                                                                                                                      | Figures/Tables           |
| ----- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| §1    | Introduction              | Problem (multi-scale centrality cost), existing bounds (source vs path sampling), the gap (no localised validation), our contribution (adapt EW, calibrate ε, validate)                                                                      | —                        |
| §2    | Methods                   | Centrality measures, reach, accuracy metrics (normalised additive error primary, Spearman secondary), experimental design (synthetic topologies, boundary buffering), validation networks (GLA, Madrid)                                      | —                        |
| §3.1  | The Opportunity           | Sampling works, speedup grows with distance                                                                                                                                                                                                  | Fig 1 (keep)             |
| §3.2  | The Hoeffding Model       | Adapt EW to localised centrality. Each source contributes bounded variable. Union bound over r targets. k = log(2r/δ)/(2ε²). Properties: p → 1.0 at low reach (floor emerges), p ~ O(log r / r) at high reach. Connection to Hájek variance. | **Fig 2 (NEW)**          |
| §3.3  | Calibrating ε             | ε = 0.1 validated on synthetic data (100% success). ε = 0.125 drops to 95%. Failures are pathological. ε has direct interpretation.                                                                                                          | **Fig 3 (NEW)**          |
| §3.4  | Error Structure           | Precision scales with importance. Absolute MAE up, normalised MAE down.                                                                                                                                                                      | Fig 4 (was Fig 5)        |
| §4.1  | GLA Validation            | Hoeffding-prescribed p achieves ρ > 0.97 at all distances.                                                                                                                                                                                   | Fig 5 (was Fig 6), Tab 1 |
| §4.2  | Madrid Validation         | External validation. Hoeffding vs √r side-by-side.                                                                                                                                                                                           | Fig 6 (was Fig 7), Tab 2 |
| §5.1  | Practical Guidelines      | ε as user-facing parameter. Lookup charts.                                                                                                                                                                                                   | Fig 7 (was Fig 8), Tab 3 |
| §5.2  | Limitations               | Uniform sampling, betweenness margin thinner, shortest-path only, two European cities, top-k harmonic is 0.86–0.95                                                                                                                           | —                        |
| §5.3  | Conclusion                | Summary, √r as secondary observation, future work                                                                                                                                                                                            | —                        |
| App A | Technical Details         | Synthetic generation, GLA preprocessing, NetworkX verification, Hájek estimator, cityseer implementation                                                                                                                                     | —                        |
| App B | Empirical √r Scaling      | The √r observation, power exponent diagnostics, model comparison (demoted from main text)                                                                                                                                                    | Fig S1, Tab S1           |
| App C | Bernstein Comparison      | Bernstein vs Hoeffding (from script 10)                                                                                                                                                                                                      | Fig Bernstein            |
| App D | Detailed Bound Evaluation | EW bound on synthetic data (success rates, conservatism)                                                                                                                                                                                     | Fig EW synthetic, Tab EW |

### Key rewrites in `main.tex`

**Abstract**: Replace entirely. New arc: "We adapt the EW source-sampling bound to localised centrality, replacing node count with distance-bounded reach. With conventional ε = 0.1, the resulting Hoeffding model prescribes per-distance sampling probabilities that deliver both additive error guarantees and rank preservation (ρ ≥ 0.97) on two real networks, with speedups of X–Y×."

**Introduction (§1)**:

- Keep ¶1 (computational cost problem) mostly as-is.
- Reframe ¶2: the EW bound adapts straightforwardly; the question is whether it delivers practical speedups (not whether it's "too conservative").
- Reframe ¶3 (contributions): (1) formal adaptation of EW to localised centrality, (2) calibration of ε = 0.1, (3) validation showing ρ ≥ 0.97 as a consequence.
- §1.1 (computational cost): keep as-is.
- §1.2 (background): mostly keep, but adjust final paragraph to frame the question as "does the adapted EW bound work?" rather than "can we replace it?"

**Methods (§2)**:

- §2.1 (centrality measures): keep.
- §2.2 (experimental design): keep but reorder: present normalised additive error as the primary metric, Spearman ρ as the secondary/consequence metric.
- §2.3 (accuracy metric): major rewrite. Lead with normalised additive error (the Hoeffding target). Then introduce Spearman ρ as "the practitioner's metric" that we report as a consequence. Keep the structural alignment argument but frame it as explaining _why_ additive bounds also preserve ranks, not as justifying a rank-based model.
- §2.4 (validation networks): keep.

**Sampling Model (§3)**:

- §3.1 (opportunity): keep Fig 1 as-is.
- §3.2 (Hoeffding model): **entirely new**. Replace the √r derivation (current §3.2–3.3) with: (a) each source contributes a bounded random variable, (b) union bound over r targets, (c) formula, (d) properties (floor emerges, O(log r / r) scaling), (e) comparison with √r empirical observation.
- §3.3 (ε calibration): **entirely new**. Replace floor + logistic regression with: sweep ε on synthetic data, report success rates, identify ε = 0.1 as the sweet spot.
- §3.4 (error structure): keep current §3.4 content, just renumber.
- §3.5 (model selection): **remove** from main text. Demote to Appendix B.
- §3.6 (relationship to bounds): **remove** from main text. The Hoeffding bound is now the model, so there's nothing to compare against. The table comparing approaches stays (reframed) but the long discussion about why our model needs fewer samples is replaced by a brief note that the bound works better than expected on street networks due to low variance.

**Algorithm 1**: Replace formula. Input: ε, δ (not k, n_min). Formula: k_d = log(2r_d/δ)/(2ε²), p_d = min(1, k_d/r_d).

**Validation (§4)**: Mostly keep structure but present Hoeffding results as primary. The GLA and Madrid tables/figures show Hoeffding-prescribed p. √r shown as comparison in Madrid.

**Discussion (§5)**: ε is the user-facing parameter. Practical guide recast. Add top-k disclosure. Add validation breadth caveat. Mention √r as conservative alternative.

---

## Phase 3: Execution Order

1. Run `scripts/00_generate_cache.py` (if needed)
2. Run `scripts/01_fit_rank_model.py` and `scripts/07_hoeffding_model_figure.py` (figures)
3. Run `scripts/03_validate_gla.py` and `scripts/04_validate_madrid.py` (validation CSVs + tables)
4. Run `scripts/05_practical_guide.py` (figure + lookup table)
5. Run `scripts/06_generate_macros.py` (paper macros)
6. Compile `paper/main.tex`

---

## Figures Summary (New → Old mapping)

| New    | Content                    | Source                                       |
| ------ | -------------------------- | -------------------------------------------- |
| Fig 1  | Headline opportunity       | Keep as-is                                   |
| Fig 2  | **Hoeffding model**        | NEW (script 11)                              |
| Fig 3  | **ε calibration**          | NEW (script 12)                              |
| Fig 4  | Error crossover            | Was Fig 5 (keep content)                     |
| Fig 5  | GLA validation             | Was Fig 6 (modify to mark Hoeffding p)       |
| Fig 6  | Madrid validation          | Was Fig 7 (modify to show Hoeffding primary) |
| Fig 7  | Practical guide            | Was Fig 8 (modify to use ε parameter)        |
| Fig S1 | Power exponent diagnostics | Keep (appendix)                              |
| Fig S2 | √r model derivation        | Was Fig 2 (demoted)                          |
| Fig S3 | Bernstein comparison       | Keep (appendix)                              |
| Fig S4 | EW synthetic bound         | Keep (appendix)                              |

## Tables Summary

| New    | Content                         | Source                                 |
| ------ | ------------------------------- | -------------------------------------- |
| Tab 1  | GLA validation results          | Was Tab 2 (reorder, Hoeffding primary) |
| Tab 2  | Madrid validation (both models) | NEW or modify existing                 |
| Tab 3  | Practical lookup (ε-based)      | Was Tab 3 (modify)                     |
| Tab S1 | √r model parameters             | Was Tab 1 (demoted)                    |
| Tab S2 | Model comparison (AIC/Brier)    | Was in main text (demoted)             |
| Tab S3 | EW bound comparison             | Keep (appendix)                        |

---

## Items from "Carried Forward" list to address during rewrite

- **Top-k precision disclosure**: Add to §5.2 (limitations). Harmonic top-k is 0.86–0.95; betweenness 0.93–0.97.
- **HT vs Hájek**: Clarify in Algorithm 1 and appendix. The implementation uses Horvitz--Thompson-style inverse-probability weighting ($1/p$ scaling).
- **Pilot phase overhead**: Bound in §5.1 — "< 1% at distances above 5km."
- **Validation breadth caveat**: Add to §5.2 — "two European cities with similar morphologies."
- **Sensitivity analysis**: Consider adding a brief table showing speedup and ρ at ε = 0.05, 0.10, 0.15 to §5.1.
