# Argument Notes

## The argument in one sentence

Adapting the Eppstein-Wang source-sampling bound to localised centrality gives a zero-parameter Hoeffding model, k = log(2r/delta)/(2eps^2), that at eps = 0.1 delivers both additive-error guarantees and rank preservation (rho >= 0.979; minimum 0.9794 on Madrid betweenness at 20km) on real street networks, with speedups of up to 63x.

## The logical chain

1. Localised centrality at multiple distance thresholds is effectively quadratic. Sampling reduces cost by computing from a random subset of sources.
2. Each sampled source contributes a bounded random variable to each target node's centrality estimate. Hoeffding's inequality with a union bound over r reachable target nodes (the Eppstein-Wang framework, substituting reach r for global node count n) gives: k = log(2r/delta)/(2eps^2), p = min(1, k/r).
3. This is a formal bound on additive normalised error at every node simultaneously. It has zero fitted parameters: eps and delta are user-chosen (default 0.1, 0.1).
4. At low reach, k > r so p = 1 (exact computation). No separate floor parameter is needed.
5. At high reach, p decreases as O(log r / r), giving speedups that grow with network size.
6. On street networks, the bound also preserves rank order (Spearman rho >= 0.979; all configurations pass rho >= 0.95). This is structural: high-reach nodes get high effective sample sizes and precise estimates; these are precisely the high-centrality nodes whose ranking matters. Low-reach peripheral nodes get near-exact computation (p -> 1).
7. Validation on GLA (294k nodes) and Madrid (99k nodes) at 1-20km confirms this. Minimum rho = 0.9794 (Madrid betweenness, 20km). All distances pass rho >= 0.95.

## What is novel

1. **The adaptation**: applying EW to distance-bounded centrality (r for n) and showing it works at eps = 0.1. The common assumption is that EW bounds are "too conservative" -- we show they are not, for localised centrality on street networks.
2. **Rank preservation as a consequence**: additive eps = 0.1 implies rho >= 0.979 (minimum observed across both networks). No separate rank-calibrated model is needed.
3. **Validation at scale**: from small synthetic networks to GLA and Madrid, the model generalises without fitting.

## What is obvious (do not oversell)

- The localisation step (r for n) is trivially direct for source-sampling.
- Hoeffding's inequality is a standard tool.
- Distance-dependent calibration follows automatically from r varying by distance.

## Paper structure

### Section 1: Introduction

Problem: multi-scale localised centrality is expensive. Source-sampling bounds (EW) adapt to the localised setting by replacing n with r. We show this adapted bound delivers practical speedups while preserving rank order.

### Section 2: Methods

Define harmonic closeness, betweenness, reach. Describe experimental design: 3 synthetic topologies, 7 distances, 12 sampling probabilities. Accuracy metrics: Spearman rho (rank) and normalised additive error (formal). Validation networks: GLA (294k nodes), Madrid (99k nodes), both with 20km live buffers.

### Section 3: The Hoeffding sampling model

**3.1 The opportunity.** Accuracy saturates well below p = 1; speedup grows with distance.
-> Fig 1: headline (accuracy vs p; speedup vs distance)

**3.2 The model.** Adapt EW: k = log(2r/delta)/(2eps^2), p = min(1, k/r). At low reach, p = 1 (floor emerges naturally). At high reach, speedup grows as O(r / log r). Show prescribed p and speedup across eps values.
-> Fig 2: Hoeffding model (Panel A: required eff_n vs reach for 3 eps values; Panel B: speedup 1/p vs reach, log-log)

**3.3 Error structure.** Absolute error increases with reach; normalised error decreases. High-importance nodes get highest precision. The Hoeffding bound line confirms observed errors are 2-3 orders of magnitude below the worst-case prediction.
-> Fig 3: error crossover (2x2: betweenness/harmonic x absolute/normalised)

### Section 4: Validation

**4.1 GLA results.**
-> Tab 2: GLA validation (distance, reach, Hoeffding p, rho_close, rho_betw, speedup)

| Distance | Reach | p | rho_close | rho_betw | Speedup |
|----------|---------|--------|-----------|----------|---------|
| 1km | 100 | 100.0% | 0.9948 | 0.9983 | 1.0x |
| 2km | 411 | 100.0% | 0.9979 | 0.9993 | 1.0x |
| 5km | 2,783 | 19.6% | 0.9875 | 0.9912 | 5.1x |
| 10km | 11,590 | 5.3% | 0.9930 | 0.9846 | 18.8x |
| 20km | 43,165 | 1.6% | 0.9959 | 0.9824 | 63.2x |

**4.2 Madrid results.**
-> Tab 4: Madrid validation

| Distance | Reach | p | rho_close | rho_betw | Speedup |
|----------|---------|--------|-----------|----------|---------|
| 5km | 1,451 | 35.4% | 0.9969 | 0.9949 | 2.8x |
| 10km | 5,219 | 11.1% | 0.9974 | 0.9872 | 9.0x |
| 20km | 15,300 | 4.1% | 0.9976 | 0.9794 | 24.2x |

All pass rho >= 0.95. Minimum rho: 0.9794 (Madrid betweenness, 20km). Betweenness is the binding constraint at every distance.

### Section 5: Discussion and practical guidelines

Practical guidance: choose eps (default 0.1), compute p = min(1, k/r). Lookup charts and table for common scenarios.
-> Fig 4: practical guide (Panel A: required p by reach with scenario annotations; Panel B: speedup curve)
-> Tab 1: EW comparison across eps values
-> Tab 3: practical lookup table

Limitations: uniform sampling only, shortest-path distance only, two European cities. Betweenness margin thinner than closeness at high reach.

## Figure inventory

| Figure | Script | Role |
|--------|--------|------|
| Fig 1 (headline) | 01_fit_rank_model.py | Motivates the paper: sampling works |
| Fig 2 (Hoeffding model) | 07_hoeffding_model_figure.py | The model: eff_n and speedup vs reach |
| Fig 3 (error crossover) | 01_fit_rank_model.py | Precision scales with importance |
| Fig 4 (practical guide) | 05_practical_guide.py | Practitioner lookup charts |

## Table inventory

| Table | Script | Role |
|-------|--------|------|
| Tab 1 (EW comparison) | 02_fit_error_model.py | Required k and p across eps values |
| Tab 2 (GLA validation) | 03_validate_gla.py | Primary validation results |
| Tab 3 (practical lookup) | 05_practical_guide.py | Numerical reference by reach |
| Tab 4 (Madrid validation) | 04_validate_madrid.py | External validation results |
| model_macros.tex | 06_generate_macros.py | Auto-generated LaTeX macros |

## Script pipeline

| # | Script | Depends on | Outputs |
|---|--------|------------|---------|
| 00 | generate_cache.py | -- | .cache/sampling_analysis_v21.pkl |
| 01 | fit_rank_model.py | 00 | fig1, fig3 |
| 02 | fit_error_model.py | 00 | tab1, error_model_synthetic.csv |
| 03 | validate_gla.py | -- | tab2, gla_validation.csv |
| 04 | validate_madrid.py | -- | tab4, madrid_validation.csv |
| 05 | practical_guide.py | -- | fig4, tab3 |
| 06 | generate_macros.py | 03 | model_macros.tex |
| 07 | hoeffding_model_figure.py | -- | fig2 |

## Key numbers

- GLA: 294,486 nodes, 20km buffer
- Madrid: 99,166 nodes, 20km buffer
- Hoeffding model: eps = 0.1, delta = 0.1, zero fitted parameters
- GLA minimum rho: 0.9824 (betweenness, 20km)
- Madrid minimum rho: 0.9794 (betweenness, 20km)
- Maximum speedup: 63.2x (GLA 20km), 24.2x (Madrid 20km)
- EW bound success rate on synthetic data: 93.6% overall, 99.7% conditional on reach >= 100
