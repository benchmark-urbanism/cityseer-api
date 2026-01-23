# Adaptive Sampling for Network Centrality: Paper Development Strategy

This document outlines the strategy and exact steps for developing the research paper on adaptive per-distance sampling for network centrality computation.

---

## 1. Paper Overview

### Title (Working)

**"Adaptive Per-Distance Sampling for Efficient Multi-Scale Network Centrality Analysis"**

### Abstract Summary

- Problem: Computing network centrality across multiple distance scales is computationally expensive
- Key insight: Effective sample size (reachability × probability) determines accuracy
- Innovation: Per-distance adaptive sampling that calibrates probability based on network reachability
- Result: ~2× speedup while maintaining Spearman ρ ≥ 0.95 across all distance thresholds

### Target Venue

1. **Primary**: International Journal of Geographical Information Science (IJGIS)
   - Directly extends Cooper (2015) on localised centrality
   - Core urban network analysis venue
2. **Secondary**: Environment and Planning B
3. **Preprint**: arXiv (cs.SI or physics.soc-ph) for immediate visibility

---

## 2. Research Contributions

1. **Empirical relationship** between effective sample size and centrality accuracy
2. **Separate models** for harmonic closeness vs betweenness (betweenness requires ~1.5× more samples)
3. **Adaptive algorithm** that achieves consistent accuracy across all distance thresholds
4. **Open-source implementation** integrated into cityseer library

---

## 3. Paper Structure

```
paper/
├── main.tex                    # Main LaTeX document
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_methodology.tex
│   ├── 04_empirical_models.tex
│   ├── 05_algorithm.tex
│   ├── 06_validation.tex
│   ├── 07_discussion.tex
│   └── 08_conclusion.tex
├── figures/
│   ├── topologies.pdf          # Generated: Network topology visualisations
│   ├── accuracy_vs_effn.pdf    # Generated: Main results figure
│   ├── required_probability.pdf # Generated: Required p for target ρ
│   ├── adaptive_comparison.pdf  # Generated: Uniform vs adaptive per-distance
│   └── realworld_validation.pdf # Generated: London/Madrid validation
├── tables/
│   ├── model_parameters.tex    # Generated: Fitted model constants
│   ├── accuracy_by_effn.tex    # Generated: Accuracy by effective_n bins
│   ├── required_effn.tex       # Generated: Required eff_n for targets
│   └── realworld_results.tex   # Generated: Real-world validation results
├── generated/
│   └── (auto-generated data files from validation scripts)
└── references.bib
```

---

## 4. Development Steps

### Phase 1: Foundation (Current)

- [x] Core sampling analysis complete (`sampling_analysis.py`)
- [x] Adaptive sampling implementation validated (`test_adaptive_sampling.py`)
- [x] Model constants fitted and integrated into cityseer
- [ ] **Create LaTeX paper skeleton**
- [ ] **Create paper generation script**

### Phase 2: Real-World Validation

- [ ] **Create validation script for London network**
- [ ] **Create validation script for Madrid network**
- [ ] Verify empirical models generalise to real networks
- [ ] Document any topology-specific deviations

### Phase 3: Figure & Table Generation

- [ ] Convert existing PNG figures to PDF (publication quality)
- [ ] Generate LaTeX tables from analysis results
- [ ] Create comparison figures (uniform vs adaptive by distance)

### Phase 4: Writing

- [ ] Draft Introduction (positioning, contributions)
- [ ] Draft Background (localised centrality, sampling methods)
- [ ] Draft Methodology (experimental design)
- [ ] Draft Empirical Models (fitting procedure, results)
- [ ] Draft Algorithm (adaptive sampling procedure)
- [ ] Draft Validation (synthetic + real-world results)
- [ ] Draft Discussion (limitations, future work)
- [ ] Draft Conclusion

### Phase 5: Finalisation

- [ ] Internal review and revision
- [ ] Format for target venue
- [ ] Submit to arXiv
- [ ] Submit to journal

---

## 5. Scripts to Create/Modify

### 5.1 `generate_paper_assets.py`

Main script that:

1. Runs validation on London and Madrid networks
2. Generates all figures as PDF
3. Generates all LaTeX tables
4. Outputs to `paper/figures/` and `paper/tables/`

### 5.2 `validate_realworld.py`

Script that:

1. Downloads OSM networks for London and Madrid using cityseer
2. Runs the same experimental protocol as `sampling_analysis.py`
3. Compares observed accuracy with model predictions
4. Outputs validation metrics

### 5.3 Modifications to existing scripts

- `sampling_analysis.py`: Add PDF figure output, LaTeX table generation
- `test_adaptive_sampling.py`: Add per-distance comparison tables

---

## 6. Real-World Validation Protocol

### Network Selection

| City            | Centre Point   | Buffer | Expected Characteristics    |
| --------------- | -------------- | ------ | --------------------------- |
| London (Soho)   | -0.134, 51.514 | 2000m  | Dense, irregular historical |
| Madrid (Centro) | -3.703, 40.416 | 2000m  | Mediterranean grid + radial |

### Experimental Parameters

- Distances: [200, 500, 1000, 2000, 5000] metres
- Sampling probabilities: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
- Runs per configuration: 10 (for variance estimation)
- Metrics: Harmonic closeness, Betweenness

### Validation Metrics

1. **Model accuracy**: Compare predicted ρ vs observed ρ
2. **Model coverage**: Does 95% CI contain observed values?
3. **Topology effects**: Document any systematic deviations

---

## 7. Key Figures

### Figure 1: Network Topologies

- Synthetic networks (trellis, tree, linear)
- Real-world networks (London, Madrid)
- Show node counts and characteristics

### Figure 2: Accuracy vs Effective Sample Size

- Scatter plot with fitted model curve
- Separate panels for harmonic and betweenness
- Colour by topology type

### Figure 3: Required Sampling Probability

- Curves showing required p for target ρ levels (0.90, 0.95, 0.97)
- X-axis: reachability, Y-axis: required probability

### Figure 4: Adaptive vs Uniform Comparison

- Per-distance accuracy comparison
- Shows uniform fails at short distances
- Shows adaptive maintains ρ ≥ 0.95 throughout

### Figure 5: Real-World Validation

- Observed vs predicted accuracy for London/Madrid
- Error bars showing variance

---

## 8. Key Tables

### Table 1: Model Parameters

| Metric      | A     | B     | RMSE  | Required eff_n (ρ=0.95) |
| ----------- | ----- | ----- | ----- | ----------------------- |
| Harmonic    | 32.40 | 31.54 | 0.041 | 617                     |
| Betweenness | 48.31 | 49.12 | 0.049 | 917                     |

### Table 2: Accuracy by Effective Sample Size

| eff_n range | Harmonic ρ | Between ρ | Scale ratio |
| ----------- | ---------- | --------- | ----------- |
| 0-50        | ...        | ...       | ...         |
| 50-100      | ...        | ...       | ...         |
| ...         | ...        | ...       | ...         |

### Table 3: Adaptive Sampling Results

| Topology | Full time | Adaptive time | Speedup | ρ achieved |
| -------- | --------- | ------------- | ------- | ---------- |

### Table 4: Real-World Validation

| City | Nodes | Model pred ρ | Observed ρ | Within 95% CI? |

---

## 9. Commands to Execute

### Run full paper generation

```bash
cd analysis/paper
python generate_paper_assets.py
```

### Compile LaTeX

```bash
cd analysis/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Run real-world validation only

```bash
cd analysis
python validate_realworld.py --cities london madrid --output paper/generated/
```

---

## 10. Timeline Checklist

- [ ] Week 1: LaTeX skeleton, real-world validation script
- [ ] Week 2: Run validation, generate figures/tables
- [ ] Week 3: Write Introduction, Background, Methodology
- [ ] Week 4: Write Empirical Models, Algorithm, Validation
- [ ] Week 5: Write Discussion, Conclusion, revisions
- [ ] Week 6: Final formatting, arXiv submission

---

## 11. File Locations

| Item                   | Location                                  |
| ---------------------- | ----------------------------------------- |
| Strategy document      | `analysis/paper/PAPER_STRATEGY.md`        |
| Main paper             | `analysis/paper/main.tex`                 |
| Asset generation       | `analysis/paper/generate_paper_assets.py` |
| Real-world validation  | `analysis/validate_realworld.py`          |
| Existing analysis      | `analysis/sampling_analysis.py`           |
| Existing adaptive test | `analysis/test_adaptive_sampling.py`      |
| Synthetic substrates   | `analysis/utils/substrates.py`            |
| Output figures         | `analysis/paper/figures/`                 |
| Output tables          | `analysis/paper/tables/`                  |
| Generated data         | `analysis/paper/generated/`               |

---

_Last updated: 2026-01-23_
