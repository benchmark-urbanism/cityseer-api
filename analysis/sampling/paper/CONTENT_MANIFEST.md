# Content Manifest

**Model:** Hoeffding / Eppstein–Wang reach-based source sampling.
`k = log(2·r_c/δ) / (2·ε²)`, `p = min(1, k/r_c)`, canonical reach `r_c = π·d²/s²`.
Defaults: **ε = 0.05, δ = 0.1, s = 175 m** (single source of truth: `pysrc/cityseer/sampling.py`;
mirrored in `scripts/utilities.py`, which the validation scripts assert against at runtime).

The grid spacing `s` is a **fixed** canonical reference (not fitted). The tolerance `ε` is the
**single calibrated parameter**: it is tuned so the sparsest validated network (Cary, NC) preserves
node rankings (ρ ≥ 0.95); denser networks then clear the target comfortably.

> Authoritative pipeline = `scripts/run_all.py` + the individual script headers. This manifest is a
> high-level index. Do not hardcode paper values — they all come from the generated macros in
> `paper/tables/model_macros.tex`.

## Validation networks (official road data; OSM boundaries)

| Network | Road source | Character |
| ------- | ----------- | --------- |
| Greater London | Ordnance Survey Open Roads | dense metro |
| Greater Madrid | official *Red Viaria* network | dense metro |
| Cary, NC       | US Census TIGER/Line **edges** (`ROADFLG=Y`) | low-density suburb (binding case for ε) |

Each live study area is delimited by the administrative boundary geocoded from OpenStreetMap via
OSMnx; only the road geometry differs by source.

## Pipeline (`scripts/run_all.py`, in order)

| #  | Script                     | Purpose                                            | Key outputs                                                                                         |
| -- | -------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 01 | `01_validate_gla.py`       | Greater London validation                          | `output/gla_*.csv`, `.cache/gla_n_nodes.json`                                                        |
| 02 | `02_validate_madrid.py`    | Greater Madrid validation                          | `output/madrid_*.csv`, `.cache/madrid_n_nodes.json`                                                  |
| 03 | `03_validate_cary.py`      | Cary, NC (suburban) validation                     | `output/cary_*.csv`, `.cache/cary_n_nodes.json`                                                      |
| 04 | `04_validate_woodlands.py` | Held-out validation: The Woodlands, TX (suburban) | `output/woodlands_*.csv`, `tables/woodlands_n_nodes.json` |
| 05 | `05_figures_validation.py` | Validation figures (accuracy, speedup, reach)      | `figures/fig2_error_vs_reach.pdf`, `fig4_validation_accuracy.pdf`, `fig5_validation_speedup.pdf`, `fig6_reach_comparison.pdf` |
| 06 | `06_generate_macros.py`    | LaTeX macros, validation tables, practical-guide   | `tables/model_macros.tex`, `tab2/tab4/tab5/tab6_*.tex`, `tab_distance_lookup.tex`, `figures/fig3_practical_guide.pdf` |
| 08 | `08_figure_method.py` | Method schematic (worked example) | `figures/fig1_method_schematic.pdf/.svg` |
| 09 | `09_figure_adaptive_comparison.py` | Baseline vs adaptive comparison | `figures/fig12_baseline_vs_adaptive.pdf/.svg` |
| 07 | `07_figures_spatial.py`    | Spatial-error figures (all networks)               | `figures/fig7_rank_shift.png`, `figures/fig11_decile_transition.pdf`                         |
| –  | `utilities.py`             | Shared constants / utilities                       | (imported by the others)                                                                            |
| –  | `fetch_tiger.py`      | Download TIGER edges for a suburb + buffer (Cary, Woodlands)    | `temp/tiger_{place}/*.zip`                                                                              |

Adaptive validation (not part of `run_all.py`): `validate_adaptive.py` runs the per-node runtime path against the cached exact baselines and writes `output/{network}_validation_adaptive.csv` (consumed by `06_generate_macros.py` for `tab7` and `\adaptiveMinRho`).

Standalone calibration diagnostics (not part of `run_all.py`): `epsilon_sweep.py`,
`cary_s_sweep.py` — map ρ vs ε / s on Cary to locate the ρ = 0.95 crossing.

## Paper

`paper/main.tex` `\input`s `tables/model_macros.tex` and the `tab*.tex` tables, and `\includegraphics`
the `fig*` figures above. The build artifacts (`paper/figures/`, `paper/tables/`, and the `.cache/`
network caches) are **generated, not committed** — run the pipeline before compiling.

## Build

```bash
cd analysis/sampling/scripts && python run_all.py            # regenerate all (needs network caches)
cd ../paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
