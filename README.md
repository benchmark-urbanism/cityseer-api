# cityseer

A `Python` package for pedestrian-scale network-based urban analysis: network analysis, landuse accessibilities & mixed uses, statistical aggregations.

[![PyPI version](https://badge.fury.io/py/cityseer.svg)](https://badge.fury.io/py/cityseer)

[![publish package](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml)

[![deploy docs](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml)

Examples: <https://benchmark-urbanism.github.io/cityseer-examples/>

API Documentation: <https://cityseer.benchmarkurbanism.com/>

Issues: <https://github.com/benchmark-urbanism/cityseer-api/issues>

Questions: <https://github.com/benchmark-urbanism/cityseer-api/discussions>

## Installation

```bash
pip install cityseer
```

## Development

Contributions are welcome — please open an [issue](https://github.com/benchmark-urbanism/cityseer-api/issues) or [discussion](https://github.com/benchmark-urbanism/cityseer-api/discussions) before larger changes.

> [!IMPORTANT]
> Active development happens on the **`dev`** branch; `master` tracks the latest released version. Please **base branches and target pull requests against `dev`**, not `master`:
>
> ```bash
> gh pr create --base dev
> ```
>
> Pushing an alpha tag (e.g. `4.25.0b1`) from `dev` publishes a pre-release to PyPI for testing ahead of a stable release off `master`.

### Setup

The loop-intensive algorithms are written in `rust` and exposed to `Python` via [`maturin`](https://www.maturin.rs/), so a `rust` toolchain is required alongside [`uv`](https://docs.astral.sh/uv/):

```bash
brew install uv rust rust-analyzer rustfmt   # or your platform's equivalent
uv sync                                       # creates the venv and builds the rust extension
```

After editing `rust` sources, rebuild the extension before re-running Python:

```bash
uv run maturin develop                        # or re-run `uv sync`
```

### Verify before pushing

Run the same formatting, linting, type-checking, and test suite that CI runs:

```bash
uv run poe verify_project                     # ruff format && ruff check && ty check && pytest ./tests
```

To preview the documentation site locally:

```bash
uv run poe docs_dev
```

## Cite

Cite as: [The cityseer Python package for pedestrian-scale network-based urban analysis](https://journals.sagepub.com/doi/full/10.1177/23998083221133827)

## Background

The `cityseer-api` `Python` package addresses a range of issues specific to computational workflows for urban analytics from an urbanist's point of view and contributes a combination of techniques to support developments in this field:

- High-resolution workflows including localised moving-window analysis with strict network-based distance thresholds; spatially precise assignment of land-use or other data points to adjacent street-fronts for improved contextual sensitivity; dynamic aggregation workflows which aggregate and compute distances on-the-fly from any selected point on the network to any accessible land-use or data point within a selected distance threshold; facilitation of workflows eschewing intervening steps of aggregation and associated issues such as ecological correlations; and the optional use of network decomposition to increase the resolution of the analysis.
- Localised computation of network centralities using shortest paths on primal or dual graphs, and simplest-path heuristics on dual graphs, including tailored methods such as harmonic closeness centrality, and segmented versions of centrality (which convert centrality methods from a discretised to an explicitly continuous form). For more information, see [_"Network centrality measures and their correlation to mixed-uses at the pedestrian-scale"_](https://arxiv.org/abs/2106.14040).
- Land-use accessibilities and mixed-use calculations incorporate dynamic and directional aggregation workflows with the optional use of spatial-impedance-weighted forms. Shortest-path workflows operate on primal or dual graphs, while simplest-path workflows require dual graphs. For more information, see [_"The application of mixed-use measures at the pedestrian-scale"_](https://arxiv.org/abs/2106.14048).
- Network centralities dovetailed with land-use accessibilities, mixed-uses, and general statistical aggregations from the same points of analysis to generate multi-scalar and multi-variable datasets facilitating downstream data science and machine learning workflows. For examples, see [_"Untangling urban data signatures: unsupervised machine learning methods for the detection of urban archetypes at the pedestrian scale"_](https://arxiv.org/abs/2106.15363) and [_"Prediction of 'artificial' urban archetypes at the pedestrian-scale through a synthesis of domain expertise with machine learning methods"_](https://arxiv.org/abs/2106.15364).
- The inclusion of graph cleaning methods reduce topological distortions for higher quality network analysis and aggregation workflows while accommodating workflows bridging the wider `NumPy` ecosystem of scientific and geospatial packages.
- Underlying loop-intensive algorithms are implemented in `rust`, allowing these methods to be applied to large and, optionally, decomposed graphs, which have substantial computational demands.
