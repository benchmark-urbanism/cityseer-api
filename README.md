# cityseer

A `Python` package for pedestrian-scale network-based urban analysis: network analysis, landuse accessibilities and mixed uses, statistical aggregations.

[![PyPI version](https://badge.fury.io/py/cityseer.svg)](https://badge.fury.io/py/cityseer)

[![publish package](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml)

[![deploy docs](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml)

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- Documentation for `v1.x`: see documented code per tagged release [v1](https://github.com/benchmark-urbanism/cityseer-api/tree/v1.2.1/cityseer)
- Documentation for `v2.x`: see documented code per tagged release [v2](https://github.com/benchmark-urbanism/cityseer-api/tree/v2.0.0/cityseer)
- Documentation for `v3.x`: see documented code per tagged release [v3](https://github.com/benchmark-urbanism/cityseer-api/tree/v3.7.2)
- Documentation for `v4+`: <https://cityseer.benchmarkurbanism.com/>

Demo Notebooks: <https://benchmark-urbanism.github.io/cityseer-examples/>

Issues: <https://github.com/benchmark-urbanism/cityseer-api/issues>

Questions: <https://github.com/benchmark-urbanism/cityseer-api/discussions>

Cite as: [The cityseer Python package for pedestrian-scale network-based urban analysis](https://journals.sagepub.com/doi/full/10.1177/23998083221133827)

The `cityseer-api` `Python` package addresses a range of issues specific to computational workflows for urban analytics from an urbanist's point of view and contributes a combination of techniques to support developments in this field:

- High-resolution workflows including localised moving-window analysis with strict network-based distance thresholds; spatially precise assignment of land-use or other data points to adjacent street-fronts for improved contextual sensitivity; dynamic aggregation workflows which aggregate and compute distances on-the-fly from any selected point on the network to any accessible land-use or data point within a selected distance threshold; facilitation of workflows eschewing intervening steps of aggregation and associated issues such as ecological correlations; and the optional use of network decomposition to increase the resolution of the analysis.
- Localised computation of network centralities using either shortest or simplest path heuristics on either primal or dual graphs, including tailored methods such as harmonic closeness centrality (which behaves more suitably than traditional variants of closeness), and segmented versions of centrality (which convert centrality methods from a discretised to an explicitly continuous form). For more information, see [_"Network centrality measures and their correlation to mixed-uses at the pedestrian-scale"_](https://arxiv.org/abs/2106.14040).
- Land-use accessibilities and mixed-use calculations incorporate dynamic and directional aggregation workflows with the optional use of spatial-impedance-weighted forms. These can likewise be applied with either shortest or simplest path heuristics and on either primal or dual graphs. For more information, see [_"The application of mixed-use measures at the pedestrian-scale"_](https://arxiv.org/abs/2106.14048).
- Network centralities dovetailed with land-use accessibilities, mixed-uses, and general statistical aggregations from the same points of analysis to generate multi-scalar and multi-variable datasets facilitating downstream data science and machine learning workflows. For examples, see [_"Untangling urban data signatures: unsupervised machine learning methods for the detection of urban archetypes at the pedestrian scale"_](https://arxiv.org/abs/2106.15363) and [_"Prediction of 'artificial' urban archetypes at the pedestrian-scale through a synthesis of domain expertise with machine learning methods"_](https://arxiv.org/abs/2106.15364).
- The inclusion of graph cleaning methods reduce topological distortions for higher quality network analysis and aggregation workflows while accommodating workflows bridging the wider `NumPy` ecosystem of scientific and geospatial packages. See the [Graph Cleaning Guide](https://cityseer.benchmarkurbanism.com/guide#graph-cleaning).
- Underlying loop-intensive algorithms are implemented in `rust`, allowing these methods to be applied to large and, optionally, decomposed graphs, which have substantial computational demands.

## Development

`pdm install`
`python -m ensurepip --default-pip`
`brew install rust rust-analyzer rustfmt`
