# cityseer

[![publish package](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml)

[![deploy docs](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml)

The `cityseer` package addresses a range of issues specific to computational workflows addressing urban analytics from an urbanist's point of view. It contributes a range of techniques and computational advancements to further support developments in this field:

- High-resolution workflows including localised moving-window analysis with strict network-based distance thresholds; spatially precise assignment of landuse or other data-points to adjacent street-fronts for improved contextual sensitivity; dynamic aggregation workflows which can aggregate and compute distances on-the-fly from any selected point on the network to any accessible land-use or data-point within a selected distance threshold; facilitates workflows eschewing intervening steps of aggregation and associated issues such as ecological correlations; and affords the optional use of network decomposition to increase the resolution of analysis.
- Localised computation of network centralities using either shortest or simplest path heuristics on either primal or dual graphs, and includes tailored methods such as harmonic closeness centrality which behaves more suitably than traditional globalised variants of closeness and segmentised versions of centrality which can be used to explicitly convert centrality methods from a discretised to continuous form.
- Landuse accessibilities and mixed-use calculations are computed using the aforementioned dynamic and directional aggregation workflows with the optional use of spatial-impedance-weighted forms. These can be applied with either shortest or simplest path heuristics and on either primal or dual graphs.
- Network centralities can be dovetailed with landuse accessibilities, mixed-uses, and general statistical aggregations to generate multi-scalar and multi-variable datasets facilitating downstream data-science and machine-learning workflows.
- The inclusion of graph cleaning methods can be used to help reduce topological distortions for higher quality network analysis and aggregation workflows while accommodating workflows bridging to `OSMnx` and to the `numpy` ecosystem of scientific and geospatial packages.
- `Numba` JIT compilation of underlying loop-intensive algorithms allows for these methods to be scaled to large and, optionally, decomposed graphs.

Documentation: <https://cityseer.benchmarkurbanism.com/>

Issues: <https://github.com/benchmark-urbanism/cityseer-api/issues>

Questions: <https://github.com/benchmark-urbanism/cityseer-api/discussions>
