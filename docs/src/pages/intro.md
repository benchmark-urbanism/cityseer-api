---
layout: '@src/layouts/PageLayout.astro'
---

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by network-based methods that have been developed from the ground-up for micro-morphological urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given street-front location. Importantly, `cityseer` computes metrics directly over the street network and offers distance-weighted variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian walking distances into account.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io), etc.), Open Street Map workflows with [`OSMnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/), [`geopandas`](https://geopandas.org/en/stable/) (and [`momepy`](http://docs.momepy.org)) stack of packages. The underlying algorithms are parallelised and implemented in `rust` so that the methods can be scaled to large networks. In-out convenience methods are provided for interfacing with `networkX` and graph cleaning tools aid the incorporation of complex network representations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api). Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package.

Code tests are run against Python versions `3.11` - `3.13`.

## Getting Started

A growing collection of recipes and examples is available via the [`Cityseer Examples`](https://benchmark-urbanism.github.io/cityseer-examples/) site. The example notebooks include workflows showing how to run graph cleaning, network centralities, and land-use accessibility analysis from data sources such as OSM or geospatial files (e.g. GeoPackages & Shapefiles).

## Local-Scale Analysis

`cityseer` is developed from the ground-up for pedestrian-scale urban analysis. It builds-on and further best-practices for urban analytics:

- It uses localised network analysis (as opposed to global forms of analysis) using a 'moving-window' methodology. A node is selected, the graph is then isolated at a selected distance threshold around the node, metrics are then computed, and then the process subsequently repeats for every other node in the network. `cityseer` exclusively uses localised methods for network analysis because they do not suffer from the same issues as global methods, which are inherently problematic because of edge roll-off effects. Localised methods have the distinct advantage of being comparable across different locations and cities, while also being capable of targeting both smaller and larger distance thresholds to reveal patterns at different scales of analysis.
- It is common to use either shortest-distance (metric) or simplest-path (shortest angular or geometric distance) heuristics for network analysis. When using simplest-path (angular) distances, it is necessary to modify the underlying shortest-path algorithms to prevent side-stepping of sharp angular turns; otherwise, two smaller side-steps can be combined to 'short-cut' sharp corners.
- `cityseer` supports analysis for both primal and dual graph representations, and contains methods for converting from primal (intersection-based) to dual (street-segment-based) representations. The dual representation retains accurate street lengths and geometry (angles) while affording the opportunity to measure and visualise metrics relative to streets instead of intersections.
- `cityseer` supports both unweighted and weighted (spatial impedance) forms of centrality, accessibility, and mixed-use methods.
- To support the evaluation of measures at finely-spaced intervals along street fronts, `cityseer` includes support for network decomposition.
- Granular evaluation of land-use accessibilities and mixed-uses requires that land uses be assigned to the street network in a contextually precise manner. `cityseer` assigns data-points to the nearest adjacent street segment and then allows access over the network from both sides, thereby allowing precise distances to be calculated dynamically based on the direction of approach.
- Centrality methods are susceptible to topological distortions arising from 'messy' graph representations as well as due to the conflation of topological and geometrical properties of street networks. `cityseer` addresses these through the inclusion of graph cleaning functions and procedures for splitting geometrical properties from topological representations.

The broader emphasis on localised methods and how `cityseer` addresses these is broached in the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827). `cityseer` includes a variety of convenience methods for the general preparation of networks and their conversion into (and out of) the lower-level data structures used by the underlying algorithms. These graph utility methods are designed to work with `NetworkX` to facilitate ease of use. A complement of code tests has been developed to maintain the codebase's integrity through general package maintenance and upgrade cycles. Shortest-path algorithms, harmonic closeness, and betweenness algorithms are tested against `NetworkX`. Mock data and test plots have been used to visually confirm the intended behaviour for divergent simplest and shortest-path heuristics and for testing data assignment to network nodes given various scenarios.

The best way to get started is to see the [`Cityseer Examples`](https://benchmark-urbanism.github.io/cityseer-examples/) site, which contains a number of recipes for a variety of use-cases.

## Support

Please report bugs to the [github issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues) and direct more general questions to [Github Discussions](https://github.com/benchmark-urbanism/cityseer-api/discussions).

Time permitting, for general help with workflows or feedback in support of research projects or papers, please start a new [discussion on Github](https://github.com/benchmark-urbanism/cityseer-api/discussions).

## Attribution

Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package for producing research.

`cityseer` is licensed as AGPLv3. Please [get in touch](mailto:info@benchmarkurbanism.com) if you need technical support developing related workflows, or if you wish to sponsor the development of additional or bespoke functionality.

If using the package to produce visual plots and outputs, please display the cityseer logo and a link to the documentation website.

<img src="/logos/cityseer_logo_white.png" alt="Cityseer white logo." width="350"></img>

<img src="/logos/cityseer_logo_light_red.png" alt="Cityseer red logo." width="350"></img>
