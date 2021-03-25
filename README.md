# cityseer

[![publish package](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/publish_package.yml)

[![deploy docs](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml/badge.svg)](https://github.com/benchmark-urbanism/cityseer-api/actions/workflows/firebase-hosting-merge.yml)

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods.

`cityseer` is underpinned by network-based methods that have been developed from the ground-up for localised urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given streetfront location. It can be used to compute a variety of node or segment-based centrality methods, landuse accessibility and mixed-use measures, and statistical aggregations. Aggregations are computed dynamically --- directly over the network while taking into account the direction of approach --- and can incorporate spatial impedances and network decomposition to further accentuate spatial precision.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io)), Open Street Map workflows [`osmnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/) stack of scientific packages. The underlying algorithms are are implemented in [`numba`](https://numba.pydata.org/) JIT compiled code so that the methods can be applied to large decomposed networks. In-out convenience methods are provided for interfacing with [`networkX`](https://networkx.github.io/) and graph cleaning tools aid the incorporation of messier network respresentations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The documentation is available from [cityseer.benchmarkurbanism.com](https://cityseer.benchmarkurbanism.com) and the github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api).
