---
layout: '@src/layouts/PageLayout.astro'
---

# Getting Started

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods.

The getting started guide and a growing collection of other examples is available via the [`Cityseer Examples`](https://benchmark-urbanism.github.io/cityseer-examples/) site. The example notebooks include workflows showing how to run graph cleaning, network centralities, and land-use accessibility analysis for some real-world situations.

`cityseer` is underpinned by network-based methods that have been developed from the ground-up for micro-morphological urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given streetfront location. Importantly, `cityseer` computes metrics directly over the street network and offers distance-weighted variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian walking distances into account.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io), etc.), Open Street Map workflows with [`OSMnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/), [`geopandas`](https://geopandas.org/en/stable/) (and [`momepy`](http://docs.momepy.org)) stack of packages. The underlying algorithms are parallelised and implemented in `rust` so that the methods can be scaled to large networks. In-out convenience methods are provided for interfacing with `networkX` and graph cleaning tools aid the incorporation of messier network representations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api). Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package. Associated papers introducing the package and demonstrating the forms of analysis that can be done with it are [available at `arXiv`](https://arxiv.org/a/simons_g_1.html).

Code tests are run against Python versions `3.10` - `3.12`.

## Old versions

For documentations of older versions of `cityseer`, please refer to the docstrings which are directly embedded in the code for the respective version:

- Documentation for [`v1.x`](https://github.com/benchmark-urbanism/cityseer-api/tree/v1.2.1/cityseer)
- Documentation for [`v2.x`](https://github.com/benchmark-urbanism/cityseer-api/tree/v2.0.0/cityseer)
- Documentation for [`v3.x`](https://github.com/benchmark-urbanism/cityseer-api/tree/v3.7.2)
