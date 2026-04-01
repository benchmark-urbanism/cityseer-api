---
layout: '@src/layouts/PageLayout.astro'
---

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by network-based methods that have been developed from the ground-up for micro-morphological urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given street-front location. Importantly, `cityseer` computes metrics directly over the street network and offers distance-weighted variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian walking distances into account.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io), etc.), Open Street Map workflows with [`OSMnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/), [`geopandas`](https://geopandas.org/en/stable/) (and [`momepy`](http://docs.momepy.org)) stack of packages. The underlying algorithms are parallelised and implemented in `rust` so that the methods can be scaled to large networks. In-out convenience methods are provided for interfacing with `networkX` and graph cleaning tools aid the incorporation of complex network representations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api). Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package.

Code tests are run against Python versions `3.10` - `3.13`.

## Getting Started

The [Quickstart](https://benchmark-urbanism.github.io/cityseer-examples/recipes/quickstart.html) notebook is the fastest way to see `cityseer` in action. For a comprehensive walkthrough of concepts, conventions, and features, see the [Guide](/guide). For practical worked examples covering network preparation, centrality, accessibility, statistics, visibility, and continuity analysis, see the [Cityseer Examples](https://benchmark-urbanism.github.io/cityseer-examples/) site.

```python
from shapely.geometry import box
from cityseer.network import CityNetwork

polygon = box(-0.13, 51.51, -0.12, 51.52)
cn = CityNetwork.from_osm(polygon, to_crs_code=32630)
cn.centrality_shortest(distances=[400, 800])
result_gdf = cn.to_geopandas()
result_gdf.to_file("centrality.gpkg")
```

## QGIS Plugin

A [QGIS plugin](/plugin) is available for computing localised network centrality metrics directly within QGIS without writing code. See the [plugin page](/plugin) for installation and usage instructions.

## Support

Please report bugs to the [github issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues) and direct more general questions to [Github Discussions](https://github.com/benchmark-urbanism/cityseer-api/discussions).

Time permitting, for general help with workflows or feedback in support of research projects or papers, please start a new [discussion on Github](https://github.com/benchmark-urbanism/cityseer-api/discussions).

## Attribution

Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package for producing research.

`cityseer` is licensed as AGPLv3. Please [get in touch](mailto:info@benchmarkurbanism.com) if you need technical support developing related workflows, or if you wish to sponsor the development of additional or bespoke functionality.

If using the package to produce visual plots and outputs, please display the cityseer logo and a link to the documentation website.

<img src="/logos/cityseer_logo_white.png" alt="Cityseer white logo." width="350"></img>

<img src="/logos/cityseer_logo_light_red.png" alt="Cityseer red logo." width="350"></img>
