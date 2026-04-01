---
layout: '@src/layouts/PageLayout.astro'
---

`cityseer` provides tools for street-network and land-use analysis at the pedestrian scale. It measures how central each street is in the walking network (centrality), how easily pedestrians can reach amenities like shops or parks (accessibility), and how numerical attributes such as property prices are distributed across neighbourhoods (statistical aggregation) — all computed along actual walking routes rather than straight-line distances.

`cityseer` integrates with [`NetworkX`](https://networkx.github.io/), [`GeoPandas`](https://geopandas.org/en/stable/), [`OSMnx`](https://osmnx.readthedocs.io/), and the broader Python geospatial ecosystem including [`shapely`](https://shapely.readthedocs.io), [`numpy`](http://www.numpy.org/), and [`momepy`](http://docs.momepy.org). The underlying algorithms are implemented in Rust for performance and scale to large networks. Graph cleaning tools and convenience methods for [Open Street Map](https://www.openstreetmap.org) data are included.

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api).

Code tests are run against Python versions `3.10` - `3.13`.

## Getting Started

The [Quickstart](https://benchmark-urbanism.github.io/cityseer-examples/recipes/quickstart.html) notebook is the fastest way to see `cityseer` in action. For a detailed explanation of how `cityseer` represents networks, computes metrics, and handles distance decay, see the [Guide](/guide). For practical worked examples covering network preparation, centrality, accessibility, statistics, visibility, and continuity analysis, see the [Cityseer Examples](https://benchmark-urbanism.github.io/cityseer-examples/) site.

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

Please report bugs to the [github issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues). For general questions, workflow help, or research feedback, start a [discussion on Github](https://github.com/benchmark-urbanism/cityseer-api/discussions).

## Attribution

Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package for producing research.

`cityseer` is licensed as AGPLv3. Please [get in touch](mailto:info@benchmarkurbanism.com) if you need technical support developing related workflows, or if you wish to sponsor the development of additional or bespoke functionality.

If using the package to produce visual plots and outputs, please display the cityseer logo and a link to the documentation website.

<img src="/logos/cityseer_logo_white.png" alt="Cityseer white logo." width="350"></img>

<img src="/logos/cityseer_logo_light_red.png" alt="Cityseer red logo." width="350"></img>
