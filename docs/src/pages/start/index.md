---
layout: '@src/layouts/PageLayout.astro'
---

# Getting Started

`cityseer` provides tools for street-network and land-use analysis at the pedestrian scale. It measures how central each street is in the walking network (centrality), how easily pedestrians can reach amenities such as shops or parks (accessibility), how varied those amenities are (mixed-use diversity), and how numerical attributes such as property prices vary across neighbourhoods (statistical aggregation). It also models pedestrian movement between origins and destinations (demand and origin-destination betweenness) and computes street-level visibility. Every measure is computed over the walking network, so distances follow real routes.

:::note
**Working with an LLM?** If you use an AI coding assistant, point it at [`llms.txt`](/llms.txt), a compact machine-readable index of this documentation, and at the [GitHub repository](https://github.com/benchmark-urbanism/cityseer-api), so its answers reflect the current API.
:::

`cityseer` integrates with [`NetworkX`](https://networkx.github.io/), [`GeoPandas`](https://geopandas.org/en/stable/), [`OSMnx`](https://osmnx.readthedocs.io/), and the broader Python geospatial ecosystem including [`shapely`](https://shapely.readthedocs.io), [`numpy`](http://www.numpy.org/), and [`momepy`](http://docs.momepy.org). The underlying algorithms are implemented in Rust, so the same analyses scale from a single neighbourhood to entire cities and large regions without giving up the strict pedestrian-scale distance thresholds. Code tests are run against Python versions `3.10` - `3.14`.

## Installation

```bash
pip install --upgrade cityseer
```

`cityseer` requires Python 3.10 or later. The underlying algorithms are implemented in Rust and distributed as pre-compiled wheels, so no Rust toolchain is needed. A projected coordinate reference system (CRS) is required for all analyses; coordinates must be in metres, not degrees. Use [epsg.io](https://epsg.io/) to find the appropriate EPSG code for your study area (e.g. `EPSG:32630` for London, `EPSG:32632` for central Europe, `EPSG:2154` for France).

:::note
For users who prefer a GUI workflow, the [QGIS plugin](/plugin) runs cityseer's centrality, accessibility, mixed-use, statistics, and demand analyses without writing Python code.
:::

## Quick start

The [Quickstart](/examples/recipes/quickstart) notebook provides a complete worked example. The following minimal example downloads a street network from OpenStreetMap, computes centrality, and plots the result:

```python
from shapely.geometry import box
from cityseer.network import CityNetwork

# Define a bounding box in WGS84 (lon, lat)
polygon = box(-0.13, 51.51, -0.12, 51.52)

# Build the network (projected to UTM zone 30N)
cn = CityNetwork.from_osm(polygon, to_crs_code=32630)

# Compute shortest-path centrality at 400m and 800m walking distance
cn.centrality_shortest(distances=[400, 800])

# Export as a GeoDataFrame with original street geometries
result_gdf = cn.to_geopandas()

# Visualise betweenness at 800m (in a Jupyter notebook; for scripts, call plt.show())
result_gdf.plot(column="cc_betweenness_800", cmap="inferno", linewidth=0.5)
```

![Centrality on a street network.](/images/graph_colour.png) _A worked result: centrality coloured across a street network, from low (blue) to high (red)._

Distance thresholds can also be specified as walking times using the `minutes` parameter:

```python
cn.centrality_shortest(minutes=[5, 10, 20])  # assumes default walking speed of 1.33 m/s
```

### Land-use accessibility

The same network answers land-use questions. Download features for the area and measure how reachable they are:

```python
from osmnx import features

# parks in the same area, projected to match the network
parks = features.features_from_polygon(polygon, tags={"leisure": "park"}).to_crs(32630)

cn.compute_accessibilities(
    data_gdf=parks,
    landuse_column_label="leisure",
    accessibility_keys=["park"],
    distances=[400, 800],
)
result_gdf = cn.to_geopandas()
print(result_gdf["cc_park_nearest_max_800"])  # distance to the nearest park within 800m
```

The same feature data also feeds mixed-use diversity and statistical aggregation, and pedestrian movement between origins and destinations is modelled with demand and origin-destination betweenness. See the [land-use guide](/guide/land-use) and the [flows guide](/guide/flows).

### Saving and loading

Networks can be saved to disk and restored later, preserving all computed metrics:

```python
cn.save("my_network")
# Creates: my_network.nodes.parquet, my_network.state.pkl

cn_restored = CityNetwork.load("my_network")
```

:::note
The lower-level API (`cityseer.tools`, `cityseer.metrics`) offers step-by-step control over graph cleaning, network construction, and metric computation. Most users should start with `CityNetwork`; the lower-level API is useful when integrating cityseer into an existing NetworkX pipeline or when fine-grained control over processing steps is needed. See the [`tools`](/tools/io) and [`metrics`](/metrics/networks) module references for details.
:::

## Learning path

New to Python or computational notebooks? Work through the Python 101 course first:

1. [Notebooks](/start/1-notebooks): What computational notebooks are and how to use marimo.
2. [Python Basics](/start/2-basics): Variables, data types, collections, control flow, and functions.
3. [Spatial Data](/start/3-spatial): Points, lines, and polygons with the `shapely` package.
4. [GeoPandas](/start/4-geopandas): Handling geospatial datasets with `geopandas`.
5. [Urban Analytics](/start/5-urban): Downloading OSM data with `osmnx` and analysing urban morphology with `momepy`.
6. [Data Science](/start/6-data-science): Dimensionality reduction, clustering, and prediction with `seaborn` and `scikit-learn`.

Each lesson page renders the executed notebook and offers the raw `.py` file for download; open a downloaded lesson interactively with `uv run marimo edit <file>`. Alternatively, create a notebook in whichever environment you prefer and copy the cells across as you work through a lesson: the code is plain Python.

## Where next

- The [Quickstart notebook](/examples/recipes/quickstart) covers a complete first analysis.
- The [Guide](/guide/fundamentals) explains how `cityseer` frames analysis: fundamentals, [networks](/guide/networks), [centrality](/guide/centrality), [land-use](/guide/land-use), and [flows](/guide/flows).
- The [examples](/examples) section holds worked recipes for every module, built on bundled real-world data.
- The [API reference](/api/network) documents every function and parameter.

## Support and attribution

Report bugs to the [issues tracker](https://github.com/benchmark-urbanism/cityseer-api/issues); for questions and workflow help, start a [discussion](https://github.com/benchmark-urbanism/cityseer-api/discussions).

Please cite the tools you use: for `cityseer`, cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) in research outputs, and as a minimum of good conscience, openly link to this documentation site from work that builds on the package. `cityseer` is licensed AGPLv3; [get in touch](mailto:info@benchmarkurbanism.com) for technical support or sponsored development.
