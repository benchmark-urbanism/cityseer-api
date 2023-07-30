---
layout: '@src/layouts/PageLayout.astro'
---

# Getting Started

`cityseer` is a collection of computational tools for fine-grained street-network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods.

`cityseer` is underpinned by network-based methods that have been developed from the ground-up for micro-morphological urban analysis at the pedestrian scale, with the intention of providing contextually specific metrics for any given streetfront location. Importantly, `cityseer` computes metrics directly over the street network and offers distance-weighted variants. The combination of these strategies makes `cityseer` more contextually sensitive than methods otherwise based on crow-flies aggregation methods that do not take the network structure and its affect on pedestrian walking distances into account.

The use of `python` facilitates interaction with popular computational tools for network manipulation (e.g. [`networkX`](https://networkx.github.io/)), geospatial data processing (e.g. [`shapely`](https://shapely.readthedocs.io), etc.), Open Street Map workflows with [`OSMnx`](https://osmnx.readthedocs.io/), and interaction with the [`numpy`](http://www.numpy.org/), [`geopandas`](https://geopandas.org/en/stable/) (and [`momepy`](http://docs.momepy.org)) stack of packages. The underlying algorithms are parallelised and implemented in `rust` so that the methods can be scaled to large networks. In-out convenience methods are provided for interfacing with `networkX` and graph cleaning tools aid the incorporation of messier network representations such as those derived from [Open Street Map](https://www.openstreetmap.org).

The github repository is available at [github.com/benchmark-urbanism/cityseer-api](https://github.com/benchmark-urbanism/cityseer-api). Please cite the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827) when using this package. Associated papers introducing the package and demonstrating the forms of analysis that can be done with it are [available at `arXiv`](https://arxiv.org/a/simons_g_1.html).

## Installation

`cityseer` is a `python` package that can be installed with `pip`:

```bash
pip install cityseer
```

Code tests are run against `python 3.10`, though the code base will generally be compatible with Python 3.8+.

## Notebooks

The getting started guide on this page, and a growing collection of other examples, is available as an Jupyter Notebooks which can be accessed via the [`Examples`](/examples) page. The examples include workflows showing how to run network centralities and land-use accessibility analysis for some real-world situations.

## Quickstart

`cityseer` revolves around networks (graphs). If you're comfortable with `numpy` and abstract data handling, then the underlying data structures can be created and manipulated directly. However, it is generally more convenient to sketch the graph using [`NetworkX`](https://networkx.github.io/) and to let `cityseer` take care of initialising and converting the graph for you.

```python
# any networkX MultiGraph with 'x' and 'y' node attributes will do
# here we'll use the cityseer mock module to generate an example networkX graph
import networkx as nx
from cityseer.tools import mock, graphs, plot

G = mock.mock_graph()
print(G)
# let's plot the network
plot.plot_nx(G, labels=True, node_size=80, dpi=200, figsize=(4, 4))
```

![An example graph](/images/graph.png)
_An example graph._

## Graph Preparation

The [`tools.graphs`](/tools/graphs) module contains a collection of convenience functions for the preparation and conversion of `networkX` `MultiGraphs`, i.e. undirected graphs allowing for parallel edges. The [`tools.graphs`](/tools/graphs) module is designed to work with raw `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries that have been assigned to the graph's edge (link) `geom` attributes. The benefit to this approach is that the geometry of the network is decoupled from the topology: the topology is consequently free from distortions which would otherwise confound centrality and other metrics.

There are generally two scenarios when creating a street network graph:

1. In the ideal case, if you have access to a high-quality street network dataset -- which keeps the topology of the network separate from the geometry of the streets -- then you would construct the network based on the topology while assigning the roadway geometries to the respective edges spanning the nodes. [OS Open Roads](https://www.ordnancesurvey.co.uk/business-and-government/products/os-open-roads.html) is a good example of this type of dataset. Assigning the geometries to an edge involves A) casting the geometry to a [`shapely`](https://shapely.readthedocs.io) `LineString`, and B) assigning this geometry to the respective edge by adding the `LineString` geometry as a `geom` attribute. e.g. `G.add-edge(start_node, end_node, geom=a_linestring_geom)`.

2. In reality, most data-sources are not this refined and will represent roadway geometries by adding additional nodes to the network. For a variety of reasons, this is not ideal and you may want to follow the [`Graph Cleaning`](/guide#graph-cleaning) guide; in these cases, the [`graphs.nx_simple_geoms`](/tools/graphs#nx-simple-geoms) method can be used to generate the street geometries, after which several methods can be applied to clean and prepare the graph. For example, [`nx-wgs_to_utm`](/tools/graphs#nx-wgs-to-utm) aids coordinate conversions; [`nx_remove_dangling_nodes`](/tools/graphs#nx-remove-dangling-nodes) removes remove roadway stubs, [`nx_remove_filler_nodes`](/tools/graphs#nx-remove-filler-nodes) strips-out filler nodes, and [`nx-consolidate-nodes`](/tools/graphs#nx-consolidate-nodes) assists in cleaning-up the network.

## Example

Here, we'll walk through a high-level overview showing how to use `cityseer`. You can provide your own shapely geometries if available; else, you can auto-infer simple geometries from the start to end node of each network edge, which works well for graphs where nodes have been used to inscribe roadway geometries.

```python
# use nx_simple_geoms to infer geoms for your edges
G = graphs.nx_simple_geoms(G)
plot.plot_nx(G, labels=True, node_size=80, plot_geoms=True, dpi=200, figsize=(4, 4))
```

![An example graph](/images/graph_example.png)
_A graph with inferred geometries. In this case the geometries are all exactly straight._

We have now inferred geometries for each edge, meaning that each edge now has an associated `LineString` geometry. Any further manipulation of the graph using the `cityseer.graph` module will retain and further manipulate these geometries in-place.

Once the geoms are readied, we can use tools such as [`nx_decompose`](/tools/graphs#nx-decompose) for generating granular graph representations and [`nx_to_dual`](/tools/graphs#nx-to-dual) for casting a primal graph representation to its dual.

```python
# this will (optionally) decompose the graph
G_decomp = graphs.nx_decompose(G, 50)
plot.plot_nx(G_decomp, plot_geoms=True, labels=False, dpi=200, figsize=(4, 4))
```

![An example decomposed graph](/images/graph_decomposed.png)
_A decomposed graph._

```python
# this will (optionally) cast to a dual network
G_dual = graphs.nx_to_dual(G)
# here we are plotting the newly decomposed graph (blue) against the original graph (red)
plot.plot_nx_primal_or_dual(G, G_dual, plot_geoms=False, dpi=200, figsize=(4, 4))
```

![An example dual graph](/images/graph_dual.png)
_A dual graph (blue) plotted against the primal source graph (red). In this case, the true geometry has not been plotted so that the dual graph is easily delineated from the primal graph._

## Metrics

After graph preparation and cleaning has been completed, the `networkX` graph can be transformed into data structures for efficiently computing centralities, land-use measures, or statistical aggregations.

Use [network_structure_from_nx](/tools/graphs#network-structure-from-nx) to convert a `networkX` graph into GeoPandas [`GeoDataFrames`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html) and a [`rustalgos.NetworkStructure`](/rustalgos#networkstructure), the latter of which is used by `cityseer` for efficiently computing the measures with the underlying `rust` algorithms.

### Network Centralities

The [`networks.node_centrality_shortest`](/metrics/networks#node-centrality-shortest), [`networks.node_centrality_simplest`](/metrics/networks#node-centrality-simplest), and [`networks.segment_centrality`](/metrics/networks#segment-centrality) methods wrap underlying `rust` functions that compute the centrality methods. All selected measures and distance thresholds are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar workflows. The results of the computations will be written to the `GeoDataFrame`.

```python
from cityseer.metrics import networks

# create a Network layer from the networkX graph
# use a CRS EPSG code matching the projected coordinate reference system for your data
nodes_gdf, edges_gdf, network_structure = graphs.network_structure_from_nx(G_decomp, crs=3395)
# the underlying method allows the computation of various centralities simultaneously, e.g.
nodes_gdf = networks.segment_centrality(
    network_structure=network_structure,  # the network structure for which to compute the measures
    nodes_gdf=nodes_gdf,  # the nodes GeoDataFrame, to which the results will be written
    distances=[200, 400, 800, 1600],  # the distance thresholds for which to compute centralities
)
nodes_gdf.head()  # the results are now in the GeoDataFrame
```

```python
# plot centrality
from matplotlib import colors

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list("cityseer", ["#64c1ff", "#d32f2f"])
# normalise the values
segment_harmonic_vals = nodes_gdf["cc_metric_segment_harmonic_800"]
segment_harmonic_vals = colors.Normalize()(segment_harmonic_vals)
# cast against the colour map
segment_harmonic_cols = cmap(segment_harmonic_vals)
# plot segment_harmonic
# cityseer's plot methods are used here and in tests for convenience
# that said, rather use plotting methods directly from networkX or GeoPandas where possible
plot.plot_nx(G_decomp, labels=False, node_colour=segment_harmonic_cols, dpi=200, figsize=(4, 4))
```

![Example centrality plot](/images/intro_segment_harmonic.png)
_800m segmentised harmonic centrality._

### Land-use and statistical measures

Landuse and statistical measures require a GeoPandas [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html) consisting of `Point` geometries. Columns representing categorical landuse information ("pub", "shop", "school") can be passed to landuse methods, whereas columns representing numerical information can be used for statistical methods.

When computing these measures, `cityseer` will assign each data point to the two closest network nodes — one in either direction — based on the closest adjacent street edge. This enables `cityseer` to use dynamic spatial aggregation methods that more accurately describe distances from the perspective of pedestrians travelling over the network, and relative to the direction of approach.

[`layers.compute_landuses`](/metrics/layers#compute-landuses) and [`layers.compute_mixed_uses`](/metrics/layers#compute-mixed-uses) methods are used for the calculation of land-use accessibility and mixed-use measures whereas [`layers.compute_stats`](/metrics/layers#compute-stats) can be used for statistical aggregations. As with the centrality methods, the measures are computed over the network and are computed simultaneously for all measures and distances.

```python
from cityseer.metrics import layers

# a mock data dictionary representing categorical landuse data
# here randomly generated letters represent fictitious landuse categories
data_gdf = mock.mock_landuse_categorical_data(G_decomp, random_seed=25)
data_gdf.head()
```

```python
# example easy-wrapper method for computing mixed-uses
# this is a distance weighted form of hill diversity
nodes_gdf, data_gdf = layers.compute_mixed_uses(
    data_gdf,  # the source data
    landuse_column_label="categorical_landuses",  # column in the dataframe which contains the landuse labels
    nodes_gdf=nodes_gdf,  # nodes GeoDataFrame - the results are written here
    network_structure=network_structure,  # measures will be computed relative to pedestrian distances over the network
    distances=[200, 400, 800, 1600],  # distance thresholds for which you want to compute the measures
)
print(nodes_gdf.columns)  # the GeoDataFrame will contain the results of the calculations
print(nodes_gdf["cc_metric_q0_800_hill"])  # which can be retrieved as needed
```

```python
# for curiosity's sake - plot the assignments to see which edges the data points were assigned to
plot.plot_assignment(network_structure, G_decomp, data_gdf, dpi=200, figsize=(4, 4))
```

![Data assigned to network](/images/assignment.png)
_Data points assigned to a Network Layer._

![Data assigned to a decomposed network](/images/assignment_decomposed.png)
_Data assignment becomes more precise on a decomposed Network Layer._

```python
# plot distance-weighted "hill" numbers mixed uses
mixed_uses_vals = nodes_gdf["cc_metric_q0_800_hill_weighted"]
mixed_uses_vals = colors.Normalize()(mixed_uses_vals)
mixed_uses_cols = cmap(mixed_uses_vals)
plot.plot_assignment(
    network_structure,
    G_decomp,
    data_gdf,
    node_colour=mixed_uses_cols,
    data_labels=data_gdf["categorical_landuses"].values,
    dpi=200,
    figsize=(4, 4),
)
```

![Example mixed-use plot](/images/intro_mixed_uses.png)
_400m distance-weighted mixed-uses._

```python
# compute landuse accessibilities for land-use types a, b, c
nodes_gdf, data_gdf = layers.compute_accessibilities(
    data_gdf,  # the source data
    landuse_column_label="categorical_landuses",  # column in the dataframe which contains the landuse labels
    accessibility_keys=["a", "b", "c"],  # the landuse categories for which to compute accessibilities
    nodes_gdf=nodes_gdf,  # nodes GeoDataFrame - the results are written here
    network_structure=network_structure,  # measures will be computed relative to pedestrian distances over the network
    distances=[200, 400, 800, 1600],  # distance thresholds for which you want to compute the measures
)
# accessibilities are computed in both weighted and unweighted forms, e.g. for "a" and "b" landuse codes
print(nodes_gdf[["cc_metric_a_800_weighted", "cc_metric_b_1600_non_weighted"]])  # and can be retrieved as needed
```

Aggregations can likewise be computed for numerical data. Let's generate some mock numerical data:

```python
numerical_data_gdf = mock.mock_numerical_data(G_decomp, num_arrs=3)
numerical_data_gdf.head()
# compute stats for column mock_numerical_1
nodes_gdf, numerical_data_gdf = layers.compute_stats(
    numerical_data_gdf,  # the source data
    stats_column_label="mock_numerical_1",  # numerical column to compute stats for
    nodes_gdf=nodes_gdf,  # nodes GeoDataFrame - the results are written here
    network_structure=network_structure,  # measures will be computed relative to pedestrian distances over the network
    distances=[800, 1600],  # distance thresholds for which you want to compute the measures
)
# statistical aggregations are calculated for each requested column, and in the following forms:
# max, min, sum, sum_weighted, mean, mean_weighted, variance, variance_weighted
print(nodes_gdf["cc_metric_max_800"])
print(nodes_gdf["cc_metric_mean_wt_800"])
```

The landuse metrics and statistical aggregations are computed over the street network relative to the network, with results written to each node. The mixed-use, accessibility, and statistical aggregations can therefore be compared directly to centrality computations from the same locations, and can be correlated or otherwise compared.

Data derived from metrics can be converted back into a `NetworkX` graph using the [nx_from_geopandas](/metrics/networks#nx-from-network-geopandas) method.

```python
nx_multigraph_round_trip = graphs.nx_from_geopandas(
    nodes_gdf,
    edges_gdf,
)
nx_multigraph_round_trip.nodes["0"]
```
