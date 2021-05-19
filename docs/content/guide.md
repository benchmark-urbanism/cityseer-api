# Guide

## Motivation

The overarching motivation in the development of `cityseer` has been the quantification of urban characteristics in a manner that is as sensitive as possible to local particularities and variations. `cityseer` is designed for localised urban analysis at the pedestrian scale, meaning that observations are tailored towards pedestrian walking tolerances, commonly in the range of 400m to 800m, and infrequently exceeding 1,600m. In the case of certain land-use and mixed-use measures, it can be beneficial to work with weighted measures at distance thresholds as small as 100m or even 50m so that metrics are sufficiently precise to be pertinent to the day-to-day decisions made by architects and urban designers. These use-cases require information anchored to particular locations and measures must be adequately sensitive to reflect varying outcomes in response to different planned scenarios.

`cityseer` is developed from the ground-up to address a particular range of issues that are prevalent in pedestrian-scale urban analysis:

- It uses localised forms of network analysis (as opposed to global forms of analysis) based on network methods applied over the graph through means of a 'moving-window' methodology: the graph is isolated at a specified distance threshold for the currently selected node, and the process subsequently repeats for every other node in the network. These thresholds are conventionally based on either crow-flies euclidean distances or true network distances (@Cooper2015): `cityseer` takes the position that true network distances are more representative when working at smaller pedestrian distance thresholds, particularly when applied to land-use accessibilities and mixed-use calculations;
- It is common to use either shortest-distance or simplest-path (shortest angular 'distance') impedance heuristics. When using simplest-path heuristics, it is necessary to modify the underlying shortest-path algorithms to prevent side-stepping of sharp angular turns (@Turner2007), otherwise two smaller side-steps can be combined to 'short-cut' sharp corners. It is also common for methods to be applied to either primal graph representations (generally used with shortest-path methods such as those applied by _multiple centrality assessment_ (@Porta2006) analysis) or dual graph representations (typically used with simplest-path methods in the tradition of _space syntax_(@Hillier1984));
- There are a range of possible centrality and mixed-use methods, many of which can be weighted by distances or street lengths. These methods and their implications are explored in detail in the [localised land-use diversity](https://blog.benchmarkurbanism.com/papers/mixed-uses/) and [localised land-use diversity methods](https://blog.benchmarkurbanism.com/papers/network-centrality/) papers. Some conventional methods, even if widely used, have not necessarily proved suitable for localised urban analysis;
- Centrality methods are susceptible to topological distortions arising from 'messy' graph representations as well as due to the conflation of topological and geometrical properties of street networks. `cityseer` addresses these through the inclusion of graph cleaning functions; procedures for splitting geometrical properties from topological representations; and the inclusion of segmentised centrality measures, which are less susceptible to distortions introduced by varying intensities of nodes;
- Hyperlocal analysis requires approaches facilitating the evaluation of respective measures at finely-spaced intervals along streetfronts. Further, granular evaluation of land-use accessibilities and mixed-uses requires that landuses be assigned to the street network in a contextually precise manner. These are addressed in `cityseer` through application of network decomposition combined with algorithms incorporating bidirectional assignment of data points to network nodes based on the closest adjacent street edge.

The broader emphasis on localised methods and the manner in which `cityseer` addresses these is broached in the accompanying paper (link forthcoming). `cityseer` includes a variety of convenience methods for the general preparation of networks and for their conversion into (and out of) the lower-level data structures used by the underlying algorithms. These graph utility methods are designed to work with `NetworkX` to facilitate ease-of-use. A complement of code tests has been developed for maintaining the integrity of the code-base through general package maintenance and upgrade cycles. Shortest-path algorithms, harmonic closeness, and betweenness algorithms are tested against `NetworkX`. Mock data and test plots have been used to visually confirm the intended behaviour for divergent simplest and shortest-path heuristics, and for testing data assignment to network nodes given a variety of scenarios.

## Graph Cleaning

:::tip Comment

A notebook of this guide can be found at [google colaboratory](https://colab.research.google.com/github/cityseer/cityseer/blob/master/demo_notebooks/graph_cleaning.ipynb).

:::

Good sources of street network data, such as the Ordnance Survey's [OS Open Roads](https://www.ordnancesurvey.co.uk/business-and-government/products/os-open-roads.html), typically have two distinguishing characteristics:

- The network has been simplified to its essential structure: i.e. unnecessarily complex representations of intersections; on-ramps; split roadways; etc. have been reduced to a simpler representation concurring more readily with the core topological structure of street networks. This is in contrast to network representations focusing on completeness (e.g. for route way-finding, see [OS ITN Layer](https://www.ordnancesurvey.co.uk/business-and-government/help-and-support/products/itn-layer.html)): these introduce unnecessary complexity serving to hinder rather than help shortest-path algorithms in the sense used by pedestrian centrality measures.
- The topology of the network is kept distinct from the geometry of the streets. Often-times, as can be seen with [Open Street Map](https://www.openstreetmap.org), additional nodes are added to streets for the purpose of representing geometric twists and turns along a roadway. These additional nodes cause topological distortions that impact network centrality measures.

When a high-quality source is available, it may be best not to attempt additional clean-up unless there is a particular reason to do so. On the other-hand, many indispensable sources of network information, particularly Open Street Map data, can be messy for the purposes of network analysis. This section describes how such sources can be cleaned and prepared for subsequent analysis. `cityseer` uses methods attempting to clean the topology of the graph in such a manner as to reduce topological artefacts that might otherwise confound centrality measures e.g. by attempting to remove dual carriageways and by deriving topologies and geometrical forms of edge consolidation that are as 'tidy' as possible so as not to complicate simplest-path (angular) methods.

### Downloading data

This example will make use of OSM data downloaded from the [OSM API](https://wiki.openstreetmap.org/wiki/API). To keep things interesting, let's pick London Soho, which will be buffered and cleaned for a 1,250m radius.

```python
from shapely import geometry
import utm

from cityseer.tools import graphs, plot, mock

# Let's download data within a 1,250m buffer around London Soho:
lng, lat = -0.13396079424572427, 51.51371088849723
G_utm = mock.make_buffered_osm_graph(lng, lat, 1250)

# As an alternative, you can use OSMnx to download data. Set simplify to False:
# e.g.: OSMnx_multi_di_graph = ox.graph_from_point((lat, lng), dist=1250, simplify=False)
# Then convert to a cityseer compatible MultiGraph:
# e.g.: G_utm = graphs.nX_from_OSMnx(OSMnx_multi_di_graph, tolerance=10)

# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(1000)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds


# reusable plot function
def simple_plot(_G, plot_geoms=True):
    # plot using the selected extents
    plot.plot_nX(_G,
                 labels=False,
                 plot_geoms=plot_geoms,
                 node_size=15,
                 edge_width=2,
                 x_lim=(min_x, max_x),
                 y_lim=(min_y, max_y),
                 figsize=(20, 20),
                 dpi=200)


simple_plot(G_utm, plot_geoms=False)
```

![The raw graph from OSM](../src/assets/plots/images/graph_cleaning_1.png)
_The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

### Deducing the network topology

Once OSM data has been converted to a `NetworkX` `MultiGraph`, the `tools.graphs` module can be used to clean the network.

> The convenience method used for this demonstration has already converted the graph from a geographic WGS to projected UTM coordinate system; however, if working with a graph which is otherwise in a WGS coordinate system then it must be converted to a projected coordinate system prior to further processing. This can be done with [`graphs.nX_wgs_to_utm`](/tools/graphs/#nx_wgs_to_utm).

Now that raw OSM data has been loaded into a NetworkX graph, the `cityseer.tools.graph` methods can be used to further clean and prepare the network prior to analysis.

At this stage, the raw OSM graph is going to look a bit messy. Note how that nodes have been used to represent the roadway geometry. These nodes need to be removed and will be abstracted into `shapely` `LineString` geometries assigned to the respective street edges. So doing, the geometric representation will be kept distinct from the network topology.

```py
# the raw osm nodes denote the road geometries by the placement of nodes
# the first step generates explicit LineStrings geometries for each street edge
G = graphs.nX_simple_geoms(G_utm)
# We'll now strip the "filler-nodes" from the graph
# the associated geometries will be welded into continuous LineStrings
# the new LineStrings will be assigned to the newly consolidated topological links
G = graphs.nX_remove_filler_nodes(G)
# and remove dangling nodes: short dead-end stubs
# these are often found at entrances to buildings or parking lots
# The removed_disconnected flag will removed isolated network components
# i.e. disconnected portions of network that are not joined to the main street network
G = graphs.nX_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
# removing danglers can cause newly orphaned filler nodes, which we'll remove for good measure
G = graphs.nX_remove_filler_nodes(G)
simple_plot(G)
```

![Initial graph cleaning](../src/assets/plots/images/graph_cleaning_2.png)
_After removal of filler nodes, dangling nodes, and disconnected components._

### Refining the network

Things are already looked much better, but we still have areas with large concentrations of nodes at complex intersections and many parallel roadways, which will confound centrality methods. We'll now try to remove as much of this as possible. These steps involve the consolidation of nodes to clean-up extraneous nodes, which may otherwise exaggerate the intensity or complexity of the network in certain situations.

In this case, we're trying to get rid of parallel road segments so we'll do this in three steps, though it should be noted that, depending on your use-case, Step 1 may already be sufficient:

Step 1: An initial pass to cleanup complex intersections will be performed with the [`graphs.nX_consolidate_nodes`](/tools/graphs/#nx_consolidate_nodes) function. The arguments passed to the parameters allow for a number of different strategies, such as whether to 'crawl'; minimum and maximum numbers of nodes to consider for consolidation; and to set the policies according to which nodes and edges are consolidated. These are explained more fully in the documentation. In this case, we're accepting the defaults except for explicitly setting the buffer distance and bumping the minimum size of node groups to be considered for consolidation from 2 to 3.

```py
G1 = graphs.nX_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
simple_plot(G1)
```

![First step of node consolidation](../src/assets/plots/images/graph_cleaning_3.png)
_After an initial pass of node consolidation._

Complex intersections have now been simplified, for example, the intersection of Oxford and Regent has gone from 17 nodes to a single node.

In Step 2, we'll use [`graphs.nX_split_opposing_geoms`](/tools/graphs/#nx_split_opposing_geoms) to intentionally split edges in near proximity to nodes located on an adjacent roadway. This is going to help with the final pass of consolidation in Step 3.

```py
G2 = graphs.nX_split_opposing_geoms(G1, buffer_dist=15)
simple_plot(G2)
```

![Splitting opposing geoms](../src/assets/plots/images/graph_cleaning_4.png)
_After "splitting opposing geoms" on longer parallel segments._

In the final step, we can now rerun the consolidation to clean up any remaining clusters of nodes. In this case, we're setting the `crawl` parameter to `False`, setting `min_node_degree` down to 2, and prioritising nodes of `degree=4` for determination of the newly consolidated centroids:

```py
G3 = graphs.nX_consolidate_nodes(G2,
                                 buffer_dist=15,
                                 crawl=False,
                                 min_node_degree=2,
                                 cent_min_degree=4)
simple_plot(G3)
```

![Final step of node consolidation](../src/assets/plots/images/graph_cleaning_5.png)
_After the final step of node consolidation._

:::tip Manual Cleaning

When using shortest-path methods, automated graph simplification and consolidation can arguably eliminate the need for manual checks; however, it is worth plotting the graph and performing a quick look-through to be sure there aren't any unexpectedly odd situations.

When using simplest-path (angular) centralities, manual checks become more important because automated simplification and consolidation can result in small twists and turns where nodes and edges have been merged. `cityseer` uses particular methods that attempt to keep these issues to a minimum, though there may still be some situations necessitating manual checks. From this perspective, it may be preferable to use a cleaner source of network topology (e.g. OS Open Roads) if working with simplest-path centralities; else, if only OSM data is available, to instead consider the use of shortest-path methods if the graph has too many unresolvable situations to clean-up manually.

:::

The above recipe should be enough to get you started, but there are innumerable other strategies that may also work for any variety of scenarios.

## Relation to other packages

### _OSMnx_

[`OSMnx`](https://osmnx.readthedocs.io/) connects the [Open Street Map](https://www.openstreetmap.org) (`OSM`) API to [`networkX`](https://networkx.github.io/) graphs and the wider ecosystem of python-based geospatial tools through which it provides access to a range of graph-analysis, conversion, and visualisation methods. In the first-instance, `cityseer` is not about `networkX`, nor is it about `OSM`. Earlier versions of `cityseer` emerged around graphs in a more abstract sense --- raw [`numpy`](http://www.numpy.org/) arrays and [`numba`](https://numba.pydata.org/) data structures that scale relatively well to larger graphs --- and the associated data structures were manually created from data stored on `postGIS` (`postgres`) databases. To ease the repeated use of these methods, and to lower the barrier to entry, these workflows were gradually abstracted to `networkX` based approaches to make it simpler to create the graphs and to apply methods such as "decomposition"; casting a graph to its "dual"; and subsequent conversion into `cityseer` data structures with the correct format of attributes for use by downstream `cityseer` algorithms. Nevertheless, it bears emphasis that `cityseer` uses `networkX` primarily as an in-out and graph preparation tool, not as its primary representation; similarly, it is not tailored for ingestion of `OSM` data-sources but is rather intended to be data-source agnostic.

Other differences stem accordingly: `cityseer` does not use the `networkX` package for graph analysis, but implements its own algorithms that have developed around experimental exploration of niche methods intended only for pedestrian-scale analysis on street-networks as opposed to the more general-purpose and much wider variety of algorithms available in `networkX`. The algorithms employed in `cityseer` are accordingly intended only for localised (windowed) graph analysis: they use explicit distance thresholds; employ unique variants of centrality measures; handle cases such as simplest-path heuristics and segmentised forms of analysis; and extend these algorithms to handle the derivation of land-use accessibilities, mixed-uses, and statistical aggregations using similarly windowed and network-distance-weighted methods.

Taking the following into account, it is possible to use `OSMnx` and `cityseer` together, and an example is provided in the code snippet which follows below:

- Whereas some basic `OSM` ingestion and conversion functions are included in the [`tools.mock`](/tools/mock) module, these are primarily intended for internal code development. If used directly, these assume that the end-user will have some direct knowledge of how these APIs work and how the recipes and conversion functions can be manipulated for specific situations. i.e. unless you want to roll-your-own OSM queries, it is best to stick with `OSMnx` for purposes of ingesting `OSM` networks.
- `OSMnx` prepared graphs can be converted to `cityseer` compatible graphs by using the [`tools.graphs.nX_from_OSMnx`](/tools/graphs/#nx_from_osmnx) method. In doing so, keep the following in mind:
  - `OSMnx` uses `networkX` `multiDiGraph` graph structures which use directional edges. As such, it can be used for understanding vehicular routing, i.e. where one-way routes can have a major impact on the available shortest-routes. `cityseer` --- and intentionally so --- is only concerned with pedestrian networks and therefore makes use of `networkX` `MultiGraphs` on the premise that pedestrian networks are not ordinarily directional. When using the [`tools.graphs.nX_from_OSMnx`](/tools/graphs/#nx_from_osmnx) method, be cognisant that all directional information will consequently be discarded.
  - `cityseer` graph simplification and consolidation workflows will give subtly different results to those employed in `OSMnx`. If you're using `OSMnx` to ingest networks from `OSM` but wish to simplify and consolidate the network as part of a `cityseer` workflow, then set the `OSMnx` `simplify` argument to `False` so that the network is not automatically simplified.
  - `cityseer` uses internal graph validation workflows to check that the geometries associated with an edge remain connected to the coordinates of the nodes on either side. If performing any graph manipulation outside of `cityseer` then the conversion function may complain of disconnected geometries. If so, you may need to relax the tolerance parameter which is used for error checking upon conversion to a `cityseer` `MultiGraph`. Geometries that are disconnected from their end-nodes (within the tolerance parameter) will be "snapped" to meet their endpoints as part of the conversion process.
- For graph cleaning and simplificaton: `cityseer` is oriented less towards automation and ease-of-use and more towards explicit and intentional use of potentially varied steps of processing. This involves a tradeoff, whereas some recipes are provided as a starting point (see [`Graph Cleaning`](/guide/#graph-cleaning)), you may find yourself needing to do more up-front experimentation and fine-tuning, but with the benefit of a degree of flexibility in how these methods are applied for a given network topology: e.g. steps can be included or omitted, used in different sequences, or repeated. Some of these methods, particularly [`tools.graphs.nX_consolidate_nodes`](/tools/graphs/#nx_consolidate_nodes), may have severable tunable parameters which can have markedly different outcomes. This philosophy is by design, and if you want a simplified method that you can easily repeat, then you'll need to wrap your own sequence of steps in a simplified utility function.

```py
from cityseer import tools
import osmnx as ox
from shapely import geometry
import utm

# centre-point
lng, lat = -0.13396079424572427, 51.51371088849723

# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
buff = geometry.Point(easting, northing).buffer(1000)
min_x, min_y, max_x, max_y = buff.bounds


# reusable plot function
def simple_plot(_G):
    # plot using the selected extents
    tools.plot.plot_nX(_G,
                       labels=False,
                       plot_geoms=True,
                       node_size=15,
                       edge_width=2,
                       x_lim=(min_x, max_x),
                       y_lim=(min_y, max_y),
                       figsize=(20, 20),
                       dpi=200)


# Let's use OSMnx to fetch an OSM graph
# We'll use the same raw network for both workflows (hence simplify=False)
multi_di_graph_raw = ox.graph_from_point((lat, lng),
                                         dist=1250,
                                         simplify=False)

# Workflow 1: Using OSMnx to prepare the graph
# ============================================
# explicit simplification and consolidation via OSMnx
multi_di_graph_utm = ox.project_graph(multi_di_graph_raw)
multi_di_graph_simpl = ox.simplify_graph(multi_di_graph_utm)
multi_di_graph_cons = ox.consolidate_intersections(multi_di_graph_simpl,
                                                   tolerance=10,
                                                   dead_ends=True)
# let's use the same plotting function for both scenarios to aid visual comparisons
multi_graph_cons = tools.graphs.nX_from_OSMnx(multi_di_graph_cons, tolerance=50)
simple_plot(multi_graph_cons)

# WORKFLOW 2: Using cityseer to prepare the graph
# ===============================================
# let's convert the OSMnx graph to a cityseer compatible `multiGraph`
G_raw = tools.graphs.nX_from_OSMnx(multi_di_graph_raw)
# convert to UTM
G = tools.graphs.nX_wgs_to_utm(G_raw)
# infer geoms
G = tools.graphs.nX_simple_geoms(G)
# remove degree=2 nodes
G = tools.graphs.nX_remove_filler_nodes(G)
# remove dangling nodes
G = tools.graphs.nX_remove_dangling_nodes(G, despine=10)
# repeat degree=2 removal to remove orphaned nodes due to despining
G = tools.graphs.nX_remove_filler_nodes(G)
# let's consolidate the nodes
G1 = tools.graphs.nX_consolidate_nodes(G, buffer_dist=10, min_node_group=3)
# and we'll try to remove as many parallel roadways as possible
G2 = tools.graphs.nX_split_opposing_geoms(G1, buffer_dist=15)
G3 = tools.graphs.nX_consolidate_nodes(G2,
                                       buffer_dist=15,
                                       crawl=False,
                                       min_node_degree=2,
                                       cent_min_degree=4)
simple_plot(G3)
```

![Example OSMnx simplification and consolidation](../src/assets/plots/images/osmnx_simplification.png)
_An example `OSMnx` simplification and consolidation workflow._

![Example OSMnx to cityseer workflow](../src/assets/plots/images/osmnx_cityseer_simplification.png)
_An example `OSMnx` to `cityseer` conversion followed by simplification and consolidation workflow in `cityseer`._

### Optimised packages

Computational methods for network-based centrality and land-use analysis make extensive use of shortest-path algorithms: these present substantial computational complexity due to nested-loops. Centrality methods implemented in pure `python`, such as those contained in [`NetworkX`](https://networkx.github.io/), can be particularly slow and may hinder timely application to large urban street networks (though, for the record, `NetworkX` is an exquisitely designed package). Speed improvements are offered by running intensive algorithms against packages such as [`Graph-Tool`](https://graph-tool.skewed.de) or [`igraph`](https://igraph.org/python/#docs), which wrap underlying optimised code libraries implemented in more performant languages such as `C++`. This is the approach that was adopted by the author prior to the development of `cityseer`, but whereas these performant packages offer tremendous utility for general-purpose network analysis they can remain cumbersome to piggy-back for more esoteric use-cases, some of which are briefly alluded to in the discussion on [motivation](/guide/#motivation). Doing so can lead to a degree of code complexity presenting a bottleneck to further experimentation and development, and heavily overloading such packages from `python` can cause any speed-advantages to rapidly dissipate. It was this conundrum that kickstarted the development of the current codebase which became formalised as the `cityseer` package.

Remember that `cityseer` does not explicitly implement globalised forms of network analysis, and currently has no intention of doing so because these make it tricky to compare metrics across locations. Therefore, if the aim is conventional forms of global centralities applied to the network as a global entity, then it may be worth sticking with a package such as [`Graph-Tool`](https://graph-tool.skewed.de) which wraps heavily optimised code designed for exactly these types of purposes. On the other-hand, if using localised forms of analysis that take into account factors such as localised distance thresholds, specialised centralities, shortest vs. simplest-path heuristics, or land-use accessibilities and mixed-uses, then `cityseer` offers methods that are not available through other off-the-shelf network analysis packages.

The present approach leveraged by `cityseer` consists of pure `python` and [`numpy`](http://www.numpy.org/), but with computationally intensive algorithms implemented in [`numba`](https://numba.pydata.org/) for the sake of performant JIT compilation. The use of python has provided the necessary flexibility to easily experiment with different implementations of underlying methodologies and algorithms, thereby facilitating the development of measures specific to analytics from an urbanist's perspective. The use of `numba` has made it feasible to scale these methods to large and, optionally, decomposed networks. Further, `numba` permits a style of programming more in-keeping with lower-level languages, i.e. it is possible to use loops explicitly, which can in many cases be easier to reason-with than nested array indices more typical of `numpy`. Note that `cityseer` algorithms, at present, are not necessarily heavily optimised for performance or designed explicitly for a performance-first paradigm: they have instead been implemented from a practical perspective that happens to be 'fast-enough' for day-to-day usage. There may well be opportunities for further efficiencies, however, these have not been deemed a priority at present and will only be implemented if they do not unnecessarily complicate the package.
