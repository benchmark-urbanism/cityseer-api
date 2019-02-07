---
---

<RenderMath></RenderMath>

Cityseer <Chip text="beta"/>
--------

`cityseer` is a collection of computational tools for fine-grained network and land-use analysis, useful for assessing the morphological precursors to vibrant neighbourhoods. It is underpinned by rigorous network-based methods that have been developed from the ground-up specifically for hyperlocal analysis at the pedestrian scale.

The use of `python` facilitates interaction with popular tools for network, geospatial, and scientific data analysis, i.e. [`networkX`](https://networkx.github.io/), [`shapely`](https://shapely.readthedocs.io), and the [`numpy`](http://www.numpy.org/) stack. The underlying algorithms are designed for efficient large-scale urban analysis and have been implemented in [`numba`](https://numba.pydata.org/) JIT compiled code.


Installation
------------

`cityseer` is a python package installed via `pip`:
```bash
pip install cityseer
```


Quickstart
----------

`cityseer` revolves around networks (graphs). If you're comfortable with `numpy` and abstract data handling, then the underlying data structures can be created and manipulated directly. However, it is generally more convenient to sketch the graph using [`NetworkX`](https://networkx.github.io/) and to let `cityseer` take care of initialising and converting the graph for you.

```python
# any NetworkX graph with 'x' and 'y' node attributes will do
# here we'll use the cityseer mock module to generate an example networkX graph
from cityseer.util import mock
G = mock.mock_graph()

'''
import networkx as nx
print(nx.info(G))
# Name:
# Type: Graph
# Number of nodes: 52
# Number of edges: 73
# Average degree:   2.8077
'''
# let's plot the network
from cityseer.util import plot
plot.plot_nX(G, labels=True)
```

<img src="/plots/graph.png" alt="Example graph" class="centre" style="max-height:450px;">

The [`util.graphs`](/util/graphs.html) module contains a collection of convenience functions for the preparation and conversion of `networkX` graphs, including
[`nX_wgs_to_utm`](/util/graphs.html#nx-wgs-to-utm) for coordinate conversions; [`nX_remove_filler_nodes`](/util/graphs.html#nx-remove-filler-nodes) for graph cleanup; [`nX_decompose`](/util/graphs.html#nx-decompose) for generating granular graph typologies; and [`nX_to_dual`](/util/graphs.html#nx-to-dual) for casting a primal graph representation to its dual. These functions are designed to work with raw `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries that have been assigned to the edge `geom` attributes. If working with simple graph representations — straight-line edges between nodes — then [`graphs.nX_simple_geoms`](/util/graphs.html#nx-simple-geoms) can generate these geometries for you. The benefit to the use of raw geoms is that the geometry of the network is kept distinct from the topology, and the fidelity of the geometries can therefore be retained in spite of topological transformations.

<img src="/plots/graph_decomposed.png" alt="Example decomposed graph" class="left"><img src="/plots/graph_dual.png" alt="Example dual graph" class="right">

_A decomposed variant of the graph (left) and a primal / dual transformation of the graph (right)._

Before conversion to a [`Network_Layer`](/metrics/networks.html#network-layer), the `networkX` graph must first be furnished with `length` and `impedance` edge attributes. These can be generated in one of several ways:

- If decomposing the graph, then the [`nX_decompose`](/util/graphs.html#nx-decompose) function will generate the `length` and `impedance` attributes as part of the decompositional process;

- If transposing to a dual graph, then [`nX_to_dual`](/util/graphs.html#nx-to-dual) will likewise generate the attributes; in this case `impedance` will represent total angular change over the length of of an edge segment;

- If neither of the above, then use the [`graphs.nX_auto_edge_params`](/util/graphs.html#nx-auto-edge-params) function;

- The attributes can also be set manually, if that's your thing :muscle:.

```python
from cityseer.util import graphs
# provide your own shapely geometries if you need precise street lengths / angles
# else, auto-generate simple geometries from the start to end node of each network edge
G = graphs.nX_simple_geoms(G)

# auto-set edge length and impedance attributes from the geoms
G = graphs.nX_auto_edge_params(G)
```

Once the `networkX` graph has been prepared, it can be transformed into a [`Network_Layer`](/metrics/networks.html#network-layer) by invoking [`Network_Layer_From_nX`](/metrics/networks.html#network-layer-from-nx). Network layers are used for network centrality computations and provide the backbone for landuse and statistical aggregations. A `Network_Layer` requires a set of distances $d_{max}$ specifying the local distance thresholds at which the centrality methods will be computed.

The [@compute_centrality](/metrics/networks.html#compute-centrality) method wraps underlying numba optimised functions providing access to all the available centrality methods. These are computed simultaneously for any required combinations of measures (and distances), which can have significant speed implications. However, situations requiring only a single measure can instead make use of the simpler [`@gravity`](/metrics/networks.html#gravity), [`@harmonic_closeness`](/metrics/networks.html#harmonic-closeness), [`@improved_closeness`](/metrics/networks.html#improved-closeness), [`@betweenness`](/metrics/networks.html#betweenness), or [`@weighted_betweenness`](/metrics/networks.html#betweenness-gravity) methods. 

The results of the computations will be written to the `Network_Layer` class, and can be access at the `Network_Layer.metrics` property. It is also possible to extract the data to a `python` dictionary through use of the [@metrics_to_dict](/metrics/networks.html#metrics-to-dict) method, or to simply convert the network — data and all — back into a `networkX` layer through use of the [@to_networkX](/metrics/networks.html#to-networkx) method.

```python
from cityseer.metrics import networks
# create a Network layer
N = networks.Network_Layer_From_nX(G, distances=[200, 400, 800, 1600], angular=False)
# one of several easy-wrapper methods for computing centrality
N.improved_closeness()
# the full underlying method allows the computation of various centralities at once, e.g.
N.compute_centrality(close_metrics=['improved', 'gravity', 'cycles'],
                     between_metrics=['betweenness_gravity'])
```

<img src="imp_close_800.png" alt="Improved Closeness 800m" class="centre">

_Example $800m$ improved closeness centrality for inner London._

_Contains OS data © Crown copyright and database right 2019._

```python
from cityseer.metrics import layers
# optionally, add land-uses from a dictionary with 'x', 'y' attributes
data_dict = mock.mock_data_dict(G)
# generate a data layer
D = layers.Data_Layer_From_Dict(data_dict)
# assign to the above Network Layer
D.assign_to_network(N, max_dist=400)
```

```python
# landuse labels can be used to generate mixed-use and land-use accessibility measures
landuse_labels = mock.mock_categorical_data(len(data_dict))
# example easy-wrapper method for computing mixed-uses
D.hill_branch_wt_diversity(landuse_labels, qs=[0, 1, 2])
# example easy-wrapper method for computing accessibilities
D.compute_accessibilities(landuse_labels, accessibility_labels=['a', 'c'])
# or custom, e.g.: 
D.compute_aggregated(mixed_use_metrics=['hill', 'shannon'], accessibility_labels=['a', 'b'])
```

Generate contextually sensitive statistics
```python
# statistics can be generated for numerical layers
mock_valuations_data = mock.mock_numerical_data(len(data_dict))
# compute max, min, mean, mean-weighted, range, and range-weighted using local distance thresholds
D.compute_stats_single(numerical_label='valuations', numerical_array=mock_valuations_data)
```

Compute network centrality
```python
# convert back to NetworkX
G_metrics = N.to_networkX()
# or a dictionary:
N.metrics_to_dict()
# or simply access the metrics directly from numpy arrays at N.metrics
```


Issues & Contributions
----------------------

Please report issues to the [`issues`](https://github.com/cityseer/cityseer-api/issues) page of the `cityseer` `github` repo.

Suggestions, contributions, and pull requests are welcome. Please discuss significant proposals prior to implementation.

