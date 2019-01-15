---

---

## Installation

Cityseer is a python package, which can be installed with `pip`:
```bash
pip install cityseer
```

## Overview

Cityseer includes methods for converting and preparing your data for analysis:
```python
from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs

# any NetworkX graph with 'x' and 'y' attributes will do
# here we'll generate a mock-graph from Cityseer's mock module
G, pos = mock.mock_graph()

# you can provide shapely geometries if you need accurate street lengths (or angles in the case of the dual graph)
# for this example, we'll simply auto-generate geometries from the start to end node of each network edge
G = graphs.networkX_simple_geoms(G)

# auto-set edge lengths and impedances from geom lengths
G = graphs.networkX_edge_defaults(G)
# optionally, convert to dual: G.networkX_to_dual()
```

Cityseer uses network algorithms to compute street network centralities.
```python
# create a Network layer
N = networks.Network_Layer_From_NetworkX(G, distances=[100, 400, 800, 1600], angular=False)
# example easy-wrapper method for computing centrality
N.harmonic_closeness()
# or access the full underlying wrapper method, e.g.
N.compute_centrality(close_metrics=['improved', 'gravity', 'cycles'], between_metrics=['betweenness_gravity'])
```

Compute mixed-use measures and land-use accessibility
```python
# optionally, add land-uses from a dictionary with 'x', 'y' attributes
data_dict = mock.mock_data_dict(G)
# generate a data layer
D = layers.Data_Layer_From_Dict(data_dict)
# assign to the above Network Layer
D.assign_to_network(N, max_dist=400)

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


Modules
-------

### [/metrics/](metrics/README.md)

### [/util/](util/README.md)


Development
----------


License & Attribution
---------------------

Attribution is required.

Apache License v2.0 + Commons Clause License v1.0

Copyright © 2018-present Gareth Simons