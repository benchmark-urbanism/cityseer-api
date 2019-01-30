---

---

::: slot boo

# Here might be a page title

:::


```python
from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs

# any NetworkX graph with 'x' and 'y' attributes will do
G, pos = mock.mock_graph()
# supply your own street geometries, or auto-generate
G = graphs.networkX_simple_geoms(G)
# optionally, convert to dual: G.networkX_to_dual()
# optionally, auto-set edge parameters from geom lengths
G = graphs.networkX_edge_defaults(G)
# create a Network layer
N = networks.Network_Layer_From_NetworkX(G, distances=[100, 400, 800, 1600], angular=False)
# example easy-wrapper method for computing centrality
N.harmonic_closeness()
# or access the full underlying wrapper method, e.g.
N.compute_centrality(close_metrics=['improved', 'gravity', 'cycles'], between_metrics=['betweenness_gravity'])

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

# statistics can be generated for numerical layers
mock_valuations_data = mock.mock_numerical_data(len(data_dict))
# compute max, min, mean, mean-weighted, range, and range-weighted using local distance thresholds
D.compute_stats_single(numerical_label='valuations', numerical_array=mock_valuations_data)

# convert back to NetworkX
G_metrics = N.to_networkX()
# or a dictionary:
N.metrics_to_dict()
# or simply access the metrics directly from numpy arrays at N.metrics
```

<RenderMath></RenderMath>

<FuncSignature>distance_from_beta(beta, min_threshold_wt=0.01831563888873418)</FuncSignature>

- A Paragraph
- Another Paragraph

::: slot baa

Here's some contact info

:::