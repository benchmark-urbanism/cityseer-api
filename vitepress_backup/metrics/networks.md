

### Node attributes



### Edge attributes

The start and end edge `idx` attributes point to the corresponding node indices in the `node_data` array.

The `length` edge attribute (index $2$) should always correspond to the edge lengths in metres. This is used when calculating the distances traversed by the shortest-path algorithm so that the respective $d_{max}$ maximum distance thresholds can be enforced: these distance thresholds are based on the actual network-paths traversed by the algorithm as opposed to crow-flies distances.

The `angle_sum` edge bearing (index $3$) should correspond to the total angular change along the length of the segment. This is used when calculating angular impedances for simplest-path measures. The `start_bearing` (index $5$) and `end_bearing` (index $6$) attributes respectively represent the starting and ending bearing of the segment. This is also used when calculating simplest-path measures when the algorithm steps from one edge to another.

The `imp_factor` edge attribute (index $4$) represents an impedance multiplier for increasing or diminishing the impedance of an edge. This is ordinarily set to $1$, therefor not impacting calculations. By setting this to greater or less than $1$, the edge will have a correspondingly higher or lower impedance. This can be used to take considerations such as street gradients into account, but should be used with caution.

::: tip Hint

It is possible to represent unlimited $d_{max}$ distance thresholds by setting one of the specified `distance` parameter values to `np.inf`. Note that this may substantially increase the computational time required for the completion of the algorithms on large networks.

:::

## @metrics_to_dict

<FuncSignature>NetworkLayer.metrics_to_dict()</FuncSignature>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="metrics_dict" type="dict">

Unpacks all calculated metrics from the `NetworkLayer.metrics` property into a `python` dictionary. The dictionary `keys` will correspond to the node `uids`.

</FuncElement>

```python
from cityseer.metrics import networks
from cityseer.tools import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)

# generate the network layer and compute some metrics
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
N.node_centrality(measures=['node_harmonic'])

# let's select a random node id
random_idx = 6
random_uid = N.uids[random_idx]

# the data is directly available at N.metrics
# in this case the data is stored in arrays corresponding to the node indices
print(N.metrics['centrality']['node_harmonic'][200][random_idx])
# prints: 0.023120252

# let's convert the data to a dictionary
# the unpacked data is now stored by the uid of the node identifier
data_dict = N.metrics_to_dict()
print(data_dict[random_uid]['centrality']['node_harmonic'][200])
# prints: 0.023120252
```

## @to_networkX

<FuncSignature>NetworkLayer.to_networkX()</FuncSignature>

Transposes a `NetworkLayer` into a `networkX` graph. This method calls [nX_from_graph_maps](/util/graphs.html#nx-from-graph-maps) internally.

<FuncHeading>Returns</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph.

`x`, `y`, `live`, `ghosted` node attributes will be copied from `node_data` to the graph nodes. `length`, `angle_sum`, `imp_factor`, `start_bearing`, and `end_bearing` attributes will be copied from the `edge_data` to the graph edges.

If a `metrics_dict` is provided, all data will be copied to the graph nodes based on matching node identifiers.

</FuncElement>

```python
from cityseer.metrics import networks
from cityseer.tools import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)

# generate the network layer and compute some metrics
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
# compute some-or-other metrics
N.node_centrality(measures=['node_harmonic'])
# convert back to networkX
G_post = N.to_networkX()

# let's select a random node id
random_idx = 6
random_uid = N.uids[random_idx]

print(N.metrics['centrality']['node_harmonic'][200][random_idx])
# prints: 0.023120252

# the metrics have been copied to the new networkX graph
print(G_post.nodes[random_uid]['metrics']['centrality']['node_harmonic'][200])
# prints: 0.023120252
```

<img src="../images/plots/graph_before.png" alt="Graph before conversion" class="left"><img src="../images/plots/graph_after.png" alt="graph after conversion back to networkX" class="right">

_A `networkX` graph before conversion to a `NetworkLayer` (left) and after conversion back to `networkX` (right)._

## Network centrality methods

There are two network centrality methods available depending on whether you're using a node-based or segment-based approach:

- [compute_node_centrality](#compute-node-centrality)
- [compute_segment_centrality](#compute-segment-centrality)

These methods wrap the underlying `numba` optimised functions for computing centralities, and provides access to all of the underlying node-based or segment-based centrality methods. Multiple selected measures and distances are computed simultaneously to reduce the amount of time required for multi-variable and multi-scalar strategies.

::: tip Hints
The reasons for picking one approach over another are varied:

- Node based centralities compute the measures relative to each reachable node within the threshold distances. For this reason, they can be susceptible to distortions caused by messy graph topologies such redundant and varied concentrations of $degree=2$ nodes (e.g. to describe roadway geometry) or needlessly complex representations of street intersections. In these cases, the network should first be cleaned using methods such as those available in the [graph](/util/graphs) module (see the [graph cleaning guide](/guide/cleaning) for examples). If a network topology has varied intensities of nodes but the street segments are less spurious, then segmentised methods can be preferable because they are based on segment distances: segment aggregations remain the same regardless of the number of intervening nodes;
- Node-based `harmonic` centrality can be problematic on graphs where nodes are erroneously placed too close together or where impedances otherwise approach zero, as may be the case for simplest-path measures or small distance thesholds. This happens because the outcome of the division step can balloon towards $\infty$ once impedances decrease below $1$.
- Note that `cityseer`'s implementation of simplest (angular) measures work on both primal (node or segment based) and dual graphs (node only).
- Measures should only be directly compared on the same topology because different topologies can otherwise affect the expression of a measure. Accordingly, measures computed on dual graphs cannot be compared to measures computed on primal graphs because this does not account for the impact of differing topologies. Dual graph representations can have substantially greater numbers of nodes and edges for the same underlying street network; for example, a four-way intersection consisting of one node with four edges translates to four nodes and six edges on the dual. This effect is amplified for denser regions of the network.
- Segmentised versions of centrality measures should not be computed on dual graph topologies because street segment lengths would be duplicated for each permutation of dual edge spanning street intersections. By way of example, the contribution of a single edge segment at a four-way intersection would be duplicated three times.
- Global closeness is strongly discouraged because it does not behave suitably for localised graphs. Harmonic closeness or improved closeness should be used instead. Note that Global closeness ($\frac{nodes}{farness}$) and improved closeness ($\frac{nodes}{farness / nodes}$) can be recovered from the available metrics, if so desired, through additional (manual) steps.
- Network decomposition can be a useful strategy when working at small distance thresholds, and confers advantages such as more regularly spaced snapshots and fewer artefacts at small distance thresholds where street edges intersect distance thresholds. However, the regular spacing of the decomposed segments will introduce spikes in the distributions of node-based centrality measures when working at very small distance thresholds. Segmentised versions may therefore be preferable when working at small thresholds on decomposed networks.
  :::

The computed metrics will be written to a dictionary available at the `NetworkLayer.metrics` property and will be categorised by the respective centrality and distance keys:

<small>`NetworkLayer.metrics['centrality'][<<measure key>>][<<distance key>>][<<node idx>>]`</small>

For example, if `node_density`, and `node_betweenness_beta` centrality keys are computed at $800m$ and $1600m$, then the dictionary would assume the following structure:

```python
# example structure
NetworkLayer.metrics = {
    'centrality': {
        'node_density': {
            800: [<numpy array>],
            1600: [<numpy array>]
        },
        'node_betweenness_beta': {
            800: [<numpy array>],
            1600: [<numpy array>]
        }
    }
}
```

A working example:

```python
from cityseer.metrics import networks
from cityseer.tools import mock, graphs

# prepare a mock graph
G = mock.mock_graph()
G = graphs.nX_simple_geoms(G)

# generate the network layer and compute some metrics
N = networks.NetworkLayerFromNX(G, distances=[200, 400, 800, 1600])
# compute a centrality measure
N.node_centrality(measures=['node_density', 'node_betweenness_beta'])

# fetch node density for 400m threshold for the first 5 nodes
print(N.metrics['centrality']['node_density'][400][:5])
# prints [15, 13, 10, 11, 12]

# fetch betweenness beta for 1600m threshold for the first 5 nodes
print(N.metrics['centrality']['node_betweenness_beta'][1600][:5])
# prints [75.83173, 45.188183, 6.805982, 11.478158, 33.74703]
```

The data can be handled using the underlying `numpy` arrays, and can also be unpacked to a dictionary using [`NetworkLayer.metrics_to_dict`](#metrics-to-dict) or transposed to a `networkX` graph using [`NetworkLayer.to_networkX`](#to-networkx).

## @compute_node_centrality <Chip text="v0.12.0"/>

<FuncSignature>
<pre>
NetworkLayer.compute_node_centrality(measures=None, angular=False)
</pre>
</FuncSignature>

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="measures" type="list[str], tuple[str]">

A list or tuple of strings, containing any combination of the following `key` values, computed within the respective distance thresholds of $d_{max}$.

</FuncElement>

<FuncElement name="angular" type="bool">

A boolean indicating whether to use shortest or simplest path heuristics.

</FuncElement>

The following keys use the shortest-path heuristic, and are available when the `angular` parameter is set to the default value of `False`:

| key                                  |                                       formula                                       | notes                                                                                                                                                                                                                                                                 |
| ------------------------------------ | :---------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <small>node_density</small>          |                         $\scriptstyle\sum_{j\neq{i}}^{n}1$                          | <small>A summation of nodes.</small>                                                                                                                                                                                                                                  |
| <small>node_farness</small>          |                     $\scriptstyle\sum_{j\neq{i}}^{n}d_{(i,j)}$                      | <small>A summation of distances in metres.</small>                                                                                                                                                                                                                    |
| <small>node_cycles</small>           |                      $\scriptstyle\sum_{j\neq{i}j=cycle}^{n}1$                      | <small>A summation of network cycles.</small>                                                                                                                                                                                                                         |
| <small>node_harmonic</small>         |                $\scriptstyle\sum_{j\neq{i}}^{n}\frac{1}{Z_{(i,j)}}$                 | <small>Harmonic closeness is an appropriate form of closeness centrality for localised implementations constrained by the threshold $d_{max}$.</small>                                                                                                                |
| <small>node_beta</small>             |              $\scriptstyle\sum_{j\neq{i}}^{n}\\exp(\beta\cdot d[i,j])$              | <small>Also known as the '_gravity index_'. This is a spatial impedance metric differentiated from other closeness centralities by the use of an explicit $\beta$ parameter, which can be used to model the decay in walking tolerance as distances increase.</small> |
| <small>node_betweenness</small>      |            $\scriptstyle\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$             | <small>Betweenness centrality summing all shortest-paths traversing each node $i$.</small>                                                                                                                                                                            |
| <small>node_betweenness_beta</small> | $\scriptstyle\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}\\exp(\beta\cdot d[j,k])$ | <small>Applies a spatial impedance decay function to betweenness centrality. $d$ represents the full distance from any $j$ to $k$ node pair passing through node $i$.</small>                                                                                         |

The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular` parameter is explicitly set to `True`:

| key                                     |                           formula                            | notes                                                                                                                                                                                                                                                      |
| --------------------------------------- | :----------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <small>node_harmonic_angular</small>    |     $\scriptstyle\sum_{j\neq{i}}^{n}\frac{1}{Z_{(i,j)}}$     | <small>The simplest-path implementation of harmonic closeness uses angular-distances for the impedance parameter. Angular-distances are normalised by 180 and added to $1$ to avoid division by zero: $\scriptstyle{Z = 1 + (angularchange/180)}$.</small> |
| <small>node_betweenness_angular</small> | $\scriptstyle\sum_{j\neq{i}}^{n}\sum_{k\neq{j}\neq{i}}^{n}1$ | <small>The simplest-path version of betweenness centrality. This is distinguished from the shortest-path version by use of a simplest-path heuristic (shortest angular distance).</small>                                                                  |

## @compute_segment_centrality <Chip text="v0.12.0"/>

<FuncSignature>
<pre>
NetworkLayer.compute_segment_centrality(measures=None, angular=False)
</pre>
</FuncSignature>

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="measures" type="list[str], tuple[str]">

A list or tuple of strings, containing any combination of the following `key` values, computed within the respective distance thresholds of $d_{max}$.

</FuncElement>

<FuncElement name="angular" type="bool">

A boolean indicating whether to use shortest or simplest path heuristics.

</FuncElement>

The following keys use the shortest-path heuristic, and are available when the `angular` parameter is set to the default value of `False`:

| key                                |                                               formula                                               | notes                                                                                                                                                                    |
| ---------------------------------- | :-------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <small>segment_density</small>     |                          $\scriptstyle\sum_{(a, b)}^{edges}d_{b} - d_{a}$                           | <small>A summation of edge lengths.</small>                                                                                                                              |
| <small>segment_harmonic</small>    |                    $\scriptstyle\sum_{(a, b)}^{edges}\int_{a}^{b}\ln(b) -\ln(a)$                    | <small>A continuous form of harmonic closeness centrality applied to edge lengths.</small>                                                                               |
| <small>segment_beta</small>        | $\scriptstyle\sum_{(a, b)}^{edges}\int_{a}^{b}\frac{\exp(\beta\cdot b) -\exp(\beta\cdot a)}{\beta}$ | <small>A continuous form of beta-weighted (gravity index) centrality applied to edge lengths.</small>                                                                    |
| <small>segment_betweenness</small> |                                                                                                     | <small>A continuous form of betweenness: Resembles `segment_beta` applied to edges situated on shortest paths between all nodes $j$ and $k$ passing through $i$.</small> |

The following keys use the simplest-path (shortest-angular-path) heuristic, and are available when the `angular` parameter is explicitly set to `True`.

| key                                      |                          formula                           | notes                                                                                                                                                                                       |
| ---------------------------------------- | :--------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <small>segment_harmonic_hybrid</small>   | $\scriptstyle\sum_{(a, b)}^{edges}\frac{d_{b} - d_{a}}{Z}$ | <small>Weights angular harmonic centrality by the lengths of the edges. See `node_harmonic_angular`.</small>                                                                                |
| <small>segment_betweeness_hybrid</small> |                                                            | <small>A continuous form of angular betweenness: Resembles `segment_harmonic_hybrid` applied to edges situated on shortest paths between all nodes $j$ and $k$ passing through $i$.</small> |

## NetworkLayerFromNX <Chip text="class"/>

<FuncSignature>
<pre>
NetworkLayerFromNX(networkX_graph,
                      distances=None,
                      betas=None,
                      min_threshold_wt=0.01831563888873418)
</pre>
</FuncSignature>

Directly transposes a `networkX` graph into a `NetworkLayer`. This `class` simplifies the conversion of `NetworkX` graphs by calling [`graph_maps_from_nX`](/util/graphs.html#graph-maps-from-nx) internally. Methods and properties are inherited from the parent [`NetworkLayer`](#network-layer) class.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

A `networkX` graph.

`x` and `y` node attributes are required. The `live` node attribute is optional, but recommended. See [`NetworkLayer`](#network-layer) for more information about what these attributes represent.

</FuncElement>

<FuncElement name="distances" type="int, float, list, tuple, np.ndarray">

A distance, or `list`, `tuple`, or `numpy` array of distances corresponding to the local $d_{max}$ thresholds to be used for centrality (and land-use) calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead. Use a distance of `np.inf` where no distance threshold should be enforced.

</FuncElement>

<FuncElement name="betas" type="float, list, tuple, np.ndarray">

A $\beta$, or `list`, `tuple`, or `numpy` array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.

</FuncElement>

<FuncElement name="min_threshold_wt" type="float">

The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [distance_from_beta](#distance-from-beta) for more information.

</FuncElement>

<FuncHeading>Returns</FuncHeading>

<FuncElement name="Network_Layer" type="class">

A `NetworkLayer`.

</FuncElement>
