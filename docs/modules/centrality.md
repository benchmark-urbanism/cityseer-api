---
functions:
    distance_from_beta:
      name: distance_from_beta
      version: 0.1+
      intro: A convenience function mapping $-\beta$ decay parameters to equivalent $d_{max}$ distance thresholds.
      params:
        - name: beta
          type: float, list[float], ndarray
          desc: $-\beta$ value/s to convert to distance thresholds $d_{max}$.
        - name: min_threshold_wt
          type: float
          def: 0.01831563888873418
          desc: $w_{min}$ threshold at which to set the distance threshold $d_{max}$.
      returns:
        - name: betas
          type: numpy.ndarray
          desc: A numpy array of effective $d_{max}$ distances.
        - name: min_threshold_wt
          type: float
          desc: The corresponding $w_{min}$ threshold.
    graph_from_networkx:
      name: graph_from_networkx
      version: 0.1+
      intro: A convenience function for generating a `node_map` and `edge_map` from a [NetworkX](https://networkx.github.io/documentation/networkx-1.10/index.html) undirected Graph, which can then be passed to [`centrality.compute_centrality`](#compute-centrality).
      params:
        - name: network_x_graph
          type: networkx.Graph
          desc: A NetworkX undirected `Graph`. Requires node attributes `x` and `y` for spatial coordinates and accepts optional `length` and `weight` edge attributes. See notes.
        - name: wgs84_coords
          type: bool
          desc: Set to `True` if the `x` and `y` node attribute keys reference [`WGS84`](https://epsg.io/4326) lng, lat values instead of a projected coordinate system.
        - name: decompose
          type: int, float
          def: None
          desc: Generates a decomposed version of the graph wherein edges are broken into smaller sections no longer than the specified distance in metres.
        - name: geom
          type: shapely.geometry.Polygon
          def: None
          desc: Shapely geometry defining the original area of interest. Recommended for avoidance of boundary roll-off in computed metrics.
      returns:
        - name: node_map
          type: numpy.ndarray
          desc: Node data
        - name: edge_map
          type: numpy.ndarray
          desc: Edge data
---

<RenderMath></RenderMath>


centrality <Chip text="beta" :important="true"/>
==========


distance\_from\_beta() <Chip :text='$page.frontmatter.functions.distance_from_beta.version'/>
----------------------

<DisplayFunction :func='$page.frontmatter.functions.distance_from_beta'></DisplayFunction>

::: tip Note
There is no need to use this function unless:

- Providing custom beta values for weighted centrality measures in lieu of the automatically generated defaults;
- Using custom `min_threshold_wt` parameters.
:::

::: warning Important
Pass both $d_{max}$ and $w_{min}$ to [`centrality.compute_centrality`](#compute-centrality) for the desired behaviour.
:::

The weighted variants of centrality, i.e. gravity and weighted betweenness, are computed using a negative exponential decay function of the form:

$$weight = exp(-\beta \cdot distance)$$

The strength of the decay is controlled by the $-\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
For example, if $-\beta=0.005$ were to represent a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at 13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At some point, it becomes futile to consider locations any farther away, so it is necessary to set a a minimum weight threshold $w_{min}$ corresponding to a maximum distance of $d_{max}$.

The [`centrality.compute_centrality`](#compute-centrality) method computes the $-\beta$ parameters automatically using the following formula, and a default `min_threshold_wt` of $w_{min}=0.01831563888873418$:

$$\beta = \frac{log\Big(1 / w_{min}\Big)}{d_{max}}$$

Therefore, $-\beta$ weights corresponding to $d_{max}$ walking thresholds of 400m, 800m, and 1600m would yield:

| $d_{max}$ | $-\beta$ |
|-----------|:----------|
| $400m$ | $-0.01$ |
| $800m$ | $-0.005$ |
| $1600m$ | $-0.0025$ |

In reality, people may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context. If overriding the default $-\beta$ or $w_{min}$ threshold, then this function can be used to generate the effective $d_{max}$ values, which can then be passed to [`centrality.compute_centrality`](#compute-centrality) along with the specified $w_{min}$. For example, the following $-\beta$ and $w_{min}$ thresholds yield these effective $d_{max}$ distances:

| $-\beta$ | $w_{min}$ | $d_{max}$ |
|----------|:----------|:----------|
| $-0.01$ | $0.01$ | $461m$ |
| $-0.005$ | $0.01$ | $921m$ |
| $-0.0025$ | $0.01$ | $1842m$ |


graph\_from\_networkx() <Chip :text='$page.frontmatter.functions.distance_from_beta.version'/>
-----------------------

<DisplayFunction :func='$page.frontmatter.functions.graph_from_networkx'></DisplayFunction>

The node attributes `x` and `y` determine the spatial coordinates of the node, and should be in a suitable projected (flat) coordinate reference system in metres unless the `wgs84_coords` parameter is set to `True`.

The optional edge attribute `length` indicates the original edge length in metres. If not provided, lengths will be computed using crow-flies distances between either end of the edges.

If provided, the optional edge attribute `weight` will be used by shortest path algorithms instead of distances in metres.

::: tip Note
When calculating local network centralities, it is best-practice for the area of interest to have been buffered by a distance equal to the maximum distance threshold to be considered. This prevents misleading results arising due to a boundary roll-off effect. If provided, the `geom` geometry is used to identify nodes falling within the original non-buffered area of interest. Metrics will then only be computed for these nodes, thus avoiding roll-off effects and reducing frivolous computation. (The algorithms still have access to the full buffered network.)
:::

::: warning Important
Graph decomposition provides a more granular representation of variations along street lengths. However, setting the `decompose` parameter too small can increase the computation time unnecessarily for subsequent analysis. It is generally not necessary to go smaller $20m$, and $50m$ may already be sufficient for many cases.
:::


compute\_centrality()
---------------------

