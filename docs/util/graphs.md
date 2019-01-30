---

---

<RenderMath></RenderMath>

cityseer.util.graphs
====================

A collection of convenience functions for the preparation and conversion of [`NetworkX`](https://networkx.github.io/) graphs.

`cityseer`'s network data structures can be created and manipulated directly, where so desired.

networkX_simple_geoms
---------------------

<FuncSignature>networkX_simple_geoms(networkX_graph)</FuncSignature>

Generates straight-line geometries for each edge based on the the `x` and `y` coordinates of the adjacent nodes.

<FuncHeading>Parameters</FuncHeading>

<FuncElement name="networkX_graph" type="nx.Graph">

$-\beta$ value/s to convert to distance thresholds $d_{max}$.

</FuncElement>
<FuncElement name="min_threshold_wt" type="float">

The minimum weight $w_{min}$ at which to set the distance threshold $d_{max}$.

</FuncElement>

<FuncHeading>Returns</FuncHeading>
<FuncElement name="betas" type="numpy.ndarray">

A numpy array of $d_{max}$ distances.

</FuncElement>
<FuncElement name="min_threshold_wt" type="float">

The $w_{min}$ threshold.

</FuncElement>