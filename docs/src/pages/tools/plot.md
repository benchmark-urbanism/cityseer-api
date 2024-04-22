---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# plot


 Convenience methods for plotting graphs within the cityseer API context. Custom behaviour can be achieved by directly manipulating the underlying [`NetworkX`](https://networkx.github.io) and [`matplotlib`](https://matplotlib.org) figures. This module is predominately used for basic plots or visual verification of behaviour in code tests. Users are encouraged to use matplotlib or other plotting packages directly where possible.


<div class="class">


## ColourMap



 Specifies global colour presets.



<div class="function">

## ColourMap


<div class="content">
<span class="name">ColourMap</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="function">

## plot_nx_primal_or_dual


<div class="content">
<span class="name">plot_nx_primal_or_dual</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">primal_graph</span>
  </div>
  <div class="param">
    <span class="pn">dual_graph</span>
  </div>
  <div class="param">
    <span class="pn">path</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">labels</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">primal_node_size</span>
    <span class="pc">:</span>
    <span class="pa"> int = 30</span>
  </div>
  <div class="param">
    <span class="pn">primal_node_colour</span>
  </div>
  <div class="param">
    <span class="pn">primal_edge_colour</span>
  </div>
  <div class="param">
    <span class="pn">dual_node_size</span>
    <span class="pc">:</span>
    <span class="pa"> int = 30</span>
  </div>
  <div class="param">
    <span class="pn">dual_node_colour</span>
  </div>
  <div class="param">
    <span class="pn">dual_edge_colour</span>
  </div>
  <div class="param">
    <span class="pn">primal_edge_width</span>
    <span class="pc">:</span>
    <span class="pa"> int | float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">dual_edge_width</span>
    <span class="pc">:</span>
    <span class="pa"> int | float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">plot_geoms</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">x_lim</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">y_lim</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">ax</span>
    <span class="pc">:</span>
    <span class="pa"> matplotlib.axes._axes.Axes | None = None</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Plot a primal or dual cityseer graph.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">primal_graph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 An optional `NetworkX` MultiGraph to plot in the primal representation. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dual_graph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 An optional `NetworkX` MultiGraph to plot in the dual representation. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">labels</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to display node labels. Defaults to False.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">primal_node_size</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The diameter for the primal graph's nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">primal_node_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Primal node colour or colours. When passing an iterable of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">primal_edge_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Primal edge colour or colours. When passing an iterable of colours, the number of colours should match the order and number of edges in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dual_node_size</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The diameter for the dual graph's nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dual_node_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Dual node colour or colours. When passing a list of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dual_edge_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Dual edge colour or colours. When passing an iterable of colours, the number of colours should match the order and number of edges in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">primal_edge_width</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Linewidths for the primal edge. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">dual_edge_width</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Linewidths for the dual edge. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">plot_geoms</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to represent edges. Defaults to True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x_lim</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple or list with the minimum and maxium `x` extents to be plotted. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">y_lim</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple or list with the minimum and maxium `y` extents to be plotted. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">ax</div>
    <div class="type">plt.Axes</div>
  </div>
  <div class="desc">

 An optional `matplotlib` `ax` to which to plot. If not provided, a figure and ax will be generated.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">**kwargs</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the default figure size or dpi parameters.</div>
</div>

### Notes

 Plot either or both primal and dual representations of a `networkX MultiGraph`. Only call this function directly if explicitly printing both primal and dual graphs. Otherwise, use the simplified [`plot_nx`](/tools/plot#plot-nx) method instead.

```py
from cityseer.tools import mock, graphs, plot
G = mock.mock_graph()
G_simple = graphs.nx_simple_geoms(G)
G_dual = graphs.nx_to_dual(G_simple)
plot.plot_nx_primal_or_dual(G_simple,
                            G_dual,
                            plot_geoms=False)
```

![Example primal and dual graph plot.](/images/graph_dual.png) _A dual graph in blue overlaid on the source primal graph in red._

</div>


<div class="function">

## plot_nx


<div class="content">
<span class="name">plot_nx</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">path</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">labels</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">node_size</span>
    <span class="pc">:</span>
    <span class="pa"> int = 20</span>
  </div>
  <div class="param">
    <span class="pn">node_colour</span>
  </div>
  <div class="param">
    <span class="pn">edge_colour</span>
  </div>
  <div class="param">
    <span class="pn">edge_width</span>
    <span class="pc">:</span>
    <span class="pa"> int | float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">plot_geoms</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">x_lim</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">y_lim</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">ax</span>
    <span class="pc">:</span>
    <span class="pa"> matplotlib.axes._axes.Axes | None = None</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Plot a `networkX` MultiGraph.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` MultiGraph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">labels</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to display node labels. Defaults to False.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">node_size</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The diameter for the graph's nodes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">node_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Node colour or colours. When passing an iterable of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Edges colour as a `matplotlib` compatible colour string. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_width</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Linewidths for edges. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">plot_geoms</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to represent edges. Defaults to True.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x_lim</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple or list with the minimum and maximum `x` extents to be plotted. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">y_lim</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple or list with the minimum and maximum `y` extents to be plotted. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">ax</div>
    <div class="type">plt.Axes</div>
  </div>
  <div class="desc">

 An optional `matplotlib` `ax` to which to plot. If not provided, a figure and ax will be generated.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">**kwargs</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the default figure size or dpi parameters.</div>
</div>

### Notes

```py
from cityseer.tools import mock, graphs, plot, io
from cityseer.metrics import networks
from matplotlib import colors

# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 50)
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
networks.node_centrality_shortest(
    network_structure=network_structure,
    nodes_gdf=nodes_gdf,
    distances=[800]
)
G_after = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data["cc_beta_800"])
# let's create a custom colourmap using matplotlib
cmap = colors.LinearSegmentedColormap.from_list(
    "cityseer", [(100 / 255, 193 / 255, 255 / 255, 255 / 255), (211 / 255, 47 / 255, 47 / 255, 1 / 255)]
)
# normalise the values
vals = colors.Normalize()(vals)
# cast against the colour map
cols = cmap(vals)
# plot
plot.plot_nx(G_after, node_colour=cols)
```


![Example Colour Plot.](/images/graph_colour.png) _Colour plot of 800m gravity index centrality on a 50m decomposed graph._

</div>


<div class="function">

## plot_assignment


<div class="content">
<span class="name">plot_assignment</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">nx_multigraph</span>
  </div>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">path</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">node_colour</span>
  </div>
  <div class="param">
    <span class="pn">node_labels</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">data_labels</span>
  </div>
  <div class="param">
    <span class="pn">**kwargs</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Plot a `network_structure` and `data_gdf` for visualising assignment of data points to respective nodes.
:::warning
This method is primarily intended for package testing and development.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.NetworkStructure</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure) instance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` MultiGraph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `data_gdf` `GeoDataFrame` with `nearest_assigned` and `next_neareset_assign` columns.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">node_colour</div>
    <div class="type">str | float | ndarray</div>
  </div>
  <div class="desc">

 Node colour or colours. When passing a list of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long method and should be formatted accordingly. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">node_labels</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to plot the node labels. Defaults to False.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_labels</div>
    <div class="type">ndarray[int | str]</div>
  </div>
  <div class="desc">

 An optional iterable of categorical data labels which will be mapped to colours. The number of labels should match the number of data points in `data_layer`. Defaults to None.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">**kwargs</div>
    <div class="type">None</div>
  </div>
  <div class="desc">

 `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the default figure size or dpi parameters.</div>
</div>

### Notes

![Example assignment plot.](/images/assignment_plot.png) _An assignment plot to a 50m decomposed graph, with the data points coloured by categorical labels._

</div>


<div class="function">

## plot_network_structure


<div class="content">
<span class="name">plot_network_structure</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">data_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">poly</span>
    <span class="pc">:</span>
    <span class="pa"> shapely.geometry.polygon.Polygon | None = None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Plot a graph from raw `cityseer` network structure.
:::note
Note that this function is subject to frequent revision pending short-term development requirements. It is used
mainly to visually confirm the correct behaviour of particular algorithms during the software development cycle.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">network_structure</div>
    <div class="type">rustalgos.NetworkStructure</div>
  </div>
  <div class="desc">

 A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure) instance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_gdf</div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `data_gdf` `GeoDataFrame` with `nearest_assigned` and `next_neareset_assign` columns.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">poly</div>
    <div class="type">geometry.Polygon</div>
  </div>
  <div class="desc">

 An optional polygon. Defaults to None.</div>
</div>


</div>


<div class="function">

## plot_scatter


<div class="content">
<span class="name">plot_scatter</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">ax</span>
    <span class="pc">:</span>
    <span class="pa"> matplotlib.axes._axes.Axes</span>
  </div>
  <div class="param">
    <span class="pn">xs</span>
  </div>
  <div class="param">
    <span class="pn">ys</span>
  </div>
  <div class="param">
    <span class="pn">vals</span>
  </div>
  <div class="param">
    <span class="pn">bbox_extents</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[int, int, int, int] | tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">perc_range</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] = (0.01, 99.99)</span>
  </div>
  <div class="param">
    <span class="pn">cmap_key</span>
    <span class="pc">:</span>
    <span class="pa"> str = 'viridis'</span>
  </div>
  <div class="param">
    <span class="pn">shape_exp</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1</span>
  </div>
  <div class="param">
    <span class="pn">s_min</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">s_max</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1</span>
  </div>
  <div class="param">
    <span class="pn">rasterized</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">face_colour</span>
    <span class="pc">:</span>
    <span class="pa"> str = '#111'</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Convenience plotting function for plotting outputs from examples in the Cityseer Guide.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">ax</div>
    <div class="type">plt.Axes</div>
  </div>
  <div class="desc">

 A 'matplotlib' `Ax` to which to plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">xs</div>
    <div class="type">ndarray[float]</div>
  </div>
  <div class="desc">

 A numpy array of floats representing the `x` coordinates for points to plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">ys</div>
    <div class="type">ndarray[float]</div>
  </div>
  <div class="desc">

 A numpy array of floats representing the `y` coordinates for points to plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">vals</div>
    <div class="type">ndarray[float]</div>
  </div>
  <div class="desc">

 A numpy array of floats representing the data values for the provided points.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">bbox_extents</div>
    <div class="type">tuple[int, int, int, int]</div>
  </div>
  <div class="desc">

 A tuple or list containing the `[s, w, n, e]` bounding box extents for clipping the plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">perc_range</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple of two floats, representing the minimum and maximum percentiles at which to clip the data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cmap_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A `matplotlib` colour map key.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">shape_exp</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing an exponential for reshaping the values distribution. Defaults to 1 which returns the values as provided. An exponential greater than or less than 1 will shape the values distribution accordingly.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">s_min</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing the minimum size for a plotted point.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">s_max</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing the maximum size for a plotted point.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">rasterized</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether or not to rasterise the output. Recommended for plots with a large number of points.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">face_colour</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A hex or other valid `matplotlib` colour value for the ax and figure faces (backgrounds).</div>
</div>


</div>


<div class="function">

## plot_nx_edges


<div class="content">
<span class="name">plot_nx_edges</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">ax</span>
    <span class="pc">:</span>
    <span class="pa"> matplotlib.axes._axes.Axes</span>
  </div>
  <div class="param">
    <span class="pn">nx_multigraph</span>
    <span class="pc">:</span>
    <span class="pa"> networkx.classes.multigraph.MultiGraph</span>
  </div>
  <div class="param">
    <span class="pn">edge_metrics_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">bbox_extents</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[int, int, int, int] | tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">perc_range</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float] = (0.01, 99.99)</span>
  </div>
  <div class="param">
    <span class="pn">cmap_key</span>
    <span class="pc">:</span>
    <span class="pa"> str = 'viridis'</span>
  </div>
  <div class="param">
    <span class="pn">shape_exp</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1</span>
  </div>
  <div class="param">
    <span class="pn">lw_min</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.1</span>
  </div>
  <div class="param">
    <span class="pn">lw_max</span>
    <span class="pc">:</span>
    <span class="pa"> float = 1</span>
  </div>
  <div class="param">
    <span class="pn">edge_label_key</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">colour_by_categorical</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <div class="param">
    <span class="pn">max_n_categorical</span>
    <span class="pc">:</span>
    <span class="pa"> int = 10</span>
  </div>
  <div class="param">
    <span class="pn">rasterized</span>
    <span class="pc">:</span>
    <span class="pa"> bool = True</span>
  </div>
  <div class="param">
    <span class="pn">face_colour</span>
    <span class="pc">:</span>
    <span class="pa"> str = '#111'</span>
  </div>
  <div class="param">
    <span class="pn">invert_plot_order</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Convenience plotting function for plotting outputs from examples in the Cityseer Guide.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">ax</div>
    <div class="type">plt.Axes</div>
  </div>
  <div class="desc">

 A 'matplotlib' `Ax` to which to plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` MultiGraph.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_metrics_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An edge key for the provided `nx_multigraph`. Plotted values will be retrieved from this edge key.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">bbox_extents</div>
    <div class="type">tuple[int, int, int, int]</div>
  </div>
  <div class="desc">

 A tuple or list containing the `[s, w, n, e]` bounding box extents for clipping the plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">perc_range</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">

 A tuple of two floats, representing the minimum and maximum percentiles at which to clip the data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cmap_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A `matplotlib` colour map key.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">shape_exp</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing an exponential for reshaping the values distribution. Defaults to 1 which returns the values as provided. An exponential greater than or less than 1 will shape the values distribution accordingly.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">lw_min</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing the minimum line width for a plotted edge.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">lw_max</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A float representing the maximum line width for a plotted edge.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_label_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A key for retrieving categorical labels from edges.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">colour_by_categorical</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to plot colours by categorical. This requires an `edge_label_key` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_n_categorical</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of categorical values (sorted in decreasing order) to plot.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">rasterized</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether or not to rasterise the output. Recommended for plots with a large number of edges.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">face_colour</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 A hex or other valid `matplotlib` colour value for the ax and figure faces (backgrounds).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">invert_plot_order</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 Whether to invert the plot order, e.g. if using an inverse colour map.</div>
</div>


</div>



</section>
