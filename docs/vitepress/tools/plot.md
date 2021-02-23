# cityseer.tools.plot
### plot_nX_primal_or_dual

<FuncSignature>
<pre>
plot_nX_primal_or_dual(primal_graph = None,                       dual_graph = None,                       path = None,                       labels = False,                       primal_node_colour = None,                       primal_edge_colour = None,                       dual_node_colour = None,                       dual_edge_colour = None,                       primal_edge_width = None,                       dual_edge_width = None,                       plot_geoms = True,                       x_lim = None,                       y_lim = None,                       **figure_kwargs)
</pre>
</FuncSignature>

Plot either or both primal and dual representations of a `networkX MultiGraph`. Only call this function directly if explicitly printing both primal and dual graphs. Otherwise, use the simplified [`plot_nX()`](plot#plot-nx) method instead.<FuncHeading>Arguments</FuncHeading>
<FuncElement name="figure_kwargs" type="None">
 An optional `NetworkX` MultiGraph to plot in the primal representation. Defaults to None.
 An optional `NetworkX` MultiGraph to plot in the dual representation. Defaults to None.
 An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to None.
 Whether to display node labels. Defaults to False.
 Primal node colour or colours. When passing an iterable of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx) method and should be formatted accordingly. Defaults to None.
 Primal edge colour as a `matplotlib` compatible colour string. Defaults to None.
 Dual node colour or colours. When passing a list of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx) method and should be formatted accordingly. Defaults to None.
 Dual edge colour as a `matplotlib` compatible colour string. Defaults to None.
 Linewidths for the primal edge. Defaults to None.
 Linewidths for the dual edge. Defaults to None.
 Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to represent edges. Defaults to True.
 A tuple or list with the minimum and maxium `x` extents to be plotted. Defaults to None.
 A tuple or list with the minimum and maxium `y` extents to be plotted. Defaults to None.
 `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the default figure size or dpi parameters.
</FuncElement>
<FuncHeading>Example</FuncHeading>
```pyfrom cityseer.tools import mock, graphs, plotG = mock.mock_graph()G_simple = graphs.nX_simple_geoms(G)G_dual = graphs.nX_to_dual(G_simple)plot.plot_nX_primal_or_dual(G_simple,                            G_dual,                            plot_geoms=False)### plot_nX

<FuncSignature>
<pre>
plot_nX(networkX_graph,        path = None,        labels = False,        node_colour = None,        edge_colour = None,        edge_width = None,        plot_geoms = False,        x_lim = None,        y_lim = None,        **figure_kwargs)
</pre>
</FuncSignature>

Plot a `networkX` MultiGraph.<FuncHeading>Arguments</FuncHeading>
<FuncElement name="figure_kwargs" type="None">
 A `NetworkX` MultiGraph.
 An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to None.
 Whether to display node labels. Defaults to False.
 Node colour or colours. When passing an iterable of colours, the number of colours should match the order and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx) method and should be formatted accordingly. Defaults to None.
 Edges colour as a `matplotlib` compatible colour string. Defaults to None.
 Linewidths for edges. Defaults to None.
 Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to represent edges. Defaults to True.
 A tuple or list with the minimum and maxium `x` extents to be plotted. Defaults to None.
 A tuple or list with the minimum and maxium `y` extents to be plotted. Defaults to None.
 `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the default figure size or dpi parameters.
</FuncElement>
<FuncHeading>Example</FuncHeading>
