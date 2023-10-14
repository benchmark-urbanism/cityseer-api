---
layout: ../../layouts/PageLayout.astro
---

# plot


 Convenience methods for plotting graphs within the cityseer API context. Custom behaviour can be achieved by directly manipulating the underlying [`NetworkX`](https://networkx.github.io) and [`matplotlib`](https://matplotlib.org) figures. This module is predominately used for basic plots or visual verification of behaviour in code tests. Users are encouraged to use matplotlib or other plotting packages directly where possible. See the demos section for examples.


<div class="class">


## ColourMap



 Specifies global colour presets.



<div class="function">

## ColourMap


<div class="content">
<span class="name">ColourMap</span><span class="signature pdoc-code condensed">()</span>
</div>

</div>

 
</div>


<div class="function">

## plot_nx_primal_or_dual


<div class="content">
<span class="name">plot_nx_primal_or_dual</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">primal_graph</span><span class="p">:</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">dual_graph</span><span class="p">:</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">path</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">labels</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">primal_node_size</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">30</span>,</span><span class="param">	<span class="n">primal_node_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">primal_edge_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">dual_node_size</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">30</span>,</span><span class="param">	<span class="n">dual_node_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">dual_edge_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">primal_edge_width</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="nb">float</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">dual_edge_width</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="nb">float</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">plot_geoms</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">x_lim</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">y_lim</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">ax</span><span class="p">:</span> <span class="n">matplotlib</span><span class="o">.</span><span class="n">axes</span><span class="o">.</span><span class="n">_axes</span><span class="o">.</span><span class="n">Axes</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="o">**</span><span class="n">kwargs</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]</span></span><span class="return-annotation">):</span></span>
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
<span class="name">plot_nx</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">path</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">labels</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">node_size</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">20</span>,</span><span class="param">	<span class="n">node_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">edge_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">edge_width</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="nb">float</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">plot_geoms</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">x_lim</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">y_lim</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">ax</span><span class="p">:</span> <span class="n">matplotlib</span><span class="o">.</span><span class="n">axes</span><span class="o">.</span><span class="n">_axes</span><span class="o">.</span><span class="n">Axes</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="o">**</span><span class="n">kwargs</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]</span></span><span class="return-annotation">):</span></span>
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

```py
from cityseer.tools import mock, graphs, plot
from cityseer.metrics import networks
from matplotlib import colors

# generate a MultiGraph and compute gravity
G = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
G = graphs.nx_decompose(G, 50)
nodes_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
networks.node_centrality(
    measures=["node_beta"], network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[800]
)
G_after = graphs.nx_from_network_structure(nodes_gdf, network_structure, G)
# let's extract and normalise the values
vals = []
for node, data in G_after.nodes(data=True):
    vals.append(data["cc_metric_node_beta_800"])
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
<span class="name">plot_assignment</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">network_structure</span><span class="p">:</span> <span class="n">NetworkStructure</span>,</span><span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">data_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>,</span><span class="param">	<span class="n">path</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">node_colour</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">node_labels</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">data_labels</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">]</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="o">**</span><span class="n">kwargs</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">typing</span><span class="o">.</span><span class="n">Any</span><span class="p">]</span></span><span class="return-annotation">):</span></span>
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

 A [`rustalgos.NetworkStructure`](/rustalgos#networkstructure) instance.</div>
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
<span class="name">plot_network_structure</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">network_structure</span><span class="p">:</span> <span class="n">NetworkStructure</span>,</span><span class="param">	<span class="n">data_gdf</span><span class="p">:</span> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>,</span><span class="param">	<span class="n">poly</span><span class="p">:</span> <span class="n">shapely</span><span class="o">.</span><span class="n">geometry</span><span class="o">.</span><span class="n">polygon</span><span class="o">.</span><span class="n">Polygon</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span></span><span class="return-annotation">):</span></span>
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

 A [`rustalgos.NetworkStructure`](/rustalgos#networkstructure) instance.</div>
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
<span class="name">plot_scatter</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">ax</span><span class="p">:</span> <span class="n">matplotlib</span><span class="o">.</span><span class="n">axes</span><span class="o">.</span><span class="n">_axes</span><span class="o">.</span><span class="n">Axes</span>,</span><span class="param">	<span class="n">xs</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">float</span><span class="p">],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]]]</span>,</span><span class="param">	<span class="n">ys</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">float</span><span class="p">],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]]]</span>,</span><span class="param">	<span class="n">vals</span><span class="p">:</span> <span class="n">Union</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]],</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_array_like</span><span class="o">.</span><span class="n">_SupportsArray</span><span class="p">[</span><span class="n">numpy</span><span class="o">.</span><span class="n">dtype</span><span class="p">[</span><span class="n">Any</span><span class="p">]]],</span> <span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">,</span> <span class="n">numpy</span><span class="o">.</span><span class="n">_typing</span><span class="o">.</span><span class="n">_nested_sequence</span><span class="o">.</span><span class="n">_NestedSequence</span><span class="p">[</span><span class="n">Union</span><span class="p">[</span><span class="nb">bool</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">complex</span><span class="p">,</span> <span class="nb">str</span><span class="p">,</span> <span class="nb">bytes</span><span class="p">]]]</span>,</span><span class="param">	<span class="n">bbox_extents</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">]</span> <span class="o">|</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span>,</span><span class="param">	<span class="n">perc_range</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="mf">0.01</span><span class="p">,</span> <span class="mf">99.99</span><span class="p">)</span>,</span><span class="param">	<span class="n">cmap_key</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;viridis&#39;</span>,</span><span class="param">	<span class="n">shape_exp</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">1</span>,</span><span class="param">	<span class="n">s_min</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.1</span>,</span><span class="param">	<span class="n">s_max</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">1</span>,</span><span class="param">	<span class="n">rasterized</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">face_colour</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;#111&#39;</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Convenience plotting function for plotting outputs from examples in demo notebooks.
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
<span class="name">plot_nx_edges</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">ax</span><span class="p">:</span> <span class="n">matplotlib</span><span class="o">.</span><span class="n">axes</span><span class="o">.</span><span class="n">_axes</span><span class="o">.</span><span class="n">Axes</span>,</span><span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">networkx</span><span class="o">.</span><span class="n">classes</span><span class="o">.</span><span class="n">multigraph</span><span class="o">.</span><span class="n">MultiGraph</span>,</span><span class="param">	<span class="n">edge_metrics_key</span><span class="p">:</span> <span class="nb">str</span>,</span><span class="param">	<span class="n">bbox_extents</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">,</span> <span class="nb">int</span><span class="p">]</span> <span class="o">|</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span>,</span><span class="param">	<span class="n">perc_range</span><span class="p">:</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="mf">0.01</span><span class="p">,</span> <span class="mf">99.99</span><span class="p">)</span>,</span><span class="param">	<span class="n">cmap_key</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;viridis&#39;</span>,</span><span class="param">	<span class="n">shape_exp</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">1</span>,</span><span class="param">	<span class="n">lw_min</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.1</span>,</span><span class="param">	<span class="n">lw_max</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mi">1</span>,</span><span class="param">	<span class="n">edge_label_key</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>,</span><span class="param">	<span class="n">colour_by_categorical</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>,</span><span class="param">	<span class="n">max_n_categorical</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10</span>,</span><span class="param">	<span class="n">rasterized</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>,</span><span class="param">	<span class="n">face_colour</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;#111&#39;</span>,</span><span class="param">	<span class="n">invert_plot_order</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span></span><span class="return-annotation">):</span></span>
</div>


 Convenience plotting function for plotting edge outputs from examples in demo notebooks.
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



