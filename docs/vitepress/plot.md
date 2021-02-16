# cityseer.util.plot API documentation

<main class="pdoc"><section>cityseer.util.plot<div class="docstring"><p>These plot methods are mainly for testing and debugging</p>
</div>
</section><section id="plot_nX_primal_or_dual"><div class="attr function"><a class="headerlink" href="#plot_nX_primal_or_dual">#&nbsp;&nbsp</a>

<span class="def">def</span>
<span class="name">plot_nX_primal_or_dual</span><span class="signature">(
primal_graph: networkx.classes.multigraph.MultiGraph = None,
dual_graph: networkx.classes.multigraph.MultiGraph = None,
path: str = None,
labels: bool = False,
primal_node_colour: (&lt;class &#39;str&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
primal_edge_colour: str = None,
dual_node_colour: (&lt;class &#39;str&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
dual_edge_colour: str = None,
primal_edge_width: (&lt;class &#39;int&#39;&gt;, &lt;class &#39;float&#39;&gt;) = None,
dual_edge_width: (&lt;class &#39;int&#39;&gt;, &lt;class &#39;float&#39;&gt;) = None,
plot_geoms: bool = True,
x_lim: (&lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
y_lim: (&lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
\*\*figure_kwargs
)</span>:

</div>

<div class="docstring"><p>Plots either or both a primal or dual MultiGraph</p>
</div>

</section><section id="plot_nX"><div class="attr function"><a class="headerlink" href="#plot_nX">#&nbsp;&nbsp</a>

<span class="def">def</span>
<span class="name">plot_nX</span><span class="signature">(
networkX_graph: networkx.classes.multigraph.MultiGraph,
path: str = None,
labels: bool = False,
node_colour: (&lt;class &#39;str&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
edge_colour: (&lt;class &#39;str&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
edge_width: (&lt;class &#39;int&#39;&gt;, &lt;class &#39;float&#39;&gt;) = None,
plot_geoms: bool = False,
x_lim: (&lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
y_lim: (&lt;class &#39;tuple&#39;&gt;, &lt;class &#39;list&#39;&gt;) = None,
\*\*kwargs
)</span>:

</div>

<div class="docstring"><p>Convenience method for plotting a primal graph.</p>
</div>

</section><section id="plot_assignment"><div class="attr function"><a class="headerlink" href="#plot_assignment">#&nbsp;&nbsp</a>

<span class="def">def</span>
<span class="name">plot_assignment</span><span class="signature">(
Network_Layer,
Data_Layer,
path: str = None,
node_colour: (&lt;class &#39;list&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;numpy.ndarray&#39;&gt;) = None,
node_labels: bool = False,
data_labels: (&lt;class &#39;list&#39;&gt;, &lt;class &#39;tuple&#39;&gt;, &lt;class &#39;numpy.ndarray&#39;&gt;) = None,
\*\*kwargs
)</span>:

</div>

<div class="docstring"><p>Plots POI assigments to the network. Mostly used for debugging.</p>
</div>

</section><section id="plot_graph_maps"><div class="attr function"><a class="headerlink" href="#plot_graph_maps">#&nbsp;&nbsp</a>

<span class="def">def</span>
<span class="name">plot_graph_maps</span><span class="signature">(
node_data: numpy.ndarray,
edge_data: numpy.ndarray,
data_map: numpy.ndarray = None,
poly: shapely.geometry.polygon.Polygon = None
)</span>:

</div>

<div class="docstring"><p>Plots node and edge data maps. Mostly used for debugging.</p>
</div>

</section></main>
