# Table of Contents

* [cityseer.tools.plot](#cityseer.tools.plot)
  * [plot\_nX\_primal\_or\_dual](#cityseer.tools.plot.plot_nX_primal_or_dual)
  * [plot\_nX](#cityseer.tools.plot.plot_nX)
  * [plot\_assignment](#cityseer.tools.plot.plot_assignment)
  * [plot\_graph\_maps](#cityseer.tools.plot.plot_graph_maps)

---
sidebar_label: plot
title: cityseer.tools.plot
---

Convenience methods for plotting graphs within the cityseer API context. This module is predominately used for basic plots or visual verification of behaviour in code tests. Custom behaviour can be achieved by directly manipulating the underlying [`NetworkX`](https://networkx.github.io) and [`matplotlib`](https://matplotlib.org) figures.

<a name="cityseer.tools.plot.plot_nX_primal_or_dual"></a>
#### plot\_nX\_primal\_or\_dual

<FuncSignature>

plot_nX_primal_or_dual(primal_graph = None, **figure_kwargs)

</FuncSignature>

**Arguments**:

<FuncElement name="param1" type="int">The first parameter.</FuncElement>
<FuncElement name="param2" type=":obj:`str`, optional">The second parameter. Defaults to None.</FuncElement>
  Second line of description should be indented.
<FuncElement name="*args">Variable length argument list.</FuncElement>
<FuncElement name="**kwargs">Arbitrary keyword arguments.</FuncElement>
  

**Returns**:

<FuncElement name="bool">True if successful, False otherwise.</FuncElement>
  
  The return type is optional and may be specified at the beginning of
  the ``Returns`` section followed by a colon.
  
  The ``Returns`` section may span multiple lines and paragraphs.
  Following lines should be indented to match the first line.
  
  The ``Returns`` section supports any reStructuredText formatting,
  including literal blocks::
  
  {
<FuncElement name="'param1'">param1,</FuncElement>
<FuncElement name="'param2'">param2</FuncElement>
  }
  

**Raises**:

<FuncElement name="AttributeError">The ``Raises`` section is a list of all exceptions</FuncElement>
  that are relevant to the interface.
<FuncElement name="ValueError">If `param2` is equal to `param1`.</FuncElement>

<a name="cityseer.tools.plot.plot_nX"></a>
#### plot\_nX

<FuncSignature>

plot_nX(networkX_graph = None, **figure_kwargs)

</FuncSignature>

Convenience method for plotting a primal graph.

<a name="cityseer.tools.plot.plot_assignment"></a>
#### plot\_assignment

<FuncSignature>

plot_assignment(Network_Layer, Data_Layer, path = None, **figure_kwargs)

</FuncSignature>

Plots POI assigments to the network. Mostly used for debugging.

<a name="cityseer.tools.plot.plot_graph_maps"></a>
#### plot\_graph\_maps

<FuncSignature>

plot_graph_maps(node_data = None)

</FuncSignature>

Plots node and edge data maps. Mostly used for debugging.

