---
layout: ../../layouts/PageLayout.astro
---

# mock


 A collection of functions for the generation of mock data. This module is intended for project development and writing code tests, but may otherwise be useful for demonstration and utility purposes.


<div class="function">

## mock_graph


<div class="content">
<span class="name">mock_graph</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">wgs84_coords</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span></span><span class="return-annotation">) -> <span class="n">Any</span>:</span></span>
</div>


 Generate a `NetworkX` `MultiGraph` for testing or experimentation purposes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">wgs84_coords</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 If set to `True`, the `x` and `y` attributes will be in [WGS84](https://epsg.io/4326) geographic coordinates instead of a projected cartesion coordinate system.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` `MultiGraph` with `x` and `y` node attributes.</div>
</div>

### Notes

```python
from cityseer.tools import mock, plot
nx_multigraph = mock.mock_graph()
plot.plot_nx(nx_multigraph)
```


![Example graph](/images/graph_example.png) _Mock graph._

</div>


<div class="function">

## get_graph_extents


<div class="content">
<span class="name">get_graph_extents</span><span class="signature pdoc-code condensed">(<span class="param"><span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span></span><span class="return-annotation">) -> <span class="nb">tuple</span><span class="p">[</span><span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">,</span> <span class="nb">float</span><span class="p">]</span>:</span></span>
</div>


 Derive geographic bounds for a given networkX graph.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` `MultiGraph` with `x` and `y` node parameters.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">min_x</div>
    <div class="type">float</div>
  </div>
  <div class="desc">


</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_y</div>
    <div class="type">float</div>
  </div>
  <div class="desc">


</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_x</div>
    <div class="type">float</div>
  </div>
  <div class="desc">


</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_y</div>
    <div class="type">float</div>
  </div>
  <div class="desc">


</div>
</div>


</div>


<div class="function">

## mock_data_gdf


<div class="content">
<span class="name">mock_data_gdf</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">length</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span>,</span><span class="param">	<span class="n">random_seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span></span><span class="return-annotation">) -> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>:</span></span>
</div>


 Generate a `GeoDataFrame` containing mock data for testing or experimentation purposes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the network. The returned data will be within these extents.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">length</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of data elements to return in the `GeoDataFrame`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional random seed.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with data points for testing purposes.</div>
</div>


</div>


<div class="function">

## mock_landuse_categorical_data


<div class="content">
<span class="name">mock_landuse_categorical_data</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">length</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span>,</span><span class="param">	<span class="n">num_classes</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">10</span>,</span><span class="param">	<span class="n">random_seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span></span><span class="return-annotation">) -> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>:</span></span>
</div>


 Generate a `numpy` array containing mock categorical data for testing or experimentation purposes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the network. The returned data will be within these extents.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">length</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of categorical elements to return in the array.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">num_classes</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The maximum number of unique classes to return in the randomly assigned categorical data. The classes are randomly generated from a pool of unique class labels of length `num_classes`. The number of returned unique classes will be less than or equal to `num_classes`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional random seed.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with a &quot;categorical_landuses&quot; data column for testing purposes. The number of rows will match the `length` parameter. The categorical data will consist of randomly selected characters from `num_classes`.</div>
</div>


</div>


<div class="function">

## mock_numerical_data


<div class="content">
<span class="name">mock_numerical_data</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">nx_multigraph</span><span class="p">:</span> <span class="n">Any</span>,</span><span class="param">	<span class="n">length</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">50</span>,</span><span class="param">	<span class="n">val_min</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span>,</span><span class="param">	<span class="n">val_max</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">100000</span>,</span><span class="param">	<span class="n">num_arrs</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1</span>,</span><span class="param">	<span class="n">floating_pt</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span>,</span><span class="param">	<span class="n">random_seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span></span><span class="return-annotation">) -> <span class="n">geopandas</span><span class="o">.</span><span class="n">geodataframe</span><span class="o">.</span><span class="n">GeoDataFrame</span>:</span></span>
</div>


 Generate a 2d `numpy` array containing mock numerical data for testing or experimentation purposes.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">nx_multigraph</div>
    <div class="type">MultiGraph</div>
  </div>
  <div class="desc">

 A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the network. The returned data will be within these extents.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">length</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of numerical elements to return in the array.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">val_min</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The (inclusive) minimum value in the `val_min`, `val_max` range of randomly generated integers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">val_max</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The (exclusive) maximum value in the `val_min`, `val_max` range of randomly generated integers.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">num_arrs</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The number of arrays to nest in the returned 2d array.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">floating_pt</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The floating point precision</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional random seed.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">GeoDataFrame</div>
  </div>
  <div class="desc">

 A `GeoDataFrame` with a &quot;mock_numerical_x&quot; data columns for testing purposes. The number of rows will match the `length` parameter. The numer of numerical columns will match the `num_arrs` paramter.</div>
</div>


</div>


<div class="function">

## mock_species_data


<div class="content">
<span class="name">mock_species_data</span><span class="signature pdoc-code multiline">(<span class="param">	<span class="n">random_seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span></span><span class="return-annotation">) -> <span class="n">Generator</span><span class="p">[</span><span class="nb">tuple</span><span class="p">[</span><span class="nb">list</span><span class="p">[</span><span class="nb">int</span><span class="p">],</span> <span class="nb">list</span><span class="p">[</span><span class="nb">float</span><span class="p">]],</span> <span class="n">NoneType</span><span class="p">,</span> <span class="n">NoneType</span><span class="p">]</span>:</span></span>
</div>


 Generate a series of randomly generated counts and corresponding probabilities. This function is used for testing diversity measures. The data is generated in varying lengths from randomly assigned integers between 1 and 10. Matching integers are then collapsed into species "classes" with probabilities computed accordingly.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">random_seed</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 An optional random seed.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">counts</div>
    <div class="type">ndarray[int]</div>
  </div>
  <div class="desc">

 The number of members for each species class.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">probs</div>
    <div class="type">ndarray[float]</div>
  </div>
  <div class="desc">

 The probability of encountering the respective species classes.</div>
</div>

### Notes

```python
from cityseer.tools import mock

for counts, probs in mock.mock_species_data():
    cs = [c for c in counts]
    print(f'c = {cs}')
    ps = [round(p, 3) for p in probs]
    print(f'p = {ps}')

# c = [1]
# p = [1.0]

# c = [1, 1, 2, 2]
# p = [0.167, 0.167, 0.333, 0.333]

# c = [3, 2, 1, 1, 1, 3]
# p = [0.273, 0.182, 0.091, 0.091, 0.091, 0.273]

# c = [3, 3, 2, 2, 1, 1, 1, 2, 1]
# p = [0.188, 0.188, 0.125, 0.125, 0.062, 0.062, 0.062, 0.125, 0.062]

# etc.
```


</div>



