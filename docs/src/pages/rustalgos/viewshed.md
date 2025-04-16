---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# viewshed


 Viewshed analysis utilities for spatial visibility studies.


<div class="class">


## Viewshed




<div class="function">

## Viewshed


<div class="content">
<span class="name">Viewshed</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## progress_init


<div class="content">
<span class="name">progress_init</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Reset the progress counter to zero.

</div>

 

<div class="function">

## progress


<div class="content">
<span class="name">progress</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Get the current progress value.

</div>

 

<div class="function">

## visibility_graph


<div class="content">
<span class="name">visibility_graph</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">bldgs_rast</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute the visibility graph for the given raster and view distance.

</div>

 

<div class="function">

## viewshed


<div class="content">
<span class="name">viewshed</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">bldgs_rast</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
  </div>
  <div class="param">
    <span class="pn">origin_x</span>
  </div>
  <div class="param">
    <span class="pn">origin_y</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Compute the viewshed for a single origin cell.

</div>

 
</div>



</section>
