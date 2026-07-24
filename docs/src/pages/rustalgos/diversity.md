---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# diversity

:::warning
**Low-level internals.** These are the Rust-backed structures and functions that power `cityseer`. They carry no stability guarantee and are subject to breaking changes from time to time. Users are encouraged to use the higher-level wrappers instead, which are more stable: the [`CityNetwork`](/api/network) class, or the [`metrics`](/metrics/networks) and [`tools`](/tools/graphs) modules. The basic information below is provided for those who do wish to work with the lower-level internals.
:::



 Functions for calculating diversity metrics in spatial analysis.


<div class="function">

## gini_simpson_diversity


<div class="content">
<span class="name">gini_simpson_diversity</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>


<div class="function">

## hill_diversity


<div class="content">
<span class="name">hill_diversity</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>


<div class="function">

## hill_diversity_branch_distance_wt


<div class="content">
<span class="name">hill_diversity_branch_distance_wt</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <div class="param">
    <span class="pn">class_distances</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
  </div>
  <div class="param">
    <span class="pn">max_curve_wt</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>


<div class="function">

## hill_diversity_pairwise_distance_wt


<div class="content">
<span class="name">hill_diversity_pairwise_distance_wt</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <div class="param">
    <span class="pn">class_distances</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
  </div>
  <div class="param">
    <span class="pn">max_curve_wt</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>


<div class="function">

## raos_quadratic_diversity


<div class="content">
<span class="name">raos_quadratic_diversity</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <div class="param">
    <span class="pn">wt_matrix</span>
  </div>
  <div class="param">
    <span class="pn">alpha</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>


<div class="function">

## shannon_diversity


<div class="content">
<span class="name">shannon_diversity</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


</div>



</section>
