---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# diversity

:::warning
**Low-level internals.** This section documents the Rust-backed structures and functions that power `cityseer`. They are provided for reference. For analysis, use the higher-level wrappers instead: the [`CityNetwork`](/api/network) class, or the [`metrics`](/metrics/networks) and [`tools`](/tools/graphs) modules. Symbols on these pages are not part of the public API and may change between releases without a deprecation cycle.
:::



 Functions for calculating diversity metrics in spatial analysis.


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



</section>
