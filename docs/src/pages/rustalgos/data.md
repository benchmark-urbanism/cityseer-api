---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# data


 Data structures and utilities for spatial data analysis.


<div class="class">


## DataEntry



 Data entry for spatial analysis.



<div class="function">

## DataEntry


<div class="content">
<span class="name">DataEntry</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## DataMap



 Map of data entries for spatial analysis.



<div class="function">

## DataMap


<div class="content">
<span class="name">DataMap</span><div class="signature">
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

</div>

 

<div class="function">

## insert


<div class="content">
<span class="name">insert</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_key_py</span>
  </div>
  <div class="param">
    <span class="pn">geom_wkt</span>
  </div>
  <div class="param">
    <span class="pn">dedupe_key_py=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## entry_keys


<div class="content">
<span class="name">entry_keys</span><div class="signature multiline">
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

</div>

 

<div class="function">

## get_entry


<div class="content">
<span class="name">get_entry</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">data_key</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## count


<div class="content">
<span class="name">count</span><div class="signature multiline">
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

</div>

 

<div class="function">

## is_empty


<div class="content">
<span class="name">is_empty</span><div class="signature multiline">
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

</div>

 

<div class="function">

## assign_data_to_network


<div class="content">
<span class="name">assign_data_to_network</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
  </div>
  <div class="param">
    <span class="pn">max_assignment_dist</span>
  </div>
  <div class="param">
    <span class="pn">n_nearest_candidates</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Assigns data entries to network nodes based on proximity and accessibility checks. This method iterates through all data entries and uses `NetworkStructure::find_assignments_for_entry` to determine valid node assignments for each entry. The results are collected and stored in the `node_data_map`.

</div>

 

<div class="function">

## aggregate_to_src_idx


<div class="content">
<span class="name">aggregate_to_src_idx</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">netw_src_idx</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
  </div>
  <div class="param">
    <span class="pn">max_walk_seconds</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">angular=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## accessibility


<div class="content">
<span class="name">accessibility</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
  </div>
  <div class="param">
    <span class="pn">landuses_map</span>
  </div>
  <div class="param">
    <span class="pn">accessibility_keys</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">angular=None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## mixed_uses


<div class="content">
<span class="name">mixed_uses</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
  </div>
  <div class="param">
    <span class="pn">landuses_map</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill_weighted=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_shannon=None</span>
  </div>
  <div class="param">
    <span class="pn">compute_gini=None</span>
  </div>
  <div class="param">
    <span class="pn">angular=None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## stats


<div class="content">
<span class="name">stats</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
  </div>
  <div class="param">
    <span class="pn">numerical_maps</span>
  </div>
  <div class="param">
    <span class="pn">distances=None</span>
  </div>
  <div class="param">
    <span class="pn">betas=None</span>
  </div>
  <div class="param">
    <span class="pn">minutes=None</span>
  </div>
  <div class="param">
    <span class="pn">angular=None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance=None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt=None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s=None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale=None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled=None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## AccessibilityResult



 Accessibility computation result.



<div class="function">

## AccessibilityResult


<div class="content">
<span class="name">AccessibilityResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## MixedUsesResult



 Mixed uses computation result.



<div class="function">

## MixedUsesResult


<div class="content">
<span class="name">MixedUsesResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## StatsResult



 Statistics computation result.



<div class="function">

## StatsResult


<div class="content">
<span class="name">StatsResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>



</section>
