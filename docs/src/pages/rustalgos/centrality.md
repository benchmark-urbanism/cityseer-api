---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# centrality


 Centrality analysis utilities for network structures.


<div class="class">


## OdMatrix



 Sparse origin-destination weight matrix for OD-weighted centrality. Stores per-pair trip weights in a nested HashMap for O(1) lookup. Constructed once and passed to centrality functions; can be reused across calls.



<div class="function">

## OdMatrix


<div class="content">
<span class="name">OdMatrix</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## len


<div class="content">
<span class="name">len</span><div class="signature multiline">
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


 Number of non-zero OD pairs.

</div>

 

<div class="function">

## n_origins


<div class="content">
<span class="name">n_origins</span><div class="signature multiline">
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


 Number of unique origin nodes.

</div>

 
</div>


<div class="class">


## ClosenessShortestResult




<div class="function">

## ClosenessShortestResult


<div class="content">
<span class="name">ClosenessShortestResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## ClosenessSimplestResult




<div class="function">

## ClosenessSimplestResult


<div class="content">
<span class="name">ClosenessSimplestResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## BetweennessShortestResult




<div class="function">

## BetweennessShortestResult


<div class="content">
<span class="name">BetweennessShortestResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## BetweennessSimplestResult




<div class="function">

## BetweennessSimplestResult


<div class="content">
<span class="name">BetweennessSimplestResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## CentralitySegmentResult




<div class="function">

## CentralitySegmentResult


<div class="content">
<span class="name">CentralitySegmentResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>



</section>
