---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# decay


 Functions for generating decay expression strings. Decay expressions control how feature importance decreases with distance from an analysis point. They use a single variable `p` that ranges from 0 at the analysis point (source) to 1 at the distance threshold (cutoff), where `p = network_distance / max_distance`. The functions below produce expression strings that can be passed directly to the `decay_fn` parameter of [`betweenness_demand`](/metrics/networks#betweenness-demand), [`compute_stats`](/metrics/layers#compute-stats), [`compute_accessibilities`](/metrics/layers#compute-accessibilities), and other analysis functions.

```python
from cityseer import decay
from cityseer.metrics import layers

nodes_gdf, data_gdf = layers.compute_stats(
    ...,
    distances=[1200],
    decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
)
```

 Available presets:

```text
exponential (default)       gaussian(peak, cutoff)        linear
╷                           ╷       ╭─╮                   ╷╲
│╲                          │      ╱   ╲                  │ ╲
│ ╲                         │     ╱     ╲                 │  ╲
│  ╲                        │    ╱       ╲                │   ╲
│   ╲                       │   ╱         ╲               │    ╲
│    ╰─╮                    │  ╱           ╲              │     ╲
│      ╰───╮                │ ╱             ╰╮            │      ╲
│          ╰────────        │╱               ╰───         │       ╲
╵───────────────────        ╵───────────────────          ╵────────╳
p=0                p=1      p=0                p=1        p=0      p=1

logistic(midpoint, cutoff)        flat (no decay)
╷──────╮                          ╷───────────────
│      ╲                          │
│       ╲                         │
│        │                        │
│        ╲                        │
│         ╲                       │
│          ╰╮                     │
│           ╰────────             │
╵───────────────────              ╵───────────────
p=0                p=1            p=0            p=1
```



<div class="function">

## exponential


<div class="content">
<span class="name">exponential</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">steepness</span>
    <span class="pc">:</span>
    <span class="pa"> float = 4.0</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pt">]</span>
</div>
</div>


 Exponential decay: weight = exp(-steepness * p). At p=1 (the cutoff), weight = exp(-steepness). The default steepness of 4 gives ~1.8% weight at the cutoff boundary, matching cityseer's historical default.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">steepness</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Controls how quickly weight decays. Higher values produce steeper decay. Common values: 2 (~13.5% at cutoff), 4 (~1.8%), 6 (~0.25%).</div>
</div>


</div>


<div class="function">

## linear


<div class="content">
<span class="name">linear</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pt">]</span>
</div>
</div>


 Linear decay from 1 at the source to 0 at the cutoff.

</div>


<div class="function">

## flat


<div class="content">
<span class="name">flat</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pt">]</span>
</div>
</div>


 No decay: constant weight of 1 everywhere within the cutoff.

</div>


<div class="function">

## gaussian


<div class="content">
<span class="name">gaussian</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">peak</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">cutoff</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">std</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pt">]</span>
</div>
</div>


 Gaussian bell curve peaking at a specified location.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">peak</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The point at which the weight is highest, in the same units as ``cutoff`` (e.g. metres or minutes).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cutoff</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The cutoff threshold in the same units as ``peak`` (must match the ``distances`` or ``minutes`` parameter).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">std</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Standard deviation controlling the width of the bell curve, in the same units as ``peak``/``cutoff``. If not provided, defaults to ``peak / 2``.</div>
</div>


</div>


<div class="function">

## logistic


<div class="content">
<span class="name">logistic</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">midpoint</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">cutoff</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">rate</span>
    <span class="pc">:</span>
    <span class="pa"> float = 0.05</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">str</span>
  <span class="pt">]</span>
</div>
</div>


 Logistic (sigmoid) decay centred at a specified location. Weight transitions from ~1 (near source) to ~0 (beyond midpoint). The transition is centred at ``midpoint``.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">midpoint</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The point at which weight = 0.5, in the same units as ``cutoff``.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">cutoff</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The cutoff threshold in the same units as ``midpoint`` (must match the ``distances`` or ``minutes`` parameter).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">rate</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Steepness of the transition. Higher values produce a sharper step. The rate is specified in per-unit terms; internally it is scaled to normalised ``p`` coordinates.</div>
</div>


</div>



</section>
