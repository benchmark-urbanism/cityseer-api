---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# rustalgos


<div class="class">


## Coord



 Class representing a coordinate.



<div class="function">

## Coord


<div class="content">
<span class="name">Coord</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Creates a `Coord` with `x` and `y` coordinates.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">`x`</div>
    <div class="type">x coordinate.</div>
  </div>
  <div class="desc">


</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">`y`</div>
    <div class="type">y coordinate.</div>
  </div>
  <div class="desc">


</div>
</div>


</div>

 

<div class="function">

## xy


<div class="content">
<span class="name">xy</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Returns the `Coord` as a `tuple` of `x` and `y`.
### Returns
<div class="param-set">
  <div class="def">
    <div class="name">`xy`</div>
    <div class="type">tuple[float, float]</div>
  </div>
  <div class="desc">


</div>
</div>


</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>


 Validates the Coord.

</div>

 

<div class="function">

## hypot


<div class="content">
<span class="name">hypot</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">other_coord</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Returns the pythagorean distance from this `Coord` to another.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">`other_coord`</div>
    <div class="type">Coord</div>
  </div>
  <div class="desc">

 The other coordinate to which to compute the Pythagorean distance.</div>
</div>


</div>

 

<div class="function">

## difference


<div class="content">
<span class="name">difference</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">other_coord</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Coord</span>
  <span class="pt">]</span>
</div>
</div>


 Returns the vector of the spatial difference between this `Coord` and another.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">`other_coord`</div>
    <div class="type">Coord</div>
  </div>
  <div class="desc">

 The other coordinate to which to compute the Pythagorean distance.</div>
</div>


</div>

 
</div>


<div class="function">

## calculate_rotation


<div class="content">
<span class="name">calculate_rotation</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">point_a</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <div class="param">
    <span class="pn">point_b</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>

</div>


<div class="function">

## calculate_rotation_smallest


<div class="content">
<span class="name">calculate_rotation_smallest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">vec_a</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <div class="param">
    <span class="pn">vec_b</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.Coord</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>


 Calculates the angle between `vec_a` and `vec_b`.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">`vec_a`</div>
    <div class="type">Coord</div>
  </div>
  <div class="desc">

 The vector of `vec_a`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">`vec_b`</div>
    <div class="type">Coord</div>
  </div>
  <div class="desc">

 The vector of `vec_b`.</div>
</div>


</div>


<div class="function">

## check_numerical_data


<div class="content">
<span class="name">check_numerical_data</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_arr</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Checks the integrity of a numerical data array. data_arr: list[float]

</div>


<div class="function">

## distances_from_betas


<div class="content">
<span class="name">distances_from_betas</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int</span>
  <span class="pt">]</span>
</div>
</div>


 Map distance thresholds $d_{max}$ to equivalent decay parameters $\beta$ at the specified cutoff weight $w_{min}$. See [`distance_from_beta`](#distance-from-beta) for additional discussion.

:::note
It is generally not necessary to utilise this function directly.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distance</div>
    <div class="type">int | tuple[int]</div>
  </div>
  <div class="desc">

 $d_{max}$ value/s to convert to decay parameters $\beta$.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 A numpy array of decay parameters $\beta$.</div>
</div>

### Notes

```python
from cityseer.metrics import networks
# a list of betas
distances = [400, 200]
# convert to betas
betas = networks.beta_from_distance(distances)
print(betas)  # prints: array([0.01, 0.02])
```

 Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using the `distances` parameter, then this function will be called in order to extrapolate the decay parameters implicitly, using:

 $$\beta = -\frac{log(w_{min})}{d_{max}}$$

 The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ parameters, for example:

| $d_{max}$ | $\beta$ |
|:---------:|:-------:|
| 200m | 0.02 |
| 400m | 0.01 |
| 800m | 0.005 |
| 1600m | 0.0025 |

</div>


<div class="function">

## betas_from_distances


<div class="content">
<span class="name">betas_from_distances</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[float</span>
  <span class="pt">]</span>
</div>
</div>


 Map decay parameters $\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.
:::note
It is generally not necessary to utilise this function directly.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 $\beta$ value/s to convert to distance thresholds $d_{max}$.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float | None</div>
  </div>
  <div class="desc">

 An optional cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 A list of distance thresholds $d_{max}$.</div>
</div>

### Notes

```python
from cityseer import rustalgos
# a list of betas
betas = [0.01, 0.02]
# convert to distance thresholds
d_max = rustalgos.distances_from_betas(betas)
print(d_max)
# prints: [400, 200]
```

 Weighted measures such as the gravity index, weighted betweenness, and weighted land-use accessibilities are computed using a negative exponential decay function in the form of:

 $$weight = exp(-\beta \cdot distance)$$

 The strength of the decay is controlled by the $\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances. For example, if $\beta=0.005$ were to represent a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at 13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity; thus, once a sufficiently small weight is encountered it becomes computationally expensive to consider locations any farther away. The minimum weight at which this cutoff occurs is represented by $w_{min}$, and the corresponding maximum distance threshold by $d_{max}$.

![Example beta decays](/images/betas.png)

 Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using the `betas` parameter, then this function will be called in order to extrapolate the distance thresholds implicitly, using:

 $$d_{max} = \frac{log(w_{min})}{-\beta}$$

 The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking thresholds, for example:

| $\beta$ | $d_{max}$ |
|:-------:|:---------:|
| 0.02 | 200m |
| 0.01 | 400m |
| 0.005 | 800m |
| 0.0025 | 1600m |

Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly, for example:

| $\beta$ | $w_{min}$ | $d_{max}$ |
|:-------:|:---------:|:---------:|
| 0.02 | 0.01 | 230m |
| 0.01 | 0.01 | 461m |
| 0.005 | 0.01 | 921m |
| 0.0025 | 0.01 | 1842m |

</div>


<div class="function">

## pair_distances_and_betas


<div class="content">
<span class="name">pair_distances_and_betas</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int]</span>
  <span class="pr">list[float</span>
  <span class="pt">]</span>
</div>
</div>


 Pair distances and betas, where one or the other parameter is provided.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int] | tuple[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">tuple[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
</div>

### Notes

:::warning
Networks should be buffered according to the largest distance threshold that will be used for analysis. This
protects nodes near network boundaries from edge falloffs. Nodes outside the area of interest but within these
buffered extents should be set to 'dead' so that centralities or other forms of measures are not calculated.
Whereas metrics are not calculated for 'dead' nodes, they can still be traversed by network analysis algorithms
when calculating shortest paths and landuse accessibilities.
:::

</div>


<div class="function">

## avg_distances_for_betas


<div class="content">
<span class="name">avg_distances_for_betas</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[float</span>
  <span class="pt">]</span>
</div>
</div>


 Calculate the mean distance for a given $\beta$ parameter.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">beta</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 $\beta$ representing a spatial impedance / distance decay for which to compute the average walking distance.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 The average walking distance for a given $\beta$.</div>
</div>

### Notes

```python
from cityseer.metrics import networks
import numpy as np

distances = [100, 200, 400, 800, 1600]
print('distances', distances)
# distances [ 100  200  400  800 1600]

betas = networks.beta_from_distance(distances)
print('betas', betas)
# betas [0.04   0.02   0.01   0.005  0.0025]

print('avg', networks.avg_distance_for_beta(betas))
# avg [ 35.11949  70.23898 140.47797 280.95593 561.91187]
```


</div>


<div class="function">

## clip_wts_curve


<div class="content">
<span class="name">clip_wts_curve</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[float</span>
  <span class="pt">]</span>
</div>
</div>


 Calculate the upper bounds for clipping weights produced by spatial impedance functions. Determine the upper weights threshold of the distance decay curve for a given $\beta$ based on the `spatial_tolerance` parameter. This is used by downstream functions to determine the upper extent at which weights derived for spatial impedance functions are flattened and normalised. This functionality is only intended for situations where the location of datapoints is uncertain for a given spatial tolerance.

:::warning
Use distance based clipping with caution for smaller distance thresholds. For example, if using a 200m distance
threshold clipped by 100m, then substantial distortion is introduced by the process of clipping and normalising the
distance decay curve. More generally, smaller distance thresholds should generally be avoided for situations where
datapoints are not located with high spatial precision.
:::
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">tuple[int]</div>
  </div>
  <div class="desc">

 An array of distances corresponding to the local $d_{max}$ thresholds to be used for calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 An array of $\beta$ to be used for the exponential decay function for weighted metrics.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">spatial_tolerance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The spatial buffer distance corresponding to the tolerance for spatial inaccuracy.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">max_curve_wts</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 An array of maximum weights at which curves for corresponding $\beta$ will be clipped.</div>
</div>


</div>


<div class="function">

## clipped_beta_wt


<div class="content">
<span class="name">clipped_beta_wt</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">beta</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">max_curve_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">data_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>

</div>


<div class="class">


## DataEntry




<div class="function">

## DataEntry


<div class="content">
<span class="name">DataEntry</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">data_id</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">nearest_assign</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">next_nearest_assign</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## is_assigned


<div class="content">
<span class="name">is_assigned</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## DataMap




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
<span class="name">progress</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
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
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">data_id</span>
    <span class="pc">:</span>
    <span class="pa"> str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">nearest_assign</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">next_nearest_assign</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 data_key: str The key for the added node. data_x: float The x coordinate for the added node. data_y: float The y coordinate for the added node. data_id: str | None An optional key for each datapoint. Used for deduplication.

</div>

 

<div class="function">

## entry_keys


<div class="content">
<span class="name">entry_keys</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[str</span>
  <span class="pt">]</span>
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
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">DataEntry | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## get_data_coord


<div class="content">
<span class="name">get_data_coord</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Coord | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## count


<div class="content">
<span class="name">count</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## is_empty


<div class="content">
<span class="name">is_empty</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## all_assigned


<div class="content">
<span class="name">all_assigned</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## none_assigned


<div class="content">
<span class="name">none_assigned</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## set_nearest_assign


<div class="content">
<span class="name">set_nearest_assign</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">assign_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## set_next_nearest_assign


<div class="content">
<span class="name">set_next_nearest_assign</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">assign_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

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
    <span class="pn">netw_src_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">dict[str</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">landuses_map</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str]</span>
  </div>
  <div class="param">
    <span class="pn">accessibility_keys</span>
    <span class="pc">:</span>
    <span class="pa"> list[str]</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">dict[str</span>
  <span class="pr">AccessibilityResult</span>
  <span class="pt">]</span>
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
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">landuses_map</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, str]</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_hill_weighted</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_shannon</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = False</span>
  </div>
  <div class="param">
    <span class="pn">compute_gini</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = False</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">MixedUsesResult</span>
  <span class="pt">]</span>
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
    <span class="pn">network_structure</span>
    <span class="pc">:</span>
    <span class="pa"> cityseer.rustalgos.NetworkStructure</span>
  </div>
  <div class="param">
    <span class="pn">numerical_map</span>
    <span class="pc">:</span>
    <span class="pa"> dict[str, float]</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">angular</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <div class="param">
    <span class="pn">spatial_tolerance</span>
    <span class="pc">:</span>
    <span class="pa"> int | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">StatsResult</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## AccessibilityResult




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


<div class="function">

## hill_diversity


<div class="content">
<span class="name">hill_diversity</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">class_counts</span>
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">class_distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">max_curve_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">class_distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[float]</span>
  </div>
  <div class="param">
    <span class="pn">q</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">max_curve_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
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
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">wt_matrix</span>
    <span class="pc">:</span>
    <span class="pa"> list[list[float]]</span>
  </div>
  <div class="param">
    <span class="pn">alpha</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">beta</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pt">]</span>
</div>
</div>

</div>


<div class="class">


## NodePayload




<div class="function">

## NodePayload


<div class="content">
<span class="name">NodePayload</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## EdgePayload




<div class="function">

## EdgePayload


<div class="content">
<span class="name">EdgePayload</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## NetworkStructure




<div class="function">

## NetworkStructure


<div class="content">
<span class="name">NetworkStructure</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## dijkstra_tree_shortest


<div class="content">
<span class="name">dijkstra_tree_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">src_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int]</span>
  <span class="pr">NodeVisit</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## dijkstra_tree_simplest


<div class="content">
<span class="name">dijkstra_tree_simplest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">src_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int]</span>
  <span class="pr">NodeVisit</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## dijkstra_tree_segment


<div class="content">
<span class="name">dijkstra_tree_segment</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">src_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int]</span>
  <span class="pr">list[int]</span>
  <span class="pr">NodeVisit]</span>
  <span class="pr">EdgeVisit</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## local_node_centrality_shortest


<div class="content">
<span class="name">local_node_centrality_shortest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CentralityShortestResult</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## local_node_centrality_simplest


<div class="content">
<span class="name">local_node_centrality_simplest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CentralitySimplestResult</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## local_segment_centrality


<div class="content">
<span class="name">local_segment_centrality</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">betas</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">compute_closeness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">compute_betweenness</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = True</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">jitter_scale</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">CentralitySegmentResult</span>
  <span class="pt">]</span>
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
<span class="name">progress</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## add_node


<div class="content">
<span class="name">add_node</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">node_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">live</span>
    <span class="pc">:</span>
    <span class="pa"> bool</span>
  </div>
  <div class="param">
    <span class="pn">weight</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">node_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 The node key as `str`.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">x</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The node's `x` coordinate.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">y</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The node's `y` coordinate.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">live</div>
    <div class="type">bool</div>
  </div>
  <div class="desc">

 The `live` node attribute identifying if this node falls within the areal boundary of interest as opposed to those that fall within the surrounding buffered area. See the [edge-rolloff](/guide#edge-rolloff) section in the guide.</div>
</div>


</div>

 

<div class="function">

## get_node_payload


<div class="content">
<span class="name">get_node_payload</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">NodePayload</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## get_node_weight


<div class="content">
<span class="name">get_node_weight</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">/</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
  </div>
  <span class="pt">)</span>
</div>
</div>

</div>

 

<div class="function">

## is_node_live


<div class="content">
<span class="name">is_node_live</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">node_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## node_count


<div class="content">
<span class="name">node_count</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## node_indices


<div class="content">
<span class="name">node_indices</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[int</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## add_edge


<div class="content">
<span class="name">add_edge</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_key</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">length</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">angle_sum</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">imp_factor</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">in_bearing</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">out_bearing</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>


 Add an edge to the `NetworkStructure`. Edges are directed, meaning that each bidirectional street is represented twice: once in each direction; start/end nodes and in/out bearings will differ accordingly.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">start_node_idx</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Node index for the starting node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">end_node_idx</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Node index for the ending node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">edge_idx</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 The edge index, such that multiple edges can span between the same node pair.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">start_node_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Node key for the starting node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">end_node_key</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 Node key for the ending node.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">length</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The `length` edge attribute should always correspond to the edge lengths in metres. This is used when calculating the distances traversed by the shortest-path algorithm so that the respective $d_{max}$ maximum distance thresholds can be enforced: these distance thresholds are based on the actual network-paths traversed by the algorithm as opposed to crow-flies distances.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">angle_sum</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The `angle_sum` edge bearing should correspond to the total angular change along the length of the segment. This is used when calculating angular impedances for simplest-path measures. The `in_bearing` and `out_bearing` attributes respectively represent the starting and ending bearing of the segment. This is also used when calculating simplest-path measures when the algorithm steps from one edge to another.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">imp_factor</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The `imp_factor` edge attribute represents an impedance multiplier for increasing or diminishing the impedance of an edge. This is ordinarily set to 1, therefore not impacting calculations. By setting this to greater or less than 1, the edge will have a correspondingly higher or lower impedance. This can be used to take considerations such as street gradients into account, but should be used with caution.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">in_bearing</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The edge's inwards angular bearing.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">out_bearing</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The edge's outwards angular bearing.</div>
</div>


</div>

 

<div class="function">

## edge_references


<div class="content">
<span class="name">edge_references</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">ist[tuple[int</span>
  <span class="pr">int</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## get_edge_payload


<div class="content">
<span class="name">get_edge_payload</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">start_nd_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">end_nd_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">edge_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">EdgePayload</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## validate


<div class="content">
<span class="name">validate</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">bool</span>
  <span class="pt">]</span>
</div>
</div>


 Validate Network Structure.

</div>

 

<div class="function">

## find_nearest


<div class="content">
<span class="name">find_nearest</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int | None</span>
  <span class="pr">float</span>
  <span class="pr">int | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## road_distance


<div class="content">
<span class="name">road_distance</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">nd_a_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">nd_b_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pr">int | None</span>
  <span class="pr">int | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## closest_intersections


<div class="content">
<span class="name">closest_intersections</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">pred_map</span>
    <span class="pc">:</span>
    <span class="pa"> list[int | None]</span>
  </div>
  <div class="param">
    <span class="pn">last_nd_idx</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">float</span>
  <span class="pr">int | None</span>
  <span class="pr">int | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<div class="function">

## assign_to_network


<div class="content">
<span class="name">assign_to_network</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <div class="param">
    <span class="pn">data_coord</span>
  </div>
  <div class="param">
    <span class="pn">max_dist</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int | None</span>
  <span class="pr">int | None</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 

<span class="name">node_xys</span><span class="annotation">: list[tuple[float, float]]</span>


 

<span class="name">node_ys</span><span class="annotation">: list[float]</span>


 

<span class="name">node_xs</span><span class="annotation">: list[float]</span>


 

<span class="name">node_lives</span><span class="annotation">: list[bool]</span>


 
</div>


<div class="class">


## CentralityShortestResult




<div class="function">

## CentralityShortestResult


<div class="content">
<span class="name">CentralityShortestResult</span><div class="signature">
  <span class="pt">(</span>
  <span class="pt">)</span>
</div>
</div>

</div>

 
</div>


<div class="class">


## CentralitySimplestResult




<div class="function">

## CentralitySimplestResult


<div class="content">
<span class="name">CentralitySimplestResult</span><div class="signature">
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

</div>

 

<div class="function">

## progress


<div class="content">
<span class="name">progress</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">int</span>
  <span class="pt">]</span>
</div>
</div>

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
    <span class="pn">bldgs_rast</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Any]]</span>
  <span class="pr">Any]]]</span>
  <span class="pr">bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pr">Union[bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes]]]</span>
  <span class="pr">Any]]</span>
  <span class="pr">Any]]]</span>
  <span class="pr">bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pr">Union[bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes]]]</span>
  <span class="pr">Any]]</span>
  <span class="pr">Any]]]</span>
  <span class="pr">bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pr">Union[bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pt">]</span>
</div>
</div>

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
    <span class="pn">bldgs_rast</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">origin_x</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">origin_y</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">pbar_disabled</span>
    <span class="pc">:</span>
    <span class="pa"> bool = False</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">dtype[Any]]</span>
  <span class="pr">dtype[Any]]]</span>
  <span class="pr">bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pr">_NestedSequence[Union[bool</span>
  <span class="pr">int</span>
  <span class="pr">float</span>
  <span class="pr">complex</span>
  <span class="pr">str</span>
  <span class="pr">bytes</span>
  <span class="pt">]</span>
</div>
</div>

</div>

 
</div>



</section>
