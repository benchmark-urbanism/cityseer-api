---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# rustalgos


 Common geometry and mathematical utilities.


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
    <span class="pa"> Self</span>
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
    <span class="pa"> Self</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">Self</span>
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


 Calculates the rotation angle between two points relative to the origin.

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
  <span class="pr">list[int]</span>
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
    <div class="type">list[int]</div>
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
    <div class="type">list[float]</div>
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
  <span class="pr">list[float]</span>
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

Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly.

</div>


<div class="function">

## distances_from_seconds


<div class="content">
<span class="name">distances_from_seconds</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">seconds</span>
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[int]</span>
  <span class="pt">]</span>
</div>
</div>


 Map seconds to equivalent distance thresholds $d_{max}$.
:::note
It is generally not necessary to utilise this function directly.
:::

 The default `speed_m_s` of $1.333$ yields the following $d_{max}$ walking thresholds:

| $seconds$ | $d_{max}$ |
|:-------:|:---------:|
| 300 | 400m |
| 600 | 800m |
| 1200 | 1600m |

Setting the `speed_m_s` to a higher or lower number will affect the $d_{max}$ accordingly.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">seconds</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Seconds to convert to distance thresholds $d_{max}$.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The walking speed in metres per second.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 A numpy array of distance thresholds $d_{max}$.</div>
</div>


</div>


<div class="function">

## seconds_from_distances


<div class="content">
<span class="name">seconds_from_distances</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">distances</span>
    <span class="pc">:</span>
    <span class="pa"> list[int]</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[float]</span>
  <span class="pt">]</span>
</div>
</div>


 Map distances into equivalent walking time in seconds.
:::note
It is generally not necessary to utilise this function directly.
:::

 The default `speed_m_s` of $1.33333$ yields the following walking times:

| $d_{max}$ | $seconds$ |
|:-------:|:---------:|
| 400m | 300 |
| 800m | 600 |
| 1600m | 1200 |

Setting the `speed_m_s` to a higher or lower number will affect the walking time accordingly.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distances to convert to seconds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The walking speed in metres per second.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 A numpy array of walking time in seconds.</div>
</div>


</div>


<div class="function">

## pair_distances_betas_time


<div class="content">
<span class="name">pair_distances_betas_time</span><div class="signature multiline">
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
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[int]</span>
  <span class="pr">list[float]</span>
  <span class="pr">list[float]</span>
  <span class="pt">]</span>
</div>
</div>


 Pair distances, betas, and time, where only one parameter is provided.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `betas` or `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">tuple[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distances` or `minutes` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Walking times in minutes to be used for calculations. The `distance` and `beta` parameters will be determined implicitly. If the `minutes` parameter is not provided, then the `distances` or `betas` parameters must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The default walking speed in meters per second can optionally be overridden to configure the distances covered by the respective walking times.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided, then the `beta` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not provided, then the `distance` parameter must be provided instead.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">seconds</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Walking times in seconds corresponding to the distances used for calculations.</div>
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
  <span class="pr">list[float]</span>
  <span class="pt">]</span>
</div>
</div>


 Calculate the mean distance for a given $\beta$ parameter.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">beta</div>
    <div class="type">list[float]</div>
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
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 The average walking distance for a given $\beta$.</div>
</div>

### Notes

```python
from cityseer.metrics import networks
import numpy as np

distances = [100, 200, 400, 800, 1600]
print("distances", distances)
# distances [ 100  200  400  800 1600]

betas = networks.beta_from_distance(distances)
print("betas", betas)
# betas [0.04   0.02   0.01   0.005  0.0025]

print("avg", networks.avg_distance_for_beta(betas))
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
  <span class="pr">list[float]</span>
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
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 An array of distances corresponding to the local $d_{max}$ thresholds to be used for calculations.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
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
    <div class="type">list[float]</div>
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


 Computes a clipped weight based on a beta value and maximum curve weight.

</div>



</section>
