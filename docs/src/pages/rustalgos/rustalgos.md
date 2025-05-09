---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# rustalgos


 Cityseer high-performance algorithms implemented in Rust.


<div class="function">

## check_numerical_data


<div class="content">
<span class="name">check_numerical_data</span><div class="signature">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">data_arr</span>
    <span class="pc">:</span>
    <span class="pa"> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float32]]</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Validates that all elements in a 2D numerical array are finite.
### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">ValueError</div>
  </div>
  <div class="desc">

 If any element is not finite (NaN or infinity).</div>
</div>


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


 Convert decay parameters (betas) to distance thresholds ($d_{max}$). Requires betas > 0 and sorted in strictly decreasing order. Uses a default minimum weight threshold.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 $\beta$ values (&gt; 0, strictly decreasing) to convert.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float | None</div>
  </div>
  <div class="desc">

 Optional cutoff weight $w_{min}$ (default: ~0.0183).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Corresponding distance thresholds $d_{max}$.</div>
</div>

### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">ValueError</div>
  </div>
  <div class="desc">

 If inputs are invalid (empty, non-positive, not decreasing).</div>
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


 Convert distance thresholds ($d_{max}$) to decay parameters (betas). Requires distances > 0 and sorted in strictly increasing order. Uses a default minimum weight threshold.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 $d_{max}$ values (&gt; 0, strictly increasing) to convert.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float | None</div>
  </div>
  <div class="desc">

 Optional cutoff weight $w_{min}$ (default: ~0.0183).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Corresponding decay parameters $\beta$.</div>
</div>

### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">ValueError</div>
  </div>
  <div class="desc">

 If inputs are invalid (empty, non-positive, not increasing).</div>
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
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[int]</span>
  <span class="pt">]</span>
</div>
</div>


 Convert time in seconds to distance thresholds ($d_{max}$) based on speed.
:::note
It is generally not necessary to utilise this function directly.
:::

 The default `speed_m_s` of $1.333$ yields the following $d_{max}$ walking thresholds:

| $seconds$ | $d_{max}$ |
|:-------:|:---------:|
| 300 | 400m |
| 600 | 800m |
| 1200 | 1600m |

Setting the `speed_m_s` to a higher or lower number will affect the $d_{max}$ accordingly.]
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">seconds</div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Time values in seconds.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in meters per second.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Corresponding distance thresholds $d_{max}$.</div>
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
    <span class="pa"> float</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[int]</span>
  <span class="pt">]</span>
</div>
</div>


 Convert distance thresholds ($d_{max}$) to time in seconds based on speed.
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

 Distance thresholds $d_{max}$.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Speed in meters per second.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[int]</div>
  </div>
  <div class="desc">

 Corresponding time values in seconds.</div>
</div>


</div>


<div class="function">

## pair_distances_betas_time


<div class="content">
<span class="name">pair_distances_betas_time</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">speed_m_s</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
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
    <span class="pn">minutes</span>
    <span class="pc">:</span>
    <span class="pa"> list[float] | None = None</span>
  </div>
  <div class="param">
    <span class="pn">min_threshold_wt</span>
    <span class="pc">:</span>
    <span class="pa"> float | None = None</span>
  </div>
  <span class="pt">)-&gt;[</span>
  <span class="pr">list[int]</span>
  <span class="pr">list[float]</span>
  <span class="pr">list[int]</span>
  <span class="pt">]</span>
</div>
</div>


 Calculate distances, betas, and seconds, given exactly one of them. Requires exactly one of `distances`, `betas`, or `minutes` to be provided.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">speed_m_s</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 Walking speed in meters per second.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">distances</div>
    <div class="type">list[int] | None</div>
  </div>
  <div class="desc">

 Distance thresholds ($d_{max}$).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float] | None</div>
  </div>
  <div class="desc">

 Decay parameters ($\beta$).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">minutes</div>
    <div class="type">list[float] | None</div>
  </div>
  <div class="desc">

 Time in minutes.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float | None</div>
  </div>
  <div class="desc">

 Optional cutoff weight $w_{min}$ for conversions.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">tuple[list[int], list[float], list[int]]</div>
  </div>
  <div class="desc">

 A tuple containing (distances, betas, seconds).</div>
</div>

### Raises
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">ValueError</div>
  </div>
  <div class="desc">

 If not exactly one of `distances`, `betas`, `minutes` is provided, or if inputs are invalid.</div>
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


 Calculate the mean distance corresponding to given beta parameters.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 $\beta$ parameters.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">min_threshold_wt</div>
    <div class="type">float | None</div>
  </div>
  <div class="desc">

 Optional cutoff weight $w_{min}$.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 The average walking distance for each beta.</div>
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


 Calculate upper weight bounds for clipping distance decay curves based on spatial tolerance. Used when data point location has uncertainty defined by `spatial_tolerance`. Determine the upper weights threshold of the distance decay curve for a given $\beta$ based on the `spatial_tolerance` parameter. This is used by downstream functions to determine the upper extent at which weights derived for spatial impedance functions are flattened and normalised. This functionality is only intended for situations where the location of datapoints is uncertain for a given spatial tolerance.

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

 Distance thresholds ($d_{max}$).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">betas</div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Decay parameters ($\beta$).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">spatial_tolerance</div>
    <div class="type">int</div>
  </div>
  <div class="desc">

 Spatial buffer distance (uncertainty).</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">list[float]</div>
  </div>
  <div class="desc">

 Maximum weights for clipping the decay curve for each beta.</div>
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


 Calculate a single weight using beta decay, clipped by a maximum weight. Applies $weight = exp(-\beta \cdot distance)$, ensuring the result does not exceed `max_curve_wt`.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">beta</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The decay parameter $\beta$.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">max_curve_wt</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The maximum allowed weight (from `clip_wts_curve`).</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">data_dist</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The distance to the data point.</div>
</div>

### Returns
<div class="param-set">
  <div class="def">
    <div class="name"></div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 The calculated (potentially clipped) weight. Returns 0.0 if calculation fails.</div>
</div>


</div>



</section>
