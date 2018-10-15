# centrality

## distance from beta

This is a convenience function for mapping $-\beta$ decay parameters to equivalent $d_{max}$ distance thresholds.

#### distance_from_beta(beta, min_threshold_wt=0.01831563888873418)

##### params
___

**beta** (`float`, `list[float]`): The $-\beta$ that you wish to convert to distance thresholds.

**min_threshold_wt** (`float`): The $w_{min}$ threshold.

##### returns
___

**betas** (`np.ndarray`): A numpy array of effective $d_{max}$ distances.

**min_threshold_wt** (`float`): The corresponding $w_{min}$ threshold.


::: tip
There is no need to use this function unless:

- Providing custom beta values for weighted centrality measures, instead of using the generated defaults;
- Using custom `min_threshold_wt` parameters.
:::

::: warning Important
Pass both $d_{max}$ and $w_{min}$ to [`centrality.compute_centrality`](#compute-centrality-a) for the desired behaviour.
:::



The weighted variants of centrality, e.g. gravity or weighted betweenness, are computed using a negative exponential decay function of the form:

$$
weight = exp(-\beta \cdot distance)
$$

The strength of the decay is controlled by the $-\beta$ parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
For example, if $-\beta=0.005$ represents a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at 13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At some point, it becomes futile to consider locations any farther away, so it is necessary to set a a minimum weight threshold $w_{min}$ corresponding to a maximum distance of $d_{max}$.

The [`centrality.compute_centrality`](#compute-centrality-a) method computes the $-\beta$ parameters automatically, using a default `min_threshold_wt` of $w_{min}=0.01831563888873418$.

$$
\beta = \frac{log\Big(1 / w_{min}\Big)}{d_{max}}
$$

Therefore, $-\beta$ weights corresponding to $d_{max}$ walking thresholds of 400m, 800m, and 1600m would give:

| $d_{max}$ | $-\beta$ |
|-----------|:----------|
| $400m$ | $-0.01$ |
| $800m$ | $-0.005$ |
| $1600m$ | $-0.0025$ |

In reality, people may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context. If overriding the defaults, or to use a custom $-\beta$ or a different $w_{min}$ threshold, then this function can be used to generate the effective $d_{max}$ values, which can then be passed to [`centrality.compute_centrality`](#compute-centrality-a) along with the specified $w_{min}$. For example, the following $-\beta$ and $w_{min}$ thresholds yield these effective $d_{max}$ distances:

| $-\beta$ | $w_{min}$ | $d_{max}$ |
|----------|:----------|:----------|
| $-0.01$ | $0.01$ | $461m$ |
| $-0.005$ | $0.01$ | $921m$ |
| $-0.0025$ | $0.01$ | $1842m$ |

## compute centrality a

test