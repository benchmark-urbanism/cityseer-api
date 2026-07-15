---
layout: '@src/layouts/PageLayout.astro'
---

# Network Centrality

Accessible via **Processing > Cityseer > Network Centrality**. Computes localised closeness and betweenness centrality on a street network using a dual graph representation.

## Input Parameters

| Parameter                     | Description                                                                                                                               | Default      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer** | A line layer in a projected metre-based CRS                                                                                               | _(required)_ |
| **Distance thresholds**       | Comma-separated distances in metres                                                                                                       | `400,800`    |
| **Betweenness tolerance %**   | Controls betweenness spread across near-shortest paths. 0 = exact shortest paths only. Keep below 2%.                                     | `0.0`        |
| **Simplest-path tolerance %** | Tolerance on angular route cost for near-simplest routes. Keep below 20%.                                                                 | `0.0`        |
| **Boundary polygon**          | Optional polygon layer. Nodes inside the boundary are used as centrality sources; nodes outside provide network context only.             | _(none)_     |
| **Use adaptive sampling**     | _Experimental._ When enabled, a pilot poll measures per-node reach and distances run sampled only when predicted to be faster than exact. | `False`      |
| **Error tolerance epsilon**   | Advanced. Sampling accuracy tolerance. The default 0.05 preserves node rankings; loosen towards 0.1 for exploratory work.                 | `0.05`       |
| **Time thresholds**           | Advanced. Comma-separated minutes; overrides distances when set. Converted to metres using the walking speed.                             | _(none)_     |
| **Walking speed**             | Advanced. Metres per second, used to convert minutes to distances.                                                                        | `1.33`       |

## Metric Selection

The algorithm dialog provides a 2x2 grid of metric categories. Each category can be toggled on or off independently, and individual metrics within each category are selected independently. Enabling a metric in one category does not affect other categories.

|                 | Shortest path                                      | Simplest path (angular)             |
| --------------- | -------------------------------------------------- | ----------------------------------- |
| **Closeness**   | harmonic, density, farness, decay, cycles, hillier | harmonic, density, farness, hillier |
| **Betweenness** | betweenness, betweenness_decay                     | betweenness                         |

By default, harmonic closeness and betweenness are enabled for shortest paths. All simplest path categories are off by default; when simplest-path closeness is enabled, Hillier (improved closeness) is the default metric.

## Output

The output is a line layer with the original street segments and computed centrality values as attributes. Output fields follow the naming convention:

```text
cc_<metric>_<distance>[_ang]
```

For example, with distances `400,800`:

- `cc_harmonic_400`, `cc_harmonic_800`
- `cc_betweenness_400`, `cc_betweenness_800`
- `cc_harmonic_400_ang` (if simplest path closeness is enabled)

## Sampling

Adaptive sampling is optional and off by default. When enabled, a pilot poll measures each segment's network reach at every distance threshold, and per-segment sampling probabilities are derived from the Hoeffding bound so that every catchment accumulates approximately the required number of samples. A work test then decides per distance threshold whether sampling is predicted to be faster than exact computation; distances that would not benefit run exactly. Inverse-probability weighting keeps the resulting estimates unbiased.

The error tolerance epsilon (advanced parameter, default 0.05) is calibrated so that node rankings are preserved relative to exact computation. Speed-ups are largest on dense networks at long distance thresholds. See the [sampling module](/metrics/sampling) documentation for the methodology.
