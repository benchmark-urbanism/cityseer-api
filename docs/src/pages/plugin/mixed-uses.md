---
layout: '@src/layouts/PageLayout.astro'
---

# Mixed Uses

Accessible via **Processing > Cityseer > Mixed Uses**. Computes land-use diversity metrics within distance thresholds along the street network, from a point or polygon data layer with a land-use category column.

## Input Parameters

| Parameter                       | Description                                                                                              | Default      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer**   | A line layer in a projected metre-based CRS                                                              | _(required)_ |
| **Data layer**                  | A point or polygon layer containing land-use features                                                    | _(required)_ |
| **Land-use category field**     | Text column containing the land-use category for each feature                                            | _(required)_ |
| **Distance thresholds**         | Comma-separated distances in metres                                                                      | `400,800`    |
| **Max assignment distance**     | Maximum distance (metres) to snap data points to the nearest street segment                              | `400`        |
| **Use simplest path (angular)** | Use angular (simplest) paths instead of shortest (metric) paths                                          | `False`      |
| **Hill diversity**              | Compute Hill diversity at q = 0, 1, 2. q0 counts distinct reachable land uses.                           | `True`       |
| **Shannon entropy**             | Advanced. Prefer Hill q = 1 unless you specifically need entropy.                                        | `False`      |
| **Gini-Simpson**                | Advanced. Prefer Hill q = 2 unless you specifically need it.                                             | `False`      |
| **Boundary polygon**            | Optional polygon layer. Nodes inside the boundary are used as sources.                                   | _(none)_     |
| **Decay expression**            | Advanced. Distance-decay weighting for Hill diversity. Default 1 weights all reachable features equally. | `1`          |
| **Time thresholds**             | Advanced. Comma-separated minutes; overrides distances when set. Converted using the walking speed.      | _(none)_     |
| **Walking speed**               | Advanced. Metres per second, used to convert minutes to distances.                                       | `1.33`       |

## Output

The output is a line layer with the original street segments and diversity values as attributes:

```text
cc_hill_q0_<distance>[_ang]   -> count of distinct reachable land uses
cc_hill_q1_<distance>[_ang]   -> Hill diversity of order 1 (exponential of Shannon)
cc_hill_q2_<distance>[_ang]   -> Hill diversity of order 2 (inverse Simpson)
cc_shannon_<distance>[_ang]   -> Shannon entropy (if enabled)
cc_gini_<distance>[_ang]      -> Gini-Simpson diversity (if enabled)
```

Hill diversity is the recommended measure: q = 0 is generally the best choice for granular land-use data, and higher orders progressively discount rare categories. Shannon and Gini-Simpson are provided under the advanced parameters for comparability with older studies; the Hill equivalents (q = 1 and q = 2) are preferable in most cases. See the [land-use guide](/guide/land-use) for interpretation.
