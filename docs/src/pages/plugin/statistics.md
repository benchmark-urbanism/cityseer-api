---
layout: '@src/layouts/PageLayout.astro'
---

# Statistics

Accessible via **Processing > Cityseer > Statistics**. Computes localised statistics for one or more numerical data columns within distance thresholds along the street network.

## Input Parameters

| Parameter                       | Description                                                                                                        | Default      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------ |
| **Street network line layer**   | A line layer in a projected metre-based CRS                                                                        | _(required)_ |
| **Data layer**                  | A point or polygon layer containing numerical data                                                                 | _(required)_ |
| **Numerical field(s)**          | One or more numeric columns to compute statistics on; all selected fields are computed in a single network pass    | _(required)_ |
| **Distance thresholds**         | Comma-separated distances in metres                                                                                | `400,800`    |
| **Max assignment distance**     | Maximum distance (metres) to snap data points to the nearest street segment                                        | `400`        |
| **Use simplest path (angular)** | Use angular (simplest) paths instead of shortest (metric) paths on the internally constructed dual graph           | `False`      |
| **Boundary polygon**            | Optional polygon layer. Nodes inside the boundary are used as sources; nodes outside provide network context only. | _(none)_     |
| **Decay expression**            | Advanced. Distance-decay weighting for the statistics. Default 1 weights all contributions equally.                | `1`          |
| **Time thresholds**             | Advanced. Comma-separated minutes; overrides distances when set. Converted to metres using the walking speed.      | _(none)_     |
| **Walking speed**               | Advanced. Metres per second, used to convert minutes to distances.                                                 | `1.33`       |

Only numeric columns are available for the numerical field selector. Features with missing, null, or non-finite values in a given field are skipped for that field's statistics.

## Available Statistics

The algorithm dialog provides checkboxes for selecting which statistics to compute. Only the selected statistics are computed, so deselecting the median and MAD (the costly measures) speeds up large runs.

| Statistic    | Description               | Default |
| ------------ | ------------------------- | :-----: |
| **Sum**      | Sum of values             |   On    |
| **Mean**     | Mean of values            |   On    |
| **Count**    | Number of data points     |   On    |
| **Median**   | Median of values          |   Off   |
| **Variance** | Variance of values        |   Off   |
| **MAD**      | Median Absolute Deviation |   Off   |
| **Max**      | Maximum value             |   Off   |
| **Min**      | Minimum value             |   Off   |

Each statistic produces one column per field per distance threshold. All statistics use the decay function to weight contributions by distance; the default is flat (no decay).

## Output

The output is a line layer with the original street segments and computed statistics as attributes. Output fields follow the naming convention:

```text
cc_<field>_<statistic>_<distance>[_ang]    -> computed statistic
```

For example, with a `price` field and distances `400,800`:

- `cc_price_sum_400`, `cc_price_sum_800`
- `cc_price_mean_400`, `cc_price_mean_800`
- `cc_price_max_400`, `cc_price_max_800`
