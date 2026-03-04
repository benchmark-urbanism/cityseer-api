---
layout: '@src/layouts/PageLayout.astro'
---

## QGIS Plugin

The `cityseer` QGIS plugin provides Processing algorithms for computing localised network centrality metrics, land-use accessibility, and localised statistics directly within QGIS. It uses a dual graph representation where each road segment becomes a node connected to its neighbours, with support for multiple distance thresholds, shortest and simplest (angular) paths, and deterministic distance-based sampling.

The plugin is experimental and requires QGIS 4.0+.

### Installation

Install the plugin from the QGIS plugin repository: go to **Plugins > Manage and Install Plugins**, search for "Cityseer", and click **Install**. Enable the "Show also experimental plugins" option in the **Settings** tab if the plugin is not visible.

On first load, the plugin will prompt to install the `cityseer` Python library if it is not already available in the QGIS Python environment.

### Common Concepts

The following concepts apply to all three algorithms:

- **Projected CRS**: All input layers must use a projected metre-based coordinate reference system (not geographic/degrees). All layers in a given analysis must share the same CRS.
- **Distance thresholds**: Metrics are computed independently at each distance threshold. Shorter distances capture local patterns; longer distances capture wider-area structure.
- **Boundary polygon**: An optional polygon layer. Nodes whose midpoints fall inside the boundary are "live" (used as analysis sources); nodes outside provide network context only. Multi-polygon layers are supported (features are merged automatically).
- **Simplest path (angular)**: When enabled, paths minimise cumulative angular change instead of metric distance. This models route choice based on cognitive simplicity rather than physical distance.

### Network Centrality

Accessible via **Processing > Cityseer > Network Centrality**.

#### Input Parameters

| Parameter                                     | Description                                                                                                                                                                                 | Default      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer**                 | A line layer in a projected metre-based CRS                                                                                                                                                 | _(required)_ |
| **Distance thresholds**                       | Comma-separated distances in metres                                                                                                                                                         | `400,800`    |
| **Betweenness tolerance %**                   | Controls betweenness spread across near-shortest paths. 0 = exact shortest paths only. Keep below 1%.                                                                                       | `0.0`        |
| **Boundary polygon**                          | Optional polygon layer. Nodes inside the boundary are used as centrality sources; nodes outside provide network context only.                                                               | _(none)_     |
| **Use deterministic distance-based sampling** | _Experimental._ When enabled, sampling probability is computed per distance threshold. Distances where the probability is 1.0 are computed exactly; larger distances are sampled for speed. | `True`       |

#### Metric Selection

The algorithm dialog provides a 2x2 grid of metric categories. Each category can be toggled on or off independently, and individual metrics within each category are selected independently — enabling a metric in one category does not affect other categories.

|                 | Shortest path                                     | Simplest path (angular)             |
| --------------- | ------------------------------------------------- | ----------------------------------- |
| **Closeness**   | harmonic, density, farness, beta, cycles, hillier | harmonic, density, farness, hillier |
| **Betweenness** | betweenness, betweenness_beta                     | betweenness, betweenness_beta       |

By default, harmonic closeness and betweenness are enabled for shortest paths. All simplest path categories are off by default.

#### Centrality Output

The output is a line layer with the original street segments and computed centrality values as attributes. Output fields follow the naming convention:

```
cc_<metric>_<distance>[_ang]
```

For example, with distances `400,800`:

- `cc_harmonic_400`, `cc_harmonic_800`
- `cc_betweenness_400`, `cc_betweenness_800`
- `cc_harmonic_400_ang` (if simplest path closeness is enabled)

### Sampling

Deterministic distance-based sampling is enabled by default. Sampling probability depends only on the distance threshold: smaller distances run exactly while larger distances are sampled for a speed-up. The following table shows approximate sampling rates:

| Distance (m) | Sampled sources | Approx. speed-up |
| -----------: | --------------: | ---------------: |
|          400 |            100% |               1x |
|          800 |            100% |               1x |
|         1600 |            100% |               1x |
|         3200 |            100% |               1x |
|         5000 |           58.7% |             1.7x |
|         8000 |           24.9% |             4.0x |
|        10000 |           16.6% |             6.0x |
|        15000 |            7.8% |            12.7x |
|        20000 |            4.6% |            21.7x |

Disable sampling for exact computation at all distances.

### Accessibility

Accessible via **Processing > Cityseer > Accessibility**. Computes land-use accessibility by counting reachable features (from a point or polygon data layer) within distance thresholds along the street network.

#### Accessibility Parameters

| Parameter                       | Description                                                                                                          | Default      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer**   | A line layer in a projected metre-based CRS                                                                          | _(required)_ |
| **Data layer**                  | A point or polygon layer containing land-use features                                                                | _(required)_ |
| **Land-use category field**     | Text column containing the land-use category for each feature. If not set, all features are treated as one category. | _(none)_     |
| **Distance thresholds**         | Comma-separated distances in metres                                                                                  | `400,800`    |
| **Max assignment distance**     | Maximum distance (metres) to snap data points to the nearest street segment                                          | `400`        |
| **Use simplest path (angular)** | Use angular (simplest) paths instead of shortest (metric) paths                                                      | `False`      |
| **Boundary polygon**            | Optional polygon layer. Nodes inside the boundary are used as sources; nodes outside provide network context only.   | _(none)_     |

#### Land-Use Categories

If a land-use field is selected, the dialog provides a **Load categories** button that reads unique values from the selected field. Individual categories can then be checked or unchecked. Categories are unchecked by default — use **Select all** to enable all categories, or check specific ones. Only text (string) columns are available for the land-use field.

If no land-use field is selected, all features are counted together under a single category (`all`).

#### Accessibility Output

The output is a line layer with the original street segments and computed accessibility values as attributes. For each land-use category and distance threshold, three types of columns are produced:

```text
cc_<category>_<distance>[_ang]_nw    — unweighted count of reachable features
cc_<category>_<distance>[_ang]_wt    — distance-weighted count (exponential decay)
cc_<category>_nearest_max_<max_distance>[_ang]  — distance to nearest feature
```

For example, with a `type` field containing `pub` and `shop`, and distances `400,800`:

- `cc_pub_400_nw`, `cc_pub_400_wt`, `cc_pub_800_nw`, `cc_pub_800_wt`
- `cc_pub_nearest_max_800`
- `cc_shop_400_nw`, `cc_shop_400_wt`, `cc_shop_800_nw`, `cc_shop_800_wt`
- `cc_shop_nearest_max_800`

### Statistics

Accessible via **Processing > Cityseer > Statistics**. Computes localised statistics for a numerical data column within distance thresholds along the street network.

#### Statistics Parameters

| Parameter                       | Description                                                                                                        | Default      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------ |
| **Street network line layer**   | A line layer in a projected metre-based CRS                                                                        | _(required)_ |
| **Data layer**                  | A point or polygon layer containing numerical data                                                                 | _(required)_ |
| **Numerical field**             | Numeric column to compute statistics on                                                                            | _(required)_ |
| **Distance thresholds**         | Comma-separated distances in metres                                                                                | `400,800`    |
| **Max assignment distance**     | Maximum distance (metres) to snap data points to the nearest street segment                                        | `400`        |
| **Use simplest path (angular)** | Use angular (simplest) paths instead of shortest (metric) paths                                                    | `False`      |
| **Boundary polygon**            | Optional polygon layer. Nodes inside the boundary are used as sources; nodes outside provide network context only. | _(none)_     |

Only numeric columns are available for the numerical field selector. Features with missing, null, or non-finite values are skipped automatically.

#### Available Statistics

The algorithm dialog provides checkboxes for selecting which statistics to compute:

| Statistic    | Description               | Weighted variant | Default |
| ------------ | ------------------------- | :--------------: | :-----: |
| **Sum**      | Sum of values             |       Yes        |   On    |
| **Mean**     | Mean of values            |       Yes        |   On    |
| **Count**    | Number of data points     |       Yes        |   On    |
| **Median**   | Median of values          |       Yes        |   Off   |
| **Variance** | Variance of values        |       Yes        |   Off   |
| **MAD**      | Median Absolute Deviation |       Yes        |   Off   |
| **Max**      | Maximum value             |        No        |   Off   |
| **Min**      | Minimum value             |        No        |   Off   |

Statistics with weighted variants produce both unweighted (`_nw`) and distance-weighted (`_wt`) columns. The weighted variant uses exponential distance decay. Max and min have no weighted variant and produce a single column per distance.

To compute statistics for multiple numerical columns, run the algorithm once per column.

#### Statistics Output

The output is a line layer with the original street segments and computed statistics as attributes. Output fields follow the naming convention:

```text
cc_<field>_<statistic>_<distance>[_ang]_nw    — unweighted statistic
cc_<field>_<statistic>_<distance>[_ang]_wt    — distance-weighted statistic
cc_<field>_<statistic>_<distance>[_ang]       — for max/min (no weighted variant)
```

For example, with a `price` field and distances `400,800`:

- `cc_price_sum_400_nw`, `cc_price_sum_400_wt`, `cc_price_sum_800_nw`, `cc_price_sum_800_wt`
- `cc_price_mean_400_nw`, `cc_price_mean_400_wt`, `cc_price_mean_800_nw`, `cc_price_mean_800_wt`
- `cc_price_max_400`, `cc_price_max_800`
