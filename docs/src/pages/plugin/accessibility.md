---
layout: '@src/layouts/PageLayout.astro'
---

# Accessibility

Accessible via **Processing > Cityseer > Accessibility**. Computes land-use accessibility by counting reachable features (from a point or polygon data layer) within distance thresholds along the street network.

## Input Parameters

| Parameter                       | Description                                                                                                          | Default      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer**   | A line layer in a projected metre-based CRS                                                                          | _(required)_ |
| **Data layer**                  | A point or polygon layer containing land-use features                                                                | _(required)_ |
| **Land-use category field**     | Text column containing the land-use category for each feature. If not set, all features are treated as one category. | _(none)_     |
| **Distance thresholds**         | Comma-separated distances in metres                                                                                  | `400,800`    |
| **Max assignment distance**     | Maximum distance (metres) to snap data points to the nearest street segment                                          | `400`        |
| **Use simplest path (angular)** | Use angular (simplest) paths instead of shortest (metric) paths on the internally constructed dual graph             | `False`      |
| **Boundary polygon**            | Optional polygon layer. Nodes inside the boundary are used as sources; nodes outside provide network context only.   | _(none)_     |
| **Decay expression**            | Advanced. Distance-decay weighting for the counts. Default 1 counts all reachable features equally.                  | `1`          |
| **Time thresholds**             | Advanced. Comma-separated minutes; overrides distances when set. Converted to metres using the walking speed.        | _(none)_     |
| **Walking speed**               | Advanced. Metres per second, used to convert minutes to distances.                                                   | `1.33`       |

## Land-Use Categories

If a land-use field is selected, the dialog provides a **Load categories** button that reads unique values from the selected field. Individual categories can then be checked or unchecked. Categories are unchecked by default; use **Select all** to enable all categories, or check specific ones. Only text (string) columns are available for the land-use field.

If no land-use field is selected, all features are counted together under a single category (`all`).

## Output

The output is a line layer with the original street segments and computed accessibility values as attributes. For each land-use category and distance threshold, two types of columns are produced:

```text
cc_<category>_<distance>[_ang]                  — count of reachable features within the threshold
cc_<category>_nearest_max_<max_distance>[_ang]  — distance to nearest feature
```

For example, with a `type` field containing `pub` and `shop`, and distances `400,800`:

- `cc_pub_400`, `cc_pub_800`
- `cc_pub_nearest_max_800`
- `cc_shop_400`, `cc_shop_800`
- `cc_shop_nearest_max_800`
