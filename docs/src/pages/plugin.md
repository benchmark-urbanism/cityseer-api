---
layout: '@src/layouts/PageLayout.astro'
---

## QGIS Plugin

The `cityseer` QGIS plugin provides a Processing algorithm for computing localised network centrality metrics (closeness and betweenness) directly within QGIS. It uses a dual graph representation where each road segment becomes a node connected to its neighbours, with support for multiple distance thresholds and deterministic distance-based sampling.

The plugin is experimental and requires QGIS 4.0+.

### Installation

Install the plugin from the QGIS plugin repository: go to **Plugins > Manage and Install Plugins**, search for "Cityseer", and click **Install**. Enable the "Show also experimental plugins" option in the **Settings** tab if the plugin is not visible.

On first load, the plugin will prompt to install the `cityseer` Python library if it is not already available in the QGIS Python environment.

### Usage

The algorithm is accessible via **Processing > Cityseer > Network Centrality**.

#### Input Parameters

| Parameter                                     | Description                                                                                                                                                                 | Default      |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| **Street network line layer**                 | A line layer in a projected metre-based CRS                                                                                                                                 | _(required)_ |
| **Distance thresholds**                       | Comma-separated distances in metres                                                                                                                                         | `400,800`    |
| **Betweenness tolerance %**                   | Controls betweenness spread across near-shortest paths. 0 = exact shortest paths only.                                                                                      | `10.0`       |
| **Boundary polygon**                          | Optional polygon layer. Nodes inside the boundary are used as centrality sources; nodes outside provide network context only.                                               | _(none)_     |
| **Use deterministic distance-based sampling** | When enabled, sampling probability is computed per distance threshold. Distances where the probability is 1.0 are computed exactly; larger distances are sampled for speed. | `True`       |

The input layer must be in a **projected metre-based CRS** (not geographic/degrees).

#### Metric Selection

The algorithm dialog provides a 2x2 grid of metric categories:

|                 | Shortest path                                     | Simplest path (angular)       |
| --------------- | ------------------------------------------------- | ----------------------------- |
| **Closeness**   | harmonic, density, farness, beta, cycles, hillier | harmonic, density, farness    |
| **Betweenness** | betweenness, betweenness_beta                     | betweenness, betweenness_beta |

Each category can be toggled on or off, and individual metrics within each category can be selected independently. By default, harmonic closeness and betweenness are enabled for shortest paths.

#### Output

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
|         5000 |           84.6% |             1.2x |
|         8000 |           35.9% |             2.8x |
|        10000 |           23.8% |             4.2x |
|        15000 |           11.3% |             8.9x |
|        20000 |            6.6% |              15x |

Disable sampling for exact computation at all distances.
