---
layout: '@src/layouts/PageLayout.astro'
---

# QGIS Plugin

The `cityseer` QGIS plugin provides Processing algorithms for urban network analysis directly within QGIS, without writing Python code. It uses a dual graph representation where each road segment becomes a node connected to its neighbours, with support for multiple distance thresholds, shortest and simplest (angular) paths, and optional adaptive sampling.

The plugin is experimental and requires QGIS 4.0+.

## Algorithms

- [Network Centrality](/plugin/centrality): localised closeness and betweenness centrality with independent metric selection per category.
- [Demand Betweenness (OD Flow)](/plugin/demand): demand-weighted flow betweenness from weighted origins and destinations using a spatial interaction model.
- [Accessibility](/plugin/accessibility): counts of reachable land-use features and distances to the nearest feature per category.
- [Mixed Uses](/plugin/mixed-uses): land-use diversity metrics (Hill, Shannon, Gini-Simpson) within distance thresholds.
- [Statistics](/plugin/statistics): localised statistics for numerical data columns aggregated over the street network.

## Installation

Install the plugin from the QGIS plugin repository: go to **Plugins > Manage and Install Plugins**, search for "Cityseer", and click **Install**. Enable the "Show also experimental plugins" option in the **Settings** tab if the plugin is not visible.

On first load, the plugin will prompt to install the `cityseer` Python library if it is not already available in the QGIS Python environment.

## Common Concepts

The following concepts apply to all algorithms:

- **Network distances**: Every metric is computed over the street network. Distances are shortest-path distances walked along the streets, not straight-line (crow-flies) distances, and catchments are network catchments: a feature on the far side of a river or railway is only reachable via an actual crossing, however close it sits on the map.
- **Projected CRS**: All input layers must use a projected metre-based coordinate reference system (not geographic/degrees). All layers in a given analysis must share the same CRS.
- **Distance thresholds**: Metrics are computed independently at each distance threshold, with shorter distances corresponding to localised patterns and longer distances to wider-area structure. Time thresholds (comma-separated minutes with a walking speed) are available as advanced parameters on every algorithm; they are converted to metre distances and output columns are named by the resolved metres.
- **Boundary polygon**: An optional polygon layer. Nodes whose midpoints fall inside the boundary are "live" (used as analysis sources); nodes outside provide network context only. Multi-polygon layers are supported (features are merged automatically).
- **Simplest path (angular)**: When enabled, paths minimise cumulative angular change instead of metric distance. This models route choice based on cognitive simplicity rather than physical distance, and uses the plugin's internal dual graph representation.
- **Output columns**: Each algorithm returns the street segments as a line layer with computed values as attributes, named `cc_<metric>_<distance>` with an `_ang` suffix for simplest-path variants.

For the equivalent Python workflow, from computed metrics to a styled QGIS layer or publication figure, see the [From Results to Maps](/examples/networks/results-to-maps) recipe.
