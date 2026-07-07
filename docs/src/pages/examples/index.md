---
layout: '@src/layouts/PageLayout.astro'
---

# Examples

Worked recipes for every part of `cityseer`, built on bundled real-world data. Each recipe page renders the executed notebook and offers the raw `.py` file for download. To work with a notebook interactively, download it (or clone the [repository](https://github.com/benchmark-urbanism/cityseer-api)) and open it with `uv run marimo edit <file>`. The notebooks are [marimo](https://marimo.io) files, but nothing ties the code to marimo: the cells are plain Python, so you can equally create a notebook in your own preferred environment and copy the cells across as you follow along.

The recipes build on each other, so if you are wondering why or how to do something, you may find the answer in a preceding recipe. Start with the [Quickstart](/examples/recipes/quickstart); the underlying concepts (network conversion, distance thresholds, edge rolloff) are explained in the [guide](/guide/fundamentals). [Open an issue](https://github.com/benchmark-urbanism/cityseer-api/issues) to request additional examples.

## Topics

- [Network Preparation](/examples/networks): building `cityseer`-compatible graphs from OSM, GeoDataFrames, OSMnx, and momepy; simplification; dual graphs; directed networks.
- [Network Centrality](/examples/centrality): metric and angular centrality, custom expressions, adaptive sampling, demand-weighted betweenness, GTFS.
- [Accessibility](/examples/accessibility): land-use accessibility and mixed-use metrics.
- [Statistics](/examples/stats): aggregating numeric feature properties over the network.
- [Visibility](/examples/visibility): street enclosure and openness from footprints and rasters.
- [Continuity](/examples/continuity): street name, route, and classification continuity.

## Datasets

The real-world datasets used by the recipes are documented on the [datasets](/examples/datasets) page, with source links and licensing information.
