# Cityseer QGIS Plugin

QGIS Processing algorithms for urban network analysis using the cityseer library: network centrality (closeness and betweenness), land-use accessibility, and localised statistics.

## Sampling Behaviour

The `Network Centrality` algorithm offers optional adaptive sampling (off by default). When enabled, it mirrors the runtime behaviour of `cityseer.metrics.networks` with `sample=True`:

- A pilot poll (`cityseer.sampling.estimate_polled_reach`) measures each node's network reach at every distance threshold with bounded shortest-path traversals from a small set of sampled sources.
- Per-node inclusion probabilities derive from the Hoeffding bound applied to the lower confidence bound on polled reach (`cityseer.sampling.compute_node_p`), so every catchment accumulates approximately the required number of samples.
- A work test compares predicted sampled work against exact work per distance threshold. A distance runs sampled only when sampling is predicted to be faster; otherwise it runs exactly.
- Sampled runs pass per-node probabilities to the Rust backend as `sampling_weights`, with inverse-probability weighting keeping estimates unbiased.

The error tolerance `epsilon` is exposed as an advanced parameter (default 0.05, which preserves node rankings at Spearman rho >= 0.95 on the validation networks; loosen towards 0.1 for exploratory work). The methodology is documented in `analysis/sampling/` and in the `cityseer.sampling` module.

## Development

The plugin source lives in `qgis_plugin/cityseer_qgis/`. When the repository checkout is present, the plugin prefers importing cityseer from `pysrc/` (development mode); otherwise it uses the pip-installed package and checks the version against `metadata.txt`.

Deploy a development symlink into the local QGIS profile:

```bash
python qgis_plugin/build_plugin.py --deploy
```

## Building the QGIS Plugin ZIP

Create a distributable QGIS plugin zip:

```bash
python qgis_plugin/build_plugin.py
```

This stamps the plugin version from `pyproject.toml`, ensures required plugin assets (`metadata.txt`, `LICENSE`, `icon.png`) are present, and writes a ZIP to `qgis_plugin/`.
