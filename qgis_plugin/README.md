# Cityseer QGIS Plugin

## Default Sampling Behaviour

The `Network Centrality` algorithm enables sampling by default.

- Sampling is deterministic by distance threshold using `cityseer.config.compute_distance_p(distance)`.
- The probability depends only on the distance threshold (canonical model), not local graph size or local reach.
- The plugin Bernoulli-samples source segments with inclusion probability `p(d)` and passes `sample_probability=p(d)` to cityseer.
- Distances with `p(d) >= 1` run exactly (no approximation).

Default canonical parameters (from `pysrc/cityseer/config.py`):

- `epsilon = 0.06`
- `delta = 0.1`
- `grid_spacing = 175 m`

With these defaults, sampling stays exact up to about `4558 m` (`p(d) = 1.0`).

### Default `p(d)` and Approximate Speed-up

Approximate speed-up is shown as `~1/p(d)` and is only a rough upper-bound guide.
Actual speed-up depends on overheads, graph structure, and the selected metrics.

| Distance (m) | `p(d)` | Approx. sampled sources | Approx. speed-up |
|---:|---:|---:|---:|
| 400 | 1.000 | 100.0% | 1.00x |
| 800 | 1.000 | 100.0% | 1.00x |
| 1600 | 1.000 | 100.0% | 1.00x |
| 3200 | 1.000 | 100.0% | 1.00x |
| 5000 | 0.846 | 84.6% | 1.18x |
| 6000 | 0.607 | 60.7% | 1.65x |
| 8000 | 0.359 | 35.9% | 2.79x |
| 10000 | 0.238 | 23.8% | 4.19x |
| 12000 | 0.171 | 17.1% | 5.86x |
| 15000 | 0.113 | 11.3% | 8.85x |
| 20000 | 0.066 | 6.6% | 15.07x |

## Building the QGIS Plugin ZIP

Create a distributable QGIS plugin zip:

```bash
python qgis_plugin/build_plugin.py
```

This stamps the plugin version from `pyproject.toml`, ensures required plugin assets
(`metadata.txt`, `LICENSE`, `icon.png`) are present, and writes a ZIP to `qgis_plugin/`.
