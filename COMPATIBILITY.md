# Backwards compatibility (4.25)

4.25 generalises the metrics API — hardcoded β-metrics became configurable **decay expressions**, and a
high-level **`CityNetwork`** class was added. This document records the contract: **the functional API you
already use stays behaviour-compatible by default**, and the new expression engine is **opt-in**. Nothing in the
documented high-level API breaks.

## Two surfaces

| Surface | Default behaviour | Status |
| --- | --- | --- |
| `cityseer.metrics.networks` / `layers` functions | **Classic 4.24 output** — same columns, same numbers | Stable; deprecated names/params emit warnings; removed a few majors on |
| `cityseer.CityNetwork` class | Modern expression defaults | The recommended API going forward |

Low-level surfaces (`rustalgos.*` `NetworkStructure`/`DataMap` methods, the result objects, `pair_distances_*`)
**do** change in 4.25 — these were never the stable public contract.

## What's preserved (high-level functional API)

- **Function names** — `node_centrality_shortest`, `node_centrality_simplest`, `segment_centrality` are restored
  as deprecated aliases that emit `DeprecationWarning`.
- **Parameters** — `betas`, `min_threshold_wt`, `compute_closeness` / `compute_betweenness`, `spatial_tolerance`,
  `compute_hill_weighted`, and the angular scaling args are accepted and translated into the expression engine.
- **Columns & numbers** — legacy column names (`cc_beta_*`, `cc_betweenness_beta_*`, land-use `_wt` / `_nw`) and
  their values are reproduced exactly. The old default β-weighting equals `exp(-k · p)` with
  `k = -ln(min_threshold_wt)` (= 4 by default and `p = c / threshold`), so the numbers are identical.

## Opting into the new behaviour

Pass expression dicts / `decay_fn` explicitly, or use `CityNetwork`:

```python
# classic (unchanged): emits cc_..._wt and cc_..._nw
networks.compute_accessibilities(..., distances=[800])

# new expression engine, opt-in: emits cc_..._800
networks.compute_accessibilities(..., distances=[800], decay_fn="exp(-4 * p)")
```

## Genuinely unavoidable (rare)

- Direct readers of removed low-level result attributes (e.g. `CentralityShortestResult.node_beta`) — read the
  returned GeoDataFrame columns instead.
- Truly bespoke **per-distance** β shapes can't collapse to a single expression; the shim covers the standard case
  (one β per distance, derived from `min_threshold_wt`).

## Deprecation timeline

- **4.25** — compat layer active; deprecated names/params warn.
- **~4.26–4.27** — warnings remain; docs steer users to `CityNetwork`.
- **~4.28** (a few majors on) — remove the compat layer.

## How it's built (so it stays contained)

A single clearly-delimited *Deprecated 4.24 compatibility* section in `networks.py` / `layers.py` that only
**translates** old calls into the new core and **relabels** outputs — no duplicated algorithms, one source of
truth. Each shim is pinned by a parity test asserting its output matches the 4.24 result.
