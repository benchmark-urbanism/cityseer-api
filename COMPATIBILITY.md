# Backwards compatibility (4.25)

4.25 generalises the metrics API — hardcoded β-metrics became configurable **decay expressions**, and a
high-level **`CityNetwork`** class was added. This document records the contract: **the functional API you
already use stays behaviour-compatible by default**, and the new expression engine is **opt-in**. Nothing in the
documented high-level API breaks.

## Two surfaces

| Surface | Default output |
| --- | --- |
| `cityseer.metrics.networks` / `layers` functions (the old API you already use) | **Classic 4.24** — the full metric set; land-use returns both `_nw` and `_wt` columns |
| `cityseer.CityNetwork` (the new API) | **Lean** — a single harmonic closeness + a single betweenness, cycles off; land-use returns one column |

The old functions are preserved exactly so existing scripts keep working. The new `CityNetwork` is a clean
surface with no back-compat debt, so it defaults to a tidy, single-metric output. Either way the expression
engine is opt-in: pass `closeness` / `betweenness` dicts (centrality) or `decay_fn` (land-use) to compute exactly
what you want. Nothing in the documented old API breaks.

Low-level surfaces (`rustalgos.*` `NetworkStructure`/`DataMap` methods, the result objects, `pair_distances_*`)
**do** change in 4.25 — these were never the stable public contract.

## What's preserved — the default call

The common path is preserved exactly. A call using only the everyday arguments (`distances`,
`landuse_column_label`, …) returns the **same columns and the same numbers** as 4.24:

- **Renamed functions** — `node_centrality_shortest` and `node_centrality_simplest` are restored as deprecated
  aliases that emit `DeprecationWarning`. (`segment_centrality` is *not* restored — see "Removed" below.)
- **Default output** — legacy column names (`cc_beta_*`, `cc_betweenness_beta_*`, land-use `_wt` / `_nw`) and their
  values are reproduced. The old default β-weighting equals `exp(-k · p)` with `k = -ln(min_threshold_wt)` (= 4 by
  default, `p = c / threshold`), so the numbers are identical.

We deliberately **do not** reproduce every removed *parameter*. Those are the rare calls and — unlike the default
output — they fail **loudly** (a `TypeError`), never silently with wrong numbers. So they get a clear error and a
documented migration, not a full shim.

## Changed or removed parameters

Pass one of these (beyond the everyday args) and here is exactly what to expect — every one fails loudly, so none
can silently mislead:

| Old parameter | 4.25 | If you pass it | Do this instead |
| --- | --- | --- | --- |
| `betas=[...]` | removed | `TypeError` | rely on `distances` (default weighting is unchanged), or `decay_fn="exp(-beta * c)"` |
| `min_threshold_wt=` | removed | `TypeError` | only affected custom β scaling; fold into the `decay_fn` expression |
| `spatial_tolerance=` | removed | `TypeError` | no direct equivalent — note per use |
| `compute_hill_weighted=` | removed | `TypeError` | pass a `decay_fn` to weight Hill diversity |
| `angular_scaling_unit=`, `farness_scaling_offset=` | removed | `TypeError` | bake into the simplest expression, e.g. `farness="1 + c / 90"` |
| `source_indices=` | removed | `TypeError` | nearest equivalent is `sample=True` (subset of sources) |

## Opting into the new behaviour

Pass expression dicts / `decay_fn` explicitly, or use `CityNetwork`:

```python
# classic (unchanged): emits cc_..._wt and cc_..._nw
networks.compute_accessibilities(..., distances=[800])

# new expression engine, opt-in: emits cc_..._800
networks.compute_accessibilities(..., distances=[800], decay_fn="exp(-4 * p)")
```

The efficiency tips for the new API (computing only the metrics/columns you need) live on the new functions'
own docstrings — see `centrality_shortest` / `centrality_simplest` and `compute_accessibilities` /
`compute_mixed_uses` / `compute_stats`.

## Removed (not restored)

- **`segment_centrality`** — the underlying continuous-segment routine (`segment_density` / `harmonic` / `beta` /
  `betweenness`) was removed at the low level in 4.25, so the old numbers cannot be reproduced. The nearest modern
  equivalent is `centrality_shortest(..., segment_weighted=True)`, which weights node centrality by street-segment
  length — related, but a different calculation. Calling `segment_centrality` raises a clear error pointing here.

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

Three contained moves, no duplicated algorithms (one source of truth), each pinned by a parity test against the
4.24 result:

- **Renamed functions** → thin deprecated shims that translate into the new core and relabel outputs
  (`node_centrality_shortest`, `node_centrality_simplest` — done).
- **Kept-name functions** → their **default** reverts to classic output (the shared `_resolve_decay_fns` returns
  the `_nw` + `_wt` pair); the new expression engine is reached by passing `decay_fn` / expression dicts.
- **Removed parameters / functions** → raise a clear error pointing at the table above; not reproduced.
