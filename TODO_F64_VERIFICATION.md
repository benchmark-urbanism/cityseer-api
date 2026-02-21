# f64 Accumulation Fix & Validation Pipeline Hardening

## 1. f64 accumulation fix

The `MetricResult` struct used `AtomicF32` for parallel accumulation. Once betweenness
values reached 2^24 = 16,777,216, `float32 + 1.0 = float32` — increments were silently
dropped. This capped GLA 20km betweenness at exactly 2^24 for 6.7% of nodes.

Changed to `AtomicF64` internally (exact for integers up to 2^53), with `f32` output
preserved (single rounding at the end, ~7 significant digits).

### Rust files changed

- `rust/src/common.rs` — `AtomicF32` → `AtomicF64`; `load()` casts f64→f32 at output
- `rust/src/centrality.rs` — `as f64` casts on all `fetch_add` calls
- `rust/src/data.rs` — `as f64` casts on all `fetch_add` and `store` calls

All 106 tests pass.

## 2. Validation pipeline hardening

### Live-node masking

Previously, `compute_accuracy_metrics` filtered by `true_vals > 0`, which incorrectly
excluded legitimate zero-betweenness nodes (leaf nodes) and included dead buffer-zone
nodes. Now:

- Both `03_validate_gla.py` and `04_validate_madrid.py` build a `live_mask` from
  `net.node_lives` and apply it when extracting arrays from centrality results.
- Ground truth caches now store only live-node data.
- `mean_reach` is computed over live nodes only (was previously diluted by dead nodes).

### Node-count semantics

`get_n_nodes()` in both GLA and Madrid scripts now counts **live** nodes only (was
previously counting all nodes including buffer zone).

### Bound check methodology

`max_abs_error` now uses `np.max` across runs (worst-case, correct for bound testing).
Previously used `np.mean` (mean-of-max), which underestimates the bound violation.

### Richer output statistics

Both GLA and Madrid validation CSVs now store:
- `spearman`, `spearman_min`, `spearman_median`, `spearman_max`, `spearman_std`
- `max_abs_error`, `mean_abs_error`, `median_abs_error`, `min_abs_error`, `max_abs_error_std`
- `n_live` in ground truth cache

### GLA `compute_theoretical_bounds` fix

Previously only handled betweenness with a single `max_abs_error` column (which no
longer existed after the pivot renamed it to `max_abs_error_h`/`max_abs_error_b`).
Now iterates over both metrics with metric-appropriate normalisation, matching the
Madrid implementation. `generate_validation_table` now carries forward the error
columns into its summary DataFrame.

### Files changed

- `rust/src/common.rs`
- `rust/src/centrality.rs`
- `rust/src/data.rs`
- `analysis/sampling/scripts/03_validate_gla.py`
- `analysis/sampling/scripts/04_validate_madrid.py`

## Still to do

- [ ] **Recompute GLA ground truth** with `--force` to confirm 2^24 cap is gone.
      Check that `max(betweenness_20000m)` exceeds 16,777,216.
- [ ] **Recompute Madrid ground truth** with `--force`.
- [ ] **Verify Hoeffding/EW bound** now holds for betweenness at 20km.
- [ ] **Update paper validation tables** with corrected values.
- [ ] **Delete stale n_nodes caches** (`gla_n_nodes.json`, `madrid_n_nodes.json`)
      if they exist, so `get_n_nodes` recomputes with live-node counting.
- [ ] **Delete this file** once verification is complete.

### Commands to run

```bash
# Rebuild from GLA validation onwards (skips synthetic data, which is unaffected)
uv run python analysis/sampling/scripts/run_all.py --force --from 3

# Or run individually:
uv run python analysis/sampling/scripts/03_validate_gla.py --force
uv run python analysis/sampling/scripts/04_validate_madrid.py --force
```
