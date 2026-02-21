# f64 Accumulation Fix — Verification TODO

## What was fixed

The `MetricResult` struct used `AtomicF32` for parallel accumulation. Once betweenness
values reached 2^24 = 16,777,216, `float32 + 1.0 = float32` — increments were silently
dropped. This capped GLA 20km betweenness at exactly 2^24 for 6.7% of nodes.

Changed to `AtomicF64` internally (exact for integers up to 2^53), with `f32` output
preserved (single rounding at the end, ~7 significant digits).

### Files changed

- `rust/src/common.rs` — `AtomicF32` → `AtomicF64`; `load()` casts f64→f32 at output
- `rust/src/centrality.rs` — `as f64` casts on all `fetch_add` calls
- `rust/src/data.rs` — `as f64` casts on all `fetch_add` and `store` calls

### Tests

All 106 tests pass.

## Still to do

- [ ] **Recompute GLA ground truth** at all distances (especially 20km) to confirm the
      2^24 cap is gone. Check that `max(betweenness_20000m)` exceeds 16,777,216.
- [ ] **Re-run the paper validation pipeline** (`analysis/sampling/scripts/03_validate_gla.py`)
      to check whether the Hoeffding formal bound now holds for betweenness at 20km
      (previously `eps_observed ~0.12 > eps_predicted ~0.10`).
- [ ] **Re-run the Madrid EW analysis** (`analysis/sampling/scripts/02_ew_analysis.py` or
      equivalent) to update `madrid_ew_analysis.csv` — betweenness bound failures at 20km
      should resolve.
- [ ] **Update paper validation tables** with corrected `eps_observed` values.
- [ ] **Delete this file** once verification is complete.
