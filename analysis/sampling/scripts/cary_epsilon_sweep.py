#!/usr/bin/env python
"""
cary_epsilon_sweep.py - Map where Cary's rank correlation crosses 0.95 vs epsilon.

Cary (sparse suburban) closeness lands at rho=0.945 at 20km with the default
eps=0.06. This sweeps eps at the distances where sampling engages to locate the
rho=0.95 crossing precisely, informing whether to keep eps=0.06 (design margin)
or tighten it.

Reuses the cached Cary network + per-distance ground truth (run 04 first); only
the cheap sampled passes re-run, via the low-level Rust API with an explicit
sample_probability (so eps is swept directly, bypassing the phi fallback).
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cityseer.sampling import compute_distance_p
from utilities import CACHE_DIR, OUTPUT_DIR, compute_accuracy_metrics

DISTANCES = [10000, 20000]
EPSILONS = [0.04, 0.05, 0.06, 0.08]
N_RUNS = 3

# load_cary_network from the (digit-prefixed) validation module
_spec = importlib.util.spec_from_file_location("cary_val", Path(__file__).parent / "03_validate_cary.py")
_cary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cary)


def main():
    net, nodes_gdf, live_mask, _ = _cary.load_cary_network()
    rows = []
    for dist in DISTANCES:
        with open(CACHE_DIR / f"cary_ground_truth_{dist}m.pkl", "rb") as f:
            gt = pickle.load(f)
        true_h, true_b = gt["harmonic"], gt["betweenness"]
        for eps in EPSILONS:
            p = compute_distance_p(dist, epsilon=eps)
            rc, rb = [], []
            for seed in range(N_RUNS):
                res = net.centrality_shortest(
                    distances=[dist],
                    closeness_exprs=[("harmonic", "1/c")],
                    betweenness_exprs=[("betweenness", "1")],
                    compute_cycles=False,
                    sample_probability=float(p),
                    random_seed=42 + seed,
                    pbar_disabled=True,
                )
                est_h = np.array(res.metrics["harmonic"][dist])[live_mask]
                est_b = np.array(res.metrics["betweenness"][dist])[live_mask]
                rc.append(compute_accuracy_metrics(true_h, est_h)[0])
                rb.append(compute_accuracy_metrics(true_b, est_b)[0])
            row = {
                "distance": dist,
                "epsilon": eps,
                "p": p,
                "rho_closeness": float(np.nanmean(rc)),
                "rho_betweenness": float(np.nanmean(rb)),
            }
            rows.append(row)
            print(
                f"  d={dist // 1000}km eps={eps}: p={p:.4f} "
                f"rho_c={row['rho_closeness']:.4f} rho_b={row['rho_betweenness']:.4f}"
            )
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "cary_epsilon_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("\n0.95 crossing (closeness): tighter eps -> larger p -> higher rho")
    return 0


if __name__ == "__main__":
    exit(main())
