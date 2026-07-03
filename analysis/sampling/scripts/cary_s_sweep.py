#!/usr/bin/env python
"""
cary_s_sweep.py - Find the minimal grid spacing s that keeps Cary above rho=0.95.

s=250 (the first guess) lands Cary's 20km closeness at rho=0.969 — a ~2pt margin
over the 0.95 target, bought with ~half the speedup. This sweeps s at the binding
distances (fixed eps=0.06) to locate the smallest s that still clears 0.95 with a
modest safety margin, recovering the speedup s=250 gave away.

Reuses the cached Cary network + ground truth; only the cheap sampled passes run.
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cityseer.sampling import compute_distance_p
from utilities import CACHE_DIR, OUTPUT_DIR, compute_accuracy_metrics

DISTANCES = [10000, 20000]
S_VALUES = [175, 185, 190, 195, 200, 210, 225, 250]
EPSILON = 0.06
N_RUNS = 3

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
        for s in S_VALUES:
            p = compute_distance_p(dist, epsilon=EPSILON, grid_spacing=float(s))
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
                rc.append(compute_accuracy_metrics(true_h, np.array(res.metrics["harmonic"][dist])[live_mask])[0])
                rb.append(compute_accuracy_metrics(true_b, np.array(res.metrics["betweenness"][dist])[live_mask])[0])
            row = {
                "distance": dist,
                "s": s,
                "p": p,
                "rho_closeness": float(np.nanmean(rc)),
                "rho_betweenness": float(np.nanmean(rb)),
            }
            rows.append(row)
            print(
                f"  d={dist // 1000}km s={s}: p={p:.4f} "
                f"rho_c={row['rho_closeness']:.4f} rho_b={row['rho_betweenness']:.4f}"
            )
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "cary_s_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    exit(main())
