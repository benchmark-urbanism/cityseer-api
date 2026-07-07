#!/usr/bin/env python
"""
epsilon_sweep.py - Map where a network's rank correlation crosses 0.95 as a function of epsilon.

Usage:
    python epsilon_sweep.py --network cary
    python epsilon_sweep.py --network woodlands

For Cary this sweep is the calibration evidence behind the eps=0.05 default:
under the pre-calibration eps=0.06, Cary closeness landed at rho=0.945 at 20km,
below the 0.95 target, while eps=0.05 clears it (rho=0.961). For The Woodlands
(held out from calibration) the sweep quantifies the tolerance required to
restore the target where the shipped schedule falls short. Results are written
to output/{network}_epsilon_sweep.csv.

Reuses the cached network and per-distance ground truth (run the corresponding
validation script first); only the cheap sampled passes re-run, via the
low-level Rust API with an explicit sample_probability (so eps is swept
directly, bypassing the phi fallback).
"""

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cityseer.sampling import compute_distance_p
from utilities import CACHE_DIR, OUTPUT_DIR, compute_accuracy_metrics

DISTANCES = [10000, 20000]
EPSILONS = [0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08]
N_RUNS = 3

NETWORKS = {
    "cary": ("03_validate_cary.py", "load_cary_network"),
    "woodlands": ("04_validate_woodlands.py", "load_woodlands_network"),
}


def _load_network(network: str):
    script, loader_name = NETWORKS[network]
    spec = importlib.util.spec_from_file_location(f"{network}_val", Path(__file__).parent / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, loader_name)


def main():
    parser = argparse.ArgumentParser(description="Epsilon sweep for a validation network")
    parser.add_argument("--network", choices=sorted(NETWORKS), required=True)
    args = parser.parse_args()
    network = args.network
    loader = _load_network(network)
    net, nodes_gdf, live_mask, _ = loader()
    rows = []
    for dist in DISTANCES:
        with open(CACHE_DIR / f"{network}_ground_truth_{dist}m.pkl", "rb") as f:
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
    out = OUTPUT_DIR / f"{network}_epsilon_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print("\n0.95 crossing (closeness): tighter eps -> larger p -> higher rho")
    return 0


if __name__ == "__main__":
    exit(main())
