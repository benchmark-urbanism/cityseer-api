#!/usr/bin/env python
"""
validate_adaptive.py - Validate per-node (adaptive) sampling on all four networks.

Runs the runtime `sample=True` path, which now assigns per-node inclusion
probabilities from pilot reach estimates (see cityseer.sampling), against the
cached exact ground truths produced by the per-network validation scripts. The
canonical-schedule results in {network}_validation.csv are left untouched; they
are the baseline the paper compares against. Adaptive results are written to
{network}_validation_adaptive.csv.

Usage:
    python validate_adaptive.py                  # all four networks
    python validate_adaptive.py --networks cary woodlands
"""

import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cityseer.metrics import networks
from utilities import CACHE_DIR, OUTPUT_DIR, compute_accuracy_metrics, compute_quartile_accuracy, mean_quartiles

DISTANCES = [1000, 2000, 5000, 10000, 20000]
N_RUNS = 3

NETWORKS = {
    "gla": ("01_validate_gla.py", "load_gla_network"),
    "madrid": ("02_validate_madrid.py", "load_madrid_network"),
    "cary": ("03_validate_cary.py", "load_cary_network"),
    "woodlands": ("04_validate_woodlands.py", "load_woodlands_network"),
}


def _load_network(network: str):
    script, loader_name = NETWORKS[network]
    spec = importlib.util.spec_from_file_location(f"{network}_val", Path(__file__).parent / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, loader_name)()


def _baseline_times(gt: dict) -> tuple[float | None, float | None]:
    close = gt.get("baseline_time_closeness", gt.get("baseline_close_time"))
    betw = gt.get("baseline_time_betweenness", gt.get("baseline_betw_time"))
    return close, betw


def validate_network(network: str) -> pd.DataFrame:
    print(f"\n{'=' * 60}\n{network.upper()}: adaptive sampling validation\n{'=' * 60}")
    net, nodes_gdf, live_mask, _ = _load_network(network)
    rows = []
    for dist in DISTANCES:
        gt_path = CACHE_DIR / f"{network}_ground_truth_{dist}m.pkl"
        if not gt_path.exists():
            print(f"  {dist}m: no ground truth cache, skipping")
            continue
        with open(gt_path, "rb") as f:
            gt = pickle.load(f)
        true_h = np.asarray(gt["harmonic"], dtype=float)
        true_b = np.asarray(gt["betweenness"], dtype=float)
        node_reach = np.asarray(gt["node_reach"], dtype=float) if gt.get("node_reach") is not None else None
        base_c, base_b = _baseline_times(gt)

        rhos_h, rhos_b, times_c, times_b, quartiles_h, quartiles_b = [], [], [], [], [], []
        for seed in range(N_RUNS):
            t0 = time.time()
            gdf_c = networks.closeness_shortest(
                net, nodes_gdf.copy(), distances=[dist], random_seed=42 + seed, sample=True
            )
            times_c.append(time.time() - t0)
            est_h = gdf_c[f"cc_harmonic_{dist}"].to_numpy(float)[live_mask]
            sp_h, _, _, _, _ = compute_accuracy_metrics(true_h, est_h)
            if not np.isnan(sp_h):
                rhos_h.append(sp_h)
            if node_reach is not None:
                quartiles_h.append(compute_quartile_accuracy(true_h, est_h, node_reach))

            t0 = time.time()
            gdf_b = networks.betweenness_shortest(
                net, nodes_gdf.copy(), distances=[dist], random_seed=42 + seed, sample=True
            )
            times_b.append(time.time() - t0)
            est_b = gdf_b[f"cc_betweenness_{dist}"].to_numpy(float)[live_mask]
            sp_b, _, _, _, _ = compute_accuracy_metrics(true_b, est_b)
            if not np.isnan(sp_b):
                rhos_b.append(sp_b)
            if node_reach is not None:
                quartiles_b.append(compute_quartile_accuracy(true_b, est_b, node_reach))
            print(".", end="", flush=True)

        rho_c = float(np.mean(rhos_h)) if rhos_h else float("nan")
        rho_b = float(np.mean(rhos_b)) if rhos_b else float("nan")
        mean_tc, mean_tb = float(np.mean(times_c)), float(np.mean(times_b))
        row = {
            "distance": dist,
            "rho_closeness": rho_c,
            "rho_closeness_std": float(np.std(rhos_h)) if len(rhos_h) > 1 else 0.0,
            "rho_betweenness": rho_b,
            "rho_betweenness_std": float(np.std(rhos_b)) if len(rhos_b) > 1 else 0.0,
            "sampled_time_c": mean_tc,
            "sampled_time_b": mean_tb,
            "speedup_closeness": (base_c / mean_tc) if base_c and mean_tc > 0 else float("nan"),
            "speedup_betweenness": (base_b / mean_tb) if base_b and mean_tb > 0 else float("nan"),
        }
        for k_q, v_q in mean_quartiles(quartiles_h).items():
            row[f"h_{k_q}"] = v_q
        for k_q, v_q in mean_quartiles(quartiles_b).items():
            row[f"b_{k_q}"] = v_q
        rows.append(row)
        print(f" {dist}m: rho_c={rho_c:.4f} rho_b={rho_b:.4f}")

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"{network}_validation_adaptive.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate adaptive sampling against cached exact baselines")
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS), default=sorted(NETWORKS))
    parser.add_argument("--distances", nargs="+", type=int, default=None, help="Restrict to these distances (m)")
    args = parser.parse_args()
    if args.distances:
        global DISTANCES
        DISTANCES = sorted(args.distances)
    for network in args.networks:
        validate_network(network)
    return 0


if __name__ == "__main__":
    exit(main())
