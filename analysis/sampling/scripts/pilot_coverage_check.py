#!/usr/bin/env python
"""
pilot_coverage_check.py - Direct validation of the pilot poll's reach bounds.

The method's accuracy claims rest on the pilot: inclusion probabilities derive from the
one-sided lower Clopper-Pearson bound on polled reach, and work predictions from the
upper bound. The bounds are conservative by construction (exact binomial bounds on a
hypergeometric count), but the paper should also demonstrate the coverage empirically.
This script polls each validation network (three seeds) and compares the per-node
bounds against the true reach recorded in the ground-truth caches (cc_density):

- lcb_coverage: mean share of live nodes whose lower bound sits at or below the true
  reach (nominal >= 90% at alpha = 0.1; violations risk under-sampling)
- lcb_coverage_min/max: the per-seed range. All nodes share one pilot draw, so
  coverage events are correlated: an unlucky draw overestimates many nodes at once and
  a single seed's share can sit far below the nominal rate even though the per-node
  rate holds. The mean over seeds estimates the per-node rate; the range shows the
  correlation.
- ucb_coverage: mean share of live nodes whose upper bound sits at or above the true
  reach (nominal >= 90%; violations risk under-priced exact work)
- point_median_rel_err: median relative error of the point estimate
- lcb_violation_ratio: among lower-bound violations, the median ratio lcb / true reach
  (how far the risky tail overshoots; 1.0 means marginal)

Reads the cached graphs and ground truths only; no centrality is computed. Writes
output/pilot_coverage.csv.

Usage:
    python pilot_coverage_check.py                 # all four networks
    python pilot_coverage_check.py --networks gla  # subset
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from cityseer.sampling import estimate_polled_reach
from utilities import CACHE_DIR, OUTPUT_DIR
from validate_adaptive import DISTANCES, NETWORKS, _load_network

N_SEEDS = 10


def check_network(network: str) -> list[dict]:
    print(f"\n{'=' * 60}\n{network.upper()}: pilot reach-bound coverage\n{'=' * 60}")
    net, nodes_gdf, live_mask, _ = _load_network(network)
    rows = []
    per_seed = []
    for seed in range(N_SEEDS):
        lcb, point, ucb = estimate_polled_reach(net, DISTANCES, random_seed=42 + seed)
        per_seed.append((lcb, point, ucb))
    for dist in DISTANCES:
        gt_path = CACHE_DIR / f"{network}_ground_truth_{dist}m.pkl"
        if not gt_path.exists():
            print(f"  {dist}m: no ground truth cache, skipping")
            continue
        with open(gt_path, "rb") as f:
            gt = pickle.load(f)
        true_reach = np.asarray(gt["node_reach"], dtype=float)
        lcb_cov, ucb_cov, rel_errs, viol_ratios = [], [], [], []
        for lcb, point, ucb in per_seed:
            lo = np.asarray(lcb[dist], float)[live_mask]
            pt = np.asarray(point[dist], float)[live_mask]
            hi = np.asarray(ucb[dist], float)[live_mask]
            ok = true_reach > 0
            lcb_cov.append(float(np.mean(lo[ok] <= true_reach[ok])))
            ucb_cov.append(float(np.mean(hi[ok] >= true_reach[ok])))
            rel_errs.append(float(np.median(np.abs(pt[ok] - true_reach[ok]) / true_reach[ok])))
            viol = lo[ok] > true_reach[ok]
            if viol.any():
                viol_ratios.append(float(np.median(lo[ok][viol] / true_reach[ok][viol])))
        row = {
            "network": network,
            "distance": dist,
            "lcb_coverage": float(np.mean(lcb_cov)),
            "lcb_coverage_min": float(np.min(lcb_cov)),
            "lcb_coverage_max": float(np.max(lcb_cov)),
            "ucb_coverage": float(np.mean(ucb_cov)),
            "point_median_rel_err": float(np.mean(rel_errs)),
            "lcb_violation_ratio": float(np.mean(viol_ratios)) if viol_ratios else np.nan,
        }
        rows.append(row)
        print(
            f"  {dist}m: lcb_cov={row['lcb_coverage']:.4f} "
            f"[{row['lcb_coverage_min']:.2f}-{row['lcb_coverage_max']:.2f}] "
            f"ucb_cov={row['ucb_coverage']:.4f} "
            f"pt_med_rel_err={row['point_median_rel_err']:.3f} viol_ratio={row['lcb_violation_ratio']:.3f}"
        )
    return rows


def emit_table() -> None:
    """Emit the appendix coverage table (20 km rows) from the CSV."""
    df = pd.read_csv(OUTPUT_DIR / "pilot_coverage.csv")
    df20 = df[df["distance"] == 20000]
    names = {"gla": "London", "madrid": "Madrid", "cary": "Cary", "woodlands": "The Woodlands"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pilot reach-bound coverage at 20\,km, ten pilot draws per network"
        r" against the exact reach. Coverage is the share of live nodes whose lower"
        r" confidence bound sits at or below the true reach (nominal 0.90 at $\alpha = 0.1$)."
        r" All nodes share each draw, so coverage is correlated across nodes and single"
        r" draws range widely; violations overshoot the true reach by the stated median"
        r" ratio. Per-distance detail in \texttt{output/pilot\_coverage.csv}.}",
        r"\label{tab:pilot_coverage}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Network} & \textbf{Coverage (mean)} & \textbf{Single-draw range} &"
        r" \textbf{Violation ratio} & \textbf{Point median rel.\ err.} \\",
        r"\midrule",
    ]
    for key, name in names.items():
        r = df20[df20["network"] == key].iloc[0]
        lines.append(
            f"{name} & {r['lcb_coverage']:.2f} & {r['lcb_coverage_min']:.2f}--{r['lcb_coverage_max']:.2f}"
            f" & {r['lcb_violation_ratio']:.3f} & {r['point_median_rel_err']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = OUTPUT_DIR.parent / "paper" / "tables" / "tab9_pilot_coverage.tex"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot reach-bound coverage check")
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS.keys()))
    parser.add_argument("--table-only", action="store_true", help="Emit the LaTeX table from the existing CSV")
    args = parser.parse_args()
    if not args.table_only:
        rows = []
        for network in args.networks:
            rows.extend(check_network(network))
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "pilot_coverage.csv", index=False)
        print(f"\nSaved {OUTPUT_DIR / 'pilot_coverage.csv'}")
    emit_table()
    return 0


if __name__ == "__main__":
    exit(main())
