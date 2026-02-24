#!/usr/bin/env python
"""
04_global_betweenness.py - Spatially stratified progressive source sampling.

For each distance d:
  1. Partition live nodes into spatial grid cells (cell_size = d/2)
  2. Each round: select one random live source per cell
  3. Run betweenness_exact_shortest(source_indices=...) for that round
  4. Accumulate betweenness across rounds, check convergence
  5. Stop when Spearman ρ(current, previous) > threshold or max rounds

Spatial stratification ensures uniform coverage. Progressive convergence
determines sample size empirically — no theoretical budget formula needed.

Usage:
    python 04_global_betweenness.py           # Run (skips cache if exists)
    python 04_global_betweenness.py --force   # Force regeneration

Outputs:
    - output/gla_stratified_betweenness.csv
"""

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cityseer.tools import io
from scipy import stats as scipy_stats
from utilities import (
    CACHE_DIR,
    OUTPUT_DIR,
    compute_accuracy_metrics,
)

SCRIPT_DIR = Path(__file__).parent

# Distance thresholds
DISTANCES = [1000, 2000, 5000, 10000, 20000]

# Convergence parameters
RHO_THRESHOLD = 0.999  # Stop when ρ(current, previous) exceeds this
MAX_ROUNDS = 50  # Safety cap
MIN_ROUNDS = 3  # Always run at least this many rounds

# Reproducibility
BASE_SEED = 42


def build_spatial_grid(net, cell_size: float) -> dict[tuple[int, int], list[int]]:
    """Partition live nodes into spatial grid cells.

    Returns dict mapping (cx, cy) -> [node_indices].
    """
    all_xs = net.node_xs
    all_ys = net.node_ys
    live_indices = [i for i in net.node_indices() if net.is_node_live(i)]

    if not live_indices:
        return {}

    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]
    x_min = min(xs)
    y_min = min(ys)

    cells: dict[tuple[int, int], list[int]] = {}
    for idx, x, y in zip(live_indices, xs, ys):
        cx = int((x - x_min) / cell_size)
        cy = int((y - y_min) / cell_size)
        cells.setdefault((cx, cy), []).append(idx)

    return cells


class CellSampler:
    """Without-replacement sampler across grid cells.

    Each cell maintains a shuffled deck of node indices. Each call to
    `sample_round()` draws one node per cell. When a cell's deck is
    exhausted it reshuffles — so every node is used before any repeats.
    """

    def __init__(self, cells: dict[tuple[int, int], list[int]], rng: random.Random):
        self.rng = rng
        self.decks: dict[tuple[int, int], list[int]] = {}
        self.pools: dict[tuple[int, int], list[int]] = {}
        for key, nodes in cells.items():
            if nodes:
                self.pools[key] = list(nodes)
                deck = list(nodes)
                rng.shuffle(deck)
                self.decks[key] = deck

    def sample_round(self) -> list[int]:
        """Draw one node per cell (without replacement until exhausted)."""
        sources = []
        for key, deck in self.decks.items():
            if not deck:
                # Reshuffle from the full pool
                deck = list(self.pools[key])
                self.rng.shuffle(deck)
                self.decks[key] = deck
            sources.append(deck.pop())
        return sources


def run_stratified_betweenness(force: bool = False):
    """Run spatially stratified progressive source sampling."""
    output_csv = OUTPUT_DIR / "gla_stratified_betweenness.csv"

    if output_csv.exists() and not force:
        print(f"Results already exist: {output_csv}")
        print("  Use --force to regenerate")
        return pd.read_csv(output_csv)

    # ---------------------------------------------------------------
    # Load GLA graph
    # ---------------------------------------------------------------
    gla_cache = CACHE_DIR / "gla_graph.pkl"
    if not gla_cache.exists():
        raise FileNotFoundError(
            f"GLA graph cache not found: {gla_cache}\n"
            "  Run 03_validate_gla.py first to build the graph cache."
        )

    print("\n" + "=" * 70)
    print("SPATIALLY STRATIFIED PROGRESSIVE SOURCE SAMPLING")
    print("=" * 70)

    print(f"Loading GLA graph from {gla_cache}")
    with open(gla_cache, "rb") as f:
        G = pickle.load(f)
    print(f"GLA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load GLA boundary for live mask
    import geopandas as gpd
    from shapely.geometry import Point

    boundary_cache = CACHE_DIR / "gla_boundary.geojson"
    if not boundary_cache.exists():
        raise FileNotFoundError(
            f"GLA boundary cache not found: {boundary_cache}\n"
            "  Run 03_validate_gla.py first."
        )
    gla_boundary = gpd.read_file(boundary_cache).geometry.iloc[0]

    print("Marking live nodes...")
    n_live = 0
    for _n, data in G.nodes(data=True):
        data["live"] = gla_boundary.contains(Point(data["x"], data["y"]))
        n_live += data["live"]
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()}")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, _, net = io.network_structure_from_nx(G)
    live_mask = nodes_gdf["live"].values
    n_nodes = len(nodes_gdf)

    # ---------------------------------------------------------------
    # Per-distance progressive sampling
    # ---------------------------------------------------------------
    results = []

    for dist in DISTANCES:
        print(f"\n{'=' * 60}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 60}")

        # Ground truth
        gt_cache = CACHE_DIR / f"gla_ground_truth_{dist}m.pkl"
        if gt_cache.exists():
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_betw = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            baseline_time = gt_data.get("baseline_betw_time", None)
        else:
            print("  Computing exact betweenness (ground truth)...")
            t0 = time.time()
            betw_r = net.betweenness_exact_shortest(
                distances=[dist], pbar_disabled=False,
            )
            baseline_time = time.time() - t0
            true_betw = np.array(betw_r.node_betweenness[dist])[live_mask]
            close_r = net.closeness_shortest(distances=[dist], pbar_disabled=True)
            node_reach = np.array(close_r.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))
            with open(gt_cache, "wb") as f:
                pickle.dump({
                    "betweenness": true_betw,
                    "mean_reach": mean_reach,
                    "node_reach": node_reach,
                    "n_live": n_live,
                    "baseline_betw_time": baseline_time,
                }, f)
            print(f"    {baseline_time:.1f}s")

        # Build spatial grid for this distance
        cell_size = dist / 2.0
        cells = build_spatial_grid(net, cell_size)
        n_cells = len(cells)
        sources_per_round = sum(1 for c in cells.values() if c)

        print(f"  Reach: {mean_reach:,.0f}")
        print(f"  Grid: cell_size={cell_size:.0f}m, {n_cells} cells, {sources_per_round} sources/round")
        if baseline_time:
            print(f"  Exact baseline: {baseline_time:.1f}s")

        # Progressive sampling
        rng = random.Random(BASE_SEED)
        sampler = CellSampler(cells, rng)
        accumulated_betw = np.zeros(n_nodes, dtype=np.float64)
        total_sources = 0
        t_start = time.time()
        prev_betw_live = None
        converged_round = None

        round_data = []

        for rnd in range(1, MAX_ROUNDS + 1):
            # Select sources for this round (without replacement per cell)
            round_sources = sampler.sample_round()
            n_round = len(round_sources)

            # Run Brandes from these sources
            r = net.betweenness_exact_shortest(
                distances=[dist],
                source_indices=round_sources,
                pbar_disabled=True,
            )
            # The result is scaled by n_live / (2 * n_sources) internally.
            # We need to accumulate raw (unscaled) and rescale at the end.
            # Actually, each call returns an independent estimate.
            # To combine: average across rounds.
            round_betw = np.array(r.node_betweenness[dist], dtype=np.float64)

            # Running average: accumulated = sum of per-round estimates / n_rounds
            accumulated_betw += round_betw
            total_sources += n_round
            current_avg = accumulated_betw / rnd

            # Extract live nodes for convergence check
            current_betw_live = current_avg[live_mask]

            elapsed = time.time() - t_start

            # Compute accuracy against ground truth
            sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_betw, current_betw_live)

            # Convergence check: ρ between successive rounds
            conv_rho = np.nan
            if prev_betw_live is not None:
                mask = (current_betw_live > 0) & (prev_betw_live > 0)
                if mask.sum() > 10:
                    conv_rho, _ = scipy_stats.spearmanr(
                        current_betw_live[mask], prev_betw_live[mask]
                    )

            speedup = baseline_time / elapsed if baseline_time and elapsed > 0 else np.nan

            round_data.append({
                "distance": dist,
                "round": rnd,
                "n_sources_round": n_round,
                "total_sources": total_sources,
                "frac_sources": total_sources / n_live,
                "elapsed": elapsed,
                "spearman": sp,
                "top_k_precision": prec,
                "scale_ratio": scale,
                "scale_iqr": iqr,
                "max_abs_error": mae,
                "conv_rho": conv_rho,
                "speedup": speedup,
                "mean_reach": mean_reach,
                "baseline_time": baseline_time,
            })

            status = f"ρ_gt={sp:.4f}" if not np.isnan(sp) else "ρ_gt=nan"
            if not np.isnan(conv_rho):
                status += f", ρ_conv={conv_rho:.6f}"

            print(f"  Round {rnd:>2}: {n_round} src, total={total_sources}, "
                  f"{elapsed:.1f}s, {status}, speedup={speedup:.1f}x")

            prev_betw_live = current_betw_live.copy()

            # Check convergence
            if rnd >= MIN_ROUNDS and conv_rho >= RHO_THRESHOLD:
                converged_round = rnd
                print(f"  ** Converged at round {rnd} (ρ_conv={conv_rho:.6f} >= {RHO_THRESHOLD})")
                break

        if converged_round is None:
            print(f"  ** Did not converge after {MAX_ROUNDS} rounds")

        results.extend(round_data)

    # ---------------------------------------------------------------
    # Save and summarise
    # ---------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: Spatially Stratified Progressive Source Sampling")
    print("=" * 70)

    print(
        f"\n{'Dist':>8} | {'Reach':>10} | {'Rounds':>7} | {'Sources':>10} | "
        f"{'Frac':>7} | {'ρ_gt':>8} | {'Top-k':>7} | {'Time':>8} | {'Speedup':>8}"
    )
    print("-" * 95)

    for dist in DISTANCES:
        d_rows = df[df["distance"] == dist]
        last = d_rows.iloc[-1]
        print(
            f"{dist // 1000}km    | {last['mean_reach']:>10,.0f} | "
            f"{int(last['round']):>7} | {int(last['total_sources']):>10,} | "
            f"{last['frac_sources']:>6.1%} | "
            f"{last['spearman']:>8.4f} | {last['top_k_precision']:>7.3f} | "
            f"{last['elapsed']:>7.1f}s | {last['speedup']:>7.1f}x"
        )

    print(f"\n  Saved: {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Spatially stratified progressive source sampling for betweenness"
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()

    print("=" * 70)
    print("04_global_betweenness.py - Stratified Progressive Sampling")
    print("=" * 70)

    run_stratified_betweenness(force=args.force)
    return 0


if __name__ == "__main__":
    exit(main())
