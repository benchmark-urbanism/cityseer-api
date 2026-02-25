#!/usr/bin/env python
"""
Smoke test: spatially stratified progressive source sampling on synthetic graphs.

Uses the same substrate generation as 00_generate_cache.py (generate_keyed_template).

Tests:
  1. source_indices=all_live matches exact (correctness)
  2. Progressive stratified sampling converges (rankings improve with rounds)
"""

import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from cityseer.tools import io
from scipy import stats as scipy_stats

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities import apply_live_buffer_nx, compute_accuracy_metrics
from utils.substrates import generate_keyed_template


def build_spatial_grid(net, cell_size):
    """Partition live nodes into spatial grid cells."""
    all_xs = net.node_xs
    all_ys = net.node_ys
    live_indices = [i for i in net.node_indices() if net.is_node_live(i)]
    if not live_indices:
        return {}
    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]
    x_min, y_min = min(xs), min(ys)
    cells = {}
    for idx, x, y in zip(live_indices, xs, ys):
        cx = int((x - x_min) / cell_size)
        cy = int((y - y_min) / cell_size)
        cells.setdefault((cx, cy), []).append(idx)
    return cells


class CellSampler:
    """Without-replacement sampler across grid cells."""

    def __init__(self, cells, rng):
        self.rng = rng
        self.decks = {}
        self.pools = {}
        for key, nodes in cells.items():
            if nodes:
                self.pools[key] = list(nodes)
                deck = list(nodes)
                rng.shuffle(deck)
                self.decks[key] = deck

    def sample_round(self):
        sources = []
        for key, deck in self.decks.items():
            if not deck:
                deck = list(self.pools[key])
                self.rng.shuffle(deck)
                self.decks[key] = deck
            sources.append(deck.pop())
        return sources


def test_correctness(net, live_mask, dist):
    """Verify source_indices=all_live matches exact."""
    print(f"\n--- Test 1: source_indices correctness (dist={dist}m) ---")
    exact_r = net.betweenness_shortest(distances=[dist], pbar_disabled=True)
    true_betw = np.array(exact_r.node_betweenness[dist], dtype=np.float64)

    all_live = [i for i in net.node_indices() if net.is_node_live(i)]
    check_r = net.betweenness_shortest(
        distances=[dist], source_indices=all_live, pbar_disabled=True,
    )
    check_betw = np.array(check_r.node_betweenness[dist], dtype=np.float64)

    max_diff = np.max(np.abs(true_betw - check_betw))
    max_val = np.max(np.abs(true_betw))
    rel_diff = max_diff / (max_val + 1e-10)
    print(f"  Max abs diff: {max_diff:.6f}, max betw: {max_val:.1f}, relative: {rel_diff:.8f}")

    if rel_diff > 0.01:
        print(f"  FAIL — relative diff {rel_diff:.4f} > 0.01")
        return False, true_betw
    print("  PASS")
    return True, true_betw


def test_progressive(net, live_mask, dist, true_betw, max_rounds=30):
    """Run progressive stratified sampling and check that accuracy improves."""
    n_nodes = len(live_mask)
    n_live = int(live_mask.sum())
    true_betw_live = true_betw[live_mask]
    nonzero = np.sum(true_betw_live > 0)
    betw_cv = np.std(true_betw_live) / (np.mean(true_betw_live) + 1e-10)

    print(f"\n--- Test 2: progressive stratified sampling (dist={dist}m) ---")
    print(f"  n_live={n_live}, nonzero={nonzero}, max={true_betw_live.max():.1f}, "
          f"mean={true_betw_live.mean():.1f}, CV={betw_cv:.2f}")

    cell_size = dist / 2.0
    cells = build_spatial_grid(net, cell_size)
    n_cells = len(cells)
    sources_per_round = sum(1 for c in cells.values() if c)
    print(f"  Grid: cell_size={cell_size:.0f}m, {n_cells} cells, {sources_per_round} src/round")

    rng = random.Random(42)
    sampler = CellSampler(cells, rng)
    accumulated = np.zeros(n_nodes, dtype=np.float64)
    prev_live = None
    rho_history = []

    for rnd in range(1, max_rounds + 1):
        round_sources = sampler.sample_round()
        r = net.betweenness_shortest(
            distances=[dist], source_indices=round_sources, pbar_disabled=True,
        )
        round_betw = np.array(r.node_betweenness[dist], dtype=np.float64)
        accumulated += round_betw
        current_avg = accumulated / rnd
        current_live = current_avg[live_mask]

        sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_betw_live, current_live)

        conv_rho = np.nan
        if prev_live is not None:
            mask = (current_live > 0) & (prev_live > 0)
            if mask.sum() > 10:
                conv_rho, _ = scipy_stats.spearmanr(current_live[mask], prev_live[mask])

        total_src = rnd * sources_per_round
        rho_history.append(sp)
        print(f"  Round {rnd:>2}: {sources_per_round} src ({total_src}/{n_live}={total_src / n_live:.0%}), "
              f"ρ_gt={sp:.4f}, top-k={prec:.3f}, scale={scale:.3f}, "
              f"ρ_conv={conv_rho:.6f}")

        prev_live = current_live.copy()

    # Check that accuracy monotonically improves (trend)
    # Compare first half vs second half
    mid = len(rho_history) // 2
    first_half = np.mean(rho_history[:mid])
    second_half = np.mean(rho_history[mid:])
    final_rho = rho_history[-1]

    print(f"\n  First-half avg ρ: {first_half:.4f}")
    print(f"  Second-half avg ρ: {second_half:.4f}")
    print(f"  Final ρ: {final_rho:.4f}")

    # Pass criteria:
    # 1. ρ should improve (second half better than first half)
    # 2. Final ρ should be positive (better than random)
    improving = second_half > first_half
    positive = final_rho > 0.0

    if improving and positive:
        print("  PASS — accuracy improves with more sources")
        return True
    else:
        if not improving:
            print("  FAIL — accuracy not improving")
        if not positive:
            print("  FAIL — final ρ not positive")
        return False


def main():
    print("=" * 60)
    print("SMOKE TEST: Stratified Progressive Source Sampling")
    print("=" * 60)

    # Use "tree" topology — more heterogeneous betweenness than trellis
    topo = "tree"
    tiles = 6  # ~6km extent, manageable size
    dist = 1000  # 1km catchments
    buffer = 2000

    print(f"\nGenerating '{topo}' substrate (tiles={tiles})...")
    G, _, _ = generate_keyed_template(template_key=topo, tiles=tiles, decompose=None, plot=False)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    G = apply_live_buffer_nx(G, buffer)

    nodes_gdf, _, net = io.network_structure_from_nx(G)
    live_mask = nodes_gdf["live"].values
    n_live = int(live_mask.sum())
    print(f"  Live nodes: {n_live}/{len(nodes_gdf)}")

    ok1, true_betw = test_correctness(net, live_mask, dist)
    ok2 = test_progressive(net, live_mask, dist, true_betw, max_rounds=20)

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("ALL TESTS PASSED")
        return 0
    else:
        fails = []
        if not ok1:
            fails.append("correctness")
        if not ok2:
            fails.append("progressive convergence")
        print(f"FAILED: {', '.join(fails)}")
        return 1


if __name__ == "__main__":
    exit(main())
