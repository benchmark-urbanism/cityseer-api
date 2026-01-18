# %% Test substrates for sampling analysis
"""
Explore the substrates module to understand network statistics.
Outputs visualizations and stats tables for:
- Named quadrant templates: trellis, tree, neighbourhood, linear, islands
- Gradated templates: connectivity levels 1-10 (tiles=1 and tiles=2)

Parameters demonstrated:
- gradation: 1-10, controls connectivity (1=sparse/tree-like, 10=dense/grid-like)
- tiles: number of tiles in each direction (tiles=2 gives 2×2 grid)
- decompose: edge decomposition length (None=no decomposition)
- weld_edges: whether to weld opposite boundaries (False for realistic bounded networks)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils.substrates import generate_gradated_template, generate_keyed_template, quadrant_templates


def calculate_stats(G: nx.MultiGraph, label: str | int, tiles: int = 1) -> dict:
    """Calculate network statistics for a graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    edge_lengths = []
    for _u, _v, d in G.edges(data=True):
        if "geom" in d:
            edge_lengths.append(d["geom"].length)
    avg_length = np.mean(edge_lengths) if edge_lengths else 0

    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    x_extent = max(xs) - min(xs)
    y_extent = max(ys) - min(ys)

    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0

    return {
        "label": label,
        "tiles": tiles,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "avg_length": avg_length,
        "x_extent": x_extent,
        "y_extent": y_extent,
    }


def print_stats_table(stats: list[dict], title: str) -> None:
    """Print a formatted statistics table."""
    print(f"\n{title}")
    print("-" * 105)
    print(
        f"{'Label':>13} | {'Tiles':>5} | {'Nodes':>6} | {'Edges':>6} | "
        f"{'AvgDeg':>6} | {'AvgLen':>7} | {'X Extent':>8} | {'Y Extent':>8}"
    )
    print("-" * 105)
    for s in stats:
        print(
            f"{str(s['label']):>13} | {s['tiles']:>5} | {s['n_nodes']:>6} | {s['n_edges']:>6} | "
            f"{s['avg_degree']:>6.2f} | {s['avg_length']:>7.1f} | {s['x_extent']:>8.0f} | {s['y_extent']:>8.0f}"
        )
    print("-" * 105)


# %% Generate and visualize named templates (tiles=1)
template_names = list(quadrant_templates.keys())
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

named_stats = []

for i, name in enumerate(template_names):
    G, _, _ = generate_keyed_template(
        template_key=name,
        tiles=1,
        decompose=None,
        weld_edges=False,
        plot=False,
    )

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    stats = calculate_stats(G, name, tiles=1)
    named_stats.append(stats)

    ax = axes[i]
    nx.draw(G, pos, ax=ax, node_size=5, width=0.5, with_labels=False)
    ax.set_title(f"{name.title()}\n{stats['n_nodes']} nodes, deg={stats['avg_degree']:.2f}")
    ax.set_aspect("equal")

plt.suptitle("Named Templates (tiles=1, ~980m extent)", fontsize=14)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "output" / "test_substrates_named.png", dpi=150)
plt.show()

# %% Generate and visualize named templates (tiles=2)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

named_tiled_stats = []

for i, name in enumerate(template_names):
    G, _, _ = generate_keyed_template(
        template_key=name,
        tiles=2,
        decompose=None,
        weld_edges=False,
        plot=False,
    )

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    stats = calculate_stats(G, name, tiles=2)
    named_tiled_stats.append(stats)

    ax = axes[i]
    nx.draw(G, pos, ax=ax, node_size=3, width=0.3, with_labels=False)
    ax.set_title(f"{name.title()}\n{stats['n_nodes']} nodes, deg={stats['avg_degree']:.2f}")
    ax.set_aspect("equal")

plt.suptitle("Named Templates (tiles=2, ~1960m extent)", fontsize=14)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "output" / "test_substrates_named_tiled.png", dpi=150)
plt.show()

# %% Generate and visualize base gradations (tiles=1)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

base_stats = []

for i, gradation in enumerate(range(1, 11)):
    G, _, _ = generate_gradated_template(
        gradation=gradation,
        tiles=1,
        decompose=None,
        weld_edges=False,
        plot=False,
    )

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    stats = calculate_stats(G, gradation, tiles=1)
    base_stats.append(stats)

    ax = axes[i]
    nx.draw(G, pos, ax=ax, node_size=5, width=0.5, with_labels=False)
    ax.set_title(f"Gradation {gradation}\n{stats['n_nodes']} nodes, deg={stats['avg_degree']:.2f}")
    ax.set_aspect("equal")

plt.suptitle("Gradated Substrates (tiles=1, ~980m extent)", fontsize=14)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "output" / "test_substrates_base.png", dpi=150)
plt.show()

# %% Generate and visualize tiled gradations (tiles=2 for ~1960m extent)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

tiled_stats = []

for i, gradation in enumerate(range(1, 11)):
    G, _, _ = generate_gradated_template(
        gradation=gradation,
        tiles=2,
        decompose=None,
        weld_edges=False,
        plot=False,
    )

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    stats = calculate_stats(G, gradation, tiles=2)
    tiled_stats.append(stats)

    ax = axes[i]
    nx.draw(G, pos, ax=ax, node_size=2, width=0.3, with_labels=False)
    ax.set_title(f"Gradation {gradation}\n{stats['n_nodes']} nodes, deg={stats['avg_degree']:.2f}")
    ax.set_aspect("equal")

plt.suptitle("Tiled Substrates (tiles=2, ~1960m extent)", fontsize=14)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "output" / "test_substrates_tiled.png", dpi=150)
plt.show()

# %% Print statistics
print_stats_table(named_stats, "Named Template Statistics (tiles=1, ~980m extent)")
print_stats_table(named_tiled_stats, "Named Template Statistics (tiles=2, ~1960m extent)")
print_stats_table(base_stats, "Gradated Substrate Statistics (tiles=1, ~980m extent)")
print_stats_table(tiled_stats, "Gradated Substrate Statistics (tiles=2, ~1960m extent)")
