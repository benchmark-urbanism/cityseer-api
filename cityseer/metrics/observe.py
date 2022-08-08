"""
Observe module for computing observations derived from `networkX` graphs.

These methods are generally sufficiently simple that further computational optimisation is not required. Network
centrality methods (which do require further computational optimisation due to their complexity) are handled separately
in the [`networks`](/metrics/networks/) module.

"""
from typing import Any

import networkx as nx
from tqdm import tqdm

from cityseer.tools.graphs import EdgeData, NodeKey


def route_continuity(nx_multigraph: nx.MultiGraph, method: str) -> nx.MultiGraph:
    """
    Compute the route continuity for a given graph.
    """
    nx_multi_copy: nx.MultiGraph = nx_multigraph.copy()

    def _clean_vals(vals: list[str]) -> set[str]:
        clean_vals: list[str] = []
        for val in vals:
            # highways category has residential, service, etc.
            if val not in [None, "residential", "service", "footway"]:
                clean_vals.append(val)
        return set(clean_vals)

    def _intersect_vals(vals_a: list[str], vals_b: list[str]) -> bool:
        """Find set overlaps between values for set A and set B."""
        clean_vals_a = _clean_vals(vals_a)
        clean_vals_b = _clean_vals(vals_b)
        itx = clean_vals_a.intersection(clean_vals_b)
        return len(itx) > 0

    def _recurse_edges(
        _nx_multigraph: nx.MultiGraph,
        _target_key: str,
        _target_vals: list[str],
        _a_nd_key: NodeKey,
        _b_nd_key: NodeKey,
        _edge_idx: int,
        agg_edge_lengths: list[float],
        visited_edges: list[str],
    ):
        edge_nodes = tuple(sorted([str(_a_nd_key), str(_b_nd_key)]))
        edge_key = f"{edge_nodes[0]}_{edge_nodes[1]}_{_edge_idx}"
        if edge_key in visited_edges:
            return
        visited_edges.append(edge_key)
        nested_edge_data: EdgeData = _nx_multigraph[_a_nd_key][_b_nd_key][_edge_idx]
        if _target_key not in nested_edge_data:
            return
        nested_target_vals: list[str] = nested_edge_data[_target_key]
        if not _intersect_vals(_target_vals, nested_target_vals):
            return
        agg_edge_lengths.append(nested_edge_data["geom"].length)
        # find all neighbouring edge pairs
        a_nb_pairs: list[tuple[NodeKey, NodeKey]] = [
            (_a_nd_key, ann) for ann in nx.neighbors(_nx_multigraph, _a_nd_key) if ann != _b_nd_key  # type: ignore
        ]
        b_nb_pairs: list[tuple[NodeKey, NodeKey]] = [
            (_b_nd_key, bnn) for bnn in nx.neighbors(_nx_multigraph, _b_nd_key) if bnn != _a_nd_key  # type: ignore
        ]
        for nested_a_nd_key, nested_b_nd_key in a_nb_pairs + b_nb_pairs:
            nested_edge_idx: int
            for nested_edge_idx in _nx_multigraph[nested_a_nd_key][nested_b_nd_key].keys():
                _recurse_edges(
                    _nx_multigraph,
                    _target_key,
                    _target_vals,
                    nested_a_nd_key,
                    nested_b_nd_key,
                    nested_edge_idx,
                    agg_edge_lengths,
                    visited_edges,
                )

    if method in ["names", "refs", "highways"]:
        target_key: str = method
    else:
        raise ValueError(f"Method of {method} is not recognised.")

    # iter edges
    edge_data: dict[str, Any]
    a_nd_key: NodeKey
    b_nd_key: NodeKey
    edge_idx: int
    for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
        if target_key not in edge_data:
            nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg"] = None  # type: ignore
        target_vals = edge_data[target_key]
        agg_edge_lengths: list[float] = []
        visited_edges: list[str] = []
        _recurse_edges(
            nx_multi_copy, target_key, target_vals, a_nd_key, b_nd_key, edge_idx, agg_edge_lengths, visited_edges
        )
        # length sum
        agg_len = sum(agg_edge_lengths)
        if f"{target_key}_agg" in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]:
            current_agg_len: float = nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_length"]
            if agg_len > current_agg_len:
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_length"] = agg_len / 1000
        else:
            nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_length"] = agg_len / 1000
        # counts
        agg_count = len(agg_edge_lengths)
        if f"{target_key}_agg_count" in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]:
            current_agg_count: float = nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_count"]
            if agg_count > current_agg_count:
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_count"] = agg_count
        else:
            nx_multi_copy[a_nd_key][b_nd_key][edge_idx][f"{target_key}_agg_count"] = agg_count

    return nx_multi_copy
