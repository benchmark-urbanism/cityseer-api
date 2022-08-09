"""
Observe module for computing observations derived from `networkX` graphs.

These methods are generally sufficiently simple that further computational optimisation is not required. Network
centrality methods (which do require further computational optimisation due to their complexity) are handled separately
in the [`networks`](/metrics/networks/) module.

"""
from typing import Any

import networkx as nx
from tqdm import tqdm
import numpy as np
import numpy.typing as npt

from cityseer.tools import graphs


def route_continuity(nx_multigraph: nx.MultiGraph, method: str) -> nx.MultiGraph:
    """
    Compute the route continuity for a given graph.
    """
    nx_multi_copy: nx.MultiGraph = nx_multigraph.copy()

    # clean values and save as sets
    available_targets = ["names", "refs", "highways"]
    ignore_vals = set(
        [
            # street name cleaning
            "street",
            "way",
            "st",
            "ave",
            "avenue",
            "highway",
            "drive",
            "gardens",
            "lane",
            "close",
            "gate",
            "the",
            "road",
            "court",
            "mews",
            "crescent",
            "place",
            "grove",
            "bridge",
            "walk",
            "of",
            "park",
            "hill",
            "square",
            "passage",
            "wharf",
            "row",
            "terrace",
            "harbour",
            "junction",
            "end",
            "rise",
            "upper",
            "lower",
            "quays",
            "mall",
            "yard",
            "approach",
            # highway cleaning
            "local",
            "unnumbered",
            "minor",
            "access",
        ]
    )
    for target in available_targets:
        start_nd_key: graphs.NodeKey
        end_nd_key: graphs.NodeKey
        edge_idx: int
        edge_data: graphs.EdgeData
        for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(data=True, keys=True)):  # type: ignore
            if target not in edge_data:
                nx_multi_copy[start_nd_key][end_nd_key][edge_idx][target] = None
            elif edge_data[target] is not None:
                vals = edge_data[target]
                # cleaned: set[str] = set()
                # NOTE: deep cleaning seems to work against intention
                # for v in vals:
                #     v = v.lower()
                #     vs = v.split(" ")
                #     cleaned.update(vs)
                # cleaned.difference_update(ignore_vals)
                vals = [v.lower() for v in vals]
                nx_multi_copy[start_nd_key][end_nd_key][edge_idx][f"{target}_clean"] = vals

    def _recurse_edges(
        _nx_multigraph: nx.MultiGraph,
        _target_key: str,
        _target_vals: set[str],
        _a_nd_key: graphs.NodeKey,
        _b_nd_key: graphs.NodeKey,
        _edge_idx: int,
        agg_edge_lengths: list[float],
        visited_edges: list[str],
    ):
        edge_nodes = tuple(sorted([str(_a_nd_key), str(_b_nd_key)]))
        edge_key = f"{edge_nodes[0]}_{edge_nodes[1]}_{_edge_idx}"
        if edge_key in visited_edges:
            return
        visited_edges.append(edge_key)
        nested_edge_data: graphs.EdgeData = _nx_multigraph[_a_nd_key][_b_nd_key][_edge_idx]
        if _target_key not in nested_edge_data:
            raise ValueError(f"Missing target key of {_target_key}")
        nested_target_vals: set[str] = nested_edge_data[_target_key]
        if not set(_target_vals).intersection(set(nested_target_vals)):
            return
        agg_edge_lengths.append(nested_edge_data["geom"].length)
        # find all neighbouring edge pairs
        a_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
            (_a_nd_key, ann) for ann in nx.neighbors(_nx_multigraph, _a_nd_key) if ann != _b_nd_key  # type: ignore
        ]
        b_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
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

    if method in available_targets:
        target_key: str = f"{method}_clean"
    else:
        raise ValueError(f"Method of {method} is not recognised.")

    # iter edges
    a_nd_key: graphs.NodeKey
    b_nd_key: graphs.NodeKey
    edge_idx: int
    edge_data: dict[str, Any]
    edge_counts: list[int] = []
    edge_names: list[str] = []
    for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
        if target_key not in edge_data:
            raise ValueError(f"Missing target key of {target_key}")
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
        # reporting
        edge_counts.append(agg_count)
        edge_names.append(target_vals)

    arg_sorted: npt.NDArray[np.uint] = np.argsort(np.array(edge_counts))[::-1]
    names_dict: dict[str, list[int]] = {}
    counter = 0
    for arg_idx in arg_sorted:
        if counter > 100:
            break
        name = str.join(", ", edge_names[arg_idx])
        count = edge_counts[arg_idx]
        if name not in names_dict:
            names_dict[name] = [count]
            counter += 1
        else:
            names_dict[name].append(count)
    print(names_dict)

    return nx_multi_copy
