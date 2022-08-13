"""
Observe module for computing observations derived from `networkX` graphs.

These methods are generally sufficiently simple that further computational optimisation is not required. Network
centrality methods (which do require further computational optimisation due to their complexity) are handled separately
in the [`networks`](/metrics/networks/) module.

"""
from __future__ import annotations

import logging
from typing import Any, Union

import networkx as nx
import numpy as np
from tqdm import tqdm

from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuityEntry:
    """State management for an individual street continuity entry."""

    entry_name: str
    count: int
    length: float
    edges: list[tuple[graphs.NodeKey, graphs.NodeKey, int]]

    def __init__(self, entry_name: str) -> None:
        """Instances a continuity entry."""
        self.entry_name = entry_name
        self.count = 0
        self.length = 0
        self.edges = []

    def add_edge(self, length: float, start_nd_key: graphs.NodeKey, end_nd_key: graphs.NodeKey, edge_idx: int) -> None:
        """Adds edge details to a continuity entry."""
        self.count += 1
        self.length += length
        self.edges.append((start_nd_key, end_nd_key, edge_idx))


class StreetContinuityReport:
    """State management for a collection of street continuity metrics."""

    entries: dict[str, ContinuityEntry]

    def __init__(self) -> None:
        """Instance a street contunity report."""
        self.entries = {}
        self.sorted_entries_idx = []

    def scaffold_entry(self, entry_name: str) -> None:
        """Adds a new continuity entry to the report's entries."""
        if entry_name in self.entries:
            raise ValueError(f"Cannot create a duplicate entry for {entry_name}")
        self.entries[entry_name] = ContinuityEntry(entry_name)

    def report_by_count(self, n_items: int = 10) -> None:
        """Print a report sorted by entry counts."""
        logger.info(f"Reporting top {n_items} continuity observations by street counts.")
        item_keys = list(self.entries.keys())
        counts: list[int] = [v.count for v in self.entries.values()]
        for n, arg_idx in enumerate(np.argsort(counts)[::-1]):  # type: ignore
            if n == n_items:
                break
            logger.info(f"Count: {counts[arg_idx]} - {item_keys[arg_idx]}")

    def report_by_length(self, n_items: int = 10) -> None:
        """Print a report sorted by entry lengths."""
        logger.info(f"Reporting top {n_items} continuity observations by street lengths.")
        item_keys = list(self.entries.keys())
        counts: list[float] = [v.length for v in self.entries.values()]
        for n, arg_idx in enumerate(np.argsort(counts)[::-1]):  # type: ignore
            if n == n_items:
                break
            logger.info(f"Length: {round(counts[arg_idx] / 1000, 2)}km - {item_keys[arg_idx]}")


def route_continuity(
    nx_multigraph: nx.MultiGraph, method: Union[str, tuple[str, str]]
) -> tuple[nx.MultiGraph, StreetContinuityReport]:
    """
    Compute the route continuity for a given graph.
    """
    # NOTE: experimented with removing generic descriptors but this worked contrary to intentions, so has been removed.

    nx_multi_copy: nx.MultiGraph = nx_multigraph.copy()
    # check intended method keys
    available_targets = ["names", "routes", "highways"]
    if method not in available_targets:
        raise ValueError(f"Method of {method} is not recognised.")

    # keys for writing data back to graph
    count_edge_key = f"{method}_cont_by_count"
    length_edge_key = f"{method}_cont_by_length"

    def _recurse_edges(
        _nx_multigraph: nx.MultiGraph,
        _method: str,
        _match_target: str,
        _a_nd_key: graphs.NodeKey,
        _b_nd_key: graphs.NodeKey,
        _edge_idx: int,
        _visited_edges: list[str],
        _contunity_report: StreetContinuityReport,
    ):
        """Used when using strictly continuous implementations, e.g. for street names or routes."""
        # generate an edge key
        edge_nodes = tuple(sorted([str(_a_nd_key), str(_b_nd_key)]))
        edge_key = f"{edge_nodes[0]}_{edge_nodes[1]}_{_edge_idx}"
        # skip if already visited
        if edge_key in _visited_edges:
            return
        _visited_edges.append(edge_key)
        # extract edge data for the current target key
        nested_edge_data: graphs.EdgeData = _nx_multigraph[_a_nd_key][_b_nd_key][_edge_idx]
        if _method not in nested_edge_data:
            raise ValueError(f"Missing target key of {_method}")
        nested_match_targets: set[str] = nested_edge_data[_method]
        # bail if this edge info (e.g. street name) doesn't intersect with the previous
        if not _match_target.lower() in [nmt.lower() for nmt in nested_match_targets if nmt is not None]:
            return
        _contunity_report.entries[_match_target].add_edge(
            length=nested_edge_data["geom"].length, start_nd_key=_a_nd_key, end_nd_key=_b_nd_key, edge_idx=_edge_idx
        )
        # find all neighbouring edge pairs
        a_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
            (_a_nd_key, ann) for ann in nx.neighbors(_nx_multigraph, _a_nd_key) if ann != _b_nd_key  # type: ignore
        ]
        b_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
            (_b_nd_key, bnn) for bnn in nx.neighbors(_nx_multigraph, _b_nd_key) if bnn != _a_nd_key  # type: ignore
        ]
        # recurse into neighbours
        for nested_a_nd_key, nested_b_nd_key in a_nb_pairs + b_nb_pairs:
            nested_edge_idx: int
            for nested_edge_idx in _nx_multigraph[nested_a_nd_key][nested_b_nd_key].keys():
                _recurse_edges(
                    _nx_multigraph,
                    _method,
                    _match_target,
                    nested_a_nd_key,
                    nested_b_nd_key,
                    nested_edge_idx,
                    _visited_edges,
                    _contunity_report,
                )

    # instance StreetContinuityReport
    street_continuity = StreetContinuityReport()

    # iter edges
    a_nd_key: graphs.NodeKey
    b_nd_key: graphs.NodeKey
    edge_idx: int
    edge_data: dict[str, Any]
    for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
        # raise if the key doesn't exist
        if method not in edge_data:
            raise ValueError(f"Could not find {method} in edge data for edge: {a_nd_key} - {b_nd_key} idx: {edge_idx}")
        # get target values, e.g. street names
        match_targets: list[str] = edge_data[method]
        # can be multiple values, e.g. route numbers per edge, so explore individually
        for match_target in match_targets:
            if match_target is None:
                continue
            # if using street names or route refs, then use continuous method
            if method in ["names", "routes"]:
                # no need to redo already explored values
                # NOTE: this assumes street names or routes are continuous
                if match_target not in street_continuity.entries:
                    street_continuity.scaffold_entry(entry_name=match_target)
                    visited_edges: list[str] = []
                    _recurse_edges(
                        nx_multi_copy,
                        method,
                        match_target,
                        a_nd_key,
                        b_nd_key,
                        edge_idx,
                        visited_edges,
                        street_continuity,
                    )
            # don't use recursion for highway types because these are not continuous
            else:
                if match_target not in street_continuity.entries:
                    street_continuity.scaffold_entry(entry_name=match_target)
                street_continuity.entries[match_target].add_edge(
                    edge_data["geom"].length, start_nd_key=a_nd_key, end_nd_key=b_nd_key, edge_idx=edge_idx
                )

    # write data from StreetContinuityReport to graph
    for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
        match_targets: list[str] = edge_data[method]
        # if empty, then zero
        if not match_targets:
            nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key] = 0
            nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key] = 0
            continue
        # in case of multiple valid values, use the greater
        for match_target in match_targets:
            if match_target is None:
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key] = 0
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key] = 0
                continue
            # update graph count key
            agg_count = street_continuity.entries[match_target].count
            if (
                count_edge_key not in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]
                or agg_count > nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key]
            ):
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key] = agg_count
            # update graph length key
            agg_length = street_continuity.entries[match_target].length
            if (
                length_edge_key not in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]
                or agg_length > nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key]
            ):
                nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key] = agg_length

    return nx_multi_copy, street_continuity
