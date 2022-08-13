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


class _ContinuityEntry:
    """
    State management for an individual street continuity entry.

    This corresponds to an individual street name, route name or number, or highway type.
    """

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


class _StreetContinuityReport:
    """
    State management for a collection of street continuity metrics.

    Each key in the `entries` attribute corresponds to a `_ContinuityEntry`.
    """

    entries: dict[str, _ContinuityEntry]

    def __init__(self) -> None:
        """Instance a street contunity report."""
        self.entries = {}
        self.sorted_entries_idx = []

    def scaffold_entry(self, entry_name: str) -> None:
        """Adds a new continuity entry to the report's entries."""
        if entry_name in self.entries:
            raise ValueError(f"Cannot create a duplicate entry for {entry_name}")
        self.entries[entry_name] = _ContinuityEntry(entry_name)

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
) -> tuple[nx.MultiGraph, _StreetContinuityReport]:
    """
    Compute the route continuity for a given graph.

    This requires a graph with `names`, `routes`, or `highways` edge keys corresponding to the selected `method`
    parameter. These keys are available if importing an OSM network with
    [`osm_graph_from_poly_wgs`](/tools/io#osm-graph-from-poly-wgs) or
    [nx_from_open_roads](/tools/io#nx-from-open-roads).

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms. Edges should contain "names", "routes", or "highways" keys
        corresponding to the specified `method` parameter.
    method: str
        The type of continuity metric to compute, where available options are "names", "routes", "highways", or
        "hybrid". The latter of which computes a combines metric derived from the "names" and "routes" keys.

    Returns
    -------
    MultiGraph
        A copy of the input `networkX` `MultiGraph` with new edge keys corresponding to the calculated route continuity
        metric. The metrics will be stored in 'length' and 'count' forms for the specified method, with keys formatted
        according to `f"{method}_cont_by_{form}"`. For example, when computing "names" continuity, the
        `names_cont_by_count` and `names_cont_by_length` keys will be added to the returned `networkX` `MultiGraph`.
    _StreetContinuityReport:
        An instance of [`_StreetContinuityReport](/metrics/observe#streetcontinuityreport) containing the computed
        state for the selected method.

    """
    # NOTE: experimented with string cleaning and removal of generic descriptors but this worked contrary to intentions.

    nx_multi_copy: nx.MultiGraph = nx_multigraph.copy()
    # check intended method keys
    available_targets = ["names", "routes", "highways", "hybrid"]
    if method not in available_targets:
        raise ValueError(f"Method of {method} is not recognised.")

    def _recurse_edges(
        _nx_multigraph: nx.MultiGraph,
        _method: str,
        _match_target: str,
        _a_nd_key: graphs.NodeKey,
        _b_nd_key: graphs.NodeKey,
        _edge_idx: int,
        _visited_edges: list[str],
        _contunity_report: _StreetContinuityReport,
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

    # hybrid requires names and routes methods
    if method == "hybrid":
        do_methods = ["routes", "names"]
        method_reports = [_StreetContinuityReport(), _StreetContinuityReport()]
    else:
        do_methods = [method]
        method_reports = [_StreetContinuityReport()]

    for do_method, method_report in zip(do_methods, method_reports):
        logger.info(f"Calculating metrics for {do_method}.")
        # keys for writing data back to graph
        count_edge_key = f"{do_method}_cont_by_count"
        length_edge_key = f"{do_method}_cont_by_length"
        # iter edges
        a_nd_key: graphs.NodeKey
        b_nd_key: graphs.NodeKey
        edge_idx: int
        edge_data: dict[str, Any]
        for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
            # raise if the key doesn't exist
            if do_method not in edge_data:
                raise ValueError(
                    f"Could not find {do_method} in edge data for edge: {a_nd_key} - {b_nd_key} idx: {edge_idx}"
                )
            # get target values, e.g. street names
            match_targets: list[str] = edge_data[do_method]
            match_targets = [mt.lower() for mt in match_targets if mt is not None]
            # can be multiple values, e.g. route numbers per edge, so explore individually
            for match_target in match_targets:
                if match_target is None:
                    continue
                # if using street names or route refs, then use the recursive / continuous method
                if do_method in ["names", "routes"]:
                    # no need to redo already explored values
                    # NOTE: this assumes street names or routes are continuous
                    if match_target not in method_report.entries:
                        method_report.scaffold_entry(entry_name=match_target)
                        visited_edges: list[str] = []
                        _recurse_edges(
                            nx_multi_copy,
                            do_method,
                            match_target,
                            a_nd_key,
                            b_nd_key,
                            edge_idx,
                            visited_edges,
                            method_report,
                        )
                # don't use recursion for highway types because these types are not continuous
                else:
                    if match_target not in method_report.entries:
                        method_report.scaffold_entry(entry_name=match_target)
                    method_report.entries[match_target].add_edge(
                        edge_data["geom"].length, start_nd_key=a_nd_key, end_nd_key=b_nd_key, edge_idx=edge_idx
                    )

        # write data from _StreetContinuityReport to graph
        logger.info("Writing observations to graph.")
        for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(nx_multi_copy.edges(keys=True, data=True)):  # type: ignore
            match_targets: list[str] = edge_data[do_method]
            match_targets = [mt.lower() for mt in match_targets if mt is not None]
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
                agg_count = method_report.entries[match_target].count
                if (
                    count_edge_key not in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]
                    or agg_count > nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key]
                ):
                    nx_multi_copy[a_nd_key][b_nd_key][edge_idx][count_edge_key] = agg_count
                # update graph length key
                agg_length = method_report.entries[match_target].length
                if (
                    length_edge_key not in nx_multi_copy[a_nd_key][b_nd_key][edge_idx]
                    or agg_length > nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key]
                ):
                    nx_multi_copy[a_nd_key][b_nd_key][edge_idx][length_edge_key] = agg_length

    # return if not hybrid
    if not method == "hybrid":
        return nx_multi_copy, method_reports.pop()

    # otherwise, compute hybrid using routes welded to overlapping street names
    logger.info("Post-processing routes and overlapping street names per hybrid method")
    hybrid_report = _StreetContinuityReport()
    route_report = method_reports[0]
    names_report = method_reports[1]
    # first iter routes and look for related street names
    for route_name, route_report_stats in route_report.entries.items():
        # add to hybrid report
        hybrid_report.scaffold_entry(route_name)
        # copy across
        hybrid_report.entries[route_name].count = route_report_stats.count
        hybrid_report.entries[route_name].length = route_report_stats.length
        hybrid_report.entries[route_name].edges = route_report_stats.edges
        for rt_start_nd_key, rt_end_nd_key, rt_edge_idx in route_report_stats.edges:
            # get street names
            rt_edge_data: graphs.EdgeData = nx_multi_copy[rt_start_nd_key][rt_end_nd_key][rt_edge_idx]
            edge_names = rt_edge_data["names"]
            edge_names = [name.lower() for name in edge_names if name is not None]
            for target_name in edge_names:
                # fetch the street name entry from the names continuity report
                target_name_report_stats = names_report.entries[target_name]
                # iterate the entries
                for nm_start_nd_key, nm_end_nd_key, nm_edge_idx in target_name_report_stats.edges:
                    # if the entry isn't already in the hybrid continuity report's edge list (from routes) then add
                    if (nm_start_nd_key, nm_end_nd_key, nm_edge_idx) not in hybrid_report.entries[route_name].edges:
                        nm_edge_data: graphs.EdgeData = nx_multi_copy[nm_start_nd_key][nm_end_nd_key][nm_edge_idx]
                        hybrid_report.entries[route_name].add_edge(
                            nm_edge_data["geom"].length, nm_start_nd_key, nm_end_nd_key, nm_edge_idx
                        )
    # copy to graph
    count_edge_key = f"hybrid_cont_by_count"
    length_edge_key = f"hybrid_cont_by_length"
    for hybrid_report_stats in hybrid_report.entries.values():
        for hb_start_nd_key, hb_end_nd_key, hb_edge_idx in hybrid_report_stats.edges:
            nx_multi_copy[hb_start_nd_key][hb_end_nd_key][hb_edge_idx][count_edge_key] = hybrid_report_stats.count
            nx_multi_copy[hb_start_nd_key][hb_end_nd_key][hb_edge_idx][length_edge_key] = hybrid_report_stats.length
    # write zeros
    for start_nd_key, end_nd_key, edge_idx, edge_data in nx_multi_copy.edges(keys=True, data=True):  # type: ignore
        if count_edge_key not in edge_data or length_edge_key not in edge_data:
            nx_multi_copy[start_nd_key][end_nd_key][edge_idx][count_edge_key] = 0  # type: ignore
            nx_multi_copy[start_nd_key][end_nd_key][edge_idx][length_edge_key] = 0  # type: ignore

    return nx_multi_copy, hybrid_report
