"""
Observe module for computing observations derived from `networkX` graphs.

These methods are generally sufficiently simple that further computational optimisation is not required. Network
centrality methods (which do require further computational optimisation due to their complexity) are handled separately
in the [`networks`](/metrics/networks) module.

"""

# workaround until networkx adopts types
# pyright: basic

from __future__ import annotations

import copy
import logging
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm

from cityseer import config
from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuityEntry:
    """
    State management for an individual street continuity entry.

    This corresponds to an individual street name, route name or number, or highway type.
    """

    entry_name: str
    count: int
    length: float
    edges: dict[str, tuple[graphs.NodeKey, graphs.NodeKey, int]]

    def __init__(self, entry_name: str) -> None:
        """Instances a continuity entry."""
        self.entry_name = entry_name
        self.count = 0
        self.length = 0
        self.edges = {}

    @staticmethod
    def generate_key(start_nd_key: graphs.NodeKey, end_nd_key: graphs.NodeKey, edge_idx: int):
        """Generate a unique key given uncertainty of start and end node order."""
        sorted_keys = sorted([str(start_nd_key), str(end_nd_key)])
        return f"{sorted_keys[0]}|{sorted_keys[1]}|{edge_idx}"

    def add_edge(self, length: float, start_nd_key: graphs.NodeKey, end_nd_key: graphs.NodeKey, edge_idx: int) -> None:
        """Adds edge details to a continuity entry."""
        key = self.generate_key(start_nd_key, end_nd_key, edge_idx)
        if key not in self.edges:
            self.count += 1
            self.length += length
            self.edges[key] = (start_nd_key, end_nd_key, edge_idx)


class StreetContinuityReport:
    """
    State management for a collection of street continuity metrics.

    Each key in the `entries` attribute corresponds to a `ContinuityEntry`.
    """

    method: str
    entries: dict[str, ContinuityEntry]

    def __init__(self, method: str) -> None:
        """Instance a street continuity report."""
        self.method = method
        self.entries = {}

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
        for n, arg_idx in enumerate(np.argsort(counts)[::-1]):
            if n == n_items:
                break
            logger.info(f"Count: {counts[arg_idx]} - {item_keys[arg_idx]}")

    def report_by_length(self, n_items: int = 10) -> None:
        """Print a report sorted by entry lengths."""
        logger.info(f"Reporting top {n_items} continuity observations by street lengths.")
        item_keys = list(self.entries.keys())
        lengths: list[float] = [v.length for v in self.entries.values()]
        for n, arg_idx in enumerate(np.argsort(lengths)[::-1]):
            if n == n_items:
                break
            len_idx: float = lengths[arg_idx]
            logger.info(f"Length: {round(len_idx / 1000, 2)}km - {item_keys[arg_idx]}")


def _continuity_report_to_nx(
    edge_key: str, nx_multigraph: nx.MultiGraph, continuity_report: StreetContinuityReport
) -> nx.MultiGraph:
    """Copies data from a continuity report to a graph."""
    # prepare edge keys
    label_edge_key = f"{edge_key}_cont_by_label"
    count_edge_key = f"{edge_key}_cont_by_count"
    length_edge_key = f"{edge_key}_cont_by_length"
    # iterate from smallest to largest so that larger routes overwrite smaller if overlapping
    route_keys = list(continuity_report.entries.keys())
    route_counts: list[float] = [v.count for v in continuity_report.entries.values()]
    for arg_idx in np.argsort(route_counts):
        route_key: str = route_keys[arg_idx]
        continuity_entry: ContinuityEntry = continuity_report.entries[route_key]
        for hb_start_nd_key, hb_end_nd_key, hb_edge_idx in continuity_entry.edges.values():
            nx_multigraph[hb_start_nd_key][hb_end_nd_key][hb_edge_idx][label_edge_key] = route_key
            nx_multigraph[hb_start_nd_key][hb_end_nd_key][hb_edge_idx][count_edge_key] = continuity_entry.count
            nx_multigraph[hb_start_nd_key][hb_end_nd_key][hb_edge_idx][length_edge_key] = continuity_entry.length
    # write zeros to empties
    edge_data: graphs.EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in nx_multigraph.edges(keys=True, data=True):  # type: ignore
        if count_edge_key not in edge_data or length_edge_key not in edge_data:
            nx_multigraph[start_nd_key][end_nd_key][edge_idx][label_edge_key] = None
            nx_multigraph[start_nd_key][end_nd_key][edge_idx][count_edge_key] = 0
            nx_multigraph[start_nd_key][end_nd_key][edge_idx][length_edge_key] = 0

    return nx_multigraph


def _recurse_edges(
    _nx_multigraph: nx.MultiGraph,
    _method: str,
    _match_target: str,
    _a_nd_key: graphs.NodeKey,
    _b_nd_key: graphs.NodeKey,
    _edge_idx: int,
    _visited_edges: set[str],
    _continuity_report: StreetContinuityReport,
    _report_key: str,
):
    """Used when using strictly continuous implementations, e.g. for street names or routes."""
    # generate an edge key
    edge_key = ContinuityEntry.generate_key(_a_nd_key, _b_nd_key, _edge_idx)
    # skip if already visited
    if edge_key in _visited_edges:
        return
    _visited_edges.add(edge_key)
    # extract edge data for the current target key
    nested_edge_data: graphs.EdgeData = _nx_multigraph[_a_nd_key][_b_nd_key][_edge_idx]
    if _method not in nested_edge_data:
        raise ValueError(f"Missing target key of {_method}")
    nested_match_targets: list[str | None] = nested_edge_data[_method]
    nested_match_targets = [nmt.lower() for nmt in nested_match_targets if nmt is not None]
    # bail if this edge info (e.g. street name) doesn't intersect with the previous
    if not _match_target.lower() in nested_match_targets:
        return
    _continuity_report.entries[_report_key].add_edge(
        length=nested_edge_data["geom"].length, start_nd_key=_a_nd_key, end_nd_key=_b_nd_key, edge_idx=_edge_idx
    )
    # find all neighbouring edge pairs
    a_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
        (_a_nd_key, ann) for ann in nx.neighbors(_nx_multigraph, _a_nd_key) if ann != _b_nd_key
    ]
    b_nb_pairs: list[tuple[graphs.NodeKey, graphs.NodeKey]] = [
        (_b_nd_key, bnn) for bnn in nx.neighbors(_nx_multigraph, _b_nd_key) if bnn != _a_nd_key
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
                _continuity_report,
                _report_key,
            )


def street_continuity(
    nx_multigraph: nx.MultiGraph, method: str | tuple[str, str]
) -> tuple[nx.MultiGraph, StreetContinuityReport]:
    """
    Compute the street continuity for a given graph.

    This requires a graph with `names`, `routes`, or `highways` edge keys corresponding to the selected `method`
    parameter. These keys are available if importing an OSM network with
    [`osm_graph_from_poly`](/tools/io#osm-graph-from-poly) or if importing OS Open Roads data with
    [nx_from_open_roads](/tools/io#nx-from-open-roads).

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms. Edges should contain "names", "routes", or "highways" keys
        corresponding to the specified `method` parameter.
    method: str
        The type of continuity metric to compute, where available options are "names", "routes", or "highways".

    Returns
    -------
    MultiGraph
        A copy of the input `networkX` `MultiGraph` with new edge keys corresponding to the calculated route continuity
        metric. The metrics will be stored in 'length' and 'count' forms for the specified method, with keys formatted
        according to `f"{method}_cont_by_{form}"`. For example, when computing "names" continuity, the
        `names_cont_by_count` and `names_cont_by_length` keys will be added to the returned `networkX` `MultiGraph`.
    StreetContinuityReport
        An instance of [`StreetContinuityReport`](/metrics/observe#streetcontinuityreport) containing the computed
        state for the selected method.

    """
    # NOTE: experimented with string cleaning and removal of generic descriptors but this worked contrary to intentions.

    nx_multi_copy: nx.MultiGraph = nx_multigraph.copy()
    # check intended method keys
    available_targets = ["names", "routes", "highways"]
    if method not in available_targets:
        raise ValueError(f"Method of {method} is not recognised.")

    method_report = StreetContinuityReport(method=method)

    logger.info(f"Calculating metrics for {method}.")
    # iter edges
    a_nd_key: graphs.NodeKey
    b_nd_key: graphs.NodeKey
    edge_idx: int
    edge_data: dict[str, Any]
    for a_nd_key, b_nd_key, edge_idx, edge_data in tqdm(  # type: ignore
        nx_multi_copy.edges(keys=True, data=True), disable=config.QUIET_MODE  # type: ignore
    ):
        # raise if the key doesn't exist
        if method not in edge_data:
            raise ValueError(f"Could not find {method} in edge data for edge: {a_nd_key} - {b_nd_key} idx: {edge_idx}")
        # get target values, e.g. street names
        match_targets: list[str | None] = edge_data[method]
        match_targets = [mt.lower() for mt in match_targets if mt is not None]
        # can be multiple values, e.g. route numbers per edge, so explore individually
        for match_target in match_targets:
            if match_target is None:
                continue
            # if using street names or route refs, then use the recursive / continuous method
            if method in ["names", "routes"]:
                # no need to redo already explored values
                # NOTE: this assumes street names or routes are continuous
                if match_target not in method_report.entries:
                    method_report.scaffold_entry(entry_name=match_target)
                    visited_edges: set[str] = set()
                    _recurse_edges(
                        nx_multi_copy,
                        method,
                        match_target,
                        a_nd_key,
                        b_nd_key,
                        edge_idx,
                        visited_edges,
                        method_report,
                        match_target,  # report key is same as match target in this instance
                    )
            # don't use recursion for highway types because these types are not continuous
            else:
                if match_target not in method_report.entries:
                    method_report.scaffold_entry(entry_name=match_target)
                method_report.entries[match_target].add_edge(
                    edge_data["geom"].length, start_nd_key=a_nd_key, end_nd_key=b_nd_key, edge_idx=edge_idx
                )
    # copy to networkx input graph
    nx_multi_copy = _continuity_report_to_nx(
        edge_key=method, nx_multigraph=nx_multi_copy, continuity_report=method_report
    )

    return nx_multi_copy, method_report


def hybrid_street_continuity(
    nx_multigraph: nx.MultiGraph,
) -> tuple[nx.MultiGraph, StreetContinuityReport]:
    """
    Compute the street continuity for a given graph using a hybridisation of routes and names continuity.

    Hybrid continuity merges route continuity and street continuity information where a route overlaps a street
    continuity.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms. Edges should contain "names", "routes", or "highways" keys
        corresponding to the specified `method` parameter.

    Returns
    -------
    MultiGraph
        A copy of the input `networkX` `MultiGraph` with new edge keys corresponding to the calculated route continuity
        metric. The metrics will be stored in 'hybrid_cont_by_length' and 'hybrid_cont_by_count' keys.
    StreetContinuityReport
        An instance of [`StreetContinuityReport`](/metrics/observe#streetcontinuityreport) containing the computed
        state for the "hybrid" method.

    """
    nx_multi_copy, routes_continuity_report = street_continuity(nx_multigraph, method="routes")
    hybrid_report = StreetContinuityReport(method="hybrid")
    # copy route entries across to hybrid as starting point
    for route_key, continuity_entry in routes_continuity_report.entries.items():
        hybrid_report.scaffold_entry(route_key)
        # copy across
        hybrid_report.entries[route_key].count = continuity_entry.count
        hybrid_report.entries[route_key].length = continuity_entry.length
        # make sure to copy list otherwise changes leak from hybrid to routes continuity reports!!
        hybrid_report.entries[route_key].edges = copy.deepcopy(continuity_entry.edges)
    # process in order - useful for debugging
    route_keys = list(routes_continuity_report.entries.keys())
    route_lengths: list[float] = [v.length for v in routes_continuity_report.entries.values()]
    for arg_idx in np.argsort(route_lengths)[::-1]:
        route_key: str = route_keys[arg_idx]
        routes_continuity_entry: ContinuityEntry = routes_continuity_report.entries[route_key]
        # iterate the edges associated with an entry
        for rt_start_nd_key, rt_end_nd_key, rt_edge_idx in routes_continuity_entry.edges.values():
            # get associated street names
            rt_edge_data: graphs.EdgeData = nx_multi_copy[rt_start_nd_key][rt_end_nd_key][rt_edge_idx]
            rt_edge_names = rt_edge_data["names"]
            rt_edge_names = [name.lower() for name in rt_edge_names if name is not None]
            # find corresponding street name related continuity entries from names continuity report
            for target_name in rt_edge_names:
                # recursively add adjoining streets of the same name
                visited_edges: set[str] = set()
                _recurse_edges(
                    nx_multi_copy,
                    "names",  # match target - i.e. edge names
                    target_name,
                    rt_start_nd_key,
                    rt_end_nd_key,
                    rt_edge_idx,
                    visited_edges,
                    hybrid_report,
                    route_key,  # report key must write back to route label
                )
    # write to report
    nx_multi_copy = _continuity_report_to_nx(
        edge_key="hybrid", nx_multigraph=nx_multi_copy, continuity_report=hybrid_report
    )

    return nx_multi_copy, hybrid_report
