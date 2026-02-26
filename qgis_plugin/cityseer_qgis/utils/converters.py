"""
Utilities for converting QGIS line layers into a cityseer dual NetworkStructure.

Each road segment becomes a node positioned at its midpoint. Adjacent segments
that share an endpoint are connected by edges whose geometry is the merge of
the two half-segments meeting at that endpoint.

Supports incremental updates: if the layer changes between runs, only the
affected nodes and edges are removed/added rather than rebuilding from scratch.
The underlying Rust NetworkStructure uses petgraph's StableGraph, which
preserves node/edge indices on removal (no swap-and-compact), so existing
indices for unchanged features remain valid across incremental updates.

No NetworkX or geopandas required — only shapely (a cityseer dependency) and
cityseer.rustalgos.
"""

from __future__ import annotations

import collections
import itertools

# Module-level state for incremental updates.
# Keys: wkts, ns, fid_list, geoms, midpoints, node_idx, endpoint_to_fids,
#        edge_counter, seen
_inc_state: dict | None = None


def parse_distances(s: str) -> list[int]:
    """Parse a comma-separated string of distances into a list of ints."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("No distances provided.")
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid distance value: {p!r}. Expected integers.")
    return result


def build_dual_network(layer, feedback=None, step_start=1, n_steps=3, boundary=None):
    """
    Build a cityseer dual NetworkStructure directly from a QGIS line layer.

    Each line feature becomes one node at its midpoint. Pairs of features that
    share an endpoint are connected by bidirectional edges whose geometry is
    the merge of the two half-segments meeting at that shared endpoint.

    On the first call a full build is performed. Subsequent calls diff the
    layer against the previous state and apply an incremental update (remove
    changed/deleted features, add new/modified features).

    Parameters
    ----------
    layer : QgsVectorLayer
        A line layer in a projected (metre-based) CRS.
    feedback : QgsProcessingFeedback or None
        Optional feedback object for progress reporting.
    step_start : int
        Step number for the first phase (nodes). Edges phase is step_start + 1.
    n_steps : int
        Total number of steps in the overall workflow.
    boundary : shapely.geometry.BaseGeometry or None
        Optional boundary polygon. Nodes whose midpoints fall inside the
        boundary are marked ``live=True`` (used as centrality sources); nodes
        outside are ``live=False`` (provide network context only). When None,
        all nodes are live.

    Returns
    -------
    tuple[NetworkStructure, list[int], dict[int, tuple], dict[int, LineString]]
        ns        — cityseer NetworkStructure ready for centrality calls
        fid_list  — ordered list of feature IDs (matches node_key order in ns)
        midpoints — dict mapping fid -> (x, y) midpoint coordinates
        geoms     — dict mapping fid -> shapely LineString (original geometry)
    """
    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge, substring

    from cityseer import rustalgos

    global _inc_state

    def _is_live(mid):
        """Return True if the midpoint falls inside the boundary (or no boundary)."""
        if boundary is None:
            return True
        return boundary.contains(Point(mid.x, mid.y))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_line(wkt):
        """Parse WKT into a 2D shapely LineString, or None if degenerate.

        Handles MultiLineString by merging parts (linemerge) or, if that
        yields a MultiLineString, taking the longest part.
        """
        geom = shapely_wkt.loads(wkt)
        if geom.is_empty:
            return None
        if geom.geom_type == "MultiLineString":
            merged = linemerge(geom)
            if merged.geom_type == "MultiLineString":
                # Take the longest component
                geom = max(merged.geoms, key=lambda g: g.length)
            else:
                geom = merged
        if geom.geom_type != "LineString" or len(geom.coords) < 2:
            return None
        if geom.has_z:
            geom = LineString([(c[0], c[1]) for c in geom.coords])
        return geom

    def _ep_key(pt):
        """Rounded endpoint key for adjacency lookup."""
        return (round(pt[0], 1), round(pt[1], 1))

    def _half_toward(line, endpoint):
        """Return the half of *line* running from its midpoint toward *endpoint*."""
        ep = (round(endpoint[0], 1), round(endpoint[1], 1))
        end = (round(line.coords[-1][0], 1), round(line.coords[-1][1], 1))
        if ep == end:
            return substring(line, 0.5, 1.0, normalized=True)
        reversed_line = line.reverse()
        return substring(reversed_line, 0.5, 1.0, normalized=True)

    def _make_edge_wkt(line_a, line_b, endpoint_key):
        """Create merged WKT for the dual edge between two adjacent segments."""
        ha = _half_toward(line_a, endpoint_key)
        hb = _half_toward(line_b, endpoint_key)
        merged = linemerge([ha, hb])
        if merged.geom_type != "LineString":
            merged = LineString(list(ha.coords) + list(hb.coords))
        return merged.wkt

    # ------------------------------------------------------------------
    # Extract current WKTs from QGIS layer
    # ------------------------------------------------------------------
    current_wkts: dict[int, str] = {}
    for feat in layer.getFeatures():
        qgeom = feat.geometry()
        if qgeom is None or qgeom.isEmpty():
            continue
        current_wkts[feat.id()] = qgeom.asWkt()

    # ------------------------------------------------------------------
    # Incremental path — diff against previous state
    # ------------------------------------------------------------------
    # Track boundary identity so we can update live flags when it changes.
    boundary_wkt = boundary.wkt if boundary is not None else None

    if _inc_state is not None:
        prev_boundary_wkt = _inc_state.get("boundary_wkt")
        boundary_changed = boundary_wkt != prev_boundary_wkt

        prev_wkts = _inc_state["wkts"]
        prev_fids = set(prev_wkts.keys())
        curr_fids = set(current_wkts.keys())

        removed = prev_fids - curr_fids
        added = curr_fids - prev_fids
        modified = {
            fid
            for fid in prev_fids & curr_fids
            if prev_wkts[fid] != current_wkts[fid]
        }

        to_remove = removed | modified
        to_add = added | modified

        if feedback:
            if removed:
                feedback.pushInfo(f"Removed fids: {sorted(removed)}")
            if added:
                feedback.pushInfo(f"Added fids: {sorted(added)}")
            if modified:
                feedback.pushInfo(f"Modified fids: {sorted(modified)}")

        if not to_remove and not to_add and not boundary_changed:
            if feedback:
                feedback.pushInfo("Using cached network structure (no changes).")
            s = _inc_state
            return (s["ns"], s["fid_list"], s["midpoints"], s["geoms"])

        # Unpack mutable state
        ns = _inc_state["ns"]
        fid_list = _inc_state["fid_list"]
        geoms = _inc_state["geoms"]
        midpoints = _inc_state["midpoints"]
        node_idx = _inc_state["node_idx"]
        endpoint_to_fids = _inc_state["endpoint_to_fids"]
        edge_counter = _inc_state["edge_counter"]
        seen = _inc_state["seen"]

        # ---- Remove phase ----
        # StableGraph.remove_node() cascades to all connected edges, so we
        # only need to remove nodes here — no explicit remove_street_edge()
        # calls are required.  StableGraph also preserves existing indices
        # (no swap-and-compact), so node_idx values for untouched features
        # remain valid after removals.
        if feedback:
            feedback.setProgressText(
                f"Step {step_start} of {n_steps}: "
                f"Removing {len(to_remove)} features…"
            )
            feedback.setProgress(0)

        for i, fid in enumerate(to_remove):
            ns.remove_street_node(node_idx[fid])
            line = geoms[fid]
            for pt in (line.coords[0], line.coords[-1]):
                key = _ep_key(pt)
                if key in endpoint_to_fids:
                    try:
                        endpoint_to_fids[key].remove(fid)
                    except ValueError:
                        pass
                    if not endpoint_to_fids[key]:
                        del endpoint_to_fids[key]
            del geoms[fid]
            del midpoints[fid]
            del node_idx[fid]
            if feedback:
                feedback.setProgress(int(100 * (i + 1) / len(to_remove)))

        # Batch-clean fid_list and seen set
        fid_list = [f for f in fid_list if f not in to_remove]
        seen = {p for p in seen if not (p & to_remove)}

        # ---- Add phase: create nodes first ----
        if feedback:
            feedback.setProgress(100)
            feedback.setProgressText(
                f"Step {step_start + 1} of {n_steps}: "
                f"Adding {len(to_add)} features…"
            )
            feedback.setProgress(0)

        for fid in to_add:
            line = _parse_line(current_wkts[fid])
            if line is None:
                continue
            geoms[fid] = line
            fid_list.append(fid)
            mid = line.interpolate(0.5, normalized=True)
            idx = ns.add_street_node(
                node_key=fid, x=mid.x, y=mid.y, live=_is_live(mid), weight=1.0,
            )
            node_idx[fid] = idx
            midpoints[fid] = (mid.x, mid.y)
            for pt in (line.coords[0], line.coords[-1]):
                endpoint_to_fids[_ep_key(pt)].append(fid)

        # Keep fid_list in stable sorted order after remove+append.
        fid_list.sort()

        # ---- Add phase: create edges for new fids ----
        added_with_geom = [fid for fid in to_add if fid in geoms]
        for i, fid in enumerate(added_with_geom):
            line = geoms[fid]
            for pt in (line.coords[0], line.coords[-1]):
                key = _ep_key(pt)
                for other_fid in endpoint_to_fids.get(key, []):
                    if other_fid == fid:
                        continue
                    pair = frozenset({fid, other_fid})
                    if pair in seen:
                        continue
                    seen.add(pair)
                    merged_wkt = _make_edge_wkt(geoms[fid], geoms[other_fid], key)
                    ns.add_street_edge(
                        node_idx[fid], node_idx[other_fid],
                        edge_counter, fid, other_fid, merged_wkt,
                    )
                    edge_counter += 1
                    ns.add_street_edge(
                        node_idx[other_fid], node_idx[fid],
                        edge_counter, other_fid, fid, merged_wkt,
                    )
                    edge_counter += 1
            if feedback and added_with_geom:
                feedback.setProgress(int(100 * (i + 1) / len(added_with_geom)))

        # If the boundary polygon changed, update live status on all nodes.
        if boundary_changed:
            for fid in fid_list:
                if fid in node_idx:
                    mx, my = midpoints[fid]
                    live = boundary is None or boundary.contains(Point(mx, my))
                    ns.set_node_live(node_idx[fid], live)
            if feedback:
                feedback.pushInfo("Boundary changed — updated live status for all nodes.")

        ns.validate()
        ns.build_edge_rtree()

        # Persist updated state
        _inc_state["wkts"] = current_wkts
        _inc_state["fid_list"] = fid_list
        _inc_state["edge_counter"] = edge_counter
        _inc_state["seen"] = seen
        _inc_state["boundary_wkt"] = boundary_wkt

        if feedback:
            feedback.pushInfo(
                f"Incremental update: {len(to_remove)} removed, "
                f"{len(to_add)} added."
            )
            feedback.setProgress(100)

        return (ns, fid_list, midpoints, geoms)

    # ------------------------------------------------------------------
    # Full build path (first run)
    # ------------------------------------------------------------------
    geoms: dict[int, object] = {}
    fid_list: list[int] = []

    for fid, wkt in current_wkts.items():
        line = _parse_line(wkt)
        if line is None:
            continue
        geoms[fid] = line
        fid_list.append(fid)

    n_segments = len(fid_list)

    # ---- Build dual nodes (0–100%) ----
    if feedback:
        feedback.setProgressText(
            f"Step {step_start} of {n_steps}: Building dual nodes…"
        )
        feedback.setProgress(0)

    endpoint_to_fids: dict[tuple, list[int]] = collections.defaultdict(list)
    ns = rustalgos.graph.NetworkStructure()
    node_idx: dict[int, int] = {}
    midpoints: dict[int, tuple] = {}

    for i, (fid, line) in enumerate(geoms.items()):
        for pt in (line.coords[0], line.coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)
        mid = line.interpolate(0.5, normalized=True)
        idx = ns.add_street_node(
            node_key=fid, x=mid.x, y=mid.y, live=_is_live(mid), weight=1.0,
        )
        node_idx[fid] = idx
        midpoints[fid] = (mid.x, mid.y)
        if feedback and i % 200 == 0:
            feedback.setProgress(int(100 * i / n_segments))

    n_live = sum(1 for fid in fid_list if ns.is_node_live(node_idx[fid]))
    if feedback:
        feedback.pushInfo(f"Added {n_segments} dual nodes ({n_live} live).")
        feedback.setProgress(100)

    # ---- Build dual edges (0–100%) ----
    if feedback:
        feedback.setProgressText(
            f"Step {step_start + 1} of {n_steps}: Building dual edges…"
        )
        feedback.setProgress(0)

    edge_counter = 0
    seen: set = set()
    n_endpoints = len(endpoint_to_fids)

    for j, (endpoint, fids) in enumerate(endpoint_to_fids.items()):
        for fid_a, fid_b in itertools.combinations(fids, 2):
            pair = frozenset({fid_a, fid_b})
            if pair in seen:
                continue
            seen.add(pair)
            merged_wkt = _make_edge_wkt(geoms[fid_a], geoms[fid_b], endpoint)
            ns.add_street_edge(
                node_idx[fid_a], node_idx[fid_b],
                edge_counter, fid_a, fid_b, merged_wkt,
            )
            edge_counter += 1
            ns.add_street_edge(
                node_idx[fid_b], node_idx[fid_a],
                edge_counter, fid_b, fid_a, merged_wkt,
            )
            edge_counter += 1
        if feedback and j % 200 == 0:
            feedback.setProgress(int(100 * j / n_endpoints))

    ns.validate()
    ns.build_edge_rtree()

    if feedback:
        feedback.pushInfo(f"Added {edge_counter} dual edges.")
        feedback.setProgress(100)

    # Persist state for incremental updates
    _inc_state = {
        "wkts": current_wkts,
        "ns": ns,
        "fid_list": fid_list,
        "geoms": geoms,
        "midpoints": midpoints,
        "node_idx": node_idx,
        "endpoint_to_fids": endpoint_to_fids,
        "edge_counter": edge_counter,
        "seen": seen,
        "boundary_wkt": boundary_wkt,
    }

    return (ns, fid_list, midpoints, geoms)
