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
import contextlib
import itertools
import math

# Module-level state for incremental updates.
# Keys: wkts, ns, fid_list, geoms, midpoints, node_idx, endpoint_to_fids,
#        edge_counter, seen
_inc_state: dict | None = None


# ---------------------------------------------------------------------------
# Pure-Python coordinate arithmetic (replaces shapely substring/interpolate)
# ---------------------------------------------------------------------------
def _cumulative_lengths(coords: list[tuple[float, float]]) -> list[float]:
    """Return cumulative distances along a coordinate list.

    Result[i] = distance from coords[0] to coords[i].
    Result[0] = 0.0, result[-1] = total line length.
    """
    cum = [0.0]
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    return cum


def _interpolate_at(
    coords: list[tuple[float, float]],
    cum: list[float],
    frac: float,
) -> tuple[float, float]:
    """Return (x, y) at a normalised fraction [0..1] of total line length."""
    target = frac * cum[-1]
    # Find the segment containing the target distance
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg_len = cum[i] - cum[i - 1]
            if seg_len == 0.0:
                return coords[i]
            t = (target - cum[i - 1]) / seg_len
            x = coords[i - 1][0] + t * (coords[i][0] - coords[i - 1][0])
            y = coords[i - 1][1] + t * (coords[i][1] - coords[i - 1][1])
            return (x, y)
    return coords[-1]


def _substring_coords(
    coords: list[tuple[float, float]],
    cum: list[float],
    start_frac: float,
    end_frac: float,
) -> list[tuple[float, float]]:
    """Return coordinate list for the normalised substring [start_frac, end_frac].

    Interpolates new vertices at the start and end fractions if they fall
    between existing vertices. Includes all original vertices in between.
    """
    total = cum[-1]
    if total == 0.0:
        return list(coords)
    d_start = start_frac * total
    d_end = end_frac * total
    result: list[tuple[float, float]] = []
    # Find and interpolate the start point
    started = False
    for i in range(1, len(cum)):
        if not started:
            if cum[i] >= d_start:
                seg_len = cum[i] - cum[i - 1]
                if seg_len == 0.0:
                    result.append(coords[i])
                else:
                    t = (d_start - cum[i - 1]) / seg_len
                    x = coords[i - 1][0] + t * (coords[i][0] - coords[i - 1][0])
                    y = coords[i - 1][1] + t * (coords[i][1] - coords[i - 1][1])
                    result.append((x, y))
                started = True
                # If end is also in this segment, interpolate and return
                if cum[i] >= d_end:
                    if seg_len == 0.0:
                        if len(result) == 0 or result[-1] != coords[i]:
                            result.append(coords[i])
                    else:
                        t2 = (d_end - cum[i - 1]) / seg_len
                        x2 = coords[i - 1][0] + t2 * (coords[i][0] - coords[i - 1][0])
                        y2 = coords[i - 1][1] + t2 * (coords[i][1] - coords[i - 1][1])
                        end_pt = (x2, y2)
                        if end_pt != result[-1]:
                            result.append(end_pt)
                    return result
                # Add the vertex at cum[i] if distinct from start point
                if coords[i] != result[-1]:
                    result.append(coords[i])
        else:
            # We've started — add vertices until we reach or pass d_end
            if cum[i] >= d_end:
                seg_len = cum[i] - cum[i - 1]
                if seg_len == 0.0:
                    if coords[i] != result[-1]:
                        result.append(coords[i])
                else:
                    t = (d_end - cum[i - 1]) / seg_len
                    x = coords[i - 1][0] + t * (coords[i][0] - coords[i - 1][0])
                    y = coords[i - 1][1] + t * (coords[i][1] - coords[i - 1][1])
                    end_pt = (x, y)
                    if end_pt != result[-1]:
                        result.append(end_pt)
                return result
            if coords[i] != result[-1]:
                result.append(coords[i])
    return result


def _coords_to_wkt(coords: list[tuple[float, float]]) -> str:
    """Format a coordinate list as a WKT LINESTRING."""
    pairs = ", ".join(f"{x} {y}" for x, y in coords)
    return f"LINESTRING ({pairs})"


def parse_distances(s: str) -> list[int]:
    """Parse a comma-separated string of distances into a list of ints."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("No distances provided.")
    result = []
    seen: set[int] = set()
    for p in parts:
        try:
            d = int(p)
        except ValueError as err:
            raise ValueError(f"Invalid distance value: {p!r}. Expected integers.") from err
        if d <= 0:
            raise ValueError(f"Invalid distance value: {p!r}. Distances must be positive integers.")
        if d in seen:
            raise ValueError(f"Duplicate distance value: {d}. Distances must be unique.")
        seen.add(d)
        result.append(d)
    return result


def build_dual_network(
    layer,
    feedback=None,
    step=1,
    n_steps=3,
    progress_base=0,
    progress_span=100,
    boundary=None,
):
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
    step : int
        Step number displayed in the progress text.
    n_steps : int
        Total number of steps in the overall workflow.
    progress_base : float
        Start of this step's slice of the overall 0–100% progress bar.
    progress_span : float
        Width of this step's slice (e.g. 25 means base..base+25%).
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
    from cityseer import rustalgos
    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge
    from shapely.prepared import prep
    from shapely.validation import make_valid

    global _inc_state
    layer_cache_key = (
        layer.id(),
        layer.crs().authid(),
        layer.wkbType(),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_line(wkt):
        """Parse WKT into a 2D shapely LineString, or None if degenerate.

        Handles MultiLineString by merging parts (linemerge) or, if that
        yields a MultiLineString, taking the longest part. Attempts to fix
        invalid geometries via make_valid and skips zero-length geometries.
        """
        geom = shapely_wkt.loads(wkt)
        if geom.is_empty:
            return None
        if not geom.is_valid:
            geom = make_valid(geom)
            if geom.is_empty:
                return None
        if geom.geom_type == "MultiLineString":
            merged = linemerge(geom)
            geom = max(merged.geoms, key=lambda g: g.length) if merged.geom_type == "MultiLineString" else merged
        elif geom.geom_type == "GeometryCollection":
            lines = [g for g in geom.geoms if g.geom_type == "LineString" and g.length > 0]
            if not lines:
                return None
            geom = max(lines, key=lambda g: g.length)
        if geom.geom_type != "LineString" or len(geom.coords) < 2:
            return None
        if geom.has_z:
            geom = LineString([(c[0], c[1]) for c in geom.coords])
        if geom.length < 1e-3:
            return None
        return geom

    def _ep_key(pt):
        """Rounded endpoint key for adjacency lookup."""
        return (round(pt[0], 1), round(pt[1], 1))

    def _half_toward_coords(fid, endpoint):
        """Return coords for the half of line *fid* from midpoint toward *endpoint*."""
        coords, cum = _line_data[fid]
        ep = (round(endpoint[0], 1), round(endpoint[1], 1))
        end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        if ep == end:
            return _substring_coords(coords, cum, 0.5, 1.0)
        rev = coords[::-1]
        rev_cum = _cumulative_lengths(rev)
        return _substring_coords(rev, rev_cum, 0.5, 1.0)

    def _make_edge_wkt(fid_a, fid_b, endpoint_key):
        """Create directed merged WKT from fid_a midpoint to fid_b midpoint via endpoint."""
        ha = _half_toward_coords(fid_a, endpoint_key)
        hb = _half_toward_coords(fid_b, endpoint_key)
        hb_rev = hb[::-1]
        merged = ha + hb_rev[1:]
        return _coords_to_wkt(merged)

    # Sub-phase boundaries within this step's progress_span:
    # reading features: 0–10%, building nodes: 10–50%, building edges: 50–100%
    read_base = progress_base
    read_span = progress_span * 0.1
    node_base = progress_base + read_span
    node_span = progress_span * 0.4
    edge_base = node_base + node_span
    edge_span = progress_span * 0.5

    # ------------------------------------------------------------------
    # Extract current WKTs from QGIS layer
    # ------------------------------------------------------------------
    if feedback:
        feedback.setProgressText(f"Step {step} of {n_steps}: Reading input features…")
        feedback.setProgress(int(read_base))
    current_wkts: dict[int, str] = {}
    feat_count = layer.featureCount() if hasattr(layer, "featureCount") else -1
    for i, feat in enumerate(layer.getFeatures()):
        qgeom = feat.geometry()
        if qgeom is None or qgeom.isEmpty():
            if feedback and feat_count and feat_count > 0:
                pct = (i + 1) / feat_count
                feedback.setProgress(int(read_base + pct * read_span))
            continue
        current_wkts[feat.id()] = qgeom.asWkt()
        if feedback and feat_count and feat_count > 0:
            pct = (i + 1) / feat_count
            feedback.setProgress(int(read_base + pct * read_span))
    if feedback and (feat_count <= 0):
        feedback.setProgress(int(read_base + read_span))

    if not current_wkts:
        feat_total = layer.featureCount() if hasattr(layer, "featureCount") else 0
        if feat_total > 0:
            msg = (
                f"Layer reports {feat_total} features but none could be read. "
                "The data source may have moved or become unavailable."
            )
        else:
            msg = "Input layer contains no features."
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Incremental path — diff against previous state
    # ------------------------------------------------------------------
    # Track boundary identity so we can update live flags when it changes.
    boundary_wkt = boundary.wkt if boundary is not None else None
    prepared_boundary = prep(boundary) if boundary is not None else None

    if _inc_state is not None and _inc_state.get("layer_cache_key") == layer_cache_key:
        prev_boundary_wkt = _inc_state.get("boundary_wkt")
        boundary_changed = boundary_wkt != prev_boundary_wkt

        prev_wkts = _inc_state["wkts"]
        prev_fids = set(prev_wkts.keys())
        curr_fids = set(current_wkts.keys())

        removed = prev_fids - curr_fids
        added = curr_fids - prev_fids
        modified = {fid for fid in prev_fids & curr_fids if prev_wkts[fid] != current_wkts[fid]}

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
                feedback.setProgress(int(progress_base + progress_span))
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
        _line_data = _inc_state["_line_data"]

        # ---- Remove phase ----
        # StableGraph.remove_node() cascades to all connected edges, so we
        # only need to remove nodes here — no explicit remove_street_edge()
        # calls are required.  StableGraph also preserves existing indices
        # (no swap-and-compact), so node_idx values for untouched features
        # remain valid after removals.
        if feedback:
            feedback.setProgressText(f"Step {step} of {n_steps}: Removing {len(to_remove)} features…")
            feedback.setProgress(int(node_base))

        for i, fid in enumerate(to_remove):
            ns.remove_street_node(node_idx[fid])
            coords, _cum = _line_data[fid]
            for pt in (coords[0], coords[-1]):
                key = _ep_key(pt)
                if key in endpoint_to_fids:
                    with contextlib.suppress(ValueError):
                        endpoint_to_fids[key].remove(fid)
                    if not endpoint_to_fids[key]:
                        del endpoint_to_fids[key]
            del geoms[fid]
            del _line_data[fid]
            del midpoints[fid]
            del node_idx[fid]
            if feedback:
                pct = (i + 1) / len(to_remove)
                feedback.setProgress(int(node_base + pct * node_span))

        # Batch-clean fid_list and seen set
        fid_list = [f for f in fid_list if f not in to_remove]
        seen = {p for p in seen if not (p & to_remove)}

        # ---- Add phase: create nodes first ----
        if feedback:
            feedback.setProgressText(f"Step {step} of {n_steps}: Adding {len(to_add)} features…")
            feedback.setProgress(int(edge_base))

        # Build endpoint-pair index for duplicate detection
        ep_pair_best: dict[frozenset, tuple[int, float]] = {}
        for existing_fid, existing_line in geoms.items():
            ec = list(existing_line.coords)
            pair = frozenset({_ep_key(ec[0]), _ep_key(ec[-1])})
            if pair not in ep_pair_best or existing_line.length > ep_pair_best[pair][1]:
                ep_pair_best[pair] = (existing_fid, existing_line.length)

        n_skipped = 0
        for fid in to_add:
            line = _parse_line(current_wkts[fid])
            if line is None:
                n_skipped += 1
                continue
            c = list(line.coords)
            # Skip short self-loops
            if _ep_key(c[0]) == _ep_key(c[-1]) and line.length < 10.0:
                n_skipped += 1
                continue
            # Skip duplicates (same endpoint pair, similar length to existing)
            pair = frozenset({_ep_key(c[0]), _ep_key(c[-1])})
            if pair in ep_pair_best and line.length >= ep_pair_best[pair][1] * 0.8:
                n_skipped += 1
                continue
            ep_pair_best[pair] = (fid, line.length)
            geoms[fid] = line
            fid_list.append(fid)
            coords = [(c[0], c[1]) for c in line.coords]
            cum = _cumulative_lengths(coords)
            _line_data[fid] = (coords, cum)
            mid = _interpolate_at(coords, cum, 0.5)
            live = prepared_boundary is None or prepared_boundary.contains(Point(mid[0], mid[1]))
            idx = ns.add_street_node(
                node_key=fid,
                x=mid[0],
                y=mid[1],
                live=live,
                weight=1.0,
            )
            node_idx[fid] = idx
            midpoints[fid] = mid
            for pt in (coords[0], coords[-1]):
                endpoint_to_fids[_ep_key(pt)].append(fid)

        if n_skipped > 0 and feedback:
            feedback.pushInfo(f"Skipped {n_skipped} features with invalid or zero-length geometry.")

        # Keep fid_list in stable sorted order after remove+append.
        fid_list.sort()

        # ---- Add phase: create edges for new fids ----
        added_with_geom = [fid for fid in to_add if fid in geoms]
        for i, fid in enumerate(added_with_geom):
            coords, _cum = _line_data[fid]
            for pt in (coords[0], coords[-1]):
                key = _ep_key(pt)
                for other_fid in endpoint_to_fids.get(key, []):
                    if other_fid == fid:
                        continue
                    pair = frozenset({fid, other_fid})
                    if pair in seen:
                        continue
                    seen.add(pair)
                    primal_key = str(key)
                    merged_wkt = _make_edge_wkt(fid, other_fid, key)
                    ns.add_street_edge(
                        node_idx[fid],
                        node_idx[other_fid],
                        edge_counter,
                        fid,
                        other_fid,
                        merged_wkt,
                        shared_primal_node_key=primal_key,
                    )
                    edge_counter += 1
                    merged_wkt_rev = _make_edge_wkt(other_fid, fid, key)
                    ns.add_street_edge(
                        node_idx[other_fid],
                        node_idx[fid],
                        edge_counter,
                        other_fid,
                        fid,
                        merged_wkt_rev,
                        shared_primal_node_key=primal_key,
                    )
                    edge_counter += 1
            if feedback and added_with_geom:
                pct = (i + 1) / len(added_with_geom)
                feedback.setProgress(int(edge_base + pct * edge_span))

        # If the boundary polygon changed, update live status on all nodes.
        if boundary_changed:
            for fid in fid_list:
                if fid in node_idx:
                    mx, my = midpoints[fid]
                    live = prepared_boundary is None or prepared_boundary.contains(Point(mx, my))
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
            feedback.pushInfo(f"Incremental update: {len(to_remove)} removed, {len(to_add)} added.")
            feedback.setProgress(int(progress_base + progress_span))

        return (ns, fid_list, midpoints, geoms)
    elif _inc_state is not None and feedback:
        feedback.pushInfo("Input layer changed — rebuilding cached dual network.")

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

    n_skipped = len(current_wkts) - len(fid_list)
    if n_skipped > 0 and feedback:
        feedback.pushInfo(f"Skipped {n_skipped} features with invalid or zero-length geometry.")

    # ---- Geometry cleanup ----
    n_self_loops = 0
    n_duplicates = 0
    n_danglers = 0

    # Remove short self-loops (start and end endpoints round to same key, < 10m)
    SELF_LOOP_MAX = 10.0
    for fid in list(geoms.keys()):
        c = list(geoms[fid].coords)
        if _ep_key(c[0]) == _ep_key(c[-1]) and geoms[fid].length < SELF_LOOP_MAX:
            del geoms[fid]
            n_self_loops += 1

    # Remove duplicate geometries (same endpoint pair, similar length — keep longest)
    DUPLICATE_LENGTH_RATIO = 0.8  # remove shorter if within 80% of longest
    ep_pairs: dict[frozenset, list[tuple[int, float]]] = collections.defaultdict(list)
    for fid, line in geoms.items():
        c = list(line.coords)
        pair = frozenset({_ep_key(c[0]), _ep_key(c[-1])})
        ep_pairs[pair].append((fid, line.length))
    for items in ep_pairs.values():
        if len(items) > 1:
            items.sort(key=lambda x: x[1], reverse=True)
            longest = items[0][1]
            for fid, length in items[1:]:
                if fid in geoms and length >= longest * DUPLICATE_LENGTH_RATIO:
                    del geoms[fid]
                    n_duplicates += 1

    # Remove short danglers (primal degree-1, iterative)
    DANGLER_MAX = 10.0
    while True:
        temp_ep: dict[tuple, list[int]] = collections.defaultdict(list)
        for fid, line in geoms.items():
            c = list(line.coords)
            for pt in (c[0], c[-1]):
                temp_ep[_ep_key(pt)].append(fid)
        to_remove: set[int] = set()
        for fid, line in geoms.items():
            if line.length > DANGLER_MAX:
                continue
            c = list(line.coords)
            if len(temp_ep.get(_ep_key(c[0]), [])) <= 1 or len(temp_ep.get(_ep_key(c[-1]), [])) <= 1:
                to_remove.add(fid)
        if not to_remove:
            break
        for fid in to_remove:
            del geoms[fid]
        n_danglers += len(to_remove)

    fid_list = sorted(geoms.keys())
    n_segments = len(fid_list)

    if feedback:
        if n_self_loops:
            feedback.pushInfo(f"Removed {n_self_loops} short self-loop geometries (< {SELF_LOOP_MAX}m).")
        if n_duplicates:
            feedback.pushInfo(f"Removed {n_duplicates} duplicate geometries.")
        if n_danglers:
            feedback.pushInfo(f"Removed {n_danglers} short dangling segments (< {DANGLER_MAX}m).")

    # ---- Build dual nodes ----
    if feedback:
        feedback.setProgressText(f"Step {step} of {n_steps}: Building dual nodes…")
        feedback.setProgress(int(node_base))

    endpoint_to_fids: dict[tuple, list[int]] = collections.defaultdict(list)
    ns = rustalgos.graph.NetworkStructure()
    ns.set_is_dual(True)
    node_idx: dict[int, int] = {}
    midpoints: dict[int, tuple] = {}
    _line_data: dict[int, tuple] = {}

    n_items = len(geoms)
    node_tick = max(1, n_items // 100) if n_items > 0 else 1
    for i, (fid, line) in enumerate(geoms.items()):
        coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(coords)
        _line_data[fid] = (coords, cum)
        for pt in (coords[0], coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)
        mid = _interpolate_at(coords, cum, 0.5)
        live = prepared_boundary is None or prepared_boundary.contains(Point(mid[0], mid[1]))
        idx = ns.add_street_node(
            node_key=fid,
            x=mid[0],
            y=mid[1],
            live=live,
            weight=1.0,
        )
        node_idx[fid] = idx
        midpoints[fid] = mid
        if feedback and (i % node_tick == 0 or i == n_items - 1):
            pct = (i + 1) / n_segments
            feedback.setProgress(int(node_base + pct * node_span))

    n_live = sum(1 for fid in fid_list if ns.is_node_live(node_idx[fid]))
    if feedback:
        feedback.pushInfo(f"Added {n_segments} dual nodes ({n_live} live).")

    # ---- Build dual edges ----
    if feedback:
        feedback.setProgressText(f"Step {step} of {n_steps}: Building dual edges…")
        feedback.setProgress(int(edge_base))

    edge_counter = 0
    seen: set = set()
    n_endpoints = len(endpoint_to_fids)

    edge_tick = max(1, n_endpoints // 100) if n_endpoints > 0 else 1
    for j, (endpoint, fids) in enumerate(endpoint_to_fids.items()):
        for fid_a, fid_b in itertools.combinations(fids, 2):
            pair = frozenset({fid_a, fid_b})
            if pair in seen:
                continue
            seen.add(pair)
            primal_key = str(endpoint)
            merged_wkt = _make_edge_wkt(fid_a, fid_b, endpoint)
            ns.add_street_edge(
                node_idx[fid_a],
                node_idx[fid_b],
                edge_counter,
                fid_a,
                fid_b,
                merged_wkt,
                shared_primal_node_key=primal_key,
            )
            edge_counter += 1
            merged_wkt_rev = _make_edge_wkt(fid_b, fid_a, endpoint)
            ns.add_street_edge(
                node_idx[fid_b],
                node_idx[fid_a],
                edge_counter,
                fid_b,
                fid_a,
                merged_wkt_rev,
                shared_primal_node_key=primal_key,
            )
            edge_counter += 1
        if feedback and (j % edge_tick == 0 or j == n_endpoints - 1):
            pct = (j + 1) / n_endpoints
            feedback.setProgress(int(edge_base + pct * edge_span))

    ns.validate()
    ns.build_edge_rtree()

    if feedback:
        feedback.pushInfo(f"Added {edge_counter} dual edges.")
        feedback.setProgress(int(progress_base + progress_span))

    # Persist state for incremental updates
    _inc_state = {
        "layer_cache_key": layer_cache_key,
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
        "_line_data": _line_data,
    }

    return (ns, fid_list, midpoints, geoms)
