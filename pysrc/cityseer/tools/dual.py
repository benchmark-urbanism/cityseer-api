from __future__ import annotations

import collections
import contextlib
import itertools
import math
from typing import Any

from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge
from shapely.prepared import prep
from shapely.validation import make_valid

from .. import rustalgos

DualInput = dict[Any, str] | dict[Any, BaseGeometry]
DualState = dict[str, Any]

# Treat only tiny loops as corrupted geometry. Longer loops remain valid features.
SELF_LOOP_MIN_LENGTH = 1.0
# Only features with almost identical endpoint-to-endpoint lengths count as duplicates.
# This stays intentionally narrow so distinct curved alternatives are preserved.
DUPLICATE_LENGTH_RATIO = 0.98
DANGLER_MAX = 10.0


def _cumulative_lengths(coords: list[tuple[float, float]]) -> list[float]:
    """Return cumulative distances along a coordinate list."""
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
    """Return the coordinate at a normalized fraction along a line."""
    target = frac * cum[-1]
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
    """Return the coordinate sequence for a normalized substring."""
    total = cum[-1]
    if total == 0.0:
        return list(coords)
    d_start = start_frac * total
    d_end = end_frac * total
    result: list[tuple[float, float]] = []
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
                if cum[i] >= d_end:
                    if seg_len == 0.0:
                        if not result or result[-1] != coords[i]:
                            result.append(coords[i])
                    else:
                        t2 = (d_end - cum[i - 1]) / seg_len
                        x2 = coords[i - 1][0] + t2 * (coords[i][0] - coords[i - 1][0])
                        y2 = coords[i - 1][1] + t2 * (coords[i][1] - coords[i - 1][1])
                        end_pt = (x2, y2)
                        if end_pt != result[-1]:
                            result.append(end_pt)
                    return result
                if coords[i] != result[-1]:
                    result.append(coords[i])
        else:
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
    """Format a coordinate list as a WKT LineString."""
    pairs = ", ".join(f"{x} {y}" for x, y in coords)
    return f"LINESTRING ({pairs})"


def _ep_key(pt: tuple[float, float]) -> tuple[float, float]:
    """Rounded endpoint key for adjacency lookup."""
    return (round(pt[0], 1), round(pt[1], 1))


def extract_wkts(data: DualInput | Any) -> tuple[dict[Any, str], Any | None]:
    """Normalize dict or GeoDataFrame input to a key->WKT mapping."""
    try:
        import geopandas as gpd
    except ImportError:
        gpd = None  # type: ignore[assignment]
    if gpd is not None and isinstance(data, gpd.GeoDataFrame):
        if data.index.duplicated().any():
            raise ValueError("The GeoDataFrame index must contain unique entries.")
        geom_name = data.geometry.name
        wkts = {idx: geom.wkt for idx, geom in data[geom_name].items() if geom is not None and not geom.is_empty}
        return wkts, data.crs
    wkts: dict[Any, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            wkts[key] = value
        elif isinstance(value, BaseGeometry):
            if value.is_empty:
                continue
            wkts[key] = value.wkt
        else:
            raise TypeError(f"Unsupported geometry type for key {key!r}: {type(value)!r}")
    return wkts, None


def _parse_line(value: str | BaseGeometry) -> LineString | None:
    """Parse WKT or shapely geometry into a clean 2D LineString."""
    geom = shapely_wkt.loads(value) if isinstance(value, str) else value
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


def _half_toward_coords(
    fid: Any,
    endpoint: tuple[float, float],
    line_data: dict[Any, tuple[list[tuple[float, float]], list[float]]],
) -> list[tuple[float, float]]:
    """Return coords for the half of a line from midpoint toward an endpoint."""
    coords, _cum = line_data[fid]
    ep = _ep_key(endpoint)
    end = _ep_key(coords[-1])
    if ep == end:
        return _substring_coords(coords, _cum, 0.5, 1.0)
    rev = coords[::-1]
    rev_cum = _cumulative_lengths(rev)
    return _substring_coords(rev, rev_cum, 0.5, 1.0)


def _make_edge_wkt(
    fid_a: Any,
    fid_b: Any,
    endpoint_key: tuple[float, float],
    line_data: dict[Any, tuple[list[tuple[float, float]], list[float]]],
) -> str:
    """Create directed merged WKT from one segment midpoint to another."""
    ha = _half_toward_coords(fid_a, endpoint_key, line_data)
    hb = _half_toward_coords(fid_b, endpoint_key, line_data)
    merged = ha + hb[::-1][1:]
    return _coords_to_wkt(merged)


def _clean_geometries(
    geoms: dict[Any, LineString],
) -> tuple[dict[Any, LineString], dict[Any, str], int, int, int]:
    """Apply the same self-loop, duplicate, and dangler cleanup as the QGIS fast path."""
    geoms = dict(geoms)
    statuses = {fid: "active" for fid in geoms}
    n_self_loops = 0
    n_duplicates = 0
    n_danglers = 0

    for fid in list(geoms.keys()):
        coords = list(geoms[fid].coords)
        if _ep_key(coords[0]) == _ep_key(coords[-1]) and geoms[fid].length < SELF_LOOP_MIN_LENGTH:
            statuses[fid] = "short_self_loop"
            del geoms[fid]
            n_self_loops += 1

    ep_pairs: dict[frozenset[tuple[float, float]], list[tuple[Any, float]]] = collections.defaultdict(list)
    for fid, line in geoms.items():
        coords = list(line.coords)
        pair = frozenset({_ep_key(coords[0]), _ep_key(coords[-1])})
        ep_pairs[pair].append((fid, line.length))
    for items in ep_pairs.values():
        if len(items) > 1:
            items.sort(key=lambda x: x[1], reverse=True)
            longest = items[0][1]
            for fid, length in items[1:]:
                if fid in geoms and length >= longest * DUPLICATE_LENGTH_RATIO:
                    statuses[fid] = "duplicate"
                    del geoms[fid]
                    n_duplicates += 1

    while True:
        temp_ep: dict[tuple[float, float], list[Any]] = collections.defaultdict(list)
        for fid, line in geoms.items():
            coords = list(line.coords)
            for pt in (coords[0], coords[-1]):
                temp_ep[_ep_key(pt)].append(fid)
        to_remove: set[Any] = set()
        for fid, line in geoms.items():
            if line.length > DANGLER_MAX:
                continue
            coords = list(line.coords)
            if len(temp_ep.get(_ep_key(coords[0]), [])) <= 1 or len(temp_ep.get(_ep_key(coords[-1]), [])) <= 1:
                to_remove.add(fid)
        if not to_remove:
            break
        for fid in to_remove:
            statuses[fid] = "short_dangler"
            del geoms[fid]
        n_danglers += len(to_remove)

    return geoms, statuses, n_self_loops, n_duplicates, n_danglers


def _build_nodes_gdf(
    ns: rustalgos.graph.NetworkStructure,
    fid_list: list[Any],
    node_idx: dict[Any, int],
    midpoints: dict[Any, tuple[float, float]],
    crs: Any | None,
) -> Any:
    import geopandas as gpd

    return gpd.GeoDataFrame(  # type: ignore[call-overload]
        {
            "ns_node_idx": [node_idx[fid] for fid in fid_list],
            "x": [midpoints[fid][0] for fid in fid_list],
            "y": [midpoints[fid][1] for fid in fid_list],
            "live": [ns.is_node_live(node_idx[fid]) for fid in fid_list],
            "weight": 1.0,
        },
        index=fid_list,
        geometry=[Point(midpoints[fid]) for fid in fid_list],
        crs=crs,
    )


def _edge_record(
    start_key: Any,
    end_key: Any,
    edge_idx: int,
    geom_wkt: str,
    shared_primal_node_key: str | None,
) -> dict[str, Any]:
    return {
        "start_key": start_key,
        "end_key": end_key,
        "edge_idx": edge_idx,
        "geom_wkt": geom_wkt,
        "imp_factor": 1.0,
        "shared_primal_node_key": shared_primal_node_key,
    }


def _active_wkts(source_wkts: dict[Any, str], fid_list: list[Any]) -> dict[Any, str]:
    return {fid: source_wkts[fid] for fid in fid_list if fid in source_wkts}


def _neighboring_fids(state: DualState, fid: Any) -> set[Any]:
    neighbors: set[Any] = set()
    geoms = state["geoms"]
    endpoint_to_fids = state["endpoint_to_fids"]
    if fid not in geoms:
        return neighbors
    coords = list(geoms[fid].coords)
    for pt in (coords[0], coords[-1]):
        for other_fid in endpoint_to_fids.get(_ep_key(pt), []):
            if other_fid != fid:
                neighbors.add(other_fid)
    return neighbors


def _requires_full_rebuild(
    state: DualState,
    current_source_wkts: dict[Any, str],
    to_remove: set[Any],
    to_add: set[Any],
) -> bool:
    geoms = state["geoms"]
    node_idx = state["node_idx"]

    if any(fid not in node_idx for fid in to_remove):
        return True

    for fid in to_remove:
        if fid in geoms and geoms[fid].length <= DANGLER_MAX:
            return True
        for other_fid in _neighboring_fids(state, fid):
            if other_fid in geoms and geoms[other_fid].length <= DANGLER_MAX:
                return True

    ep_pair_best: dict[frozenset[tuple[float, float]], tuple[Any, float]] = {}
    for existing_fid, existing_line in geoms.items():
        if existing_fid in to_remove:
            continue
        coords = list(existing_line.coords)
        pair = frozenset({_ep_key(coords[0]), _ep_key(coords[-1])})
        if pair not in ep_pair_best or existing_line.length > ep_pair_best[pair][1]:
            ep_pair_best[pair] = (existing_fid, existing_line.length)

    for fid in to_add:
        line = _parse_line(current_source_wkts[fid])
        if line is None:
            return True
        coords = list(line.coords)
        if _ep_key(coords[0]) == _ep_key(coords[-1]) and line.length < SELF_LOOP_MIN_LENGTH:
            return True
        if line.length <= DANGLER_MAX:
            return True
        pair = frozenset({_ep_key(coords[0]), _ep_key(coords[-1])})
        if pair in ep_pair_best and line.length >= ep_pair_best[pair][1] * DUPLICATE_LENGTH_RATIO:
            return True
        for pt in (coords[0], coords[-1]):
            for other_fid in state["endpoint_to_fids"].get(_ep_key(pt), []):
                if other_fid in geoms and geoms[other_fid].length <= DANGLER_MAX:
                    return True

    return False


def build_dual(
    data: DualInput | Any,
    *,
    crs: Any | None = None,
    boundary: BaseGeometry | None = None,
    build_nodes_gdf: bool = True,
    progress: bool = True,
) -> tuple[rustalgos.graph.NetworkStructure, Any | None, DualState]:
    """Build a dual NetworkStructure directly from line geometries."""
    if progress:
        from tqdm import tqdm
    else:

        def tqdm(iterable, **_kwargs):
            return iterable

    source_wkts, discovered_crs = extract_wkts(data)
    crs = crs if crs is not None else discovered_crs
    prepared_boundary = prep(boundary) if boundary is not None else None
    boundary_wkt = boundary.wkt if boundary is not None else None

    if not source_wkts:
        raise ValueError("Input contains no readable line geometries.")

    raw_geoms: dict[Any, LineString] = {}
    for fid, value in tqdm(source_wkts.items(), total=len(source_wkts), desc="Parsing geometries", mininterval=0.1):
        line = _parse_line(value)
        if line is not None:
            raw_geoms[fid] = line

    feature_status = {fid: "invalid_geometry" for fid in source_wkts}
    for fid in raw_geoms:
        feature_status[fid] = "active"

    geoms, cleaned_status, _n_self_loops, _n_duplicates, _n_danglers = _clean_geometries(raw_geoms)
    feature_status.update(cleaned_status)
    if not geoms:
        raise ValueError("No valid network geometries remained after cleanup.")

    fid_list = sorted(geoms.keys())
    ns = rustalgos.graph.NetworkStructure()
    ns.set_is_dual(True)
    endpoint_to_fids: dict[tuple[float, float], list[Any]] = collections.defaultdict(list)
    node_idx: dict[Any, int] = {}
    midpoints: dict[Any, tuple[float, float]] = {}
    line_data: dict[Any, tuple[list[tuple[float, float]], list[float]]] = {}

    for fid in tqdm(fid_list, desc="Building nodes", mininterval=0.1):
        line = geoms[fid]
        coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(coords)
        line_data[fid] = (coords, cum)
        for pt in (coords[0], coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)
        mid = _interpolate_at(coords, cum, 0.5)
        live = prepared_boundary is None or prepared_boundary.contains(Point(mid))
        idx = ns.add_street_node(
            node_key=fid,
            x=mid[0],
            y=mid[1],
            live=live,
            weight=1.0,
        )
        node_idx[fid] = idx
        midpoints[fid] = mid

    edge_counter = 0
    seen: set[frozenset[Any]] = set()
    edge_records: dict[tuple[Any, Any, int], dict[str, Any]] = {}
    # Pre-collect unique edge pairs for smooth progress tracking
    edge_pairs: list[tuple[Any, Any, tuple[float, float]]] = []
    for endpoint, fids in endpoint_to_fids.items():
        for fid_a, fid_b in itertools.combinations(fids, 2):
            pair = frozenset({fid_a, fid_b})
            if pair not in seen:
                seen.add(pair)
                edge_pairs.append((fid_a, fid_b, endpoint))
    seen.clear()
    for fid_a, fid_b, endpoint in tqdm(edge_pairs, desc="Building edges", mininterval=0.1):
        pair = frozenset({fid_a, fid_b})
        seen.add(pair)
        shared_key = str(endpoint)
        merged_wkt = _make_edge_wkt(fid_a, fid_b, endpoint, line_data)
        ns.add_street_edge(
            node_idx[fid_a],
            node_idx[fid_b],
            edge_counter,
            fid_a,
            fid_b,
            merged_wkt,
            shared_primal_node_key=shared_key,
        )
        edge_records[(fid_a, fid_b, edge_counter)] = _edge_record(
            fid_a,
            fid_b,
            edge_counter,
            merged_wkt,
            shared_key,
        )
        edge_counter += 1

        merged_wkt_rev = _make_edge_wkt(fid_b, fid_a, endpoint, line_data)
        ns.add_street_edge(
            node_idx[fid_b],
            node_idx[fid_a],
            edge_counter,
            fid_b,
            fid_a,
            merged_wkt_rev,
            shared_primal_node_key=shared_key,
        )
        edge_records[(fid_b, fid_a, edge_counter)] = _edge_record(
            fid_b,
            fid_a,
            edge_counter,
            merged_wkt_rev,
            shared_key,
        )
        edge_counter += 1

    ns.validate()
    ns.build_edge_rtree()
    nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs) if build_nodes_gdf else None
    state: DualState = {
        "ns": ns,
        "wkts": _active_wkts(source_wkts, fid_list),
        "source_wkts": dict(source_wkts),
        "feature_status": feature_status,
        "fid_list": fid_list,
        "geoms": geoms,
        "midpoints": midpoints,
        "node_idx": node_idx,
        "endpoint_to_fids": endpoint_to_fids,
        "edge_counter": edge_counter,
        "seen": seen,
        "boundary_wkt": boundary_wkt,
        "_line_data": line_data,
        "crs": crs,
        "edge_records": edge_records,
    }
    return ns, nodes_gdf, state


def incremental_update(
    state: DualState,
    data: DualInput | Any,
    *,
    crs: Any | None = None,
    boundary: BaseGeometry | None = None,
    build_nodes_gdf: bool = True,
    progress: bool = True,
) -> tuple[rustalgos.graph.NetworkStructure, Any | None, DualState]:
    """Apply an incremental diff to a previously built dual network."""
    current_source_wkts, discovered_crs = extract_wkts(data)
    crs = crs if crs is not None else discovered_crs if discovered_crs is not None else state.get("crs")
    prepared_boundary = prep(boundary) if boundary is not None else None
    boundary_wkt = boundary.wkt if boundary is not None else None

    prev_source_wkts = state.get("source_wkts", state["wkts"])
    prev_fids = set(prev_source_wkts.keys())
    curr_fids = set(current_source_wkts.keys())
    removed = prev_fids - curr_fids
    added = curr_fids - prev_fids
    modified = {fid for fid in prev_fids & curr_fids if prev_source_wkts[fid] != current_source_wkts[fid]}
    to_remove = removed | modified
    to_add = added | modified
    boundary_changed = boundary_wkt != state.get("boundary_wkt")

    ns = state.get("ns", None)
    if ns is None:
        ns, _nodes_gdf, state = build_dual(
            current_source_wkts,
            crs=crs,
            boundary=boundary,
            build_nodes_gdf=build_nodes_gdf,
            progress=progress,
        )
        state["ns"] = ns
        return ns, _nodes_gdf, state

    if (to_remove or to_add) and _requires_full_rebuild(state, current_source_wkts, to_remove, to_add):
        ns, nodes_gdf, rebuilt_state = build_dual(
            current_source_wkts,
            crs=crs,
            boundary=boundary,
            build_nodes_gdf=build_nodes_gdf,
            progress=progress,
        )
        feature_status = rebuilt_state.get("feature_status", {})
        for fid in removed:
            feature_status[fid] = "deleted"
        rebuilt_state["feature_status"] = feature_status
        return ns, nodes_gdf, rebuilt_state

    fid_list = state["fid_list"]
    geoms = state["geoms"]
    midpoints = state["midpoints"]
    node_idx = state["node_idx"]
    endpoint_to_fids = state["endpoint_to_fids"]
    edge_counter = state["edge_counter"]
    seen = state["seen"]
    line_data = state["_line_data"]
    edge_records = state["edge_records"]
    feature_status = dict(state.get("feature_status", {}))

    if not to_remove and not to_add and not boundary_changed:
        nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs) if build_nodes_gdf else None
        state["crs"] = crs
        state["ns"] = ns
        state["source_wkts"] = dict(current_source_wkts)
        state["feature_status"] = feature_status
        return ns, nodes_gdf, state

    for fid in to_remove:
        feature_status[fid] = "deleted"
        if fid not in node_idx:
            continue
        ns.remove_street_node(node_idx[fid])
        coords, _cum = line_data[fid]
        for pt in (coords[0], coords[-1]):
            key = _ep_key(pt)
            if key in endpoint_to_fids:
                with contextlib.suppress(ValueError):
                    endpoint_to_fids[key].remove(fid)
                if not endpoint_to_fids[key]:
                    del endpoint_to_fids[key]
        del geoms[fid]
        del line_data[fid]
        del midpoints[fid]
        del node_idx[fid]

    fid_list = [fid for fid in fid_list if fid not in to_remove]
    seen = {pair for pair in seen if not (pair & to_remove)}
    edge_records = {
        ref: record
        for ref, record in edge_records.items()
        if record["start_key"] not in to_remove and record["end_key"] not in to_remove
    }

    ep_pair_best: dict[frozenset[tuple[float, float]], tuple[Any, float]] = {}
    for existing_fid, existing_line in geoms.items():
        coords = list(existing_line.coords)
        pair = frozenset({_ep_key(coords[0]), _ep_key(coords[-1])})
        if pair not in ep_pair_best or existing_line.length > ep_pair_best[pair][1]:
            ep_pair_best[pair] = (existing_fid, existing_line.length)

    for fid in to_add:
        line = _parse_line(current_source_wkts[fid])
        if line is None:
            continue
        coords = list(line.coords)
        if _ep_key(coords[0]) == _ep_key(coords[-1]) and line.length < SELF_LOOP_MIN_LENGTH:
            continue
        pair = frozenset({_ep_key(coords[0]), _ep_key(coords[-1])})
        if pair in ep_pair_best and line.length >= ep_pair_best[pair][1] * DUPLICATE_LENGTH_RATIO:
            continue
        ep_pair_best[pair] = (fid, line.length)
        geoms[fid] = line
        fid_list.append(fid)
        clean_coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(clean_coords)
        line_data[fid] = (clean_coords, cum)
        mid = _interpolate_at(clean_coords, cum, 0.5)
        live = prepared_boundary is None or prepared_boundary.contains(Point(mid))
        idx = ns.add_street_node(
            node_key=fid,
            x=mid[0],
            y=mid[1],
            live=live,
            weight=1.0,
        )
        node_idx[fid] = idx
        midpoints[fid] = mid
        feature_status[fid] = "active"
        for pt in (clean_coords[0], clean_coords[-1]):
            endpoint_to_fids[_ep_key(pt)].append(fid)

    fid_list.sort()
    for fid in [fid for fid in to_add if fid in geoms]:
        coords, _cum = line_data[fid]
        for pt in (coords[0], coords[-1]):
            key = _ep_key(pt)
            for other_fid in endpoint_to_fids.get(key, []):
                if other_fid == fid:
                    continue
                pair = frozenset({fid, other_fid})
                if pair in seen:
                    continue
                seen.add(pair)
                shared_key = str(key)
                merged_wkt = _make_edge_wkt(fid, other_fid, key, line_data)
                ns.add_street_edge(
                    node_idx[fid],
                    node_idx[other_fid],
                    edge_counter,
                    fid,
                    other_fid,
                    merged_wkt,
                    shared_primal_node_key=shared_key,
                )
                edge_records[(fid, other_fid, edge_counter)] = _edge_record(
                    fid,
                    other_fid,
                    edge_counter,
                    merged_wkt,
                    shared_key,
                )
                edge_counter += 1

                merged_wkt_rev = _make_edge_wkt(other_fid, fid, key, line_data)
                ns.add_street_edge(
                    node_idx[other_fid],
                    node_idx[fid],
                    edge_counter,
                    other_fid,
                    fid,
                    merged_wkt_rev,
                    shared_primal_node_key=shared_key,
                )
                edge_records[(other_fid, fid, edge_counter)] = _edge_record(
                    other_fid,
                    fid,
                    edge_counter,
                    merged_wkt_rev,
                    shared_key,
                )
                edge_counter += 1

    if boundary_changed:
        for fid in fid_list:
            mid = midpoints[fid]
            live = prepared_boundary is None or prepared_boundary.contains(Point(mid))
            ns.set_node_live(node_idx[fid], live)

    ns.validate()
    ns.build_edge_rtree()
    state.update(
        {
            "wkts": _active_wkts(current_source_wkts, fid_list),
            "source_wkts": dict(current_source_wkts),
            "feature_status": feature_status,
            "fid_list": fid_list,
            "geoms": geoms,
            "midpoints": midpoints,
            "node_idx": node_idx,
            "endpoint_to_fids": endpoint_to_fids,
            "edge_counter": edge_counter,
            "seen": seen,
            "boundary_wkt": boundary_wkt,
            "_line_data": line_data,
            "crs": crs,
            "edge_records": edge_records,
            "ns": ns,
        }
    )
    nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs) if build_nodes_gdf else None
    return ns, nodes_gdf, state
