from __future__ import annotations

import collections
import contextlib
import itertools
import math
from dataclasses import dataclass
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


@dataclass
class _DualBuildContext:
    """Bundles the per-build shared state that `_try_add_edge` needs.

    All fields hold mutable references that the caller (re)uses across many edge insertions;
    the dataclass exists purely to keep `_try_add_edge`'s argument list manageable.
    """

    ns: rustalgos.graph.NetworkStructure
    line_data: dict[Any, tuple[list[tuple[float, float]], list[float]]]
    directions: dict[Any, tuple[bool, bool]] | None
    ep_keys: dict[Any, tuple[tuple[float, float], tuple[float, float]]]
    node_idx: dict[Any, int]
    edge_records: dict[tuple[Any, Any, int], dict[str, Any]]
    impedances: dict[Any, float]


# Treat only tiny loops as corrupted geometry. Longer loops remain valid features.
SELF_LOOP_MIN_LENGTH = 1.0
# Only features with almost identical endpoint-to-endpoint lengths count as duplicates.
# This stays intentionally narrow so distinct curved alternatives are preserved.
DUPLICATE_LENGTH_RATIO = 0.98
DANGLER_MAX = 10.0
# Default endpoint tolerance for merging parallel (near-duplicate) edges: edges whose two
# endpoints each fall within this distance of another edge's endpoints, and whose lengths are
# near-identical (DUPLICATE_LENGTH_RATIO), are the same street drawn twice.
MERGE_PARALLEL_DIST = 2.0

# Pre-5.5 cleaning behaviour: no filler welding, and near-duplicate detection at the endpoint
# rounding precision only. The QGIS plugin pins these to avoid changing plugin outputs, and
# saved states from older versions fall back to them.
LEGACY_CLEAN_PARAMS: dict[str, Any] = {
    "remove_fillers": False,
    "remove_danglers": DANGLER_MAX,
    "merge_parallel_dist": 0.1,
}


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
        gpd = None  # type: ignore
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


def _weld_fillers(
    geoms: dict[Any, LineString],
    statuses: dict[Any, str],
) -> dict[Any, Any]:
    """Weld chains of segments meeting at filler (degree-2) endpoints into single segments.

    A filler endpoint joins exactly two distinct segments and no others: it subdivides one
    continuous street rather than marking a junction. Each weld keeps the id of the longer
    constituent; absorbed ids are marked ``"merged"`` and recorded in the returned mapping
    (absorbed id -> kept id), resolved transitively across chained welds. Welds that would
    close a segment into a ring are skipped, since a ring would detach from the dual graph.
    Undirected only: callers skip this pass for directed graphs, where welding could join
    edges with conflicting one-way orientations.
    """
    merges: dict[Any, Any] = {}
    ep_map: dict[tuple[float, float], set[Any]] = collections.defaultdict(set)
    for fid, line in geoms.items():
        coords = list(line.coords)
        k_start, k_end = _ep_key(coords[0]), _ep_key(coords[-1])
        if k_start == k_end:
            continue  # rings have no weldable endpoints
        ep_map[k_start].add(fid)
        ep_map[k_end].add(fid)
    queue = collections.deque(key for key, fids in ep_map.items() if len(fids) == 2)
    while queue:
        key = queue.popleft()
        fids = ep_map.get(key)
        if fids is None or len(fids) != 2:
            continue
        fid_a, fid_b = tuple(fids)
        if fid_a not in geoms or fid_b not in geoms:
            continue
        coords_a = list(geoms[fid_a].coords)
        coords_b = list(geoms[fid_b].coords)
        # orient a to end at the weld point and b to start at it
        if _ep_key(coords_a[0]) == key:
            coords_a = coords_a[::-1]
        if _ep_key(coords_b[-1]) == key:
            coords_b = coords_b[::-1]
        far_a = _ep_key(coords_a[0])
        far_b = _ep_key(coords_b[-1])
        if far_a == far_b:
            continue  # welding would create a ring
        keep, drop = (fid_a, fid_b) if geoms[fid_a].length >= geoms[fid_b].length else (fid_b, fid_a)
        geoms[keep] = LineString(coords_a + coords_b[1:])
        del geoms[drop]
        statuses[drop] = "merged"
        merges[drop] = keep
        for absorbed, kept in merges.items():
            if kept == drop:
                merges[absorbed] = keep
        # update the endpoint map in place: the weld point closes; the far endpoints now
        # reference the kept id, and may themselves have become weldable
        del ep_map[key]
        for far_key, old_fid in ((far_a, fid_a), (far_b, fid_b)):
            fid_set = ep_map.get(far_key)
            if fid_set is None:
                continue
            fid_set.discard(old_fid)
            fid_set.add(keep)
            if len(fid_set) == 2:
                queue.append(far_key)
    return merges


def _remove_danglers(
    geoms: dict[Any, LineString],
    statuses: dict[Any, str],
    dangler_max: float,
) -> int:
    """Iteratively remove short dead-end segments (an endpoint shared with no other segment)."""
    n_danglers = 0
    if dangler_max <= 0:
        return n_danglers
    while True:
        temp_ep: dict[tuple[float, float], list[Any]] = collections.defaultdict(list)
        for fid, line in geoms.items():
            coords = list(line.coords)
            for pt in (coords[0], coords[-1]):
                temp_ep[_ep_key(pt)].append(fid)
        to_remove: set[Any] = set()
        for fid, line in geoms.items():
            if line.length > dangler_max:
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
    return n_danglers


def _cluster_endpoints(
    geoms: dict[Any, LineString],
    dist: float,
) -> dict[tuple[float, float], int]:
    """Group endpoint keys lying within ``dist`` of each other via grid hashing + union-find."""
    keys: list[tuple[float, float]] = []
    key_idx: dict[tuple[float, float], int] = {}
    for line in geoms.values():
        coords = list(line.coords)
        for pt in (coords[0], coords[-1]):
            key = _ep_key(pt)
            if key not in key_idx:
                key_idx[key] = len(keys)
                keys.append(key)
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i, (x, y) in enumerate(keys):
        grid[(int(x // dist), int(y // dist))].append(i)
    for i, (x, y) in enumerate(keys):
        cx, cy = int(x // dist), int(y // dist)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):  # noqa: B905
                    if j <= i:
                        continue
                    jx, jy = keys[j]
                    if math.hypot(jx - x, jy - y) <= dist:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
    return {key: find(i) for key, i in key_idx.items()}


def _merge_parallel(
    geoms: dict[Any, LineString],
    statuses: dict[Any, str],
    parallel_dist: float,
) -> int:
    """Remove near-duplicate parallel edges: endpoints within ``parallel_dist``, near-equal length.

    This generalises exact endpoint-pair duplicate detection: endpoints are clustered within
    the tolerance, and edges spanning the same endpoint clusters with lengths within
    ``DUPLICATE_LENGTH_RATIO`` of the longest are the same street drawn twice. The longest is
    kept. Distinctly shorter or longer alternatives between the same clusters are preserved.
    """
    n_duplicates = 0
    if parallel_dist <= 0:
        return n_duplicates
    if parallel_dist <= 0.1:
        # at (or below) the endpoint rounding precision, exact key matching is the legacy
        # behaviour: only edges sharing both rounded endpoints count as duplicates
        clusters = {}
        for line in geoms.values():
            coords = list(line.coords)
            for pt in (coords[0], coords[-1]):
                key = _ep_key(pt)
                clusters.setdefault(key, len(clusters))
    else:
        clusters = _cluster_endpoints(geoms, parallel_dist)
    ep_pairs: dict[frozenset[int], list[tuple[Any, float]]] = collections.defaultdict(list)
    for fid, line in geoms.items():
        coords = list(line.coords)
        c_start = clusters[_ep_key(coords[0])]
        c_end = clusters[_ep_key(coords[-1])]
        if c_start == c_end:
            continue  # both endpoints in one cluster: a loop at the tolerance, not a parallel pair
        ep_pairs[frozenset({c_start, c_end})].append((fid, line.length))
    for items in ep_pairs.values():
        if len(items) > 1:
            # pairwise near-equality: an edge is a duplicate of any longer KEPT edge with a
            # near-identical length, so a distinctly longer alternative between the same
            # endpoints does not shield true twins from each other
            items.sort(key=lambda x: x[1], reverse=True)
            kept_lengths: list[float] = []
            for fid, length in items:
                if fid not in geoms:
                    continue
                if any(length >= kept_len * DUPLICATE_LENGTH_RATIO for kept_len in kept_lengths):
                    statuses[fid] = "duplicate"
                    del geoms[fid]
                    n_duplicates += 1
                else:
                    kept_lengths.append(length)
    return n_duplicates


def _clean_geometries(
    geoms: dict[Any, LineString],
    directed: bool = False,
    *,
    remove_fillers: bool = True,
    remove_danglers: float = DANGLER_MAX,
    merge_parallel_dist: float = MERGE_PARALLEL_DIST,
) -> tuple[dict[Any, LineString], dict[Any, str], dict[Any, Any]]:
    """Clean the primal edge set ahead of dual conversion.

    Order matters and is deliberate: tiny self-loops (corrupt geometry) first, then filler
    welding so that subdivided streets are treated as whole segments, then dangler removal so
    that a welded chain is judged on its full length, then parallel-edge merging.

    Directed graphs skip filler welding and parallel merging: welding can join edges with
    conflicting one-way orientations, and reverse edges sharing endpoints are distinct traffic
    directions rather than duplicates.

    Returns the cleaned geoms, per-feature statuses, and the weld mapping (absorbed id -> kept id).
    """
    geoms = dict(geoms)
    statuses = {fid: "active" for fid in geoms}
    merges: dict[Any, Any] = {}

    for fid in list(geoms.keys()):
        coords = list(geoms[fid].coords)
        if _ep_key(coords[0]) == _ep_key(coords[-1]) and geoms[fid].length < SELF_LOOP_MIN_LENGTH:
            statuses[fid] = "short_self_loop"
            del geoms[fid]

    if remove_fillers and not directed:
        merges = _weld_fillers(geoms, statuses)

    _remove_danglers(geoms, statuses, remove_danglers)

    if not directed:
        _merge_parallel(geoms, statuses, merge_parallel_dist)

    return geoms, statuses, merges


def _build_nodes_gdf(
    ns: rustalgos.graph.NetworkStructure,
    fid_list: list[Any],
    node_idx: dict[Any, int],
    midpoints: dict[Any, tuple[float, float]],
    crs: Any | None,
    geoms: dict[Any, LineString] | None = None,
) -> Any:
    import geopandas as gpd

    data: dict[str, Any] = {
        "ns_node_idx": [node_idx[fid] for fid in fid_list],
        "x": [midpoints[fid][0] for fid in fid_list],
        "y": [midpoints[fid][1] for fid in fid_list],
        "live": [ns.is_node_live(node_idx[fid]) for fid in fid_list],
        "weight": 1.0,
    }
    if geoms is not None:
        # primal segment length: enables segment_weighted centrality (plain float, parquet-safe)
        data["seg_length"] = [geoms[fid].length for fid in fid_list]
    return gpd.GeoDataFrame(  # type: ignore
        data,
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
    imp_factor: float = 1.0,
) -> dict[str, Any]:
    return {
        "start_key": start_key,
        "end_key": end_key,
        "edge_idx": edge_idx,
        "geom_wkt": geom_wkt,
        "imp_factor": imp_factor,
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


def _can_traverse(
    fid_from: Any,
    fid_to: Any,
    endpoint: tuple[float, float],
    ep_keys: dict[Any, tuple[tuple[float, float], tuple[float, float]]],
    directions: dict[Any, tuple[bool, bool]],
) -> bool:
    """Check if traffic can flow from fid_from to fid_to at a shared endpoint."""
    start_from, end_from = ep_keys[fid_from]
    start_to, end_to = ep_keys[fid_to]
    fwd_from, rev_from = directions[fid_from]
    fwd_to, rev_to = directions[fid_to]
    can_exit = (end_from == endpoint and fwd_from) or (start_from == endpoint and rev_from)
    can_enter = (start_to == endpoint and fwd_to) or (end_to == endpoint and rev_to)
    return can_exit and can_enter


def _try_add_edge(
    ctx: _DualBuildContext,
    fid_from: Any,
    fid_to: Any,
    endpoint: tuple[float, float],
    edge_counter: int,
    shared_key: str,
) -> int:
    """Add a single directed edge if traversal is allowed. Returns updated edge_counter."""
    if ctx.directions is not None and not _can_traverse(fid_from, fid_to, endpoint, ctx.ep_keys, ctx.directions):
        return edge_counter
    merged_wkt = _make_edge_wkt(fid_from, fid_to, endpoint, ctx.line_data)
    # The dual edge traverses half of each adjacent primal segment, so propagate impedance as
    # the length-weighted mean of the two primal imp_factors. All-1.0 primals -> 1.0 on the dual.
    len_from = ctx.line_data[fid_from][1][-1]
    len_to = ctx.line_data[fid_to][1][-1]
    imp_from = ctx.impedances.get(fid_from, 1.0)
    imp_to = ctx.impedances.get(fid_to, 1.0)
    total_len = len_from + len_to
    dual_imp = (len_from * imp_from + len_to * imp_to) / total_len if total_len > 0.0 else 1.0
    ctx.ns.add_street_edge(
        ctx.node_idx[fid_from],
        ctx.node_idx[fid_to],
        edge_counter,
        fid_from,
        fid_to,
        merged_wkt,
        imp_factor=dual_imp,
        shared_primal_node_key=shared_key,
    )
    ctx.edge_records[(fid_from, fid_to, edge_counter)] = _edge_record(
        fid_from,
        fid_to,
        edge_counter,
        merged_wkt,
        shared_key,
        imp_factor=dual_imp,
    )
    return edge_counter + 1


def build_dual(
    data: DualInput | Any,
    *,
    crs: Any | None = None,
    boundary: BaseGeometry | None = None,
    build_nodes_gdf: bool = True,
    progress: bool = True,
    directions: dict[Any, tuple[bool, bool]] | None = None,
    impedances: dict[Any, float] | None = None,
    remove_fillers: bool = True,
    remove_danglers: float = DANGLER_MAX,
    merge_parallel_dist: float = MERGE_PARALLEL_DIST,
) -> tuple[rustalgos.graph.NetworkStructure, Any | None, DualState]:
    """Build a dual NetworkStructure directly from line geometries.

    Cleaning runs on the primal edge set before dual conversion, in this order: tiny self-loop
    removal, filler welding, dangler removal, parallel-edge merging.

    Parameters
    ----------
    directions: dict[Any, tuple[bool, bool]] | None
        Optional mapping from feature ID to ``(forward_allowed, reverse_allowed)`` where forward
        follows the LineString coordinate order. When provided, the graph is built as directed:
        dual edges are only added where traffic flow permits. ``None`` (default) builds an
        undirected graph with edges in both directions.
    impedances: dict[Any, float] | None
        Optional mapping from primal feature ID to its impedance factor. Each dual edge's
        ``imp_factor`` is computed as the length-weighted mean of the two adjacent primal
        segments' impedances. Missing entries default to ``1.0``; ``None`` leaves every dual edge
        at ``1.0``.
    remove_fillers: bool
        Weld chains of segments meeting at filler (degree-2) endpoints into single segments, so
        a street subdivided during digitisation is treated as one segment. Welded features keep
        the id of the longest constituent; absorbed features are marked ``"merged"`` in the
        feature statuses. Skipped for directed graphs. Default ``True``.
    remove_danglers: float
        Remove dead-end stubs up to this length in metres, iteratively. ``0`` disables.
        Default ``10.0``.
    merge_parallel_dist: float
        Merge near-duplicate parallel edges: edges whose endpoints fall within this distance of
        another edge's endpoints and whose lengths are near-identical. The longest is kept.
        ``0`` disables. Skipped for directed graphs. Default ``2.0``.
    """
    if impedances is None:
        impedances = {}
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

    geoms, cleaned_status, merges = _clean_geometries(
        raw_geoms,
        directed=directions is not None,
        remove_fillers=remove_fillers,
        remove_danglers=remove_danglers,
        merge_parallel_dist=merge_parallel_dist,
    )
    feature_status.update(cleaned_status)
    if not geoms:
        raise ValueError("No valid network geometries remained after cleanup.")

    # a welded segment inherits the length-weighted mean of its constituents' impedances,
    # consistent with how dual edges combine the impedances of their two primal halves.
    # combined values go into a working copy only: the state keeps the caller's input
    # impedances so that save/load or incremental rebuilds recombine from the originals
    # rather than compounding already-combined values.
    effective_impedances = dict(impedances)
    if impedances and merges:
        constituents: dict[Any, list[Any]] = collections.defaultdict(list)
        for absorbed, kept in merges.items():
            constituents[kept].append(absorbed)
        for kept, absorbed_fids in constituents.items():
            if kept not in geoms:
                continue  # the welded segment was itself removed downstream (e.g. as a dangler)
            parts = [kept, *absorbed_fids]
            total_len = sum(raw_geoms[fid].length for fid in parts)
            if total_len > 0.0:
                effective_impedances[kept] = (
                    sum(raw_geoms[fid].length * impedances.get(fid, 1.0) for fid in parts) / total_len
                )

    fid_list = sorted(geoms.keys())
    ns = rustalgos.graph.NetworkStructure()
    ns.set_is_dual(True)
    if directions is not None:
        ns.set_is_directed(True)
    endpoint_to_fids: dict[tuple[float, float], list[Any]] = collections.defaultdict(list)
    node_idx: dict[Any, int] = {}
    midpoints: dict[Any, tuple[float, float]] = {}
    line_data: dict[Any, tuple[list[tuple[float, float]], list[float]]] = {}

    ep_keys: dict[Any, tuple[tuple[float, float], tuple[float, float]]] = {}
    for fid in tqdm(fid_list, desc="Building nodes", mininterval=0.1):
        line = geoms[fid]
        coords = [(c[0], c[1]) for c in line.coords]
        cum = _cumulative_lengths(coords)
        line_data[fid] = (coords, cum)
        start_key = _ep_key(coords[0])
        end_key = _ep_key(coords[-1])
        ep_keys[fid] = (start_key, end_key)
        for pt in (start_key, end_key):
            endpoint_to_fids[pt].append(fid)
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
    ctx = _DualBuildContext(
        ns=ns,
        line_data=line_data,
        directions=directions,
        ep_keys=ep_keys,
        node_idx=node_idx,
        edge_records=edge_records,
        impedances=effective_impedances,
    )
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
        edge_counter = _try_add_edge(ctx, fid_a, fid_b, endpoint, edge_counter, shared_key)
        edge_counter = _try_add_edge(ctx, fid_b, fid_a, endpoint, edge_counter, shared_key)

    ns.validate()
    ns.build_edge_rtree()
    nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs, geoms=geoms) if build_nodes_gdf else None
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
        "_ep_keys": ep_keys,
        "crs": crs,
        "edge_records": edge_records,
        "directions": directions,
        "impedances": dict(impedances),
        "effective_impedances": dict(effective_impedances),
        "merges": merges,
        "clean_params": {
            "remove_fillers": remove_fillers,
            "remove_danglers": remove_danglers,
            "merge_parallel_dist": merge_parallel_dist,
        },
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
    directions: dict[Any, tuple[bool, bool]] | None = None,
) -> tuple[rustalgos.graph.NetworkStructure, Any | None, DualState]:
    """Apply an incremental diff to a previously built dual network."""
    current_source_wkts, discovered_crs = extract_wkts(data)
    crs = crs if crs is not None else discovered_crs if discovered_crs is not None else state.get("crs")
    # Carry forward the impedances captured at the original build; new fids fall back to 1.0.
    impedances: dict[Any, float] = state.get("impedances", {})
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

    # Cleaning params travel with the state; states saved before they existed imply the legacy
    # behaviour. The per-fid incremental path below reproduces only the legacy cleaning
    # (exact-endpoint duplicates, no welding), so any topological diff under non-legacy params,
    # or against a state where welds occurred, falls back to a full rebuild.
    clean_params: dict[str, Any] = state.get("clean_params", dict(LEGACY_CLEAN_PARAMS))
    non_legacy_clean = clean_params != LEGACY_CLEAN_PARAMS or bool(state.get("merges"))

    ns = state.get("ns", None)
    if ns is None:
        ns, _nodes_gdf, state = build_dual(
            current_source_wkts,
            crs=crs,
            boundary=boundary,
            build_nodes_gdf=build_nodes_gdf,
            progress=progress,
            directions=directions,
            **clean_params,
        )
        state["ns"] = ns
        return ns, _nodes_gdf, state

    if (to_remove or to_add) and (
        non_legacy_clean or _requires_full_rebuild(state, current_source_wkts, to_remove, to_add)
    ):
        ns, nodes_gdf, rebuilt_state = build_dual(
            current_source_wkts,
            crs=crs,
            boundary=boundary,
            build_nodes_gdf=build_nodes_gdf,
            progress=progress,
            directions=directions,
            **clean_params,
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
    ep_keys: dict[Any, tuple[tuple[float, float], tuple[float, float]]] = state.get("_ep_keys", {})
    edge_records = state["edge_records"]
    feature_status = dict(state.get("feature_status", {}))

    if not to_remove and not to_add and not boundary_changed:
        nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs, geoms=geoms) if build_nodes_gdf else None
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
        start_key = _ep_key(clean_coords[0])
        end_key = _ep_key(clean_coords[-1])
        ep_keys[fid] = (start_key, end_key)
        for pt in (start_key, end_key):
            endpoint_to_fids[pt].append(fid)

    fid_list.sort()
    ctx = _DualBuildContext(
        ns=ns,
        line_data=line_data,
        directions=directions,
        ep_keys=ep_keys,
        node_idx=node_idx,
        edge_records=edge_records,
        impedances=impedances,
    )
    for fid in [fid for fid in to_add if fid in geoms]:
        start_key, end_key = ep_keys[fid]
        for key in (start_key, end_key):
            for other_fid in endpoint_to_fids.get(key, []):
                if other_fid == fid:
                    continue
                pair = frozenset({fid, other_fid})
                if pair in seen:
                    continue
                seen.add(pair)
                shared_key = str(key)
                edge_counter = _try_add_edge(ctx, fid, other_fid, key, edge_counter, shared_key)
                edge_counter = _try_add_edge(ctx, other_fid, fid, key, edge_counter, shared_key)

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
            "_ep_keys": ep_keys,
            "crs": crs,
            "edge_records": edge_records,
            "ns": ns,
            "directions": directions,
        }
    )
    nodes_gdf = _build_nodes_gdf(ns, fid_list, node_idx, midpoints, crs, geoms=geoms) if build_nodes_gdf else None
    return ns, nodes_gdf, state
