"""
Utilities for converting QGIS line layers into a cityseer dual NetworkStructure.

The QGIS plugin keeps the feature-reading and cache-key logic here, but delegates
the actual dual build and incremental diff logic to ``cityseer.tools.dual``.
"""

from __future__ import annotations

import math

from cityseer.tools import dual

_inc_state: dict | None = None


def parse_distances(s: str) -> list[int]:
    """Parse a comma-separated string of distances into a list of ints."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("No distances provided.")
    result = []
    seen: set[int] = set()
    for p in parts:
        try:
            distance = int(p)
        except ValueError as err:
            raise ValueError(f"Invalid distance value: {p!r}. Expected integers.") from err
        if distance <= 0:
            raise ValueError(f"Invalid distance value: {p!r}. Distances must be positive integers.")
        if distance in seen:
            raise ValueError(f"Duplicate distance value: {distance}. Distances must be unique.")
        seen.add(distance)
        result.append(distance)
    return result


def _read_layer_wkts(layer, feedback=None, progress_base=0, progress_span=100) -> dict[int, str]:
    current_wkts: dict[int, str] = {}
    feat_count = layer.featureCount() if hasattr(layer, "featureCount") else -1
    for i, feat in enumerate(layer.getFeatures()):
        qgeom = feat.geometry()
        if qgeom is None or qgeom.isEmpty():
            if feedback and feat_count and feat_count > 0:
                pct = (i + 1) / feat_count
                feedback.setProgress(int(progress_base + pct * progress_span))
            continue
        current_wkts[feat.id()] = qgeom.asWkt()
        if feedback and feat_count and feat_count > 0:
            pct = (i + 1) / feat_count
            feedback.setProgress(int(progress_base + pct * progress_span))
    if feedback and feat_count <= 0:
        feedback.setProgress(int(progress_base + progress_span))
    return current_wkts


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

    Returns
    -------
    tuple[NetworkStructure, list[int], dict[int, tuple], dict[int, LineString]]
        ns, fid_list, midpoints, geoms
    """
    global _inc_state

    if feedback:
        feedback.setProgressText(f"Step {step} of {n_steps}: Reading input features…")
        feedback.setProgress(int(progress_base))

    read_span = progress_span * 0.1
    current_wkts = _read_layer_wkts(layer, feedback=feedback, progress_base=progress_base, progress_span=read_span)
    if not current_wkts:
        feat_total = layer.featureCount() if hasattr(layer, "featureCount") else 0
        if feat_total > 0:
            raise ValueError(
                f"Layer reports {feat_total} features but none could be read. "
                "The data source may have moved or become unavailable."
            )
        raise ValueError("Input layer contains no features.")

    layer_cache_key = (layer.id(), layer.crs().authid(), layer.wkbType())
    if _inc_state is not None and _inc_state.get("layer_cache_key") != layer_cache_key and feedback:
        feedback.pushInfo("Input layer changed — rebuilding cached dual network.")

    use_incremental = _inc_state is not None and _inc_state.get("layer_cache_key") == layer_cache_key
    if feedback:
        action = "Updating" if use_incremental else "Building"
        feedback.setProgressText(f"Step {step} of {n_steps}: {action} dual network…")
        feedback.setProgress(int(progress_base + math.ceil(read_span)))

    if use_incremental:
        ns, _nodes_gdf, state = dual.incremental_update(
            _inc_state,
            current_wkts,
            crs=layer.crs().authid(),
            boundary=boundary,
            build_nodes_gdf=False,
            progress=False,
        )
    else:
        ns, _nodes_gdf, state = dual.build_dual(
            current_wkts,
            crs=layer.crs().authid(),
            boundary=boundary,
            build_nodes_gdf=False,
            progress=False,
        )

    state["layer_cache_key"] = layer_cache_key
    _inc_state = state

    if feedback:
        feedback.setProgress(int(progress_base + progress_span))

    return state["ns"], state["fid_list"], state["midpoints"], state["geoms"]
