from __future__ import annotations

import math
import threading
import time
from queue import Queue

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from .base import CityseerAlgorithmBase


def _run_with_feedback(ns, func, total, feedback, progress_base=0, progress_span=100):
    """
    Run a Rust centrality function in a background thread, polling
    ns.progress to drive the QGIS feedback progress bar.

    progress_base and progress_span map the sub-task's 0–100% onto a
    slice of the overall algorithm progress, e.g. progress_base=40,
    progress_span=20 means this sub-task fills 40%–60%.

    Mirrors cityseer's config.wrap_progress but replaces tqdm with
    QgsProcessingFeedback.
    """
    result_queue: Queue = Queue()

    def _worker():
        try:
            result_queue.put(func())
        except Exception as e:
            result_queue.put(e)

    feedback.setProgress(int(progress_base))
    thread = threading.Thread(target=_worker)
    thread.daemon = True
    thread.start()
    cancelled = False

    while thread.is_alive():
        time.sleep(0.1)
        if total > 0:
            pct = min(ns.progress() / total, 1.0)
            feedback.setProgress(int(progress_base + pct * progress_span))
        if feedback.isCanceled():
            cancelled = True
            break

    thread.join()
    feedback.setProgress(int(progress_base + progress_span))

    if cancelled:
        raise QgsProcessingException("Computation was cancelled.")

    if result_queue.empty():
        raise QgsProcessingException("Computation was cancelled.")

    result = result_queue.get()
    if isinstance(result, Exception):
        raise QgsProcessingException(str(result)) from result
    return result


# Per-category metric parameter definitions.
# Each entry: (metric_suffix, label, default_on)
# The full hidden param name is METRIC_{suffix}_{category_short}.
# Category shorts: CS=closeness-shortest, CA=closeness-simplest,
#                  BS=betweenness-shortest, BA=betweenness-simplest
_CLOSENESS_SHORTEST_METRICS = [
    ("HARMONIC", "Harmonic closeness (shortest)", True),
    ("DENSITY", "Density (shortest)", False),
    ("FARNESS", "Farness (shortest)", False),
    ("BETA", "Beta-weighted closeness (shortest)", False),
    ("CYCLES", "Cycles (shortest)", False),
    ("HILLIER", "Hillier closeness (shortest)", False),
]
_CLOSENESS_SIMPLEST_METRICS = [
    ("HARMONIC", "Harmonic closeness (simplest)", True),
    ("DENSITY", "Density (simplest)", False),
    ("FARNESS", "Farness (simplest)", False),
    ("HILLIER", "Hillier closeness (simplest)", False),
]
_BETWEENNESS_SHORTEST_METRICS = [
    ("BETWEENNESS", "Betweenness (shortest)", True),
    ("BETWEENNESS_BETA", "Beta-weighted betweenness (shortest)", False),
]
_BETWEENNESS_SIMPLEST_METRICS = [
    ("BETWEENNESS", "Betweenness (simplest)", True),
    ("BETWEENNESS_BETA", "Beta-weighted betweenness (simplest)", False),
]


def _param_name(suffix: str, cat_short: str) -> str:
    return f"METRIC_{suffix}_{cat_short}"


class CityseerCentralityAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    DISTANCES = "DISTANCES"
    SAMPLE = "SAMPLE"
    CLOSENESS_SHORTEST = "CLOSENESS_SHORTEST"
    CLOSENESS_SIMPLEST = "CLOSENESS_SIMPLEST"
    BETWEENNESS_SHORTEST = "BETWEENNESS_SHORTEST"
    BETWEENNESS_SIMPLEST = "BETWEENNESS_SIMPLEST"
    TOLERANCE = "TOLERANCE"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "network_centrality"

    def displayName(self) -> str:
        return self.tr("Network Centrality")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute localised closeness and betweenness centrality on a street network "
            "using a dual graph representation. Deterministic distance-based sampling is "
            "enabled by default (exact for smaller distance thresholds)."
        )

    def createInstance(self):
        return CityseerCentralityAlgorithm()

    def createCustomParametersWidget(self, parent=None):
        from .centrality_widget import CentralityDialog

        return CentralityDialog(self, parent=parent)

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                self.tr("Street network line layer"),
                [QgsProcessing.SourceType.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BOUNDARY_LAYER,
                self.tr("Boundary polygon (optional — nodes inside are 'live')"),
                [QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DISTANCES,
                self.tr("Distance thresholds (comma-separated metres)"),
                defaultValue="400,800",
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SAMPLE,
                self.tr("Use deterministic distance-based sampling (faster at larger distances)"),
                defaultValue=True,
            )
        )
        # -- Category toggles (hidden — custom widget handles these) --
        for name, label, default in [
            (self.CLOSENESS_SHORTEST, "Closeness (shortest path)", True),
            (self.CLOSENESS_SIMPLEST, "Closeness (simplest path)", False),
            (self.BETWEENNESS_SHORTEST, "Betweenness (shortest path)", True),
            (self.BETWEENNESS_SIMPLEST, "Betweenness (simplest path)", False),
        ]:
            p = QgsProcessingParameterBoolean(name, self.tr(label), defaultValue=default)
            p.setFlags(p.flags() | QgsProcessingParameterDefinition.Flag.FlagHidden)
            self.addParameter(p)
        # -- Per-category metric toggles (hidden — custom widget handles these) --
        for cat_short, metrics in [
            ("CS", _CLOSENESS_SHORTEST_METRICS),
            ("CA", _CLOSENESS_SIMPLEST_METRICS),
            ("BS", _BETWEENNESS_SHORTEST_METRICS),
            ("BA", _BETWEENNESS_SIMPLEST_METRICS),
        ]:
            for suffix, label, default in metrics:
                pname = _param_name(suffix, cat_short)
                p = QgsProcessingParameterBoolean(pname, self.tr(label), defaultValue=default)
                p.setFlags(p.flags() | QgsProcessingParameterDefinition.Flag.FlagHidden)
                self.addParameter(p)
        tol_param = QgsProcessingParameterNumber(
            self.TOLERANCE,
            self.tr(
                "Shortest-path betweenness tolerance % (0 = exact shortest paths only). "
                "Spreads betweenness across near-shortest routes. Keep below 1% "
                "— higher values increasingly diffuse route concentration, "
                "especially at larger distance thresholds. "
                "Does not apply to simplest (angular) path betweenness."
            ),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0,
            optional=False,
            minValue=0.0,
            maxValue=20.0,
        )
        self.addParameter(tol_param)
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr("Output layer (street segments with centrality values)"),
            )
        )

    def _get_metric(self, parameters, suffix, cat_short, context):
        """Read a per-category metric boolean parameter."""
        return self.parameterAsBool(parameters, _param_name(suffix, cat_short), context)

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network, parse_distances

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer plugin workflow.")
        self.import_cityseer()
        feedback.setProgressText("Preparing workflow (reading inputs)…")

        # ------------------------------------------------------------------
        # 1. Resolve inputs
        # ------------------------------------------------------------------
        layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        if layer is None:
            raise QgsProcessingException("Could not load input layer.")

        if layer.geometryType() != QgsWkbTypes.GeometryType.LineGeometry:
            raise QgsProcessingException("Input layer must be a line (street network) layer.")

        crs = layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                f"Input layer CRS ({crs.authid()}) is geographic (degrees). "
                "Reproject the layer to a projected metre-based CRS before running."
            )
        feedback.pushInfo(f"Input layer loaded: {layer.name()} ({crs.authid()})")

        # Optional boundary polygon
        boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY_LAYER, context)
        boundary_poly = None
        if boundary_layer is not None:
            if boundary_layer.crs().isValid() and crs.isValid() and boundary_layer.crs() != crs:
                raise QgsProcessingException(
                    "Boundary layer CRS does not match input layer CRS. "
                    f"Input: {crs.authid()}, boundary: {boundary_layer.crs().authid()}. "
                    "Reproject the boundary to the same projected CRS as the street layer."
                )
            try:
                from shapely import wkt as shapely_wkt
                from shapely.ops import unary_union

                polys = []
                for feat in boundary_layer.getFeatures():
                    qgeom = feat.geometry()
                    if qgeom is not None and not qgeom.isEmpty():
                        polys.append(shapely_wkt.loads(qgeom.asWkt()))
                if polys:
                    boundary_poly = unary_union(polys)
                    feedback.pushInfo(f"Boundary polygon loaded ({len(polys)} feature(s)).")
                else:
                    feedback.reportError(
                        "Boundary layer has no valid geometries — ignoring boundary. "
                        "All segments will be treated as live sources."
                    )
            except Exception as exc:
                raise QgsProcessingException(f"Failed to parse boundary polygon layer: {exc}") from exc
        else:
            feedback.pushInfo("No boundary polygon provided (all segments are live sources).")

        distances_str = self.parameterAsString(parameters, self.DISTANCES, context)
        try:
            distances = parse_distances(distances_str)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        do_sample = self.parameterAsBool(parameters, self.SAMPLE, context)

        # -- Category toggles --
        closeness_shortest = self.parameterAsBool(parameters, self.CLOSENESS_SHORTEST, context)
        closeness_simplest = self.parameterAsBool(parameters, self.CLOSENESS_SIMPLEST, context)
        betweenness_shortest = self.parameterAsBool(parameters, self.BETWEENNESS_SHORTEST, context)
        betweenness_simplest = self.parameterAsBool(parameters, self.BETWEENNESS_SIMPLEST, context)

        # -- Per-category metric flags --
        # Closeness shortest (CS)
        cs_harmonic = self._get_metric(parameters, "HARMONIC", "CS", context)
        cs_density = self._get_metric(parameters, "DENSITY", "CS", context)
        cs_farness = self._get_metric(parameters, "FARNESS", "CS", context)
        cs_beta = self._get_metric(parameters, "BETA", "CS", context)
        cs_cycles = self._get_metric(parameters, "CYCLES", "CS", context)
        cs_hillier = self._get_metric(parameters, "HILLIER", "CS", context)
        # Closeness simplest (CA)
        ca_harmonic = self._get_metric(parameters, "HARMONIC", "CA", context)
        ca_density = self._get_metric(parameters, "DENSITY", "CA", context)
        ca_farness = self._get_metric(parameters, "FARNESS", "CA", context)
        ca_hillier = self._get_metric(parameters, "HILLIER", "CA", context)
        # Betweenness shortest (BS)
        bs_betweenness = self._get_metric(parameters, "BETWEENNESS", "BS", context)
        bs_betweenness_beta = self._get_metric(parameters, "BETWEENNESS_BETA", "BS", context)
        # Betweenness simplest (BA)
        ba_betweenness = self._get_metric(parameters, "BETWEENNESS", "BA", context)

        tolerance = self.parameterAsDouble(parameters, self.TOLERANCE, context) / 100.0

        # Derive path types from category toggles
        do_shortest = closeness_shortest or betweenness_shortest
        do_simplest = closeness_simplest or betweenness_simplest

        if not do_shortest and not do_simplest:
            raise QgsProcessingException(
                "Enable at least one category (closeness or betweenness for shortest or simplest path)."
            )

        # Determine which combined traversals to run
        n_combined = sum([do_shortest, do_simplest])

        # Step numbering: 1 (network build) + n_combined + 1 (write output)
        n_steps = 1 + n_combined + 1
        step = 1

        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(
            "Sampling mode: " + ("deterministic distance-based (default)" if do_sample else "exact (sampling disabled)")
        )
        # Log selected categories
        categories = []
        if closeness_shortest:
            categories.append("closeness-shortest")
        if closeness_simplest:
            categories.append("closeness-simplest")
        if betweenness_shortest:
            categories.append("betweenness-shortest")
        if betweenness_simplest:
            categories.append("betweenness-simplest")
        feedback.pushInfo("Categories: " + ", ".join(categories))
        if betweenness_shortest and tolerance > 0:
            feedback.pushInfo(f"Betweenness tolerance: {tolerance * 100:.1f}%")

        # Overall progress: divide 0–100% equally among steps.
        step_pct = 100.0 / n_steps

        # ------------------------------------------------------------------
        # Step 1: Build dual NetworkStructure
        # ------------------------------------------------------------------
        ns, fid_list, _midpoints, geoms = build_dual_network(
            layer,
            feedback,
            step=step,
            n_steps=n_steps,
            progress_base=0,
            progress_span=step_pct,
            boundary=boundary_poly,
        )
        node_count = ns.street_node_count()
        if node_count == 0:
            raise QgsProcessingException(
                "No valid street segments found. Check that the input layer contains line features "
                "with valid geometries in a projected CRS."
            )
        feedback.pushInfo(f"Network built: {node_count} segments.")
        step += 1

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Sampling: split distances into full (exact) and sampled batches.
        # ------------------------------------------------------------------
        from cityseer import sampling as cs_sampling

        full_distances: list[int] = []
        sampled_distances: list[tuple[int, float]] = []  # (distance, p)
        if not do_sample:
            full_distances = sorted(distances)
            feedback.pushInfo("Sampling disabled: all thresholds will run exactly.")
        else:
            for d in sorted(distances):
                p = cs_sampling.compute_distance_p(d)
                if p >= 1.0:
                    full_distances.append(d)
                else:
                    sampled_distances.append((d, p))
            if sampled_distances:
                feedback.pushInfo("Sampling: " + ", ".join(f"{d}m @ {p:.0%}" for d, p in sampled_distances))
            if full_distances:
                feedback.pushInfo("Exact distances (p=1): " + ", ".join(f"{d}m" for d in full_distances))

        # ------------------------------------------------------------------
        # Compute centrality metrics
        # ------------------------------------------------------------------
        results: dict[int, dict[str, float]] = {fid: {} for fid in fid_list}

        def _store(result, col_prefix, attr_names):
            """Unpack a Rust result object into the results dict."""
            for d in result.distances:
                for attr in attr_names:
                    arr = getattr(result, attr)[d]
                    base = attr.replace("node_", "")
                    col = f"cc_{base}_{d}_{col_prefix}" if col_prefix else f"cc_{base}_{d}"
                    for i, fid in enumerate(result.node_keys_py):
                        if fid in results:
                            val = float(arr[i])
                            results[fid][col] = val if math.isfinite(val) else None

        def _run_metric_batches(
            label,
            metric_func,
            total_exact,
            attrs,
            col_prefix,
            derive_hillier=False,
            **extra_kwargs,
        ):
            """Run exact + sampled batches for one metric, distributing progress across the step."""
            nonlocal step
            base = (step - 1) * step_pct
            feedback.setProgressText(f"Step {step} of {n_steps}: Computing {label}…")
            # Count sub-batches: 1 for exact (if any) + 1 per sampled distance
            n_batches = (1 if full_distances else 0) + len(sampled_distances)
            if n_batches == 0:
                n_batches = 1
            batch_span = step_pct / n_batches
            batch_idx = 0
            if full_distances:
                _fd = full_distances
                feedback.pushInfo(f"Running {label} exact batch: " + ", ".join(f"{d}m" for d in _fd))
                r = _run_with_feedback(
                    ns,
                    lambda: metric_func(distances=_fd, **extra_kwargs),
                    total_exact,
                    feedback,
                    progress_base=base + batch_idx * batch_span,
                    progress_span=batch_span,
                )
                _store(r, col_prefix, attrs)
                if derive_hillier:
                    for d in r.distances:
                        density = r.node_density[d]
                        farness = r.node_farness[d]
                        for i, fid in enumerate(r.node_keys_py):
                            if fid in results and farness[i] > 0:
                                val = float(density[i] ** 2 / farness[i])
                                hcol = f"cc_hillier_{d}_{col_prefix}" if col_prefix else f"cc_hillier_{d}"
                                results[fid][hcol] = val if math.isfinite(val) else None
                batch_idx += 1
            for d, p in sampled_distances:
                _d, _p = [d], p
                feedback.pushInfo(f"Running {label} sampled {d}m: p={p:.1%}")
                r = _run_with_feedback(
                    ns,
                    lambda _d=_d, _p=_p: metric_func(
                        distances=_d,
                        sample_probability=_p,
                        **extra_kwargs,
                    ),
                    total_exact,
                    feedback,
                    progress_base=base + batch_idx * batch_span,
                    progress_span=batch_span,
                )
                _store(r, col_prefix, attrs)
                if derive_hillier:
                    for dd in r.distances:
                        density = r.node_density[dd]
                        farness = r.node_farness[dd]
                        for i, fid in enumerate(r.node_keys_py):
                            if fid in results and farness[i] > 0:
                                val = float(density[i] ** 2 / farness[i])
                                hcol = f"cc_hillier_{dd}_{col_prefix}" if col_prefix else f"cc_hillier_{dd}"
                                results[fid][hcol] = val if math.isfinite(val) else None
                batch_idx += 1
            step += 1

        if do_shortest:
            # Build attr list for shortest path from per-category flags
            shortest_attrs = []
            if closeness_shortest:
                if cs_harmonic:
                    shortest_attrs.append("node_harmonic")
                if cs_density:
                    shortest_attrs.append("node_density")
                if cs_farness:
                    shortest_attrs.append("node_farness")
                if cs_beta:
                    shortest_attrs.append("node_beta")
                if cs_cycles:
                    shortest_attrs.append("node_cycles")
            if betweenness_shortest:
                if bs_betweenness:
                    shortest_attrs.append("node_betweenness")
                if bs_betweenness_beta:
                    shortest_attrs.append("node_betweenness_beta")
            _run_metric_batches(
                "centrality (shortest path)",
                ns.centrality_shortest,
                node_count,
                shortest_attrs,
                "",
                derive_hillier=cs_hillier and closeness_shortest,
                compute_closeness=closeness_shortest,
                compute_betweenness=betweenness_shortest,
                tolerance=tolerance,
            )

        if feedback.isCanceled():
            return {}

        if do_simplest:
            # Build attr list for simplest path from per-category flags
            simplest_attrs = []
            if closeness_simplest:
                if ca_harmonic:
                    simplest_attrs.append("node_harmonic")
                if ca_density:
                    simplest_attrs.append("node_density")
                if ca_farness:
                    simplest_attrs.append("node_farness")
            if betweenness_simplest and ba_betweenness:
                simplest_attrs.append("node_betweenness")
            if not simplest_attrs:
                feedback.pushInfo("Simplest path: no applicable metrics selected. Skipping.")
                step += 1
            else:
                _run_metric_batches(
                    "centrality (simplest / angular path)",
                    ns.centrality_simplest,
                    node_count,
                    simplest_attrs,
                    "ang",
                    derive_hillier=ca_hillier and closeness_simplest,
                    compute_closeness=closeness_simplest,
                    compute_betweenness=betweenness_simplest,
                )

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Collect all column names in stable order
        # ------------------------------------------------------------------
        all_cols: list[str] = []
        seen_cols: set[str] = set()
        for fid in fid_list:
            for col in results[fid]:
                if col not in seen_cols:
                    all_cols.append(col)
                    seen_cols.add(col)

        # ------------------------------------------------------------------
        # Final step: Write output layer
        # ------------------------------------------------------------------
        write_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Writing output layer…")
        feedback.setProgress(int(write_base))
        fields = QgsFields()
        fields.append(QgsField("fid", QVariant.Int))
        for col in all_cols:
            fields.append(QgsField(col, QVariant.Double, "double", 30, 6))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            QgsWkbTypes.Type.LineString,
            crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create output layer.")

        live_fid_set = set(
            key for key, idx in zip(ns.node_keys_py(), ns.node_indices(), strict=True) if ns.is_node_live(idx)
        )
        live_fids = [fid for fid in fid_list if fid in live_fid_set]
        n_features = len(live_fids)
        if n_features == 0:
            feedback.reportError(
                "No live segments to write. If using a boundary polygon, check that it overlaps the street network."
            )
        for i, fid in enumerate(live_fids):
            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromWkt(geoms[fid].wkt))
            attrs = [fid] + [results[fid].get(col, None) for col in all_cols]
            feat.setAttributes(attrs)
            sink.addFeature(feat)
            if n_features > 0 and (((i + 1) % max(1, n_features // 100)) == 0 or i == n_features - 1):
                pct = (i + 1) / n_features
                feedback.setProgress(int(write_base + pct * step_pct))

        feedback.setProgress(100)
        feedback.pushInfo("Done.")

        return {self.OUTPUT: dest_id}
