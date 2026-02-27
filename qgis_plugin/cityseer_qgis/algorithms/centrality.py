from __future__ import annotations

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


class CityseerCentralityAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    DISTANCES = "DISTANCES"
    SAMPLE = "SAMPLE"
    CLOSENESS_SHORTEST = "CLOSENESS_SHORTEST"
    BETWEENNESS_SHORTEST = "BETWEENNESS_SHORTEST"
    CLOSENESS_SIMPLEST = "CLOSENESS_SIMPLEST"
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
                self.tr("Use deterministic distance-based sampling (default; faster at larger distances)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLOSENESS_SHORTEST,
                self.tr("Closeness (shortest path)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BETWEENNESS_SHORTEST,
                self.tr("Betweenness (shortest path)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLOSENESS_SIMPLEST,
                self.tr("Closeness (simplest / angular path)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BETWEENNESS_SIMPLEST,
                self.tr("Betweenness (simplest / angular path)"),
                defaultValue=False,
            )
        )
        tol_param = QgsProcessingParameterNumber(
            self.TOLERANCE,
            self.tr(
                "Betweenness tolerance % (0 = exact shortest paths only; "
                "e.g. 1% spreads betweenness across routes within 1% of the shortest)"
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

        # Optional boundary polygon — nodes inside are "live" (analysis sources);
        # nodes outside provide network context but are not used as sources.
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

        do_closeness_shortest = self.parameterAsBool(parameters, self.CLOSENESS_SHORTEST, context)
        do_betweenness_shortest = self.parameterAsBool(parameters, self.BETWEENNESS_SHORTEST, context)
        do_closeness_simplest = self.parameterAsBool(parameters, self.CLOSENESS_SIMPLEST, context)
        do_betweenness_simplest = self.parameterAsBool(parameters, self.BETWEENNESS_SIMPLEST, context)
        tolerance = self.parameterAsDouble(parameters, self.TOLERANCE, context) / 100.0

        # Group into combined calls: one for shortest, one for simplest
        do_shortest = do_closeness_shortest or do_betweenness_shortest
        do_simplest = do_closeness_simplest or do_betweenness_simplest
        n_combined = sum([do_shortest, do_simplest])
        if n_combined == 0:
            raise QgsProcessingException("Select at least one metric to compute.")

        # Step numbering: 1 (network build) + n_combined + 1 (write output)
        n_steps = 1 + n_combined + 1
        step = 1

        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(
            "Sampling mode: " + ("deterministic distance-based (default)" if do_sample else "exact (sampling disabled)")
        )
        selected = []
        if do_closeness_shortest:
            selected.append("closeness-shortest")
        if do_betweenness_shortest:
            selected.append("betweenness-shortest")
        if do_closeness_simplest:
            selected.append("closeness-simplest")
        if do_betweenness_simplest:
            selected.append("betweenness-simplest")
        feedback.pushInfo("Metrics selected: " + ", ".join(selected))
        if (do_betweenness_shortest or do_betweenness_simplest) and tolerance > 0:
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
        feedback.pushInfo(f"Network built: {node_count} segments.")
        step += 1

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Sampling: split distances into full (exact) and sampled batches.
        # When sample=True, per-distance probability is derived from a
        # canonical grid model (config.compute_distance_p).  Distances where
        # p >= 1.0 run exact; others get source_indices + sample_probability.
        # ------------------------------------------------------------------
        from cityseer import config as cs_config

        live_indices = [idx for idx in ns.node_indices() if ns.is_node_live(idx)]
        n_live = len(live_indices)

        full_distances: list[int] = []
        sampled_distances: list[tuple[int, float]] = []  # (distance, p)
        if not do_sample:
            full_distances = sorted(distances)
            feedback.pushInfo("Sampling disabled: all thresholds will run exactly.")
        else:
            import random as _random

            feedback.pushInfo(f"Live segments available as sources: {n_live}")
            for d in sorted(distances):
                p = cs_config.compute_distance_p(d)
                if p >= 1.0:
                    full_distances.append(d)
                else:
                    sampled_distances.append((d, p))
            if sampled_distances:
                feedback.pushInfo("Sampling: " + ", ".join(f"{d}m @ {p:.0%}" for d, p in sampled_distances))
            if sampled_distances and n_live == 0:
                raise QgsProcessingException(
                    "No live segments are available for sampling. Adjust or remove the boundary polygon."
                )
            if full_distances:
                feedback.pushInfo("Exact distances (p=1): " + ", ".join(f"{d}m" for d in full_distances))

        def _sample_sources(p):
            """Bernoulli-sample source_indices at probability p (with 1-source fallback)."""
            if n_live == 0:
                return []
            sources = [idx for idx in live_indices if _random.random() < p]
            if not sources:
                sources = [_random.choice(live_indices)]
            return sources

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
                            results[fid][col] = float(arr[i])

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
                                results[fid][f"cc_hillier_{d}"] = float(density[i] ** 2 / farness[i])
                batch_idx += 1
            for d, p in sampled_distances:
                sources = _sample_sources(p)
                _d, _s, _p = [d], sources, p
                feedback.pushInfo(f"Running {label} sampled {d}m: p={p:.1%}, sources={len(sources)}/{n_live}")
                r = _run_with_feedback(
                    ns,
                    lambda: metric_func(
                        distances=_d,
                        source_indices=_s,
                        sample_probability=_p,
                        **extra_kwargs,
                    ),
                    len(sources),
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
                                results[fid][f"cc_hillier_{dd}"] = float(density[i] ** 2 / farness[i])
                batch_idx += 1
            step += 1

        if do_shortest:
            # Combine closeness + betweenness into a single Dijkstra traversal
            shortest_attrs = []
            if do_closeness_shortest:
                shortest_attrs.extend(["node_density", "node_farness", "node_harmonic", "node_beta", "node_cycles"])
            if do_betweenness_shortest:
                shortest_attrs.extend(["node_betweenness", "node_betweenness_beta"])
            _run_metric_batches(
                "centrality (shortest path)",
                ns.centrality_shortest,
                node_count,
                shortest_attrs,
                "",
                derive_hillier=do_closeness_shortest,
                compute_closeness=do_closeness_shortest,
                compute_betweenness=do_betweenness_shortest,
                tolerance=tolerance,
            )

        if feedback.isCanceled():
            return {}

        if do_simplest:
            # Combine closeness + betweenness into a single Dijkstra traversal
            simplest_attrs = []
            if do_closeness_simplest:
                simplest_attrs.extend(["node_density", "node_farness", "node_harmonic"])
            if do_betweenness_simplest:
                simplest_attrs.extend(["node_betweenness", "node_betweenness_beta"])
            _run_metric_batches(
                "centrality (simplest / angular path)",
                ns.centrality_simplest,
                node_count,
                simplest_attrs,
                "ang",
                compute_closeness=do_closeness_simplest,
                compute_betweenness=do_betweenness_simplest,
                tolerance=tolerance,
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
            fields.append(QgsField(col, QVariant.Double))

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
