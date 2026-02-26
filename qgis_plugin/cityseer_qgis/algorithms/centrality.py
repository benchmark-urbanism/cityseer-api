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
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from .base import CityseerAlgorithmBase


def _run_with_feedback(ns, func, total, feedback):
    """
    Run a Rust centrality function in a background thread, polling
    ns.progress to drive the QGIS feedback progress bar from 0–100%.

    Mirrors cityseer's config.wrap_progress but replaces tqdm with
    QgsProcessingFeedback.
    """
    result_queue: Queue = Queue()

    def _worker():
        try:
            result_queue.put(func())
        except Exception as e:
            result_queue.put(e)

    feedback.setProgress(0)
    ns.progress_init()
    thread = threading.Thread(target=_worker)
    thread.start()

    while thread.is_alive():
        time.sleep(0.1)
        if total > 0:
            pct = min(ns.progress() / total, 1.0)
            feedback.setProgress(int(pct * 100))
        if feedback.isCanceled():
            break

    thread.join()
    feedback.setProgress(100)

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
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "network_centrality"

    def displayName(self) -> str:
        return self.tr("Network Centrality")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute localised closeness and betweenness centrality on a street network "
            "using a dual graph representation."
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
                self.tr("Use adaptive sampling (faster, approximate)"),
                defaultValue=False,
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
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr("Output layer (street segments with centrality values)"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network, parse_distances

        self.import_cityseer()

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

        # Optional boundary polygon — nodes inside are "live" (analysis sources);
        # nodes outside provide network context but are not used as sources.
        boundary_layer = self.parameterAsVectorLayer(
            parameters, self.BOUNDARY_LAYER, context
        )
        boundary_poly = None
        if boundary_layer is not None:
            from shapely import wkt as shapely_wkt
            from shapely.ops import unary_union

            polys = []
            for feat in boundary_layer.getFeatures():
                qgeom = feat.geometry()
                if qgeom is not None and not qgeom.isEmpty():
                    polys.append(shapely_wkt.loads(qgeom.asWkt()))
            if polys:
                boundary_poly = unary_union(polys)
                feedback.pushInfo(
                    f"Boundary polygon loaded ({len(polys)} feature(s))."
                )

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

        n_metrics = sum([do_closeness_shortest, do_betweenness_shortest, do_closeness_simplest, do_betweenness_simplest])
        if n_metrics == 0:
            raise QgsProcessingException("Select at least one metric to compute.")

        # Step numbering: 2 (nodes + edges) + n_metrics + 1 (write output)
        n_steps = 2 + n_metrics + 1
        step = 1

        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")

        # ------------------------------------------------------------------
        # Steps 1–2: Build dual NetworkStructure (nodes then edges)
        # ------------------------------------------------------------------
        ns, fid_list, _midpoints, geoms = build_dual_network(
            layer, feedback, step_start=step, n_steps=n_steps,
            boundary=boundary_poly,
        )
        node_count = ns.street_node_count()
        feedback.pushInfo(f"Network built: {node_count} segments.")
        step += 2

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Sampling: split distances into full (exact) and sampled batches.
        # When sample=True, per-distance probability is derived from a
        # canonical grid model (config.compute_distance_p).  Distances where
        # p >= 1.0 run exact; others get source_indices + sample_probability.
        # ------------------------------------------------------------------
        from cityseer import config as cs_config

        full_distances: list[int] = []
        sampled_distances: list[tuple[int, float]] = []  # (distance, p)
        if not do_sample:
            full_distances = sorted(distances)
        else:
            import random as _random

            live_indices = [idx for idx in ns.node_indices() if ns.is_node_live(idx)]
            n_live = len(live_indices)
            for d in sorted(distances):
                p = cs_config.compute_distance_p(d)
                if p >= 1.0:
                    full_distances.append(d)
                else:
                    sampled_distances.append((d, p))
            if sampled_distances:
                feedback.pushInfo(
                    "Sampling: "
                    + ", ".join(f"{d}m @ {p:.0%}" for d, p in sampled_distances)
                )

        def _sample_sources(p):
            """Generate source_indices for a given probability."""
            n_sources = max(1, int(p * n_live))
            sources = _random.sample(live_indices, min(n_sources, n_live))
            actual_p = len(sources) / n_live if n_live > 0 else 1.0
            return sources, actual_p

        # ------------------------------------------------------------------
        # Steps 3+: Run centrality — each metric gets its own 0–100%
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

        if do_closeness_shortest:
            feedback.setProgressText(f"Step {step} of {n_steps}: Computing closeness (shortest path)…")
            if full_distances:
                _fd = full_distances
                r = _run_with_feedback(
                    ns, lambda: ns.closeness_shortest(distances=_fd),
                    node_count, feedback,
                )
                _store(r, "", ["node_density", "node_farness", "node_harmonic", "node_beta", "node_cycles"])
                for d in r.distances:
                    density = r.node_density[d]
                    farness = r.node_farness[d]
                    for i, fid in enumerate(r.node_keys_py):
                        if fid in results and farness[i] > 0:
                            results[fid][f"cc_hillier_{d}"] = float(density[i] ** 2 / farness[i])
            for d, p in sampled_distances:
                sources, actual_p = _sample_sources(p)
                _d, _s, _ap = [d], sources, actual_p
                r = _run_with_feedback(
                    ns, lambda: ns.closeness_shortest(
                        distances=_d, source_indices=_s, sample_probability=_ap,
                    ),
                    node_count, feedback,
                )
                _store(r, "", ["node_density", "node_farness", "node_harmonic", "node_beta", "node_cycles"])
                for dd in r.distances:
                    density = r.node_density[dd]
                    farness = r.node_farness[dd]
                    for i, fid in enumerate(r.node_keys_py):
                        if fid in results and farness[i] > 0:
                            results[fid][f"cc_hillier_{dd}"] = float(density[i] ** 2 / farness[i])
            step += 1

        if feedback.isCanceled():
            return {}

        if do_betweenness_shortest:
            feedback.setProgressText(f"Step {step} of {n_steps}: Computing betweenness (shortest path)…")
            if full_distances:
                _fd = full_distances
                r = _run_with_feedback(
                    ns, lambda: ns.betweenness_shortest(distances=_fd),
                    node_count, feedback,
                )
                _store(r, "", ["node_betweenness", "node_betweenness_beta"])
            for d, p in sampled_distances:
                sources, actual_p = _sample_sources(p)
                _d, _s, _ap = [d], sources, actual_p
                r = _run_with_feedback(
                    ns, lambda: ns.betweenness_shortest(
                        distances=_d, source_indices=_s, sample_probability=_ap,
                    ),
                    node_count, feedback,
                )
                _store(r, "", ["node_betweenness", "node_betweenness_beta"])
            step += 1

        if feedback.isCanceled():
            return {}

        if do_closeness_simplest:
            feedback.setProgressText(f"Step {step} of {n_steps}: Computing closeness (simplest / angular path)…")
            if full_distances:
                _fd = full_distances
                r = _run_with_feedback(
                    ns, lambda: ns.closeness_simplest(distances=_fd),
                    node_count, feedback,
                )
                _store(r, "ang", ["node_density", "node_farness", "node_harmonic"])
            for d, p in sampled_distances:
                sources, actual_p = _sample_sources(p)
                _d, _s, _ap = [d], sources, actual_p
                r = _run_with_feedback(
                    ns, lambda: ns.closeness_simplest(
                        distances=_d, source_indices=_s, sample_probability=_ap,
                    ),
                    node_count, feedback,
                )
                _store(r, "ang", ["node_density", "node_farness", "node_harmonic"])
            step += 1

        if feedback.isCanceled():
            return {}

        if do_betweenness_simplest:
            feedback.setProgressText(f"Step {step} of {n_steps}: Computing betweenness (simplest / angular path)…")
            if full_distances:
                _fd = full_distances
                r = _run_with_feedback(
                    ns, lambda: ns.betweenness_simplest(distances=_fd),
                    node_count, feedback,
                )
                _store(r, "ang", ["node_betweenness", "node_betweenness_beta"])
            for d, p in sampled_distances:
                sources, actual_p = _sample_sources(p)
                _d, _s, _ap = [d], sources, actual_p
                r = _run_with_feedback(
                    ns, lambda: ns.betweenness_simplest(
                        distances=_d, source_indices=_s, sample_probability=_ap,
                    ),
                    node_count, feedback,
                )
                _store(r, "ang", ["node_betweenness", "node_betweenness_beta"])
            step += 1

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
        feedback.setProgressText(f"Step {step} of {n_steps}: Writing output layer…")
        feedback.setProgress(0)
        fields = QgsFields()
        fields.append(QgsField("fid", QVariant.Int))
        for col in all_cols:
            fields.append(QgsField(col, QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, QgsWkbTypes.Type.LineString, crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create output layer.")

        for fid in fid_list:
            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromWkt(geoms[fid].wkt))
            attrs = [fid] + [results[fid].get(col, None) for col in all_cols]
            feat.setAttributes(attrs)
            sink.addFeature(feat)

        feedback.setProgress(100)
        feedback.pushInfo("Done.")

        return {self.OUTPUT: dest_id}
