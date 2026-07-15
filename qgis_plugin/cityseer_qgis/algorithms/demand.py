from __future__ import annotations

import math

from qgis.core import (
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)

from .base import CityseerAlgorithmBase, run_with_feedback


class CityseerDemandAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    ORIGINS_LAYER = "ORIGINS_LAYER"
    ORIGIN_WEIGHT_FIELD = "ORIGIN_WEIGHT_FIELD"
    DESTINATIONS_LAYER = "DESTINATIONS_LAYER"
    DESTINATION_WEIGHT_FIELD = "DESTINATION_WEIGHT_FIELD"
    DISTANCES = "DISTANCES"
    DECAY_FN = "DECAY_FN"
    CLOSEST_DESTINATION = "CLOSEST_DESTINATION"
    MAX_SNAP_DIST = "MAX_SNAP_DIST"
    TOLERANCE = "TOLERANCE"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "betweenness_demand"

    def displayName(self) -> str:
        return self.tr("Demand Betweenness (OD Flow)")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute demand-weighted (flow) betweenness from a spatial interaction model. "
            "Trips are allocated from weighted origins (e.g. population) to weighted "
            "destinations (e.g. attractors) with distance decay, then routed along shortest "
            "network paths so intermediate streets accumulate the flow passing through them."
        )

    def createInstance(self):
        return CityseerDemandAlgorithm()

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
                self.ORIGINS_LAYER,
                self.tr("Origins layer (points or polygons; polygon centroids are used)"),
                [
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorPolygon,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ORIGIN_WEIGHT_FIELD,
                self.tr("Origin weight field (e.g. population)"),
                parentLayerParameterName=self.ORIGINS_LAYER,
                type=QgsProcessingParameterField.DataType.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.DESTINATIONS_LAYER,
                self.tr("Destinations layer (points or polygons; polygon centroids are used)"),
                [
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorPolygon,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DESTINATION_WEIGHT_FIELD,
                self.tr("Destination weight field (attractiveness)"),
                parentLayerParameterName=self.DESTINATIONS_LAYER,
                type=QgsProcessingParameterField.DataType.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DISTANCES,
                self.tr("Distance thresholds (comma-separated metres)"),
                defaultValue="800",
            )
        )
        self.add_time_parameters()
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SNAP_DIST,
                self.tr("Max distance to snap origins/destinations to network (metres)"),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=100,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLOSEST_DESTINATION,
                self.tr("Route each origin's full weight to its single nearest destination"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BOUNDARY_LAYER,
                self.tr("Boundary polygon (optional — segments inside are 'live')"),
                [QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        decay_param = QgsProcessingParameterString(
            self.DECAY_FN,
            self.tr(
                "Distance-decay expression using c (metric distance) and p (progress = c / threshold). "
                "Default exp(-4 * p) is scale-free; use e.g. exp(-0.002 * c) for a classic gravity model."
            ),
            defaultValue="exp(-4 * p)",
        )
        decay_param.setFlags(decay_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(decay_param)
        tol_param = QgsProcessingParameterNumber(
            self.TOLERANCE,
            self.tr(
                "Shortest-path tolerance % (0 = exact shortest paths only). "
                "Spreads flow across near-shortest routes. Recommend staying below 2%."
            ),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0,
            optional=False,
            minValue=0.0,
            maxValue=20.0,
        )
        tol_param.setFlags(tol_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(tol_param)
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr("Output layer (street segments with flow values)"),
            )
        )

    def _load_weighted_points(self, parameters, context, crs, feedback, layer_param, field_param, label):
        """Read a weighted point/polygon layer as (coords, weights) lists.

        Polygon features contribute their centroid. Features with empty geometry or
        missing / non-finite / non-positive weights are dropped with logged counts.
        """
        layer = self.parameterAsVectorLayer(parameters, layer_param, context)
        if layer is None:
            raise QgsProcessingException(f"Could not load {label} layer.")
        if layer.crs().isValid() and crs.isValid() and layer.crs() != crs:
            raise QgsProcessingException(
                f"{label.capitalize()} layer CRS does not match input layer CRS. "
                f"Input: {crs.authid()}, {label}: {layer.crs().authid()}. "
                f"Reproject the {label} layer to the same projected CRS as the street layer."
            )
        field_name = self.parameterAsString(parameters, field_param, context)
        if not field_name:
            raise QgsProcessingException(f"A weight field must be selected for the {label} layer.")
        if field_name not in [f.name() for f in layer.fields()]:
            raise QgsProcessingException(f"Field '{field_name}' not found in {label} layer.")

        coords: list[tuple[float, float]] = []
        weights: list[float] = []
        skipped_geom = 0
        skipped_val = 0
        for feat in layer.getFeatures():
            qgeom = feat.geometry()
            if qgeom is None or qgeom.isEmpty():
                skipped_geom += 1
                continue
            val = feat[field_name]
            if val is None:
                skipped_val += 1
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                skipped_val += 1
                continue
            if not math.isfinite(fval) or fval <= 0:
                skipped_val += 1
                continue
            point = qgeom.centroid().asPoint()
            coords.append((point.x(), point.y()))
            weights.append(fval)
        if skipped_geom > 0:
            feedback.pushInfo(f"Skipped {skipped_geom} {label} features with empty geometry.")
        if skipped_val > 0:
            feedback.pushInfo(f"Skipped {skipped_val} {label} features with missing or non-positive weights.")
        if not coords:
            raise QgsProcessingException(f"No valid weighted features found in the {label} layer.")
        feedback.pushInfo(f"{label.capitalize()}: {len(coords)} weighted features ({layer.name()}).")
        return coords, weights

    @staticmethod
    def _snap_to_nodes(ns, coords, weights, max_snap_dist, feedback, label):
        """Snap coordinates to nearest network nodes; returns [(raw_node_idx, weight), ...].

        Uses a grid hash with cell size equal to the max snap distance, so each point only
        compares against nodes in its 3x3 cell neighbourhood (which contains every node
        within range). Pure numpy: no scipy dependency. Positions in node_xys (present
        nodes in iteration order) can diverge from raw graph indices after incremental
        updates, so matches are mapped through node_indices().
        """
        import numpy as np

        node_idxs = np.asarray(ns.node_indices(), dtype=np.int64)
        xys = np.asarray(ns.node_xys, dtype=np.float64)
        cell = float(max_snap_dist)
        cell_x = np.floor(xys[:, 0] / cell).astype(np.int64)
        cell_y = np.floor(xys[:, 1] / cell).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i in range(len(xys)):
            buckets.setdefault((int(cell_x[i]), int(cell_y[i])), []).append(i)

        max_d2 = cell * cell
        pairs: list[tuple[int, float]] = []
        n_far = 0
        for (x, y), w in zip(coords, weights, strict=True):
            cx = math.floor(x / cell)
            cy = math.floor(y / cell)
            candidates: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(buckets.get((cx + dx, cy + dy), ()))
            if not candidates:
                n_far += 1
                continue
            cand_xys = xys[candidates]
            d2 = (cand_xys[:, 0] - x) ** 2 + (cand_xys[:, 1] - y) ** 2
            best = int(np.argmin(d2))
            if d2[best] > max_d2:
                n_far += 1
                continue
            pairs.append((int(node_idxs[candidates[best]]), float(w)))
        if n_far > 0:
            feedback.pushInfo(f"Excluded {n_far} {label} beyond the max snap distance ({max_snap_dist}m).")
        if not pairs:
            raise QgsProcessingException(
                f"No {label} could be snapped to the network. "
                "Check that the layer overlaps the street network and that "
                "the max snap distance is large enough."
            )
        return pairs

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer demand betweenness workflow.")
        self.import_cityseer()
        feedback.setProgressText("Preparing workflow (reading inputs)…")

        # ------------------------------------------------------------------
        # 1. Resolve inputs
        # ------------------------------------------------------------------
        layer, crs = self.resolve_network_layer(parameters, context, feedback)
        boundary_poly = self.load_boundary(parameters, context, crs, feedback)

        distances, speed_m_s = self.resolve_thresholds(parameters, context, feedback)

        max_snap_dist = self.parameterAsInt(parameters, self.MAX_SNAP_DIST, context)
        closest_destination = self.parameterAsBool(parameters, self.CLOSEST_DESTINATION, context)
        decay_fn = self.parameterAsString(parameters, self.DECAY_FN, context).strip()
        if not decay_fn:
            decay_fn = "exp(-4 * p)"
        tolerance = self.parameterAsDouble(parameters, self.TOLERANCE, context)

        origin_coords, origin_weights = self._load_weighted_points(
            parameters, context, crs, feedback, self.ORIGINS_LAYER, self.ORIGIN_WEIGHT_FIELD, "origins"
        )
        dest_coords, dest_weights = self._load_weighted_points(
            parameters, context, crs, feedback, self.DESTINATIONS_LAYER, self.DESTINATION_WEIGHT_FIELD, "destinations"
        )

        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(f"Decay function: {decay_fn}")
        feedback.pushInfo(
            "Allocation: "
            + ("closest destination only" if closest_destination else "across all reachable destinations")
        )
        if tolerance > 0:
            feedback.pushInfo(f"Path tolerance: {tolerance:.1f}%")

        # Progress: 3 steps (build network, compute, write output)
        n_steps = 3
        step_pct = 100.0 / n_steps
        step = 1

        # ------------------------------------------------------------------
        # Step 1: Build dual network
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
        # Step 2: Snap origins/destinations and compute demand betweenness
        # ------------------------------------------------------------------
        origins = self._snap_to_nodes(ns, origin_coords, origin_weights, max_snap_dist, feedback, "origins")
        destinations = self._snap_to_nodes(ns, dest_coords, dest_weights, max_snap_dist, feedback, "destinations")
        feedback.pushInfo(f"Snapped {len(origins)} origins and {len(destinations)} destinations.")

        compute_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Computing demand betweenness…")
        result = run_with_feedback(
            ns,
            lambda: ns.betweenness_demand_shortest(
                origins=origins,
                destinations=destinations,
                decay_fn=decay_fn,
                distances=distances,
                closest_destination=closest_destination,
                speed_m_s=speed_m_s,
                tolerance=tolerance,
            ),
            node_count,
            feedback,
            progress_base=compute_base,
            progress_span=step_pct,
        )
        step += 1

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Step 3: Write output layer
        # ------------------------------------------------------------------
        results: dict[int, dict[str, float]] = {fid: {} for fid in fid_list}
        metrics = result.metrics
        for d in result.distances:
            if "demand" not in metrics or d not in metrics["demand"]:
                continue
            arr = metrics["demand"][d]
            col = f"cc_demand_{d}"
            for i, fid in enumerate(result.node_keys_py):
                if fid in results:
                    val = float(arr[i])
                    results[fid][col] = val if math.isfinite(val) else None

        write_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Writing output layer…")
        feedback.setProgress(int(write_base))
        dest_id = self.write_segments_output(
            parameters,
            context,
            feedback,
            ns,
            fid_list,
            geoms,
            results,
            crs,
            progress_base=write_base,
            progress_span=step_pct,
        )

        feedback.setProgress(100)
        feedback.pushInfo("Done.")

        return {self.OUTPUT: dest_id}
