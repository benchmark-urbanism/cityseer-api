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
    FLOW_EXPRS = "FLOW_EXPRS"
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
        flow_param = QgsProcessingParameterString(
            self.FLOW_EXPRS,
            self.tr(
                "Flow-weighting expressions as semicolon-separated 'name: expression' pairs, "
                "e.g. 'demand: 1; demand_decay: exp(-4 * p)'. Each expression weights the allocated "
                "flow by trip distance (c = metres, p = c / threshold) and emits its own output "
                "column, all from a single traversal. Leave empty for the paired default: a "
                "conserved 'demand' column and a distance-attenuated 'demand_decay' column. Note "
                "the distinct roles: the decay expression above shapes destination choice; these "
                "expressions scale the flow itself (e.g. trip frequency falling with distance)."
            ),
            defaultValue="",
            optional=True,
        )
        flow_param.setFlags(flow_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(flow_param)
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
    def _build_demand_data_map(ns, coords, weights, max_snap_dist, feedback, label):
        """Assign weighted points to the network via the shared DataMap workflow.

        Uses the same edge-based assignment as the library's data layers and
        ``networks.betweenness_demand`` (``DataMap.assign_data_to_network``: both endpoints of
        the nearest barrier-valid edge, with straight-line offsets carried into all routed
        distances by the Rust core). Returns ``(data_map, weights_map)`` with entries keyed
        positionally. Keep in sync with ``networks._demand_data_map``.
        """
        from cityseer import rustalgos

        data_map = rustalgos.data.DataMap()
        for i, (x, y) in enumerate(coords):
            data_map.insert(i, f"POINT ({x} {y})")
        # n_nearest_candidates matches the library default (layers.build_data_map)
        data_map.assign_data_to_network(ns, float(max_snap_dist), 50)
        weights_map = {i: float(w) for i, w in enumerate(weights)}
        assigned_keys = {assignment[0] for pairs in data_map.node_data_map.values() for assignment in pairs}
        n_far = len(coords) - len(assigned_keys)
        if n_far > 0:
            feedback.pushInfo(f"Excluded {n_far} {label} with no valid network assignment within {max_snap_dist}m.")
        if not assigned_keys:
            raise QgsProcessingException(
                f"No {label} could be assigned to the network. "
                "Check that the layer overlaps the street network and that "
                "the max snap distance is large enough."
            )
        feedback.pushInfo(f"Assigned {len(assigned_keys)} {label} to the network.")
        return data_map, weights_map

    @staticmethod
    def _parse_flow_exprs(raw: str) -> list[tuple[str, str]] | None:
        """Parse 'name: expr; name: expr' into [(name, expr), ...]; empty input returns None."""
        raw = (raw or "").strip()
        if not raw:
            return None
        exprs: list[tuple[str, str]] = []
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise QgsProcessingException(f"Flow expression '{part}' must use the form 'name: expression'.")
            name, expr = part.split(":", 1)
            name = name.strip()
            expr = expr.strip()
            if not name.replace("_", "").isalnum():
                raise QgsProcessingException(f"Flow expression name '{name}' must be alphanumeric/underscore.")
            if not expr:
                raise QgsProcessingException(f"Flow expression '{name}' has an empty expression.")
            exprs.append((name, expr))
        if not exprs:
            return None
        return exprs

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
        flow_exprs = self._parse_flow_exprs(self.parameterAsString(parameters, self.FLOW_EXPRS, context))
        if flow_exprs is None:
            # paired default matching the library (networks.betweenness_demand): a conserved flow
            # channel and a distance-attenuated one, from a single traversal
            flow_exprs = [("demand", "1"), ("demand_decay", "exp(-4 * p)")]
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
        if flow_exprs:
            feedback.pushInfo("Flow expressions: " + "; ".join(f"{n}: {e}" for n, e in flow_exprs))
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
        # Step 2: Assign origins/destinations and compute demand betweenness.
        # QGIS divergence: build the DataMaps and call NetworkStructure.betweenness_demand_shortest
        # on the Rust core directly, in place of cityseer.metrics.networks.betweenness_demand (which
        # takes GeoDataFrames and so needs geopandas). The assignment mechanism is identical
        # (DataMap.assign_data_to_network). Keep in sync with networks.betweenness_demand.
        # ------------------------------------------------------------------
        origins_map, origin_weights_map = self._build_demand_data_map(
            ns, origin_coords, origin_weights, max_snap_dist, feedback, "origins"
        )
        destinations_map, destination_weights_map = self._build_demand_data_map(
            ns, dest_coords, dest_weights, max_snap_dist, feedback, "destinations"
        )

        compute_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Computing demand betweenness…")
        result = run_with_feedback(
            ns,
            lambda: ns.betweenness_demand_shortest(
                origins=origins_map,
                origin_weights_map=origin_weights_map,
                destinations=destinations_map,
                destination_weights_map=destination_weights_map,
                decay_fn=decay_fn,
                distances=distances,
                betweenness_exprs=flow_exprs,
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
        # Step 3: Write output layer (one column per flow expression per distance)
        # ------------------------------------------------------------------
        metric_names = [name for name, _expr in flow_exprs] if flow_exprs else ["demand"]
        results: dict[int, dict[str, float]] = {fid: {} for fid in fid_list}
        metrics = result.metrics
        for metric_name in metric_names:
            for d in result.distances:
                if metric_name not in metrics or d not in metrics[metric_name]:
                    continue
                arr = metrics[metric_name][d]
                col = f"cc_{metric_name}_{d}"
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
