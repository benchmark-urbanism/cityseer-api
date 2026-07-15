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


class CityseerMixedUsesAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    DATA_LAYER = "DATA_LAYER"
    LANDUSE_FIELD = "LANDUSE_FIELD"
    DISTANCES = "DISTANCES"
    DECAY_FN = "DECAY_FN"
    MAX_ASSIGN_DIST = "MAX_ASSIGN_DIST"
    ANGULAR = "ANGULAR"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    COMPUTE_HILL = "COMPUTE_HILL"
    COMPUTE_SHANNON = "COMPUTE_SHANNON"
    COMPUTE_GINI = "COMPUTE_GINI"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "mixed_uses"

    def displayName(self) -> str:
        return self.tr("Mixed Uses")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute land-use diversity (mixed-use) metrics within network distance thresholds, "
            "aggregated over the street network: Hill diversity (q = 0, 1, 2), Shannon entropy, "
            "and Gini-Simpson."
        )

    def createInstance(self):
        return CityseerMixedUsesAlgorithm()

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
                self.DATA_LAYER,
                self.tr("Data layer (points or polygons with land-use categories)"),
                [
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorPolygon,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.LANDUSE_FIELD,
                self.tr("Land-use category field"),
                parentLayerParameterName=self.DATA_LAYER,
                type=QgsProcessingParameterField.DataType.String,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DISTANCES,
                self.tr("Distance thresholds (comma-separated metres)"),
                defaultValue="400,800",
            )
        )
        self.add_time_parameters()
        decay_param = QgsProcessingParameterString(
            self.DECAY_FN,
            self.tr(
                "Distance-decay weighting for Hill diversity, using c (metric distance) and "
                "p (progress = c / threshold). Default 1 weights all reachable features equally."
            ),
            defaultValue="1",
        )
        decay_param.setFlags(decay_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(decay_param)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ASSIGN_DIST,
                self.tr("Max distance to snap data points to network (metres)"),
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=400,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ANGULAR,
                self.tr("Use simplest path (angular) instead of shortest path"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_HILL,
                self.tr("Hill diversity (q = 0, 1, 2; q0 counts distinct land uses)"),
                defaultValue=True,
            )
        )
        shannon_param = QgsProcessingParameterBoolean(
            self.COMPUTE_SHANNON,
            self.tr("Shannon entropy (prefer Hill q = 1 unless you specifically need entropy)"),
            defaultValue=False,
        )
        shannon_param.setFlags(shannon_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(shannon_param)
        gini_param = QgsProcessingParameterBoolean(
            self.COMPUTE_GINI,
            self.tr("Gini-Simpson diversity (prefer Hill q = 2 unless you specifically need it)"),
            defaultValue=False,
        )
        gini_param.setFlags(gini_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(gini_param)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BOUNDARY_LAYER,
                self.tr("Boundary polygon (optional — nodes inside are 'live')"),
                [QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr("Output layer (street segments with diversity values)"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer mixed-uses workflow.")
        self.import_cityseer()
        feedback.setProgressText("Preparing workflow (reading inputs)…")

        # ------------------------------------------------------------------
        # 1. Resolve inputs
        # ------------------------------------------------------------------
        layer, crs = self.resolve_network_layer(parameters, context, feedback)

        data_layer = self.parameterAsVectorLayer(parameters, self.DATA_LAYER, context)
        if data_layer is None:
            raise QgsProcessingException("Could not load data layer.")
        if data_layer.crs().isValid() and crs.isValid() and data_layer.crs() != crs:
            raise QgsProcessingException(
                "Data layer CRS does not match input layer CRS. "
                f"Input: {crs.authid()}, data: {data_layer.crs().authid()}. "
                "Reproject the data layer to the same projected CRS as the street layer."
            )
        feedback.pushInfo(f"Data layer loaded: {data_layer.name()} ({data_layer.featureCount()} features)")

        landuse_field = self.parameterAsString(parameters, self.LANDUSE_FIELD, context)
        if not landuse_field:
            raise QgsProcessingException("A land-use category field must be selected.")
        if landuse_field not in [f.name() for f in data_layer.fields()]:
            raise QgsProcessingException(f"Field '{landuse_field}' not found in data layer.")
        # Fail before the network build if the category field is empty.
        field_idx = data_layer.fields().indexFromName(landuse_field)
        available = set(str(v) for v in data_layer.uniqueValues(field_idx) if v is not None and str(v).strip())
        if not available:
            raise QgsProcessingException(
                f"Field '{landuse_field}' in layer '{data_layer.name()}' contains no non-empty values, "
                "so there are no land-use categories to compute diversity over. Choose a different field."
            )
        if len(available) == 1:
            feedback.reportError(
                f"Field '{landuse_field}' contains a single category; diversity metrics will be trivial "
                "(Hill q0 = 1, Shannon and Gini = 0 wherever anything is reachable)."
            )

        distances, speed_m_s = self.resolve_thresholds(parameters, context, feedback)
        max_assign_dist = self.parameterAsInt(parameters, self.MAX_ASSIGN_DIST, context)
        angular = self.parameterAsBool(parameters, self.ANGULAR, context)
        decay_fn = self.parameterAsString(parameters, self.DECAY_FN, context).strip() or "1"
        compute_hill = self.parameterAsBool(parameters, self.COMPUTE_HILL, context)
        compute_shannon = self.parameterAsBool(parameters, self.COMPUTE_SHANNON, context)
        compute_gini = self.parameterAsBool(parameters, self.COMPUTE_GINI, context)
        if not (compute_hill or compute_shannon or compute_gini):
            raise QgsProcessingException("Enable at least one diversity measure.")

        boundary_poly = self.load_boundary(parameters, context, crs, feedback)

        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Land-use field: {landuse_field}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(f"Max assignment distance: {max_assign_dist}m")
        feedback.pushInfo(f"Path type: {'simplest (angular)' if angular else 'shortest'}")
        measures = [
            name
            for name, on in [("hill q0/q1/q2", compute_hill), ("shannon", compute_shannon), ("gini", compute_gini)]
            if on
        ]
        feedback.pushInfo("Diversity measures: " + ", ".join(measures))
        if decay_fn != "1":
            feedback.pushInfo(f"Decay function: {decay_fn}")

        # Progress: 4 steps
        n_steps = 4
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
        # Step 2: Build DataMap and assign to network
        # ------------------------------------------------------------------
        from cityseer import rustalgos

        feedback.setProgressText(f"Step {step} of {n_steps}: Building data map…")
        assign_base = (step - 1) * step_pct
        feedback.setProgress(int(assign_base))

        # QGIS divergence: build the DataMap and land-use map directly from QGIS features, in place of
        # cityseer.metrics.layers.compute_mixed_uses (which takes a GeoDataFrame and so needs geopandas).
        # None/blank categories are skipped before insertion, so an uncategorised feature never registers
        # as its own diversity class, matching the library. Keep in sync with layers.compute_mixed_uses.
        data_map = rustalgos.data.DataMap()
        landuses_map: dict = {}
        skipped = 0
        for feat in data_layer.getFeatures():
            qgeom = feat.geometry()
            if qgeom is None or qgeom.isEmpty():
                skipped += 1
                continue
            val = feat[landuse_field]
            if val is None or not str(val).strip():
                skipped += 1
                continue
            fid = feat.id()
            data_map.insert(fid, qgeom.asWkt())
            landuses_map[fid] = str(val)

        if data_map.is_empty():
            raise QgsProcessingException(
                f"No usable features found in data layer '{data_layer.name()}': every feature has an "
                f"empty geometry or an empty '{landuse_field}' value."
            )
        if skipped > 0:
            feedback.pushInfo(f"Skipped {skipped} features with empty geometry or missing category.")
        feedback.pushInfo(f"Data entries: {data_map.count()}")

        feedback.pushInfo("Assigning data points to network…")
        try:
            data_map.assign_data_to_network(ns, max_assign_dist, 50)
        except Exception as exc:
            raise QgsProcessingException(
                f"Failed to assign data points to network: {exc}. "
                "Check that the data layer overlaps the street network and that "
                "the max assignment distance is large enough."
            ) from exc
        feedback.setProgress(int(assign_base + step_pct))
        feedback.pushInfo("Data assigned to network.")
        step += 1

        if feedback.isCanceled():
            return {}

        # ------------------------------------------------------------------
        # Step 3: Compute mixed-use diversity
        # ------------------------------------------------------------------
        compute_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Computing mixed-use diversity…")

        mu_result = run_with_feedback(
            data_map,
            lambda: data_map.mixed_uses(
                network_structure=ns,
                landuses_map=landuses_map,
                distances=distances,
                compute_hill=compute_hill,
                compute_shannon=compute_shannon,
                compute_gini=compute_gini,
                angular=angular,
                speed_m_s=speed_m_s,
                decay_fn=decay_fn,
                pbar_disabled=False,
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
        # Step 4: Write output layer
        # ------------------------------------------------------------------
        results: dict[int, dict[str, float]] = {fid: {} for fid in fid_list}
        ang_suffix = "_ang" if angular else ""

        def _store(metric_by_dist, col_base):
            for dist_key in distances:
                arr = metric_by_dist[dist_key]
                col = f"cc_{col_base}_{dist_key}{ang_suffix}"
                for i, node_key in enumerate(mu_result.node_keys_py):
                    if node_key in results:
                        val = float(arr[i])
                        results[node_key][col] = val if math.isfinite(val) else None

        if compute_hill:
            for q_key in [0, 1, 2]:
                _store(mu_result.hill[q_key], f"hill_q{q_key}")
        if compute_shannon:
            _store(mu_result.shannon, "shannon")
        if compute_gini:
            _store(mu_result.gini, "gini")

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
