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


class CityseerAccessibilityAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    DATA_LAYER = "DATA_LAYER"
    LANDUSE_FIELD = "LANDUSE_FIELD"
    ACCESSIBILITY_KEYS = "ACCESSIBILITY_KEYS"
    DISTANCES = "DISTANCES"
    DECAY_FN = "DECAY_FN"
    MAX_ASSIGN_DIST = "MAX_ASSIGN_DIST"
    ANGULAR = "ANGULAR"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "accessibility"

    def displayName(self) -> str:
        return self.tr("Accessibility")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute land-use accessibility over a street network. Counts the features "
            "reachable within network distance thresholds (walked along the streets, not "
            "straight-line) and reports the network distance to the nearest feature per category."
        )

    def createInstance(self):
        return CityseerAccessibilityAlgorithm()

    def createCustomParametersWidget(self, parent=None):
        from .accessibility_widget import AccessibilityDialog

        return AccessibilityDialog(self, parent=parent)

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
                self.tr("Land-use category field (leave empty to treat all features as one category)"),
                parentLayerParameterName=self.DATA_LAYER,
                type=QgsProcessingParameterField.DataType.String,
                optional=True,
            )
        )
        # Hidden: comma-separated selected categories (managed by custom widget)
        keys_param = QgsProcessingParameterString(
            self.ACCESSIBILITY_KEYS,
            self.tr("Selected land-use categories (comma-separated)"),
            defaultValue="",
            optional=True,
        )
        keys_param.setFlags(keys_param.flags() | QgsProcessingParameterDefinition.Flag.FlagHidden)
        self.addParameter(keys_param)
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
                "Distance-decay weighting using c (metric distance) and p (progress = c / threshold). "
                "Default 1 counts all reachable features equally; use e.g. exp(-4 * p) for decay-weighted counts."
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
                self.tr("Output layer (street segments with accessibility values)"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer accessibility workflow.")
        self.import_cityseer()
        feedback.setProgressText("Preparing workflow (reading inputs)…")

        # ------------------------------------------------------------------
        # 1. Resolve inputs
        # ------------------------------------------------------------------
        layer, crs = self.resolve_network_layer(parameters, context, feedback)

        # Data layer
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

        # Land-use field
        landuse_field = self.parameterAsString(parameters, self.LANDUSE_FIELD, context)
        if landuse_field and landuse_field not in [f.name() for f in data_layer.fields()]:
            raise QgsProcessingException(f"Field '{landuse_field}' not found in data layer.")

        # Distances
        distances, speed_m_s = self.resolve_thresholds(parameters, context, feedback)

        max_assign_dist = self.parameterAsInt(parameters, self.MAX_ASSIGN_DIST, context)
        angular = self.parameterAsBool(parameters, self.ANGULAR, context)
        decay_fn = self.parameterAsString(parameters, self.DECAY_FN, context).strip() or "1"

        # Boundary polygon
        boundary_poly = self.load_boundary(parameters, context, crs, feedback)

        # Validate the category selection against the field BEFORE the network build,
        # so a stale or empty selection fails in seconds rather than after minutes.
        selected_keys_str = self.parameterAsString(parameters, self.ACCESSIBILITY_KEYS, context)
        selected = (
            set(k.strip() for k in selected_keys_str.split(",") if k.strip())
            if selected_keys_str and selected_keys_str.strip()
            else None
        )
        if landuse_field:
            field_idx = data_layer.fields().indexFromName(landuse_field)
            available = sorted(
                set(str(v) for v in data_layer.uniqueValues(field_idx) if v is not None and str(v).strip())
            )
            if not available:
                raise QgsProcessingException(
                    f"Field '{landuse_field}' in layer '{data_layer.name()}' contains no non-empty values, "
                    "so there are no land-use categories to compute. Choose a different field, or clear "
                    "the field selection to treat all features as one category."
                )
            if selected is not None:
                matched = [k for k in available if k in selected]
                missing = sorted(selected - set(available))
                if not matched:
                    shown = ", ".join(available[:20]) + ("…" if len(available) > 20 else "")
                    raise QgsProcessingException(
                        f"None of the selected land-use categories exist in field '{landuse_field}' of "
                        f"layer '{data_layer.name()}'. Selected: {', '.join(sorted(selected))}. "
                        f"Available: {shown}. Click 'Load categories' again after changing the data "
                        "layer or field, then reselect."
                    )
                if missing:
                    feedback.reportError(
                        f"Ignoring selected categories not present in field '{landuse_field}': {', '.join(missing)}."
                    )

        # Log configuration
        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(f"Max assignment distance: {max_assign_dist}m")
        feedback.pushInfo(f"Path type: {'simplest (angular)' if angular else 'shortest'}")
        if decay_fn != "1":
            feedback.pushInfo(f"Decay function: {decay_fn}")
        if landuse_field:
            feedback.pushInfo(f"Land-use field: {landuse_field}")
        else:
            feedback.pushInfo("No land-use field — all features treated as category 'all'.")

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

        data_map = rustalgos.data.DataMap()
        landuses_map: dict = {}
        skipped = 0

        for feat in data_layer.getFeatures():
            qgeom = feat.geometry()
            if qgeom is None or qgeom.isEmpty():
                skipped += 1
                continue
            fid = feat.id()
            data_map.insert(fid, qgeom.asWkt())
            if landuse_field:
                val = feat[landuse_field]
                landuses_map[fid] = str(val) if val is not None else ""
            else:
                landuses_map[fid] = "all"

        if data_map.is_empty():
            raise QgsProcessingException(
                f"No usable features found in data layer '{data_layer.name()}': all "
                f"{data_layer.featureCount()} features have empty geometries."
            )

        if skipped > 0:
            feedback.pushInfo(f"Skipped {skipped} features with empty geometry.")

        # Derive accessibility keys from unique values, filtered by user selection
        all_categories = sorted(set(v for v in landuses_map.values() if v))
        if not all_categories:
            raise QgsProcessingException(
                f"No land-use categories remain after reading layer '{data_layer.name()}': every feature "
                f"has an empty geometry or an empty '{landuse_field}' value."
            )
        accessibility_keys = [k for k in all_categories if k in selected] if selected is not None else all_categories
        if not accessibility_keys:
            shown = ", ".join(all_categories[:20]) + ("…" if len(all_categories) > 20 else "")
            raise QgsProcessingException(
                "None of the selected land-use categories match the categories read from "
                f"layer '{data_layer.name()}'. Selected: {', '.join(sorted(selected or set()))}. "
                f"Available: {shown}. Click 'Load categories' in the dialog and reselect."
            )
        feedback.pushInfo(f"Land-use categories ({len(accessibility_keys)}): {', '.join(accessibility_keys)}")
        feedback.pushInfo(f"Data entries: {data_map.count()}")

        # Assign data to network
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
        # Step 3: Compute accessibility
        # ------------------------------------------------------------------
        compute_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Computing accessibility…")

        acc_result = run_with_feedback(
            data_map,
            lambda: data_map.accessibility(
                network_structure=ns,
                landuses_map=landuses_map,
                accessibility_keys=accessibility_keys,
                distances=distances,
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
        write_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Writing output layer…")
        feedback.setProgress(int(write_base))

        # Build results dict: fid -> {col: value}
        results: dict[int, dict[str, float]] = {fid: {} for fid in fid_list}
        max_dist = max(distances)

        for acc_key in accessibility_keys:
            if acc_key not in acc_result.result:
                feedback.reportError(f"Category '{acc_key}' not found in results — skipping.")
                continue
            lu_access = acc_result.result[acc_key]
            for dist_key in distances:
                col = f"cc_{acc_key}_{dist_key}"
                if angular:
                    col += "_ang"
                count_arr = lu_access.count[dist_key]
                for i, node_key in enumerate(acc_result.node_keys_py):
                    if node_key in results:
                        val = float(count_arr[i])
                        results[node_key][col] = val if math.isfinite(val) else None
                # Nearest distance (only for max distance)
                if dist_key == max_dist:
                    col_dist = f"cc_{acc_key}_nearest_max_{dist_key}"
                    if angular:
                        col_dist = f"cc_{acc_key}_nearest_max_{dist_key}_ang"
                    dist_arr = lu_access.distance[dist_key]
                    for i, node_key in enumerate(acc_result.node_keys_py):
                        if node_key in results:
                            val = float(dist_arr[i])
                            results[node_key][col_dist] = val if math.isfinite(val) else None

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
