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
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from .base import CityseerAlgorithmBase


def _run_with_feedback(progress_src, func, total, feedback, progress_base=0, progress_span=100):
    """
    Run a Rust function in a background thread, polling
    progress_src.progress to drive the QGIS feedback progress bar.
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
            pct = min(progress_src.progress() / total, 1.0)
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


class CityseerAccessibilityAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    DATA_LAYER = "DATA_LAYER"
    LANDUSE_FIELD = "LANDUSE_FIELD"
    ACCESSIBILITY_KEYS = "ACCESSIBILITY_KEYS"
    DISTANCES = "DISTANCES"
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
            "Compute land-use accessibility on a street network. "
            "Counts reachable features (weighted and unweighted) within distance thresholds."
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
        keys_param.setFlags(
            keys_param.flags() | QgsProcessingParameterDefinition.Flag.FlagHidden
        )
        self.addParameter(keys_param)
        self.addParameter(
            QgsProcessingParameterString(
                self.DISTANCES,
                self.tr("Distance thresholds (comma-separated metres)"),
                defaultValue="400,800",
            )
        )
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
        from ..utils.converters import build_dual_network, parse_distances

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer accessibility workflow.")
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
        distances_str = self.parameterAsString(parameters, self.DISTANCES, context)
        try:
            distances = parse_distances(distances_str)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        max_assign_dist = self.parameterAsInt(parameters, self.MAX_ASSIGN_DIST, context)
        angular = self.parameterAsBool(parameters, self.ANGULAR, context)

        # Boundary polygon
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

        # Log configuration
        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(f"Max assignment distance: {max_assign_dist}m")
        feedback.pushInfo(f"Path type: {'simplest (angular)' if angular else 'shortest'}")
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
            raise QgsProcessingException("No valid features found in data layer.")

        if skipped > 0:
            feedback.pushInfo(f"Skipped {skipped} features with empty geometry.")

        # Derive accessibility keys from unique values, filtered by user selection
        all_categories = sorted(set(v for v in landuses_map.values() if v))
        if not all_categories:
            raise QgsProcessingException("No valid land-use categories found.")
        selected_keys_str = self.parameterAsString(parameters, self.ACCESSIBILITY_KEYS, context)
        if selected_keys_str and selected_keys_str.strip():
            selected = set(k.strip() for k in selected_keys_str.split(",") if k.strip())
            accessibility_keys = [k for k in all_categories if k in selected]
        else:
            accessibility_keys = all_categories
        if not accessibility_keys:
            raise QgsProcessingException("No land-use categories selected.")
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

        acc_result = _run_with_feedback(
            data_map,
            lambda: data_map.accessibility(
                network_structure=ns,
                landuses_map=landuses_map,
                accessibility_keys=accessibility_keys,
                distances=distances,
                angular=angular,
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
                # Unweighted
                col_nw = f"cc_{acc_key}_{dist_key}"
                if angular:
                    col_nw += "_ang"
                col_nw += "_nw"
                nw_arr = lu_access.unweighted[dist_key]
                # Weighted
                col_wt = f"cc_{acc_key}_{dist_key}"
                if angular:
                    col_wt += "_ang"
                col_wt += "_wt"
                wt_arr = lu_access.weighted[dist_key]
                for i, node_key in enumerate(acc_result.node_keys_py):
                    if node_key in results:
                        val_nw = float(nw_arr[i])
                        results[node_key][col_nw] = val_nw if math.isfinite(val_nw) else None
                        val_wt = float(wt_arr[i])
                        results[node_key][col_wt] = val_wt if math.isfinite(val_wt) else None
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

        # Collect column names in stable order
        all_cols: list[str] = []
        seen_cols: set[str] = set()
        for fid in fid_list:
            for col in results[fid]:
                if col not in seen_cols:
                    all_cols.append(col)
                    seen_cols.add(col)

        # Create output fields
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

        # Only write live nodes
        live_fid_set = set(
            key for key, idx in zip(ns.node_keys_py(), ns.node_indices(), strict=True) if ns.is_node_live(idx)
        )
        live_fids = [fid for fid in fid_list if fid in live_fid_set]
        n_features = len(live_fids)
        if n_features == 0:
            feedback.reportError(
                "No live segments to write. If using a boundary polygon, check that it "
                "overlaps the street network."
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
