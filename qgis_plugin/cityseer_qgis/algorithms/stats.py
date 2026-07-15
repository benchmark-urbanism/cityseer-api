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

# Core statistics (decay-weighted via decay_fn parameter)
_PAIRED_STATS = [
    ("sum", "STAT_SUM"),
    ("mean", "STAT_MEAN"),
    ("median", "STAT_MEDIAN"),
    ("count", "STAT_COUNT"),
    ("variance", "STAT_VARIANCE"),
    ("mad", "STAT_MAD"),
]
# Extrema statistics
_UNPAIRED_STATS = [
    ("max", "STAT_MAX"),
    ("min", "STAT_MIN"),
]


class CityseerStatsAlgorithm(CityseerAlgorithmBase):
    INPUT_LAYER = "INPUT_LAYER"
    DATA_LAYER = "DATA_LAYER"
    NUMERICAL_FIELD = "NUMERICAL_FIELD"
    DISTANCES = "DISTANCES"
    MAX_ASSIGN_DIST = "MAX_ASSIGN_DIST"
    ANGULAR = "ANGULAR"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    STAT_SUM = "STAT_SUM"
    STAT_MEAN = "STAT_MEAN"
    STAT_MEDIAN = "STAT_MEDIAN"
    STAT_COUNT = "STAT_COUNT"
    STAT_VARIANCE = "STAT_VARIANCE"
    STAT_MAD = "STAT_MAD"
    STAT_MAX = "STAT_MAX"
    STAT_MIN = "STAT_MIN"
    DECAY_FN = "DECAY_FN"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "statistics"

    def displayName(self) -> str:
        return self.tr("Statistics")

    def shortDescription(self) -> str:
        return self.tr(
            "Compute localised statistics (sum, mean, count, etc.) for numerical data columns "
            "within network distance thresholds, aggregated over the street network rather "
            "than straight-line buffers."
        )

    def createInstance(self):
        return CityseerStatsAlgorithm()

    def createCustomParametersWidget(self, parent=None):
        from .stats_widget import StatsDialog

        return StatsDialog(self, parent=parent)

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
                self.tr("Data layer (points or polygons with numerical values)"),
                [
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorPolygon,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.NUMERICAL_FIELD,
                self.tr("Numerical field(s) to compute statistics on"),
                parentLayerParameterName=self.DATA_LAYER,
                type=QgsProcessingParameterField.DataType.Numeric,
                allowMultiple=True,
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
                "Distance-decay weighting using c (metric distance) and p (progress = c / threshold). "
                "Default 1 weights all contributions equally; use e.g. exp(-4 * p) for decay-weighted statistics."
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
        # Hidden stat toggles (managed by custom widget)
        for param_name, label, default in [
            (self.STAT_SUM, "Sum", True),
            (self.STAT_MEAN, "Mean", True),
            (self.STAT_MEDIAN, "Median", False),
            (self.STAT_COUNT, "Count", True),
            (self.STAT_VARIANCE, "Variance", False),
            (self.STAT_MAD, "Median Absolute Deviation (MAD)", False),
            (self.STAT_MAX, "Maximum", False),
            (self.STAT_MIN, "Minimum", False),
        ]:
            p = QgsProcessingParameterBoolean(param_name, self.tr(label), defaultValue=default)
            p.setFlags(p.flags() | QgsProcessingParameterDefinition.Flag.FlagHidden)
            self.addParameter(p)
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr("Output layer (street segments with statistics values)"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..utils.converters import build_dual_network

        feedback.setProgressText("Preparing workflow (loading dependencies)…")
        feedback.setProgress(0)
        feedback.pushInfo("Initialising cityseer statistics workflow.")
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

        # Numerical fields
        num_fields = self.parameterAsFields(parameters, self.NUMERICAL_FIELD, context)
        if not num_fields:
            raise QgsProcessingException("At least one numerical field must be selected.")
        layer_fields = [f.name() for f in data_layer.fields()]
        for num_field in num_fields:
            if num_field not in layer_fields:
                raise QgsProcessingException(f"Field '{num_field}' not found in data layer.")

        # Distances
        distances, speed_m_s = self.resolve_thresholds(parameters, context, feedback)

        max_assign_dist = self.parameterAsInt(parameters, self.MAX_ASSIGN_DIST, context)
        angular = self.parameterAsBool(parameters, self.ANGULAR, context)
        decay_fn = self.parameterAsString(parameters, self.DECAY_FN, context).strip() or "1"

        # Resolve enabled stats
        enabled_paired = [
            stat_name
            for stat_name, param_name in _PAIRED_STATS
            if self.parameterAsBool(parameters, param_name, context)
        ]
        enabled_unpaired = [
            stat_name
            for stat_name, param_name in _UNPAIRED_STATS
            if self.parameterAsBool(parameters, param_name, context)
        ]
        if not enabled_paired and not enabled_unpaired:
            raise QgsProcessingException("At least one statistic must be selected.")

        # Boundary polygon
        boundary_poly = self.load_boundary(parameters, context, crs, feedback)

        # Log configuration
        feedback.pushInfo(f"CRS: {crs.authid()}")
        feedback.pushInfo(f"Numerical fields: {', '.join(num_fields)}")
        feedback.pushInfo(f"Distances: {distances}")
        feedback.pushInfo(f"Max assignment distance: {max_assign_dist}m")
        feedback.pushInfo(f"Path type: {'simplest (angular)' if angular else 'shortest'}")
        all_enabled = enabled_paired + enabled_unpaired
        feedback.pushInfo(f"Statistics: {', '.join(all_enabled)}")
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

        # QGIS divergence: build the DataMap and numerical maps directly from QGIS features, in place of
        # cityseer.metrics.layers.compute_stats (which takes a GeoDataFrame and so needs geopandas).
        # Every inserted entry must appear in every numerical map (a Rust invariant); missing or invalid
        # values are stored as NaN, which the Rust aggregation skips. Keep in sync with layers.compute_stats.
        data_map = rustalgos.data.DataMap()
        numerical_maps: list[dict] = [{} for _ in num_fields]
        skipped_geom = 0
        n_missing = [0] * len(num_fields)

        for feat in data_layer.getFeatures():
            qgeom = feat.geometry()
            if qgeom is None or qgeom.isEmpty():
                skipped_geom += 1
                continue
            fid = feat.id()
            data_map.insert(fid, qgeom.asWkt())
            for j, num_field in enumerate(num_fields):
                val = feat[num_field]
                try:
                    fval = float(val) if val is not None else float("nan")
                except (TypeError, ValueError):
                    fval = float("nan")
                if not math.isfinite(fval):
                    fval = float("nan")
                    n_missing[j] += 1
                numerical_maps[j][fid] = fval

        if data_map.is_empty():
            raise QgsProcessingException(
                f"No usable features found in data layer '{data_layer.name()}': all "
                f"{data_layer.featureCount()} features have empty geometries."
            )

        if skipped_geom > 0:
            feedback.pushInfo(f"Skipped {skipped_geom} features with empty geometry.")
        for num_field, n_bad in zip(num_fields, n_missing, strict=True):
            if n_bad > 0:
                feedback.pushInfo(f"Field '{num_field}': {n_bad} features with missing or non-finite values.")
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
        # Step 3: Compute statistics
        # ------------------------------------------------------------------
        compute_base = (step - 1) * step_pct
        feedback.setProgressText(f"Step {step} of {n_steps}: Computing statistics…")

        # Rust measure keys: "variance" is exposed as attribute `variance` but selected as "var"
        measures = [{"variance": "var"}.get(s, s) for s in enabled_paired + enabled_unpaired]
        stats_result = run_with_feedback(
            data_map,
            lambda: data_map.stats(
                network_structure=ns,
                numerical_maps=numerical_maps,
                distances=distances,
                angular=angular,
                speed_m_s=speed_m_s,
                decay_fn=decay_fn,
                measures=measures,
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
        ang_suffix = "_ang" if angular else ""

        for j, num_field in enumerate(num_fields):
            stats_obj = stats_result.result[j]
            for stat_name in enabled_paired + enabled_unpaired:
                attr = getattr(stats_obj, stat_name)
                for dist_key in distances:
                    col = f"cc_{num_field}_{stat_name}_{dist_key}{ang_suffix}"
                    arr = attr[dist_key]
                    for i, node_key in enumerate(stats_result.node_keys_py):
                        if node_key in results:
                            val = float(arr[i])
                            results[node_key][col] = val if math.isfinite(val) else None

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
