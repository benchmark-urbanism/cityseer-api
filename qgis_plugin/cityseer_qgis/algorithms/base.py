"""Shared base classes and helpers for the cityseer QGIS processing algorithms.

Dependency-avoidance design
---------------------------
The plugin installs cityseer with ``--no-deps`` (see ``cityseer_qgis/__init__.py``) so QGIS does
not have to pull in the library's heavier dependencies (geopandas, networkx, osmnx, matplotlib,
tqdm). It therefore cannot use the GeoDataFrame-based high-level API — ``cityseer.metrics.layers``,
``cityseer.metrics.networks``, and ``cityseer.tools.io`` all take and return ``GeoDataFrame``
objects and so require geopandas.

Instead the plugin works directly against the lightweight surfaces whose only runtime
dependencies (numpy, shapely, scipy) ship with QGIS:

- ``cityseer.rustalgos`` — the compiled Rust core (``NetworkStructure``, ``DataMap``).
- ``cityseer.tools.dual`` — the dual-graph builder, used by ``utils.converters.build_dual_network``
  in place of ``cityseer.tools.io.network_structure_from_nx`` (which needs networkx + geopandas).
- ``cityseer.sampling`` — the adaptive-sampling model, kept dependency-light for exactly this use.

Every place that reimplements high-level behaviour for this reason is flagged inline with a
``# QGIS divergence:`` comment naming the library function it stands in for. When that library
function's behaviour changes (aggregation semantics, null handling, the sampling plan), the
mirrored code here must be updated to match.
"""

from __future__ import annotations

import threading
import time
from queue import Queue

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType

# Default pedestrian speed, matching the Rust WALKING_SPEED constant.
DEFAULT_SPEED_M_S = 1.33333


def run_with_feedback(progress_src, func, total, feedback, progress_base=0, progress_span=100):
    """
    Run a Rust function in a background thread, polling
    progress_src.progress() to drive the QGIS feedback progress bar.

    progress_base and progress_span map the sub-task's 0-100% onto a
    slice of the overall algorithm progress, e.g. progress_base=40,
    progress_span=20 means this sub-task fills 40%-60%.

    QGIS divergence: mirrors cityseer.config.wrap_progress but drives
    QgsProcessingFeedback instead of tqdm, to avoid the tqdm dependency
    (see the module docstring's dependency-avoidance note).
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


class CityseerAlgorithmBase(QgsProcessingAlgorithm):
    @staticmethod
    def tr(string: str) -> str:
        from qgis.PyQt.QtCore import QCoreApplication

        return QCoreApplication.translate("CityseerAlgorithm", string)

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def helpUrl(self) -> str:
        return "https://cityseer.benchmarkurbanism.com"

    @staticmethod
    def import_cityseer():
        try:
            import cityseer  # noqa: F401
        except ImportError as exc:
            raise QgsProcessingException(
                f"cityseer is not installed. Install it with: pip install cityseer\n{exc}"
            ) from exc

    def add_time_parameters(self):
        """Add advanced MINUTES and SPEED_M_S parameters for time-based thresholds."""
        minutes_param = QgsProcessingParameterString(
            "MINUTES",
            self.tr(
                "Time thresholds (comma-separated minutes; overrides distances when set). "
                "Converted to metres using the walking speed; output columns are named by "
                "the converted metre distances."
            ),
            defaultValue="",
            optional=True,
        )
        minutes_param.setFlags(minutes_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(minutes_param)
        speed_param = QgsProcessingParameterNumber(
            "SPEED_M_S",
            self.tr("Walking speed in metres per second (converts minutes to distances)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=DEFAULT_SPEED_M_S,
            optional=False,
            minValue=0.1,
            maxValue=40.0,
        )
        speed_param.setFlags(speed_param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(speed_param)

    def resolve_thresholds(self, parameters, context, feedback, distances_param="DISTANCES"):
        """Resolve distance thresholds from DISTANCES or MINUTES + SPEED_M_S.

        Returns (distances, speed_m_s). When minutes are supplied they are converted to
        metre distances with the walking speed (matching the library's
        pair_distances_and_time), so downstream column naming keys on resolved metres.
        """
        from ..utils.converters import parse_distances

        speed = self.parameterAsDouble(parameters, "SPEED_M_S", context)
        minutes_str = self.parameterAsString(parameters, "MINUTES", context) or ""
        if minutes_str.strip():
            try:
                minutes = [float(p.strip()) for p in minutes_str.split(",") if p.strip()]
            except ValueError as exc:
                raise QgsProcessingException(f"Invalid minutes value in: {minutes_str!r}") from exc
            if not minutes or any(m <= 0 for m in minutes):
                raise QgsProcessingException("Time thresholds must be positive minutes.")
            from cityseer import rustalgos

            distances, _seconds = rustalgos.pair_distances_and_time(speed, None, minutes)
            distances = sorted(set(int(d) for d in distances))
            feedback.pushInfo(
                f"Minutes {minutes} at {speed:.2f} m/s resolve to distances {distances} (columns use metres)."
            )
            return distances, speed

        distances_str = self.parameterAsString(parameters, distances_param, context)
        try:
            distances = parse_distances(distances_str)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        return distances, speed

    def resolve_network_layer(self, parameters, context, feedback, param_name="INPUT_LAYER"):
        """Load and validate the street network line layer; returns (layer, crs)."""
        layer = self.parameterAsVectorLayer(parameters, param_name, context)
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
        return layer, crs

    def load_boundary(self, parameters, context, crs, feedback, param_name="BOUNDARY_LAYER"):
        """Load the optional boundary polygon layer as a merged shapely geometry, or None."""
        boundary_layer = self.parameterAsVectorLayer(parameters, param_name, context)
        if boundary_layer is None:
            feedback.pushInfo("No boundary polygon provided (all segments are live sources).")
            return None
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
        except Exception as exc:
            raise QgsProcessingException(f"Failed to parse boundary polygon layer: {exc}") from exc
        if not polys:
            feedback.reportError(
                "Boundary layer has no valid geometries — ignoring boundary. "
                "All segments will be treated as live sources."
            )
            return None
        feedback.pushInfo(f"Boundary polygon loaded ({len(polys)} feature(s)).")
        return unary_union(polys)

    def write_segments_output(
        self,
        parameters,
        context,
        feedback,
        ns,
        fid_list,
        geoms,
        results,
        crs,
        progress_base,
        progress_span,
        output_param="OUTPUT",
    ):
        """Write live street segments with result columns to the output sink; returns dest_id.

        ``results`` maps fid -> {column: value}. Column order follows first appearance
        across ``fid_list``.
        """
        all_cols: list[str] = []
        seen_cols: set[str] = set()
        for fid in fid_list:
            for col in results[fid]:
                if col not in seen_cols:
                    all_cols.append(col)
                    seen_cols.add(col)

        fields = QgsFields()
        fields.append(QgsField("fid", QMetaType.Type.Int))
        for col in all_cols:
            fields.append(QgsField(col, QMetaType.Type.Double, "double", 30, 6))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            output_param,
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
                feedback.setProgress(int(progress_base + pct * progress_span))

        return dest_id
