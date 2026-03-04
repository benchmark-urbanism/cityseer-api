"""Custom dialog and parameters widget for the Network Centrality algorithm.

Provides a 4-box layout (closeness/betweenness x shortest/simplest)
with fully independent per-box metric checkboxes.
"""

from __future__ import annotations

import traceback

from processing.gui.AlgorithmDialog import AlgorithmDialog
from processing.gui.ParametersPanel import ParametersPanel
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

# Metric definitions per box: (param_suffix, label, tooltip_template, default)
# The full param name is built as METRIC_{suffix}_{category_short}
# e.g. METRIC_HARMONIC_CS for closeness-shortest
_CLOSENESS_METRICS_SHORTEST = [
    ("HARMONIC", "Harmonic", "Sum of inverse distances to reachable nodes → cc_harmonic_<d>", True),
    ("DENSITY", "Density", "Number of nodes reachable within the distance threshold → cc_density_<d>", False),
    ("FARNESS", "Farness", "Sum of distances to all reachable nodes → cc_farness_<d>", False),
    ("BETA", "Beta-weighted", "Closeness with negative-exponential distance decay → cc_beta_<d>", False),
    ("CYCLES", "Cycles", "Count of network cycles through each node → cc_cycles_<d>", False),
    ("HILLIER", "Hillier (n²/farness)", "Derived closeness variant (density² / farness) → cc_hillier_<d>", False),
]

_CLOSENESS_METRICS_SIMPLEST = [
    ("HARMONIC", "Harmonic", "Sum of inverse distances to reachable nodes → cc_harmonic_<d>_ang", True),
    ("DENSITY", "Density", "Number of nodes reachable within the distance threshold → cc_density_<d>_ang", False),
    ("FARNESS", "Farness", "Sum of distances to all reachable nodes → cc_farness_<d>_ang", False),
    ("HILLIER", "Hillier (n²/farness)", "Derived closeness variant (density² / farness) → cc_hillier_<d>_ang", False),
]

_BETWEENNESS_METRICS_SHORTEST = [
    ("BETWEENNESS", "Betweenness", "Count of shortest paths passing through each node → cc_betweenness_<d>", True),
    (
        "BETWEENNESS_BETA",
        "Beta-weighted",
        "Betweenness with negative-exponential distance decay → cc_betweenness_beta_<d>",
        False,
    ),
]

_BETWEENNESS_METRICS_SIMPLEST = [
    ("BETWEENNESS", "Betweenness", "Count of shortest paths passing through each node → cc_betweenness_<d>_ang", True),
    (
        "BETWEENNESS_BETA",
        "Beta-weighted",
        "Betweenness with negative-exponential distance decay → cc_betweenness_beta_<d>_ang",
        False,
    ),
]

# Category short codes used to build param names
_CATEGORY_SHORTS = {
    "CLOSENESS_SHORTEST": "CS",
    "CLOSENESS_SIMPLEST": "CA",
    "BETWEENNESS_SHORTEST": "BS",
    "BETWEENNESS_SIMPLEST": "BA",
}


class CentralityParametersPanel(ParametersPanel):
    """Custom parameters panel with 4 group boxes."""

    def __init__(self, parent, alg):
        self._group_boxes = {}
        # key: full param name (e.g. "METRIC_HARMONIC_CS"), value: QCheckBox
        self._metric_cbs: dict[str, QCheckBox] = {}
        # super().__init__ calls self.initWidgets()
        super().__init__(parent, alg)

    def initWidgets(self):
        """Create standard widgets, then insert group boxes into the layout."""
        # Let the base ParametersPanel create widgets for visible params.
        super().initWidgets()
        try:
            self._insert_group_boxes()
        except Exception:
            traceback.print_exc()

    def _insert_group_boxes(self):
        """Build and insert the 4 metric group boxes into the scroll area layout."""
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(
            self._make_group_box(
                "Closeness (Shortest Path)",
                "CLOSENESS_SHORTEST",
                _CLOSENESS_METRICS_SHORTEST,
                checked=True,
            ),
            0,
            0,
        )
        grid.addWidget(
            self._make_group_box(
                "Closeness (Simplest Path)",
                "CLOSENESS_SIMPLEST",
                _CLOSENESS_METRICS_SIMPLEST,
                checked=False,
            ),
            0,
            1,
        )
        grid.addWidget(
            self._make_group_box(
                "Betweenness (Shortest Path)",
                "BETWEENNESS_SHORTEST",
                _BETWEENNESS_METRICS_SHORTEST,
                checked=True,
            ),
            1,
            0,
        )
        grid.addWidget(
            self._make_group_box(
                "Betweenness (Simplest Path)",
                "BETWEENNESS_SIMPLEST",
                _BETWEENNESS_METRICS_SIMPLEST,
                checked=False,
            ),
            1,
            1,
        )
        grid_widget = QWidget()
        grid_widget.setLayout(grid)

        # Insert the grid into the scroll area layout, before TOLERANCE.
        content = self.findChild(QWidget, "scrollAreaWidgetContents")
        if content is not None and content.layout() is not None:
            layout = content.layout()
            tol_wrapper = self.wrappers.get("TOLERANCE")
            if tol_wrapper is not None:
                tol_label = tol_wrapper.wrappedLabel()
                if tol_label is not None:
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item is not None and item.widget() is tol_label:
                            layout.insertWidget(i, grid_widget)
                            grid_widget.show()
                            return
            # Fallback: insert before the trailing stretch.
            layout.insertWidget(max(0, layout.count() - 1), grid_widget)
            grid_widget.show()
            return
        self.addExtraWidget(grid_widget)

    def _make_group_box(self, title, category_param, metrics, checked):
        """Create a checkable QGroupBox with metric checkboxes inside."""
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(checked)
        self._group_boxes[category_param] = box

        cat_short = _CATEGORY_SHORTS[category_param]
        layout = QVBoxLayout()
        for metric_suffix, label, tooltip, default in metrics:
            full_param = f"METRIC_{metric_suffix}_{cat_short}"
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setChecked(default)
            layout.addWidget(cb)
            self._metric_cbs[full_param] = cb

        layout.addStretch()
        box.setLayout(layout)
        return box

    def createProcessingParameters(self, flags=None):
        """Read parameter values, adding category and metric booleans from our custom widgets."""
        if flags is not None:
            params = super().createProcessingParameters(flags)
        else:
            params = super().createProcessingParameters()

        # Add category booleans from group box states.
        for category_param, box in self._group_boxes.items():
            params[category_param] = box.isChecked()

        # Add metric booleans from our custom checkboxes.
        for param_name, cb in self._metric_cbs.items():
            params[param_name] = cb.isChecked()

        return params


class CentralityDialog(AlgorithmDialog):
    """Custom dialog that uses the 4-box parameters panel."""

    def getParametersPanel(self, alg, parent):
        return CentralityParametersPanel(parent, alg)
