"""Custom dialog and parameters widget for the Network Centrality algorithm.

Provides a 4-box layout (closeness/betweenness x shortest/simplest)
with per-box metric checkboxes, replacing the flat parameter list.
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


# Metric definitions per box: (param_name, label, tooltip, default)
_CLOSENESS_METRICS_SHORTEST = [
    ("METRIC_HARMONIC", "Harmonic", "Sum of inverse distances to reachable nodes \u2192 cc_harmonic_<d>", True),
    ("METRIC_DENSITY", "Density", "Number of nodes reachable within the distance threshold \u2192 cc_density_<d>", False),
    ("METRIC_FARNESS", "Farness", "Sum of distances to all reachable nodes \u2192 cc_farness_<d>", False),
    ("METRIC_BETA", "Beta-weighted", "Closeness with negative-exponential distance decay \u2192 cc_beta_<d>", False),
    ("METRIC_CYCLES", "Cycles", "Count of network cycles through each node \u2192 cc_cycles_<d>", False),
    ("METRIC_HILLIER", "Hillier (n\u00b2/farness)", "Derived closeness variant (density\u00b2 / farness) \u2192 cc_hillier_<d>", False),
]

_CLOSENESS_METRICS_SIMPLEST = [
    ("METRIC_HARMONIC", "Harmonic", "Sum of inverse distances to reachable nodes \u2192 cc_harmonic_<d>_ang", True),
    ("METRIC_DENSITY", "Density", "Number of nodes reachable within the distance threshold \u2192 cc_density_<d>_ang", False),
    ("METRIC_FARNESS", "Farness", "Sum of distances to all reachable nodes \u2192 cc_farness_<d>_ang", False),
]

_BETWEENNESS_METRICS_SHORTEST = [
    ("METRIC_BETWEENNESS", "Betweenness", "Count of shortest paths passing through each node \u2192 cc_betweenness_<d>", True),
    ("METRIC_BETWEENNESS_BETA", "Beta-weighted", "Betweenness with negative-exponential distance decay \u2192 cc_betweenness_beta_<d>", False),
]

_BETWEENNESS_METRICS_SIMPLEST = [
    ("METRIC_BETWEENNESS", "Betweenness", "Count of shortest paths passing through each node \u2192 cc_betweenness_<d>_ang", True),
    ("METRIC_BETWEENNESS_BETA", "Beta-weighted", "Betweenness with negative-exponential distance decay \u2192 cc_betweenness_beta_<d>_ang", False),
]


class CentralityParametersPanel(ParametersPanel):
    """Custom parameters panel with 4 group boxes."""

    def __init__(self, parent, alg):
        self._group_boxes = {}
        self._metric_cbs = {}
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
        # Build the 4 group boxes in a 2x2 grid.
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(
            self._make_group_box(
                "Closeness (Shortest Path)", "CLOSENESS_SHORTEST",
                _CLOSENESS_METRICS_SHORTEST, checked=True,
            ),
            0, 0,
        )
        grid.addWidget(
            self._make_group_box(
                "Closeness (Simplest Path)", "CLOSENESS_SIMPLEST",
                _CLOSENESS_METRICS_SIMPLEST, checked=False,
            ),
            0, 1,
        )
        grid.addWidget(
            self._make_group_box(
                "Betweenness (Shortest Path)", "BETWEENNESS_SHORTEST",
                _BETWEENNESS_METRICS_SHORTEST, checked=True,
            ),
            1, 0,
        )
        grid.addWidget(
            self._make_group_box(
                "Betweenness (Simplest Path)", "BETWEENNESS_SIMPLEST",
                _BETWEENNESS_METRICS_SIMPLEST, checked=False,
            ),
            1, 1,
        )
        grid_widget = QWidget()
        grid_widget.setLayout(grid)

        # Insert the grid into the scroll area layout, before TOLERANCE.
        # Find the layout via the named content widget from the .ui file.
        content = self.findChild(QWidget, "scrollAreaWidgetContents")
        if content is not None and content.layout() is not None:
            layout = content.layout()
            # Find the TOLERANCE wrapper's label to insert just before it.
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
        # Last resort: append via C++ method (may have a gap).
        self.addExtraWidget(grid_widget)

    def _make_group_box(self, title, category_param, metrics, checked):
        """Create a checkable QGroupBox with metric checkboxes inside."""
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(checked)
        self._group_boxes[category_param] = box

        layout = QVBoxLayout()
        for param_name, label, tooltip, default in metrics:
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setChecked(default)
            layout.addWidget(cb)

            # Sync with any existing checkbox for the same param.
            if param_name in self._metric_cbs:
                existing = self._metric_cbs[param_name]
                cb.setChecked(existing[0].isChecked())
                for other in existing:
                    cb.toggled.connect(lambda state, o=other: _sync_cb(o, state))
                    other.toggled.connect(lambda state, c=cb: _sync_cb(c, state))
                existing.append(cb)
            else:
                self._metric_cbs[param_name] = [cb]

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
        for param_name, cb_list in self._metric_cbs.items():
            params[param_name] = cb_list[0].isChecked()

        return params


class CentralityDialog(AlgorithmDialog):
    """Custom dialog that uses the 4-box parameters panel."""

    def getParametersPanel(self, alg, parent):
        return CentralityParametersPanel(parent, alg)


def _sync_cb(target, state):
    """Set target checkbox to state without triggering infinite recursion."""
    if target.isChecked() != state:
        target.setChecked(state)
