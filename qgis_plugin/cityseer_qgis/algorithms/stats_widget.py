"""Custom dialog for the Statistics algorithm.

Provides a statistics selector: checkboxes for each statistic type
(sum, mean, median, count, variance, MAD, max, min) with select/deselect all.
"""

from __future__ import annotations

import traceback

from processing.gui.AlgorithmDialog import AlgorithmDialog
from processing.gui.ParametersPanel import ParametersPanel
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# (param_name, label, tooltip, default_checked)
_STAT_DEFS = [
    ("STAT_SUM", "Sum", "Sum of values within distance threshold", True),
    ("STAT_MEAN", "Mean", "Mean of values within distance threshold", True),
    ("STAT_COUNT", "Count", "Count of data points within distance threshold", True),
    ("STAT_MEDIAN", "Median", "Median of values within distance threshold", False),
    ("STAT_VARIANCE", "Variance", "Variance of values within distance threshold", False),
    ("STAT_MAD", "MAD", "Median Absolute Deviation within distance threshold", False),
    ("STAT_MAX", "Max", "Maximum value within distance threshold", False),
    ("STAT_MIN", "Min", "Minimum value within distance threshold", False),
]


class StatsParametersPanel(ParametersPanel):
    """Custom parameters panel with a statistics selector."""

    def __init__(self, parent, alg):
        self._stat_cbs: dict[str, QCheckBox] = {}
        super().__init__(parent, alg)

    def initWidgets(self):
        """Create standard widgets, then insert the statistics selector."""
        super().initWidgets()
        try:
            self._insert_stats_selector()
        except Exception:
            traceback.print_exc()

    def _insert_stats_selector(self):
        """Build and insert the statistics selector group box."""
        box = QGroupBox("Statistics to compute")
        outer = QVBoxLayout()

        # Buttons row
        btn_layout = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(self._select_all)
        deselect_all = QPushButton("Deselect all")
        deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(select_all)
        btn_layout.addWidget(deselect_all)
        btn_layout.addStretch()
        outer.addLayout(btn_layout)

        # Checkboxes
        for param_name, label, tooltip, default in _STAT_DEFS:
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setChecked(default)
            self._stat_cbs[param_name] = cb
            outer.addWidget(cb)

        outer.addStretch()
        box.setLayout(outer)

        # Insert before BOUNDARY_LAYER parameter
        content = self.findChild(QWidget, "scrollAreaWidgetContents")
        if content is not None and content.layout() is not None:
            layout = content.layout()
            boundary_wrapper = self.wrappers.get("BOUNDARY_LAYER")
            if boundary_wrapper is not None:
                boundary_label = boundary_wrapper.wrappedLabel()
                if boundary_label is not None:
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item is not None and item.widget() is boundary_label:
                            layout.insertWidget(i, box)
                            box.show()
                            return
            # Fallback: insert before trailing stretch
            layout.insertWidget(max(0, layout.count() - 1), box)
            box.show()
            return
        self.addExtraWidget(box)

    def _select_all(self):
        for cb in self._stat_cbs.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self._stat_cbs.values():
            cb.setChecked(False)

    def createProcessingParameters(self, flags=None):
        """Read parameter values, adding the selected statistics."""
        if flags is not None:
            params = super().createProcessingParameters(flags)
        else:
            params = super().createProcessingParameters()

        for param_name, cb in self._stat_cbs.items():
            params[param_name] = cb.isChecked()

        return params


class StatsDialog(AlgorithmDialog):
    """Custom dialog that uses the statistics parameters panel."""

    def getParametersPanel(self, alg, parent):
        return StatsParametersPanel(parent, alg)
