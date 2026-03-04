"""Custom dialog for the Accessibility algorithm.

Provides a dynamic land-use category selector: the user selects a field,
clicks "Load categories", and checkboxes appear for each unique value.
"""

from __future__ import annotations

import traceback

from processing.gui.AlgorithmDialog import AlgorithmDialog
from processing.gui.ParametersPanel import ParametersPanel
from qgis.core import QgsMapLayerProxyModel, QgsProject, QgsVectorLayer
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class AccessibilityParametersPanel(ParametersPanel):
    """Custom parameters panel with a dynamic land-use category selector."""

    def __init__(self, parent, alg):
        self._category_box = None
        self._category_layout = None
        self._category_cbs: list[QCheckBox] = []
        super().__init__(parent, alg)

    def initWidgets(self):
        """Create standard widgets, then insert the category selector."""
        super().initWidgets()
        try:
            self._insert_category_selector()
        except Exception:
            traceback.print_exc()

    def _insert_category_selector(self):
        """Build and insert the category selector group box."""
        self._category_box = QGroupBox("Land-use categories")

        outer = QVBoxLayout()

        # Buttons row
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load categories")
        load_btn.setToolTip("Read unique values from the selected land-use field")
        load_btn.clicked.connect(self._reload_categories)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(self._select_all)
        deselect_all = QPushButton("Deselect all")
        deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(select_all)
        btn_layout.addWidget(deselect_all)
        btn_layout.addStretch()
        outer.addLayout(btn_layout)

        # Scrollable area for category checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll_content = QWidget()
        self._category_layout = QVBoxLayout()
        self._category_layout.setContentsMargins(4, 4, 4, 4)
        self._category_layout.addWidget(
            QLabel("Select a land-use field and click 'Load categories'.")
        )
        self._category_layout.addStretch()
        scroll_content.setLayout(self._category_layout)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll)

        self._category_box.setLayout(outer)

        # Insert after the LANDUSE_FIELD parameter
        content = self.findChild(QWidget, "scrollAreaWidgetContents")
        if content is not None and content.layout() is not None:
            layout = content.layout()
            # Find the DISTANCES wrapper to insert before it
            dist_wrapper = self.wrappers.get("DISTANCES")
            if dist_wrapper is not None:
                dist_label = dist_wrapper.wrappedLabel()
                if dist_label is not None:
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item is not None and item.widget() is dist_label:
                            layout.insertWidget(i, self._category_box)
                            self._category_box.show()
                            return
            # Fallback: insert before trailing stretch
            layout.insertWidget(max(0, layout.count() - 1), self._category_box)
            self._category_box.show()
            return
        self.addExtraWidget(self._category_box)

    def _get_current_layer(self) -> QgsVectorLayer | None:
        """Get the currently selected data layer."""
        layer_wrapper = self.wrappers.get("DATA_LAYER")
        if layer_wrapper is None:
            return None
        try:
            # Try the widget directly (QgsMapLayerComboBox)
            widget = layer_wrapper.wrappedWidget()
            if widget is not None and hasattr(widget, "currentLayer"):
                layer = widget.currentLayer()
                if isinstance(layer, QgsVectorLayer):
                    return layer
            # Fallback: parameterValue may return a layer ID string
            val = layer_wrapper.parameterValue()
            if isinstance(val, QgsVectorLayer):
                return val
            if isinstance(val, str) and val:
                layer = QgsProject.instance().mapLayer(val)
                if isinstance(layer, QgsVectorLayer):
                    return layer
        except Exception:
            traceback.print_exc()
        return None

    def _get_current_field(self) -> str:
        """Get the currently selected field name."""
        field_wrapper = self.wrappers.get("LANDUSE_FIELD")
        if field_wrapper is None:
            return ""
        try:
            # Try the widget directly (QgsFieldComboBox)
            widget = field_wrapper.wrappedWidget()
            if widget is not None and hasattr(widget, "currentField"):
                field = widget.currentField()
                if field:
                    return field
            # Fallback: parameterValue
            val = field_wrapper.parameterValue()
            return str(val) if val else ""
        except Exception:
            traceback.print_exc()
            return ""

    def _reload_categories(self):
        """Read unique values from the selected field and populate checkboxes."""
        # Clear existing checkboxes
        self._category_cbs.clear()
        while self._category_layout.count() > 0:
            item = self._category_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        layer = self._get_current_layer()
        field_name = self._get_current_field()

        if layer is None or not field_name:
            self._category_layout.addWidget(
                QLabel("Select a data layer and land-use field first.")
            )
            self._category_layout.addStretch()
            return

        # Check field exists
        if field_name not in [f.name() for f in layer.fields()]:
            self._category_layout.addWidget(
                QLabel(f"Field '{field_name}' not found in layer.")
            )
            self._category_layout.addStretch()
            return

        # Read unique values
        idx = layer.fields().indexFromName(field_name)
        unique_values = sorted(
            set(str(v) for v in layer.uniqueValues(idx) if v is not None and str(v).strip())
        )

        if not unique_values:
            self._category_layout.addWidget(
                QLabel("No values found in selected field.")
            )
            self._category_layout.addStretch()
            return

        for val in unique_values:
            cb = QCheckBox(val)
            cb.setChecked(False)
            self._category_cbs.append(cb)
            self._category_layout.addWidget(cb)
        self._category_layout.addStretch()

    def _select_all(self):
        for cb in self._category_cbs:
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self._category_cbs:
            cb.setChecked(False)

    def createProcessingParameters(self, flags=None):
        """Read parameter values, adding the selected categories."""
        if flags is not None:
            params = super().createProcessingParameters(flags)
        else:
            params = super().createProcessingParameters()

        # Build comma-separated list of selected categories
        selected = [cb.text() for cb in self._category_cbs if cb.isChecked()]
        params["ACCESSIBILITY_KEYS"] = ",".join(selected)

        return params


class AccessibilityDialog(AlgorithmDialog):
    """Custom dialog that uses the accessibility parameters panel."""

    def getParametersPanel(self, alg, parent):
        return AccessibilityParametersPanel(parent, alg)
