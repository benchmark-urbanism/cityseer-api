from __future__ import annotations

from pathlib import Path

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon


class CityseerProvider(QgsProcessingProvider):
    def id(self) -> str:
        return "cityseer"

    def name(self) -> str:
        return "Cityseer"

    def longName(self) -> str:
        return "Cityseer — Urban Network Centrality"

    def icon(self) -> QIcon:
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            return QIcon(str(icon_path))
        return super().icon()

    def loadAlgorithms(self) -> None:
        from .algorithms.centrality import CityseerCentralityAlgorithm

        self.addAlgorithm(CityseerCentralityAlgorithm())
