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
        return "Cityseer - Urban Network Analysis"

    def icon(self) -> QIcon:
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            return QIcon(str(icon_path))
        return super().icon()

    def loadAlgorithms(self) -> None:
        from .algorithms.accessibility import CityseerAccessibilityAlgorithm
        from .algorithms.centrality import CityseerCentralityAlgorithm
        from .algorithms.demand import CityseerDemandAlgorithm
        from .algorithms.mixed_uses import CityseerMixedUsesAlgorithm
        from .algorithms.stats import CityseerStatsAlgorithm

        self.addAlgorithm(CityseerCentralityAlgorithm())
        self.addAlgorithm(CityseerDemandAlgorithm())
        self.addAlgorithm(CityseerAccessibilityAlgorithm())
        self.addAlgorithm(CityseerMixedUsesAlgorithm())
        self.addAlgorithm(CityseerStatsAlgorithm())
