from __future__ import annotations

from qgis.core import QgsProcessingAlgorithm, QgsProcessingException


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
