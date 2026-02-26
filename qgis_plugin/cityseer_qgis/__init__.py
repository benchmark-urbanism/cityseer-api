"""
Cityseer QGIS Plugin

Provides QGIS Processing algorithms for computing urban network centrality
metrics (closeness and betweenness) using the cityseer library.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
_CITYSEER_AVAILABLE = False
_CITYSEER_IMPORT_ERROR = None


def _try_import_system() -> bool:
    global _CITYSEER_AVAILABLE, _CITYSEER_IMPORT_ERROR
    try:
        import cityseer  # noqa: F401

        _CITYSEER_AVAILABLE = True
        _CITYSEER_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _CITYSEER_IMPORT_ERROR = f"system import failed: {exc}"
        return False


def _try_import_dev() -> bool:
    global _CITYSEER_AVAILABLE, _CITYSEER_IMPORT_ERROR
    # Walk up from the plugin dir looking for pysrc/cityseer
    candidate = _PLUGIN_DIR
    for _ in range(6):
        candidate = candidate.parent
        pysrc = candidate / "pysrc"
        if pysrc.is_dir() and (pysrc / "cityseer").is_dir():
            inserted = False
            if str(pysrc) not in sys.path:
                sys.path.insert(0, str(pysrc))
                inserted = True
            # Clear any stale cached modules
            stale = [k for k in sys.modules if k == "cityseer" or k.startswith("cityseer.")]
            for k in stale:
                del sys.modules[k]
            try:
                import cityseer  # noqa: F401

                _CITYSEER_AVAILABLE = True
                _CITYSEER_IMPORT_ERROR = None
                return True
            except Exception as exc:
                _CITYSEER_IMPORT_ERROR = f"development import failed: {exc}"
                if inserted:
                    sys.path.remove(str(pysrc))
                return False
    return False


def _setup_cityseer() -> None:
    # Prefer dev path if we're inside a repository checkout
    repo_root = _PLUGIN_DIR.parent.parent
    prefer_dev = (repo_root / "pyproject.toml").exists() and (repo_root / "pysrc" / "cityseer").is_dir()
    if prefer_dev:
        if _try_import_dev():
            return
        _try_import_system()
    else:
        if _try_import_system():
            return
        _try_import_dev()


def _install_cityseer() -> tuple[bool, str]:
    """
    Install or upgrade cityseer via pip in-process.

    Uses pip's internal API rather than subprocess because QGIS embeds Python
    and sys.executable may point to the QGIS binary, not a usable interpreter.

    Installs with --no-deps to avoid pulling in heavy dependencies (networkx,
    geopandas, matplotlib, etc.) that are not needed by the QGIS plugin.
    The plugin only uses cityseer.rustalgos (the compiled Rust extension).
    """
    import contextlib
    import io

    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        return False, "pip is not available in this QGIS Python environment."

    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exit_code = pip_main(["install", "--upgrade", "--no-deps", "cityseer"])
        if exit_code == 0:
            return True, "cityseer installed successfully."
        return False, f"pip install failed (exit code {exit_code}):\n{output.getvalue()}"
    except Exception as e:
        return False, f"Installation failed: {e}"


_setup_cityseer()


class CityseerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        from .provider import CityseerProvider

        self.provider = CityseerProvider()
        from qgis.core import QgsApplication

        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()
        if not _CITYSEER_AVAILABLE:
            self._prompt_install()

    def _prompt_install(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Cityseer plugin")
        msg.setText("The cityseer library is not installed.")
        msg.setInformativeText(
            "Would you like to install it now?\n\n"
            f"({_CITYSEER_IMPORT_ERROR})"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)

        if msg.exec() == QMessageBox.StandardButton.Yes:
            success, message = _install_cityseer()
            if success:
                # Re-attempt import after install
                _setup_cityseer()
                QMessageBox.information(
                    None,
                    "Cityseer plugin",
                    "cityseer installed successfully.\n\n"
                    "Please restart QGIS to activate the plugin.",
                )
            else:
                QMessageBox.critical(
                    None,
                    "Cityseer plugin",
                    f"Failed to install cityseer.\n\n{message}\n\n"
                    "Try manually: pip install cityseer",
                )

    def unload(self):
        if self.provider is not None:
            from qgis.core import QgsApplication

            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None


def classFactory(iface):
    return CityseerPlugin(iface)
