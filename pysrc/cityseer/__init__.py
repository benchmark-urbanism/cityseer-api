from __future__ import annotations

import os

from . import rustalgos

__all__ = ["metrics", "tools", "config", "sampling", "rustalgos", "CityNetwork"]

# for handling GeoPandas warnings
os.environ["USE_PYGEOS"] = "0"


def __getattr__(name: str):
    """Lazy-load heavy submodules so lightweight consumers (e.g. QGIS plugin) can import cityseer
    without pulling in networkx, geopandas, tqdm, etc."""
    if name in ("config", "metrics", "sampling", "tools"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name == "CityNetwork":
        import importlib

        return importlib.import_module(".network", __name__).CityNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
