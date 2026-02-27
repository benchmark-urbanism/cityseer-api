from __future__ import annotations

__all__ = ["layers", "networks", "observe"]


def __getattr__(name: str):
    """Lazy-load submodules so that importing cityseer.metrics does not pull in geopandas, pandas, etc."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
