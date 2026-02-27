from __future__ import annotations

__all__ = ["graphs", "io", "mock", "plot", "util"]


def __getattr__(name: str):
    """Lazy-load submodules so that importing cityseer.tools does not pull in networkx, matplotlib, etc."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
