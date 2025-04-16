from __future__ import annotations

import os

from . import config, metrics, rustalgos, tools

__all__ = ["metrics", "tools", "config", "rustalgos"]

# for handling GeoPandas warnings
os.environ["USE_PYGEOS"] = "0"
