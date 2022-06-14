# Stubs for networkx.algorithms.connectivity.kcutsets (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from networkx.algorithms.flow import edmonds_karp
from typing import Any, Optional

default_flow_func = edmonds_karp

def all_node_cuts(G: Any, k: Optional[Any] = ..., flow_func: Optional[Any] = ...) -> None: ...
