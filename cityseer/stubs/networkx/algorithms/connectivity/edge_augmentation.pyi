# Stubs for networkx.algorithms.connectivity.edge_augmentation (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from collections import namedtuple
from typing import Any, Optional

def is_k_edge_connected(G: Any, k: Any): ...
def is_locally_k_edge_connected(G: Any, s: Any, t: Any, k: Any): ...
def k_edge_augmentation(G: Any, k: Any, avail: Optional[Any] = ..., weight: Optional[Any] = ..., partial: bool = ...) -> None: ...

MetaEdge = namedtuple('MetaEdge', ['meta_uv', 'uv', 'w'])
