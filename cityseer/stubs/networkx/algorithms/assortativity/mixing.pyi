# Stubs for networkx.algorithms.assortativity.mixing (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any, Optional

def attribute_mixing_dict(G: Any, attribute: Any, nodes: Optional[Any] = ..., normalized: bool = ...): ...
def attribute_mixing_matrix(G: Any, attribute: Any, nodes: Optional[Any] = ..., mapping: Optional[Any] = ..., normalized: bool = ...): ...
def degree_mixing_dict(G: Any, x: str = ..., y: str = ..., weight: Optional[Any] = ..., nodes: Optional[Any] = ..., normalized: bool = ...): ...
def degree_mixing_matrix(G: Any, x: str = ..., y: str = ..., weight: Optional[Any] = ..., nodes: Optional[Any] = ..., normalized: bool = ...): ...
def numeric_mixing_matrix(G: Any, attribute: Any, nodes: Optional[Any] = ..., normalized: bool = ...): ...
def mixing_dict(xy: Any, normalized: bool = ...): ...
