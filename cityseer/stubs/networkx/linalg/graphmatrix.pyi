# Stubs for networkx.linalg.graphmatrix (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any, Optional

def incidence_matrix(G: Any, nodelist: Optional[Any] = ..., edgelist: Optional[Any] = ..., oriented: bool = ..., weight: Optional[Any] = ...): ...
def adjacency_matrix(G: Any, nodelist: Optional[Any] = ..., weight: str = ...): ...
adj_matrix = adjacency_matrix
