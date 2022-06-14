# Stubs for networkx.drawing.layout (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any, Optional

def random_layout(G: Any, center: Optional[Any] = ..., dim: int = ..., seed: Optional[Any] = ...): ...
def circular_layout(G: Any, scale: int = ..., center: Optional[Any] = ..., dim: int = ...): ...
def shell_layout(G: Any, nlist: Optional[Any] = ..., scale: int = ..., center: Optional[Any] = ..., dim: int = ...): ...
def bipartite_layout(G: Any, nodes: Any, align: str = ..., scale: int = ..., center: Optional[Any] = ..., aspect_ratio: Any = ...): ...
def fruchterman_reingold_layout(G: Any, k: Optional[Any] = ..., pos: Optional[Any] = ..., fixed: Optional[Any] = ..., iterations: int = ..., threshold: float = ..., weight: str = ..., scale: int = ..., center: Optional[Any] = ..., dim: int = ..., seed: Optional[Any] = ...): ...
spring_layout = fruchterman_reingold_layout

def kamada_kawai_layout(G: Any, dist: Optional[Any] = ..., pos: Optional[Any] = ..., weight: str = ..., scale: int = ..., center: Optional[Any] = ..., dim: int = ...): ...
def spectral_layout(G: Any, weight: str = ..., scale: int = ..., center: Optional[Any] = ..., dim: int = ...): ...
def planar_layout(G: Any, scale: int = ..., center: Optional[Any] = ..., dim: int = ...): ...
def rescale_layout(pos: Any, scale: int = ...): ...
