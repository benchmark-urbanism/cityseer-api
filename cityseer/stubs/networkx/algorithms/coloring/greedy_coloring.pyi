# Stubs for networkx.algorithms.coloring.greedy_coloring (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any, Optional

def strategy_largest_first(G: Any, colors: Any): ...
def strategy_random_sequential(G: Any, colors: Any, seed: Optional[Any] = ...): ...
def strategy_smallest_last(G: Any, colors: Any): ...
def strategy_independent_set(G: Any, colors: Any) -> None: ...
def strategy_connected_sequential_bfs(G: Any, colors: Any): ...
def strategy_connected_sequential_dfs(G: Any, colors: Any): ...
def strategy_connected_sequential(G: Any, colors: Any, traversal: str = ...) -> None: ...
def strategy_saturation_largest_first(G: Any, colors: Any): ...
def greedy_color(G: Any, strategy: str = ..., interchange: bool = ...): ...
