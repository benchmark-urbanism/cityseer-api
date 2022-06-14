# Stubs for networkx.algorithms.tree.coding (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

import networkx as nx
from typing import Any

class NotATree(nx.NetworkXException): ...

def to_nested_tuple(T: Any, root: Any, canonical_form: bool = ...): ...
def from_nested_tuple(sequence: Any, sensible_relabeling: bool = ...): ...
def to_prufer_sequence(T: Any): ...
def from_prufer_sequence(sequence: Any): ...
