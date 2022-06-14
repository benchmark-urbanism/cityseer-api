# Stubs for networkx.algorithms.chordal (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

import networkx as nx
from typing import Any

class NetworkXTreewidthBoundExceeded(nx.NetworkXException): ...

def is_chordal(G: Any): ...
def find_induced_nodes(G: Any, s: Any, t: Any, treewidth_bound: Any = ...): ...
def chordal_graph_cliques(G: Any): ...
def chordal_graph_treewidth(G: Any): ...
