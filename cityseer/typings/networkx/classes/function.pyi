"""
This type stub file was generated by pyright.
"""

from networkx.utils import not_implemented_for

"""Functional interface to graph me"""
__all__ = ["nodes", "edges", "degree", "degree_histogram", "neighbors", "number_of_nodes", "number_of_edges", "density", "is_directed", "info", "freeze", "is_frozen", "subgraph", "subgraph_view", "induced_subgraph", "reverse_view", "edge_subgraph", "restricted_view", "to_directed", "to_undirected", "add_star", "add_path", "add_cycle", "create_empty_copy", "set_node_attributes", "get_node_attributes", "set_edge_attributes", "get_edge_attributes", "all_neighbors", "non_neighbors", "non_edges", "common_neighbors", "is_weighted", "is_negatively_weighted", "is_empty", "selfloop_edges", "nodes_with_selfloops", "number_of_selfloops", "path_weight", "is_path"]
def nodes(G):
    """Returns an iterator over the gra"""
    ...

def edges(G, nbunch=...):
    """Returns an edge view of edges in"""
    ...

def degree(G, nbunch=..., weight=...):
    """Returns a degree view of single """
    ...

def neighbors(G, n):
    """Returns a list of nodes connecte"""
    ...

def number_of_nodes(G):
    """Returns the number of nodes in t"""
    ...

def number_of_edges(G):
    """Returns the number of edges in t"""
    ...

def density(G): # -> Literal[0]:
    r"""Returns the density of a graph.
"""
    ...

def degree_histogram(G): # -> list[int]:
    """Returns a list of the frequency """
    ...

def is_directed(G):
    """Return True if graph is directed"""
    ...

def frozen(*args, **kwargs):
    """Dummy method for raising errors """
    ...

def freeze(G):
    """Modify graph to prevent further """
    ...

def is_frozen(G): # -> Literal[False]:
    """Returns True if graph is frozen."""
    ...

def add_star(G_to_add_to, nodes_for_star, **attr): # -> None:
    """Add a star to Graph G_to_add_to."""
    ...

def add_path(G_to_add_to, nodes_for_path, **attr): # -> None:
    """Add a path to the Graph G_to_add"""
    ...

def add_cycle(G_to_add_to, nodes_for_cycle, **attr): # -> None:
    """Add a cycle to the Graph G_to_ad"""
    ...

def subgraph(G, nbunch):
    """Returns the subgraph induced on """
    ...

def induced_subgraph(G, nbunch):
    """Returns a SubGraph view of `G` s"""
    ...

def edge_subgraph(G, edges):
    """Returns a view of the subgraph i"""
    ...

def restricted_view(G, nodes, edges):
    """Returns a view of `G` with hidde"""
    ...

def to_directed(graph):
    """Returns a directed view of the g"""
    ...

def to_undirected(graph):
    """Returns an undirected view of th"""
    ...

def create_empty_copy(G, with_data=...):
    """Returns a copy of the graph G wi"""
    ...

def info(G, n=...): # -> str:
    """Return a summary of information """
    ...

def set_node_attributes(G, values, name=...): # -> None:
    """Sets node attributes from a give"""
    ...

def get_node_attributes(G, name): # -> dict[Unknown, Unknown]:
    """Get node attributes from graph

"""
    ...

def set_edge_attributes(G, values, name=...):
    """Sets edge attributes from a give"""
    ...

def get_edge_attributes(G, name): # -> dict[Unknown, Unknown]:
    """Get edge attributes from graph

"""
    ...

def all_neighbors(graph, node): # -> chain[Unknown]:
    """Returns all of the neighbors of """
    ...

def non_neighbors(graph, node): # -> Generator[Unknown, None, None]:
    """Returns the non-neighbors of the"""
    ...

def non_edges(graph): # -> Generator[tuple[Unknown, Unknown], None, None]:
    """Returns the non-existent edges i"""
    ...

@not_implemented_for("directed")
def common_neighbors(G, u, v): # -> Generator[Unknown, None, None]:
    """Returns the common neighbors of """
    ...

def is_weighted(G, edge=..., weight=...): # -> bool:
    """Returns True if `G` has weighted"""
    ...

def is_negatively_weighted(G, edge=..., weight=...): # -> bool:
    """Returns True if `G` has negative"""
    ...

def is_empty(G): # -> bool:
    """Returns True if `G` has no edges"""
    ...

def nodes_with_selfloops(G): # -> Generator[Unknown, None, None]:
    """Returns an iterator over nodes w"""
    ...

def selfloop_edges(G, data=..., keys=..., default=...):
    """Returns an iterator over selfloo"""
    ...

def number_of_selfloops(G): # -> Literal[1, 0]:
    """Returns the number of selfloop e"""
    ...

def is_path(G, path): # -> bool:
    """Returns whether or not the speci"""
    ...

def path_weight(G, path, weight): # -> Literal[0]:
    """Returns total cost associated wi"""
    ...

