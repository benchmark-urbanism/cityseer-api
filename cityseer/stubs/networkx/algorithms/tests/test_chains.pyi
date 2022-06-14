# Stubs for networkx.algorithms.tests.test_chains (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any
from unittest import TestCase

def cycles(seq: Any) -> None: ...
def cyclic_equals(seq1: Any, seq2: Any): ...

class TestChainDecomposition(TestCase):
    def assertContainsChain(self, chain: Any, expected: Any) -> None: ...
    def test_decomposition(self) -> None: ...
    def test_barbell_graph(self) -> None: ...
    def test_disconnected_graph(self) -> None: ...
    def test_disconnected_graph_root_node(self) -> None: ...
