# Stubs for networkx.algorithms.tests.test_graphical (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from typing import Any

def test_valid_degree_sequence1() -> None: ...
def test_valid_degree_sequence2() -> None: ...
def test_string_input() -> None: ...
def test_negative_input() -> None: ...
def test_non_integer_input() -> None: ...

class TestAtlas:
    @classmethod
    def setupClass(cls) -> None: ...
    GAG: Any = ...
    def setUp(self) -> None: ...
    def test_atlas(self) -> None: ...

def test_small_graph_true() -> None: ...
def test_small_graph_false() -> None: ...
def test_directed_degree_sequence() -> None: ...
def test_small_directed_sequences() -> None: ...
def test_multi_sequence() -> None: ...
def test_pseudo_sequence() -> None: ...
def test_numpy_degree_sequence() -> None: ...
def test_numpy_noninteger_degree_sequence() -> None: ...
