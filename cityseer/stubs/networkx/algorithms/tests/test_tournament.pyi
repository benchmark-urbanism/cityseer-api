# Stubs for networkx.algorithms.tests.test_tournament (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

class TestIsTournament:
    def test_is_tournament(self) -> None: ...
    def test_self_loops(self) -> None: ...
    def test_missing_edges(self) -> None: ...
    def test_bidirectional_edges(self) -> None: ...

class TestRandomTournament:
    def test_graph_is_tournament(self) -> None: ...
    def test_graph_is_tournament_seed(self) -> None: ...

class TestHamiltonianPath:
    def test_path_is_hamiltonian(self) -> None: ...
    def test_hamiltonian_cycle(self) -> None: ...

class TestReachability:
    def test_reachable_pair(self) -> None: ...
    def test_same_node_is_reachable(self) -> None: ...
    def test_unreachable_pair(self) -> None: ...

class TestStronglyConnected:
    def test_is_strongly_connected(self) -> None: ...
    def test_not_strongly_connected(self) -> None: ...
