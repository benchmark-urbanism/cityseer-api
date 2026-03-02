from __future__ import annotations

import pytest
from cityseer.tools import graphs, io, mock


@pytest.fixture
def network_structure():
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G)
    return ns, nodes_gdf


def _linestring_wkt(a: tuple[float, float], b: tuple[float, float]) -> str:
    return f"LINESTRING({a[0]} {a[1]}, {b[0]} {b[1]})"


class TestGraphMutationRobustness:
    def test_set_node_live_rejects_removed_node(self, network_structure):
        ns, _ = network_structure
        node_idx = ns.street_node_indices()[0]
        ns.remove_street_node(node_idx)

        with pytest.raises(ValueError, match="does not exist"):
            ns.set_node_live(node_idx, True)

    def test_node_getters_reject_removed_node(self, network_structure):
        ns, _ = network_structure
        node_idx = ns.street_node_indices()[0]
        ns.remove_street_node(node_idx)

        with pytest.raises(ValueError, match="node_idx .* does not exist"):
            ns.get_node_payload_py(node_idx)
        with pytest.raises(ValueError, match="node_idx .* does not exist"):
            ns.get_node_weight(node_idx)
        with pytest.raises(ValueError, match="node_idx .* does not exist"):
            ns.is_node_live(node_idx)

    def test_closeness_rejects_removed_source_index(self, network_structure):
        ns, _ = network_structure
        node_idx = ns.street_node_indices()[0]
        ns.remove_street_node(node_idx)

        with pytest.raises(ValueError, match="does not exist"):
            ns.centrality_shortest(
                compute_closeness=True,
                compute_betweenness=False,
                distances=[500],
                source_indices=[node_idx],
                sample_probability=1.0,
                pbar_disabled=True,
            )

    def test_betweenness_rejects_removed_source_index(self, network_structure):
        ns, _ = network_structure
        node_idx = ns.street_node_indices()[0]
        ns.remove_street_node(node_idx)

        with pytest.raises(ValueError, match="does not exist"):
            ns.centrality_shortest(
                compute_closeness=False,
                compute_betweenness=True,
                distances=[500],
                source_indices=[node_idx],
                sample_probability=1.0,
                pbar_disabled=True,
            )

    def test_add_street_edge_rejects_removed_endpoint(self, network_structure):
        ns, _ = network_structure
        valid_idx, removed_idx = ns.street_node_indices()[:2]

        valid_payload = ns.get_node_payload_py(valid_idx)
        removed_payload = ns.get_node_payload_py(removed_idx)
        geom_wkt = _linestring_wkt(valid_payload.coord, removed_payload.coord)

        ns.remove_street_node(removed_idx)

        with pytest.raises(ValueError, match="end_nd_idx .* does not exist"):
            ns.add_street_edge(
                valid_idx,
                removed_idx,
                999999,
                valid_payload.node_key,
                removed_payload.node_key,
                geom_wkt,
            )

    def test_remove_edge_and_getters_return_value_error(self, network_structure):
        ns, _ = network_structure
        start_idx, end_idx, edge_idx = ns.edge_references()[0]

        ns.remove_street_edge(start_idx, end_idx, edge_idx)

        with pytest.raises(ValueError, match="Edge not found"):
            ns.get_edge_payload_py(start_idx, end_idx, edge_idx)
        with pytest.raises(ValueError, match="Edge not found"):
            ns.get_edge_length(start_idx, end_idx, edge_idx)
        with pytest.raises(ValueError, match="Edge not found"):
            ns.get_edge_impedance(start_idx, end_idx, edge_idx)

    def test_remove_street_edge_rejects_removed_endpoint(self, network_structure):
        ns, _ = network_structure
        start_idx, end_idx, edge_idx = ns.edge_references()[0]
        ns.remove_street_node(start_idx)

        with pytest.raises(ValueError, match="start_nd_idx .* does not exist"):
            ns.remove_street_edge(start_idx, end_idx, edge_idx)
