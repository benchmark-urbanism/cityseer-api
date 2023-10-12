# pyright: basic
from __future__ import annotations

import pytest

from cityseer.metrics import networks
from cityseer.tools import io, util


def test_add_node(diamond_graph):
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 50, 50)
    assert is_dupe is False
    assert new_name == "0±1"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1"]
    assert diamond_graph.nodes["0±1"] == {"x": 50, "y": 50}

    # same name and coordinates should return None
    response, is_dupe = util.add_node(diamond_graph, ["0", "1"], 50, 50)
    assert is_dupe is True

    # same name and different coordinates should return v2
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 40, 50)
    assert is_dupe is False
    assert new_name == "0±1§v2"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2"]
    assert diamond_graph.nodes["0±1§v2"] == {"x": 40, "y": 50}

    # likewise v3
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 30, 50)
    assert is_dupe is False
    assert new_name == "0±1§v3"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2", "0±1§v3"]
    assert diamond_graph.nodes["0±1§v3"] == {"x": 30, "y": 50}

    # and names should concatenate over old merges
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "0±1"], 60, 30)
    assert is_dupe is False
    assert new_name == "0±0±1"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2", "0±1§v3", "0±0±1"]
    assert diamond_graph.nodes["0±0±1"] == {"x": 60, "y": 30}


def test_blend_metrics(primal_graph):
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    nodes_gdf = networks.node_centrality_shortest(
        network_structure=network_structure, nodes_gdf=nodes_gdf, compute_closeness=True, distances=[500, 1000]
    )
    # AVG
    merged_edges_gdf_avg = util.blend_metrics(nodes_gdf, edges_gdf, method="avg")
    for node_column in nodes_gdf.columns:
        if not node_column.startswith("cc_metric"):
            continue
        for _edge_idx, edge_row in merged_edges_gdf_avg.iterrows():
            start_val = nodes_gdf.loc[edge_row.nx_start_node_key, node_column]
            end_val = nodes_gdf.loc[edge_row.nx_end_node_key, node_column]
            assert edge_row[node_column] == (start_val + end_val) / 2
    # MIN
    merged_edges_gdf_min = util.blend_metrics(nodes_gdf, edges_gdf, method="min")
    for node_column in nodes_gdf.columns:
        if not node_column.startswith("cc_metric"):
            continue
        for _edge_idx, edge_row in merged_edges_gdf_min.iterrows():
            start_val = nodes_gdf.loc[edge_row.nx_start_node_key, node_column]
            end_val = nodes_gdf.loc[edge_row.nx_end_node_key, node_column]
            assert edge_row[node_column] == min([start_val, end_val])
    # MAX
    merged_edges_gdf_max = util.blend_metrics(nodes_gdf, edges_gdf, method="max")
    for node_column in nodes_gdf.columns:
        if not node_column.startswith("cc_metric"):
            continue
        for _edge_idx, edge_row in merged_edges_gdf_max.iterrows():
            start_val = nodes_gdf.loc[edge_row.nx_start_node_key, node_column]
            end_val = nodes_gdf.loc[edge_row.nx_end_node_key, node_column]
            assert edge_row[node_column] == max([start_val, end_val])
