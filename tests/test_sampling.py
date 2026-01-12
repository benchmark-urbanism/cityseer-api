from __future__ import annotations
import os
import time
import pytest
from cityseer import rustalgos, config
from cityseer.tools import io, graphs, mock

def test_centrality_sampling_speed():
    """
    Test the impact of sampling on computation speed.
    """
    if "GITHUB_ACTIONS" in os.environ:
        pytest.skip("Skipping performance test in CI")

    # load a mock graph
    G_primal = mock.mock_graph(nx_rows=20, nx_cols=20)
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G_primal)
    
    distances = [500]
    
    # Baseline without sampling
    start_time = time.time()
    network_structure.local_node_centrality_shortest(
        distances=distances,
        pbar_disabled=True
    )
    baseline_duration = time.time() - start_time
    print(f"\nBaseline duration (no sampling): {baseline_duration:.4f}s")

    # With 30% sampling
    start_time = time.time()
    network_structure.local_node_centrality_shortest(
        distances=distances,
        sample_probability=0.3,
        pbar_disabled=True
    )
    sampled_duration = time.time() - start_time
    print(f"Sampled duration (30% probability): {sampled_duration:.4f}s")
    
    # We expect some speedup, although for very small graphs the overhead might dominate.
    # In a 20x20 grid, there are ~400 nodes, so sampling should be noticeable.
    assert sampled_duration < baseline_duration * 0.8  # Allow some margin for overhead

def test_centrality_weighted_sampling():
    """
    Test that weighted sampling impacts the results as expected.
    """
    # load a mock graph
    G_primal = mock.mock_graph(nx_rows=10, nx_cols=10)
    G_primal = graphs.nx_simple_geoms(G_primal)
    
    # Set weights: half nodes weight 1, half weight 0
    # Actually, let's make it more extreme: one node has weight 1, others have weight 0.001
    # If sample_probability is 1 and weighted_sample is True, then:
    # node with weight 1 has 100% chance
    # nodes with weight 0.001 have 0.1% chance.
    
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G_primal)
    
    # Reset all weights to 0.001 and set one to 1.0
    # We need to recreate the network_structure because weights are set during addition
    # Or we can modify the nodes_gdf and re-import
    nodes_gdf['weight'] = 0.0001
    nodes_gdf.iloc[0, nodes_gdf.columns.get_loc('weight')] = 1.0
    
    # Re-import to apply weights
    # Note: we need to pass nodes_gdf and edges_gdf back to network_structure_from_nx
    # but that tool takes a networkx graph.
    # Let's just manually add nodes to a new network structure for full control.
    
    ns = rustalgos.graph.NetworkStructure.new()
    node_indices = []
    for i, row in nodes_gdf.iterrows():
        idx = ns.add_street_node(i, row.geometry.x, row.geometry.y, True, row['weight'])
        node_indices.append(idx)
        
    # We don't need edges for this test, we just want to see if the loop runs for the expected number of nodes
    # But Dijkstra needs edges to visit anything.
    # Let's just use the existing primal_graph and modify it.
    
    for i, node in enumerate(G_primal.nodes):
        if i == 0:
            G_primal.nodes[node]['weight'] = 1.0
        else:
            G_primal.nodes[node]['weight'] = 0.0
            
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)
    
    # With probability 1.0 and weighted_sample True, only the first node should be sampled.
    # We can check progress to see how many nodes were processed if we had access to it, 
    # but the metrics will also be telltale.
    
    res = ns.local_node_centrality_shortest(
        distances=[500],
        sample_probability=1.0,
        weighted_sample=True,
        pbar_disabled=True
    )
    
    # If only node 0 was sampled as source, then only its results should be non-zero in the density map 
    # (for the closeness part where it is the src).
    # density[dist][src_idx] += wt
    density = res.node_density[500]
    # Find which nodes have non-zero density (meaning they were used as sources)
    source_indices = [i for i, val in enumerate(density) if val > 0]
    
    # Since weight 0 nodes have 0 probability, only node 0 should be processed.
    assert source_indices == [0]

if __name__ == "__main__":
    test_centrality_sampling_speed()
    test_centrality_weighted_sampling()
