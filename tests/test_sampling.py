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

def test_centrality_sampling_reproducibility():
    """
    Test that providing a seed produces reproducible results.
    """
    G_primal = mock.mock_graph(nx_rows=10, nx_cols=10)
    G_primal = graphs.nx_simple_geoms(G_primal)
    _nodes_gdf, _edges_gdf, ns = io.network_structure_from_nx(G_primal)
    
    distances = [500]
    seed = 42
    
    # First run
    res1 = ns.local_node_centrality_shortest(
        distances=distances,
        sample_probability=0.3,
        random_seed=seed,
        pbar_disabled=True
    )
    
    # Second run
    res2 = ns.local_node_centrality_shortest(
        distances=distances,
        sample_probability=0.3,
        random_seed=seed,
        pbar_disabled=True
    )
    
    # Third run with different seed
    res3 = ns.local_node_centrality_shortest(
        distances=distances,
        sample_probability=0.3,
        random_seed=seed + 1,
        pbar_disabled=True
    )
    
    density1 = res1.node_density[500]
    density2 = res2.node_density[500]
    density3 = res3.node_density[500]
    
    import numpy as np
    assert np.allclose(density1, density2)
    assert not np.allclose(density1, density3)

def test_segment_weight_sampling():
    """
    Test a workflow where weights are on segments (LineStrings) and need to be mapped to nodes.
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    import networkx as nx
    
    # Create a simple line geometry with high weight
    line1 = LineString([(0, 0), (10, 0)])
    # Create another line with low weight
    line2 = LineString([(10, 0), (20, 0)])
    
    # GeoDataFrame with segment weights
    edges_gdf = gpd.GeoDataFrame(
        {'geometry': [line1, line2], 'segment_weight': [100.0, 1.0]},
        crs="EPSG:27700"
    )
    
    # Convert to NetworkX
    G = io.nx_from_generic_geopandas(edges_gdf)
    
    # Map segment weights to nodes
    # Strategy: Assign node weight as the maximum of incident edge weights
    for node in G.nodes():
        incident_weights = []
        for n_nbr, n_dict in G[node].items():
            for edge_key, edge_data in n_dict.items():
                if 'segment_weight' in edge_data:
                    incident_weights.append(edge_data['segment_weight'])
        
        if incident_weights:
            G.nodes[node]['weight'] = max(incident_weights)
        else:
            G.nodes[node]['weight'] = 0.0
            
    # Verify node weights
    # Node at (0,0) connects to line1 (wt 100) -> weight 100
    # Node at (10,0) connects to line1 (wt 100) and line2 (wt 1) -> weight 100
    # Node at (20,0) connects to line2 (wt 1) -> weight 1
    
    # Get node keys (io.nx_from_generic_geopandas uses coordinates as keys string formatted)
    # We can check values directly
    weights = [d['weight'] for n, d in G.nodes(data=True)]
    assert 100.0 in weights
    assert 1.0 in weights
    
    # Convert to NetworkStructure
    nodes_gdf, edges_gdf, ns = io.network_structure_from_nx(G)
    
    # Run weighted sampling
    # Nodes with weight 100 should be sampled ~100x more than node with weight 1
    # But since probs are capped at 1.0, let's normalize or use a small probability
    # If sample_probability=0.01:
    # High weight nodes (100) -> prob 1.0 (capped)
    # Low weight node (1) -> prob 0.01
    
    res = ns.local_node_centrality_shortest(
        distances=[500],
        sample_probability=0.01,
        weighted_sample=True,
        random_seed=42,
        pbar_disabled=True
    )
    
    # Check density to see which nodes were sources
    density = res.node_density[500]
    sampled_indices = [i for i, val in enumerate(density) if val > 0]
    
    # Get the indices of high weight nodes
    high_weight_indices = nodes_gdf[nodes_gdf['weight'] == 100.0]['ns_node_idx'].tolist()
    low_weight_indices = nodes_gdf[nodes_gdf['weight'] == 1.0]['ns_node_idx'].tolist()
    
    # We expect high weight nodes to be in sampled_indices
    for idx in high_weight_indices:
        assert idx in sampled_indices
    
    # Low weight index might or might not be sampled, but much less likely. 
    # With seed 42 and only 1 low weight node, it's deterministic.
    # Let's just assert that we successfully processed weights.

if __name__ == "__main__":
    test_centrality_sampling_speed()
    test_centrality_weighted_sampling()
    test_centrality_sampling_reproducibility()
    test_segment_weight_sampling()
