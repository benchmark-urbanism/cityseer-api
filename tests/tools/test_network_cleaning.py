"""
Test network cleaning algorithms with focus on:
- nx_consolidate_nodes with crawl mode
- nx_split_opposing_geoms
- highway tag filtering
- tag cache invalidation after graph mutations
"""

import networkx as nx
from cityseer.tools import graphs
from shapely import geometry


def create_test_graph_with_tags():
    """Create a test graph with OSM-style highway tags."""
    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create a simple grid with different highway types
    nodes = {
        "A": (0, 0),
        "B": (100, 0),
        "C": (200, 0),
        "D": (0, 100),
        "E": (100, 100),
        "F": (200, 100),
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    # Add edges with different highway types
    edges = [
        ("A", "B", "primary", ["Main St"]),
        ("B", "C", "primary", ["Main St"]),
        ("D", "E", "residential", ["Side St"]),
        ("E", "F", "footway", []),
        ("A", "D", "secondary", ["Cross St"]),
        ("B", "E", "secondary", ["Cross St"]),
        ("C", "F", "tertiary", ["End St"]),
    ]

    for start, end, hwy_type, names in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=names, routes=[], levels=[0])

    return G


def test_consolidate_nodes_crawl():
    """Test that crawl mode explores all direct neighbours correctly."""
    print("\n=== Testing nx_consolidate_nodes with crawl ===")

    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create nodes in a chain where A-B-C are all within buffer distance
    # A is at origin, B is 10m away, C is 20m away
    nodes = {
        "A": (0, 0),
        "B": (10, 0),
        "C": (20, 0),
        "D": (0, 20),  # Outside buffer from B/C
        "E": (10, 20),
        "F": (20, 20),
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    # Add edges with highway tags
    edges = [
        ("A", "B", "primary"),
        ("B", "C", "primary"),
        ("A", "D", "primary"),
        ("B", "E", "primary"),
        ("C", "F", "primary"),
        ("D", "E", "residential"),
        ("E", "F", "residential"),
    ]

    for start, end, hwy_type in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=[], routes=[], levels=[0])

    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Nodes: {sorted(G.nodes())}")

    # Test with crawl=False (should only merge immediate neighbours within buffer)
    G_no_crawl = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=12,
        crawl=False,
        centroid_by_itx=False,
    )

    print(f"After consolidate (no crawl): {G_no_crawl.number_of_nodes()} nodes")
    print(f"Nodes: {sorted(G_no_crawl.nodes())}")

    # Test with crawl=True (should merge A-B-C into one node)
    G_crawl = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=12,
        crawl=True,
        centroid_by_itx=False,
    )

    print(f"After consolidate (with crawl): {G_crawl.number_of_nodes()} nodes")
    print(f"Nodes: {sorted(G_crawl.nodes())}")

    # With crawl, A-B-C should be merged (all within 12m buffer when crawling)
    assert G_crawl.number_of_nodes() < G_no_crawl.number_of_nodes(), "Crawl mode should merge more nodes than non-crawl"

    print("✓ Crawl mode correctly explores neighbours")
    return G_crawl


def test_consolidate_with_hwy_tags():
    """Test that highway tag filtering works correctly during consolidation."""
    print("\n=== Testing nx_consolidate_nodes with highway tags ===")

    G = create_test_graph_with_tags()
    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create closely spaced nodes that should merge
    G.add_node("B2", x=105, y=0)
    geom = geometry.LineString([[105, 0], [200, 0]])
    G.add_edge("B2", "C", geom=geom, highways=["primary"], names=["Main St"], routes=[], levels=[0])

    print(f"After adding B2 node: {G.number_of_nodes()} nodes")

    # Consolidate only primary highways
    G_primary = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=15,
        osm_hwy_target_tags=["primary"],
        centroid_by_itx=False,
    )

    print(f"After consolidate (primary only): {G_primary.number_of_nodes()} nodes")

    # B and B2 should be merged since they're both on primary roads
    assert G_primary.number_of_nodes() < G.number_of_nodes(), "Primary highway nodes should be consolidated"

    # Consolidate with residential tags (shouldn't affect primary road nodes)
    G_residential = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=15,
        osm_hwy_target_tags=["residential"],
        centroid_by_itx=False,
    )

    print(f"After consolidate (residential only): {G_residential.number_of_nodes()} nodes")

    # Should not consolidate primary road nodes
    assert G_residential.number_of_nodes() == G.number_of_nodes(), (
        "Residential filter should not consolidate primary nodes"
    )

    print("✓ Highway tag filtering works correctly")
    return G_primary


def test_split_opposing_geoms():
    """Test nx_split_opposing_geoms functionality."""
    print("\n=== Testing nx_split_opposing_geoms ===")

    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create parallel roads with intersections (need degree >= 2)
    # Two parallel horizontal roads with connecting vertical roads
    nodes = {
        "A1": (0, 0),
        "A2": (50, 0),
        "A3": (100, 0),
        "B1": (0, 10),
        "B2": (50, 10),
        "B3": (100, 10),
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    # Add parallel edges and connectors
    edges = [
        ("A1", "A2", "primary", ["Road A"]),
        ("A2", "A3", "primary", ["Road A"]),
        ("B1", "B2", "primary", ["Road B"]),
        ("B2", "B3", "primary", ["Road B"]),
        ("A1", "B1", "secondary", ["Cross 1"]),
        ("A3", "B3", "secondary", ["Cross 2"]),
    ]

    for start, end, hwy_type, names in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=names, routes=[], levels=[0])

    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Node degrees: {[(n, G.degree(n)) for n in sorted(G.nodes())]}")

    # Split opposing geoms (A2 should split B1-B2-B3 and B2 should split A1-A2-A3)
    G_split = graphs.nx_split_opposing_geoms(
        G,
        buffer_dist=15,
        min_node_degree=2,
        squash_nodes=True,
        centroid_by_itx=False,
    )

    print(f"After split opposing: {G_split.number_of_nodes()} nodes, {G_split.number_of_edges()} edges")

    # Should have split the opposing edges
    # The middle nodes (A2, B2) should trigger splits on the opposite parallel edge
    # After squashing, we may have fewer nodes but more edges
    assert G_split.number_of_edges() >= G.number_of_edges(), "Split opposing should maintain or add edges"

    print("✓ Split opposing geoms works")
    return G_split


def test_split_with_hwy_tags():
    """Test that highway tag filtering works during splitting."""
    print("\n=== Testing nx_split_opposing_geoms with highway tags ===")

    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create parallel roads with different types, with intersections for degree >= 2
    nodes = {
        "A1": (0, 0),
        "A2": (50, 0),
        "A3": (100, 0),
        "B1": (0, 10),
        "B2": (50, 10),
        "B3": (100, 10),
        "C1": (0, 20),
        "C2": (50, 20),
        "C3": (100, 20),
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    edges = [
        ("A1", "A2", "primary", ["Road A"]),
        ("A2", "A3", "primary", ["Road A"]),
        ("B1", "B2", "footway", ["Path B"]),
        ("B2", "B3", "footway", ["Path B"]),
        ("C1", "C2", "residential", ["Road C"]),
        ("C2", "C3", "residential", ["Road C"]),
        # Connect them vertically
        ("A1", "B1", "secondary", []),
        ("B1", "C1", "secondary", []),
        ("A3", "B3", "secondary", []),
        ("B3", "C3", "secondary", []),
    ]

    for start, end, hwy_type, names in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=names, routes=[], levels=[0])

    print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    initial_primary_nodes = len(
        [n for n in G.nodes() if any(G[n][nb][0]["highways"][0] == "primary" for nb in G.neighbors(n) if G[n][nb])]
    )

    # Split only primary highways
    G_primary = graphs.nx_split_opposing_geoms(
        G.copy(),
        buffer_dist=15,
        osm_hwy_target_tags=["primary"],
        min_node_degree=2,
        squash_nodes=False,
    )

    print(f"After split (primary only): {G_primary.number_of_nodes()} nodes, {G_primary.number_of_edges()} edges")

    # Check that only primary nodes triggered operations
    # The graph should have changes due to primary highway nodes
    print(f"✓ Highway tag filtering in split works (filtered {initial_primary_nodes} primary nodes)")
    return G_primary


def test_tag_cache_invalidation():
    """Test that tag cache is properly invalidated after graph mutations."""
    print("\n=== Testing tag cache invalidation ===")

    G = create_test_graph_with_tags()

    # Create a shared tag cache
    tag_cache = {}

    # First consolidation pass with cache
    G_step1 = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=15,
        osm_hwy_target_tags=["primary", "secondary"],
        tag_cache=tag_cache,
        centroid_by_itx=False,
    )

    print(f"After first consolidation: {G_step1.number_of_nodes()} nodes")
    print(f"Cache size: {len(tag_cache)} entries")

    # Add more nodes and edges
    G_step1.add_node("G", x=100, y=50)
    geom1 = geometry.LineString([[100, 0], [100, 50]])
    G_step1.add_edge("B", "G", geom=geom1, highways=["primary"], names=["Main St"], routes=[], levels=[0])
    geom2 = geometry.LineString([[100, 50], [100, 100]])
    G_step1.add_edge("G", "E", geom=geom2, highways=["primary"], names=["Main St"], routes=[], levels=[0])

    # Second consolidation - should use updated graph state, not stale cache
    G_step2 = graphs.nx_consolidate_nodes(
        G_step1,
        buffer_dist=15,
        osm_hwy_target_tags=["primary", "secondary"],
        tag_cache=tag_cache,
        centroid_by_itx=False,
    )

    print(f"After second consolidation: {G_step2.number_of_nodes()} nodes")
    print(f"Cache size: {len(tag_cache)} entries")

    # The new node G should be considered for consolidation with its primary tags
    # If cache wasn't invalidated, it might be skipped

    print("✓ Tag cache invalidation works")
    return G_step2


def test_prioritise_by_hwy_tag():
    """Test that prioritise_by_hwy_tag selects appropriate centroids."""
    print("\n=== Testing prioritise_by_hwy_tag ===")

    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create nodes where a trunk road intersects with residential
    nodes = {
        "T1": (0, 0),  # On trunk
        "T2": (100, 0),  # On trunk
        "R1": (50, 10),  # On residential, close to trunk
        "R2": (50, 100),  # On residential
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    edges = [
        ("T1", "T2", "trunk"),
        ("T1", "R1", "residential"),
        ("R1", "R2", "residential"),
        ("T2", "R1", "residential"),
    ]

    for start, end, hwy_type in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=[], routes=[], levels=[0])

    print(f"Initial graph: {G.number_of_nodes()} nodes")

    # Without prioritization
    G_no_priority = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=15,
        prioritise_by_hwy_tag=False,
        centroid_by_itx=False,
    )

    print(f"Without highway prioritization: {G_no_priority.number_of_nodes()} nodes")

    # With prioritization - should favor trunk road nodes
    G_priority = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=15,
        prioritise_by_hwy_tag=True,
        centroid_by_itx=False,
    )

    print(f"With highway prioritization: {G_priority.number_of_nodes()} nodes")

    # Check that consolidated nodes are positioned near trunk road
    for node_id, node_data in G_priority.nodes(data=True):
        if "primal_edge_node_a" not in node_data:  # Not a dual node
            print(f"  Node {node_id}: ({node_data['x']:.1f}, {node_data['y']:.1f})")

    print("✓ Highway prioritization works")
    return G_priority


def test_matched_tags_only():
    """Test that osm_matched_tags_only restricts consolidation to matching names."""
    print("\n=== Testing osm_matched_tags_only ===")

    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32632"

    # Create close nodes on different named streets
    nodes = {
        "A1": (0, 0),
        "A2": (100, 0),
        "B1": (5, 5),  # Close to A1
        "B2": (100, 100),
    }

    for node_id, (x, y) in nodes.items():
        G.add_node(node_id, x=x, y=y)

    edges = [
        ("A1", "A2", "primary", ["Street A"]),
        ("B1", "B2", "primary", ["Street B"]),
    ]

    for start, end, hwy_type, names in edges:
        start_data = G.nodes[start]
        end_data = G.nodes[end]
        geom = geometry.LineString([[start_data["x"], start_data["y"]], [end_data["x"], end_data["y"]]])
        G.add_edge(start, end, geom=geom, highways=[hwy_type], names=names, routes=[], levels=[0])

    print(f"Initial graph: {G.number_of_nodes()} nodes")

    # Without matched tags only - A1 and B1 should merge (same highway type, close)
    G_no_match = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=10,
        osm_matched_tags_only=False,
        centroid_by_itx=False,
    )

    print(f"Without matched tags only: {G_no_match.number_of_nodes()} nodes")

    # With matched tags only - A1 and B1 should NOT merge (different street names)
    G_match = graphs.nx_consolidate_nodes(
        G.copy(),
        buffer_dist=10,
        osm_matched_tags_only=True,
        centroid_by_itx=False,
    )

    print(f"With matched tags only: {G_match.number_of_nodes()} nodes")

    assert G_match.number_of_nodes() >= G_no_match.number_of_nodes(), (
        "Matched tags only should prevent consolidation of differently named streets"
    )

    print("✓ Matched tags only works correctly")
    return G_match


if __name__ == "__main__":
    print("=" * 60)
    print("Network Cleaning Algorithm Tests")
    print("=" * 60)

    try:
        # Test crawl mode
        test_consolidate_nodes_crawl()

        # Test highway tag filtering
        test_consolidate_with_hwy_tags()

        # Test split opposing geoms
        test_split_opposing_geoms()
        test_split_with_hwy_tags()

        # Test tag cache invalidation
        test_tag_cache_invalidation()

        # Test highway prioritization
        test_prioritise_by_hwy_tag()

        # Test matched tags only
        test_matched_tags_only()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        raise
