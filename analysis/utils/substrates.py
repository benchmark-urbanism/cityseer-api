"""
Substrate generators for creating realistic urban network patterns.

Provides synthetic street networks with controllable connectivity gradations,
useful for benchmarking and analysis where consistent spatial extents are needed.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from cityseer.tools import graphs
from pyproj import CRS
from shapely import geometry, ops

from utils import plot as tbox_plot


def make_coord(idx: int, units: int, block_length: float) -> int:
    if idx == 0:
        return 0
    elif idx < units:
        return int(idx * block_length)
    else:
        return int(idx * block_length - block_length / 2)


@dataclass
class Tiers:
    edge_map: list[
        list[int, int, int, int],
        list[int, int, int, int],
        list[int, int, int, int],
        list[int, int, int, int],
    ]


@dataclass
class Quadrant:
    x_tiers: Tiers
    y_tiers: Tiers

    def retrieve_tiers(self, x_dir: int, y_dir: int) -> [Tiers, Tiers]:
        """
        switch x and y tiers to match the current quadrant rotation
        keep in mind that x y directions are inflected in the iterations
        i.e. only swap x, y, don't reverse
        """
        if (x_dir, y_dir) in [(1, 1), (-1, -1)]:
            x_stack = self.x_tiers
            y_stack = self.y_tiers
        else:
            assert (x_dir, y_dir) in [(-1, 1), (1, -1)]
            x_stack = self.y_tiers
            y_stack = self.x_tiers
        return x_stack.edge_map, y_stack.edge_map


@dataclass
class Tiler:
    quadrant: Quadrant
    block_size: int

    def __init__(self, quadrant: Quadrant, block_size: int = 70):
        self.quadrant = quadrant
        self.block_size = block_size

    def node_coords(self, start_coord: int, dir: int) -> list[int]:
        """ """
        assert dir in [-1, 1]
        node_coords = [start_coord]
        # there are three full steps
        for step_count in range(1, 4):
            node_coords.append(start_coord + step_count * self.block_size * dir)
        # and one half step
        node_coords.append(start_coord + 3.5 * self.block_size * dir)
        return node_coords

    @staticmethod
    def generate_node_key(x_coord: int, y_coord: int) -> str:
        return f"{int(x_coord)}_{int(y_coord)}"

    def make_graph(self, decompose: int | None = None, weld_edges: bool = False) -> nx.MultiGraph:
        """
        Returns a graph consisting of four "outer" quadrants.
        Each outer quadrant consists of an identical "tile".
        Each tile consists of four inner quadrants.
        Each inner quadrant is rotated around the midpoint of the tile.
        Outer and inner quadrants are generated in an anti-clockwise order from the top-right quadrant.

        Parameters
        ----------
        decompose : int | None
            Maximum edge length for decomposition. None = no decomposition.
        weld_edges : bool
            If True, connect opposite edges to create a toroidal network (wraps around).
            If False (default), leave edges open for a realistic bounded network.
        """
        # each inner quadrant contains 3.5 block units in the x and y directions
        inner_span = 3.5 * self.block_size
        # each outer quadrant (tile) consists of four inner quadrants
        # i.e. each tile consists of two inner quadrants in the x and y directions, respectively
        outer_span = 2 * inner_span
        # prepare the graph
        G = nx.MultiGraph()
        G.graph["crs"] = CRS("EPSG:32630")  # UTM zone 30N (arbitrary but consistent)
        # the tile coordinate offsets are based on an anti-clockwise order from top-right quadrant
        outer_x_offsets = [True, False, False, True]
        outer_y_offsets = [True, True, False, False]
        # the inner quadrants rotate around the tile's centre point
        inner_xy_directions = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        # iterate the outer quadrants
        for is_outer_x_offset, is_outer_y_offset in zip(outer_x_offsets, outer_y_offsets, strict=False):
            outer_x_offset = outer_span if is_outer_x_offset else 0
            outer_y_offset = outer_span if is_outer_y_offset else 0
            # the inner quadrants rotate around the tile's centre point
            x_base = int(outer_x_offset + inner_span)
            y_base = int(outer_y_offset + inner_span)
            # iterate the inner quadrants (i.e. rotations)
            for x_dir, y_dir in inner_xy_directions:
                x_coords = self.node_coords(x_base, x_dir)
                y_coords = self.node_coords(y_base, y_dir)
                for x_coord in x_coords:
                    for y_coord in y_coords:
                        node_key = self.generate_node_key(x_coord, y_coord)
                        if node_key not in G:
                            G.add_node(node_key, x=x_coord, y=y_coord)
        # add the edges
        for is_outer_x_offset, is_outer_y_offset in zip(outer_x_offsets, outer_y_offsets, strict=False):
            outer_x_offset = outer_span if is_outer_x_offset else 0
            outer_y_offset = outer_span if is_outer_y_offset else 0
            # the inner quadrants rotate around the tile's centre point
            x_base = int(outer_x_offset + inner_span)
            y_base = int(outer_y_offset + inner_span)
            # iterate the inner quadrants (i.e. rotations)
            for x_dir, y_dir in inner_xy_directions:
                x_coords = self.node_coords(x_base, x_dir)
                y_coords = self.node_coords(y_base, y_dir)
                x_edge_tiers, y_edge_tiers = self.quadrant.retrieve_tiers(x_dir=x_dir, y_dir=y_dir)
                # add x direction edges
                for y_idx, y_coord in enumerate(y_coords[:-1]):
                    for x_idx in range(len(x_coords) - 1):
                        if x_edge_tiers[y_idx][x_idx]:
                            x_start_coord = x_coords[x_idx]
                            x_end_coord = x_coords[x_idx + 1]
                            start_node_key = self.generate_node_key(x_start_coord, y_coord)
                            end_node_key = self.generate_node_key(x_end_coord, y_coord)
                            if (start_node_key, end_node_key) not in G.edges():
                                G.add_edge(start_node_key, end_node_key)
                # add y direction edges
                for x_idx, x_coord in enumerate(x_coords[:-1]):
                    for y_idx in range(len(y_coords) - 1):
                        if y_edge_tiers[x_idx][y_idx]:
                            y_start_coord = y_coords[y_idx]
                            y_end_coord = y_coords[y_idx + 1]
                            start_node_key = self.generate_node_key(x_coord, y_start_coord)
                            end_node_key = self.generate_node_key(x_coord, y_end_coord)
                            if (start_node_key, end_node_key) not in G.edges():
                                G.add_edge(start_node_key, end_node_key)
        # remove redundant nodes
        G = graphs.nx_simple_geoms(G)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G, remove_disconnected=100)
        # decompose if requested (0 or None = no decomposition)
        if decompose and decompose > 0:
            G = graphs.nx_decompose(G, decompose)
        # weld endpoints
        labels = []
        coords = []
        for n, d in G.nodes(data=True):
            labels.append(n)
            coords.append([d["x"], d["y"]])
        labels = np.array(labels)
        coords = np.array(coords)
        # Optionally weld opposite edges to create a toroidal network
        if weld_edges:
            # weld endpoints in x direction
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            touches_left = np.where(coords[:, 0] == x_min)[0]
            touches_right = np.where(coords[:, 0] == x_max)[0]
            assert len(touches_left) == len(touches_right)
            n_edges_added = 0
            for left_n, (left_x, left_y) in zip(labels[touches_left], coords[touches_left], strict=False):
                for right_n, (right_x, right_y) in zip(labels[touches_right], coords[touches_right], strict=False):
                    if left_y == right_y:
                        geom = geometry.LineString([(left_x, left_y), (right_x, right_y)])
                        G.add_edge(left_n, right_n, geom=geom, imp_factor=0)
                        n_edges_added += 1
                        break
            assert n_edges_added == len(touches_left)
            # weld endpoints in y direction
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            touches_top = np.where(coords[:, 1] == y_min)[0]
            touches_bottom = np.where(coords[:, 1] == y_max)[0]
            assert len(touches_top) == len(touches_bottom)
            n_edges_added = 0
            for top_n, (top_x, top_y) in zip(labels[touches_top], coords[touches_top], strict=False):
                for bottom_n, (bottom_x, bottom_y) in zip(labels[touches_bottom], coords[touches_bottom], strict=False):
                    if top_x == bottom_x:
                        geom = geometry.LineString([(top_x, top_y), (bottom_x, bottom_y)])
                        G.add_edge(top_n, bottom_n, geom=geom, imp_factor=0)
                        n_edges_added += 1
                        break
            assert n_edges_added == len(touches_top)

        return G


def generate_centroids(
    graph: nx.MultiGraph,
    block_size: int = 70,
    street_width: int = 10,
    street_frontage: int = 5,
    sidewalk_width: int = 5,
    setback_distance: int = 0,
) -> tuple[list[geometry.Point], list[geometry.Point]]:
    """
    Generates the individual plots of lands.
    Each plot should have access to a street.
    The average plot size should match the intended level of granularity.
    """
    # gather linestrings
    street_geoms = []
    for _s, _e, d in graph.edges(data=True):
        # skip welded edges
        if "imp_factor" in d and d["imp_factor"] == 0:
            continue
        street_geoms.append(d["geom"])
    # combine all line geoms into a MultiLineString
    street_multi_geoms = geometry.MultiLineString(street_geoms)
    # buffer the line geoms to generate the streets - use a "flat" cap style
    street_multi_geoms = street_multi_geoms.buffer(street_width / 2, cap_style=2)
    span = 14 * block_size
    bounds = geometry.box(0, 0, span, span)
    blocks = bounds.difference(street_multi_geoms)
    inset_buffer_dist = -(sidewalk_width + setback_distance)
    # gather residential centroids
    residential_centroids = []
    # reverse buffer each block
    for block_geom in blocks.geoms:
        # use double inset for residential centroids
        inset_block = block_geom.buffer(inset_buffer_dist * 2)
        # also reverse buffer the bounds to check for overlaps
        inset_bounds = bounds.buffer(inset_buffer_dist * 2)
        # extract the exterior ring and iterate each line
        inset_ext_ring_coords = list(
            zip(inset_block.exterior.coords.xy[0], inset_block.exterior.coords.xy[1], strict=False)
        )
        for c_idx in range(len(inset_ext_ring_coords) - 1):
            start_coord = inset_ext_ring_coords[c_idx]
            end_coord = inset_ext_ring_coords[c_idx + 1]
            setback_line = geometry.LineString([start_coord, end_coord])
            # ignore the line if it overlaps the bounds' exterior linear ring or if it is too short
            if inset_bounds.exterior.contains(setback_line) or setback_line.length < street_frontage:
                continue
            # otherwise divide through the number of locations
            # use twice shop frontage interval for residential
            divisions = int(np.ceil(setback_line.length / street_frontage) / 2)
            step_size = setback_line.length / divisions
            for div_idx in range(divisions):
                start_distance = div_idx * step_size
                end_distance = (div_idx + 1) * step_size
                unit_frontage = ops.substring(setback_line, start_distance, end_distance)
                residential_centroids.append(unit_frontage.centroid)
    # gather retail centroids
    retail_centroids = []
    # reverse buffer each block
    for block_geom in blocks.geoms:
        # generate the sidewalk and setback insets
        inset_buffer_dist = -(sidewalk_width + setback_distance)
        inset_block = block_geom.buffer(inset_buffer_dist)
        # also reverse buffer the bounds to check for overlaps
        inset_bounds = bounds.buffer(inset_buffer_dist)
        # extract the exterior ring and iterate each line
        inset_ext_ring_coords = list(
            zip(inset_block.exterior.coords.xy[0], inset_block.exterior.coords.xy[1], strict=False)
        )
        for c_idx in range(len(inset_ext_ring_coords) - 1):
            start_coord = inset_ext_ring_coords[c_idx]
            end_coord = inset_ext_ring_coords[c_idx + 1]
            setback_line = geometry.LineString([start_coord, end_coord])
            # ignore the line if it overlaps the bounds' exterior linear ring or if it is too short
            if inset_bounds.exterior.contains(setback_line) or setback_line.length < street_frontage:
                continue
            # otherwise divide through the number of locations
            divisions = int(np.ceil(setback_line.length / street_frontage))
            step_size = setback_line.length / divisions
            for div_idx in range(divisions):
                start_distance = div_idx * step_size
                end_distance = (div_idx + 1) * step_size
                unit_frontage = ops.substring(setback_line, start_distance, end_distance)
                retail_centroids.append(unit_frontage.centroid)
    return residential_centroids, retail_centroids


quadrant_templates = {
    "trellis": Quadrant(
        x_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        y_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
    ),
    "tree": Quadrant(
        x_tiers=Tiers([[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]),
        y_tiers=Tiers([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]),
    ),
    "neighbourhood": Quadrant(
        x_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]),
        y_tiers=Tiers([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]),
    ),
    "linear": Quadrant(
        x_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]),
        y_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]]),
    ),
    "islands": Quadrant(
        x_tiers=Tiers([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]),
        y_tiers=Tiers([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]),
    ),
}


def generate_quadrant(
    quad_tile: Quadrant,
    decompose: int | None = None,
    weld_edges: bool = False,
    plot: bool = False,
) -> tuple[nx.MultiGraph, list[geometry.Point], list[geometry.Point]]:
    """
    Generate a single quadrant substrate.

    Parameters
    ----------
    quad_tile : Quadrant
        The quadrant configuration defining edge connectivity
    decompose : int | None
        Maximum edge length for decomposition. None = no decomposition.
    weld_edges : bool
        If True, connect opposite edges to create a toroidal network.
        If False (default), leave edges open for a realistic bounded network.
    plot : bool
        Whether to display the generated network
    """
    tiler = Tiler(quadrant=quad_tile)
    # prepare graph
    G = tiler.make_graph(decompose=decompose, weld_edges=weld_edges)
    # generate centroids
    residential_centroids, retail_centroids = generate_centroids(G)
    # plot
    if plot:
        tbox_plot.plot_substrate(G, residential_centroids, retail_centroids, figsize=(10, 10), dpi=200)
    return G, residential_centroids, retail_centroids


def generate_keyed_template(
    template_key: str,
    tiles: int = 1,
    decompose: int | None = None,
    weld_edges: bool = False,
    plot: bool = False,
) -> tuple[nx.MultiGraph, list[geometry.Point], list[geometry.Point]]:
    """
    Generate a substrate network from a named template.

    Parameters
    ----------
    template_key : str
        One of: trellis, tree, neighbourhood, linear, islands
    tiles : int
        Number of tiles in each direction (1 = ~980m, 2 = ~1960m, 3 = ~2940m extent)
    decompose : int | None
        Maximum edge length for decomposition. None = no decomposition.
    weld_edges : bool
        If True, connect opposite edges to create a toroidal network.
        If False (default), leave edges open for a realistic bounded network.
    plot : bool
        Whether to display the generated network
    """
    if template_key not in quadrant_templates:
        raise ValueError(
            f"Template key {template_key} is not in the available list of templates. "
            f"Please select one of {', '.join(quadrant_templates.keys())}."
        )
    quad_tile = quadrant_templates[template_key]
    G, residential_centroids, retail_centroids = generate_quadrant(
        quad_tile=quad_tile, decompose=decompose, weld_edges=weld_edges, plot=False
    )

    # Tile if requested
    if tiles > 1:
        G = tile_graph(G, tiles_x=tiles, tiles_y=tiles)
        # Note: centroids are not tiled (not needed for sampling analysis)
        residential_centroids = []
        retail_centroids = []

    if plot:
        tbox_plot.plot_substrate(G, residential_centroids, retail_centroids, figsize=(10, 10), dpi=200)

    return G, residential_centroids, retail_centroids


def tile_graph(base_graph: nx.MultiGraph, tiles_x: int = 1, tiles_y: int = 1) -> nx.MultiGraph:
    """
    Tile a graph in a grid pattern to create a larger network.

    Parameters
    ----------
    base_graph : nx.MultiGraph
        The base graph to tile (should have x, y node attributes and geom edge attributes)
    tiles_x : int
        Number of tiles in x direction
    tiles_y : int
        Number of tiles in y direction

    Returns
    -------
    nx.MultiGraph
        Combined tiled graph with merged nodes at tile boundaries
    """
    if tiles_x == 1 and tiles_y == 1:
        return base_graph

    # Get base graph extent
    xs = [base_graph.nodes[n]["x"] for n in base_graph.nodes()]
    ys = [base_graph.nodes[n]["y"] for n in base_graph.nodes()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min

    # Create combined graph
    G = nx.MultiGraph()
    G.graph["crs"] = base_graph.graph.get("crs", CRS("EPSG:32630"))

    # Use position-based node keys to automatically merge boundary nodes
    # Map from (tile, original_node) -> position-based key
    def pos_key(x: float, y: float) -> str:
        """Generate node key from position (handles boundary merging)."""
        return f"{int(round(x))}_{int(round(y))}"

    # First pass: add all nodes (duplicates at boundaries will be same key)
    for tx in range(tiles_x):
        for ty in range(tiles_y):
            x_offset = tx * width
            y_offset = ty * height
            for _n, d in base_graph.nodes(data=True):
                new_x = d["x"] + x_offset
                new_y = d["y"] + y_offset
                new_key = pos_key(new_x, new_y)
                if new_key not in G:
                    G.add_node(new_key, x=new_x, y=new_y)

    # Second pass: add edges with correct node references
    for tx in range(tiles_x):
        for ty in range(tiles_y):
            x_offset = tx * width
            y_offset = ty * height
            for u, v, d in base_graph.edges(data=True):
                # Get positions of endpoints
                u_x = base_graph.nodes[u]["x"] + x_offset
                u_y = base_graph.nodes[u]["y"] + y_offset
                v_x = base_graph.nodes[v]["x"] + x_offset
                v_y = base_graph.nodes[v]["y"] + y_offset
                # Map to position-based keys
                new_u = pos_key(u_x, u_y)
                new_v = pos_key(v_x, v_y)
                # Copy edge data with offset geometry
                new_d = d.copy()
                if "geom" in new_d:
                    coords = [(x + x_offset, y + y_offset) for x, y in new_d["geom"].coords]
                    new_d["geom"] = geometry.LineString(coords)
                G.add_edge(new_u, new_v, **new_d)

    # Remove degree-2 filler nodes created at tile boundaries
    G = graphs.nx_remove_filler_nodes(G)

    return G


def generate_gradated_template(
    gradation: int,
    tiles: int = 1,
    decompose: int | None = None,
    weld_edges: bool = False,
    plot: bool = False,
) -> tuple[nx.MultiGraph, list[geometry.Point], list[geometry.Point]]:
    """
    Generate a substrate network with controllable connectivity.

    Parameters
    ----------
    gradation : int
        Connectivity level from 1 (sparse/tree-like) to 10 (dense/grid-like)
    tiles : int
        Number of tiles in each direction (1 = ~980m, 2 = ~1960m, 3 = ~2940m extent)
    decompose : int | None
        Maximum edge length for decomposition. None = no decomposition.
    weld_edges : bool
        If True, connect opposite edges to create a toroidal network (for periodic boundary).
        If False (default), leave edges open for a realistic bounded network.
    plot : bool
        Whether to display the generated network

    Returns
    -------
    tuple[nx.MultiGraph, list[Point], list[Point]]
        Graph, residential centroids, retail centroids
    """
    valid_gradations = list(range(1, 11))
    if gradation not in valid_gradations:
        raise ValueError(f"Gradation must be one of {valid_gradations}")
    x_tier_map = np.array([[1, 1, 1, 1], [4, 1, 1, 6], [2, 1, 1, 3], [5, 1, 1, 7]])
    y_tier_map = np.array([[1, 1, 1, 1], [8, 9, 10, 6], [1, 1, 1, 3], [10, 9, 8, 7]])
    # x map
    x_tier_bools = np.zeros_like(x_tier_map)
    x_tier_bools[x_tier_map > gradation] = 0
    x_tier_bools[x_tier_map <= gradation] = 1
    # y map
    y_tier_bools = np.zeros_like(y_tier_map)
    y_tier_bools[y_tier_map > gradation] = 0
    y_tier_bools[y_tier_map <= gradation] = 1
    # generate base quadrant
    quad_tile = Quadrant(x_tiers=Tiers(x_tier_bools.tolist()), y_tiers=Tiers(y_tier_bools.tolist()))
    G, residential_centroids, retail_centroids = generate_quadrant(
        quad_tile=quad_tile, decompose=decompose, weld_edges=weld_edges, plot=False
    )

    # Tile if requested
    if tiles > 1:
        G = tile_graph(G, tiles_x=tiles, tiles_y=tiles)
        # Note: centroids are not tiled (not needed for sampling analysis)
        residential_centroids = []
        retail_centroids = []

    if plot:
        tbox_plot.plot_substrate(G, residential_centroids, retail_centroids, figsize=(10, 10), dpi=200)

    return G, residential_centroids, retail_centroids
