import sys

sys.path.insert(0, "pysrc")
import cProfile
import pstats

from cityseer.tools import io

LNG = -0.1270
LAT = 51.5195
BUFFER = 1000

poly_wgs, _ = io.buffered_point_poly(LNG, LAT, BUFFER)

# Profile the cleaning function
profiler = cProfile.Profile()
profiler.enable()

# This will fetch OSM data and clean - use a small area to avoid long waits
try:
    G = io.osm_graph_from_poly(poly_wgs, simplify=True)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
except Exception as e:
    print(f"Error: {e}")

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats("tottime").print_stats("pysrc/cityseer", 20)
