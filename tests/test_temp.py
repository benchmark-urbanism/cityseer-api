from cityseer.metrics import networks
from cityseer.tools import graphs, mock

G_primal = mock.mock_graph()
G_primal = graphs.nx_simple_geoms(G_primal)
# load the test graph
_nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(G_primal, 3395)
# needs a large enough beta so that distance thresholds aren't encountered
distances, betas = networks.pair_distances_betas(distances=[5000])

close_short = network_structure.local_node_centrality_shortest(
    distances,
    betas,
    True,
)
print("here")
