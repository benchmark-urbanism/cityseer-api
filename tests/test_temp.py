import numpy as np

from cityseer.tools import graphs


def test_network_structure_from_nx(diamond_graph):
    import os

    os.environ["NUMBA_DEBUGINFO"] = "1"
    # test maps vs. networkX
    G_test = diamond_graph.copy()
    G_test_dual = graphs.nx_to_dual(G_test)
    for G, is_dual in zip((G_test, G_test_dual), (False, True)):
        # set some random 'live' statuses
        for n in G.nodes():
            G.nodes[n]["live"] = bool(np.random.randint(0, 1))
        # generate test maps
        network_structure = graphs.network_structure_from_nx(G)
