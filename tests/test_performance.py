# %%
from __future__ import annotations

import os
import timeit

from cityseer.algos import centrality
from cityseer.metrics import networks
from cityseer.tools import graphs


def test_local_centrality_time(primal_graph):
    """
    Keep in mind there are several extraneous variables:
    e.g. may be fairly dramatic differences in timing on larger graphs and larger search distances

    originally based on node_harmonic and node_betweenness:
    OLD VERSION with trim maps:
    Timing: 10.490865555 for 10000 iters
    version with numba typed list - faster and removes arcane full vs. trim maps workflow
    8.24 for 10000 iters
    version with node_edge_map Dict - tad slower but worthwhile for cleaner and more intuitive code
    8.88 for 10000 iters
    version with shortest path tree algo simplified to nodes and non-angular only
    8.19 for 10000 iters
    version with jitclasses and float32
    <7 for 10000 iters
    without jitclasses again
    5.2 for 10000 iters
    without proto funcs (cacheing)
    5.15

    notes:
    - Segments of unreachable code used to add to timing: this seems to have been fixed in more recent versions of numba
    - Separating the logic into functions results in ever so slightly slower times...
      though this may be due to function setup at invocation (x10000) which wouldn't be incurred in real scenarios...?
    - Tests on using a List(Dict('x', 'y', etc.) structure proved almost four times slower, so sticking with arrays
    - Experiments with golang proved too complex re: bindings...
    """

    if "GITHUB_ACTIONS" in os.environ:
        return
    os.environ["CITYSEER_QUIET_MODE"] = "1"
    # load the test graph
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    # needs a large enough beta so that distance thresholds aren't encountered
    distances, betas = networks.pair_distances_betas(distances=[5000])

    def shortest_path_tree_wrapper():
        centrality.shortest_path_tree(
            network_structure.edges.start,
            network_structure.edges.end,
            network_structure.edges.length,
            network_structure.edges.angle_sum,
            network_structure.edges.imp_factor,
            network_structure.edges.in_bearing,
            network_structure.edges.out_bearing,
            network_structure.node_edge_map,
            0,
            max(distances),
            angular=False,
        )

    # prime the function
    shortest_path_tree_wrapper()
    iters = 100000
    # time and report
    func_time = timeit.timeit(shortest_path_tree_wrapper, number=iters)
    print(f"shortest_path_tree_wrapper: {func_time} for {iters} iterations")
    assert func_time < 5

    def node_cent_wrapper():
        centrality.local_node_centrality(
            distances,
            betas,
            ("node_harmonic", "node_betweenness"),
            network_structure.nodes.live,
            network_structure.edges.start,
            network_structure.edges.end,
            network_structure.edges.length,
            network_structure.edges.angle_sum,
            network_structure.edges.imp_factor,
            network_structure.edges.in_bearing,
            network_structure.edges.out_bearing,
            network_structure.node_edge_map,
            angular=False,
            progress_proxy=None,
        )

    # prime the function
    node_cent_wrapper()
    iters = 10000
    # time and report
    func_time = timeit.timeit(node_cent_wrapper, number=iters)
    print(f"node_cent_wrapper: {func_time} for {iters} iterations")
    assert func_time < 7

    def segment_cent_wrapper():
        centrality.local_segment_centrality(
            distances,
            betas,
            ("segment_harmonic", "segment_betweenness"),
            network_structure.nodes.live,
            network_structure.edges.start,
            network_structure.edges.end,
            network_structure.edges.length,
            network_structure.edges.angle_sum,
            network_structure.edges.imp_factor,
            network_structure.edges.in_bearing,
            network_structure.edges.out_bearing,
            network_structure.node_edge_map,
            angular=False,
            progress_proxy=None,
        )

    # prime the function
    segment_cent_wrapper()
    iters = 10000
    # time and report - roughly 9.36s on 4.2GHz i7
    func_time = timeit.timeit(segment_cent_wrapper, number=iters)
    print(f"segment_cent_wrapper: {func_time} for {iters} iterations")
    assert func_time < 9


if __name__ == "__main__":
    from cityseer.tools import graphs
    from cityseer.tools.mock import mock_graph

    G_primal = mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    test_local_centrality_time(G_primal)
