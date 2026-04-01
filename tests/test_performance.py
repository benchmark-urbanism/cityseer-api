# %%
from __future__ import annotations

import os
import timeit

from cityseer import config, rustalgos
from cityseer.tools import graphs, io


def test_local_centrality_time(primal_graph):
    """
    NOTE - rust built in development mode will be slow - build via PDM install instead

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
    computing all closeness and with rust
    ~4.58 for 10000 iterations with hashmap metrics
    ~4.09 for 10000 iterations with vec metrics
    ~3.72 for 10000 iterations for single closeness vs all five
    ~2.16 for 10000 iterations with vecs instead of hashmaps in closest path tree
    ~2.14 for 10000 iterations with vecs converted to numpy
    ~3.05 for 10000 iterations with both closeness and betweenness

    notes:
    - Segments of unreachable code used to add to timing: this seems to have been fixed in more recent versions of numba
    - Separating the logic into functions results in ever so slightly slower times...
      though this may be due to function setup at invocation (x10000) which wouldn't be incurred in real scenarios...?
    - Tests on using a List(Dict('x', 'y', etc.) structure proved almost four times slower, so sticking with arrays
    - Experiments with golang proved too complex re: bindings...
    - Ended up with rust

    Rust
    shortest_path_tree_wrapper: 0.28707868605852127 for 10000 iterations
    node_cent_wrapper: 3.1882867829408497 for 10000 iterations
    segment_cent_wrapper: 5.971783181885257 for 10000 iterations

    Heap
    dijkstra_tree_shortest_wrapper: 0.032350792083889246 for 10000 iterations
    dijkstra_tree_simplest_wrapper: 0.2574775000102818 for 10000 iterations
    node_cent_wrapper: 2.880786875030026 for 10000 iterations
    """

    if "GITHUB_ACTIONS" in os.environ:
        return
    os.environ["CITYSEER_QUIET_MODE"] = "1"
    # load the test graph
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    distances, _seconds = rustalgos.pair_distances_and_time(config.SPEED_M_S, distances=[5000])

    speed_m_s = 1.3333
    max_seconds = int(5000 / speed_m_s)

    def dijkstra_tree_shortest_wrapper():
        network_structure.dijkstra_tree_shortest(
            src_idx=0,
            max_seconds=max_seconds,
            speed_m_s=speed_m_s,
        )

    # prime the function
    dijkstra_tree_shortest_wrapper()
    iters = 10000
    # time and report
    func_time = timeit.timeit(dijkstra_tree_shortest_wrapper, number=iters)
    print(f"dijkstra_tree_shortest_wrapper: {func_time} for {iters} iterations")
    assert func_time < 1

    # simplest path operates on dual graph
    dual_graph = graphs.nx_to_dual(primal_graph)
    _dual_nodes_gdf, _dual_edges_gdf, dual_network_structure = io.network_structure_from_nx(dual_graph)

    def dijkstra_tree_simplest_wrapper():
        dual_network_structure.dijkstra_tree_simplest(
            src_idx=0,
            max_seconds=max_seconds,
            speed_m_s=speed_m_s,
        )

    # prime the function
    dijkstra_tree_simplest_wrapper()
    iters = 10000
    # time and report
    func_time = timeit.timeit(dijkstra_tree_simplest_wrapper, number=iters)
    print(f"dijkstra_tree_simplest_wrapper: {func_time} for {iters} iterations")
    assert func_time < 1

    def node_cent_wrapper():
        network_structure.centrality_shortest(
            closeness_exprs=[("density", "1"), ("harmonic", "1/c")],
            betweenness_exprs=[],
            distances=distances,
            pbar_disabled=True,
        )

    # filtering by street node indices slows wrappers
    # prime the function
    node_cent_wrapper()
    iters = 10000
    # time and report
    func_time = timeit.timeit(node_cent_wrapper, number=iters)
    print(f"node_cent_wrapper: {func_time} for {iters} iterations")
    # assert func_time < 5
    # node_cent_wrapper: 3.5858502141200006 for 10000 iterations

    print("Done!")


if __name__ == "__main__":
    from cityseer.tools import graphs
    from cityseer.tools.mock import mock_graph

    G_primal = mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    test_local_centrality_time(G_primal)


# %%
