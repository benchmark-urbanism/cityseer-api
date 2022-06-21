# pyright: basic
from __future__ import annotations

import os
import timeit

import numpy as np
import numpy.typing as npt

from cityseer.algos import centrality, data
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


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
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    # needs a large enough beta so that distance thresholds aren't encountered
    distances: npt.NDArray[np.int_] = np.array([np.inf], np.float32)
    betas = networks.beta_from_distance(distances)

    def node_cent_wrapper():
        centrality.local_node_centrality(
            network_structure,
            distances,
            betas,
            ("node_harmonic", "node_betweenness"),
            angular=False,
            progress_proxy=None,
        )

    # prime the function
    node_cent_wrapper()
    iters = 10000
    # time and report
    func_time = timeit.timeit(node_cent_wrapper, number=iters)
    print(f"node_cent_wrapper: {func_time} for {iters} iterations")
    assert func_time < 10

    def segment_cent_wrapper():
        centrality.local_segment_centrality(
            network_structure,
            distances,
            betas,
            ("segment_harmonic", "segment_betweenness"),
            angular=False,
            progress_proxy=None,
        )

    # prime the function
    segment_cent_wrapper()
    iters = 10000
    # time and report - roughly 9.36s on 4.2GHz i7
    func_time = timeit.timeit(segment_cent_wrapper, number=iters)
    print(f"segment_cent_wrapper: {func_time} for {iters} iterations")
    assert func_time < 10


def test_local_agg_time(primal_graph):
    """
    Timing tests for landuse and stats aggregations
    """
    if "GITHUB_ACTIONS" in os.environ:
        return
    os.environ["CITYSEER_QUIET_MODE"] = "1"

    # generate node and edge maps
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    # setup data
    data_dict = mock.mock_data_dict(primal_graph, random_seed=13)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, network_structure, 500)
    # needs a large enough beta so that distance thresholds aren't encountered
    distances = np.array([np.inf])
    betas = networks.beta_from_distance(distances)
    qs = np.array([0, 1, 2])
    mock_categorical = mock.mock_categorical_data(data_map.count)
    landuse_classes, landuse_encodings = layers.encode_categorical(mock_categorical)
    mock_numerical = mock.mock_numerical_data(len(data_dict), num_arrs=2, random_seed=0)

    def assign_wrapper():
        data.assign_to_network(data_map, network_structure, 500)

    # prime the function
    assign_wrapper()
    iters = 10000
    # time and report - roughly 5.675
    func_time = timeit.timeit(assign_wrapper, number=iters)
    print(f"assign_wrapper: {func_time} for {iters} iterations")
    assert func_time < 10

    def landuse_agg_wrapper():
        mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
            network_structure,
            data_map,
            distances,
            betas,
            mixed_use_hill_keys=np.array([0, 1]),
            landuse_encodings=landuse_encodings,
            qs=qs,
            angular=False,
        )

    # prime the function
    landuse_agg_wrapper()
    iters = 10000
    # time and report - roughly 10.10
    func_time = timeit.timeit(landuse_agg_wrapper, number=iters)
    print(f"node_cent_wrapper: {func_time} for {iters} iterations")
    assert func_time < 15

    def stats_agg_wrapper():
        # compute
        data.aggregate_stats(
            network_structure,
            data_map,
            distances,
            betas,
            numerical_arrays=mock_numerical,
            angular=False,
        )

    # prime the function
    stats_agg_wrapper()
    iters = 10000
    # time and report - roughly 4.96
    func_time = timeit.timeit(stats_agg_wrapper, number=iters)
    print(f"landuse_agg_wrapper: {func_time} for {iters} iterations")
    assert func_time < 10
