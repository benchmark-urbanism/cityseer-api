import pytest
import numpy as np
import networkx as nx
from cityseer import centrality
from . import generate_graph

def test_custom_decay_betas():

    assert centrality.custom_decay_betas(0.04) == (np.array([100.]), 0.01831563888873418)

    assert centrality.custom_decay_betas(0.0025) == (np.array([1600.]), 0.01831563888873418)

    arr, t_w = centrality.custom_decay_betas(0.04, threshold_weight=0.001)
    assert np.array_equal(arr.round(8), np.array([172.69388197]).round(8))
    assert t_w == 0.001

    arr, t_w = centrality.custom_decay_betas([0.04, 0.0025])
    assert np.array_equal(arr, np.array([100, 1600]))
    assert t_w == 0.01831563888873418

    with pytest.raises(ValueError):
        centrality.custom_decay_betas(-0.04)


def test_generate_graph():

    G, pos = generate_graph.tutte_graph()

    assert G.number_of_nodes() == 46

    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435
