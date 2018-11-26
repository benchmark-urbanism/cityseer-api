from cityseer.util import mock
import networkx as nx
import numpy as np


def test_mock_graph():

    G, pos = mock.mock_graph()
    G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)

    # debugging
    # plot.plot_graphs(primal=G)

    for g in [G, G_wgs]:
        assert g.number_of_nodes() == 46
        assert g.number_of_edges() == 69
        assert nx.average_degree_connectivity(g) == { 3: 3.0 }
        assert nx.average_shortest_path_length(g) == 4.356521739130435

        for n, d in g.nodes(data=True):
            assert 'x' in d and isinstance(d['y'], (int, float))
            assert 'y' in d and isinstance(d['y'], (int, float))


def test_mock_data():

    G, pos = mock.mock_graph()
    data = mock.mock_data(G)

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for n, d in G.nodes(data=True):
        if d['x'] < min_x:
            min_x = d['x']
        if d['x'] > max_x:
            max_x = d['x']
        if d['y'] < min_y:
            min_y = d['y']
        if d['y'] > max_y:
            max_y = d['y']

    for v in data.values():
        # check that attributes are present
        assert 'x' in v and isinstance(v['y'], (int, float))
        assert 'y' in v and isinstance(v['y'], (int, float))
        assert 'live' in v and isinstance(v['live'], bool)
        assert 'class' in v and isinstance(v['class'], str)
        assert v['x'] >= min_x and v['x'] <= max_x
        assert v['y'] >= min_y and v['y'] <= max_y


def test_mock_species_diversity():

    for counts, probs in mock.mock_species_diversity():
        assert np.array_equal(counts / counts.sum(), probs)
        assert round(probs.sum(), 8) == 1


def test_mock_landuse_classifications():

    classes, distances = mock.mock_landuse_classifications()
    assert len(classes) == len(distances)
