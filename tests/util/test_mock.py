import networkx as nx
import numpy as np
import pytest
import string

from cityseer.util import mock


def test_mock_graph():
    G = mock.mock_graph()
    G_wgs = mock.mock_graph(wgs84_coords=True)

    for g in [G, G_wgs]:
        assert g.number_of_nodes() == 56
        assert g.number_of_edges() == 77
        assert nx.average_degree_connectivity(g) == {
            4: 3.0,
            3: 3.0,
            2: 2.0,
            1: 2.0,
            0: 0
        }

        for n, d in g.nodes(data=True):
            assert 'x' in d and isinstance(d['y'], (int, float))
            assert 'y' in d and isinstance(d['y'], (int, float))

    # from cityseer.util import plot
    # plot.plot_nX(G)


def test_mock_data_dict():
    G = mock.mock_graph()
    data = mock.mock_data_dict(G)

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
        assert v['x'] >= min_x and v['x'] <= max_x
        assert v['y'] >= min_y and v['y'] <= max_y


def test_mock_categorical_data():
    cat_d = mock.mock_categorical_data(50)
    assert len(cat_d) == 50
    # classes are generated randomly from max number of classes
    # i.e. situations do exist where the number of classes will be less than the max permitted
    # use large enough max to reduce likelihood of this triggering issue for test
    assert len(set(cat_d)) == 10

    for c in cat_d:
        assert isinstance(c, str)
        assert c in string.ascii_lowercase

    cat_d = mock.mock_categorical_data(50, num_classes=3)
    assert len(set(cat_d)) == 3

    # test that an error is raised when requesting more than available max classes per asii_lowercase
    with pytest.raises(ValueError):
        mock.mock_categorical_data(50, num_classes=len(string.ascii_lowercase) + 1)


def test_mock_numerical_data():
    for length in [50, 100]:
        for num_arrs in range(1, 3):

            num_d = mock.mock_numerical_data(length=length, num_arrs=num_arrs)
            assert num_d.shape[0] == num_arrs
            assert num_d.shape[1] == length

            for arr in num_d:
                for n in arr:
                    assert isinstance(n, float)
                    assert 0 <= n <= 100000


def test_mock_species_data():
    for counts, probs in mock.mock_species_data():
        assert np.allclose(counts / counts.sum(), probs, atol=0.001, rtol=0)
        assert round(probs.sum(), 8) == 1


def test_mock_osm_data():
    pass
