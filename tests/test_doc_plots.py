'''
This is not a test: plots doc-related images when tests are run
'''

import matplotlib.pyplot as plt

from cityseer.util import mock, graphs, plot


def test_make_doc_plots():
    G = mock.mock_graph()
    plot.plot_nX(G, path='../docs/simple_graph.png', labels=True)

    plt.clf()
    plot.plot_nX(G, path='../docs/util/graph.png', labels=False)

    G_simple = graphs.nX_simple_geoms(G)
    G_decomposed = graphs.nX_decompose(G_simple, 100)
    plt.clf()
    plot.plot_nX(G_decomposed, path='../docs/util/graph_decomposed.png', labels=False)
