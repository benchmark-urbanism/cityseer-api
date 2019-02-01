from os import path

import matplotlib.pyplot as plt

from cityseer.util import mock, graphs, plot

base_path = path.dirname(__file__)

G = mock.mock_graph()
plot.plot_nX(G, path='graph_simple.png', labels=True)

# mock module
plt.clf()
plot.plot_nX(G, path='util/graph_example.png', labels=True)

# graph module
plt.clf()
plot.plot_nX(G, path='util/graph_simple.png', labels=False)

G_simple = graphs.nX_simple_geoms(G)
G_decomposed = graphs.nX_decompose(G_simple, 100)

plt.clf()
plot.plot_nX(G_decomposed, path='util/graph_decomposed.png', labels=False)

plt.clf()
G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple, G_dual, 'util/graph_dual.png', labels=False)
