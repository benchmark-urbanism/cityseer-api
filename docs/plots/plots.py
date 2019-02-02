from os import path

import matplotlib.pyplot as plt
import numpy as np

from cityseer.util import mock, graphs, plot

plt.style.use('./matplotlibrc')

base_path = path.dirname(__file__)

G = mock.mock_graph()
plot.plot_nX(G, path='graph.png', labels=True)

# mock module
plt.clf()
plot.plot_nX(G, path='graph_example.png', labels=True)

# graph module
plt.clf()
plot.plot_nX(G, path='graph_simple.png', labels=False)

G_simple = graphs.nX_simple_geoms(G)
G_decomposed = graphs.nX_decompose(G_simple, 100)

plt.clf()
plot.plot_nX(G_decomposed, path='graph_decomposed.png', labels=False)

plt.clf()
G_dual = graphs.nX_to_dual(G_simple)
plot.plot_nX_primal_or_dual(G_simple, G_dual, 'graph_dual.png', labels=False)

# networks module
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

w_min = 0.01831563888873418
# set the betas
betas = []
for d_max in [200, 400, 800, 1600]:
    beta = np.log(w_min) / d_max
    distances_arr = []
    for d in range(0, d_max + 1):
        distances_arr.append(d)
    distances_arr = np.array(distances_arr)
    y_falloff = np.exp(beta * distances_arr)
    ax.plot(distances_arr, y_falloff, label=f'$-\\beta={round(beta, 4)}$')

# add w_min
plt.axhline(y=w_min, color='#eeeeee', ls='--', lw=0.5)
ax.text(10, 0.035, '$w_{min}$', color='#eeeeee')

ax.set_xticks([200, 400, 800, 1600])
ax.set_xticklabels(['$d_{max}=200$', '$d_{max}=400$', '$d_{max}=800$', '$d_{max}=1600$'])
ax.set_xlim([0, 1600])
ax.set_xlabel('Distance in Metres')
ax.set_ylim([0, 1.0])
ax.set_ylabel('Weighting')
ax.legend(loc='upper right', title='$exp(-\\beta \\cdot d[i,j])$')
plt.savefig('betas.png', dpi=150)
