from cityseer import util
import networkx as nx
import matplotlib.pyplot as plt


def test_generate_graph():

    G, pos = util.tutte_graph()
    G_wgs, pos_wgs = util.tutte_graph(wgs84_coords=True)

    #nx.draw(G, pos=pos, with_labels=True)
    #plt.show()

    for g in [G, G_wgs]:

        assert g.number_of_nodes() == 46
        assert g.number_of_edges() == 69

        assert nx.average_degree_connectivity(g) == { 3: 3.0 }

        assert nx.average_shortest_path_length(g) == 4.356521739130435