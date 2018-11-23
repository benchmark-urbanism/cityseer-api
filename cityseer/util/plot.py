import matplotlib.pyplot as plt

def plot_graph_maps(node_map, edge_map, geom=None):

    # the links are undirected and therefore duplicate per edge
    # use two axes to check each copy of links
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # set extents
    for ax in (ax1, ax2):
        ax.set_xlim(node_map[:, 0].min() - 100, node_map[:, 0].max() + 100)
        ax.set_ylim(node_map[:, 1].min() - 100, node_map[:, 1].max() + 100)

    # plot nodes
    ax1.scatter(node_map[:, 0], node_map[:, 1], s=7, c=node_map[:, 2])
    ax2.scatter(node_map[:, 0], node_map[:, 1], s=7, c=node_map[:, 2])

    # check for duplicate edges
    edges = set()

    # plot edges - requires iteration through maps
    for src_idx, src_data in enumerate(node_map):
        # get the starting edge index
        edge_idx = int(src_data[3])
        # iterate the neighbours
        # don't use while True because last node's index increment won't be caught
        while edge_idx < len(edge_map):
            # get the corresponding edge data
            edge_data = edge_map[edge_idx]
            # get the src node - this is to check that still within src edge - neighbour range
            fr_idx = edge_data[0]
            # break once all neighbours visited
            if fr_idx != src_idx:
                break
            # get the neighbour node's index
            to_idx = edge_data[1]
            # fetch the neighbour node's data
            nb_data = node_map[int(to_idx)]
            # check for duplicates
            k = str(sorted([fr_idx, to_idx]))
            if k not in edges:
                edges.add(k)
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey')
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey')
            edge_idx += 1

    if geom:
        ax1.plot(geom.exterior.coords.xy[0], geom.exterior.coords.xy[1])
        ax2.plot(geom.exterior.coords.xy[0], geom.exterior.coords.xy[1])

    plt.show()
