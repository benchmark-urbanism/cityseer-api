'''
.. module:: centrality
   :synopsis: Cityseer centrality methods

.. moduleauthor:: Gareth Simons

'''

import logging
from typing import Union, Tuple
import utm
from shapely import geometry, ops
import networkx as nx
import numpy as np
from numba.pycc import CC
from numba import njit
# from . import networks, mixed_uses, accessibility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cc = CC('centrality')


def custom_decay_betas(beta:Union[float, list, np.ndarray], min_threshold_wt:float=0.01831563888873418) -> Tuple[np.ndarray, float]:
    '''
    A convenience function for mapping :math:`-\\beta` decay parameters to equivalent distance thresholds corresponding to a ``min_threshold_wt`` cutoff parameter.

    Parameters
    ----------
    beta
        The :math:`-\\beta` that you wish to convert to distance thresholds.
    min_threshold_wt
        The :math:`w_{min}` threshold.

    Returns
    -------
    betas
        A numpy array of effective :math:`d_{max}` distances.
    min_threshold_wt
        The corresponding :math:`w_{min}` threshold.

    Hint
    ----
    There is no need to use this function unless you wish to provide your own :math:`-\\beta` or ``min_threshold_wt`` parameters to the :meth:`cityseer.centrality.compute_centrality` method.

    Caution
    -------
    Remember to pass both the :math:`d_{max}` and :math:`w_{min}` to :meth:`cityseer.centrality.compute_centrality`.

    Notes
    -----
    The weighted variants of centrality, e.g. gravity or weighted betweenness, are computed using a negative exponential decay function of the form:

    .. math::
       weight = exp(-\\beta \\cdot distance)

    The strength of the decay is controlled by the :math:`-\\beta` parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
    For example, if :math:`-\\beta=0.005` represents a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at 13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At some point, it becomes futile to consider locations any farther away, so it is necessary to set a a minimum weight threshold :math:`w_{min}` corresponding to a maximum distance of :math:`d_{max}`.

    The :meth:`cityseer.centrality.compute_centrality` method computes the :math:`-\\beta` parameters automatically, using a default ``min_threshold_wt`` of :math:`w_{min}=0.01831563888873418`.

    .. math::

       \\beta = \\frac{log\\big(\\frac{1}{w_{min}}\\big)}{d_{max}}

    Therefore, :math:`-\\beta` weights corresponding to :math:`d_{max}` walking thresholds of 400m, 800m, and 1600m would give:

    .. table::
       :align: center

       ================= =================
        :math:`d_{max}`   :math:`-\\beta`
       ----------------- -----------------
              400m             -0.01
              800m             -0.005
              1600m            -0.0025
       ================= =================

    In reality, people may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context. If overriding the defaults, or to use a custom :math:`-\\beta` or a different :math:`w_{min}` threshold, then this function can be used to generate the effective :math:`d_{max}` values, which can then be passed to :meth:`cityseer.centrality.compute_centrality` along with the specified :math:`w_{min}`. For example, the following :math:`-\\beta` and :math:`w_{min}` thresholds yield these effective :math:`d_{max}` distances:

    .. table::
       :align: center

       ================= ================= =================
        :math:`-\\beta`   :math:`w_{min}`   :math:`d_{max}`
       ----------------- ----------------- -----------------
             -0.01            0.01              461m
             -0.005           0.01              921m
             -0.0025          0.01              1842m
       ================= ================= =================

    '''

    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]

    # check that the betas do not have leading negatives
    for b in beta:
        if b < 0:
            raise ValueError('Please provide the beta/s without the leading negative')

    # cast to numpy
    beta = np.array(beta)

    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / -beta, min_threshold_wt


def graph_from_networkx(network_x_graph:nx.Graph, wgs84_coords:bool=False, decompose:int=None, geom:geometry.Polygon=None) -> Tuple[np.ndarray, np.ndarray]:
    '''




    A convenience function for generating a ``node_map`` and ``edge_map`` from a `NetworkX <https://networkx.github.io/documentation/networkx-1.10/index.html>`_ undirected Graph, which can then be passed to :meth:`cityseer.centrality.compute_centrality`.

    Parameters
    ----------
    network_x_graph
        A NetworkX undirected ``Graph``. Requires node attributes ``x`` and ``y`` for spatial coordinates and accepts optional ``length`` and ``weight`` edge attributes. See notes.
    wgs84_coords
        Set to ``True`` if the ``x`` and ``y`` node attribute keys reference `WGS84 <https://epsg.io/4326>`_ lng, lat values instead of a projected coordinate system.
    decompose
        Generates a decomposed version of the graph wherein edges are broken into smaller sections no longer than the specified distance in metres.
    geom
        Shapely geometry defining the original area of interest. Recommended for avoidance of boundary roll-off in computed metrics.

    Returns
    -------
    node_map
        containing node data
    edge_map
        containing edge data

    Notes
    -----

    The node attributes ``x`` and ``y`` determine the spatial coordinates of the node, and should be in a suitable projected (flat) coordinate reference system in metres unless the ``wgs84_coords`` parameter is set to ``True``.

    The optional edge attribute ``length`` indicates the original edge length in metres. If not provided, lengths will be computed using crow-flies distances between either end of the edges.

    If provided, the optional edge attribute ``weight`` will be used by shortest path algorithms instead of distances in metres.

    Tip
    ---
    When calculating local network centralities, it is best-practise for the area of interest to have been buffered by a distance equal to the maximum distance threshold to be considered. This prevents misleading results arising due to a boundary roll-off effect. If provided, the ``geom`` geometry is used to identify nodes falling within the original non-buffered area of interest. Metrics will then only be computed for these nodes, thus avoiding roll-off effects and reducing frivolous computation. (The algorithms still have access to the full buffered network.)

    Caution
    -------
    Graph decomposition provides a more granular representation of variations along street lengths. However, setting the ``decompose`` parameter too small can increase the computation time unnecessarily for subsequent analysis. It is generally not necessary to go smaller :math:`20m`, and :math:`50m` may already be sufficient for many cases.

    '''

    # check that it is an undirected graph
    if not isinstance(network_x_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkx graph')

    # copy the graph to avoid modifying the original
    G = network_x_graph.copy()

    # decided to not discard disconnected components to avoid unintended consequences

    # if the coords are WGS84, then convert to local UTM
    if wgs84_coords:
        for n, d in G.nodes(data=True):
            # convert coords - flip from lat, lon order back to x, y order
            x, y = utm.from_latlon(d['y'], d['x'])[:2][::-1]
            G.node[n]['x'] = x
            G.node[n]['y'] = y

    if decompose:
        for s, e, d in G.edges(data=True):
            # get start coords
            s_x = G.node[s]['x']
            s_y = G.node[s]['y']
            # get end coords
            e_x = G.node[e]['x']
            e_y = G.node[e]['y']
            # generate the geometry
            g = geometry.LineString([[s_x, s_y], [e_x, e_y]])
            # write length to edge if it doesn't exist
            if 'length' not in d:
                G[s][e]['length'] = g.length
            # if both
            # see how many segments are necessary so as not to exceed decomposition max distance
            # note that a length less than the decompose threshold will result in a single 'sub'-string
            l = G[s][e]['length']
            n = np.ceil(l / decompose)
            # create the sub-links
            d_step = 0
            prior_node_id = s
            sub_node_counter = 0
            # everything inside this loop is a new node - i.e. this loop is effectively skipped if n = 1
            for i in range(int(n) - 1):
                # create the new node ID
                new_node_id = f'{s}_{sub_node_counter}_{e}'
                sub_node_counter += 1
                # create the split linestring geom for measuring the new length
                s_g = ops.substring(g, d_step, d_step + l / n)
                # get the x, y of the new end node
                x = s_g.coords.xy[0][-1]
                y = s_g.coords.xy[1][-1]
                # add the new node and edge
                G.add_node(new_node_id, x=x, y=y)
                G.add_edge(prior_node_id, new_node_id, length=s_g.length)
                # increment the step and node id
                prior_node_id = new_node_id
                d_step += l / n
            # set the last link manually to avoid rounding errors at end of linestring
            s_g = ops.substring(g, d_step, l)
            # nodes already exist, so just add link
            G.add_edge(prior_node_id, e, length=s_g.length)

    # convert the nodes to sequential - this permits implicit indices with benefits to speed and structure
    G = nx.convert_node_labels_to_integers(G, 0)

    # set lengths if missing, as may be case if decomposition is not triggered
    if not decompose:
        for s, e, d in G.edges(data=True):
            if 'length' not in d:
                # get start coords
                s_x = G.node[s]['x']
                s_y = G.node[s]['y']
                # get end coords
                e_x = G.node[e]['x']
                e_y = G.node[e]['y']
                # set length
                G[s][e]['length'] = geometry.Point(s_x, s_y).distance(geometry.Point(e_x, e_y))

    # set the live nodes and sum degrees
    total_out_degrees = 0
    for n, d in G.nodes(data=True):
        total_out_degrees += nx.degree(G, n)
        live = True
        if geom and not geom.contains(geometry.Point(d['x'], d['y'])):
            live = False
        G.node[n]['live'] = live

    # prepare the node and link maps
    n = G.number_of_nodes()
    node_map = np.full((n, 4), np.nan)
    edge_map = np.full((total_out_degrees, 3), np.nan)
    link_idx = 0
    # populate the nodes
    for n, d in G.nodes(data=True):
        idx = int(n)
        node_map[idx][0] = d['x']
        node_map[idx][1] = d['y']
        node_map[idx][2] = d['live']
        node_map[idx][3] = link_idx
        # follow all out links and add
        # this happens for both directions
        for nb in G.neighbors(n):
            # start node
            edge_map[link_idx][0] = idx
            # end node
            edge_map[link_idx][1] = int(nb)
            # length
            edge_map[link_idx][2] = G[idx][nb]['length']
            # increment the link_idx
            link_idx += 1

    return node_map, edge_map

"""
def centrality(node_map, edge_map, distances, min_threshold_wt=0.01831563888873418):
    '''

    :param node_map:
    :param edge_map:
    :param distances:
    :param min_threshold_wt:
    :return:
    '''

    if node_map.shape[0] != 4:
        raise ValueError('The node map must have a dimensionality of 4, consisting of x, y, live, and link idx parameters')

    if link_map.shape[0] < 3:
        raise ValueError('The link map must have a dimensionality of 3, consisting of start, end, and distance parameters')

    if isinstance(distances, (int, float)):
        distances = [distances]

    if not isinstance(distances, (list, np.ndarray)):
        raise ValueError('Please provide a distance or an array of distances')

    betas = []
    for d in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
        betas.append(np.log(1 / min_threshold_wt) / d)

    return compute_centrality(node_map, link_map, np.array(distances), np.array(betas))


# NOTE -> didn't work with boolean so using unsigned int...
@cc.export('compute_centrality',
           'Tuple((Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C")))'
           '(Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"))')
@njit
def compute_centrality(node_map, link_map, distances, betas):
    '''

    :param node_map:
    :param link_map:
    :return:
    '''


    # used for calculating a corresponding beta value
    y = 0.01831563888873418

    max_dist = 800
    # establish the number of nodes
    n = node_map.shape[1]
    # create the distance map
    d_map = np.full((len(distances), 2), np.nan)
    for i, d in enumerate(distances):
        d_map[i] = [d, ]

    # prepare data arrays
    closeness = np.full((4, n), 0.0)
    gravity = np.full((4, n), 0.0)
    betweenness_wt = np.full((4, n), 0.0)
    betweenness_wt = np.full((4, n), 0.0)






    # prepare data arrays
    gravity = np.zeros((4, total_count))
    betweenness_wt = np.zeros((4, total_count))
    mixed_uses_wt = np.zeros((4, total_count))
    pois = np.zeros((40, total_count))

    beta_100 = -0.04
    beta_200 = -0.02
    beta_400 = -0.01
    beta_800 = -0.005

    data_assign_map, data_assign_dist = networks.assign_accessibility_data(netw_x_arr, netw_y_arr, data_x_arr, data_y_arr, max_dist)

    # iterate through each vert and calculate the shortest path tree
    for netw_src_idx in range(total_count):

        #if netw_src_idx % 1000 == 0:
        #    print('...progress')
        #    print(round(netw_src_idx / total_count * 100, 2))

        # only compute for nodes in current city
        if not hot_node[netw_src_idx]:
            continue

        netw_src_idx_trim, netw_trim_count, netw_idx_map_trim_to_full, nbs_trim, lens_trim = \
            networks.graph_window(netw_src_idx, max_dist, netw_x_arr, netw_y_arr, nbs, lens)

        # use np.inf for max distance for data POI mapping, which uses an overshoot and backtracking workflow
        # not a huge penalty because graph is already windowed per above
        netw_dist_map_trim, netw_pred_map_trim = networks.shortest_path_tree(nbs_trim, lens_trim, netw_src_idx_trim, netw_trim_count, np.inf)

        # calculate mixed uses
        # generate the reachable classes and their respective distances
        reachable_classes, reachable_classes_dist, data_trim_to_full_idx_map = networks.accessibility_agg(netw_src_idx, max_dist,
            netw_dist_map_trim, netw_pred_map_trim, netw_idx_map_trim_to_full, netw_x_arr, netw_y_arr, data_classes,
                data_x_arr, data_y_arr, data_assign_map, data_assign_dist)

        # get unique classes, their counts, and nearest - use the default max distance of 1600m
        classes_unique, classes_counts, classes_nearest = mixed_uses.deduce_unique_species(reachable_classes, reachable_classes_dist)

        # compute mixed uses
        mixed_uses_wt[0][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_100 * classes_nearest), 0)
        mixed_uses_wt[1][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_200 * classes_nearest), 0)
        mixed_uses_wt[2][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_400 * classes_nearest), 0)
        mixed_uses_wt[3][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_800 * classes_nearest), 0)

        # compute accessibilities
        # reachable is same as for mixed uses, use the data_trim_to_full_idx_map array as an index
        poi_idx = data_trim_to_full_idx_map[np.isfinite(data_trim_to_full_idx_map)]
        # throwing errors so setting int indices with loop
        # some data_trim_to_full_idx_map values are np.nan, hence not int from get-go
        poi_idx_int = np.full(len(poi_idx), 0)
        for i, idx in enumerate(poi_idx):
            poi_idx_int[i] = np.int(idx)
        # calculate accessibilities
        pois[:,netw_src_idx] = accessibility.accessibility_osm_poi(poi_cats[poi_idx_int], reachable_classes_dist, 40, beta_800)

        # use corresponding indices for reachable verts
        ind = np.where(np.isfinite(netw_dist_map_trim))[0]
        for trim_to_idx in ind:

            # skip self node
            if trim_to_idx == netw_src_idx_trim:
                continue

            dist_m = netw_dist_map_trim[trim_to_idx]

            # some crow-flies max distance nodes won't be reached within max distance threshold over the network
            if np.isinf(dist_m):
                continue

            # remember that the shortest_path_tree is set to np.inf for mixed-uses purposes, so check here for distance
            if dist_m > max_dist:
                continue

            # calculate gravity and betweenness
            # the strength of the weight is based on the start and end vertices, not the intermediate locations
            netw_wt_100 = np.exp(beta_100 * dist_m)
            netw_wt_200 = np.exp(beta_200 * dist_m)
            netw_wt_400 = np.exp(beta_400 * dist_m)
            netw_wt_800 = np.exp(beta_800 * dist_m)

            # gravity -> an accessibility measure, or effectively a closeness consisting of inverse distance weighted node count
            gravity[0][netw_src_idx] += netw_wt_100
            gravity[1][netw_src_idx] += netw_wt_200
            gravity[2][netw_src_idx] += netw_wt_400
            gravity[3][netw_src_idx] += netw_wt_800

            # betweenness - only counting truly between vertices, not starting and ending verts
            intermediary_idx_trim = np.int(netw_pred_map_trim[trim_to_idx])
            intermediary_idx_mapped = np.int(netw_idx_map_trim_to_full[intermediary_idx_trim])  # cast to int
            # only counting betweenness in one 'direction' since the graph is symmetrical (non-directed)
            while True:
                # break out of while loop if the intermediary has reached the source node
                if intermediary_idx_trim == netw_src_idx_trim:
                    break

                # weighted variants - summed at all distances
                betweenness_wt[0][intermediary_idx_mapped] += netw_wt_100
                betweenness_wt[1][intermediary_idx_mapped] += netw_wt_200
                betweenness_wt[2][intermediary_idx_mapped] += netw_wt_400
                betweenness_wt[3][intermediary_idx_mapped] += netw_wt_800

                # unlike the dist_map the pred_map contains all vertices, so no offset required
                intermediary_idx_trim = np.int(netw_pred_map_trim[intermediary_idx_trim])
                intermediary_idx_mapped = np.int(netw_idx_map_trim_to_full[intermediary_idx_trim])  # cast to int

    return gravity, betweenness_wt, mixed_uses_wt, pois
"""
