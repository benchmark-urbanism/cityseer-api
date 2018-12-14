import numpy as np
import pytest
from scipy.stats import entropy

from cityseer.algos import diversity, data
from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs


def test_hill_diversity():
    # test hill diversity against scipy entropy
    for counts, probs in mock.mock_species_diversity():
        # check hill q=1 - this can be tested against scipy because hill q=1 is exponential of entropy
        assert np.allclose(diversity.hill_diversity(counts, q=1), np.exp(entropy(probs)))
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert np.allclose(diversity.hill_diversity(counts, 0.99999999), np.exp(entropy(probs)))
        assert np.allclose(diversity.hill_diversity(counts, 1.00000001), np.exp(entropy(probs)))
        # check for malformed q
        with pytest.raises(ValueError):
            diversity.hill_diversity(counts, q=-1)


def test_hill_diversity_branch_distance_wt():
    # test against hill diversity by setting all weights = 1
    for counts, probs in mock.mock_species_diversity():

        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            assert np.allclose(diversity.hill_diversity(counts, q),
                               diversity.hill_diversity_branch_distance_wt(counts, non_weights, q, non_beta))

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts[:-1], non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights[:-1], q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=-1, beta=-0.005)


def test_hill_diversity_pairwise_distance_wt():

    for counts, probs in mock.mock_species_diversity():

        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            # what to test against? For now, check successful run
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q, non_beta)

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts[:-1], non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights[:-1], q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=-1, beta=-0.005)


def test_hill_diversity_pairwise_matrix_wt():

    for counts, probs in mock.mock_species_diversity():

        non_matrix = np.full((len(counts), len(counts)), 1)

        for q in [0, 1, 2]:
            # what to test against? For now, check successful run
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix, q)

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts[:-1], non_matrix, q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix[:-1], q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix[:, :-1], q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix, q=-1)


def test_gini_simpson_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_diversity():
        diversity.gini_simpson_diversity(counts)


def test_shannon_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # test against scipy entropy
    for counts, probs in mock.mock_species_diversity():
        assert abs(entropy(probs) - diversity.shannon_diversity(probs)) < 0.0000000001


def test_raos_quadratic_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_diversity():
        mock_matrix = np.full((len(counts), len(counts)), 1)
        diversity.raos_quadratic_diversity(counts, mock_matrix)


def test_local_landuses():
    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    # setup data
    data_dict = mock.mock_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, node_map, edge_map, 500)

    # set parameters
    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances = networks.distance_from_beta(betas)
    qs = np.array([0, 1, 2])
    mock_matrix = np.full((len(class_labels), len(class_labels)), 1)

    # set the keys - add shuffling to be sure various orders work
    hill_keys = np.arange(4)
    np.random.shuffle(hill_keys)
    non_hill_keys = np.arange(3)
    np.random.shuffle(non_hill_keys)
    ac_keys = np.array([1, 2, 5])
    np.random.shuffle(ac_keys)

    mu_data_hill, mu_data_other, ac_data, ac_data_wt = \
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 qs=qs,
                                 mixed_use_hill_keys=hill_keys,
                                 mixed_use_other_keys=non_hill_keys,
                                 accessibility_keys=ac_keys,
                                 cl_disparity_wt_matrix=mock_matrix,
                                 angular=False)

    # hill
    hill = mu_data_hill[np.where(hill_keys == 0)][0]
    hill_branch_wt = mu_data_hill[np.where(hill_keys == 1)][0]
    hill_pw_wt = mu_data_hill[np.where(hill_keys == 2)][0]
    hill_disp_wt = mu_data_hill[np.where(hill_keys == 3)][0]
    # non hill
    shannon = mu_data_other[np.where(non_hill_keys == 0)][0]
    gini = mu_data_other[np.where(non_hill_keys == 1)][0]
    raos = mu_data_other[np.where(non_hill_keys == 2)][0]
    # access non-weighted
    ac_1_nw = ac_data[np.where(ac_keys == 1)][0]
    ac_2_nw = ac_data[np.where(ac_keys == 2)][0]
    ac_5_nw = ac_data[np.where(ac_keys == 5)][0]
    # access weighted
    ac_1_w = ac_data_wt[np.where(ac_keys == 1)][0]
    ac_2_w = ac_data_wt[np.where(ac_keys == 2)][0]
    ac_5_w = ac_data_wt[np.where(ac_keys == 5)][0]

    # test manual metrics against all nodes
    mu_max_unique = int(data_map[:, 3].max() + 1)
    # test against various distances
    for d_idx in range(len(distances)):
        dist_cutoff = distances[d_idx]
        beta = betas[d_idx]

        for src_idx in range(len(G)):

            reachable_classes_trim, reachable_classes_dist_trim, _data_trim_to_full_idx_map = \
                data.aggregate_to_src_idx(src_idx,
                                          node_map,
                                          edge_map,
                                          data_map,
                                          dist_cutoff)

            # counts of each class type (array length per max unique classes - not just those within max distance)
            cl_counts = np.full(mu_max_unique, 0)
            # nearest of each class type (likewise)
            cl_nearest = np.full(mu_max_unique, np.inf)

            a_1_nw = 0
            a_2_nw = 0
            a_5_nw = 0
            a_1_w = 0
            a_2_w = 0
            a_5_w = 0

            for i in range(len(reachable_classes_trim)):
                # classes outside of dist_cutoff will be assigned np.inf
                cl_dist = reachable_classes_dist_trim[i]
                if np.isinf(cl_dist):
                    continue
                cl = int(reachable_classes_trim[i])
                # double check distance is within threshold
                assert cl_dist <= dist_cutoff
                # update the class counts
                cl_counts[cl] += 1
                # if distance is nearer, update the nearest distance array too
                if cl_dist < cl_nearest[cl]:
                    cl_nearest[cl] = cl_dist
                # aggregate accessibility codes
                if cl == 1:
                    a_1_nw += 1
                    a_1_w += np.exp(beta * cl_dist)
                elif cl == 2:
                    a_2_nw += 1
                    a_2_w += np.exp(beta * cl_dist)
                elif cl == 5:
                    a_5_nw += 1
                    a_5_w += np.exp(beta * cl_dist)

            assert ac_1_nw[d_idx][src_idx] == a_1_nw
            assert ac_2_nw[d_idx][src_idx] == a_2_nw
            assert ac_5_nw[d_idx][src_idx] == a_5_nw

            assert ac_1_w[d_idx][src_idx] == a_1_w
            assert ac_2_w[d_idx][src_idx] == a_2_w
            assert ac_5_w[d_idx][src_idx] == a_5_w

            assert hill[0][d_idx][src_idx] == diversity.hill_diversity(cl_counts, 0)
            assert hill[1][d_idx][src_idx] == diversity.hill_diversity(cl_counts, 1)
            assert hill[2][d_idx][src_idx] == diversity.hill_diversity(cl_counts, 2)

            assert hill_branch_wt[0][d_idx][src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 0, beta)
            assert hill_branch_wt[1][d_idx][src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 1, beta)
            assert hill_branch_wt[2][d_idx][src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 2, beta)

            assert hill_pw_wt[0][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 0, beta)
            assert hill_pw_wt[1][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 1, beta)
            assert hill_pw_wt[2][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 2, beta)

            assert hill_disp_wt[0][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 0)
            assert hill_disp_wt[1][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 1)
            assert hill_disp_wt[2][d_idx][src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 2)

            assert shannon[d_idx][src_idx] == diversity.shannon_diversity(cl_counts)
            assert gini[d_idx][src_idx] == diversity.gini_simpson_diversity(cl_counts)
            assert raos[d_idx][src_idx] == diversity.raos_quadratic_diversity(cl_counts, mock_matrix)

    # check that angular is passed-through
    # actual angular tests happen in test_shortest_path_tree()
    # here the emphasis is simply on checking that the angular instruction gets chained through

    # setup dual data
    G_dual = graphs.networkX_to_dual(G)
    node_labels_dual, node_map_dual, edge_map_dual = graphs.graph_maps_from_networkX(G_dual)
    data_dict_dual = mock.mock_data(G, random_seed=13)
    data_uids_dual, data_map_dual, class_labels_dual = layers.data_map_from_dict(data_dict_dual)
    data_map_dual = data.assign_to_network(data_map_dual, node_map_dual, edge_map_dual, 500)
    mock_matrix = np.full((len(class_labels_dual), len(class_labels_dual)), 1)

    mu_hill_dual, mu_other_dual, ac_dual, ac_wt_dual = \
        diversity.local_landuses(node_map_dual,
                                 edge_map_dual,
                                 data_map_dual,
                                 distances,
                                 betas,
                                 qs=qs,
                                 mixed_use_hill_keys=hill_keys,
                                 mixed_use_other_keys=non_hill_keys,
                                 accessibility_keys=ac_keys,
                                 cl_disparity_wt_matrix=mock_matrix,
                                 angular=True)

    mu_hill_dual_sidestep, mu_other_dual_sidestep, ac_dual_sidestep, ac_wt_dual_sidestep = \
        diversity.local_landuses(node_map_dual,
                                 edge_map_dual,
                                 data_map_dual,
                                 distances,
                                 betas,
                                 qs=qs,
                                 mixed_use_hill_keys=hill_keys,
                                 mixed_use_other_keys=non_hill_keys,
                                 accessibility_keys=ac_keys,
                                 cl_disparity_wt_matrix=mock_matrix,
                                 angular=False)

    assert not np.allclose(mu_hill_dual, mu_hill_dual_sidestep)
    assert not np.allclose(mu_other_dual, mu_other_dual_sidestep)
    assert not np.allclose(ac_dual, ac_dual_sidestep)
    assert not np.allclose(ac_wt_dual, ac_wt_dual_sidestep)

    # check that missing qs are caught for hill metrics
    with pytest.raises(ValueError):
        diversity.local_landuses(node_map_dual,
                                 edge_map_dual,
                                 data_map_dual,
                                 distances,
                                 betas,
                                 qs=np.array([]),
                                 mixed_use_hill_keys=np.array([0]),
                                 angular=False)

    # check that missing matrix is caught for disparity weighted indices
    for h_key, o_key in (([3], []), ([], [2])):
        with pytest.raises(ValueError):
            diversity.local_landuses(node_map_dual,
                                     edge_map_dual,
                                     data_map_dual,
                                     distances,
                                     betas,
                                     qs=qs,
                                     mixed_use_hill_keys=np.array(h_key),
                                     mixed_use_other_keys=np.array(o_key),
                                     angular=False)

    # check that problematic keys are caught
    for mu_h_key, mu_o_key, ac_key in [([], [], []),  # missing
                                       ([-1], [1], [1]),  # negatives
                                       ([1], [-1], [1]),
                                       ([1], [1], [-1]),
                                       ([4], [1], [1]),  # out of range
                                       ([1], [3], [1]),
                                       ([1], [1], [data_map_dual[:, 3].max() + 1]),
                                       ([1, 1], [1], [1]),  # duplicates
                                       ([1], [1, 1], [1]),
                                       ([1], [1], [1, 1])]:
        with pytest.raises(ValueError):
            diversity.local_landuses(node_map_dual,
                                     edge_map_dual,
                                     data_map_dual,
                                     distances,
                                     betas,
                                     qs=qs,
                                     mixed_use_hill_keys=np.array(mu_h_key),
                                     mixed_use_other_keys=np.array(mu_o_key),
                                     accessibility_keys=np.array(ac_key))
