import time

from cityseer.algos import data
from cityseer.metrics import layers
from cityseer.util import mock, graphs


def test_filter_times():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    # generate some data
    data_dict = mock.mock_data(G, num=100000, random_seed=0)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter
    src_x = G.nodes[0]['x']
    src_y = G.nodes[0]['y']
    max_dist = 750

    iterations = 100

    # WARMUP THE FUNCTIONS FIRST
    print('')
    print('Warming up functions')
    _ = data.___distance_filter(D.index, src_x, src_y, max_dist, radial=True)
    _ = data.radial_filter(src_x, src_y, D.x_arr, D.y_arr, max_dist)

    start = time.time()
    for i in range(iterations):
        _ = data.___distance_filter(D.index, src_x, src_y, max_dist, radial=True)
    end = time.time()
    print(f'TIME: distance_filter = {end - start}')
    # distance_filter = 5.90056300163269

    start = time.time()
    for i in range(iterations):
        _ = data.radial_filter(src_x, src_y, D.x_arr, D.y_arr, max_dist)
    end = time.time()
    print(f'TIME: radial_filter = {end - start}')
    # radial_filter = 0.1493382453918457


def test_nearest_filters():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    # generate some data
    data_dict = mock.mock_data(G, num=100000, random_seed=0)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter
    src_x = G.nodes[0]['x']
    src_y = G.nodes[0]['y']
    max_dist = 750

    iterations = 100

    # WARMUP THE FUNCTIONS FIRST
    print('')
    print('Warming up functions')
    _ = data.___nearest_idx(D.index, src_x, src_y, max_dist)
    _ = data.nearest_idx_simple(src_x, src_y, D.x_arr, D.y_arr, max_dist)

    start = time.time()
    for i in range(iterations):
        _ = data.___nearest_idx(D.index, src_x, src_y, max_dist)
    end = time.time()
    print(f'TIME: nearest_idx = {end - start}')
    # nearest_idx = 4.57602596282959

    start = time.time()
    for i in range(iterations):
        _ = data.nearest_idx_simple(src_x, src_y, D.x_arr, D.y_arr, max_dist)
    end = time.time()
    print(f'TIME: nearest_idx_simple = {end - start}')
    # nearest_idx_simple = 0.06587481498718262
