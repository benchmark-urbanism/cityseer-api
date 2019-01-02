"""
Debating global class chaining-through to components - data_map - attribute_overlays...
Less redundant computation, but typically workflows not necessarily linear
i.e. centralities often called separately from landuses from statistics etc.
@jitclass can't be AOT compiled (at least don't think so) so don't bother until GCP supports numba
"""
from numba import jitclass, types

# JIT CLASS
spec = [
    ('node_map', types.float64[:, :]),
    ('edge_map', types.float64[:, :]),

    ('data_map_0_name', types.string),
    ('data_map_1_name', types.string),
    ('data_map_2_name', types.string),
    ('data_map_3_name', types.string),
    ('data_map_4_name', types.string),

    ('data_map_0', types.float64[:, :]),
    ('data_map_1', types.float64[:, :]),
    ('data_map_2', types.float64[:, :]),
    ('data_map_3', types.float64[:, :]),
    ('data_map_4', types.float64[:, :])
]


@jitclass(spec)
class System():

    def __init__(self, node_map, edge_map):
        # uninitialised fields (as defined in the spec) contain garbage data

        self.node_map = node_map
        self.edge_map = edge_map

        self.data_map_0_name = ''
        self.data_map_1_name = ''
        self.data_map_2_name = ''
        self.data_map_3_name = ''
        self.data_map_4_name = ''

    def load_data_map(self, name, data_map):

        if self.data_map_0_name == '':
            self.data_map_0_name = name
            self.data_map_0 = data_map
        elif self.data_map_0_name == name:
            raise ValueError('A data map of this name already exists. Please delete existing map first.')

        elif self.data_map_1_name == '':
            self.data_map_1_name = name
            self.data_map_1 = data_map
        elif self.data_map_1_name == name:
            raise ValueError('A data map of this name already exists. Please delete existing map first.')

        elif self.data_map_2_name == '':
            self.data_map_2_name = name
            self.data_map_2 = data_map
        elif self.data_map_2_name == name:
            raise ValueError('A data map of this name already exists. Please delete existing map first.')

        elif self.data_map_3_name == '':
            self.data_map_3_name = name
            self.data_map_3 = data_map
        elif self.data_map_3_name == name:
            raise ValueError('A data map of this name already exists. Please delete existing map first.')

        elif self.data_map_4_name == '':
            self.data_map_4_name = name
            self.data_map_4 = data_map
        elif self.data_map_4_name == name:
            raise ValueError('A data map of this name already exists. Please delete existing map first.')

        else:
            raise ValueError('Number of data maps exceeded. A maximum of 5 data maps is currently supported.')

    def remove_data_map(self, name):

        if self.data_map_0_name == name:
            self.data_map_0_name = ''

        elif self.data_map_1_name == name:
            self.data_map_1_name = ''

        elif self.data_map_2_name == name:
            self.data_map_2_name = ''

        elif self.data_map_3_name == name:
            self.data_map_3_name = ''

        elif self.data_map_4_name == name:
            self.data_map_4_name = ''

        else:
            raise ValueError('A data map matching the specified name has not been found.')
