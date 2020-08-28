# https://dbader.org/blog/python-ctypes-tutorial
# https://medium.com/learning-the-go-programming-language/calling-go-functions-from-other-languages-4c7d8bcc69bf#.n73as5d6d
# https://stackoverflow.com/questions/5081875/ctypes-beginner
# https://scipy-cookbook.readthedocs.io/items/Ctypes.html
# https://blog.filippo.io/building-python-modules-with-go-1-5/

import numpy.ctypeslib as ctl
import ctypes
import os

print(os.getcwd())

lib = ctl.load_library('test.so', 'cityseer/algos')

NodeDensity = lib.NodeDensity
NodeDensity.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
NodeDensity.restype = ctypes.c_double
r = NodeDensity(200, 100, 50, 1)
print(r)

lib.MakeGraph()
lib.PrintGraph()

class Node(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
        ('live', ctypes.c_bool),
        ('edges', ctypes.)  # TODO: slice
    ]

class Edge(ctypes.Structure):
    _fields_ = [
        ('startNodeIdx', ctypes.),  # TODO: string
        ('endNodeIdx', ctypes.),
        ('length', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('impedanceFactor', ctypes.c_double),
        ('inBearing', ctypes.c_double),
        ('outBearing', ctypes.c_double)
    ]

class Graph(ctypes.Structure):
    _fields_ = [
        ('nodes', Node),  # TODO: map
        ('edges', Edge)
    ]