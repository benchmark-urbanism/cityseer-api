# https://medium.com/learning-the-go-programming-language/calling-go-functions-from-other-languages-4c7d8bcc69bf#.n73as5d6d
# https://stackoverflow.com/questions/5081875/ctypes-beginner
# https://scipy-cookbook.readthedocs.io/items/Ctypes.html

import numpy.ctypeslib as ctl
import ctypes

lib = ctl.load_library('test.so', './')

NodeDensity = lib.NodeDensity
NodeDensity.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
NodeDensity.restype = ctypes.c_double
r = NodeDensity(200, 100, 50, 1)
print(r)