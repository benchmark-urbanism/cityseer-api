import ctypes
lib = ctypes.CDLL('cityseer/algos/test.so')
print(lib.DoubleIt(21))