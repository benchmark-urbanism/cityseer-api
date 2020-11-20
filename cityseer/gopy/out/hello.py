# python wrapper for package command-line-arguments within overall package hello
# This is what you import to use the package.
# File is generated by gopy. Do not edit.
# gopy build -output=out -vm=/usr/local/bin/python3 /Users/Shared/dev/github/cityseer/cityseer/cityseer/gopy/hello.go

# the following is required to enable dlopen to open the _go.so file
import os, sys, inspect, collections

cwd = os.getcwd()
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(currentdir)
import _hello

os.chdir(cwd)

# to use this code in your end-user python file, import it as follows:
# from hello import hello
# and then refer to everything using hello. prefix
# packages imported by this package listed below:

import go


# ---- Types ---


# ---- Enums from Go (collections of consts with same type) ---


# ---- Constants from Go: Python can only ask that you please don't change these! ---


# ---- Global Variables: can only use functions to access ---


# ---- Interfaces ---


# ---- Structs ---


# ---- Slices ---


# ---- Maps ---


# ---- Constructors ---


# ---- Functions ---
def Hello(name):
    """Hello(str name) str"""
    return _hello.hello_Hello(name)
