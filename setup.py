'''
https://docs.python.org/3.7/distutils/examples.html

in case the user doesn't have setup tools installed:
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
python3 -m pip install --user --upgrade twine
/Users/gareth/Library/Python/3.7/bin/twine upload dist/*

if compiling numba directly from file:
cc.output_dir = os.path.abspath(os.path.join(os.pardir, 'compiled'))
cc.verbose = True
cc.compile()
use numba.typeof to deduce signatures
add @njit to help aot functions find each other, see https://stackoverflow.com/questions/49326937/error-when-compiling-a-numba-module-with-function-using-other-functions-inline
'''

from setuptools import setup

setup (
    name = 'cityseer',
    version = '0.2.0',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.util'],
    description = 'Computational tools for urban analysis',
    url='https://github.com/cityseer/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/cityseer/cityseer-api/issues",
        "Documentation": "https://github.com/cityseer/cityseer-api",
        "Source Code": "https://github.com/cityseer/cityseer-api",
    },
    author='Gareth Simons',
    author_email='gareth@cityseer.io',
    license='AGPL-3.0',
    install_requires=[
        'numpy',
        'numba',
        'utm',
        'shapely>=1.7a1',
        'networkx',
        'tqdm',
        'matplotlib',
        'sklearn'
    ]
    #ext_modules = [
        # TODO: not precompiling until sorted() etc. bug is resolved?
        #data.cc.distutils_extension(),
        #diversity.cc.distutils_extension(),
        #networks.cc.distutils_extension()
    #]
)
