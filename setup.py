'''
https://docs.python.org/3.7/distutils/examples.html
https://docs.python.org/3/distutils/setupscript.html
https://scikit-build.readthedocs.io/en/latest/usage.html

# manual deployment
pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
pip install --upgrade twine
TESTING REPO: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
DOWNLOADING FROM TEST REPO: pip install --extra-index-url https://test.pypi.org/simple/ cityseer==0.7.6rc7
OTHERWISE: twine upload dist/*

For AOT: (not preferred):
-------------------------
if compiling numba directly from file:
cc.output_dir = os.path.abspath(os.path.join(os.pardir, 'compiled'))
cc.verbose = True
cc.compile()
use numba.typeof to deduce signatures
nest @njit decorator to help aot functions find each other, see https://stackoverflow.com/questions/49326937/error-when-compiling-a-numba-module-with-function-using-other-functions-inline

To test against AOT compiled modules, run python setup.py test, which will copy compiled modules to working modules:
then run pytest test, which now tests against compiled .so files...
'''
from setuptools import setup

setup (
    name = 'cityseer',
    version='0.8.8',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.util'],
    description = 'Computational tools for urban analysis',
    url='https://github.com/cityseer/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/cityseer/cityseer/issues",
        "Documentation": "https://cityseer.github.io/cityseer/",
        "Source Code": "https://github.com/cityseer/cityseer",
    },
    author='Gareth Simons',
    author_email='gareth@cityseer.io',
    license='Apache 2.0 + "Commons Clause" License Condition v1.0',
    install_requires=[
        'numpy',
        'numba',
        'utm',
        'shapely>=1.7a1',
        'networkx',
        'tqdm',
        'matplotlib',
        'sklearn'
    ],
    # ext_package='cityseer.algos',  # NB -> sets output directory for extension modules
    # some sort of issue with AOT precompilation - using njit for now...
    ext_modules = [
        # centrality.cc.distutils_extension(),
        # data.cc.distutils_extension(),
        # diversity.cc.distutils_extension(),
        #checks.cc.distutils_extension()
    ]
)
