# in case the user doesn't have setup tools installed

# python3 -m pip install --user --upgrade setuptools wheel
# python3 setup.py sdist bdist_wheel
# python3 -m pip install --user --upgrade twine
# /Users/gareth/Library/Python/3.7/bin/twine upload dist/*

from setuptools import setup
from cityseer import networks, data, landuses

setup (
    name = 'cityseer',
    version = '0.1.14',
    packages=['cityseer'],  # where the source code is located in project
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
        'tqdm',
        'shapely',
        'networkx',
        'matplotlib'
    ],
    ext_modules = [
        networks.cc.distutils_extension(),
        data.cc.distutils_extension(),
        landuses.cc.distutils_extension()
    ])
