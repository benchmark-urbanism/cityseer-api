# in case the user doesn't have setup tools installed

# python3 -m pip install --user --upgrade setuptools wheel
# python3 setup.py sdist bdist_wheel
# python3 -m pip install --user --upgrade twine
# /Users/gareth/Library/Python/3.7/bin/twine upload dist/*

# TODO: fix install requires

from setuptools import setup
from cityseer import centrality, networks, mixed_uses, accessibility

setup (
    name = 'cityseer',
    version = '0.1.11',
    packages=['cityseer'],
    description = 'Computational tools for urban analysis',
    url='https://github.com/cityseer/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/cityseer/cityseer-api/issues",
        "Documentation": "https://github.com/cityseer/cityseer-api",
        "Source Code": "https://github.com/cityseer/cityseer-api",
    },
    author='Gareth Simons',
    author_email='gareth@cityseer.io',
    license='GPL-3.0',
    install_requires=[
        'numpy',
        'numba'
      ],
    ext_modules = [
        centrality.cc.distutils_extension(),
        networks.cc.distutils_extension(),
        mixed_uses.cc.distutils_extension(),
        accessibility.cc.distutils_extension()
    ])

