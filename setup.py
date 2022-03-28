"""
DOWNLOADING FROM TEST REPO: pip install --extra-index-url https://test.pypi.org/simple/ cityseer==1.1.7b9
"""

from setuptools import setup

setup(
    name='cityseer',
    version='1.2.2',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.tools'],
    description='Computational tools for network-based pedestrian-scale urban analysis',
    url='https://github.com/benchmark-urbanism/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/benchmark-urbanism/cityseer-api/issues",
        "Documentation": "https://cityseer.benchmarkurbanism.com/",
        "Source Code": "https://github.com/benchmark-urbanism/cityseer-api",
    },
    author='Gareth Simons',
    author_email='info@benchmarkurbanism.com',
    license='GNU AGPLv3',
    install_requires=[
        'matplotlib',
        'networkx==2.6.3',
        'numba==0.55.1',
        'numba-progress==0.0.2',
        'numpy==1.21.5',
        'pyproj',
        'requests',
        'shapely==1.8.1.post1',
        'scikit-learn',
        'tqdm',
        'utm'
    ]
)
