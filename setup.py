"""
DOWNLOADING FROM TEST REPO: pip install --extra-index-url https://test.pypi.org/simple/ cityseer==1.1.4b3
"""

from setuptools import setup

setup(
    name='cityseer',
    version='1.1.5',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.tools'],
    description='Computational tools for urban analysis',
    url='https://github.com/benchmark-urbanism/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/benchmark-urbanism/cityseer-api/issues",
        "Documentation": "https://cityseer.benchmarkurbanism.com/",
        "Source Code": "https://github.com/benchmark-urbanism/cityseer-api",
    },
    author='Gareth Simons',
    author_email='info@benchmarkurbanism.com',
    license='GNU GPLv3',
    install_requires=[
        'matplotlib',
        'networkx>=2.4',
        'numba>=0.53',
        'numpy',
        'pytest',
        'shapely>=1.7',
        'sklearn',
        'tqdm',
        'utm'
    ]
)
