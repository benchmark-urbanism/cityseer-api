'''
https://docs.python.org/3.9/distutils/examples.html
https://docs.python.org/3/distutils/setupscript.html
https://scikit-build.readthedocs.io/en/latest/usage.html

# manual deployment
pip install --upgrade setuptools wheel
python setup.py sdist bdist_wheel
pip install --upgrade twine
TESTING REPO: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
DOWNLOADING FROM TEST REPO: pip install --extra-index-url https://test.pypi.org/simple/ cityseer==1.0.0a1
OTHERWISE: twine upload dist/*
'''

from setuptools import setup

setup(
    name='cityseer',
    version='1.0.0',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.tools'],
    description='Computational tools for urban analysis',
    url='https://github.com/benchmark-urbanism/cityseer-api',
    project_urls={
        "Bug Tracker": "https://github.com/benchmark-urbanism/cityseer-api/issues",
        "Documentation": "https://cityseer.benchmarkurbanism.com/",
        "Source Code": "https://github.com/benchmark-urbanism/cityseer-api",
    },
    author='Gareth Simons',
    author_email='gareth@cityseer.io',
    license='Apache 2.0 + "Commons Clause" License Condition v1.0',
    install_requires=[
        'numpy',
        'numba',
        'utm',
        'shapely>=1.7.0',
        'networkx>=2.4',
        'tqdm',
        'matplotlib',
        'sklearn'
    ]
)
