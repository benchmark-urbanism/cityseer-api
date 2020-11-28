'''
https://docs.python.org/3.7/distutils/examples.html
https://docs.python.org/3/distutils/setupscript.html
https://scikit-build.readthedocs.io/en/latest/usage.html

# manual deployment
pip install --upgrade setuptools wheel
python setup.py sdist bdist_wheel
pip install --upgrade twine
TESTING REPO: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
DOWNLOADING FROM TEST REPO: pip install --extra-index-url https://test.pypi.org/simple/ cityseer==0.10.3.dev0
OTHERWISE: twine upload dist/*
'''

from setuptools import setup

setup(
    name='cityseer',
    version='0.12.0',
    packages=['cityseer', 'cityseer.algos', 'cityseer.metrics', 'cityseer.util'],
    description='Computational tools for urban analysis',
    url='https://github.com/cityseer/cityseer',
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
        'shapely>=1.7.0',
        'networkx>=2.4',
        'tqdm',
        'matplotlib',
        'sklearn'
    ]
)
