from setuptools import setup

setup(
    name='CSCORE',
    version='0.0.1',
    author='Chang Su',
    author_email='chang.su@emory.edu',
    url='https://github.com/ChangSuBiostats/CS-CORE_python',
    license='LICENSE',
    packages=['CSCORE'],
    description='A Python package for cell-type-specific co-expression inference from single cell RNA-sequencing data',
    python_requires=">3",
    install_requires=[
        'numpy',
        "scipy",
        "scanpy"
    ]
)
