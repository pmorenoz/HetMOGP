from setuptools import setup
# This is the setup thing (written by Wil)
setup(
    name='HetMOGP',
    packages=['hetmogp', 'likelihoods'],
    version='0.1',
    author='Pablo Moreno-Muñoz',
    url='http://github.com/pmorenoz/HetMOGP/',
    description='Implementation of Heterogeneous Multi-output Gaussian Process (HetMOGP) model. The entire code is written in Python and connected with the GPy package, specially useful for Gaussian processes',
    license='Apache License 2.0',
    zip_safe=False
)
