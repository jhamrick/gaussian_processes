#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "gp.ext.gaussian_c", ["gp/ext/gaussian_c.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"]
    ),
    Extension(
        "gp.ext.periodic_c", ["gp/ext/periodic_c.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"]
    ),
    Extension(
        "gp.ext.gp_c", ["gp/ext/gp_c.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"]
    )
]

setup(
    name='gaussian_processes',
    version=open('VERSION.txt').read().strip(),
    description='Python library for gaussian processes',
    author='Jessica B. Hamrick',
    author_email='jhamrick@berkeley.edu',
    url='https://github.com/jhamrick/gaussian_processes',
    packages=['gp', 'gp.kernels', 'gp.ext', 'gp.tests'],
    ext_modules=cythonize(extensions),
    keywords='gp kernel statistics',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        'numpy',
        'sympy'
    ]
)
