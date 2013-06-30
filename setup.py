#!/usr/bin/env python

from distutils.core import setup

setup(
    name='gaussian_processes',
    version=open('VERSION.txt').read().strip(),
    description='Python library for gaussian processes',
    author='Jessica B. Hamrick',
    author_email='jhamrick@berkeley.edu',
    url='https://github.com/jhamrick/gaussian_processes',
    packages=['kernels'],
    py_modules=['gp'],
    package_dir={'': 'src'},
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
        'numba',
        'sympy'
    ]
)
