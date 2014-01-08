# Changelog

## Version 1.0.4

* Use a Cholesky solver rather than computing inverses
* Specify C-contiguous arrays and disable wraparound in Cython code
* Add `__copy__`, `__deepcopy__`, `__getstate__`, and `__setstate__`
  methods to gp and kernels
* Add methods for getting/setting kernel parameters by name

## Version 1.0.3

* Fix bugs in documentation generation
* Remove `fit_MLII` method, because it was not robust

## Version 1.0.2

* Miscellaneous bugfixes

## Version 1.0.1

* Fix bugs in compiling documentation

## Version 1.0.0

* Use Cython instead of Numba
* Create `gp.ext` module to hold Cython code
* Improve test coverage
* Enforce usage of 1-dimensional arrays

## Version 0.1.3

* Fix pip installion bug
* Add a few new methods to `GP`:
    * `dm_dtheta` -- compute the derivative of the mean of the
      gaussian process with respect to its parameters
	* `plot` -- use pyplot to plot the predictive mean and variance of
      the gaussian process
* Switch to semantic versioning

## Version 0.01.2

* Add Sphinx support
* Write better documentation

## Version 0.01.1

* More robust tests

## Version 0.01

* Basic GP functionality
* Gaussian kernel
* Periodic kernel
