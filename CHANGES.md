Changes
=======

development
-----------

* Fix definition of `norm_cdf` psychometric function, by [Alex Forrence](https://github.com/aforren1)
* Fix various warnings and `DeprecationWarnings` coming from NumPy and xarray
* Add Thurstone scaling

v2019.4
-------

* Allow JSON serialization of random number generator

v2019.3
-------

* Allow to pass a prior when instantiating `QuestPlusWeibull`

v2019.2
-------

* Allow passing a random seed via `stim_selection_options` keyword
  argument
* Better handling of `stim_selection_options` defaults (now allows
  to supply only a subset of options)

v2019.1
-------

* Allow to pass priors for only some parameters
  (the remaining parameters will be assigned an uninformative prior)
* Add more docstrings, fix typo in usage example
* Test on Python 3.8

v0.0.5
------

* Allow retrieval of marginal posterior PDFs via `QuestPlus.marginal_posterior`
* Allow `nan` values in JSON output (violating JSON std but useful)
