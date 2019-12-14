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
