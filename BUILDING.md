## Building a release

* Create `sdist` and `wheel` distributions:
  ```python
  python -m build --sdist --wheel
  ```
  (optionally append `--no-isolation` if running into group policy
   permission errors on managed Windows systems)
