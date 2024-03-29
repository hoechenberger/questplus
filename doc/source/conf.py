# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

abs_path = os.path.abspath("../../questplus")
assert os.path.exists(abs_path)
sys.path.insert(0, abs_path)


# -- Project information -----------------------------------------------------

project = "questplus"
copyright = "2019, Richard Höchenberger"
author = "Richard Höchenberger"

extensions = [
    "recommonmark",  # markdown support
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # 'sphinx.ext.viewcode',
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be loaded AFTER napoleon
    # 'sphinx.ext.coverage'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autoclass_content = "both"  # Document __init__() methods as well.
