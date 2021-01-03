# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'UltraOpt'
copyright = '2021, Qichun Tang'
author = 'Qichun Tang'

import os
import sys

import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('../../../'))
sys.path.insert(0, os.path.abspath('_ext'))

# The full version, including alpha/beta/rc tags
with open("../../../ultraopt/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
html_static_path = ['_static']
extensions = [
    # 'edit_on_github',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.inheritance_diagram',

    'sphinx.ext.graphviz',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    # 'sphinxcontrib.exceltable',
    'recommonmark',
    # 'm2r',
    # 'numpydoc',
    # 'sphinx_gallery.gen_gallery',
]
edit_on_github_project = 'auto-flow/ultraopt'
edit_on_github_branch = 'dev'
html_favicon = 'favicon.png'
html_logo = 'logo.png'
autoclass_content = 'both'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints']

source_parsers = {
    '.md': CommonMarkParser,
}

language = "zh_CN"
source_suffix = ['.rst', '.md']
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
pygments_style = 'sphinx'
