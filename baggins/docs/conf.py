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
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
print(f"pythonpath is {sys.path}")


# -- Project information -----------------------------------------------------

project = "BAGGInS"
copyright = "2025, Alex Rawlings"
author = "Alex Rawlings"

# The full version, including alpha/beta/rc tags
release = "1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/bagend.png"

# Napoleon settings
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

# Autosummary
autosummary_generate = True

# Autodoc
autodoc_typehints = "none"
autodoc_mock_imports = [
    "pygad",
    "unyt",
    "merger_ic_generator",
    "gadgetorbits",
    "cmdstanpy",
    "voronoi_binning",
    "synthesizer",
]
autodoc_default_options = {"member-order": "groupwise"}

latex_logo = "_static/bagend.png"

latex_elements = {
    "preamble": r"""
        \usepackage{xcolor}

        \makeatletter
        \AtBeginDocument{%
        % Style function/class names IF the macro exists
        \@ifundefined{sphinxstyleobjectname}{}{%
            \renewcommand{\sphinxstyleobjectname}[1]{%
            {\ttfamily\bfseries\color{blue!70!black}#1}%
            }
        }

        % Fallback: style the entire signature if objectname is not available
        \@ifundefined{sphinxstylesignature}{}{%
            \renewcommand{\sphinxstylesignature}[1]{%
            {\ttfamily\bfseries\color{blue!70!black}#1}%
            }
        }
        }
        \makeatother
    """,
    "maketitle": r"""
        \begin{titlepage}
        \centering

        \vspace*{2cm}
        \includegraphics[height=6cm]{bagend.png}

        \vspace{0.8cm}
        {\Huge\bfseries BAGGInS \par}

        \vspace{0.5cm}
        {\large Bayesian Analysis of Galaxy-Galaxy Interaction in Simulations \par}

        \vspace{0.5cm}
        {\large Alex Rawlings \par}

        \vfill
        {\large \today \par}
        \end{titlepage}
    """,
}
