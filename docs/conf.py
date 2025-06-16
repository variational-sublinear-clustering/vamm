# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

for p in Path("../build").glob("lib.*"):
    sys.path.insert(0, str(p.resolve()))
sys.path.insert(0, os.path.abspath(".."))

# need to assign some names here, otherwise autodoc won't document these classes,
# and will instead just say 'alias of ...'

from vamm.cpp import EM

EM.__name__ = "EM"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Variational Accelerated Mixture Models"
copyright = "2024, Machine Learning Lab UOL"
author = "Machine Learning Lab UOL"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"  # "sphinx_book_theme"
html_static_path = []


def process_signature(app, what, name, obj, options, signature, return_annotation):
    new_signature = signature
    new_return_annotation = return_annotation
    # remove "self:" from function signature
    # for some reason this appease in pybind11 methods
    if signature is not None and "self:" in signature:
        new_signature = f"({', '.join(signature.strip('()').split(', ')[1:])})"
    return new_signature, new_return_annotation
    # will be rendered to method(new_signature) -> new_return_annotation


def setup(app):
    app.connect("autodoc-process-signature", process_signature)
