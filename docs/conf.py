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

# # need to assign some names here, otherwise autodoc won't document these classes,
# # and will instead just say 'alias of ...'

# from vamm.cpp import EM

# EM.__name__ = "EM"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "VAMM"
copyright = "2026, ML Lab UOL & AI Lab UIBK"
author = "Sebastian Salwig, Till Kahlke"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    # "sphinx.ext.napoleon",
    "numpydoc",
]

autodoc_typehints = "none"
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"  # "sphinx_book_theme"

html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 4,
    "navigation_with_keys": True,
}


# def process_signature(app, what, name, obj, options, signature, return_annotation):
#     new_signature = signature
#     new_return_annotation = return_annotation
#     autodoc_typehints = "none"
#     # remove "self:" from function signature
#     # for some reason this appease in pybind11 methods
#     if signature is not None and "self:" in signature:
#         new_signature = f"({', '.join(signature.strip('()').split(', ')[1:])})"
#     return new_signature, new_return_annotation
#     # will be rendered to method(new_signature) -> new_return_annotation


def skip_properties(app, what, name, obj, skip, options):
    if what == "class" and isinstance(obj, property):
        return True
    return skip


def setup(app):
    # app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-skip-member", skip_properties)
    # app.add_css_file("custom.css")
