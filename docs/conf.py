"""Sphinx configuration for GraphNetz documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make ``graphnetz`` importable for autodoc.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------- #
# Project information
# --------------------------------------------------------------------------- #
project = "GraphNetz"
author = "Kleyton da Costa"
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001 (Sphinx requires this name)

try:
    from graphnetz.__about__ import __version__ as release
except Exception:  # pragma: no cover
    release = "0.0.0"
version = release

# --------------------------------------------------------------------------- #
# Extensions
# --------------------------------------------------------------------------- #
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",  # ::: admonitions
    "deflist",  # definition lists
    "tasklist",  # - [ ] checkboxes
    "smartquotes",
    "attrs_block",  # block-level attribute syntax
    "attrs_inline",
    "dollarmath",  # $...$ and $$...$$ LaTeX math
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pyg": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# --------------------------------------------------------------------------- #
# Theme: Alabaster
# --------------------------------------------------------------------------- #
html_theme = "alabaster"
html_title = "GraphNetz"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "logo": "logo.svg",
    "github_user": "kleyt0n",
    "github_repo": "graphnet",
    "github_button": True,
    "github_type": "star",
    "github_count": False,
    "description": "Statistically rigorous GNN benchmarking",
    "show_powered_by": False,
    "fixed_sidebar": True,
    "sidebar_width": "260px",
    "page_width": "960px",
}

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

html_show_sourcelink = False
html_show_sphinx = False
html_last_updated_fmt = "%Y-%m-%d"

# Pygments
pygments_style = "tango"

# Suppress noisy autodoc warnings on optional dependencies.
# ``ref.citation`` silences "Citation [Xxx] is not referenced" warnings for
# the model-paper labels that are part of each model's docstring References
# block but not cited inline.
suppress_warnings = ["autodoc.import_object", "ref.citation"]
