# conf.py — SpectralBrain documentation
#
# Sphinx + pydata-sphinx-theme + sphinx-design (home cards)
#         + myst-nb       (Markdown pages AND pre-rendered tutorial notebooks)
#         + sphinx-gallery (light, build-time example gallery with figures)
#         + autosummary    (API reference generated from NumPy-style docstrings)
#
# Build locally:   make -C docs html        (then open docs/build/html/index.html)
# Builds on Read the Docs via ../.readthedocs.yaml
from __future__ import annotations

import os
from datetime import datetime

# -- Path / package version --------------------------------------------------
# Resolve the installed package version without importing heavy submodules.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("spectralbrain")
except Exception:  # editable / not-yet-installed fallback
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- Project information -----------------------------------------------------
project = "SpectralBrain"
author = "Rodrigo Debona"
copyright = f"{datetime.now():%Y}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core autodoc stack
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # NumPy / Google docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # Authoring / UX
    "myst_nb",                    # MyST Markdown + notebook rendering (includes myst_parser)
    "sphinx_design",              # grids, cards, tabs on the home page
    "sphinx_copybutton",          # copy-to-clipboard on code blocks
    # Example gallery
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source files: Markdown (and notebooks) are handled by myst-nb.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- MyST / myst-nb ----------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives (sphinx-design cards/grids)
    "dollarmath",    # $inline$ and $$display$$ math
    "amsmath",       # LaTeX environments
    "deflist",
    "substitution",
    "linkify",
]
myst_heading_anchors = 3

# Tutorials ship as notebooks with committed outputs (heavy: real brains / GPU).
# Render them AS-IS — never execute on the build machine (pymc-style gallery).
nb_execution_mode = "off"
nb_merge_streams = True

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# Heavy / optional dependencies are mocked so the API reference builds on Read
# the Docs without a GPU, FreeSurfer, or the bayesian/viz extras installed.
autodoc_mock_imports = [
    "torch", "torchvision", "cupy", "jax", "jaxlib",
    "pymc", "pytensor", "nutpie", "numpyro", "blackjax", "arviz",
    "vedo", "scienceplots", "hippunfold_plot", "hippomaps",
    "dipy", "nilearn", "templateflow", "pybids", "bids",
    "neuroCombat", "neuroHarmonize", "neuroharmonize",
]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
}

# -- sphinx-gallery (the airy example grid) ----------------------------------
# Examples are LIGHT by design (synthetic icosphere, small k) so they execute
# in seconds during the build. PyVista renders are captured off-screen.
import pyvista  # noqa: E402

pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
if os.environ.get("READTHEDOCS") == "True":
    # Read the Docs has no display; start a virtual framebuffer for VTK.
    try:
        pyvista.start_xvfb()
    except Exception:
        pass

from sphinx_gallery.sorting import FileNameSortKey  # noqa: E402

sphinx_gallery_conf = {
    "examples_dirs": "examples",        # .py example scripts (source)
    "gallery_dirs": "auto_examples",    # generated gallery (output)
    "filename_pattern": r"plot_",
    "within_subsection_order": FileNameSortKey,
    "image_scrapers": ("matplotlib", "pyvista"),
    "remove_config_comments": True,
    "download_all_examples": False,
    "thumbnail_size": (400, 400),
    "default_thumb_file": None,
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "SpectralBrain"
html_show_sourcelink = False

_repo = "https://github.com/rdneuro/spectralbrain"
html_theme_options = {
    "icon_links": [
        {"name": "GitHub", "url": _repo, "icon": "fa-brands fa-github"},
        {"name": "PyPI", "url": "https://pypi.org/project/spectralbrain/",
         "icon": "fa-brands fa-python"},
    ],
    "use_edit_page_button": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "show_nav_level": 1,
    "logo": {"text": "SpectralBrain"},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_context = {
    "github_user": "rdneuro",
    "github_repo": "spectralbrain",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "light",
}

# Friendlier section landing pages.
html_sidebars = {
    "index": [],
    "getting_started/index": [],
}
