# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Mock imports for dependencies that may not be available during doc build
autodoc_mock_imports = [
    'jax',
    'jax.numpy',
    'jax.scipy',
    'jax.scipy.stats',
    'jax.scipy.special',
    'jax.scipy.signal',
    'jax.image',
    'jax.random',
    'jaxlib',
    'numpyro',
    'numpyro.handlers',
    'numpyro.distributions',
    'numpyro.distributions.transforms',
    'numpyro.infer',
    'numpyro.infer.initialization',
    'numpyro.infer.reparam',
    'numpyro.infer.util',
    'arviz',
    'astropy',
    'astropy.io',
    'astropy.io.fits',
    'astropy.io.ascii',
    'astropy.table',
    'astropy.wcs',
    'astropy.coordinates',
    'astropy.cosmology',
    'astropy.convolution',
    'astropy.modeling',
    'astropy.modeling.models',
    'astropy.units',
    'photutils',
    'photutils.aperture',
    'photutils.background',
    'photutils.segmentation',
    'photutils.centroids',
    'scipy',
    'scipy.ndimage',
    'scipy.constants',
    'scipy.interpolate',
    'scipy.signal',
    'matplotlib',
    'matplotlib.pyplot',
    'pysersic',
    'skimage',
    'skimage.morphology',
    'skimage.filters',
    'skimage.measure',
    'xarray',
    'pandas',
    'numpy',
    'corner',
    'jax_cosmo',
    'jax_cosmo.scipy',
    'jax_cosmo.scipy.interpolate',
    'PIL',
    'reproject'
]

# -- Project information -----------------------------------------------------

project = 'astro-geko'
copyright = '2024-2025, Geko Contributors'
author = 'Geko Contributors'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'numpydoc',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = '_static/geko_logo.png'

html_theme_options = {
    "repository_url": "https://github.com/yourusername/geko",  # Update when public
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "logo": {
        "image_light": "_static/geko_logo.png",
        "image_dark": "_static/geko_logo.png",
    }
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Show warnings for autodoc
autodoc_warningiserror = False
autodoc_inherit_docstrings = True
add_module_names = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpyro': ('https://num.pyro.ai/en/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Numpydoc settings
numpydoc_show_class_members = False

# MyST-NB settings - disable notebook execution
nb_execution_mode = "off"
