# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

import tacco

project = 'TACCO'
copyright = '2022, Broad Institute'
#author = tacco.__author__

# The full version, including alpha/beta/rc tags
#release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    #'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = dict(
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
    matplotlib=('https://matplotlib.org/stable/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy/', None),
)

## Enable "show on github" button
#html_context = {
#  'display_github': True,
#  'github_user': 'simonwm',
#  'github_repo': 'tacco',
#  'github_version': 'devel/docs/',
#}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
#autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False # Remove namespaces from class/method signatures
autosummary_imported_members = True


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Exclusions
# To exclude a module, use autodoc_mock_imports. Note this may increase build time, a lot.
# (Also, when installing on readthedocs.org, we omit installing Tensorflow and
# Tensorflow Probability so mock them here instead.)
#autodoc_mock_imports = [
    # 'tensorflow',
    # 'tensorflow_probability',
#]
# To exclude a class, function, method or attribute, use autodoc-skip-member. (Note this can also
# be used in reverse, ie. to re-include a particular member that has been excluded.)
# 'Private' and 'special' members (_ and __) are excluded using the Jinja2 templates; from the main
# doc by the absence of specific autoclass directives (ie. :private-members:), and from summary
# tables by explicit 'if-not' statements. Re-inclusion is effective for the main doc though not for
# the summary tables.
def autodoc_skip_member_callback(app, what, name, obj, skip, options):
    excluded_modules = ['numpy','scipy','matplotlib','sklearn','numba','statsmodels','difflib'] # completely exclude modules which are not essential or provide external imported members
    exclusions = {
        'module': [],
        'class': [],
        'exception': [],
        'function': [],
        'method': [],
        'attribute': [],
        'data': [],
        'property': [],
    }
    if skip != True:
        if hasattr(obj,'__module__') and obj.__module__ is not None:
            for ex_mod in excluded_modules:
                if obj.__module__.startswith(ex_mod):
                    return True
        if name in exclusions[what]:
            return True
    return skip
def setup(app):
    # Entry point to autodoc-skip-member
    app.connect("autodoc-skip-member", autodoc_skip_member_callback)

# -- Options for HTML output -------------------------------------------------

# Readthedocs theme
# on_rtd is whether on readthedocs.org, this line of code grabbed from docs.readthedocs.org...
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_css_files = [
    "readthedocs-custom.css",
    "notebook_hacks.css",
] # Override some CSS settings

# Pydata theme
#html_theme = "pydata_sphinx_theme"
#html_logo = "_static/logo-company.png"
#html_theme_options = { "show_prev_next": False}
#html_css_files = ['pydata-custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/logo.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
