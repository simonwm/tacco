# TACCO: Transfer of Annotations to Cells and their COmbinations

TACCO is a python framework for working with categorical and compositional annotations for high-dimensional observations, in particular for transferring annotations from single cell to spatial transcriptomics data. TACCO comes with an extensive ever expanding [documentation](https://simonwm.github.io/tacco/) and a set of [example notebooks](https://github.com/simonwm/tacco_examples). If TACCO is useful for your research, you can cite [Nat Biotechnol (2023)](https://doi.org/10.1038/s41587-023-01657-3).

## How to install TACCO

### Clean

The simplest way to install TACCO is to create a clean environment with `conda` using the `environment.yml` file from the TACCO repository:

```
conda env create -f "https://raw.githubusercontent.com/simonwm/tacco/master/environment.yml"
```
(For older versions of `conda` one needs to download the `environment.yml` and use the local file for installation.)

### Conda

To install TACCO in an already existing environment, use `conda` to install from the `conda-forge` channel:

```
conda install -c conda-forge tacco
```

### Pip

It is also possible to install from pypi via `pip`:

```
pip install tacco
```

This is however not recommended. Unlike `conda`, `pip` cannot treat python itself as a package, so if you start with the wrong python version, you will run into errors with dependencies (e.g. at the time of writing, `mkl-service` is not available for python 3.10 and `numba` not for 3.11).

### Github

To access the most recent pre-release versions it is also possible to pip-install directly from github:

```
pip install tacco@git+https://github.com/simonwm/tacco.git
```

Obviously, this is not recomended for production environments.

## How to use TACCO

TACCO features a fast and straightforward API for the compositional annotation of one dataset, given as an anndata object `adata`, with a categorically annotated second dataset, given as an anndata object `reference`. The annotation is wrapped in a single function call

```
import tacco as tc
tc.tl.annotate(adata, reference, annotation_key='my_categorical_annotation', result_key='my_compositional_annotation')
```

where `'my_categorical_annotation'` is the name of the categorical `.obs` annotation in `reference` and `'my_compositional_annotation'` is the name of the new compositional `.obsm` annotation to be created in `adata`. There are many options for customizing this function to call e.g. external annotation tools, which are described in the [documentation of the `annotate` function](https://simonwm.github.io/tacco/_autosummary/tacco.tools.annotate.html#tacco.tools.annotate).

As the TACCO framework contains much more than a compositional annotation method (single-molecule annotation, object-splitting, spatial co-occurrence analysis, enrichments, visualization, ...), its [documentation](https://simonwm.github.io/tacco/) does not fit into a README.
