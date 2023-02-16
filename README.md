# TACCO: Transfer of Annotations to Cells and their COmbinations

TACCO is a python framework for working with categorical and compositional annotations for high-dimensional observations, in particular for transferring annotations from single cell to spatial transcriptomics data. TACCO comes with an extensive ever expanding [documentation](https://simonwm.github.io/tacco/) and a set of [example notebooks](https://github.com/simonwm/tacco_examples). If TACCO is useful for your research, you can cite [bioRxiv (2022)](https://doi.org/10.1101/2022.10.02.508471).

## How to install TACCO

The simplest way to install TACCO is to create a clean environment with `conda` using the `environment.yml` file from the TACCO repository:

```
conda env create -f "https://raw.githubusercontent.com/simonwm/tacco/master/environment.yml"
```
(For older versions of `conda` one needs to download the `environment.yml` and use the local file for installation.)

To install TACCO in an already existing environment, one can use `pip` to directly install the latest release from github:

```
pip install tacco@git+https://github.com/simonwm/tacco.git
```

## How to use TACCO

TACCO features a fast and straightforward API for the compositional annotation of one dataset, given as an anndata object `adata`, with a categorically annotated second dataset, given as an anndata object `reference`. The annotation is wrapped in a single function call

```
import tacco as tc
tc.tl.annotate(adata, reference, annotation_key='my_categorical_annotation', result_key='my_compositional_annotation')
```

where `'my_categorical_annotation'` is the name of the categorical `.obs` annotation in `reference` and `'my_compositional_annotation'` is the name of the new compositional `.obsm` annotation to be created in `adata`. There are many options for customizing this function to call e.g. external annotation tools, which are described in the [documentation of the `annotate` function](https://simonwm.github.io/tacco/_autosummary/tacco.tools.annotate.html#tacco.tools.annotate).

As the TACCO framework contains much more than a compositional annotation method (single-molecule annotation, object-splitting, spatial co-occurrence analysis, enrichments, visualization, ...), its [documentation](https://simonwm.github.io/tacco/) does not fit into a README.
