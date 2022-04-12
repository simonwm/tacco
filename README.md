# TACCO: Transfer of Annotations to Cells and their COmbinations

TACCO is a python framework for working with categorical and compositional annotations for high-dimensional observations, in particular for transferring annotations from single cell to spatial transcriptomics data. An extensive ever expanding documentation is accessible [here](https://simonwm.github.io/tacco/).

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

