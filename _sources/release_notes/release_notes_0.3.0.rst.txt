TACCO 0.3.0 (2024-01-10)
========================

Features
--------

- :func:`tacco.plots.subplots`: support for changing the dpi and forwarding kwargs to :func:`matplotlib.pyplot.subplots`

- :func:`tacco.plots.dotplot`: new swap_axes argument

- :func:`tacco.utils.split_spatial_samples`: more flexible and intuitive reimplementation for spatial splitting samples which can account explicitly for spatial correlations; deprecates the :func:`tacco.utils.spatial_split` function

- :func:`tacco.tools.setup_orthology_converter`, :func:`tacco.tools.run_orthology_converter`: orthology conversion between species

- :func:`tacco.get.get_data_from_key`: general getter to retrieve data from an anndata given a data path

- :func:`tacco.get.get_positions`: support for data paths

Fixes
--------

- :func:`tacco.tools.annotate`: reconstruction_key now honors max_annotation. So :func:`tacco.tools.split_observations` works with reconstruction_key as well. This fixes issue `#9 <https://github.com/simonwm/tacco/issues/9>`__ .

- :func:`tacco.tools.split_observations`: fixed map_obsm_keys parameter

- :func:`tacco.plots.significances`: fix using pre-supplied ax, fix not significant annotated but significance colored cells, fix future warning, work for data with enrichment and without depletion

- :func:`tacco.plots.dotplot`: catch edge case with gene-group combinations without a match in "marks"

- :func:`tacco.plots.co_occurrence`: fixed bug for multiple anndatas in the input

- :func:`tacco.plots.co_occurrence_matrix`: fixed bug for restrict_intervals=None

- :func:`tacco.tools.annotate`: multi_center=1 changed so it now behaves the same as multi_center=0/None, fix FutureWarning from kmeans

- :func:`tacco.tools.get_contributions`: fix FutureWarning from groupby

- :func:`tacco.plots.co_occurrence`, :func:`tacco.plots.co_occurrence_matrix`: coocurrence plots now follow the show_only and show_only_center order

Documentation
-------------

- Add release notes

- Add visium example to address `#8 <https://github.com/simonwm/tacco/issues/8>`__

Miscellaneous
-------------

- Switch from setup.cfg to pyproject.toml

- Generalization of benchmarking to support conda-forge time

- Expanded testing

- Remove duplication in NOTICE.md
