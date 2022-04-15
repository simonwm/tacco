TACCO 0.5.0 (2026-04-21)
========================

Features
--------

- Refactored the conda_env specification for RCTD, tangram, and SingleR to have more flexible environment specification as well as a fallback to use the current environment.

- Implement automatic casting for bisectioning `#26 <https://github.com/simonwm/tacco/issues/26>`__

- Improve scaling of :func:`tacco.tools.fill_regions` from N**2 to N log(N)

- Prepared ARM support `#24 <https://github.com/simonwm/tacco/issues/24>`__

Fixes
-----

- Catch edge cases in the anndata2R import functionality (integer count matrices, annotation columns containing NaNs, and categorical indices) relevant for the wrapped R tools

- Fix SingleR parameters for SingleR bioconda-version 2.8.0, which corresponds to bioconductor version 3.20.

- Restore dtype after :func:`tacco.tools.fill_regions`

- Arguments of :func:`tacco.tools.fill_regions` are extended and aligned with other functions in the framework

- Fix unintended side-effect of annotation with RCTD on supplied reference anndata

- Add compatibility of :func:`tacco.tools.spectral_clustering` with scipy 1.15

- Add compatibility of :func:`tacco.testing.assert_adata_equals` with np.ndarray objects in .obsm keys

- Fix sparsity test in :func:`tacco.tools._RCTD._round_X` when running :func:`tacco.tools.annotate_RCTD`, see `#30 <https://github.com/simonwm/tacco/pull/30>`__ thanks to @kmpf

- Fix `#31 <https://github.com/simonwm/tacco/issues/31>`__: compatibility with scipy 1.17 issue 

Breaking changes
----------------

- Expose RCTD filter parameters n_max_cells, min_UMI, and counts_MIN and deactivate these filters by default to make the filtering transparent on the python level.

- Migrate deprecated github action mamba-org/provision-with-micromamba to `mamba-org/setup-micromamba <https://github.com/mamba-org/setup-micromamba>`_. Cleaned up environment.yml in the process, so the old "clean" install using that yml file is not possible anymore._

Miscellaneous
-------------

- Note that numba>=0.62.0 is incompatible with umap-learn<=0.5.12, which influences scanpy's :func:`sc.pp.neighbors` results and thereby also taccos :func:`tacco.tool.find_regions`. With a clean current installation everything should work again. `#1216 <https://github.com/lmcinnes/umap/issues/1216>`__

