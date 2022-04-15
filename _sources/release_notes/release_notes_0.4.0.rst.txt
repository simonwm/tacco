TACCO 0.4.0 (2024-07-16)
========================

Fixes
-----

- Fix incompatibility with new scipy version 1.14.0 `#20 <https://github.com/simonwm/tacco/issues/20>`__

Breaking changes
----------------

- Adapted to work with anndata>=0.9.0 and pandas>=2.0.0. This involves changes in the handling of dtypes and (floating point) precision, which leads to "small" numerical changes in the results.

- Stronger preservation of input dtype in :func:`tacco.utils.split_beads` with downstream effects on :func:`tacco.tools.split_observations` and :func:`tacco.preprocessing.normalize_platform`.

- :func:`tacco.tools.enrichments`, :func:`tacco.tools.get_contributions`, :func:`tacco.tools.get_compositions`: Always work with float64 to avoid rounding errors as far as possible.

Miscellaneous
-------------

- :func:`tacco.tools.annotate`: Report clear error message for using bisectioning with integer data

- :func:`tacco.utils.row_scale`,:func:`tacco.utils.col_scale`: Report clear error message for rescaling integer data inplace with floating rescaling factors

- :func:`tacco.tools.enrichments`, :func:`tacco.tools.get_contributions`: Deprecate on-the-fly sample split in favour of explicit use of :func:`tacco.utils.split_spatial_samples`.

- Handeled pandas `FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.`

- Handeled anndata `FutureWarning: The dtype argument is deprecated and will be removed in late 2024.` Requires anndata>=0.9.0

- Handeled anndata `FutureWarning: Use anndata.concat instead of AnnData.concatenate, AnnData.concatenate is deprecated and will be removed in the future. See the tutorial for concat at: https://anndata.readthedocs.io/en/latest/concatenation.html`  

- Handeled sklearn `FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.`

