import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from .. import utils
from .. import get
from .. import preprocessing

def get_contributions(
    adata,
    value_key,
    group_key=None,
    sample_key=None,
    position_key=None,
    position_split=2,
    min_obs=0,
    value_location=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    reduction='sum',
    normalization='gmean',
    assume_counts=None,
    reads=False,
    counts_location=None,
):

    """\
    Get the contributions of groups.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs` (and `.obsm`).
        Can also be a :class:`~pandas.DataFrame` which is then used in place of
        `.obs`.
    value_key
        The `.obs`, `.obsm` key or `.var` index value i.e. gene with the values
        to determine the enrichment for. Can also be a list of genes and
        non-categorical `.obs` keys. If `None`, use all annotation available in
        `value_location` (see below).
    group_key
        The `.obs` key with categorical group information. If `None`, determine
        the contributions for the whole dataset and names this group 'all'.
    sample_key
        The `.obs` key with categorical sample information. If not `None`,
        the data is aggregated per sample otherwise as a whole.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. If `None`, no position splits are performed.
    position_split
        The number of splits per spatial dimension before enrichment. Can be a
        tuple with the spatial dimension as length to assign a different split
        per dimension. If `None`, no position splits are performed. See also
        `min_obs`.
    min_obs
        The minimum number of observations per sample: if less observations are
        available, the sample is not used. This also limits the number of
        `position_split` to stop splitting if the split would decrease the
        number of observations below this threshold.
    value_location
        The location of `value_key` within `adata`. Possible values are:
        
        - 'obs': `value_key` is a key in `.obs`
        - 'obsm': `value_key` is a key in `.obsm`
        - 'X': `value_key` is a index value in `.var`, i.e. a gene
        - `None`: find it automatically if possible
        
        Can also be a list of specifications if `value_key` is a list. If 
        `value_key` is `None`, all keys found in `value_location` are used.
    fillna
        NAN values in the data are replaced with this value. If `None`, the
        reduction and/or normalization operation handle the NANs, e.g. by
        ignoring them in a sum.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - `None`: use observations directly
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
          
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    reads
        Whether to weight the values by the total count per observation
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`. The counts are only used if `reads` is `True`.
        
    Returns
    -------
    A :class:`~pandas.DataFrame` containing the contributions of groups.
    
    """
    
    if isinstance(adata, sc.AnnData):
        adata_obs = adata.obs
        adata_obsm = adata.obsm
        adata_var = adata.var
    else:
        adata_obs = adata
        adata_obsm = None
        adata_var = None
    
    if group_key is not None:
        if group_key not in adata_obs.columns:
            raise ValueError('`group_key` "%s" is not in `adata.obs`!' % (group_key))
        if not hasattr(adata_obs[group_key], 'cat'):
            raise ValueError(f'`adata.obs[group_key]` has to be categorical, but `adata.obs["{group_key}"]` is not!')
    if sample_key is not None and not hasattr(adata_obs[sample_key], 'cat'):
        raise ValueError(f'`adata.obs[sample_key]` has to be categorical, but `adata.obs["{sample_key}"]` is not!')
    
    if value_key is None:
        possible_locations = ['obs','X']
        if isinstance(value_location, str) and value_location in possible_locations:
            if value_location == 'obs':
                value_key = adata_obs.columns
            else: # value_location == 'X':
                if adata_var is None:
                    raise ValueError(f'"X" was specified in `value_location`, which is not possible if no AnnData object is supplied as `adata`!')
                value_key = adata_var.index
        else:
            raise ValueError(f'If `value_key` is `None`, `value_location` has to be a single string out of {possible_locations}!')
        
    if isinstance(value_key, str):
        if value_location is not None and not isinstance(value_location, str):
            raise ValueError(f'`value_location` is neither `None` nor string, but `value_key` is a single specification!')
        found = []
        if value_key in adata_obs.columns and (value_location is None or value_location == 'obs'):
            found.append('obs')
            if hasattr(adata_obs[value_key], 'cat'):
                obs = pd.get_dummies(adata_obs[value_key])
                obs.columns.name = value_key
            else:
                obs = pd.DataFrame({value_key:adata_obs[value_key]})
                obs.columns.name = value_key
                obs.columns = obs.columns.astype('category')
        if adata_obsm is not None and value_key in adata_obsm and (value_location is None or value_location == 'obsm'):
            found.append('obsm')
            obs = pd.DataFrame(adata_obsm[value_key],copy=True)
            obs.columns.name = value_key
            obs.columns = obs.columns.astype('category')
            for k in obs.columns:
                if hasattr(obs[k], 'cat'):
                    raise ValueError(f'"{k}" from `.obsm[{value_key}]` is a categorical column, but all `.obsm` columns cannot be categorical!')
        if adata_var is not None and value_key in adata_var.index and (value_location is None or value_location == 'X'):
            found.append('X')
            X = adata[:,value_key].X
            if issparse(X):
                X = X.A
            obs = pd.DataFrame({value_key:X.flatten()},index=adata_obs.index)
            obs.columns.name = value_key
            obs.columns = obs.columns.astype('category')
        
        if len(found) < 1:
            raise ValueError(f'`value_key` "{value_key}" is not in `adata.obs`, `adata.obsm`, or `adata.var`!')
        elif len(found) > 1:
            raise ValueError(f'`value_key` "{value_key}" is not uniquely specified and was found in "{found}"!')
    else:
        if value_location is None or isinstance(value_location, str):
            value_location = [value_location] * len(value_key)
        value_location = pd.Index(value_location)
        value_key = pd.Index(value_key)
        
        # get all obs keys
        obs_keys = (value_location == 'obs') | (value_location.isna() & value_key.isin(adata_obs.columns))
        # get all var keys - if explicitly specified or possible
        var_keys = (value_location == 'X')
        if adata_var is not None:
            var_keys |= (value_location.isna() & value_key.isin(adata_var.index))
        elif var_keys.sum() > 0:
            raise ValueError(f'"X" was specified in `value_location`, which is not possible if no AnnData object is supplied as `adata`!')
        # get all obsm keys - does not make sense, so find them just to give an appropriate error message
        obsm_keys = (value_location == 'obsm')
        if adata_obsm is not None:
            obsm_keys |= (value_location.isna() & value_key.isin(adata_obsm) & (~value_key.isin(var_keys)) & (~value_key.isin(obs_keys)))
        if obsm_keys.any():
            raise ValueError(f'"{value_key[obsm_keys]}" from `value_key` "{value_key}" can only be interpreted as entries in `adata.obsm`, but `adata.obsm` entries cannot appear in a list possibly together with other keys!')
        if (obs_keys & var_keys).any():
            raise ValueError(f'"{value_key[obs_keys & var_keys]}" from `value_key` "{value_key}" are found in both `adata.obs` and `adata.var.index` and cannot be automatically assigned! Specify `value_location` for these keys!')
        if sum(obs_keys | var_keys) < len(value_key):
            raise ValueError(f'"{value_key[~(obs_keys & var_keys)]}" from `value_key` "{value_key}" were found neither in `adata.obs` nor `adata.var.index`!')
        
        obs = []
        if obs_keys.any():
            obs_from_obs = adata_obs[value_key[obs_keys]]
            for _value_key in obs_from_obs:
                if hasattr(obs_from_obs[_value_key], 'cat'):
                    raise ValueError(f'"{_value_key}" from `value_key` "{value_key}" is a categorical column in `adata.obs` and cannot appear in a list possibly together with other keys!')
            obs.append(obs_from_obs)
        if var_keys.any():
            obs_from_X = adata[:,value_key[var_keys]].X
            if issparse(obs_from_X):
                obs_from_X = obs_from_X.A
            obs_from_X = pd.DataFrame(obs_from_X, index=adata_obs.index, columns=value_key[var_keys])
            obs.append(obs_from_X)
        
        if len(obs) == 0:
            raise ValueError(f'`value_key` "{value_key}" did not contain any valid `adata.obs` or `adata.var.index` keys!')
        elif len(obs) == 1:
            obs = obs[0]
        else:
            obs = pd.concat(obs,axis=1)
            # preserve the original ordering
            obs = obs.reindex(columns=value_key)
        
        obs.columns.name = value_key.name if hasattr(value_key,'name') and value_key.name is not None else 'value'
        obs.columns = obs.columns.astype('category')
    
    if sample_key is None:
        sample_column = None
    else:
        sample_column = adata_obs[sample_key]
    
    if reads:
        counts = get.counts(adata, counts_location=counts_location, annotation=False, copy=False)
        totals = utils.get_sum(counts.X, axis=1)
        obs *= totals[:,None]
    
    # prepare positions aleady here to follow obs filtering in the following steps
    positions = None
    if position_key is not None and position_split is not None:
        positions = get.positions(adata, position_key)
    
    if group_key is None:
        groups = pd.Series(np.full(shape=adata_obs.shape[0],fill_value='all'),index=adata_obs.index)
    else:
        groups = adata_obs[group_key]

        if restrict_groups is not None:
            in_groups = groups.isin(restrict_groups)
            obs = obs.loc[in_groups]
            if positions is not None:
                positions = positions.loc[in_groups]
            if len(obs) == 0:
                raise ValueError(f'Restricting the groups in {group_key} to {restrict_groups} resulted in 0 observations left!')
            groups = groups[in_groups].cat.remove_unused_categories()

    if restrict_values is not None:
        obs = obs[restrict_values]
        if hasattr(obs.columns, 'remove_unused_categories'):
            obs.columns = obs.columns.remove_unused_categories()
    
    # divide spatial samples spatially into subsamples: keeps all the correlation structure
    if position_key is not None and position_split is not None:
        
        sample_column = utils.spatial_split(positions, position_key=positions.columns, sample_key=sample_column, position_split=position_split, min_obs=min_obs)
            
    else: # filter out too small samples
        if sample_column is None:
            if len(obs) < min_obs:
                obs = obs.iloc[[]]
    
    #if len(obs) == 0:
    #    raise ValueError(f'Restricting the samples to ones with at least {min_obs} observables resulted in 0 observations left!')
    
    if fillna is not None:
        obs = obs.fillna(fillna)
    
    #groups = groups[obs.index]
    grouping = groups if sample_column is None else [groups, sample_column]#.loc[obs.index]]
    
    if normalization in ['gmean','clr'] and assume_counts is None:
        if reads:
            assume_counts = True
        else:
            assume_counts = preprocessing.check_counts_validity(obs.to_numpy(), raise_exception=False)
    
    def _normalize(x):
        if reduction is None:
            sums = x
        elif isinstance(reduction, str):
            if reduction == 'sum':
                sums = x.sum(axis=0, skipna=True)
            elif reduction == 'mean':
                sums = x.mean(axis=0, skipna=True)
            elif reduction == 'median':
                sums = x.median(axis=0, skipna=True)
            else:
                raise ValueError('`reduction` "%s" is not implemented.' % reduction)
        else:
            try:
                sums = reduction(x)
            except Exception as e:
                raise ValueError('The supplied `reduction` is neither string nor a working callable!')
            
        if normalization is None:
            return sums
        elif isinstance(normalization, str):
            if normalization == 'sum':
                # normalize total to 1 for each groupXsample
                #factor = sums.to_numpy().sum(axis=-1, skipna=True)
                factor = np.nansum(sums.to_numpy(), axis=-1)
            elif normalization == 'percent':
                # normalize total to 1 for each groupXsample
                factor = np.nansum(sums.to_numpy(), axis=-1) / 100
            elif normalization in ['gmean','clr']:
                # normalize by geometric mean for each groupXsample
                sums_le_0 = ~(sums>0) # also includes nans
                if (~sums_le_0.to_numpy()).sum() == 0: # no positive values, so the normalization cant do much about it
                    factor = 1
                else:
                    if assume_counts:
                        sums = sums.copy()
                        sums += 1
                    elif sums_le_0.to_numpy().sum() > 0:
                        min_sums = sums[~sums_le_0].to_numpy().min()
                        sums = sums.copy()
                        sums[sums_le_0] = min_sums * 1e-3
                    factor = stats.gmean(sums,axis=-1)
            elif normalization in sums.index:
                factor = sums[normalization]
            else:
                raise ValueError('`normalization` "%s" is not implemented.' % normalization)
            if hasattr(factor,'shape') and len(factor.shape) == 1:
                factor = factor[:,None]
            if normalization == 'clr':
                return np.log(sums / factor)
            else:
                return sums / factor
        else:
            try:
                return normalization(sums)
            except Exception as e:
                raise ValueError('The supplied `normalization` is neither string nor a working callable!')
    
    compositions = obs.groupby(grouping).apply(_normalize)
    
    if len(compositions.index) == len(groups.index) and (compositions.index == groups.index).all():
        compositions.index = pd.MultiIndex.from_arrays([groups,pd.Series(groups.index,index=groups.index)])
    
    compositions = compositions.dropna()
    
    return compositions

def get_compositions(
    adata,
    value_key,
    group_key=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    reads=False,
    counts_location=None,
):

    """\
    Get the compositions of groups.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs`.
    value_key
        The `.obs` or `.obsm` key with the values to determine the
        enrichment for.
    group_key
        The `.obs` key with categorical group information. If `None`, determine
        the contributions for the whole dataset and names this group 'all'.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    reads
        Whether to weight the values by the total count per observation
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`. The counts are only used if `reads` is `True`.
        
    Returns
    -------
    A :class:`~pandas.DataFrame` containing the compositions of groups.
    
    """
    
    return get_contributions(
        adata=adata,
        value_key=value_key,
        group_key=group_key,
        sample_key=None,
        fillna=fillna,
        restrict_groups=restrict_groups,
        restrict_values=restrict_values,
        reduction='sum',
        normalization='sum',
        reads=reads,
        counts_location=counts_location,
    )

def enrichments(
    adata,
    value_key,
    group_key,
    sample_key=None,
    position_key=None,
    position_split=2,
    reference_group=None,
    min_obs=0,
    value_location=None,
    p_corr='fdr_bh',
    method='mwu',
    n_boot=0,
    direction='both',
    reduction=None,
    normalization=None,
    assume_counts=None,
    fillna=None,
    restrict_groups=None,
    restrict_values=None,
    reads=False,
    counts_location=None,
    ):

    """\
    Find enrichments in groups.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs` (and `.obsm`).
        Can also be a :class:`~pandas.DataFrame` which is then used in place of
        `.obs`.
    value_key
        The `.obs`, `.obsm` key or `.var` index value i.e. gene with the values
        to determine the enrichment for. Can also be a list of genes and
        non-categorical `.obs` keys. If `None`, use all annotation available in
        `value_location` (see below).
    group_key
        The `.obs` key with categorical group information.
    sample_key
        The `.obs` key with categorical sample information. If `None`,
        the enrichment is calculated on an observation level, otherwise on
        averaged quantities per sample. See parameters `normalization` and
        `reduction` for details.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. If `None`, no position splits are performed.
    position_split
        The number of splits per spatial dimension before enrichment. Can be a
        tuple with the spatial dimension as length to assign a different split
        per dimension. If `None`, no position splits are performed. See also
        `min_obs`.
    reference_group
        The particular group value to which all other groups should be
        compared. This group will be compared to the rest. If `None`, all
        groups are compared in a 1-vs-rest scheme.
    min_obs
        The minimum number of observations per sample: if less observations are
        available, the sample is not used. This also limits the number of
        `position_split` to stop splitting if the split would decrease the
        number of observations below this threshold.
    value_location
        The location of `value_key` within `adata`. Possible values are:
        
        - 'obs': `value_key` is a key in `.obs`
        - 'obsm': `value_key` is a key in `.obsm`
        - 'X': `value_key` is a index value in `.var`, i.e. a gene
        - `None`: find it automatically if possible
        
        Can also be a list of specifications if `value_key` is a list. If 
        `value_key` is `None`, all keys found in `value_location` are used.
    p_corr
        The name of the p-value correction method to use. Possible values are
        the ones available in
        :func:`~statsmodels.stats.multitest.multipletests`. If `None`, no
        p-value correction is performed.
    method
        Specification of methods to use for enrichment. Available are:
        
        - 'fisher': Fishers exact test; only for categorical values. Ignores
          the `reduction` and `normalization` arguments.
        - 'mwu': MannWhitneyU test
        - 't': Student's t test
        - 'welch': Welch's t test
        
    n_boot
        The number of bootstrap samples which are included in addition to the
        real samples. Working with bootstrap samples is only implemented for
        the t tests.
    direction
        What should be tested for. This influences the multiple testing
        correction. Available options are:
        
        - 'enrichment': Test only for enrichment
        - 'purification': Test only for purification
        - 'both': Test for both
        
    reduction
        The reduction to apply on each (group,sample) subset of the data.
        Possible values are:
        
        - 'sum': sum of the values over observations
        - 'mean': mean of the values over observations
        - 'median': median of the values over observations
        - `None`: use observations directly
        - a callable mapping a :class:`~pandas.DataFrame` to its reduced
          counterpart
          
    normalization
        The normalization to apply on each reduced (group,sample) subset of the
        data. Possible values are:
        
        - 'sum': normalize values by their sum (yields fractions)
        - 'percent': like 'sum' scaled by 100 (yields percentages)
        - 'gmean': normalize values by their geometric mean (yields
          contributions which make more sense for enrichments than fractions,
          due to zero-sum issue; see :func:`~tacco.tools.enrichments`)
        - 'clr': "Center logratio transform"; like 'gmean' with additional log
          transform; makes the distribution more normal and better suited for t
          tests
        - `None`: no normalization
        - a value name from `value_key`: all values are normalized to this
          contribution
        - a callable mapping a :class:`~pandas.DataFrame` to its normalized
          counterpart
    
    assume_counts
        Ony relevant for `normalization=='gmean'` and `normalization=='clr'`;
        whether to regularize zeros by adding a pseudo count of 1 or by
        replacing them by 1e-3 of the minimum value. If `None`, check whether
        the data are consistent with count data and assume counts accordingly,
        except if `reads==True`, then also `assume_counts==True`.
    fillna
        If `None`, observation containing NA in the values are filtered.
        Else, NA values are replaced with this value.
    restrict_groups
        A list-like containing the groups within which the enrichment analysis
        is to be done. If `None`, all groups are included.
    restrict_values
        A list-like containing the values within which the enrichment analysis
        is to be done. If `None`, all values are included. Works only for
        categorical values.
    reads
        Whether to weight the values by the total count per observation
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`. The counts are only used if `reads` is `True`.
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """
    
    adata_obs = adata.obs if hasattr(adata, 'obs') else adata
    
    if direction == 'both':
        alternative = ['greater','less']
    elif direction == 'enrichment':
        alternative = ['greater']
    elif direction == 'purification':
        alternative = ['less']
    else:
        raise ValueError(f'`direction` can only be one of {["enrichment","purification","both"]!r}, but is {direction!r}!')
    
    boot_methods = ['t', 'welch']
    if n_boot > 0 and method not in boot_methods:
        raise ValueError(f'`n_boot` can only be larger than 0 for the methods {boot_methods}!')
    
    if method == 'fisher':
        if sample_key is not None:
            raise ValueError(f'For method "fisher", `sample_key` can only be `None`!')
        if reduction is not None and reduction != 'sum':
            raise ValueError(f'For method "fisher", `reduction` is overridden by `"sum"`!')
        if normalization is not None:
            raise ValueError(f'For method "fisher", `normalization` can only be `None`!')
    else:
        if (sample_key is not None and reduction is None) or (sample_key is None and reduction is not None):
            print(f'Having only one of `sample_key` and `reduction` not `None` is of limited use. You might want to read the doc and double check whether you really want that!')
    
    if method == 'fisher':
        contributions = get_contributions(
            adata,
            value_key=value_key,
            group_key=group_key,
            sample_key=None,
            position_key=None,
            position_split=None,
            min_obs=min_obs,
            value_location=value_location,
            fillna=fillna,
            restrict_groups=restrict_groups,
            restrict_values=restrict_values,
            reduction='sum',
            normalization=None,
            assume_counts=assume_counts,
            reads=reads,
            counts_location=counts_location,
        )
        test_method = lambda c: utils.fishers_exact(c, alternative=alternative)
        
    elif method == 'mwu':
        contributions = get_contributions(
            adata,
            value_key=value_key,
            group_key=group_key,
            sample_key=sample_key,
            position_key=position_key,
            position_split=position_split,
            min_obs=min_obs,
            value_location=value_location,
            fillna=fillna,
            restrict_groups=restrict_groups,
            restrict_values=restrict_values,
            reduction=reduction,
            normalization=normalization,
            assume_counts=assume_counts,
            reads=reads,
            counts_location=counts_location,
        )
        test_method = lambda c: utils.mannwhitneyu(c, alternative=alternative)
        
    elif method == 't':
        contributions = get_contributions(
            adata,
            value_key=value_key,
            group_key=group_key,
            sample_key=sample_key,
            position_key=position_key,
            position_split=position_split,
            min_obs=min_obs,
            value_location=value_location,
            fillna=fillna,
            restrict_groups=restrict_groups,
            restrict_values=restrict_values,
            reduction=reduction,
            normalization=normalization,
            assume_counts=assume_counts,
            reads=reads,
            counts_location=counts_location,
        )
        test_method = lambda c: utils.studentttest(c, alternative=alternative, n_boot=n_boot)
        
    elif method == 'welch':
        contributions = get_contributions(
            adata,
            value_key=value_key,
            group_key=group_key,
            sample_key=sample_key,
            position_key=position_key,
            position_split=position_split,
            min_obs=min_obs,
            value_location=value_location,
            fillna=fillna,
            restrict_groups=restrict_groups,
            restrict_values=restrict_values,
            reduction=reduction,
            normalization=normalization,
            assume_counts=assume_counts,
            reads=reads,
            counts_location=counts_location,
        )
        test_method = lambda c: utils.welchttest(c, alternative=alternative, n_boot=n_boot)

    else:
        raise ValueError(f'The `method` "{method}" is not implemented!')
    
    if reference_group is None:
        result = test_method(contributions)
        
    else: # test all groups against the reference group - and the reference against the rest
        results = []
        for group in contributions.index.get_level_values(0).unique():
            if group == reference_group:
                
                result = test_method(contributions)
                result = result[result['group'] == group].copy()
                result['group'] = f'{group} VS rest'
                results.append(result)
                
            else:
                result = test_method(contributions.loc[[reference_group,group]])
                result = result[result['group'] == group].copy()
                result['group'] = f'{group} VS {reference_group}'
                results.append(result)
                
        result = pd.concat(results,axis=0)
        result['group'] = result['group'].astype(pd.CategoricalDtype(result['group'].unique(),ordered=True))

    plabel = f'p_{method}'

    value_label = contributions.columns.name
    result.rename(columns={'p':plabel,'alternative':'enrichment','group':group_key,'value':value_label},inplace=True)
    if p_corr is not None:
        if len(result) > 0:
            result[plabel + '_' + p_corr] = multipletests(result[plabel], alpha=0.05, method=p_corr)[1]
        else:
            result[plabel + '_' + p_corr] = 1
    result['enrichment'] = result['enrichment'].astype(str).map({'greater':'enriched','less':'purified'}).astype('category') # use reasonable dtype
    
    return result
