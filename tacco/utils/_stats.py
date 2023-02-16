import numpy as np
import pandas as pd
import scipy.stats as stats
from . import _mannwhitneyu

def fishers_exact(
    df,
    alternative=['greater','less']
):

    """\
    Perform Fisher's exact test. Tests all columns in a 1 VS rest scheme.
    
    Parameters
    ----------
    df
        A contingency table as :class:`~pandas.DataFrame`, with groups in the
        rows and values in the columns.
    alternative
        The alternative for the test. Can also be a list of alternatives.
        Available are:
        
        - 'greater'
        - 'less'
        - 'two-sided'
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """

    if isinstance(alternative, str):
        alternative = [alternative]
    for a in alternative:
        if a not in ['greater','less','two-sided']:
            raise ValueError(f'`alternative` can only be "greater","less","two-sided", but got {a}!')
    
    res = []
    for values_val in df.columns:
        not_values_val = df.columns != values_val
        _values_val = ~not_values_val
        values_mat = np.array([_values_val,not_values_val])
        
        for groups_val in df.index:
            not_groups_val = df.index != groups_val
            _groups_val = ~not_groups_val
            groups_mat = np.array([_groups_val,not_groups_val])
            
            table = (groups_mat @ df.to_numpy() @ values_mat.T).T
            
            for a in alternative:
                p = stats.fisher_exact(table,alternative=a)[1] # anti-diagonal dominant
                res.append((values_val,groups_val,a,p))
    res_df = pd.DataFrame(res, columns=['value', 'group', 'alternative', 'p'])
    
    res_df['group'] = res_df['group'].astype(df.index.dtype) # get same dtype
    res_df['value'] = res_df['value'].astype(df.columns.dtype) # get same dtype
    res_df['alternative'] = res_df['alternative'].astype('category') # use reasonable dtype

    return res_df

def _wrap_test(
    test_function,
    df,
    alternative=['greater','less'],
):
    """\
    Wraps test functions in a convenient dataframe compatible format.
    
    Parameters
    ----------
    test_function
        The test function to wrap. It must return the p value and have the
        signature `f(samples0, sample1, alternative)`, where `samples0` and
        `sample1` are the two samples to test for equality, and `alternative`
        is one of the strings given below.
    df
        A table of samples as :class:`~pandas.DataFrame`, with groups in the
        rows and values in the columns. If the index is a multilevel index, the
        second level is interpreted as sample annotation.
    alternative
        The alternative for the test. Can also be a list of alternatives.
        Available are:
        
        - 'greater'
        - 'less'
        - 'two-sided'
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """

    if isinstance(alternative, str):
        alternative = [alternative]
    for a in alternative:
        if a not in ['greater','less','two-sided']:
            raise ValueError(f'`alternative` can only be "greater","less","two-sided", but got {a}!')
    
    res = []
    if isinstance(df.index, pd.MultiIndex):
        df_index = df.index.get_level_values(0)
        group_vals = df.index.levels[0]
    else:
        df_index = df.index
        group_vals = df.index
    
    _groups_vals = pd.get_dummies(df_index).astype(bool)
    for values_val in df.columns:
        for groups_val in group_vals:
            _groups_val = _groups_vals[groups_val]
            not_groups_val = ~_groups_val
            
            samples0 = df[values_val].to_numpy()[_groups_val]
            samples1 = df[values_val].to_numpy()[not_groups_val]

            if len(samples0) == 0 or ((samples0 == samples0[0]).all() and (samples1 == samples0[0]).all()):
                continue
                
            for a in alternative:
                p = test_function(samples0,samples1,alternative=a)
                res.append((values_val,groups_val,a,p))
    
    res_df = pd.DataFrame(res, columns=['value', 'group', 'alternative', 'p'])
    
    if isinstance(df.index, pd.MultiIndex):
        res_df['group'] = res_df['group'].astype(df.index.levels[0].dtype) # get same dtype
    else:
        res_df['group'] = res_df['group'].astype(df.index.dtype) # get same dtype
    res_df['value'] = res_df['value'].astype(df.columns.dtype) # get same dtype
    res_df['alternative'] = res_df['alternative'].astype('category') # use reasonable dtype
    
    return res_df

def mannwhitneyu(
    df,
    alternative=['greater','less'],
):
    """\
    Perform Mann-Whitney-U test. Tests all columns.
    
    Parameters
    ----------
    df
        A table of samples as :class:`~pandas.DataFrame`, with groups in the
        rows and values in the columns. If the index is a multilevel index, the
        second level is interpreted as sample annotation.
    alternative
        The alternative for the test. Can also be a list of alternatives.
        Available are:
        
        - 'greater'
        - 'less'
        - 'two-sided'
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """
    def test_f(samples0, samples1, alternative):
        return _mannwhitneyu.mannwhitneyu(samples0,samples1,alternative=alternative,exact=True)[1]
    return _wrap_test(test_f, df, alternative=alternative)

def studentttest(
    df,
    alternative=['greater','less'],
    n_boot=0,
):
    """\
    Perform Student's t test. Tests all columns.
    
    Parameters
    ----------
    df
        A table of samples as :class:`~pandas.DataFrame`, with groups in the
        rows and values in the columns. If the index is a multilevel index, the
        second level is interpreted as sample annotation.
    alternative
        The alternative for the test. Can also be a list of alternatives.
        Available are:
        
        - 'greater'
        - 'less'
        - 'two-sided'
        
    n_boot
        The number of bootstrap samples which are included in addition to the
        real samples.
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """
    def test_f(samples0, samples1, alternative):
        
        nobs0 = len(samples0) / (1+n_boot)
        nobs1 = len(samples1) / (1+n_boot)
        std0 = np.std(samples0, axis=0, ddof=1+n_boot)
        std1 = np.std(samples1, axis=0, ddof=1+n_boot)
        mean0 = np.mean(samples0, axis=0)
        mean1 = np.mean(samples1, axis=0)
        
        return stats.ttest_ind_from_stats(mean0, std0, nobs0, mean1, std1, nobs1, equal_var=True, alternative=alternative)[1]
    
    return _wrap_test(test_f, df, alternative=alternative)

def welchttest(
    df,
    alternative=['greater','less'],
    n_boot=0,
):
    """\
    Perform Welch's t test. Tests all columns.
    
    Parameters
    ----------
    df
        A table of samples as :class:`~pandas.DataFrame`, with groups in the
        rows and values in the columns. If the index is a multilevel index, the
        second level is interpreted as sample annotation.
    alternative
        The alternative for the test. Can also be a list of alternatives.
        Available are:
        
        - 'greater'
        - 'less'
        - 'two-sided'
        
    n_boot
        The number of bootstrap samples which are included in addition to the
        real samples.
        
    Returns
    -------
    An :class:`~pandas.DataFrame` containing the enrichment p-values.
    
    """
    def test_f(samples0, samples1, alternative):
        
        nobs0 = len(samples0) / (1+n_boot)
        nobs1 = len(samples1) / (1+n_boot)
        std0 = np.std(samples0, axis=0, ddof=1+n_boot)
        std1 = np.std(samples1, axis=0, ddof=1+n_boot)
        mean0 = np.mean(samples0, axis=0)
        mean1 = np.mean(samples1, axis=0)
        
        return stats.ttest_ind_from_stats(mean0, std0, nobs0, mean1, std1, nobs1, equal_var=False, alternative=alternative)[1]
    
    return _wrap_test(test_f, df, alternative=alternative)

