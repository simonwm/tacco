import numpy as np
import pandas as pd
from .. import utils

def proj_err(df1, df2, metric='bc'):
    err_per_cell = []
    for i in np.arange(df1.shape[0]):
        err_per_cell.append(utils.projection(df1.to_numpy()[[i], :],
                                             df2.reindex_like(df1).to_numpy()[[i], :], metric=metric, deconvolution=False, parallel=True))
    return np.nanmean(err_per_cell)


def lp_err(df1, df2, p=2):
    err_per_cell = np.linalg.norm(df1.to_numpy() - df2.reindex_like(df1).to_numpy(), axis=1, ord=p)
    return np.nanmean(err_per_cell)

def max_correct(df1, df2):
    return np.nanmean(np.argmax(df2.reindex_like(df1).to_numpy(), 1) == np.argmax(df1.to_numpy(), 1))

def corr(df1, df2):
    return np.corrcoef(df1.to_numpy().flatten(), df2.reindex_like(df1).to_numpy().flatten())[0, 1]


err_funcs = {'lp': lp_err, 'proj': proj_err, 'max_correct': max_correct, 'corr': corr}

def compute_err(adata, result_keys, annotation_keys, err_method='lp', **kwargs):
    """
    Computes annotation error of adata's annotation vs results

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including annotations in in `.obs` or
        `.obsm`.
    result_keys
        A list of keys in `.obsm` that contain annotation results
    annotation_keys
        The `.obs` or `.obsm` keys where the ground truth annotation is
        stored.
    err_method
        The method to use to calculate errors. Available are:
        
        - "lp": lp-norm, can use an additional keyword argument 'p' with
          default `2`.
        - "corr"
        - "max_correct"
        - "proj": projection, can use an additional keyword argument 'metric'
          with default "bc"
    
    Returns
    -------
    A :class:`~dict` with the errors.
    
    """
    result_keys = [result_keys] if type(result_keys) == str else result_keys

    if ~np.all([k in adata.obsm_keys() for k in result_keys]):
        raise ValueError('Not all `result_keys` are in `adata.obsm`')
    if err_method not in err_funcs.keys():
        raise ValueError('`err_method` needs to be one of: %s' % ' '.join(err_funcs.keys()))
    err_func = err_funcs[err_method]

    # generate normalized numerical dataframe of annotations
    single_annotation = isinstance(annotation_keys, str)
    annotation_keys = [annotation_keys] if single_annotation else annotation_keys

    if set(annotation_keys).issubset(set(adata.obs.columns)):
        annotation_df = adata.obs[annotation_keys]
    elif set(annotation_keys).issubset(set(adata.obsm_keys())): # should be a string then
        annotation_df = adata.obsm[annotation_keys[0]]
    else:
        raise ValueError('`annotation_keys` not found in `obs` or `obsm`')

    if isinstance(annotation_df.iloc[0], str):
        annotation_df = pd.get_dummies(annotation_df)
    elif annotation_df.shape[1] == 1 and single_annotation:
        annotation_df = pd.get_dummies(annotation_df.iloc[:,0])
    # annotation_df = (annotation_df.T / annotation_df.sum(1)).T

    errs = {}
    for k in result_keys:
        k_df = adata.obsm[k]
        
        sel_keys = list(set(annotation_df.columns).intersection(k_df.columns)) # should we throw an error for a mismatch?
        # k_df = (k_df.T / k_df.sum(1)).T
        errs[k] = err_func(k_df[sel_keys], annotation_df[sel_keys], **kwargs)

    return errs




