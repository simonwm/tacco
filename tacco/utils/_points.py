import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
from numba import njit

from ..utils import _math, min_dtype

@njit(cache=True)
def _groupby_mean(data, codes, n_codes):
    counts = np.zeros(n_codes, dtype=np.int64)
    sums = np.zeros(n_codes, dtype=data.dtype)
    assert(len(data)==len(codes))
    for i in range(len(data)):
        code_i = codes[i]
        counts[code_i] += 1
        sums[code_i] += data[i]
    for code_i in range(n_codes):
        sums[code_i] /= counts[code_i]
    return sums

def dataframe2anndata(
    data,
    obs_key,
    var_key,
    count_key=None,
    compositional_keys=None,
    mean_keys=None,
):

    """\
    Creates an :class:`~anndata.AnnData` from a long form
    :class:`~pandas.DataFrame`. The entries of the `.obs` columns in the result
    :class:`~anndata.AnnData` are only well defined if they are identical per
    observation.
    
    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame`.
    obs_key
        The name of the column containing the categorical property to become
        the `.obs` dimension. Can also be a categorical :class:`~pandas.Series`
        compatible with `data`. If `None`, use the index as `.obs` dimension
        and keep all unused annotation in `.obs`. 
    var_key
        The name of the column containing the categorical property to become
        the `.var` dimension. If `None`, the `.var` dimension will be of length
        `0`.
    count_key
        The name of the column containing counts/weights to sum. If `None`,
        bare occurences (i.e. 1) are summed over.
    compositional_keys
        The names of the columns containing categorical properties to populate
        `.obsm` dataframes with.
    mean_keys
        The names of the columns containing numerical properties to construct
        mean quantities for `.obs` columns with.
        
    Returns
    -------
    An :class:`~anndata.AnnData` containing the counts in `.X`.
    
    """

    if obs_key is None:
        row = pd.Series(data.index, index=data.index).astype('category')
    elif isinstance(obs_key, pd.Series):
        row = obs_key if hasattr(obs_key, 'cat') else obs_key.astype('category')
        obs_key = row.name
    else:
        row = data[obs_key] if hasattr(data[obs_key], 'cat') else data[obs_key].astype('category')
    if var_key is not None:
        col = data[var_key] if hasattr(data[var_key], 'cat') else data[var_key].astype('category')

    dtype = min_dtype(len(row))
    if var_key is None:
        counts = scipy.sparse.csr_matrix((len(row.cat.categories),0))
    else:
        if count_key is None:
            count = np.ones_like(row, dtype=dtype)
        else:
            count = data[count_key]
        counts = scipy.sparse.coo_matrix((count, (row.cat.codes,col.cat.codes)))
        counts = counts.tocsr()

    obs = pd.DataFrame(index=row.cat.categories.astype(str))
    var = None if var_key is None else pd.DataFrame(index=col.cat.categories.astype(str))

    adata = ad.AnnData(counts, obs=obs, var=var, dtype=dtype)

    adata.obs.index.name = obs_key
    adata.var.index.name = var_key

    if obs_key is None:
        special_keys = [var_key]
        if count_key is not None:
            special_keys.append(count_key)
        for k in data.columns:
            if k not in special_keys:
                if data[k].isna().all(): # hangs in this special case if not taken care of separately
                    adata.obs[k] = data[k]
                else:
                    adata.obs[k] = pd.Series(data[k].to_numpy(), index=adata.obs.index, dtype=data[k].dtype)

    # create compositional obsm annotations
    if compositional_keys is None:
        compositional_keys = []
    elif isinstance(compositional_keys, str):
        compositional_keys = [compositional_keys]
    for compositional_key in compositional_keys:
        obsm_data = dataframe2anndata(data, obs_key=obs_key, var_key=compositional_key)
        adata.obsm[compositional_key] = pd.DataFrame(obsm_data.X.toarray(), index=obsm_data.obs.index, columns=obsm_data.var.index)
        adata.obsm[compositional_key] /= adata.obsm[compositional_key].sum(axis=1).to_numpy()[:,None]
    
    # create mean obs annotations
    if mean_keys is None:
        mean_keys = []
    elif isinstance(mean_keys, str):
        mean_keys = [mean_keys]
    for mean_key in mean_keys:
        adata.obs[mean_key] = _groupby_mean(data[mean_key].to_numpy(), row.cat.codes.to_numpy(), len(row.cat.categories))
                    
    return adata

def anndata2dataframe(
    adata,
    obs_name=None,
    var_name=None,
    obs_keys=None,
    var_keys=None,
):

    """\
    Creates a long form :class:`~pandas.DataFrame` from an
    :class:`~anndata.AnnData`.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`.
    obs_name
        The name of the obs column in the dataframe. If `None`, tries to use
        `adata.obs.index.name`.
    var_name
        The name of the var column in the dataframe. If `None`, tries to use
        `adata.var.index.name`.
    obs_keys
        The names of the obs columns to include in the dataframe.
    var_key
        The names of the var columns to include in the dataframe.
        
    Returns
    -------
    A long form :class:`~pandas.DataFrame`.
    
    """

    if obs_name is None:
        obs_name = adata.obs.index.name
        if obs_name is None:
            raise ValueError('`obs_name` was not set and `adata.obs.index.name` is `None`!')
    if var_name is None:
        var_name = adata.var.index.name
        if var_name is None:
            raise ValueError('`var_name` was not set and `adata.var.index.name` is `None`!')

    X = scipy.sparse.coo_matrix(adata.X)

    data = pd.DataFrame({
        obs_name: adata.obs.index[X.row],
        var_name: adata.var.index[X.col],
        'X': X.data,
    })

    if obs_keys is not None:
        for obs_key in obs_keys:
            data[obs_key] = pd.Series(adata.obs[obs_key].to_numpy()[X.row], index=data.index, dtype=adata.obs[obs_key].dtype)
    if var_keys is not None:
        for var_key in var_keys:
            data[var_key] = pd.Series(adata.var[var_key].to_numpy()[X.col], index=data.index, dtype=adata.var[var_key].dtype)

    return data

