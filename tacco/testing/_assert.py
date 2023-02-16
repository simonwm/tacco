import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import pandas.testing

def assert_series_equal(left, right, **kwargs):
    """Assert equality of :class:`~pandas.Series` - thin wrapper around :func:`pandas.testing.assert_series_equal`"""
    pandas.testing.assert_series_equal(left, right, **kwargs)

def assert_index_equal(left, right, **kwargs):
    """Assert equality of :class:`~pandas.Index` - thin wrapper around :func:`pandas.testing.assert_index_equal`"""
    pandas.testing.assert_index_equal(left, right, **kwargs)

def assert_frame_equal(left, right, **kwargs):
    """Assert equality of dataframes, which handles cornercases different than :func:`pandas.testing.assert_frame_equal`"""
    if right is None:
        assert(left is None)
        return
    # need to do some corner cases manually, as pandas does not handle them properly...
    assert(isinstance(left, pd.DataFrame))
    assert(isinstance(right, pd.DataFrame))
    if len(left.index) != len(right.index):
        raise AssertionError(f'The dataframes have different index lengths {len(left.index)} and {len(right.index)}!')
    if len(left.columns) != len(right.columns):
        raise AssertionError(f'The dataframes have different columns lengths {len(left.columns)} and {len(right.columns)}!')
    if len(left.index) == 0 and len(left.columns) == 0:
        pass # empty dataframes are equal
    elif len(left.index) == 0:
        assert_index_equal(left.columns, right.columns)
        for k in left.columns:
            if left[k].dtype != right[k].dtype:
                raise AssertionError(f'The columns "{k}" have different dtypes {left[k].dtype} and {right[k].dtype}!')
    elif len(left.columns) == 0:
        assert_index_equal(left.index, right.index)
    else:
        #print(left.astype(str).to_numpy())
        #print(right.astype(str).to_numpy())
        _, counts_left = np.unique(left.index.astype(str).to_numpy(), return_counts=True)
        _, counts_right = np.unique(right.index.astype(str).to_numpy(), return_counts=True)
        if (counts_left > 1).any() or (counts_right > 1).any():
            print(f'The dataframes contained not unique indices. To work around a bug in the pandas assert_frame_equal, the indices are reset and checked separately as index.')
            assert_index_equal(left.index, right.index)
            left = left.reset_index(drop=True)
            right = right.reset_index(drop=True)
        
        pandas.testing.assert_frame_equal(left, right, **kwargs)

def assert_sparse_equal(left, right, rtol=1e-05, atol=1e-08):
    """Assert equality of sparse matrices"""
    if right is None:
        assert(left is None)
        return
    assert(scipy.sparse.issparse(left))
    assert(left.__class__ == right.__class__)
    assert(left.dtype == right.dtype)
    assert_tuple_equal(left.shape, right.shape)
    # transform to csr to have well defined order
    left = scipy.sparse.csr_matrix(left.tocoo())
    left.sort_indices()
    right = scipy.sparse.csr_matrix(right.tocoo())
    right.sort_indices()
    assert_tuple_equal(left.data.shape, right.data.shape)
    assert(np.allclose(left.data, right.data, rtol=rtol, atol=atol, equal_nan=True))
    assert_tuple_equal(left.indptr.shape, right.indptr.shape)
    assert(np.all(left.indptr == right.indptr))
    assert_tuple_equal(left.indices.shape, right.indices.shape)
    assert(np.all(left.indices == right.indices))

def assert_dense_equal(left, right, rtol=1e-05, atol=1e-08):
    """Assert equality of dense arrays"""
    if right is None:
        assert(left is None)
        return
    assert(isinstance(left, np.ndarray))
    assert(isinstance(right, np.ndarray))
    assert(left.dtype == right.dtype)
    assert_tuple_equal(left.shape, right.shape)
    assert(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))

def assert_adata_equal(left, right, rtol=1e-05, atol=1e-08):
    """Assert equality of :class:`~anndata.AnnData` instances"""
    if right is None:
        assert(left is None)
        return
    assert(isinstance(left, ad.AnnData))
    assert(isinstance(right, ad.AnnData))
    if scipy.sparse.issparse(left.X):
        assert_sparse_equal(left.X, right.X)
    else:
        assert_dense_equal(left.X, right.X)
    assert_frame_equal(left.obs, right.obs)
    assert_frame_equal(left.var, right.var)
    
    for k in left.obsm:
        if k not in right.obsm:
            raise AssertionError(f'The obsm key "{k}" is in adata left, but not in adata right!')
    for k in right.obsm:
        if k not in left.obsm:
            raise AssertionError(f'The obsm key "{k}" is in adata right, but not in adata left!')
        assert_frame_equal(left.obsm[k],right.obsm[k])
    for k in left.obsp:
        if k not in right.obsp:
            raise AssertionError(f'The obsp key "{k}" is in adata left, but not in adata right!')
    for k in right.obsp:
        if k not in left.obsp:
            raise AssertionError(f'The obsp key "{k}" is in adata right, but not in adata left!')
        if scipy.sparse.issparse(left.obsp[k]):
            assert_sparse_equal(left.obsp[k], right.obsp[k])
        else:
            assert_dense_equal(left.obsp[k], right.obsp[k])
            
    for k in left.varm:
        if k not in right.varm:
            raise AssertionError(f'The varm key "{k}" is in adata left, but not in adata right!')
    for k in right.varm:
        if k not in left.varm:
            raise AssertionError(f'The varm key "{k}" is in adata right, but not in adata left!')
        assert_frame_equal(left.varm[k],right.varm[k])
    for k in left.varp:
        if k not in right.varp:
            raise AssertionError(f'The varp key "{k}" is in adata left, but not in adata right!')
    for k in right.varp:
        if k not in left.varp:
            raise AssertionError(f'The varp key "{k}" is in adata right, but not in adata left!')
        if scipy.sparse.issparse(left.varp[k]):
            assert_sparse_equal(left.varp[k], right.varp[k])
        else:
            assert_dense_equal(left.varp[k], right.varp[k])
            
    for k in left.uns:
        if k not in right.uns:
            raise AssertionError(f'The uns key "{k}" is in adata left, but not in adata right!')
    for k in right.uns:
        if k not in left.uns:
            raise AssertionError(f'The uns key "{k}" is in adata right, but not in adata left!')
        if scipy.sparse.issparse(left.uns[k]):
            assert_sparse_equal(left.uns[k], right.uns[k])
        elif isinstance(left.uns[k], np.ndarray):
            assert_dense_equal(left.uns[k], right.uns[k])
        elif isinstance(left.uns[k], pd.DataFrame):
            assert_frame_equal(left.uns[k], right.uns[k])
        elif isinstance(left.uns[k], pd.Series):
            assert_series_equal(left.uns[k], right.uns[k])
        else:
            raise AssertionError(f'Got something of type {left.uns[k].__class__} in left.uns[{k}] - dont know what to do with it...')

def assert_tuple_equal(left, right, rtol=1e-05, atol=1e-08):
    """Assert equality of tuples"""
    assert(isinstance(left, tuple))
    assert(isinstance(right, tuple))
    assert(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
