import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, inplace_row_scale
import numba
import mkl
from numba import njit, prange

@njit(fastmath=True, cache=True)
def _divide_serial(a,b,out):
    for i in prange(len(a)):
        out[i] = a[i] / b[i]

@njit(fastmath=True, parallel=True, cache=True)
def _divide_parallel(a,b,out):
    for i in prange(len(a)):
        out[i] = a[i] / b[i]

def divide(
    a,
    b,
    out=None,
    parallel=True,
):

    """\
    Calculates division `out = a / b`.
    
    Parameters
    ----------
    a
        A :class:`~numpy.ndarray` with at least 1 dimension
    b
        A :class:`~numpy.ndarray` with at least 1 dimension of the same shape
        as `a`
    out
        A :class:`~numpy.ndarray` with at least 1 dimension of the same shape
        as `a`. If `None`, the result is returned.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the result.
        
    """

    if out is None:
        out = np.empty_like(a)

    assert(a.shape==b.shape)
    assert(a.shape==out.shape)

    if parallel:
        _divide_parallel(a,b,out)
    else:
        _divide_serial(a,b,out)

    return out

@njit(fastmath=True, parallel=True, cache=True)
def _sparse_result_gemmT(A, B, out_row, out_col, out_data):
    n = A.shape[1]
    for rc in prange(len(out_row)):
        r = out_row[rc]
        c = out_col[rc]
        temp = 0.0
        for k in range(n):
            temp += A[r,k] * B[c,k]
        out_data[rc] += temp
@njit(fastmath=True, cache=True)
def _sparse_result_gemmT_serial(A, B, out_row, out_col, out_data):
    n = A.shape[1]
    for rc in prange(len(out_row)):
        r = out_row[rc]
        c = out_col[rc]
        temp = 0.0
        for k in range(n):
            temp += A[r,k] * B[c,k]
        out_data[rc] += temp

def sparse_result_gemmT(
    A, 
    B,
    sparse_out,
    parallel=True,
    inplace=True,
    update_out=False,
):

    """\
    Perform a dense matrix-matrix multiplication A @ B.T for the case when only
    a sparse subset of the result is needed. Note that the second matrix is
    transposed.
    
    Parameters
    ----------
    A
        A 2d numpy array..
    B
        A 2d numpy array with the same second dimension as `A`.
    sparse_out
        A `scipy.sparse` matrix to contain the result and to provide the
        sparsity structure of the result.
    parallel
        Whether to work on multiple cores.
    inplace
        Whether to write the result directly into `sparse_out` or return a
        copy. `inplace` is only possible if `sparse_out` is a
        :class:`~scipy.sparse.coo_matrix`.
    update_out
        Whether to add the result to the existing values of `sparse_out` or to
        overwrite them.
        
    Returns
    -------
    Returns the :class:`~scipy.sparse.coo_matrix` containing the result.
        
    """

    result_shape = (A.shape[0],B.shape[0])
    inner_size = A.shape[1]
    if B.shape[1] != A.shape[1]:
        raise ValueError("`A` and `B` dont have the same second dimension! The shapes are %s and %s!" % (A.shape, B.shape))
    if result_shape != sparse_out.shape:
        raise ValueError("The result shape implied by `A` and `B` %s dont fit to the one supplied in `sparse_out`! The corresponding shapes are %s and %s!" % (result_shape, sparse_out.shape))
    
    _sparse_out = sparse_out.tocoo()
    if inplace and _sparse_out is not sparse_out:
        raise ValueError("`inplace` is `True` but the supplied sparse result matrix is not a coo matrix!")
    elif not inplace and _sparse_out is sparse_out:
        _sparse_out = _sparse_out.copy()
    
    if not update_out:
        _sparse_out.data[:] = 0
    
    if parallel:
        _sparse_result_gemmT(A, B, _sparse_out.row, _sparse_out.col, _sparse_out.data)
    else:
        _sparse_result_gemmT_serial(A, B, _sparse_out.row, _sparse_out.col, _sparse_out.data)
    
    return _sparse_out
    
@njit(fastmath=True, parallel=True, cache=True)
def _csrcsr_gemm_dense(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx):
    
    C = np.zeros((n_row,n_col))
    
    for i in prange(n_row):
        jj_start = Ap[i];
        jj_end   = Ap[i+1];
        for jj in range(jj_start, jj_end):
            j = Aj[jj];
            v = Ax[jj];

            kk_start = Bp[j];
            kk_end   = Bp[j+1];
            for kk in range(kk_start, kk_end):
                k = Bj[kk];

                C[i,k] += v*Bx[kk];
                
    return C

@njit(fastmath=True, parallel=True, cache=True)
def _densecsr_gemm_dense(n_row, n_col, A, Bp, Bj, Bx):
    
    nk = A.shape[1]
    
    C = np.zeros((n_row,n_col))
    
    for i in prange(n_row):
        for j in range(nk):
            v = A[i,j];

            kk_start = Bp[j];
            kk_end   = Bp[j+1];
            for kk in range(kk_start, kk_end):
                k = Bj[kk];

                C[i,k] += v*Bx[kk];
                
    return C

@njit(fastmath=True, parallel=True, cache=True)
def _cscdense_gemm_dense(n_row, n_col, Ap, Ai, Ax, B):
    
    nk = B.shape[0]
    
    C = np.zeros((n_col,n_row))
    
    for j in prange(n_col):
        for k in range(nk):
            v = B[k,j]
            
            ii_start = Ap[k];
            ii_end   = Ap[k+1];
            for ii in range(ii_start, ii_end):
                i = Ai[ii];

                C[j,i] += v*Ax[ii];
                
    return C.T

def cast_down_common(A,B):
    if not pd.api.types.is_float_dtype(A):
        A = A.astype(np.float64)
    if not pd.api.types.is_float_dtype(B):
        B = B.astype(np.float64)

    if A.dtype == B.dtype:
        return A,B
    elif A.dtype == np.float64 and B.dtype == np.float32:
        return A.astype(np.float32), B
    elif A.dtype == np.float32 and B.dtype == np.float64:
        return A, B.astype(np.float32)
    else:
        return A,B
        
def gemmT(
    A,
    B,
    parallel=True,
    sparse_result=False,
):

    """\
    Perform a matrix-matrix multiplication A @ B.T for arbitrary sparseness of
    A and B in parallel. Uses `sparse_dot_mkl` if available.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix.
    B
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix with the same
        second dimension as `A`.
    parallel
        Whether to run the multiplication in parallel.
    sparse_result
        Whether to return a sparse result when both inputs are sparse
        
    Returns
    -------
    Depending on `sparse_result` returns either a :class:`~numpy.ndarray` or a\
    scipy sparse matrix containing the result.
        
    """
    
    if not issparse(A) and not isinstance(A,np.ndarray):
        raise ValueError("`A` can only be a scipy sparse matrix or a numpy array!")
    if not issparse(B) and not isinstance(B,np.ndarray):
        raise ValueError("`B` can only be a scipy sparse matrix or a numpy array!")
    if issparse(A) and not isinstance(A, (csr_matrix, csc_matrix)):
        A = A.tocsr()
    if issparse(B) and not isinstance(B, (csr_matrix, csc_matrix)):
        B = B.tocsr()
    
    A,B = cast_down_common(A,B)
    
    numba_threads = numba.get_num_threads()
    mkl_threads = mkl.get_max_threads()
    if not parallel:
        numba.set_num_threads(1)
        mkl.set_num_threads(1)
    
    result_shape = (A.shape[0],B.shape[0])
    inner_size = A.shape[1]
    if B.shape[1] != A.shape[1]:
        raise ValueError("`A` and `B` dont have the same second dimension! The shapes are %s and %s!" % (A.shape, B.shape))
    
    try: # if mkl is available, use it.
        import sparse_dot_mkl
        return sparse_dot_mkl.dot_product_mkl(A, B.T, dense=(not sparse_result));
    
    except ImportError: # use custom numba implementation
        print('sparse_dot_mkl is not found, so a (slower) fallback is used.')
        
        if issparse(A) and issparse(B):
            if not sparse_result:
                A = A.tocsr()
                BT = B.tocsc() # avoids intermediate during transposition: .tocsc() == .T.tocsr()
                return _csrcsr_gemm_dense(result_shape[0], result_shape[1], A.indptr, A.indices, A.data, BT.indptr, BT.indices, BT.data)
            else:
                return A@(B.T)
        elif issparse(A) and not issparse(B):
            A = A.tocsc()
            return _cscdense_gemm_dense(result_shape[0], result_shape[1], A.indptr, A.indices, A.data, B.T)
        elif not issparse(A) and issparse(B):
            BT = B.tocsc() # avoids intermediate during transposition: .tocsc() == .T.tocsr()
            return _densecsr_gemm_dense(result_shape[0], result_shape[1], A, BT.indptr, BT.indices, BT.data)
        else:
            return A@(B.T)
    
    if not parallel:
        numba.set_num_threads(numba_threads)
        mkl.set_num_threads(mkl_threads)

def row_scale(
    X,
    rescaling_factors,
    round=False,
):

    """\
    Rescales rows of dense or sparse matrix inplace.
    
    Parameters
    ----------
    X
        A 2d :class:`~numpy.ndarray` array or a `scipy` sparse matrix.
    rescaling_factors
        A 1d :class:`~numpy.ndarray` containing the row-wise rescaling factors.
    round
        Whether to round the result to integer values
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """

    if hasattr(rescaling_factors, 'to_numpy'):
        rescaling_factors = rescaling_factors.to_numpy()
    rescaling_factors = rescaling_factors.flatten()
    
    if issparse(X):
        inplace_row_scale(X, rescaling_factors)
        if round:
            np.around(X.data, out=X.data)
    else:
        X *= rescaling_factors[:,None]
        if round:
            np.around(X, out=X)

def col_scale(
    X,
    rescaling_factors,
    round=False,
):

    """\
    Rescales columns of dense or sparse matrix inplace.
    
    Parameters
    ----------
    X
        A 2d :class:`~numpy.ndarray` array or a `scipy` sparse matrix.
    rescaling_factors
        A 1d :class:`~numpy.ndarray` containing the column-wise rescaling
        factors.
    round
        Whether to round the result to integer values
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """
    
    if not isinstance(X, np.ndarray) and not issparse(X):
        raise ValueError(f'`X` must be a numpy array or a scipy sparse matrix!')

    if hasattr(rescaling_factors, 'to_numpy'):
        rescaling_factors = rescaling_factors.to_numpy()
    rescaling_factors = rescaling_factors.flatten()
    
    if issparse(X):
        inplace_column_scale(X, rescaling_factors)
        if round:
            np.around(X.data, out=X.data)
    else:
        X *= rescaling_factors
        if round:
            np.around(X, out=X)

@njit(fastmath=True, parallel=True, cache=True)
def _log1p(x):
    for i in prange(len(x)):
        x[i] = np.log1p(x[i])
def log1p(
    X,
):

    """\
    Calculates log1p inplace and in parallel.
    
    Parameters
    ----------
    X
        A :class:`~numpy.ndarray` with more than 1 dimension, a `scipy` sparse
        matrix, or something which has an attribute `.X` which fits this
        description, e.g. an :class:`~anndata.AnnData`
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """

    if hasattr(X, 'X'):
        X = X.X

    if issparse(X):
        _log1p(X.data)
    else:
        _log1p(X)

@njit(fastmath=True, parallel=True, cache=True)
def _log(x):
    for i in prange(len(x)):
        x[i] = np.log(x[i])
def log(
    X,
):

    """\
    Calculates log inplace and in parallel.
    
    Parameters
    ----------
    X
        A :class:`~numpy.ndarray` with more than 1 dimension, a `scipy` sparse
        matrix, or something which has an attribute `.X` which fits this
        description, e.g. an :class:`~anndata.AnnData`
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """

    if hasattr(X, 'X'):
        X = X.X

    if issparse(X):
        _log(X.data)
    else:
        _log(X)

@njit(fastmath=True, parallel=True, cache=True)
def _sqrt(x):
    for i in prange(len(x)):
        x[i] = np.sqrt(x[i])
def sqrt(
    X,
):

    """\
    Calculates sqrt inplace and in parallel.
    
    Parameters
    ----------
    X
        A :class:`~numpy.ndarray` with more than 1 dimension, a `scipy` sparse
        matrix, or something which has an attribute `.X` which fits this
        description, e.g. an :class:`~anndata.AnnData`
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """

    if hasattr(X, 'X'):
        X = X.X

    if issparse(X):
        _sqrt(X.data)
    else:
        _sqrt(X)
    
def integrate_mean(y,x):
    # using trapezoidal rule
    heights = 0.5 * (y[1:] + y[:-1])
    widths = x[1:] - x[:-1]
    ranges = x[1:] - x[0]
    means = (heights * widths).cumsum() / ranges
    # prepend start value to get same length result
    return np.concatenate([[y[0]],means])

def get_sum(
    X,
    axis,
    dtype=None,
):

    """\
    Calculates the sum of a sparse matrix or array-like in a specified axis and
    returns a flattened result.
    
    Parameters
    ----------
    X
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix.
    axis
        The axis along which to calculate the sum
    dtype
        The dtype used in the accumulators and the result.
        
    Returns
    -------
    The flattened sums as 1d :class:`~numpy.ndarray`.
        
    """

    if issparse(X):
        result = X.sum(axis=axis, dtype=dtype).A.flatten()
    else:
        result = X.sum(axis=axis, dtype=dtype)
    
    import anndata._core.views
    if isinstance(result, anndata._core.views.ArrayView):
        result = result.toarray()
    
    return result
