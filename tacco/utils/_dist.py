import pandas as pd
import numpy as np
import scipy.spatial
from scipy.sparse import issparse
from . import _math
from . import _utils
from numba import njit, prange
import numba
import warnings
from threadpoolctl import threadpool_limits
import tempfile

def min_dtype(
    maximum,
):
    """\
    Gives the smallest signed integer dtype which is necessary to represent
    `maximum`.
    
    Parameters
    ----------
    maximum
        The integer to be representable in the dtype.
        
    Returns
    -------
    A signed integer dtype.
    
    """
    
    if maximum < 2**7 - 1:
        dtype = np.int8
    elif maximum < 2**15 - 1:
        dtype = np.int16
    elif maximum < 2**31 - 1:
        dtype = np.int32
    elif maximum < 2**63 - 1:
        dtype = np.int64
    else:
        raise ValueError('The maximum value to represent is larger than the range of 64bit integer! This is not easily possible...')
    return dtype

def bin(
    data,
    bin_size,
    position_keys=['x','y'],
    bin_keys=None,
    shift=None,
):

    """\
    Bins points in position space.
    
    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame`.
    bin_size
        The spatial size of a bin. Bins are of the same size in all
        dimensions.
    position_keys
        Array-like of column keys which contain the position of the points.
    bin_keys
        The names of the columns to contain the bin assignment. Has to be of
        the same length as `position_keys`. If `None`, a dataframe of bin
        assignments is returned.
    shift
        An array-like of the same length as `position_keys`, giving the
        shift of the start point of the bin grid in every dimension. If
        `None`, then no shift is used and the bin grid starts at the minimum
        value in each dimension.
        
    Returns
    -------
    Depending on `bin_key` returns either a :class:`~pandas.DataFrame` of bin\
    assignments or the updated input `data` contining the bin assignments\
    under `bin_keys`.
    
    """

    _coords = data[position_keys].to_numpy()
    mins, maxs = _coords.min(axis=0), _coords.max(axis=0)
    if shift is not None:
        mins += shift
    n_dim = _coords.shape[1]

    max_bin_value = (maxs - mins).max() // bin_size + 1
    dtype = min_dtype(max_bin_value)

    bin_coords = np.array([((_coords[:,d] - mins[d]) // bin_size).astype(dtype) for d in range(n_dim) ]).T

#    bind = [np.arange(mins[d],maxs[d],bin_size) for d in range(n_dim) ]
#    counts = np.array([ np.histogramdd(sub[position_keys].to_numpy(), bins=bind)[0] for g, sub in data.groupby(count_key) ])
#    bin_coords = np.array([x.T for x in np.meshgrid(*[list(range(len(b[:-1]))) for b in bind])])

    if bin_keys is None:
        return pd.DataFrame(bin_coords,index=data.index)
    else:
        if len(bin_keys) != n_dim:
            raise ValueError(f'`bin_keys` and `position_keys` have a diferent size: `(len(bin_keys),len(position_keys))=({len(bin_keys)},{n_dim})`')
        for d,bin_key in enumerate(bin_keys):
            data[bin_key] = bin_coords[:,d]
        return data

def hash(
    data,
    keys=None,
    hash_key=None,
    other=None,
    compress=True,
):

    """\
    Create collision-free hash of several categorical columns by
    lexicograhical indexing.
    
    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame`.
    keys
        The names of the columns containing the categorical properties to
        hash. Not-categorical columns are transformed to categoricals first.
        If `None`, uses all columns.
    hash_key
        The name of the column to contain the hash values. If `None`, a series
        of hash assignments is returned.
    other
        Another :class:`~pandas.DataFrame`, which also has the `keys` with the
        same datatypes and should get the same hash trasformation as `data`.
    compress
        Whether to compress the lexicographical indices into contiguous 0-based
        hash values. This can take quite some time, but can also decrease the
        object size of the hash columns.
        
    Returns
    -------
    Depending on `hash_key` returns either a :class:`~pandas.Series` of hash\
    assignments or the updated input `data` contining the hash assignments\
    under `hash_key`. Depending on `other` returns this as a pair of the\
    results for `data and `other`.
    
    """
    
    if keys is None:
        keys = data.columns

    _data = data[keys]
    if other is not None:
        _other = other[keys]

    colrange = range(_data.shape[1])
    cats = []
    for i in colrange:
        unique = _data.iloc[:,i].unique()
        if hasattr(unique,'to_numpy'):
            unique = unique.to_numpy()
    
        if other is not None:
            if _data.iloc[:,i].dtype != _other.iloc[:,i].dtype:
                raise ValueError(f'Columns {keys[i]!r} of `data` and `other` have different data types {_data.iloc[:,i].dtype!r} and {_other.iloc[:,i].dtype!r}!')
            other_unique = _other.iloc[:,i].unique()
            if hasattr(other_unique,'to_numpy'):
                other_unique = other_unique.to_numpy()
            unique = np.unique(np.concatenate([unique, other_unique]))
        
        cats.append(_data.iloc[:,i].astype(unique.dtype).astype(pd.CategoricalDtype(unique)))
    
    sizes = [len(cats[i].cat.categories) for i in colrange]
    dtype = min_dtype(np.prod(sizes))
    codes = [cats[i].cat.codes.to_numpy() for i in colrange]

    res = np.zeros(codes[0].shape, dtype=dtype)
    for size,code in zip(sizes, codes):
        res *= size
        res += code

    if other is None:
        if compress:
            res = pd.Series(res, index=data.index).astype('category').cat.codes

        if hash_key is None:
            return res
        else:
            data[hash_key] = res
            return data
    
    else:
        other_cats = [_other.iloc[:,i].astype(cats[i].cat.categories.dtype).astype(cats[i].dtype) for i in colrange]
        other_codes = [other_cats[i].cat.codes.to_numpy() for i in colrange]
        other_res = np.zeros(other_codes[0].shape, dtype=dtype)

        for size,code in zip(sizes, other_codes):
            other_res *= size
            other_res += code

        if compress:
            unique = np.unique(np.concatenate([np.unique(res), np.unique(other_res)]))
            res = pd.Series(res, index=data.index).astype(pd.CategoricalDtype(unique)).cat.codes
            other_res = pd.Series(other_res, index=other.index).astype(pd.CategoricalDtype(unique)).cat.codes

        if hash_key is None:
            return res, other_res
        else:
            data[hash_key] = res
            other[hash_key] = other_res
            return data, other

def data_copy(A):
    if issparse(A):
        # reuse indices of sparse matrix, while preseving the original data
        A_data = A.data
        A.data = A.data.copy()
        return A, A_data
    else:
        return A.copy(), A
def restore_data(A, A_data):
    if issparse(A):
        A.data = A_data
    
def kl_distance(A, B, parallel=True):
    
    # dont touch originals ; but only copy data, if sparse
    A, A_data = data_copy(A)
    B, B_data = data_copy(B)
    
    # make rows of A and B probability distributions
    _math.row_scale(A, 1/_math.get_sum(A, axis=1))
    _math.row_scale(B, 1/_math.get_sum(B, axis=1))
    
    # KL_ct = sum_g A_cg log(A_cg/B_tg) = sum_g A_cg log(A_cg) - sum_g A_cg log(B_tg)
    if issparse(B):
        B.data = np.log(B.data)
    else:
        B = np.log(B)
    ABT = _math.gemmT(A,B, parallel=parallel)
    if issparse(A):
        A.data *= np.log(A.data)
    else:
        _A0 = A != 0
        A[_A0] *= np.log(A[_A0])
    AA = _math.get_sum(A, axis=1)
    
    # restore original data, if sparse
    restore_data(A, A_data)
    restore_data(B, B_data)
    
    return ABT - AA[:,None] # change the sign to get positive distances

def naive_projection(A, B, parallel=True):
    # make rows of A and B probability distributions
    # project A on B: values between 0 and 1
    
    # as scaling and product commute, scale at the end to save memory and computation
    
    Asum = _math.get_sum(A, axis=1)
    Bsum = _math.get_sum(B, axis=1)
    
    ABT = _math.gemmT(A,B, parallel=parallel)
    
    _math.row_scale(ABT, 1/Asum)
    _math.col_scale(ABT, 1/Bsum)
    
    return ABT

def naive_projection_distance(A, B, parallel=True):
    # distance is 1 - projection
    
    ABT = naive_projection(A, B, parallel=parallel)
    
    return np.maximum(1-ABT,0)

def normalized_weighted_scalar_product(A, B, parallel=True):
    # normalize as probabilities
    Anorm = _math.get_sum(A,axis=1)
    Bnorm = _math.get_sum(B,axis=1)

    A = A.copy()
    B = B.copy()

    _math.row_scale(A, 1/Anorm)
    _math.row_scale(B, 1/Bnorm)

    # get "mean bulk expression"
    Bmean = _math.get_sum(B,axis=0) / B.shape[0]
    mean = Bmean

    # scale one of them to get Acg * Btg / Mg
    if B.shape[0] <= A.shape[0]:
        _math.col_scale(B, 1/mean)
    else:
        _math.col_scale(A, 1/mean)

    # get overlap
    ABT = _math.gemmT(A,B, parallel=parallel)

    # normalize as probabilities
    ABTnorm = _math.get_sum(ABT,axis=1)
    _math.row_scale(ABT, 1/ABTnorm)

    return ABT

def weighted_projection(A, B, parallel=True):
    # normalize as probabilities
    Anorm = _math.get_sum(A,axis=1)
    Bnorm = _math.get_sum(B,axis=1)

    A = A.copy()
    B = B.copy()

    _math.row_scale(A, 1/Anorm)
    _math.row_scale(B, 1/Bnorm)

    # get "mean bulk expression"
    Bmean = _math.get_sum(B,axis=0) / B.shape[0]
    mean = Bmean

    _mean = 1/mean

    # normalize A and B in the weighted scalar product defined by sum_g x_g * y_g / mean_g
    if issparse(A):
        # reuse indices of sparse matrix, while preseving the original data
        A, A_data = data_copy(A)
        A.data *= A.data
        Anorm = np.sqrt(A @ _mean).flatten()
        restore_data(A, A_data)
    else:
        Anorm = np.sqrt((A**2) @ _mean)
    if issparse(B):
        # reuse indices of sparse matrix, while preseving the original data
        B, B_data = data_copy(B)
        B.data *= B.data
        Bnorm = np.sqrt(B @ _mean).flatten()
        restore_data(B, B_data)
    else:
        Bnorm = np.sqrt((B**2) @ _mean)
    _math.row_scale(A, 1/Anorm)
    _math.row_scale(B, 1/Bnorm)

    # scale one of them to get Acg * Btg / Mg
    if B.shape[0] <= A.shape[0]:
        _math.col_scale(B, _mean)
    else:
        _math.col_scale(A, _mean)

    # get overlap
    ABT = _math.gemmT(A,B, parallel=parallel)

    return ABT

def get_norm(A):
    if issparse(A):
        # reuse indices of sparse matrix, while preseving the original data
        A, A_data = data_copy(A)
        A.data *= A.data
        Anorm = np.sqrt(A.sum(axis=1)).A.flatten()
        restore_data(A, A_data)
    else:
        Anorm = np.sqrt((A**2).sum(axis=1))
    return Anorm

def cosine_projection(A, B, parallel=True):
    Anorm = get_norm(A)
    Bnorm = get_norm(B)
    
    ABT = _math.gemmT(A, B, parallel=parallel)

    _math.row_scale(ABT, 1/Anorm)
    _math.col_scale(ABT, 1/Bnorm)
    
    return ABT

# Much faster than scipy.spatial.distance.cdist(A, B, 'cosine') already for dense data.
# For sparse inputs there is a minimum sparsity necessary to see speedup over the dense version.
def cosine_distance(A, B, parallel=True):

    ABT = cosine_projection(A, B, parallel=parallel)
    
    return np.maximum(1-ABT, 0)

def bhattacharyya_coefficient(A, B, parallel=True):
    # normalize as probabilities
    Anorm = _math.get_sum(A,axis=1)
    Bnorm = _math.get_sum(B,axis=1)
    
    A = A.copy()
    B = B.copy()
    
    _math.row_scale(A, 1/Anorm)
    _math.row_scale(B, 1/Bnorm)
    
    # transform to probability amplitudes
    A = np.sqrt(A)
    B = np.sqrt(B)

    # get overlap
    ABT = _math.gemmT(A,B, parallel=parallel)
    
    return ABT

# bhattacharyya variant of the cosine distance
# this is identical to the squared hellinger distance
def bhattacharyya_cosine_distance(A, B, parallel=True):
    
    # get amplitude overlap
    ABT = bhattacharyya_coefficient(A, B, parallel=parallel)

    return np.maximum(1-ABT, 0)

def hellinger_distance(A, B, parallel=True):
    return np.sqrt(bhattacharyya_cosine_distance(A, B, parallel=parallel))

def bhattacharyya2_coefficient(A, B, parallel=True):
    
    # get amplitude overlap
    ABT = bhattacharyya_coefficient(A, B, parallel=parallel)
    
    # transform back to probabilities
    ABT *= ABT
    
    return ABT

# bhattacharyya variant of the cosine distance
def bhattacharyya2_cosine_distance(A, B, parallel=True):
    
    # get overlap
    ABT = bhattacharyya2_coefficient(A, B, parallel=parallel)
    
    return np.maximum(1-ABT,0)

def get_sqnorm(A):
    if issparse(A):
        # reuse indices of sparse matrix, while preserving the original data
        A, A_data = data_copy(A)
        A.data *= A.data
        Anorm = A.sum(axis=1).A.flatten()
        restore_data(A, A_data)
    else:
        Anorm = (A**2).sum(axis=1)
    return Anorm

# Much faster than scipy.spatial.distance.cdist(A, B, 'euclidean') already for dense data (maybe an epsilon less accurate).
# For sparse inputs there is a minimum sparsity necessary to see speedup over the dense version.
def euclidean_distance(A, B, parallel=True):
    Anorm = get_sqnorm(A)
    Bnorm = get_sqnorm(B)
    
    ABT = _math.gemmT(A,B, parallel=parallel)
    
    return np.sqrt(np.maximum(Anorm[:,None]+Bnorm[None,:] - 2 * ABT,0))

def cdist(
    A,
    B=None,
    metric='euclidean',
    parallel=True,
):
 
    """\
    Calclulate a dense pairwise distance matrix of sparse and dense inputs. For
    some metrics ('euclidean', 'cosine'), this is considerably faster than
    :func:`scipy.spatial.distance.cdist`. For basically all other metrics this
    falls back to :func:`scipy.spatial.distance.cdist`. Special distances are:
    
    - 'bc': 1 - Bhattacharyya coefficient, a cosine similarity equivalent for
      the Bhattacharyya coefficient, which is the overlap of two probability
      distributions. The input vectors are normalized to sum 1 first.
    - 'bc2': 1 - (Bhattacharyya coefficient)^2, a cosine similarity equivalent
      for the squared Bhattacharyya coefficient. The input vectors are
      normalized to sum 1 first.
    - 'hellinger': The Hellinger(-Bhattacharyya) distance defined as
      sqrt(1 - Bhattacharyya coefficient)
    - 'h2': squared Hellinger Distance; synonymous to 'bc'.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix.
    B
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix with the same
        second dimension as `A`. If `None`, use `A`.
    metric
        A string specifying the metric to use.
    parallel
        Whether to run the operation in parallel - if possible.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the distances.

    """

    if B is None:
        B = A

    if B.shape[1] != A.shape[1]:
        raise ValueError("`A` and `B` dont have the same second dimension! The shapes are %s and %s!" % (A.shape, B.shape))
        
    if isinstance(A,pd.DataFrame):
        A = A.to_numpy()
    if isinstance(B,pd.DataFrame):
        B = B.to_numpy()
    
    A,B = _math.cast_down_common(A,B)
    
    if metric == 'kl':
        return kl_distance(A, B, parallel=parallel)
    elif metric == 'naive' or metric == 'projection':
        return naive_projection_distance(A, B, parallel=parallel)
    elif metric == 'cosine':
        return cosine_distance(A, B, parallel=parallel)
    elif metric == 'euclidean':
        return euclidean_distance(A, B, parallel=parallel)
    elif metric == 'bc' or metric == 'h2':
        return bhattacharyya_cosine_distance(A, B, parallel=parallel)
    elif metric == 'bc2':
        return bhattacharyya2_cosine_distance(A, B, parallel=parallel)
    elif metric == 'hellinger':
        return hellinger_distance(A, B, parallel=parallel)
    else:
        if issparse(A): # scipy.spatial.distance.cdist works only with dense matrices - and we need a dense result anyway...
            A = A.toarray()
        if issparse(B):
            B = B.toarray()
        return scipy.spatial.distance.cdist(A, B, metric)

def projection(
    A,
    B,
    metric='bc',
    deconvolution=False,
    parallel=True,
):
 
    """\
    Calculate pairwise normalized projections of sparse and dense inputs.
    Depending on the parameters this can be an asymmetric operation: the
    argument `A` here is projected on the argument `B`.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix.
    B
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix with the same
        second dimension as `A`; gives the basis to project on.
    metric
        A string specifying the metric to use. Available are
        
        - 'cosine': Euclidean projection of normalized vectors
        - 'naive': Euclidean projection of vectors normalized as probability
          distributions
        - 'bc': Bhattacharyya coefficient, which is the overlap of two
          probability distributions.
        - 'bc2': squared Bhattacharyya coefficient.
        
    deconvolution
        Which method to use for deconvolution of the projections based on the
        cross-projections of `B`. If `False`, no deconvolution is done.
        Available methods are:
        
        - 'nnls': solves nnls to get only non-negative deconvolved projections
        - 'linear': solves a linear system to disentangle contributions; can
          result in negative values which makes sense for general vectors and
          amplitudes, i.e.
          
    parallel
        Whether to run the operation in parallel.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the distances.

    """

    if B.shape[1] != A.shape[1]:
        raise ValueError("`A` and `B` dont have the same second dimension! The shapes are %s and %s!" % (A.shape, B.shape))
        
    if isinstance(A,pd.DataFrame):
        A = A.to_numpy()
    if isinstance(B,pd.DataFrame):
        B = B.to_numpy()
    
    A,B = _math.cast_down_common(A,B)
    
    if metric == 'cosine':
        projection_function = cosine_projection
    elif metric == 'naive':
        projection_function = naive_projection
    elif metric == 'bc':
        projection_function = bhattacharyya_coefficient
    elif metric == 'bc2':
        projection_function = bhattacharyya2_coefficient
    elif metric == 'nwsp':
        projection_function = normalized_weighted_scalar_product
    elif metric == 'weighted':
        projection_function = weighted_projection
    else:
        raise ValueError(f'The `metric` "{metric}" is unknown!')
    
    # get projections of data on basis
    AB = projection_function(A, B, parallel=parallel)
    
    if deconvolution:
        # get projections of basis on basis (this gives the cross-talk between basis elements)
        BB = projection_function(B, B, parallel=parallel)
    
        if deconvolution == 'linear':
            AB = scipy.linalg.solve(BB.T, AB.T, assume_a='pos').T
        elif deconvolution == 'nnls':
            AB = _utils.parallel_nnls(BB.T, AB)
        else:
            raise ValueError(f'The `deconvolution` "{deconvolution}" is unknown!')
    
    return AB

def enum_shifts(n_shifts,n_dim,shifts=None):
    if n_dim == 0:
        return np.array(shifts)
    if shifts is None:
        shifts = [ [s] for s in range(n_shifts) ]
    else:
        shifts = [ [s,*shift] for s in range(n_shifts) for shift in shifts ]
    return enum_shifts(n_shifts,n_dim-1,shifts=shifts)

@njit(cache=True)
def _lexicographic_hash(codes, sizes):
    res = 0
    for size,code in zip(sizes, codes):
        if code < 0 or code >= size:
            return -1
        res *= size
        res += code
    return res
        
@njit(cache=True)
def _numba_getNN(coords,bins,shifts):
    sizes = np.array([np.max(coords[:,i])+1 for i in range(coords.shape[1])])
    
    xy2bin_dict = {}
    for i in range(len(bins)):
        xy2bin_dict[_lexicographic_hash(coords[i],sizes)] = bins[i]
    
    # for every coord there can be 3**nDimensions - 1 neighbours contributions
    # for every coord the first entry is the number of neighbours, adding + 1 to a total of 3**nDimensions
    others = np.zeros((coords.shape[0],3**coords.shape[1]),dtype=bins.dtype)
    for xi in range(len(bins)):
        bi,ci = bins[xi], coords[xi]
        bi_counter = 0
        for shift in shifts:
            bj = xy2bin_dict.get(_lexicographic_hash(ci+shift,sizes),None)
            if bj is not None:
                if bj > bi:
                    bi_counter += 1
                    others[bi,bi_counter] = bj
        others[bi,0] = bi_counter
    
    return others

def _python_getNN(coords,bins,shifts):
    
    xy2bin_dict = {tuple(ci):bi for ci,bi in zip(coords,bins)}
    
    def _getHyperCube(xy2bin_dict,ci,shifts):
        res = []
        for shift in shifts:
            add = xy2bin_dict.get(tuple(ci+shift),None)
            if add is not None:
                res.append(add)
        return res
    
    # for every coord there can be 3**nDimensions - 1 neighbours contributions
    # for every coord the first entry is the number of neighbours, adding + 1 to a total of 3**nDimensions
    others = np.zeros((coords.shape[0],3**coords.shape[1]),dtype=bins.dtype)
    for ci,bi in zip(coords,bins):
        bi_counter = 0
        for bj in _getHyperCube(xy2bin_dict,ci,shifts):
            if bj > bi:
                bi_counter += 1
                others[bi,bi_counter] = bj
        others[bi,0] = bi_counter
    
    return others

_numba_experimental_dict = -1
def _getNN(coords, bins, numba_experimental_dict):
    global _numba_experimental_dict
    if numba_experimental_dict == 'auto':
        if _numba_experimental_dict == -1:
            _x = np.array([0,0,1,3,5,5,6], dtype=np.int16)
            _y = np.array([0,1,1,0,0,6,6], dtype=np.int16)
            _b = np.array([0,1,2,3,4,5,6], dtype=np.int16)
            _coords = np.array([_x,_y]).T
            
            result_0 = _getNN(_coords, _b, numba_experimental_dict=0)
            result_2 = _getNN(_coords, _b, numba_experimental_dict=2)
            if np.all(result_2 == result_0):
                _numba_experimental_dict = 2
            else:
                result_1 = _getNN(_coords, _b, numba_experimental_dict=1)
                if np.all(result_1 == result_0):
                    _numba_experimental_dict = 1
                else:
                    _numba_experimental_dict = 0
            
        numba_experimental_dict = _numba_experimental_dict
    
    # shifts in the bin-vector to find all neighbors
    shifts = enum_shifts(n_shifts=3,n_dim=coords.shape[1])
    shifts = shifts-1
    
    if numba_experimental_dict != 0:
        sizes = np.array([np.max(coords[:,i])+1 for i in range(coords.shape[1])])
        total_size = 1
        for size in sizes:
            total_size = total_size * size
        if total_size > 2**63-1:
            warnings.warn('The bin hypercube is too large to be enumerated in a 64bit signed integer. The current implementation for `method=="numba"` with `numba_experimental_dict != 0` cant handle this case. Use bigger bins i.e. a larger max_distance, `numba_experimental_dict==0`, or `method=="scipy".` Falling back to `numba_experimental_dict==0`.')
            numba_experimental_dict = 0
    
    if numba_experimental_dict == 2:
        binNN = _numba_getNN(coords, bins, shifts)
    elif numba_experimental_dict == 1:
        binNN = _numba_getNN(coords.astype(np.int64), bins, shifts)
    else:
        binNN = _python_getNN(coords, bins, shifts)
    
    return binNN

@njit(cache=True)
def _get_indptr(bins,max_bin,dtype):
    indptr = np.zeros((max_bin+2),dtype=dtype)
    bin0 = 0
    b = 1
    for i in range(len(bins)):
        if bins[i] != bin0:
            bin0 = bins[i]
            indptr[b] = i
            b += 1
    indptr[-1] = len(bins)
    return indptr

@njit(cache=True)
def _row_col_data(distance, max_distance, whole_row, whole_col):
    
    total_obs = distance.shape[0] * distance.shape[1]
    row  = np.empty((total_obs,),dtype=whole_row.dtype)
    col  = np.empty((total_obs,),dtype=whole_col.dtype)
    data = np.empty((total_obs,),dtype=distance.dtype)
    c = 0
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            dij = distance[i,j]
            if dij != 0 and dij <= max_distance:
                row[c] = whole_row[i]
                col[c] = whole_col[j]
                data[c] = dij
                c += 1
    row = row[:c].copy()
    col = col[:c].copy()
    data = data[:c].copy()
    
    return row,col,data

@njit(fastmath=True, parallel=True, cache=True)
def _cdist_euclidean_parallel_(A,B): # cant use this as it runs into a bug in numba https://github.com/numba/numba/issues/7051
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    # code with constants for important cases enables powerful compiler optimizations
    if _nC == 2:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(2):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    elif _nC == 3:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(3):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    elif _nC == 1:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(1):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    else:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(_nC):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    return D

@njit(fastmath=True, parallel=True, cache=True)
def _cdist_euclidean_D2_parallel(A,B):
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    for a in prange(_nA):
        for b in range(_nB):
            temp = 0.0
            for c in range(2):
                delta = A[a,c] - B[b,c]
                delta *= delta
                temp += delta
            D[a,b] = np.sqrt(temp)
    return D
@njit(fastmath=True, parallel=True, cache=True)
def _cdist_euclidean_D3_parallel(A,B):
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    for a in prange(_nA):
        for b in range(_nB):
            temp = 0.0
            for c in range(3):
                delta = A[a,c] - B[b,c]
                delta *= delta
                temp += delta
            D[a,b] = np.sqrt(temp)
    return D
@njit(fastmath=True, parallel=True, cache=True)
def _cdist_euclidean_D1_parallel(A,B):
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    for a in prange(_nA):
        for b in range(_nB):
            temp = 0.0
            for c in range(1):
                delta = A[a,c] - B[b,c]
                delta *= delta
                temp += delta
            D[a,b] = np.sqrt(temp)
    return D
@njit(fastmath=True, parallel=True, cache=True)
def _cdist_euclidean_DN_parallel(A,B):
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    for a in prange(_nA):
        for b in range(_nB):
            temp = 0.0
            for c in range(_nC):
                delta = A[a,c] - B[b,c]
                delta *= delta
                temp += delta
            D[a,b] = np.sqrt(temp)
    return D
def _cdist_euclidean_parallel(A,B): # have to use this until numba bug is resolved https://github.com/numba/numba/issues/7051
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    # code with constants for important cases enables powerful compiler optimizations
    if _nC == 2:
        return _cdist_euclidean_D2_parallel(A,B)
    elif _nC == 3:
        return _cdist_euclidean_D3_parallel(A,B)
    elif _nC == 1:
        return _cdist_euclidean_D1_parallel(A,B)
    else:
        return _cdist_euclidean_DN_parallel(A,B)

@njit(fastmath=True, parallel=False, cache=True)
def _cdist_euclidean_serial(A,B):
    _nA, _nC = A.shape
    _nB, _nC = B.shape
    D = np.empty((_nA,_nB))
    # code with constants for important cases enables powerful compiler optimizations
    if _nC == 2:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(2):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    elif _nC == 3:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(3):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    elif _nC == 1:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(1):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    else:
        for a in prange(_nA):
            for b in range(_nB):
                temp = 0.0
                for c in range(_nC):
                    delta = A[a,c] - B[b,c]
                    delta *= delta
                    temp += delta
                D[a,b] = np.sqrt(temp)
    return D

def dense_distance_matrix(
    A,
    B=None,
    parallel=True,
):
    """\
    Calclulate a dense pairwise euclidean distance matrix of dense inputs.
    Compared to :func:`tacco.tools.cdist` this has a few extra optimizations for 1D-3D
    data - and only works for dense inputs.

    For a sparse version, see :func:`~tacco.utils.sparse_distance_matrix`.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray`.
    B
        A 2d :class:`~numpy.ndarray` with the same second dimension as `A`.
        If `None`, use `A`.
    parallel
        Whether to run the operation in parallel.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the distances.
    
    """

    if B is None:
        B = A

    if B.shape[1] != A.shape[1]:
        raise ValueError("`A` and `B` dont have the same second dimension! The shapes are %s and %s!" % (A.shape, B.shape))
        
    if isinstance(A,pd.DataFrame):
        A = A.to_numpy()
    if isinstance(B,pd.DataFrame):
        B = B.to_numpy()
    
    A,B = _math.cast_down_common(A,B)
    
    if parallel:
        return _cdist_euclidean_parallel(A,B)
    else:
        return _cdist_euclidean_serial(A,B)

@njit(parallel=False, cache=True)
def _run_single(i, indptr, whole_row, pos, max_distance, binNN_i, row, col, data, rowNN, colNN, dataNN):
    
    _whole_row = whole_row[indptr[i]:indptr[i+1]]
    _pos = pos[indptr[i]:indptr[i+1]]
    # calculate distances within bin
    _distance = _cdist_euclidean_serial(_pos,_pos)
    _row,_col,_data = _row_col_data(_distance, max_distance, _whole_row, _whole_row)
    
    c1 = len(_row)
    row[0] = len(_row)
    row[1:(c1+1)] = _row
    col[0:c1] = _col
    data[0:c1] = _data

    c0 = 0
    binNN_i
    for j in binNN_i:
        # calculate distances to neighbour bins
        _whole_col = whole_row[indptr[j]:indptr[j+1]]
        _posNN = pos[indptr[j]:indptr[j+1]]
        _distance = _cdist_euclidean_serial(_pos, _posNN)
        _row,_col,_data = _row_col_data(_distance, max_distance, _whole_row, _whole_col)
    
        c1 = c0 + len(_row)
        rowNN[(c0+1):(c1+1)] = _row
        colNN[c0:c1] = _col
        dataNN[c0:c1] = _data
        c0 = c1
    rowNN[0] = c0
    
@njit(parallel=True, cache=True)
def _run_batch(i_range, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN):
    for _i in prange(len(i_range)):
        i = i_range[_i]
        li = load_balance_ordering[i]
        binNN_i = binNN[li,1:(binNN[li,0]+1)]
        _run_single(li, indptr, whole_row, pos, max_distance, binNN_i, _row[_i], _col[_i], _data[_i], _rowNN[_i], _colNN[_i], _dataNN[_i])
@njit(parallel=False, cache=True)
def _run_batch_serial(i_range, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN):
    for _i in prange(len(i_range)):
        i = i_range[_i]
        li = load_balance_ordering[i]
        binNN_i = binNN[li,1:(binNN[li,0]+1)]
        _run_single(li, indptr, whole_row, pos, max_distance, binNN_i, _row[_i], _col[_i], _data[_i], _rowNN[_i], _colNN[_i], _dataNN[_i])
    
@njit(parallel=False, cache=True)
def _run_batch_balanced_core(b, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN):
    n_extra = balance_indptr[b+1]-balance_indptr[b]
    i_range = np.empty((1+n_extra,), dtype=load_balance_ordering.dtype) 
    i_range[0] = -b-1
    i_range[1:(1+n_extra)] = np.arange(balance_indptr[b],balance_indptr[b+1])

    offset = 0
    offsetNN = 0
    for _i in range(len(i_range)):
        i = i_range[_i]
        li = load_balance_ordering[i]
        binNN_i = binNN[li,1:(binNN[li,0]+1)]

        _row0 = _row[offset] # save the last entries of the rows: will be overwritten by the number of results
        _rowNN0 = _rowNN[offsetNN]
        _run_single(li, indptr, whole_row, pos, max_distance, binNN_i,
                    _row[offset:], _col[offset:], _data[offset:],
                    _rowNN[offsetNN:], _colNN[offsetNN:], _dataNN[offsetNN:],
                   )

        new_results = _row[offset]
        new_resultsNN = _rowNN[offsetNN]
        if _i > 0: # restore the last entries of the rows
            _row[offset] = _row0
            _rowNN[offsetNN] = _rowNN0

        offset += new_results
        offsetNN += new_resultsNN
    # write the total number of results
    _row[0] = offset
    _rowNN[0] = offsetNN
    
@njit(parallel=True, cache=True)
def _run_batch_balanced(b_range, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, row, col, data, rowNN, colNN, dataNN):
    for _b in prange(len(b_range)):
        b = b_range[_b]
        _run_batch_balanced_core(b, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, row[_b], col[_b], data[_b], rowNN[_b], colNN[_b], dataNN[_b])
    
@njit(parallel=False, cache=True)
def _run_batch_balanced_serial(b_range, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, row, col, data, rowNN, colNN, dataNN):
    for _b in prange(len(b_range)):
        b = b_range[_b]
        _run_batch_balanced_core(b, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, row[_b], col[_b], data[_b], rowNN[_b], colNN[_b], dataNN[_b])

@njit(cache=True)
def _max_n_i(indptr, binNN, bin_batching):
    n = len(indptr) - 1
    max_n_i = int(bin_batching**2)
    max_n_i_NN = int(bin_batching**2 * np.log(binNN.shape[1]) / np.log(3))
    all_max = np.empty((n,),dtype=np.int64)
    all_n_i = np.empty((n,),dtype=np.int64)
    all_n_i_NN = np.empty((n,),dtype=np.int64)
    for i in range(n):
        n_i = int(indptr[i+1] - indptr[i])
        n_i *= n_i
        if n_i > max_n_i:
            max_n_i = n_i

        binNN_i_N = binNN[i][0]
        binNN_i = binNN[i][1:(binNN_i_N+1)]
        
        n_binNN_i = 0
        for j in binNN_i:
            n_binNN_i += indptr[j+1] - indptr[j]
        n_binNN_i *= int(indptr[i+1] - indptr[i])

        if max_n_i_NN < n_binNN_i:
            max_n_i_NN = n_binNN_i
            
        all_n_i[i] = n_i
        all_n_i_NN[i] = n_binNN_i
        
        all_max[i] = n_i + n_binNN_i # this should be proportional to the dense distance calculation cost - and (a bit above) the maximum distance storage requirement

    # sort the bins by cost: makes calculation in bunches better load balanced
    ordering = all_max.argsort()
    
    # create batches of a single expensive and (many) small ones to make the batches have a near equal cost
    # a batch `i` consists of a large bin `ordering[-i]` and the small ones indicated by `ordering[indptr[i]:indptr[i+1]]`
    balance_indptr = np.zeros((len(ordering)+1,),dtype=np.int64)
    back_id = n - 1
    front_id = 0
    balance_indptr[0] = 0
    batch_id = 1
    while back_id >= front_id:
        current_size = all_n_i[ordering[back_id]]
        next_size = all_n_i[ordering[front_id]]
        current_size_NN = all_n_i_NN[ordering[back_id]]
        next_size_NN = all_n_i_NN[ordering[front_id]]
        while (current_size + next_size) <= max_n_i and (current_size_NN + next_size_NN) <= max_n_i_NN and back_id > front_id:
            current_size += next_size
            front_id += 1
            next_size = all_max[ordering[front_id]]
        balance_indptr[batch_id] = front_id
        batch_id += 1
        back_id -= 1
    balance_indptr = balance_indptr[:batch_id].copy()
    
    #print('number of points/bins/jobs:', indptr[-1], '/', n, '/', len(balance_indptr)-1)
    #print(all_max)
    #print(all_max[ordering])
    #print(ordering, indptr)
    #for i in range(len(indptr)-1):
    #    print(ordering[-i-1], all_max[ordering[-i-1]], ordering[indptr[i]:indptr[i+1]], all_max[ordering[indptr[i]:indptr[i+1]]])
    
    return max_n_i, max_n_i_NN, ordering, balance_indptr

@njit(cache=True)
def _collect_results(_row, _col, _data, _rowNN, _colNN, _dataNN):
    # first sweep: check how much space is needed
    n_total = 0
    n_buffer = len(_row)
    for _i in range(n_buffer):
        n_total += _row[_i,0]
        n_total += 2 * _rowNN[_i,0] # account for distance symmetry
    
    # allocate memory
    row = np.empty((n_total,),dtype=_row.dtype)
    col = np.empty((n_total,),dtype=_col.dtype)
    data = np.empty((n_total,),dtype=_data.dtype)
    
    # second sweep: do actual collection
    r0 = 0
    r1 = 0
    for _i in range(n_buffer):
        c1 = _row[_i,0]
        if c1 > 0:
            r1 = r0 + c1
            row[r0:r1] = _row[_i,1:(c1+1)]
            col[r0:r1] = _col[_i,:c1]
            data[r0:r1] = _data[_i,:c1]
            r0 = r1

        c1 = _rowNN[_i,0]
        if c1 > 0:
            r1 = r0 + c1
            row[r0:r1] = _rowNN[_i,1:(c1+1)]
            col[r0:r1] = _colNN[_i,:c1]
            data[r0:r1] = _dataNN[_i,:c1]
            r0 = r1
            # add the same result also for the transposed bin neighbour combination
            r1 = r0 + c1
            col[r0:r1] = _rowNN[_i,1:(c1+1)]
            row[r0:r1] = _colNN[_i,:c1]
            data[r0:r1] = _dataNN[_i,:c1]
            r0 = r1
    
    return row, col, data

def _run_all(indptr, whole_row, pos, max_distance, binNN, parallel, batch_size, bin_batching, buffer_directory=None):
    max_n_i, max_n_binNN_i, load_balance_ordering, balance_indptr = _max_n_i(indptr, binNN, bin_batching)
    
    if bin_batching:
        n = len(balance_indptr) - 1
    else:
        n = len(indptr) - 1
        
    last_batch_size = n % batch_size
    n_batches = n // batch_size + (last_batch_size > 0)

    shape0 = min(batch_size, n)

    _row  = np.empty((shape0,max_n_i+1),dtype=whole_row.dtype)
    _col  = np.empty((shape0,max_n_i  ),dtype=whole_row.dtype)
    _data = np.empty((shape0,max_n_i  ),dtype=pos.dtype)

    _rowNN  = np.empty((shape0,max_n_binNN_i+1),dtype=whole_row.dtype)
    _colNN  = np.empty((shape0,max_n_binNN_i  ),dtype=whole_row.dtype)
    _dataNN = np.empty((shape0,max_n_binNN_i  ),dtype=pos.dtype)
    
    row = []
    col = []
    data = []
    
    for batch in range(n_batches):
        i_start = batch*batch_size
        if batch == n_batches - 1 and last_batch_size > 0:
            i_end = i_start + last_batch_size
        else:
            i_end = i_start + batch_size
        i_range = np.arange(i_start,i_end)

        _row[:,0] = 0
        _rowNN[:,0] = 0

        if bin_batching:
            if parallel:
                _run_batch_balanced(i_range, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN)
            else:
                _run_batch_balanced_serial(i_range, balance_indptr, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN)
        else:
            if parallel:
                _run_batch(i_range, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN)
            else:
                _run_batch_serial(i_range, load_balance_ordering, indptr, whole_row, pos, max_distance, binNN, _row, _col, _data, _rowNN, _colNN, _dataNN)

        __row, __col, __data = _collect_results(_row, _col, _data, _rowNN, _colNN, _dataNN)
        
        if len(__row) > 0:
            if buffer_directory is None:
                row.append(__row)
                col.append(__col)
                data.append(__data)
            else:
                with open(buffer_directory+'row.bin', 'ab') as f:
                    __row.tofile(f)
                with open(buffer_directory+'col.bin', 'ab') as f:
                    __col.tofile(f)
                with open(buffer_directory+'data.bin', 'ab') as f:
                    __data.tofile(f)
                row.append(len(__row))
    
    return row, col, data

def _guess_blocksize(positions, max_distance, verbose=1):
    # find a blocksize large enough that the "max" block has at a high but not too high number of points to make dense subproblems efficient while keeping memory low and sparsity gain high.
    n_dim = positions.shape[1]
    maxN = 0
    factor_i = 0
    blocks = 1
    numba_blocksize = max_distance
    while maxN < 200 and blocks >= 1:
        factor_i += 1
        factor = 2**(factor_i/n_dim)
        if factor > 10: # different ballpark...
            break
        numba_blocksize = float(f'{max_distance * 2**(factor_i/n_dim):.1e}') # round to 2 significant digits for convenience
        bins = bin(positions, bin_size=numba_blocksize, position_keys=positions.columns);
        hashes = hash(bins);
        Avc = hashes.value_counts()
        maxN = Avc.max()
        blocks = len(Avc)
    factor_i -= 1
    numba_blocksize = float(f'{max_distance * 2**(factor_i/n_dim):.1e}') # round to 2 significant digits for convenience
    if verbose > 0:
        print(f'The heuristic value for the parameter `numba_blocksize` is {numba_blocksize}. Consider specifying this directly as argument to avoid (possibly significant) overhead and/or experiment with this value on (a subset of) the actual dataset at hand to obtain an optimal value in terms of speed and memory requirements.')
    return numba_blocksize

def _numba_sparse_distance_matrix(A, max_distance, numba_blocksize, numba_experimental_dict, parallel, batch_size, bin_batching, low_mem=False, verbose=1):
    dtype = A.dtype
    n_dim = A.shape[1]
    
    A = pd.DataFrame(A)
    A.columns = A.columns.astype(str)
    
    keys = A.columns.copy()
    bin_keys = keys + '_bin'
    bin_key = 'bin'

    if batch_size is None:
        batch_size = numba.get_num_threads()
        
    if numba_blocksize is None:
        numba_blocksize = _guess_blocksize(A[keys], max_distance, verbose=verbose)
    
    if bin_key not in A:
        bin(A, bin_size=numba_blocksize, position_keys=keys, bin_keys=bin_keys);
        hash(A, keys=bin_keys, hash_key=bin_key);
    A = A.sort_values(bin_key)
    
    # get mapping from bin to all the neighbours with binID larger than itself
    binA = A.drop_duplicates(bin_key).sort_values(bin_key)
    binNN = _getNN(binA[bin_keys].to_numpy(), binA[bin_key].to_numpy(), numba_experimental_dict)

    indptr = _get_indptr(A[bin_key].to_numpy(),A[bin_key].max(),min_dtype(len(A)))

    nobs = A.shape[0]
    pos = A[keys].to_numpy().copy()
    
    whole_row = A.index.to_numpy().copy()
    
    if low_mem:
        tempdir = tempfile.TemporaryDirectory(prefix='temp_sparse_distance_matrix_',dir='.')
        buffer_directory = tempdir.name + '/'
    else:
        buffer_directory = None

    try:
        # make the matrix-vector product deep in the euclidean metric run single threaded to use more efficient higher level paralellism
        with threadpool_limits(1,'blas'):
            row, col, data = _run_all(indptr, whole_row, pos, max_distance, binNN, parallel=parallel, batch_size=batch_size, bin_batching=bin_batching, buffer_directory=buffer_directory)

        # this way of constructing a coo_matrix has slightly less overhead... 
        distance = scipy.sparse.coo_matrix((nobs,nobs),dtype=dtype)
        if len(row) > 0:
            if buffer_directory is None:
                distance.row = np.concatenate(row)
                distance.col = np.concatenate(col)
                distance.data = np.concatenate(data)
            else:
                distance.row = np.fromfile(buffer_directory+'row.bin', dtype=whole_row.dtype)
                distance.col = np.fromfile(buffer_directory+'col.bin', dtype=whole_row.dtype)
                distance.data = np.fromfile(buffer_directory+'data.bin', dtype=pos.dtype)

    finally:
        if low_mem:
            tempdir.cleanup()
    
    return distance

def sparse_distance_matrix(
    A,
    max_distance,
    method='numba',
    numba_blocksize=None,
    numba_experimental_dict='auto',
    dtype=np.float64,
    parallel=True,
    batch_size=None,
    bin_batching=200,
    low_mem=False,
    verbose=1,
):
 
    """\
    Calclulate a sparse pairwise distance matrix of dense inputs. Only
    euclidean metric is supported.

    For a dense version, see :func:`~tacco.utils.dense_distance_matrix`.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray`.
    max_distance
        A distance cutoff: all distances larger than this value are excluded
        from the result.
    method
        A string indicating the method to use. Available are:
        
        - 'scipy': Use :func:`scipy.spatial.cKDTree.sparse_distance_matrix`.
          This is most efficient for not too many points and relatively small
          `max_distance`.
        - 'numba': Use a custom `numba` based implementation, which is much
          faster for larger `max_distance` and datasets.
    numba_blocksize
        If `method` is 'numba', this gives the size of the blocks within which
        the distances should be computed densely. Has to be at least
        `max_distance` and - depending on the dataset - should be several times
        larger for optimal performance. If `None`, use a heuristic to find a
        reasonable value. Smaller values need less memory.
    numba_experimental_dict
        If `method` is 'numba', how to accelerate some parts of the code by
        using numba dictionaries, which is an experimental numba feature. If
        'auto', runs a small test set to determine whether numba dicts seem to
        work and uses them accordingly.
    dtype
        The data type to use for calculations. Internally method 'scipy' always
        uses `np.float64`, i.e. double precision. Method 'numba' can get a
        significant speedup from using lower precision - which can also lead to
        rounding errors making different nearby points have numerically 0
        distance, which is then discarded in the sparse result... All
        non-floating dtypes will be casted to `np.float64`.
    parallel
        Whether to run using multiple cores. Only method 'numba' supports this
        option.
    batch_size
        The number of blocks to calculate per batch. Small values need less
        memory while large values tend to give higher parallel speedup. If
        `None`, uses the number of available threads - even if
        `parallel==False` and only a single thread is used.
    bin_batching
        If larger than `0`, the calculations for individual blocks are grouped
        such that (mostly) at least `bin_batching` points are considered
        together. This reduces overhead and the impact of choosing a too small
        `numba_blocksize`.
    low_mem
        Whether to access harddisc to buffer big temporaries. This is slower
        than in memory operations, but can reduce the memory consumption by
        a factor of 2.
        
    Returns
    -------
    A :class:`~scipy.sparse.coo_matrix` containing the distances.

    """
    
    if not isinstance(A, np.ndarray):
        raise ValueError('`A` has to be an numpy.ndarray!')
    if A.dtype != dtype:
        A = A.astype(dtype)
    if not pd.api.types.is_float_dtype(A):
        A = A.astype(np.float64)

    if method == 'scipy':
        kd_tree = scipy.spatial.cKDTree(A)
        distance = kd_tree.sparse_distance_matrix(kd_tree, max_distance)

        distance = distance.tocoo()

        distance.eliminate_zeros()
        
    elif method == 'numba':
        
        if numba_blocksize is not None and numba_blocksize < max_distance:
            raise ValueError('`numba_blocksize < max_distance`')
        
        distance = _numba_sparse_distance_matrix(A, max_distance, numba_blocksize, numba_experimental_dict=numba_experimental_dict, parallel=parallel, batch_size=batch_size, bin_batching=bin_batching, low_mem=low_mem)
        
    else:
        raise ValueError(f'`method` "{method}" is unknown!')

    if distance.dtype != dtype:
        distance = distance.astype(dtype)

    return distance

