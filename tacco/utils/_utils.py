import warnings
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.optimize import nnls
from numpy.random import Generator, PCG64
import scipy.spatial.distance
import multiprocessing
import os
import joblib
from difflib import SequenceMatcher
import scanpy as sc
import time
import gc
import mkl
import tempfile
from numba import njit, prange
from ..get._counts import _get_counts_location
from .. import get
from . import dist
from . import math

def cpu_count():

    """\
    Return the number of available CPU cores.
        
    Returns
    -------
    The number of available CPU cores.
        
    """
    
    if 'NSLOTS' in os.environ: # if Sun Grid Engine is used, we can get the number of allocated cores here 
        return int(os.environ.get('NSLOTS'))
    mkl_threads = mkl.get_max_threads()
    if mkl_threads is not None:
        return mkl_threads
    return multiprocessing.cpu_count()

def parallel_nnls(
    A,
    B,
    n_jobs=None,
    batch_size=50
):

    """\
    Runs multiple nnls in parallel.
    
    Parameters
    ----------
    A
        A 2d :class:`~numpy.ndarray`.
    B
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix with the second
        dimension identical to the first dimension of as `A`.
    n_jobs
        How many jobs/cores should be run on. If `None`, runs on all available
        cores.
    batch_size
        The number of problems to assign to a job/core per batch.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the results.
        
    """
    def _nnls(A,B):
        if scipy.sparse.issparse(B):
            return np.array([ nnls(A, B[g].A.flatten())[0] for g in range(B.shape[0]) ])
        else:
            return np.array([ nnls(A, B[g])[0] for g in range(B.shape[0]) ])
    
    if not isinstance(A,np.ndarray):
        raise ValueError(f"`A` can only be a numpy array but is a {A.__class__}!")
    if not scipy.sparse.issparse(B) and not isinstance(B,np.ndarray):
        raise ValueError(f"`B` can only be a scipy sparse matrix or a numpy array but is a {B.__class__}!")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    result = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_nnls)(A, B[g:(g+batch_size)])
        for g in range(0,B.shape[0],batch_size)
    )
    return np.vstack(result)

def solve_OT(
    a,
    b,
    M,
    epsilon=5e-3,
    lambda_a=None,
    lambda_b=None,
    numItermax=1000,
    stopThr=1e-9,
    inplace=False,
    ):
    
    """\
    Solve optimal transport problem with entropy regularization and optionally
    Kullback-Leibler divergence penalty terms instead of exact marginal
    conservation.
    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm from in [Chizat16]_. The python
    implementation is based on [Flamary2021]_.
    
    Parameters
    ----------
    a
        A 1d array-like containing the left marginal distribution
    b
        A 1d array-like containing the right marginal distribution
    M
        A 2d array-like containing the loss matrix
    epsilon
        The entropy regularization parameter
    lambda_a
        The left marginal relaxation parameter; if `None`, enforce marginal
        exactly like in balanced OT
    lambda_b
        The right marginal relaxation parameter; if `None`, enforce marginal
        exactly like in balanced OT
    numItermax
        The maximal number of iterations
    stopThr
        The error threshold for the stopping criterion
    inplace
        Whether `M` will contain the transort matrix upon completion or be
        unchanged. `M` has to be a :class:`~numpy.ndarray` with
        `dtype=np.float64` for this.
        
    Returns
    -------
    Depending on `inplace` returns a 2d :class:`~numpy.ndarray` containing\
    the transport couplings, which either is the inplace updated `M` or a\
    newly allocated array.
        
    """

    # high precision is necessary 
    a = np.array(a, dtype=np.float64, copy=False)
    b = np.array(b, dtype=np.float64, copy=False)
    _M = M
    M = np.array(M, dtype=np.float64, copy=False)
    
    if inplace and _M is not M:
        raise Exception('`inplace` is `True` but impossible, as `M` is not a `np.float64` `np.ndarray`!')

    dim_a, dim_b = M.shape

    if len(a) != dim_a:
        raise Exception('Marginal `a` and cost `M` have incompatible shapes %s and %s!' % (a.shape, M.shape))
    if len(b) != dim_b:
        raise Exception('Cost `M` and marginal `b` have incompatible shapes %s and %s!' % (M.shape, b.shape))
    if epsilon <= 0:
        raise Exception('Entropy regularization `epsilon` must be positive!')
    if numItermax <= 0:
        raise Exception('The number of iterations `numItermax` must be positive!')
    if stopThr <= 0:
        raise Exception('The number of iterations `stopThr` must be positive!')

    u = np.ones(dim_a, dtype=np.float64) / dim_a
    v = np.ones(dim_b, dtype=np.float64) / dim_b

    if inplace:
        K = M
    else:
        K = np.empty(M.shape, dtype=np.float64)
    np.divide(M, -epsilon, out=K)
    np.exp(K, out=K)

    fa = 1 if lambda_a is None else lambda_a / (lambda_a + epsilon)
    fb = 1 if lambda_b is None else lambda_b / (lambda_b + epsilon)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = a / Kv
        if fa != 1:
            u = u ** fa
        Ktu = K.T.dot(u)
        v = b / Ktu
        if fb != 1:
            v = v ** fb

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        if i % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
            err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
            err = 0.5 * (err_u + err_v)
            if err < stopThr:
                break

    math.row_scale(K, u)
    math.col_scale(K, v)
    
    return K

def run_OT(type_cell_dist, type_prior=None, cell_prior=None, epsilon=5e-3, lamb=None, inplace=False):

    # check sanity of arguments
    if type_prior is not None and type_prior.isna().any():
        raise Exception('type_prior contains na!')
    if type_prior is not None and (type_prior<0).any():
        raise Exception('type_prior contains negative values!')
    if cell_prior is not None and cell_prior.isna().any():
        raise Exception('cell_prior contains na!')
    if cell_prior is not None and (cell_prior<0).any():
        raise Exception('cell_prior contains negative values!')
    if type_cell_dist.isna().any().any():
        raise Exception('type_cell_dist contains na!')
    
    # dont carry references to the distance around forever and thereby making garbage collection impossible, therefore copy the indices
    types = type_cell_dist.index.copy()
    cells = type_cell_dist.columns.copy()

    if type_prior is None:
        ntypes = len(types)
        p = np.full((ntypes,), 1 / ntypes)
    else:
        p = type_prior[types] / type_prior.sum()
        p = p.to_numpy()
    
    if cell_prior is None:
        ncells = len(cells)
        q = np.full((ncells,), 1 / ncells)
    else:
        q = cell_prior[cells] / cell_prior.sum()
        q = q.to_numpy()
    
    cost = type_cell_dist.loc[types].to_numpy().T
    cost /= cost.max()
    
    C = solve_OT(q, p, cost, epsilon, lambda_b=lamb, inplace=inplace)

    C = pd.DataFrame(C, columns=types, index=cells)

    return C

def run_nnls(tdata, reference, type_key, n_pca=None, zero_center=False):
    tdata = tdata.copy()
    reference = reference.copy()
    
    tX, rX, pca_offset, pca_trafo = transfer_pca(tdata, reference, n_pca, zero_center=zero_center)
    
    # create average type profiles on un-lognormalized data and use them to get the marginal distributions of celltypes via nnls
    average_profiles = { l: np.array(rX[reference.obs.index.isin(df.index)].mean(axis=0)).flatten() for l, df in reference.obs.groupby(type_key) }
    average_profiles = pd.DataFrame(average_profiles)#, index=reference.var_names)
    #print(average_profiles
    
    cell_type = parallel_nnls(average_profiles, tX)
    
    cell_type = pd.DataFrame(cell_type, columns=average_profiles.columns.copy(), index=tdata.obs.index.copy())
    
    del tdata, reference # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    return cell_type

def scale_counts(adata, rescaling_factors, counts_location=None, round=False):
    if adata.is_view:
        raise ValueError(f'Got a view of an anndata.AnnData as an argument to scale inplace! This does not (reliably) work, supply a copy.')
        
    counts_location = _get_counts_location(adata, counts_location=counts_location)
    
    # scale the counts right where they are
    if counts_location[0] == 'X':
        math.col_scale(adata.X, rescaling_factors, round=round)
    elif counts_location[0] == 'layer':
        math.col_scale(adata.layers[counts_location[1]], rescaling_factors, round=round)
    elif counts_location[0] == 'obsm':
        math.col_scale(adata.obsm[counts_location[1]], rescaling_factors, round=round)
    elif counts_location[0] == 'varm':
        math.col_scale(adata.varm[counts_location[1]], rescaling_factors, round=round)
    elif counts_location[0] == 'uns':
        math.col_scale(adata.uns[counts_location[1]], rescaling_factors, round=round)
    elif counts_location[0] == 'raw':
        scale_counts(adata.raw.to_adata(), rescaling_factors, counts_location=counts_location[1:], round=round)
    else:
        raise Exception('The counts location "%s" cannot be interpreted!' % str(counts_location))

def find_unused_key(
    dict_like,
    start='unused',
):

    """\
    Finds an unused key.
    
    Parameters
    ----------
    dict_like
        A dict-like or a list/tuple of dict-likes to find a key which is unused
        in all of them.
    start
        The keys "<start><N>" are checked for increasing `N` until an unused
        key is found.
        
    Returns
    -------
    The unused key.
        
    """
    if not isinstance(dict_like, (list,tuple)):
        dict_likes = [dict_like]
    else:
        dict_likes = dict_like
    
    new_key = start
    counter = 0
    for dict_like in dict_likes:
        while new_key in dict_like:
            new_key = start + str(counter)
            counter += 1
    return new_key
        
def get_average_profiles(type_key, reference, rX=None, pca_offset=None, pca_trafo=None, std=False, normalize=True, cell_weights=False, buffer=True):
    new_key = None
    if isinstance(type_key, pd.Series):
        new_key = find_unused_key(reference.obs)
        reference.obs[new_key] = type_key
        type_key = new_key
    elif isinstance(type_key, pd.DataFrame):
        new_key = find_unused_key(reference.obsm)
        reference.obsm[new_key] = type_key
        type_key = new_key

    found_key = []
    
    if std and pca_trafo is not None:
        raise ValueError('`pca_trafo` is not implemented for the standard deviation!')
    
    normalization = None
    
    if type_key in reference.varm:
        found_key.append('varm')
        if len(found_key) == 1: # only use this hit, if there was no better hit before
            if std:
                raise ValueError('If profiles are given explicitely, their standard deviation is not available!')
            mean_profiles = reference.varm[type_key].fillna(0)

            normalization = mean_profiles.sum(axis = 0).to_numpy()
            if pca_offset is not None:
                mean_profiles = mean_profiles - pca_offset[:,None]
            if pca_trafo is not None:
                mean_profiles = pca_trafo.T @ mean_profiles

    if type_key in reference.obs.columns:
        found_key.append('obs')
        if len(found_key) == 1: # only use this hit, if there was no better hit before
            if rX is None:
                rX = reference.X

            if cell_weights: # weight every cell equally instead of every read
                rX = rX.copy()
                math.row_scale(rX, 1/np.array(reference.X.sum(axis=1)).flatten())
            
            mean_profiles, std_profiles = {}, {}
            for l, df in reference.obs.groupby(type_key):
                rX_l = rX[reference.obs.index.isin(df.index)]
                if std:
                    if scipy.sparse.issparse(rX):
                        mean_profiles[l], std_profiles[l] = mean_variance_axis(rX_l, axis=0)
                        std_profiles[l] = np.sqrt(std_profiles[l])
                    else:
                        mean_profiles[l], std_profiles[l] = rX_l.mean(axis=0), rX_l.std(axis=0)
                else:
                    mean_profiles[l] = np.array(rX_l.mean(axis=0)).flatten()
            mean_profiles = pd.DataFrame(mean_profiles)
            if std:
                std_profiles = pd.DataFrame(std_profiles)

    if type_key in reference.obsm:
        found_key.append('obsm')
        if len(found_key) == 1: # only use this hit, if there was no better hit before
            if std:
                raise ValueError('If type annotation is given as soft clusters, the standard deviation of their profiles is not available!')
            _type_annotation = reference.obsm[type_key]
            print('OP....',end='...')
            start = time.time()
            type_gene_dist = pd.DataFrame(dist.cdist(_type_annotation.T, reference.X.T, metric='cosine'), index=_type_annotation.columns, columns=reference.var.index) # normalization is irrelevant for the cosine distance
            _type_annotation *= (np.array(reference.X.sum(axis=1)).flatten() / _type_annotation.sum(axis=1).to_numpy())[:,None] # normalize cell-type matrix to read counts per cell
            type_prior = _type_annotation.sum(axis=0) # reads per type
            gene_prior = pd.Series(np.array(reference.X.sum(axis=0)).flatten(),index=reference.var.index) # reads per gene
            mean_profiles = run_OT(type_cell_dist=type_gene_dist, type_prior=type_prior, cell_prior=gene_prior, epsilon=5e-3)
            print('time',time.time() - start)
            #print('nnls..',end='...')
            #start = time.time()
            #mean_profiles = parallel_nnls(_type_annotation.to_numpy(),rX.T)
            #print('time',time.time() - start)
            mean_profiles = pd.DataFrame(mean_profiles,columns=_type_annotation.columns)
    
    if new_key is not None:
        if new_key in reference.obs:
            del reference.obs[new_key]
        else:
            del reference.obsm[new_key]

    #if len(found_key) > 1:
    #    print('Found the key "%s" in several places %s! Used the one in \'%s\' for average profiles.' % (type_key, found_key, found_key[0]))
    if len(found_key) < 1:
        raise ValueError('The key "%s" is not found in the reference!' % (type_key))

    if normalize:
        # normalize profiles to sum to 1 over genes for every profile
        if normalization is None: # some paths might need other normalization...
            normalization = mean_profiles.sum(axis = 0).to_numpy()
        mean_profiles /= normalization
        if std:
            std_profiles /= normalization
    # recover the gene index if it makes sense 
    if len(mean_profiles.index) == len(reference.var.index):
        mean_profiles.set_index(reference.var.index, inplace=True)
        if std:
            std_profiles.set_index(reference.var.index, inplace=True)
    if buffer and 'varm' not in found_key:
        reference.varm[type_key] = mean_profiles
    #print(f'mean_profiles {mean_profiles} at the end')
    if std:
        return mean_profiles, std_profiles
    else:
        return mean_profiles

def preprocess_single_cell_data(tdata, hvg=True, scale=True, pca=True, inplace=False,min_cells=10, min_genes=10):
    print('SCprep', end='...')
    start = time.time()
    if not inplace:
        tdata = tdata.copy()
    if pd.api.types.is_integer_dtype(tdata.X):
        if inplace:
            raise ValueError(f'The integer valued data cannot be normalized inplace! Specify `inplace=False`.')
        else:
            tdata.X = tdata.X.astype(float)
    tdata.obs['log10counts'] = np.log(np.array(tdata.X.sum(axis=1)).flatten()) / np.log(10)
    if min_cells is not None:
        sc.pp.filter_genes(tdata, min_cells=min_cells)
    if min_genes is not None:
        sc.pp.filter_cells(tdata, min_genes=min_genes)
    sc.pp.normalize_per_cell(tdata, counts_per_cell_after=1e4)
    math.log1p(tdata)
    if hvg:
        n_bins = min(tdata.shape[0], 20) # catch edge case with less than default n_bins samples
        sc.pp.highly_variable_genes(tdata, subset=True, n_bins=n_bins)
    if scale:
        sc.pp.scale(tdata)
    if pca:
        sc.pp.pca(tdata, random_state=42)
    print('time', time.time() - start)
    return tdata

def umap_single_cell_data(tdata, inplace=False, **kwargs):
    print('SCumap', end='...')
    start = time.time()
    if not inplace:
        tdata = tdata.copy()
    tdata = preprocess_single_cell_data(tdata, inplace=True, **kwargs)
    sc.pp.neighbors(tdata, random_state=42)
    #sc.tl.leiden(tdata, random_state=42)
    sc.tl.umap(tdata, random_state=42)
    print('time', time.time() - start)
    return tdata

def tsne_single_cell_data(tdata, inplace=False, **kwargs):
    print('SCtsne', end='...')
    start = time.time()
    if not inplace:
        tdata = tdata.copy()
    tdata = preprocess_single_cell_data(tdata, inplace=True, **kwargs)
    # sc.pp.neighbors(tdata, random_state=42)
    #sc.tl.leiden(tdata, random_state=42)
    sc.tl.tsne(tdata, random_state=42)
    print('time', time.time() - start)
    return tdata


def transfer_pca(tdata, reference, n_pca=100, zero_center=True):
    tX = tdata.X
    offset = None
    if zero_center:
        offset = np.array(reference.X.mean(axis=0, dtype=np.float64)).flatten()
        if scipy.sparse.issparse(tX):
            tX = tX.toarray()
        tX = tX - offset
    if n_pca is None:
        rX = reference.X
        if zero_center:
            if scipy.sparse.issparse(rX):
                rX = rX.toarray()
            rX = rX - offset
        return tX, rX, offset, None
    # perform pca in reference data ...
    sc.tl.pca(reference, n_comps=n_pca, zero_center=zero_center, use_highly_variable=False)
    # ... and transfer it to the test data
    tdata.obsm['X_pca'] = tX @ reference.varm['PCs']
    return tdata.obsm['X_pca'], reference.obsm['X_pca'], offset, reference.varm['PCs']

def generate_mixture_profiles(
    average_profiles,
    include_pure_profiles=False,
):

    """\
    Creates symmetric pairwise mixture profiles.
    
    Parameters
    ----------
    average_profiles
        A 2d :class:`~numpy.ndarray` with features in the rows and profiles in
        the columns
    include_pure_profiles
        Whether to include pure profiles in the result, too.
        
    Returns
    -------
    A tuple consisting of a :class:`~numpy.ndarray` containing the mixture\
    profiles and a sparse matrix giving the mixtures.
        
    """
    
    if not isinstance(average_profiles, np.ndarray):
        raise ValueError('`average_profiles` has to be a numpy array!')
    n_genes, n_profiles = average_profiles.shape
    n_mixtures = (n_profiles * (n_profiles - 1)) // 2
    if include_pure_profiles:
        n_mixtures += n_profiles
    
    rows = np.empty(2*n_mixtures, dtype=np.int64)
    cols = np.empty(2*n_mixtures, dtype=np.int64)
    data = np.full(2*n_mixtures, 1/(2*n_mixtures), dtype=average_profiles.dtype)
    
    counter = 0
    for i in range(n_profiles):
        j0 = i if include_pure_profiles else i+1
        for j in range(j0, n_profiles):
            rows[2*counter+0] = counter
            cols[2*counter+0] = i
            rows[2*counter+1] = counter
            cols[2*counter+1] = j
            counter += 1
    
    mixtures = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_mixtures, n_profiles)).tocsr()
    
    # get mixture profiles
    _mixtures = mixtures.copy()
    math.row_scale(_mixtures,1/_mixtures.sum(axis=1).A.flatten())
    mixture_profiles = math.gemmT(_mixtures, average_profiles)
    
    return mixture_profiles, mixtures

def generate_random_mixture_profiles(
    average_profiles,
    n_levels=10,
    profiles_per_mixture=4,
):

    """\
    Creates random mixture profiles with some structure to have a deterministic
    minimum of coverage of annotation fractions per profile.
    
    Parameters
    ----------
    average_profiles
        A 2d :class:`~numpy.ndarray` with features in the rows and profiles in
        the columns
    n_levels
        The number of mixture levels: Let N be the number of profiles. Then the
        first N mixtures will be pure profiles, the next N mixtures will have
        (n_levels-1)/n_levels weight of the 'main' profiles, the next N
        mixtures will have (n_levels-2)/n_levels, ...
    profiles_per_mixture
        How many profiles should contribute to a mixture. This parameter has to
        be at least 2 and cannot be larger than the number of profiles. If it
        is larger than 2, then the type assignment is done recursively using
        until `profiles_per_mixture-1` types contribute and the rest is given
        to the `profiles_per_mixture`th type.
        
    Returns
    -------
    A tuple consisting of a :class:`~numpy.ndarray` containing the mixture\
    profiles and a sparse matrix giving the mixtures.
        
    """
    
    if not isinstance(average_profiles, np.ndarray):
        raise ValueError('`average_profiles` has to be a numpy array!')
    n_genes, n_profiles = average_profiles.shape
    if n_levels < 1:
        raise ValueError(f'`n_levels` {n_levels} is smaller than 1!')
    n_mixtures = n_levels * n_profiles
    if profiles_per_mixture < 2:
        raise ValueError(f'`profiles_per_mixture` {profiles_per_mixture} is smaller than 2!')
    if profiles_per_mixture > n_profiles:
        raise ValueError(f'`profiles_per_mixture` {profiles_per_mixture} is larger than the number of profiles {n_profiles}!')
    
    # generate sparsity and weight pattern for mixture matrix
    rg = Generator(PCG64(42))#seed=_seed))
    rows = np.array([np.arange(n_mixtures) for _ in range(profiles_per_mixture)]).T.flatten()
    # this cols construction guarantees that every annotation gets its high combination + some random number of others
    cols = np.array([rg.choice(n_profiles-1,size=profiles_per_mixture-1,replace=False) for _ in range(n_mixtures)])
    col0 = np.arange(n_mixtures)
    cols += col0[:,None] + 1
    cols = np.concatenate([col0[:,None],cols],axis=1)
    cols = np.mod(cols, n_profiles).flatten()
    data = rg.random(size=(n_mixtures,profiles_per_mixture-1),dtype=average_profiles.dtype)
    #if n_levels > 1:
    #    points = 0.5*(1+np.polynomial.chebyshev.chebpts1(n_levels-1))
    for i in range(n_levels):
    #    val = 1 if i == 0 else points[i-1]
        val = 1 - i/n_levels
        data[(n_profiles*i):(n_profiles*(i+1)),0] = val
    data = np.concatenate([data,np.full((n_mixtures,1),1/n_mixtures,dtype=average_profiles.dtype)],axis=1)
    remainder = data[:,-1]
    for i in range(profiles_per_mixture - 1):
        data[:,i] *= remainder
        remainder -= data[:,i]
    data = data.flatten()
    mixtures = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_mixtures, n_profiles)).tocsr()
    
    # get mixture profiles
    _mixtures = mixtures.copy()
    math.row_scale(_mixtures,1/_mixtures.sum(axis=1).A.flatten())
    mixture_profiles = math.gemmT(_mixtures, average_profiles)
    
    return mixture_profiles, mixtures

def pure_profile_split(type_key, reference, min_reads=0, split_type_key=None):
    from ..tools._split import split_observations, merge_observations
    if type_key not in reference.obsm:
        return type_key, reference
    
    cell_type = reference.obsm[type_key]
    
    type_profiles_guess = get_average_profiles(type_key, reference, normalize=False)
    type_profiles_guess = pd.DataFrame(type_profiles_guess, index=reference.var.index, columns=cell_type.columns)
    
    sdata = split_observations(reference, cell_type, type_profiles_guess)
    if split_type_key is None:
        split_type_key = 'split_' + type_key
    mdata = merge_observations(sdata, type_profiles_guess.columns, merge_annotation_name=split_type_key, bead_annotation_name='bc', min_reads=min_reads)

    return split_type_key, mdata

def flat_inverse_mapping(
    condensed_mapping,
):
    """\
    Given a condensed mapping construct a flat inverse mapping.
    
    Parameters
    ----------
    condensed_mapping
        A mapping from coarse to a list-like of (or a single) fine categories
        as :class:`~dict`, :class:`~pandas.Series`, etc. or anything else with
        a `.items` method.
        
    Returns
    -------
    Returns a :class:`~pandas.Series` containing a mapping from fine to coarse\
    categories.
    
    """
    
    inv_mapping = {}
    for key,values in condensed_mapping.items():
        if not pd.api.types.is_list_like(values):
            values = [values]
        for val in values:
            if val in inv_mapping:
                raise ValueError(f'There are multiple mappings given for value "{val}"!')
            inv_mapping[val] = key
    return pd.Series(inv_mapping)

def merge_annotation(
    adata,
    annotation_key,
    mapping,
    result_key=None,
):
    """\
    Merges annotation into coarser groups, e.g. from subtypes to types.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs` or `.obsm`. Can
        also be a :class:`~pandas.DataFrame`, which is then treated like the
        `.obs` of an :class:`~anndata.AnnData`.
    annotation_key
        The `.obs` or `.obsm` key where the annotation and profiles are stored
        in `adata`. If the key is found in `.obsm` and `.obs`, the `.obs` key
        is used.
    mapping
        A mapping from coarse to a list-like of (or a single) fine annotation
        categories as :class:`~dict`, :class:`~pandas.Series`, etc. or anything
        else with a `.items` method. The fine annotation categories which are
        not given coarse annotation categories are left unchanged.
    result_key
        The where to store the resulting annotation. If the `annotation_key` is
        a `.obs` key, the result will be stored as `.obs`, if `annotation_key`
        is a `.obsm` key, it will be stored as `.obsm`. If  `None`, do not
        write to `adata` and return the annotation instead.
        
    Returns
    -------
    Depending on `result_key`, either returns the original `adata` with\
    annotation written in the corresponding `.obs` or `.obsm` key, or just the\
    annotation as a new :class:`~pandas.Series` or  :class:`~pandas.DataFrame`.
    
    """
    
    inv_mapping = flat_inverse_mapping(mapping)
    
    if isinstance(adata, pd.DataFrame):
        
        if annotation_key not in adata:
        
            raise ValueError(f'The annotation_key "{annotation_key}" was not found in the columns of the dataframe!')
        
        for val in adata[annotation_key].unique():
            if val not in inv_mapping:
                inv_mapping[val] = val
        
        result = adata[annotation_key].map(inv_mapping)
        if result_key is not None:
            adata[result_key] = result
            if hasattr(adata[annotation_key], 'cat'):
                adata[result_key] = adata[result_key].astype('category')
            result = adata
            
    elif annotation_key in adata.obs:
        
        for val in adata.obs[annotation_key].unique():
            if val not in inv_mapping:
                inv_mapping[val] = val
        
        result = adata.obs[annotation_key].map(inv_mapping)
        if result_key is not None:
            adata.obs[result_key] = result
            if hasattr(adata.obs[annotation_key], 'cat'):
                adata.obs[result_key] = adata.obs[result_key].astype('category')
            result = adata
    
    elif annotation_key in adata.obsm:
        
        result = pd.DataFrame(index=adata.obsm[annotation_key].index)
        
        for anno in adata.obsm[annotation_key].columns:
            if anno in inv_mapping:
                mapped = inv_mapping[anno]
            else:
                mapped = anno
            if mapped in result.columns:
                result[mapped] += adata.obsm[annotation_key][anno]
            else:
                result[mapped] = adata.obsm[annotation_key][anno]
        
        if result_key is not None:
            adata.obsm[result_key] = result
            result = adata
    
    else:
        
        raise ValueError(f'The annotation_key "{annotation_key}" was neither found in .obs nor in .obsm!')
    
    return result

def merge_colors(
    colors,
    mapping,
):
    """\
    Merges a dict-like of colors into coarser colors.
    
    Parameters
    ----------
    colors
        A dict-like with numeric colors as values.
    mapping
        A mapping from coarse to a list-like of (or a single) fine annotation
        categories as :class:`~dict`, :class:`~pandas.Series`, etc. or anything
        else with a `.items` method. The fine annotation categories which are
        not given coarse annotation categories are left unchanged.
        
    Returns
    -------
    Returns the merged colors as a :class:`~pandas.Series`.
    
    """
    
    inv_mapping = flat_inverse_mapping(mapping)
    
    import matplotlib.colors
    rgbas = {}
    for k, vs in colors.items():
        rgbas[k] = np.array(matplotlib.colors.to_rgba(vs))
    
    result = {}
    for anno in rgbas:
        if anno in inv_mapping:
            mapped = inv_mapping[anno]
        else:
            mapped = anno
        if mapped in result:
            result[mapped].append(rgbas[anno])
        else:
            result[mapped] = [rgbas[anno]]
    
    for mapped in result:
        result[mapped] = np.mean(result[mapped], axis=0)
    
    return pd.Series(result)

def write_adata_x_var_obs(adata, filename, compression='gzip'):
    adata = ad.AnnData(adata.X, var=adata.var, obs=adata.obs)
    adata.obs = adata.obs.copy()
    adata.var = adata.var.copy()
    if scipy.sparse.issparse(adata.X): # fix edge case as in https://github.com/theislab/anndata2ri/issues/18
        if not adata.X.has_sorted_indices:
            adata.X.sort_indices()
    adata.write(filename, compression='gzip')

def complete_choice(
    a,
    size,
    seed=42,
):

    """\
    Similar to :func:`~numpy.random.choice` with `replace=False`, but with
    special behaviour for `size>a.shape[0]`. The input array is shuffled and
    concatenated as often as necessary to give `size` choices. This makes the
    multiple choices appear as equally as possible.
    
    Parameters
    ----------
    a
        A :class:`~numpy.ndarray`, or something
        with a `.to_numpy` method or behaving reasonably under
        :func:`~numpy.asarray` with at least one dimension.
    size
        The number of choices without replacement. Can be larger than the `a`.
    seed
        Random seed or random generator
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the choices.
        
    """
    
    if isinstance(seed, Generator):
        rg = seed
    else:
        rg = Generator(PCG64(seed=seed))
    if hasattr(a,'to_numpy'):
        a = a.to_numpy()
    else:
        a = np.asarray(a)
    a = a.copy()
    selection = []
    while size > 0:
        rg.shuffle(a)
        selection.append(a[:size])
        size = size - len(selection[-1])
    return np.concatenate(selection)

@njit(cache=True)
def _modes(cols, n_cat, seed):
    np.random.seed(seed)
    modes = np.zeros(cols.shape[1],dtype=cols.dtype)
    for j in range(cols.shape[1]):
        counts = np.zeros(n_cat,dtype=np.int64)
        for i in range(cols.shape[0]):
            cat = cols[i,j]
            if cat >= 0:
                counts[cat] += 1
        mode_values = [cols[0,0]] ; mode_values.pop()
        mode_count = 1
        for cat in np.arange(n_cat,dtype=cols.dtype):
            if counts[cat] == mode_count:
                mode_values.append(cat)
            elif counts[cat] > mode_count:
                mode_count = counts[cat]
                mode_values = [cat]
        if len(mode_values) == 0:
            modes[j] = -1
        elif len(mode_values) == 1:
            modes[j] = mode_values[0]
        else:
            modes[j] = np.random.choice(np.array(mode_values))
    return modes
    
def mode(
    df,
    seed=42,
):
    """\
    Calculates the most frequent value per row in a dataframe. If multiple
    values are the most frequent ones, take one of them at random.
    
    Parameters
    ----------
    df
        A :class:`~pandas.DataFrame` with the values to get the modes from
    seed
        The random seed.
        
    Returns
    -------
    Returns a :class:`~pandas.Series` with the modes.
    
    """

    cattype = pd.CategoricalDtype(sorted(list({ u for c in df for u in df[c].unique() if u == u }))) # filter out nans for the categories
    cols = np.array([ df[c].astype(cattype).cat.codes.to_numpy() for c in df ])
    codes = _modes(cols, len(cattype.categories), seed)
    modes = pd.Series(index=df.index, dtype=type(cattype.categories[0]))
    modes.iloc[codes>=0] = cattype.categories[codes[codes>=0]]
    if hasattr(df.iloc[:,0], 'cat'):
        modes = modes.astype(cattype)
    return modes

@njit(cache=True)
def _heapsort3(arr, co1, co2):
    
    # adapted from scipy
    
    n = len(arr)

    l = n//2
    while l > 0:
        tmp = arr[l-1];
        tm1 = co1[l-1];
        tm2 = co2[l-1];
        i = l
        j = l*2
        while j <= n:
            if j < n and arr[j-1] < arr[j]:
                j += 1;
                
            if tmp < arr[j-1]:
                arr[i-1] = arr[j-1];
                co1[i-1] = co1[j-1];
                co2[i-1] = co2[j-1];
                i = j;
                j += j;
            else:
                break;
            
        arr[i-1] = tmp;
        co1[i-1] = tm1;
        co2[i-1] = tm2;
        
        l-=1

    while n > 1:
        tmp = arr[n-1];
        tm1 = co1[n-1];
        tm2 = co2[n-1];
        arr[n-1] = arr[0];
        co1[n-1] = co1[0];
        co2[n-1] = co2[0];
        n -= 1;
        i = 1
        j = 2
        while j <= n:
            if j < n and arr[j-1] < arr[j]:
                j+=1;
                
            if tmp < arr[j-1]:
                arr[i-1] = arr[j-1];
                co1[i-1] = co1[j-1];
                co2[i-1] = co2[j-1];
                i = j;
                j += j;
            else:
                break;
            
        arr[i-1] = tmp;
        co1[i-1] = tm1;
        co2[i-1] = tm2;

def heapsort3(
    arr,
    co1,
    co2,
):
    """\
    Sorts an array in-place while following the reordering of the elements in
    two other arrays.
    
    Parameters
    ----------
    arr
        A 1d :class:`~nump.ndarray` with the values to sort
    co1, co2
        Two 1d :class:`~nump.ndarray`s of the same length as `arr` with should
        be reordered along with `arr`.
        
    Returns
    -------
    `None`, this is an in-place operation.
    
    """
    
    if len(arr.shape) != 1:
        raise ValueError(f'`arr` has to be a 1d array, but has shape {arr.shape}!')
    if len(co1) != len(arr):
        raise ValueError(f'`co1` has to be of the same length as `arr` ({len(arr)}) but is {len(co1)} long!')
    if len(co2) != len(arr):
        raise ValueError(f'`co2` has to be of the same length as `arr` ({len(arr)}) but is {len(co2)} long!')
    
    _heapsort3(arr, co1, co2)

@njit(cache=True)
def _coo_tocsr(n_row, Ai, Aj, Ax):
    
    nnz = len(Ai)
    Bp = np.empty(n_row+1, dtype=Ai.dtype)
    
    # ensure that all the rows are ordered
    _heapsort3(Ai, Aj, Ax)
    
    # get the start and end indices per row
    row = 0
    ptr = 0
    Bp[row] = ptr
    for n in range(nnz):
        while Ai[n] != row:
            row += 1
            Bp[row] = n
    while row < n_row:
        row += 1
        Bp[row] = nnz
    
    return Bp

def coo_tocsr_inplace(
    A,
):
    """\
    Converts a sparse matrix in coo format into a sparse matrix in csr format
    without allocating huge temporaries by reusing the memory of the input
    coo matrix. This is however slower than the out-of-place scipy version.
    For an alternative using buffering on the harddisc, see
    :func:`~coo_tocsr_buffered`.
    
    Parameters
    ----------
    A
        A :class:`~scipy.sparse.coo_matrix`. As side effect the matrix gets
        index sorted, without changing the integrity of the matrix.
        
    Returns
    -------
    Returns a :class:`~scipy.sparse.csr_matrix` which shares memory with the\
    input coo matrix.
    
    """
    
    n_row = A.shape[0]
    row = A.row
    col = A.col
    dat = A.data
    
    ptr = _coo_tocsr(n_row, row, col, dat)
    
    B = scipy.sparse.csr_matrix(A.shape)
    B.indptr = ptr
    B.indices = col
    B.data = dat
    
    return B

@njit(cache=True)
def _coo_tocsr_1(n_row, Ai, Aj, Ax):
    nnz = len(Ai)
    
    # compute number of non-zero entries per row of A 
    Bp = np.zeros(n_row+1, dtype=Ai.dtype)
    for n in range(nnz):
        Bp[Ai[n]] += 1

    # cumsum the nnz per row to get Bp
    cumsum = 0
    for i in range(n_row):     
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_row] = nnz; 
    
    return Bp

@njit(cache=True)
def _coo_tocsr_2(Bp, Ai, Aj, Ax, Bj, Bx, n):
    n_row = len(Bp)-1
    
    # write Aj,Ax into Bj,Bx
    for n in range(n):
        row  = Ai[n]
        dest = Bp[row]

        Bj[dest] = Aj[n]
        Bx[dest] = Ax[n]

        Bp[row] += 1

    return Bj,Bx

@njit(cache=True)
def _coo_tocsr_3(Bp):
    n_row = len(Bp)-1
    last = 0
    for i in range(n_row+1):
        temp = Bp[i]
        Bp[i]  = last
        last   = temp

def coo_tocsr_buffered(
    A,
    blocksize=1000000,
    buffer_directory=None,
):
    """\
    Converts a sparse matrix in coo format into a sparse matrix in csr format
    consuming less memory than working in-memory by using hard disc buffer.
    This is slower than the out-of-place in-memory scipy version, but faster
    than :func:`~coo_tocsr_inplace`.
    
    Parameters
    ----------
    A
        A :class:`~scipy.sparse.coo_matrix`. The memory of this matrix is
        reused in the construction of the csr matrix, so it is effectively
        destroyed.
    blocksize
        The number of items to read per hard disc access. This has some effect
        on performance, but usually the default value is fine.
    buffer_directory
        A directory with files containing `A.col` and `A.data` dumped to files
        named "col.bin" and "data.bin" by their `.tofile()` method. This avoids
        an additional write operation if the dumped data happens to exist
        already.
        
    Returns
    -------
    Returns a :class:`~scipy.sparse.csr_matrix` which reuses the memory of the\
    input coo matrix, thereby destroying it.
    
    """
    Bp = _coo_tocsr_1(A.shape[0],A.row,A.col,A.data)
    nnz = len(A.row)

    if buffer_directory is None:
        tempdir = tempfile.TemporaryDirectory(prefix='temp_buffered_coo_tocsr_',dir='.')
        buffer_directory = tempdir.name + '/'
    
        with open(buffer_directory+'col.bin', 'ab') as f:
            A.col.tofile(f)
        with open(buffer_directory+'data.bin', 'ab') as f:
            A.data.tofile(f)
    else:
        tempdir = None

    Bj = A.col
    Bx = A.data

    for n0 in range(0,nnz,blocksize):
        n1 = min(nnz, n0+blocksize)

        A_col = np.fromfile(buffer_directory+'col.bin', dtype=A.col.dtype, count=n1-n0, offset=n0*A.col.itemsize)
        A_data = np.fromfile(buffer_directory+'data.bin', dtype=A.data.dtype, count=n1-n0, offset=n0*A.data.itemsize)

        _coo_tocsr_2(Bp,A.row[n0:],A_col,A_data, Bj, Bx, n1-n0)
    
    if tempdir is not None:
        tempdir.cleanup()
    _coo_tocsr_3(Bp)

    # now Bp,Bj,Bx form a CSR representation (with possible duplicates)
    
    B = scipy.sparse.csr_matrix(A.shape)
    B.indptr = Bp
    B.indices = Bj
    B.data = Bx
    
    return B

def spatial_split(
    adata,
    position_key=('x','y'),
    position_split=2,
    sample_key=None,
    min_obs=0,
    result_key=None,
):
    """\
    Splits a dataset into spatial patches balancing the number of observations
    per split.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obs` (and `.obsm`).
        Can also be a :class:`~pandas.DataFrame` which is then used in place of
        `.obs`.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates.
    position_split
        The number of splits per spatial dimension. Can be a tuple with the
        spatial dimension as length to assign a different split per dimension.
        If `None`, no position splits are performed. See also `min_obs`.
    sample_key
        The `.obs` key with categorical sample information: every sample is
        split separately. Can also be a :class:`~pandas.Series` containing the
        sample information. If `None`, assume a single sample.
    min_obs
        This limits the number of `position_split` by stopping the split if the
        split would decrease the number of observations below this threshold.
    result_key
        The `.obs` key to write the split sample annotation to. If `None`,
        returns the split sample annotation as :class:`~pandas.Series`.
        
    Returns
    -------
    Depending on `result_key` returns either a :class:`~pandas.Series`\
    containing the split sample annotation or the inpt `adata` with the split\
    sample annotation written to `adata.obs[result_key]`.
    
    """
    
    if isinstance(adata, sc.AnnData):
        adata_obs = adata.obs
    else:
        adata_obs = adata
    
    if sample_key is None:
        sample_column = pd.Series(np.full(shape=adata_obs.shape[0],fill_value=''),index=adata_obs.index)
    elif isinstance(sample_key, pd.Series):
        sample_column = sample_key.reindex(index=adata_obs.index)
    elif sample_key in adata_obs:
        if not hasattr(adata_obs[sample_key], 'cat'):
            raise ValueError(f'`adata.obs[sample_key]` has to be categorical, but `adata.obs["{sample_key}"]` is not!')
        sample_column = adata_obs[sample_key]
    else:
        raise ValueError(f'The `sample_key` argument is {sample_key!r} but has to be either a key of `adata.obs["{sample_key}"]`, a `pandas.Series` or `None`!')
    
    positions = get.positions(adata, position_key)
    
    # divide spatial samples spatially into subsamples: keeps all the correlation structure
    ndim = positions.shape[1]

    position_split = np.array(position_split)
    if len(position_split.shape) == 0:
        position_split = np.array([position_split]*ndim)

    sample_column = sample_column.astype(str)
    for idim,ps in enumerate(position_split):
        new_sample_column = sample_column.copy()
        for sample, sub in positions.iloc[:,idim].groupby(sample_column):
            max_ps = ps if min_obs < 1 else min(int((len(sub)/min_obs)**(1/(ndim-idim))), ps) # split only if enough observations are available
            position_cuts = pd.qcut(sub, max_ps, duplicates='drop')
            new_sample_column.loc[sub.index] = new_sample_column.loc[sub.index] + '|' + position_cuts.astype(str)
        sample_column = new_sample_column
    
    result = sample_column
    if result_key is not None:
        adata_obs[result_key] = result
        result = adata
    
    return result

def get_maximum_annotation(adata, obsm_key, result_key=None):
    """\
    Turns a soft annotation into a categorical annotation by reporting the
    annotation categories with the maximum values per oberservation.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with annotation in `.obsm`.
        Can also be a :class:`~pandas.DataFrame` which is then used in place of
        `.obsm[obsm_key]`.
    obsm_key
        The `.obsm` key or array-like of `.obs` keys with the soft annotation.
        If `adata` is a :class:`~pandas.DataFrame`, this is ignored.
    result_key
        The `.obs` key (or `adata` key if `adata` is a
        :class:`~pandas.DataFrame`) to write the categorical annotation to. If
        `None`, returns the categorical annotation as :class:`~pandas.Series`.
        
    Returns
    -------
    Depending on `result_key` returns either a :class:`~pandas.Series`\
    containing the categorical annotation or the inpt `adata` with the split\
    sample annotation written to `adata.obs[result_key]` (or\
    `adata[result_key]` if `adata` is a :class:`~pandas.DataFrame`).
    
    """
    
    if isinstance(adata, pd.DataFrame):
        obsm = adata
        adata_obs = adata
    else:
        obsm = get.positions(adata, obsm_key)
        adata_obs = adata.obs
    
    result = obsm.columns[obsm.to_numpy().argmax(axis=1)]
    
    result = pd.Series(result, index=obsm.index)
    
    if not hasattr(result, 'cat'):
        result = result.astype(pd.CategoricalDtype(obsm.columns))
    
    if result_key is not None:
        adata_obs[result_key] = result
        result = adata
    
    return result

def AnnData_query(self, expr, **kw_args):
    """\
    Query the columns of `.obs` of an :class:`~anndata.AnnData` with a boolean
    expression and use the result to subset the :class:`~anndata.AnnData`.
    This provides a convenient shorthand notation for subsetting: E.g.
    `adata.query('A>4 & B==@value')` effectively expands to
    `adata[(adata.obs['A']>4) & (adata.obs['B']==value)]`.
    This is analogous to the functionality of :meth:`pandas.DataFrame.query`
    and implemented by :meth:`pandas.DataFrame.eval`, so similar restrictions
    apply.
    This function is implemented as a monkey patch and should be called as a
    method of an :class:`~anndata.AnnData` instance.
    
    Parameters
    ----------
    expr
        The query on `.obs` columns to subset an :class:`~anndata.AnnData`.
        Column names and literals are referenced directly, variables in python
        scope by a preceeding "@".
    **kw_args
        Additional keyword arguments are forwarded to the call to
        :meth:`pandas.DataFrame.eval`.
        
    Returns
    -------
    Returns a view of the subsetted :class:`~anndata.AnnData` instance.
    
    """
    
    selection = self.obs.eval(expr,**kw_args)
    
    if not pd.api.types.is_bool_dtype(selection):
        raise ValueError(f'Evaluating the expression {expr!r} did not result to something of boolean arry type, so it cannot be used for subsetting!')
        
    return self[selection]

sc.AnnData.query = AnnData_query

def anndata2R_header():
    """\
    Returns a string to be used as part of an R script in order to be able to
    natively read basic information (X, var, obs) from h5ad files in R.
    
    Parameters
    ----------
        
    Returns
    -------
    Returns the anndata2R code as a string.
    
    """
    
    return """\
library(data.table)

library(hdf5r)
library(Matrix) # must be loaded before SparseM to enable cast from SparseM to Matrix
library(SparseM)

read_matrix = function(file, name) {
    if (name %in% list.datasets(file,recursive=FALSE)) {
        print('dense')
        newX = t(file$open(name)$read())
    } else if ('X' %in% list.groups(file,recursive=FALSE)) {
        groupX = openGroup(file, name)
        groupX_encoding_type = h5attr(groupX,'encoding-type')
        if (groupX_encoding_type == 'csr_matrix' || groupX_encoding_type == 'csc_matrix' ) {
            ra = groupX$open('data')$read()
            ja = as.integer(groupX$open('indices')$read()+1)
            ia = as.integer(groupX$open('indptr')$read()+1)
            dim = h5attr(groupX,'shape')
            if (groupX_encoding_type == 'csr_matrix') {
                print('csr')
                newX = new("matrix.csr", ra=ra, ja=ja, ia=ia, dimension=dim)
            } else if (groupX_encoding_type == 'csc_matrix') {
                print('csc')
                newX = new("matrix.csc", ra=ra, ja=ja, ia=ia, dimension=dim)
            }
            newX = as(newX,'dgCMatrix')
        } else {
            print('unkown encoding for X...')
        }
    } else {
        print('unkown encoding for X...')
    }
    return(newX)
}
read_df = function(file, name) {
    #print(name)
    group = openGroup(file, name)
    #print('---------- group')
    #print(group)
    categories = NULL
    if (group$exists('__categories')) {
        categories = group$open('__categories')
        categories_ds = list.datasets(categories)
    }
    #print('---------- index')
    #print(h5attr(group, '_index'))
    #print('---------- group open')
    #print(group$open(h5attr(group, '_index')))
    #print('---------- group open read')
    #print(group$open(h5attr(group, '_index'))$read())
    #print('---------- df')
    df = data.frame(row.names=group$open(h5attr(group, '_index'))$read())
    if (length(list.datasets(group)) > 1) { # catch case with only the index and no additional column
        for (col in h5attr(group, 'column-order')) {
            col_item = group$open(col)
            if (col_item$get_obj_type() == 'H5I_GROUP') { # catch anndata 0.8 behaviour for categories
                #print('---------- 0.8')
                #print('---------- col_item')
                #print(col_item)
                temp = col_item$open('codes')$read()
                temp = col_item$open('categories')$read()[temp+1]
            } else {
                #print('---------- legacy')
                temp = col_item$read()
                if (!is.null(categories) && col %in% categories_ds) {
                    temp = categories$open(col)$read()[temp+1]
                }
            }
            df[col] = temp
        }
    }
    return(df)
}
read_adata = function(filename, transpose=FALSE) {
    file = H5File$new(filename, mode = "r")
    newX = read_matrix(file, 'X')
    obs_df = read_df(file, 'obs')
    var_df = read_df(file, 'var')
    #print('---------- newX')
    #print(newX)
    #print(colnames(newX))
    #print('---------- var_df')
    #print(var_df)
    #print(row.names(var_df))
    colnames(newX) = row.names(var_df)
    #print('---------- obs_df')
    #print(obs_df)
    row.names(newX) = row.names(obs_df)
    if (transpose) {
        return(list('X'=t(newX),'obs'=var_df,'var'=obs_df))
    } else {
        return(list('X'=newX,'obs'=obs_df,'var'=var_df))
    }
}
"""
