import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse
import scipy.optimize
from numba import njit,prange

from numpy.random import Generator, PCG64

from .. import get
from . _reference import construct_reference_profiles
from .. import utils
from ..utils._split import split_beads
from ..utils._utils import _infer_annotation_key, _get_unique_keys

def subsample_annotation(
    adata,
    annotation_key=None,
    modification=None,
    range_factor=1,
    seed=42,
):

    """\
    Subsamples the observations to change the annotation fractions.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing categorical annotation to
        subsample.
    annotation_key
        The `.obs` key with the categorical annotation. Is determined
        automatically if possible.
    modification
        A dict-like mapping the annotation categories to a float giving the
        fraction of this category to be kept. Unlisted categories are
        unchanged. Instead of the full name of a category, also unambiguous
        shortcuts are possible, i.e. the first few letters.
    range_factor
        A float giving a factor to scale down all categories a priori. A
        factor larger than `1` (e.g. `2`) reduces all category observation and
        makes values larger than `1` in `modification` possible without
        duplicated observations.
    seed
        The random seed to use
        
    Returns
    -------
    Returns a :class:`~pandas.Series` with the random recaling factors.
    
    """
    
    annotation_key = _infer_annotation_key(adata, annotation_key)
    
    if annotation_key not in adata.obs.columns:
        raise Exception('Only categorical `.obs` annotation columns are supported for subsampling!')

    if modification is None:
        modification = {}
    params = pd.DataFrame({'modification':pd.Series(modification)})
    
    counts = pd.DataFrame({'all':adata.obs[annotation_key].value_counts()})
    
    #print(pd.Series({s:[ p for p in params.index if s.startswith(p)] for s in counts.index}))
    params['map'] = [[ s for s in counts.index if str(s) == str(p)] for p in params.index]
    params['map_short'] = [[ s for s in counts.index if str(s).startswith(str(p))] for p in params.index]
    params['map'] = [ m if len(m) == 1 else s for m, s in zip(params['map'],params['map_short'])]
    if (params['map'].map(len) > 1).any():
        params['error'] = params['map'].map(len) > 1
        raise Exception('There is a not unique mapping of shortcuts to labels! Here is what has been mapped:\n%s' % params)
    if (params['map'].map(len) == 0).any():
        params['error'] = params['map'].map(len) == 0
        raise Exception('There is are shortcuts without mapping labels! Here is what has been mapped:\n%s' % params)
    if len(params['map']) > 0:
        params['map'] = params['map'].str[0]
    
    counts['modification'] = params.set_index('map')['modification']
    counts['modification'] = counts['modification'].fillna(1.0)
    
    counts['new'] = ((counts['all'] * counts['modification']) / range_factor).astype(int)
    
    selection = np.concatenate([
        utils.complete_choice(df.index, counts.loc[l,'new'], seed=seed)
        for l,df in adata.obs.groupby(annotation_key)
    ])
    
    return adata[selection]

def _scale_genes(adata, rescaling_factors, gene_keys=None, counts_location=None, round=False):
    counts = get.counts(adata, counts_location=counts_location)
    try:
        rescaling_factors = rescaling_factors.reindex(counts.var.index).fillna(0).to_numpy() # if we get a pd.Series, then reorder it accordingly
    except:
        pass
    utils.scale_counts(adata, rescaling_factors, counts_location=counts_location, round=round)
    if isinstance(gene_keys, bool):
        if gene_keys:
            for gene_key in adata.varm:
                try: # all varms
                    adata.varm[gene_key] *= rescaling_factors.astype(adata.varm[gene_key].iloc[0].dtype)[:,None]
                except:
                    pass
            for gene_key in adata.var:
                try: # all vars
                    adata.var[gene_key] *= rescaling_factors.astype(adata.var[gene_key].iloc[0].dtype)[:,None]
                except:
                    pass
    elif gene_keys is not None:
        if isinstance(gene_keys, str):
            gene_keys = [gene_keys]
        for gene_key in gene_keys:
            if gene_key in adata.varm:
                adata.varm[gene_key] *= rescaling_factors.astype(adata.varm[gene_key].iloc[0].dtype)[:,None]
        for gene_key in gene_keys:
            if gene_key in adata.var.columns:
                adata.var[gene_key] *= rescaling_factors.astype(adata.var[gene_key].dtype)[:,None]

def _get_random_platform_factors(adata, platform_log10_mean=0, platform_log10_std=0, seed=42):
    rg = Generator(PCG64(seed=seed))
    rescaling_factors = np.power(10,rg.laplace(loc=platform_log10_mean,scale=platform_log10_std/np.sqrt(2),size=len(adata.var.index)))
    return pd.Series(rescaling_factors, index=adata.var.index)

def apply_random_platform_effect(
    adata,
    platform_log10_mean=0,
    platform_log10_std=0.6,
    seed=42,
    round=True,
    counts_location=None,
    gene_keys=None,
):

    """\
    Applies a random log-laplace distributed rescaling per gene inplace.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the expression and/or profiles
        to rescale inplace.
    platform_log10_mean
        log10 mean of the Laplace distribution
    platform_log10_std
        log10 of the standard deviation of the Laplace distribution
    seed
        The random seed to use
    round
        Whether to round the resulting expression matrix to integer counts
        after rescaling
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    gene_keys
        String or list of strings specifying additional count-like `.var` and
        `.varm` annotations to scale along with the platform effect. If `True`,
        take all `.var` and `.varm` keys.
        
    Returns
    -------
    Returns a :class:`~pandas.Series` with the random recaling factors.
    
    """
    
    rescaling_factors = _get_random_platform_factors(adata, platform_log10_mean=platform_log10_mean, platform_log10_std=platform_log10_std, seed=seed)
    
    _scale_genes(adata, rescaling_factors.to_numpy(), gene_keys=gene_keys, counts_location=counts_location, round=round)
    
    if round and scipy.sparse.issparse(adata.X):
        adata.X.eliminate_zeros()
    
    return rescaling_factors

@njit(parallel=True, fastmath=True, cache=True)
def _platform_objective_fdf(x, profilesA, profilesB, reg_a, reg_b):
    n,m = profilesA.shape
    
    a = x[:m]
    b = x[m:]
    
    der = np.zeros(m+n)
    dera = der[:m]
    derb = der[m:]
    loss = 0.0
    for i in prange(n):
        _tempB = 0.0
        for j in range(m):
            delta = b[i] * profilesA[i,j] * a[j] - profilesB[i,j]
            loss += delta**2
            dera[j] += 2 * delta * b[i] * profilesA[i,j]
            derb[i] += 2 * delta * profilesA[i,j] * a[j]
    if reg_a > 0:
        reg_a = reg_a * n # make the terms comparable
        for j in range(m):
            loss += reg_a * (a[j]-1)**2
            dera[j] += reg_a * 2 * (a[j]-1)
    if reg_b > 0:
        reg_b = reg_b * m # make the terms comparable
        for i in range(n):
            loss += reg_b * (b[i]-1)**2
            derb[i] += reg_b * 2 * (b[i]-1)
    
    return loss, der

def get_platform_normalization_factors(
    PA,
    PB,
    reg_a=0.0,
    reg_b=0.0,
    tol=1e-12,
):
    """\
    Find platform normalization factors `a_g` per gene and `b_p` per profiles
    for scaling A to conform to B by solving `PB_pg = b_p PA_pg a_g` in least
    squares sense with `PA` and `PB` containing counts per profile and gene.
    The profiles can be e.g. celltypes or mixtures with defined celltype
    composition. `a_g` and `b_p` can be interpreted as relative capture
    efficiencies between two methods per gene and per profile (e.g. per cell
    type). This assumes that the underlying cell type frequencies are identical
    between the two sets of profiles.
    
    Parameters
    ----------
    PA
        A 2d :class:`~numpy.ndarray` containing counts with profiles in the
        first dimension and genes in the second.
    PB
        A 2d :class:`~numpy.ndarray` with identical dimensions as as `PA`.
    reg_a
        Weight of the regularization term `(a_g-1)^2` to put cost on deviations
        from 1 for the per gene rescaling factors
    reg_b
        Weight of the regularization term `(b_p-1)^2` to put cost on deviations
        from 1 for the per profiles rescaling factors
    tol
        Solver tolerance.
        
    Returns
    -------
    A :class:`~numpy.ndarray` containing the results.
        
    """
    
    if PA.shape != PB.shape:
        raise ValueError(f'PA.shape != PB.shape: {PA.shape} != {PB.shape}')
    
    if not isinstance(PA, np.ndarray):
        raise ValueError(f'PA has to be a numpy ndarray, but is a {PA.__class__}!')
    if not isinstance(PB, np.ndarray):
        raise ValueError(f'PB has to be a numpy ndarray, but is a {PB.__class__}!')
    
    # scipy only works with doubles and the numba kernel only works with homogeneous types
    if PA.dtype != np.float64:
        PA = PA.astype(np.float64)
    if PB.dtype != np.float64:
        PB = PB.astype(np.float64)
    
    PA = PA.copy()
    PB = PB.copy()
    
    n,m = PA.shape
    
    guess = np.ones(m+n)
    
    guess[:m] = PB.sum(axis=0) / PA.sum(axis=0)
    guess[m:] = 1
    result = scipy.optimize.minimize(_platform_objective_fdf, guess, args=(PA, PB, reg_a, reg_b), method='L-BFGS-B', jac=True, tol=tol, bounds=[(0,np.inf)]*len(guess)).x
    # fix gauge degree of freedom
    factor = result[m:].mean()
    result[m:] /= factor
    result[:m] *= factor
    
    return result[:m], result[m:]

def normalize_platform(
    adata,
    reference,
    annotation_key=None,
    reference_annotation_key=None,
    counts_location=None,
    gene_keys=None,
    inplace=True,
    return_rescaling_factors=False,
    reg_a=0.0,
    reg_b=0.0,
    tol=1e-12,
    verbose=1,
    ):

    """\
    Normalize the expression data and/or profiles in an
    :class:`~anndata.AnnData` to conform with the expression in another
    dataset. In effect rescales all genes.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing expression, annotation,
        and/or profiles. This data will be rescaled.
    reference
        Another :class:`~anndata.AnnData` containing expression, annotation,
        and/or profiles to be used as reference.
    annotation_key
        The `.obs`/`.obsm`/`.varm` annotation key for `adata`, or alternatively
        annotation fractions for `adata`, which have to correspond to the
        profiles given in `reference.varm[reference_annotation_key]`. If
        `None`, no annotation information is used and the overall expression
        determines the rescaling.
    reference_annotation_key
        The `.obs`/`.obsm`/`.varm` annotation key for `reference`, or
        alternatively annotation fractions for `reference`, which have to
        correspond to the profiles given in `adata.varm[annotation_key]`. If
        `None`, no annotation information is used and the overall expression
        determines the rescaling.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    gene_keys
        String or list of strings specifying additional count-like `.var` and
        `.varm` annotations to scale along with the platform normalization. The
        `annotation_key` is included automatically. Note that the any
        normalization is generally destroyed for these keys and they have to be
        renormalized if necessary. If `True`, take all `.var` and `.varm` keys.
    inplace
        Whether to update the input :class:`~anndata.AnnData` or return a copy.
    return_rescaling_factors
        Whether to return the rescaling factors instead of the rescaled data.
    reg_a
        Weight of per gene rescaling factor regularization
    reg_b
        Weight of per profile rescaling factor regularization
    tol
        Solver tolerance.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the rescaled `adata`,\
    depending on `inplace` either as copy or as a reference to the original\
    `adata`. If `return_rescaling_factors`, no rescalng is performed and the\
    factors to do so are returned instead.
    
    """
    
    if not inplace:
        adata = adata.copy()
    
    if isinstance(gene_keys, bool) and gene_keys:
        pass # gene_keys == True means take all .var and .varm keys
    else:
        gene_keys = _get_unique_keys(annotation_key, gene_keys)
    
    got_adata = annotation_key in adata.obs or annotation_key in adata.obsm or annotation_key in adata.varm
    got_other = reference_annotation_key in reference.obs or reference_annotation_key in reference.obsm or reference_annotation_key in reference.varm

    if got_adata and got_other:

        profilesA = None
        profilesB = None

        if annotation_key in adata.obs: # construct counts-weighted profiles from expression and annotation
            #print('.obs')

            dums = pd.get_dummies(adata.obs[annotation_key],dtype=adata.X.dtype)
            profilesA = adata.X.T @ dums.to_numpy()

            profilesA = pd.DataFrame(profilesA, index=adata.var.index, columns=dums.columns)

        if profilesA is None and annotation_key in adata.varm: # reconstruct counts-weighted profiles from buffered (normalized) profiles
            #print('.varm')
            
            weights = utils.parallel_nnls(adata.varm[annotation_key].to_numpy(),(utils.get_sum(adata.X,axis=0)[None,:]))
            profilesA = adata.varm[annotation_key].to_numpy()
            utils.col_scale(profilesA, weights.flatten())

            profilesA = pd.DataFrame(profilesA, index=adata.var.index, columns=adata.varm[annotation_key].columns)

        if reference_annotation_key in reference.obs: # construct counts-weighted profiles from expression and annotation
            #print('other .obs')
            
            dums = pd.get_dummies(reference.obs[reference_annotation_key],dtype=reference.X.dtype)
            profilesB = reference.X.T @ dums.to_numpy()

            profilesB = pd.DataFrame(profilesB, index=reference.var.index, columns=dums.columns)

        if profilesB is None and reference_annotation_key in reference.varm: # reconstruct counts-weighted profiles from buffered (normalized) profiles
            #print('other .varm')
            
            weights = utils.parallel_nnls(reference.varm[reference_annotation_key].to_numpy(),(utils.get_sum(reference.X,axis=0)[None,:]))
            profilesB = reference.varm[reference_annotation_key].to_numpy()
            utils.col_scale(profilesB, weights.flatten())

            profilesB = pd.DataFrame(profilesB, index=reference.var.index, columns=reference.varm[reference_annotation_key].columns)

        if profilesA is None and annotation_key in adata.obsm:

            if profilesB is not None: # reconstruct counts-weighted profiles from bead splitting with reference profiles
                #print('.obsm1')
                
                profilesA = split_beads(adata, bead_type_map=adata.obsm[annotation_key], type_profiles=profilesB, scaling_jobs=None, return_split_profiles=True)

            elif reference_annotation_key in reference.varm: # reconstruct counts-weighted profiles from bead splitting with buffered reference profiles
                #print('.obsm2')
                
                profilesA = split_beads(adata, bead_type_map=adata.obsm[annotation_key], type_profiles=reference.varm[reference_annotation_key], scaling_jobs=None, return_split_profiles=True)

        if profilesB is None and reference_annotation_key in reference.obsm:

            if profilesA is not None: # reconstruct counts-weighted profiles from bead splitting with reference profiles
                #print('other .obsm1')
                
                profilesB = split_beads(reference, bead_type_map=reference.obsm[reference_annotation_key], type_profiles=profilesA, scaling_jobs=None, return_split_profiles=True)

            elif annotation_key in adata.varm: # reconstruct counts-weighted profiles from bead splitting with buffered reference profiles
                #print('other .obsm2')
                
                profilesB = split_beads(reference, bead_type_map=reference.obsm[reference_annotation_key], type_profiles=adata.varm[annotation_key], scaling_jobs=None, return_split_profiles=True)

        if profilesA is None or profilesB is None:
            raise ValueError(f'The annotation information is not sufficient to run platform normalization with annotation! You can specify explicitly not to use annotation by setting `annotation_key` to `None`.')

    else:

        #print('.X')
        profilesA = utils.get_sum(adata.X, axis=0)[:,None]
        profilesB = utils.get_sum(reference.X, axis=0)[:,None]
    
    if isinstance(profilesA, pd.DataFrame) and isinstance(profilesB, pd.DataFrame):
        # make sure that both are ordered identically
        profilesA = profilesA.reindex_like(profilesB)
    if isinstance(profilesA, pd.DataFrame):
        profilesA = profilesA.to_numpy()
    if isinstance(profilesB, pd.DataFrame):
        profilesB = profilesB.to_numpy()
    
    gene_rescaling_factor = get_platform_normalization_factors(profilesA.T, profilesB.T, reg_a=reg_a, reg_b=reg_b, tol=tol)[0]
    if verbose > 0:
        print('mean,std( rescaling(gene) ) ', gene_rescaling_factor.mean(), gene_rescaling_factor.std())
    if return_rescaling_factors:
        return gene_rescaling_factor
    _scale_genes(adata, gene_rescaling_factor, gene_keys=gene_keys, counts_location=counts_location)
    
    return adata
