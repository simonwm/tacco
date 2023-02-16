import time

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse

from .. import get
from .. import utils
from ..utils._utils import _run_OT
from . import _qc as qc
from ..utils._utils import _infer_annotation_key

def _normalize_reference(adata, annotation_key=None):
    ''' Normalize profiles per annotation and annotation per cell - if available. '''
    
    annotation_key = _infer_annotation_key(adata, annotation_key)
    
    # normalize annotation weights to 1 per cell
    if annotation_key in adata.obsm:
        adata.obsm[annotation_key] /= np.array(adata.obsm[annotation_key].sum(axis=1))[:,None]
    # normalize profiles to 1 per annotation
    if annotation_key in adata.varm:
        adata.varm[annotation_key] /= np.array(adata.varm[annotation_key].sum(axis=0))

def construct_reference_profiles(
    adata,
    annotation_key=None,
    counts_location=None,
    inplace=True,
    normalize=True,
    target_sum=None,
    trafo=None,
    ):

    """\
    Constructs reference profiles from categorical annotations.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        a categorical annotation in `.obs`.
    annotation_key
        The `.obs` key where the annotation is stored; this will also be used
        as a `.varm` key to store the resulting annotation profiles.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    inplace
        Whether to update the input :class:`~anndata.AnnData` or return a copy.
    normalize
        Whether to normalize the reference annotation and profiles.
    target_sum
        The number of counts to normalize every observation to before computing
        profiles. If `None`, no normalization is performed. 
    trafo
        What transformation to apply to the expression before computing the
        profiles. After the computation, the transformation is inverted to get
        profiles in the same units as the expression. Possible values are:
        
        - "log1p": log(x+1)
        - "sqrt": sqrt(x)
        - `None`: no transformation
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the constructed profiles\
    depending on `inplace` either as copy or as a reference to the original\
    `adata`.
    
    """

    if adata is None:
        raise ValueError('"adata" cannot be None!')
        
    annotation_key = _infer_annotation_key(adata, annotation_key)
        
    reference = get.counts(adata, counts_location=counts_location, annotation=annotation_key, copy=False)
    #qc.check_counts_validity(reference.X)
    
    if annotation_key not in reference.obs:
        raise ValueError(f'The key {annotation_key!r} is not available in .obs!')
    
    if target_sum is not None or trafo is not None:
        reference = reference.copy() # we do not want to change the original
        if target_sum is not None:
            sc.pp.normalize_total(reference, target_sum=target_sum)
        if trafo == 'log1p':
            utils.log1p(reference)
        elif trafo == 'sqrt':
            utils.sqrt(reference)
    
    dums = pd.get_dummies(reference.obs[annotation_key],dtype=reference.X.dtype)
    ncats = dums.sum(axis=0)
    dums /= ncats.to_numpy()
    profiles = reference.X.T @ dums.to_numpy()
    profiles = pd.DataFrame(profiles, index=reference.var.index, columns=dums.columns)
    
    # invert the transformation for the profiles
    if trafo == 'log1p':
        profiles = np.expm1(profiles)
    elif trafo == 'sqrt':
        profiles *= profiles
    
    if not inplace:
        adata = adata.copy()

    adata.varm[annotation_key] = profiles.reindex(index=adata.var.index)
    
    if normalize:
        _normalize_reference(adata, annotation_key)

    return adata

def refine_reference(
    adata,
    annotation_key=None,
    counts_location=None,
    inplace=False,
    normalize=True,
    regularization=1e-3,
    ):

    """\
    Refines a reference data set by scaling profiles and
    annotation to match the expression data. Specifically, determines the
    normalization factors n(cg) in the read model
    p(cga) = n(cg) p(g|a) p(a|c) for the joint probability distribution of
    cells c, genes g, and annotation a per read to p(cg) from the expression
    data, and updates the profiles p(g|a) and the annotation p(a|c) from the
    marginals of p(cga).
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are and/or will be stored.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    inplace
        Whether to modify the input :class:`~anndata.AnnData` or return a copy.
    normalize
        Whether to normalize the reference annotation and profiles.
    regularization
        Relative factor to determine a regularization addition to the profiles
        to avoid unsolvable count distributions (e.g. for some (g,c):
        sum_a p(g|a) * p(a|c) = 0, but p(cg) != 0). If set to 0, no
        regularization is done.
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the refined reference,\
    depending on `copy` either as copy or as a reference to the original\
    `adata`.
    
    """

    if adata is None:
        raise ValueError('"adata" cannot be None!')
        
    annotation_key = _infer_annotation_key(adata, annotation_key)
    
    counts = get.counts(adata, counts_location=counts_location, annotation=False, copy=False)
    qc.check_counts_validity(counts.X)
    
    if annotation_key not in adata.obsm and annotation_key not in adata.obs:
        raise ValueError('The key "%s" is neither available in .obsm nor .obs!' % annotation_key)
    if annotation_key not in adata.varm:
        raise ValueError('The key "%s" is not available in .varm!' % annotation_key)

    if not inplace:
        adata = adata.copy()
    
    _normalize_reference(adata, annotation_key)
    
    if annotation_key in adata.obs:
        adata.obsm[annotation_key] = pd.get_dummies(adata.obs[annotation_key])
        del adata.obs[annotation_key]
        
    if regularization != 0:
        adata.varm[annotation_key] += 1 / len(adata.var.index) * regularization
        adata.obsm[annotation_key] += 1 / len(adata.obs.index) * regularization
        _normalize_reference(adata, annotation_key)
    
    p_g_a = adata.varm[annotation_key]
    p_a_c = adata.obsm[annotation_key].T
    p_g_a = p_g_a.to_numpy()
    p_a_c = p_a_c.to_numpy()
    
    #p(cga) = n(cg) p(g|a) p(a|c)
    #p(cg) = sum_a p(cga) = n(cg) sum_a p(g|a) p(a|c)

    #n(cg) = p(cg) / sum_a p(g|a) p(a|c)
    if issparse(counts.X):
        p_cg = counts.X.tocoo()
        if p_cg is counts.X:
            p_cg = p_cg.copy()
        p_cg.data *= 1/p_cg.data.sum() # normalize as joint probability
        temp_data = np.empty_like(p_cg.data)
        utils.sparse_result_gemmT(p_g_a, p_a_c.T, p_cg.col, p_cg.row, temp_data)
        utils.divide(p_cg.data,temp_data,out=p_cg.data) # p_cg now contains n_cg
    else:
        p_cg = counts.X.copy()
        p_cg *= 1/p_cg.sum(axis=None) # normalize as joint probability
        temp = p_g_a@p_a_c
        p_cg /= temp # p_cg now contains n_cg
        
    #p'(ga) = sum_c p(cga) = p(g|a) sum_c n(cg) p(a|c)
    adata.varm[annotation_key] = pd.DataFrame(p_g_a * (p_a_c@p_cg).T, index=adata.var.index, columns=adata.varm[annotation_key].columns)
    #p'(ac) = sum_g p(cga) = p(a|c) sum_g p(g|a) n(cg)
    adata.obsm[annotation_key] = pd.DataFrame(p_a_c.T * (p_cg@p_g_a), index=adata.obs.index, columns=adata.varm[annotation_key].columns)
    
    if normalize:
        _normalize_reference(adata, annotation_key)

    return adata
