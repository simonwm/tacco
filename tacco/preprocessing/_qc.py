import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import issparse

from .. import get
from .. import utils
from ..utils._utils import _infer_annotation_key

def check_counts_validity(
    X,
    delta=1e-4,
    head=1000,
    raise_exception=True,
):
    """\
    Checks whether matrix looks like it is a unnormalized uncentered count
    matrix
    
    Parameters
    ----------
    X
        Dense or sparse counts matrix.
    delta
        The maximum absolute deviation from integer numbers to tolerate.
    head
        The number of sored values of the count matrix to check for dviation
        from integer numbers.
    raise_exception
        Whether to raise an exception for invalid counts or just silently
        return `False`.
        
    Returns
    -------
    Returns whether the count matrix is valid, except when it is not and\
    `raise_exception` is `True`.
    
    """
    

    if X is None:
        if raise_exception:
            raise ValueError('The counts cannot be None!')
        else:
            return False
        
    if issparse(X):
        vals = X.data
    else:
        vals = X.flatten()
    
    if head is not None:
        vals = vals[:head]
    
    if (vals < 0).any():
        if raise_exception:
            raise ValueError('Some of the counts are negative! Provide counts which are non-negative integer counts.')
        else:
            return False
    

    if (0.5 - np.abs(np.mod(vals, 1) - 0.5) > delta).any():
        if raise_exception:
            raise ValueError('Some of the counts dont look like integers! Provide counts which are non-negative integer counts.')
        else:
            return False

    return True

def filter(
    adata,
    min_counts_per_gene=None,
    min_counts_per_cell=None,
    min_cells_per_gene=None,
    min_genes_per_cell=None,
    remove_constant_genes=False,
    remove_zero_cells=False,
    assume_valid_counts=False,
    return_view=True,
    ):

    """\
    Filter one or more :class:`~anndata.AnnData` to satisfy simple quality
    criteria. In contrast to :func:`scanpy.pp.filter_cells` and
    :func:`scanpy.pp.filter_genes` this function iterates the filters until
    convergence.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the expression to filter. It
        can also be an iterable of :class:`~anndata.AnnData` to apply the
        filter to all instances and use the intersection of genes for all.
    min_counts_per_gene
        The minimum count (per :class:`~anndata.AnnData`) genes must have to be
        kept.
    min_counts_per_cell
        The minimum count (per :class:`~anndata.AnnData`) cells must have to be
        kept.
    min_cells_per_gene
        The minimum number of cells (per :class:`~anndata.AnnData`) genes must
        have to be kept.
    min_genes_per_cell
        The minimum number of genes (per :class:`~anndata.AnnData`) cells must
        have to be kept.
    remove_constant_genes
        Whether to remove genes which do not show any variation between cells
    remove_zero_cells
        Whether to remove cells without non-zero genes
    assume_valid_counts
        Disable checking for invalid counts (e.g. non-integer or negative).
    return_view
        Instead of :class:`~anndata.AnnData` instances, return filtered views
        into the original `adata`. If nothing is filtered or permuted, return
        the original `adata`.
        
    Returns
    -------
    Returns the filtered `adata`.
    
    """
    
    adatas = adata
    if isinstance(adatas, ad.AnnData):
        adatas = [adatas]
    else:
        adatas = list(adatas)
    
    if not assume_valid_counts and (min_counts_per_gene is not None or min_counts_per_cell is not None):
        # only check for valid counts if the user is requesting filtering for counts
        for i in range(len(adatas)):
            if np.prod(adatas[i].shape) > 0:
                check_counts_validity(adatas[i].X)
    
    changed = True
    counter = 0
    while changed: # iterate filtering until no changes occur
        counter += 1
        changed = False

        good_genes = adata[0].var.index
        for i in range(len(adatas)):
            if (min_counts_per_gene is None and min_cells_per_gene is None and not remove_constant_genes) or np.prod(adatas[i].shape) == 0:
                good_genes = good_genes.intersection(adatas[i].var.index)
                continue
            if min_counts_per_gene is not None:
                good_genes = good_genes.intersection(adatas[i].var.index[sc.pp.filter_genes(adatas[i], min_counts=min_counts_per_gene, inplace=False)[0]])
            if min_cells_per_gene is not None:
                good_genes = good_genes.intersection(adatas[i].var.index[sc.pp.filter_genes(adatas[i], min_cells=min_cells_per_gene, inplace=False)[0]])
            if remove_constant_genes:
                if len(adatas[i].obs) == 1:
                    print('WARNING: Encountered a dataset with only a single observation while `remove_constant_genes` is enabled! As this would remove all genes, `remove_constant_genes` is disabled for this dataset.')
                else:
                    not_constant = adatas[i].X.max(axis=0)!=adatas[i].X.min(axis=0)
                    if issparse(not_constant):
                        not_constant = not_constant.A
                    not_constant = not_constant.flatten()
                    good_genes = good_genes.intersection(adatas[i].var.index[not_constant])
        
        for i in range(len(adatas)):
            if len(adatas[i].var.index) != len(good_genes): # filter happened
                adatas[i] = adatas[i][:,good_genes]
                changed = True
            elif (adatas[i].var.index != good_genes).any(): # reordering happened: no side effects on cell filtering
                adatas[i] = adatas[i][:,good_genes]
            
        for i in range(len(adatas)):
            if np.prod(adatas[i].shape) == 0:
                continue
            cell_mask = np.full(len(adatas[i].obs.index),True)
            if remove_zero_cells:
                cell_mask &= utils.get_sum(abs(adatas[i].X),axis=1) != 0
            if min_counts_per_cell is not None:
                cell_mask &= sc.pp.filter_cells(adatas[i], min_counts=min_counts_per_cell, inplace=False)[0]
            if min_genes_per_cell is not None:
                cell_mask &= sc.pp.filter_cells(adatas[i], min_genes=min_genes_per_cell, inplace=False)[0]
            
            if not cell_mask.all():
                adatas[i] = adatas[i][cell_mask]
                changed = True
    
    if not return_view:
        adatas = [adata.copy() for adata in adatas] # realize the views or copy the originals
        
    if len(adatas) == 1:
        return adatas[0]
    else:
        return adatas

def filter_reference_genes(
    adata,
    annotation_key=None,
    min_log2foldchange=1.0,
    min_expression=1e-5,
    remove_mito=False,
    n_hvg=None,
    return_mask=False,
    return_view=True,
    ):

    """\
    Filter :class:`~anndata.AnnData` to include only genes which pass a set of
    quality and relevance criteria.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing profiles in `.varm` to filter
        on; to use `n_hvg` also the bare counts have to be provided in `.X`.
    annotation_key
        The `.varm` key where the annotation is stored.
    min_log2foldchange
        Minimum log2-fold change a gene must have in at least one annotation
        category relative to the mean of the other categories to be kept. Only
        selects upregulated genes.
    min_expression
        Minimum expression level relative to all expression a gene must have in
        at least one annotation category to be kept.
    remove_mito
        Whether to remove genes starting with "mt-" and "MT-".
    n_hvg
        The number of highly variable genes to run on. This expects bare counts
        in `.X`. If `None`, this criterion is switched off.
    return_mask
        Instead of a (view of a) :class:`~anndata.AnnData`, return the boolen
        filter mask.
    return_view
        Instead of a :class:`~anndata.AnnData`, return a filtered view into the
        original `adata`. If nothing is filtered out, return the original
        `adata`.
        
    Returns
    -------
    Depending on `return_mask` and `return_view` returns a filter mask, a\
    filtered view, or a filtered copy of `adata`.
    
    """
    
    annotation_key = _infer_annotation_key(adata, annotation_key)
    
    if annotation_key not in adata.varm:
        raise ValueError('There are no profiles found in `.varm[%s]`!' % annotation_key)
    
    profiles = adata.varm[annotation_key]
    
    profiles = profiles / profiles.sum(axis=0).to_numpy()
    profiles = profiles.to_numpy()
    
    profiles += 1e-30 # regularization for numerical reasons
    
    good_genes = np.full(profiles.shape[0],True)
    
    if min_log2foldchange is not None and min_log2foldchange > 0:
        fold_genes = np.full(profiles.shape[0],False)
        meaner = (np.ones((profiles.shape[1],profiles.shape[1])) - np.identity(profiles.shape[1])) / (profiles.shape[1] - 1)
        for t in range(profiles.shape[1]):
            fold_genes |= (np.log(profiles[:,t] / (profiles @ meaner[t]))) >= (np.log(2) * min_log2foldchange)
        
        good_genes &= fold_genes

        print(f'min_fold: good gene fraction {good_genes.sum()/len(good_genes)} good count fraction {adata[:,good_genes].X.sum()/adata.X.sum()}')
    
    if min_expression is not None and min_expression > 0:
        good_genes &= (profiles >= min_expression).any(axis=1)

        print(f'min_exp: good gene fraction {good_genes.sum()/len(good_genes)} good count fraction {adata[:,good_genes].X.sum()/adata.X.sum()}')
    
    if remove_mito:
        good_genes &= ~np.array([ gene.startswith('mt-') or gene.startswith('MT-') for gene in adata.var.index ])

        print(f'mito: good gene fraction {good_genes.sum()/len(good_genes)} good count fraction {adata[:,good_genes].X.sum()/adata.X.sum()}')
    
    if n_hvg is not None:
        
        adata_log = adata[:,good_genes].copy()
        if adata_log.shape[1] > n_hvg:
            utils.log1p(adata_log)
            sc.pp.highly_variable_genes(adata_log, n_top_genes=n_hvg)
            good_genes[good_genes] &= adata_log.var['highly_variable']

    if return_mask:
        return good_genes
        
    if not good_genes.all():
        adata = adata[:,good_genes]
    
    if return_view:
        return adata
    else:
        return adata.copy()

def filter_reference(
    adata,
    annotation_key=None,
    fill_na=True,
    fill_negative=None,
    return_mask=False,
    return_view=True,
    mode=None,
    ):

    """\
    Filter :class:`~anndata.AnnData` to include only genes and cells where at
    least one `.varm` and `.obsm` annotation is neither negative nor na.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the annotation to filter on.
    annotation_key
        The `.varm` and/or `.obsm` key where the annotation is stored.
    fill_na
        Fills na in the annotation with 0. This is done inplace, so the original
        `adata` might be changed. If `None` and na values are discovered,
        raises a ValueError.
    fill_negative
        Fills negative values in the annotation with 0. This is done inplace,
        so the original `adata` might be changed. If `None` and negative values
        are discovered, raises a ValueError.
    return_mask
        Instead of a (view of a) :class:`~anndata.AnnData`, return the boolen
        filter mask.
    return_view
        Instead of a :class:`~anndata.AnnData`, return a filtered view into the
        original `adata`. If nothing is filtered out, return the original
        `adata`.
    mode
        String to indicate whether to restrict filtering on .varm ("profiles")
        or on .obsm ("annotation"). If `None`, do both.
        
    Returns
    -------
    Depending on `return_mask` and `return_view` returns a filter mask, a\
    filtered view, or a filtered copy of `adata`. If `return_mask` and `mode`\
    is `None`, returns a tuple (cell_mask,gene_mask).
    
    """
    
    annotation_key = _infer_annotation_key(adata, annotation_key)
    
    def _filter_anno(anno_loc):

        adata_anno = getattr(adata,anno_loc)
        
        if mode is None:
            if annotation_key not in adata_anno:
                raise ValueError('The annotation_key "%s" was not found in .%s!' % (annotation_key, anno_loc))
                
        annotation = adata_anno[annotation_key]
        filled = annotation
        
        def _fill_mask(annotation, filled, mask, inplace, anno_loc, val_name):
            if np.array(mask).any():
                if inplace is None:
                    raise ValueError('The anotation in .%s["%s"] contains %s values! Specify `fill_%s` to avoid this Exception.' % (anno_loc,annotation_key,val_name,val_name))
                elif inplace:
                    annotation[mask] = 0
                else:
                    if annotation is filled: # filled is not a separate entity
                        filled = filled.copy()
                    filled[mask] = 0
            return filled

        filled = _fill_mask(annotation=annotation, filled=filled, mask=filled.isna(), inplace=fill_na, anno_loc=anno_loc, val_name='na')

        filled = _fill_mask(annotation=annotation, filled=filled, mask=(filled < 0), inplace=fill_negative, anno_loc=anno_loc, val_name='negative')

        mask = filled.sum(axis=1) > 0
        
        return mask
        
    if mode is None:
        if annotation_key not in adata.varm or annotation_key not in adata.obsm:
            raise ValueError(f'If `mode` is `None`, the `annotation_key` {annotation_key!r} must be available in both `adata.varm` and `adata.obsm`!')
            
        gene_mask = _filter_anno('varm')
        cell_mask = _filter_anno('obsm')
        
        if return_mask:
            return cell_mask,gene_mask

        if not gene_mask.all():
            adata = adata[:,gene_mask]
        if not cell_mask.all():
            adata = adata[cell_mask]
            
    elif mode == 'profiles':
        if annotation_key not in adata.varm:
            raise ValueError(f'If `mode` is "profiles", the `annotation_key` {annotation_key!r} must be available in `adata.varm`!')
            
        gene_mask = _filter_anno('varm')
        
        if return_mask:
            return gene_mask

        if not gene_mask.all():
            adata = adata[:,gene_mask]
            
    elif mode == 'annotation':
        if annotation_key not in adata.obsm:
            raise ValueError(f'If `mode` is "annotation", the `annotation_key` {annotation_key!r} must be available in `adata.obsm`!')
            
        cell_mask = _filter_anno('obsm')
        
        if return_mask:
            return cell_mask

        if not cell_mask.all():
            adata = adata[cell_mask]
            
    else:
        raise ValueError('`mode` can only be "profiles", "annotation", or `None`!')

    if return_view:
        return adata
    else:
        return adata.copy()

def filter_profiles(
    adata,
    annotation_key=None,
    **kw_args,
    ):

    """\
    Filter :class:`~anndata.AnnData` to include only genes where at least one
    profile is neither negative nor na.
    Identical to :func:`~tacco.preprocessing.filter_reference` with `mode='profiles'`.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the annotation to filter on.
    annotation_key
        The `.varm` key where the annotation is stored.
    **kw_args
        Additional keyword arguments forwarded to
        :func:`~tacco.preprocessing.filter_reference(..., mode='profiles')`.
        
    Returns
    -------
    Depending on `return_mask` and `return_view` returns a filter mask, a\
    filtered view, or a filtered copy of `adata`.
    
    """
    
    return filter_reference(adata, annotation_key, mode='profiles', **kw_args)

def filter_annotation(
    adata,
    annotation_key=None,
    **kw_args,
    ):

    """\
    Filter :class:`~anndata.AnnData` to include only cells where at least one
    annotation is neither negative nor na.
    Identical to :func:`~tacco.preprocessing.filter_reference` with `mode='annotation'`.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the annotation to filter on.
    annotation_key
        The `.obsm` key where the annotation is stored.
    **kw_args
        Additional keyword arguments forwarded to
        :func:`~tacco.preprocessing.filter_reference(..., mode='annotation')`.
        
    Returns
    -------
    Depending on `return_mask` and `return_view` returns a filter mask, a\
    filtered view, or a filtered copy of `adata`.
    
    """
    
    return filter_reference(adata, annotation_key, mode='annotation', **kw_args)
