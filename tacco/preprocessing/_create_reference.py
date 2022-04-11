import numpy as np
import pandas as pd
import anndata as ad

from .. import get
from .. import preprocessing

def get_profiles_from_marker_genes(marker_genes):
    ''' Convert marker gene lists into (binary) reference profiles. '''
    gene_list = {g for k,v in marker_genes.items() for g in v}
    profiles = pd.DataFrame(index=gene_list)
    for profile, genes in marker_genes.items():
        profiles[profile] = 0
        profiles.loc[genes,profile] = 1
    return profiles

def infer_annotation_key(adata, annotation_key=None):
    ''' Return annotation key, if supplied or unique. '''
    if annotation_key is not None:
        return annotation_key
    if adata is None:
        raise ValueError('adata cannot be none it no annotation key is specified!')
    
    keys = set(adata.varm.keys()) | set(adata.obsm.keys()) | set(adata.obs.columns)
    if len(keys) == 1:
        return next(iter(keys))
    elif len(keys) > 1:
        raise ValueError('adata has more than a unique annotation key, so it has to be specified explicitly!')
    else:
        raise ValueError('adata has no annotation keys!')
        
def get_unique_keys(annotation_key, other_keys):
    ''' Get unique keys. '''
    if other_keys is None:
        return annotation_key
    elif isinstance(other_keys, str):
        return {annotation_key, other_keys}
    else:
        return {annotation_key, *other_keys}

def normalize_reference(adata, annotation_key=None):
    ''' Normalize profiles per annotation and annotation per cell - if available. '''
    
    annotation_key = infer_annotation_key(adata, annotation_key)
    
    # normalize annotation weights to 1 per cell
    if annotation_key in adata.obsm:
        adata.obsm[annotation_key] /= np.array(adata.obsm[annotation_key].sum(axis=1))[:,None]
    # normalize profiles to 1 per annotation
    if annotation_key in adata.varm:
        adata.varm[annotation_key] /= np.array(adata.varm[annotation_key].sum(axis=0))

def create_reference(
    adata=None,
    profiles=None,
    marker=None,
    annotation_key=None,
    normalize=True,
    counts_location=None,
    keep_annotation=False,
    ):

    """\
    Create a reference data set for the transfer of annotation from various
    inputs for use in :func:`~tc.tl.transfer_annotation`.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` for creating the reference data set. It
        should contain one of each (1) expression data (e.g. in `.X`) or
        reference  expression profiles (in `.varm`) and (2) the annotation to
        transfer as categorical annotation (in `.obs`) or as weights (in
        `.obsm`). If reference expression profiles are specified but no
        annotation, this is exactly equivalent to only supplying these profiles
        via `profiles` with `adata=None`.
    profiles
        A :class:`~pandas.DataFrame` containing a reference expression profile
        per column. Can be used standalone or together with `adata`.
    marker
        A :class:`~dict` containing a list of marker genes per key. Can be used
        standalone or together with `adata`.
    annotation_key
        If `profiles` or `marker` is not `None`, this is the `.varm` key where
        the annotation i.e. expression profiles will be stored.
        If `adata` is not `None`, this is the `.obs`, `.obsm`, and/or
        `.varm` key where the annotation is and/or will be stored.
    normalize
        Whether to normalize the reference annotation and profiles.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tc.get.counts`.
        Only relevant if `adata` is provided.
    keep_annotation
        Keys of `.var`, `.varm`, `.obs`, and `.obsm` to keep in addition to the
        `annotation_key`.
        Only relevant if `adata` is provided.
        
    Returns
    -------
    Returns a minimial :class:`~anndata.AnnData` populated only with the data\
    necessary for annotation transfer (with the exception of `keep_annotation`).\
    These may be one or more of `.X`, `.varm`, `.obsm`, and `.obs`.
    
    """

    # Handle simple and corner cases

    annotation_key = infer_annotation_key(adata, annotation_key)

    if profiles is not None and marker is not None:
        raise ValueError('"profiles" and "marker" cannot be specified at the same time!')
    
    if marker is not None:
        profiles = get_profiles_from_marker_genes(marker)

    if profiles is not None and adata is None:
        genes = profiles.index.astype(str) # AnnData prefers string indices
        reference = ad.AnnData(var=pd.DataFrame(index=genes))
        reference.varm[annotation_key] = profiles.set_index(genes)
        return reference

    if adata is None:
        raise ValueError('"adata", "profiles", and "marker" were None! At least one of them has to be provided!')

    # Handle AnnData case

    keep_annotation = get_unique_keys(annotation_key, keep_annotation)
    reference = get.counts(adata, counts_location=counts_location, annotation=keep_annotation, copy=True)
    preprocessing.check_counts_validity(reference.X)

    if annotation_key in reference.obsm and annotation_key in reference.obs:
        del reference.obsm[annotation_key] # priority for the obs annotation
        print('Found the key "%s" in .obs and .obsm! Using only the one in .obs for reference creation.' % annotation_key)
    
    if annotation_key in reference.varm and profiles is not None:
        reference.varm[annotation_key] = profiles.reindex(index=reference.var.index)
        print('Found the key "%s" in .varm and got profiles (or marker genes) explicitly specified! Using only the explicitly specfied profiles.' % annotation_key)
    
    if normalize:
        normalize_reference(reference, annotation_key)

    return reference
