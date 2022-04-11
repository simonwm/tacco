import scanpy as sc
import numpy as np
import pandas as pd

from .. import get
from .. import preprocessing
from . import _helper as helper
from .. import utils
import scipy.sparse
import scipy.linalg

try: # dont fail importing the whole module, just because a single annotation method is not available
    import wot
    HAVE_WOT = True
except ImportError:
    HAVE_WOT = False

def _annotate_wot(
    adata,
    reference,
    annotation_key,
    **kwargs
    ):

    """\
    Implements the functionality of :func:`~annotate_wot` without data
    integrity checks.
    """
    
    if not HAVE_WOT:
        raise ImportError('The module `wot` could not be imported, but is required to use the annotate method "wot"! Maybe it is not installed properly?')

    nadata = adata.shape[0]
    nref = reference.shape[0]

    adata_ref = sc.concat((adata, reference))
    adata_ref.obs['data'] = [0] * nadata + [1] * nref

    try:
        ot_model = wot.ot.OTModel(adata_ref, day_field='data', **kwargs)
        tmap_annotated = ot_model.compute_transport_map(0, 1)
    except:
        print('Exception in WaddingtonOT. Trying with cell normalization.')
        sc.pp.normalize_per_cell(adata_ref, counts_per_cell_after=10000)
        ot_model = wot.ot.OTModel(adata_ref, day_field='data', **kwargs)
        tmap_annotated = ot_model.compute_transport_map(0, 1)

    ref_cell_type = pd.get_dummies(reference.obs[annotation_key])
    cell_type = (tmap_annotated.X @ ref_cell_type)
    cell_type.index = adata.obs.index
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_wot(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    **kwargs
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by
    Waddington-OT [Schiebinger19]_.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`
        and annotation in `.obs`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tc.pp.create_reference` for options to create it.
    annotation_key
        The `.obs` key where the annotation and profiles are stored in the
        `reference`. If `None`, it is inferred from `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tc.get.counts`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    if adata is None:
        raise ValueError('"adata" cannot be None!')
    if adata.X is None:
        raise ValueError('"adata.X" cannot be None!')
    if reference is None:
        raise ValueError('"reference" cannot be None!')
        
    annotation_key = preprocessing.infer_annotation_key(reference, annotation_key)
    
    adata = get.counts(adata, counts_location=counts_location, annotation=True, copy=False)
    reference = get.counts(reference, counts_location=counts_location, annotation=annotation_key, copy=False)
    
#    if annotation_key in reference.varm:
#        reference = preprocessing.filter_profiles(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only genes in the profiles
    if annotation_key in reference.obsm:
        reference = preprocessing.filter_annotation(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only cells in the annotation

    tdata,reference = preprocessing.filter(adata=(adata, reference)) # ensure consistent gene selection
    
    # call typing without data integrity checks
    cell_type = _annotate_wot(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        **kwargs
    )
    
    return cell_type
