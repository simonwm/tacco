import numpy as np
import pandas as pd
import anndata as ad

from .. import get
from .. import preprocessing
from .. import utils
from . import _helper as helper

def _annotate_nnls(
    adata,
    reference,
    annotation_key=None,
    ):

    """\
    Implements the functionality of :func:`~annotate_nnls` without data
    integrity checks.
    """
    
    average_profiles = utils.get_average_profiles(annotation_key, reference)
    average_profiles /= average_profiles.sum(axis=0).to_numpy()
    
    types = average_profiles.columns
    
    cell_type = utils.parallel_nnls(average_profiles.to_numpy(), adata.X)
    cell_type = pd.DataFrame(cell_type, columns=types, index=adata.obs.index)
    
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_nnls(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by NNLS
    (non-negative least squares).
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tc.pp.create_reference` for options to create it.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are stored in the `reference`. If `None`, it is inferred from
        `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tc.get.counts`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=False)
    
    # call typing without data integrity checks
    cell_type = _annotate_nnls(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
    )
    
    return cell_type

