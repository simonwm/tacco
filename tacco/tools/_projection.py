import numpy as np
import pandas as pd

from .. import get
from .. import preprocessing
from .. import utils
from . import _helper as helper
import scipy.sparse
import scipy.linalg

def _annotate_projection(
    adata,
    reference,
    annotation_key,
    projection='bc2',
    deconvolution=None,
    ):

    """\
    Implements the functionality of :func:`~annotate_projection` without data
    integrity checks.
    """
    average_profiles = utils.get_average_profiles(annotation_key, reference)
    
    if projection == 'h2':
        projection = 'bc'

    if projection == 'bc':
        if deconvolution is None:
            deconvolution = 'linear'
    if deconvolution is None:
        deconvolution = 'nnls'
    
    cell_type = utils.projection(adata.X, average_profiles.to_numpy().T, metric=projection, deconvolution=deconvolution)
        
    if projection == 'bc':
        # we are still working on amplitudes, so we have to square the result
        cell_type *= cell_type
    
    cell_type = pd.DataFrame(cell_type, index=adata.obs.index, columns=average_profiles.columns)
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_projection(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    projection='bc2',
    deconvolution=None,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by projection.

    This is the direct interface to this annotation method. In practice using
    the general wrapper :func:`~tacco.tools.annotate` is recommended due to its
    higher flexibility.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`.
    reference
        Reference data to get the annotation definition from.
    annotation_key
        The `.obs` and/or `.varm` key where the annotation and/or profiles are
        stored in the `reference`. If `None`, it is inferred from `reference`,
        if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    projection
        Projection method to use. Available are:
        
        - 'naive': annotations are the matrix product of the gene frequencies
          in given by the count matrix and those in the annotation profiles
        - 'bc': like 'naive' but using the probability amplitudes
          instead of probabilities/frequencies
        - 'h2': identical to 'bc'
        - 'bc2': like 'bc' but squares the projection
          result, i.e. forms the expectation value from the overlap.
          
    deconvolution
        Which method to use for deconvolution of the results based on the
        cross-projections of the annotation categories. If `False`, no
        deconvolution is done. If `None`, the best deconvolution is selected
        for every projection method:
        
        - 'linear': special deconvolution for `projection=='bc'` which only
          works for amplitudes, does not rely on (possibly slow) nnls and works
          with (fast) solution of a linear equation.
        - 'nnls': general deconvolution for all other methods which works on
          probabilities, and therefore needs nnls.
          
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=False)
    
    # call typing without data integrity checks
    cell_type = _annotate_projection(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        projection=projection,
        deconvolution=deconvolution,
    )
    
    return cell_type
