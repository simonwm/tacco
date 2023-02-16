import scanpy as sc
import numpy as np
import pandas as pd

from .. import get
from .. import preprocessing
from . import _helper as helper
from .. import utils
from ..utils._utils import _infer_annotation_key
import scipy.sparse
import scipy.linalg

try: # dont fail importing the whole module, just because a single annotation method is not available
    import novosparc
    HAVE_NOVOSPARC = True
except ImportError:
    HAVE_NOVOSPARC = False

def _annotate_novosparc(
    adata,
    reference,
    annotation_key,
    position_key=('x','y'),
    alpha=1.0,
    reads=False,
    ):

    """\
    Implements the functionality of :func:`~annotate_novosparc` without data
    integrity checks.
    """
    
    if not HAVE_NOVOSPARC:
        raise ImportError('The module `novosparc` could not be imported, but is required to use the annotate method "novosparc"! Maybe it is not installed properly?')
    
    atlas_matrix = adata.X
    dataset = reference
    # NovoSpaRc seems to be allergic to sparse data
    if scipy.sparse.issparse(dataset.X):
        dataset = sc.AnnData(dataset.X.A, obs=dataset.obs, var=dataset.var)
    if scipy.sparse.issparse(atlas_matrix):
        atlas_matrix = atlas_matrix.A
    locations = get.positions(adata, position_key) if alpha != 1 else np.arange(0,len(adata.obs.index))[:,None]
    
    markers_to_use = np.arange(dataset.shape[1])
    
    tissue = novosparc.cm.Tissue(dataset, locations, atlas_matrix, markers_to_use=markers_to_use) # ... and here
    
    if alpha == 1.0:
        # dont need smooth cost
        tissue.setup_linear_cost(markers_metric='minkowski')
    else:
        # monkey patch NovoSpaRc ...
        tissue.setup_linear_cost.__func__.__defaults__ = (None, None, 'minkowski', 2)
        tissue.setup_reconstruction() # ... to avoid Exception here
    
    tissue.reconstruct(alpha)

    ref_cell_type = pd.get_dummies(reference.obs[annotation_key])

    if reads:
        annotation = ref_cell_type.to_numpy().astype(float)
        utils.row_scale(annotation, np.array(reference.X.sum(axis=1)).flatten() / annotation.sum(axis=1))
        cell_type = utils.gemmT(tissue.gw.T, annotation.T)
        cell_type = pd.DataFrame(cell_type, columns=ref_cell_type.columns)
    else:
        cell_type = (tissue.gw.T @ ref_cell_type)
    
    cell_type.index = adata.obs.index
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_novosparc(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    position_key=('x','y'),
    alpha=1.0,
    reads=False,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by NovoSpaRc
    [Nitzan19]_.

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
        The `.obs` key where the annotation is stored in the `reference`. If
        `None`, it is inferred from `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. If `alpha==1`, this is not referenced.
    alpha
        The alpha parameter of NovoSpaRc.
    reads
        Whether to reduce the mapping to types using the counts per reference
        observation as weights or just flat weights per cell.
        
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
        
    annotation_key = _infer_annotation_key(reference, annotation_key)
    
    adata = get.counts(adata, counts_location=counts_location, annotation=True, copy=False)
    reference = get.counts(reference, counts_location=counts_location, annotation=annotation_key, copy=False)
    
#    if annotation_key in reference.varm:
#        reference = preprocessing.filter_profiles(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only genes in the profiles
    if annotation_key in reference.obsm:
        reference = preprocessing.filter_annotation(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only cells in the annotation

    tdata,reference = preprocessing.filter(adata=(adata, reference)) # ensure consistent gene selection
    
    # call typing without data integrity checks
    cell_type = _annotate_novosparc(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        position_key=position_key,
        alpha=alpha,
        reads=reads,
    )
    
    return cell_type
