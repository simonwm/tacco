import numpy as np
import pandas as pd
import anndata as ad
from sklearn import svm
from sklearn import multioutput
from .. import get
from .. import preprocessing
from .. import utils
from . import _helper as helper

def _annotate_svm(
    adata,
    reference,
    annotation_key=None,
    mode='classification',
    trafo=None,
    seed=42,
    **kwargs,
    ):

    """\
    Implements the functionality of :func:`~annotate_svm` without data
    integrity checks.
    """
    ref_X = reference.X
    tes_X = adata.X
    if trafo is not None:
        ref_X = ref_X.copy()
        tes_X = tes_X.copy()
        if trafo == 'sqrt':
            utils.sqrt(ref_X)
            utils.sqrt(tes_X)
        elif trafo == 'log1p':
            utils.log1p(ref_X)
            utils.log1p(tes_X)
        else:
            raise ValueError(f'annotate_svm got unkown `trafo=={trafo!r}`!')
    
    if mode == 'classification':
        # train
        clf = svm.LinearSVC(random_state=seed,**kwargs)
        if annotation_key in reference.obs:
            clf.fit(ref_X, reference.obs[annotation_key].to_numpy())
            types = reference.obs[annotation_key].unique()
        else:
            raise ValueError(f'annotate_svm with `mode=="classification" can only use annotation in `.obs` in the reference, but {annotation_key!r} is not found there!')
        # test
        pred = clf.predict(tes_X)
        pred = pd.Series(pred, index=adata.obs_names)
        cell_type = pd.get_dummies(pred)
    elif mode == 'regression':
        # train
        regr = multioutput.MultiOutputRegressor(svm.LinearSVR(random_state=seed,**kwargs))
        if annotation_key in reference.obs:
            y = pd.get_dummies(reference.obs[annotation_key])
            types = reference.obs[annotation_key].unique()
        elif annotation_key in reference.obsm:
            y = reference.obsm[annotation_key]
            types = reference.obsm[annotation_key].columns
        else:
            raise ValueError(f'annotate_svm with `mode=="regression" can only use annotation in `.obs` and `.obsm` in the reference, but {annotation_key!r} is not found there!')
        regr.fit(ref_X, y.to_numpy())
        # test
        pred = regr.predict(tes_X)
        pred = pd.DataFrame(pred, index=adata.obs_names, columns=y.columns)
        cell_type = pd.get_dummies(pred)
    else:
        raise ValueError(f'`mode` can only be "classification" or "regression"!')
    
    cell_type = helper.normalize_result_format(cell_type, types=types)
    
    return cell_type

def annotate_svm(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    mode='classification',
    trafo=None,
    seed=42,
    **kwargs,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by an SVM
    [Abdelaal19]_.

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
    mode
        Selects what svm should be used. Possible values are "classification"
        to use :class:`~sklearn.svm.LinearSVC` and "regression" to use
        :class:`~sklearn.svm.LinearSVR`.
    trafo
        Selects a transformation for the data before putting it into the SVM.
        Available are:
        
        - 'sqrt': Use the squareroot of `.X`; equivalent to using probability
          amplitudes, i.e. Bhattacharyya projections
        - 'log1p': Use log1p-transformed data
        - None: Dont transform the data
    seed
        Random seed
    **kwargs
        Additional keyword arguments are forwarded to the
        :class:`~sklearn.svm.LinearSVC` or :class:`~sklearn.svm.LinearSVR`
        constructor depending on the value of `mode`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=False)
    
    # call typing without data integrity checks
    cell_type = _annotate_svm(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        mode=mode,
        trafo=trafo,
        seed=seed,
        **kwargs,
    )
    
    return cell_type

