import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import gc
import inspect
from numpy.random import Generator, PCG64
import time
from sklearn.cluster import MiniBatchKMeans

from .. import get
from .. import preprocessing
from ..utils._utils import _infer_annotation_key, _get_unique_keys
from .. import utils
from ._points import _map_hash_annotation, distance_matrix, affinity, spectral_clustering
from ._split import split_observations
from . import _helper as helper

def check_for_argument(function, argument):
    return argument in inspect.signature(function).parameters

def annotation_method(
    method,
    **kw_args,
):
    """\
    Wrap a specified method and its arguments in a standard signature
    
    Parameters
    ----------
    method
        String selecting the method to use for annotation. Can also be a
        callable of signature
        `method(adata, reference, annotation_key, **kw_args)` returning a
        :class:`~pandas.DataFrame`.
        Possible methods include:
        
        - 'OT', implemented by :func:`~tacco.tools.annotate_OT`
        - 'nnls', implemented by :func:`~tacco.tools.annotate_nnls`
        - 'tangram', implemented by :func:`~tacco.tools.annotate_tangram`
        - 'novosparc', implemented by :func:`~tacco.tools.annotate_novosparc`
        - 'projection', implemented by :func:`~tacco.tools.annotate_projection`
        - 'RCTD', implemented by :func:`~tacco.tools.annotate_RCTD`
        - 'NMFreg', implemented by :func:`~tacco.tools.annotate_NMFreg`
        - 'SingleR', implemented by :func:`~tacco.tools.annotate_SingleR`
        - 'WOT', implemented by :func:`~tacco.tools.annotate_wot`
        - 'svm', implemented by :func:`~tacco.tools.annotate_svm`
        
    **kw_args
        Additional keyword arguments are forwarded to the annotation method.
        See the functions mentioned in the documentation of the parameter
        `method` for details.
        
    Returns
    -------
    Returns a callable of signature\
    `method(adata, reference, annotation_key, annotation_prior, verbose)`\
    which performs the actual annotation.
    
    """
    
    def _method(adata, reference, annotation_key, annotation_prior, verbose):

        nonlocal method, kw_args

        if isinstance(method, str):

            if method == 'NMFreg':
                from ._NMFreg import annotate_NMFreg
                annotate = annotate_NMFreg
            elif method == 'WOT':
                from ._wot import _annotate_wot
                annotate = _annotate_wot
            elif method == 'OT':
                from ._OT import _annotate_OT
                annotate = _annotate_OT
            elif method == 'nnls':
                from ._nnls import _annotate_nnls
                annotate = _annotate_nnls
            elif method == 'svm':
                from ._svm import _annotate_svm
                annotate = _annotate_svm
            elif method == 'RCTD':
                from ._RCTD import annotate_RCTD
                annotate = annotate_RCTD
            elif method == 'SingleR':
                from ._SingleR import annotate_SingleR
                annotate = annotate_SingleR
            elif method == 'tangram':
                from ._tangram import _annotate_tangram
                annotate = _annotate_tangram
            elif method == 'novosparc':
                from ._novosparc import _annotate_novosparc
                annotate = _annotate_novosparc
            elif method == 'projection':
                from ._projection import _annotate_projection
                annotate = _annotate_projection
            else:
                raise ValueError('`method` "%s" is not implemented.' % method)

            if check_for_argument(annotate, 'verbose'):
                verbose_arg = {'verbose':verbose}
            else:
                verbose_arg = {}

            if check_for_argument(annotate, 'annotation_prior'):

                if annotation_prior is None:
                    _annotation_prior, _ = helper.prep_priors_reference(adata, reference, annotation_key, reads=True)
                elif isinstance(annotation_prior, pd.Series):
                    _annotation_prior = annotation_prior
                else:
                    _annotation_prior = annotation_prior(adata, reference, annotation_key)

                cell_type = annotate(adata, reference, annotation_key, annotation_prior=_annotation_prior, **verbose_arg, **kw_args)

                #cell_type/cell_type.sum(axis=1)[:,None]
            else:
                cell_type = annotate(adata, reference, annotation_key, **verbose_arg, **kw_args)

        else:
            try:
                if check_for_argument(method, 'verbose'):
                    verbose_arg = {'verbose':verbose}
                else:
                    verbose_arg = {}
                
                cell_type = method(adata, reference, annotation_key, **verbose_arg, **kw_args)
            except:
                raise ValueError('The supplied `method` is neither string nor a working callable!')

        return cell_type

    return _method

def boost_annotation_method(
    annotation_method,
    bisections,
    bisection_divisor,
    ):

    """\
    Boosts an annotation method by bisectioning
    
    Parameters
    ----------
    annotation_method
        A callable of signature
        `method(adata, reference, annotation_key, annotation_prior)`
    bisections
        If larger than 0, runs a boosted stronger annotator using a basis
        annotator by iteratively running the basis annotator and removing a
        reconstructed fraction of the counts. The parameter gives the number of
        recursive bisections of the annotation.
    bisection_divisor
        Number of parts for bisection: First, `bisections` times a fraction of
        1/`bisection_divisor` of the unassigned counts of every observation is
        assigned. The remainder is then split evely in `bisection_divisor`
        parts. E.g. if `bisections` is `2` and `bisection_divisor` is `3`, then
        the assigned fractions per round of typing are 1/3,2/3*(1/3,1/3,1/3),
        if `bisections` is `3` and `bisection_divisor` is `2`, then they are
        1/2,(1/2,(1/2,1/2)/2)/2. Generally, the total number of typing rounds
        is `bisections + bisection_divisor - 1`.
        
    Returns
    -------
    Returns a callable of signature\
    `method(adata, reference, annotation_key, annotation_prior, verbose)` which\
    performs the actual boosted annotation.
    
    """
    
    def _method(adata, reference, annotation_key, annotation_prior, verbose):
        
        nonlocal annotation_method, bisections, bisection_divisor
        
        adata = ad.AnnData(adata.X.copy(), obs=adata.obs[[]], var=adata.var[[]])
        average_profiles = utils.get_average_profiles(annotation_key, reference)
        average_profiles /= average_profiles.sum(axis=0).to_numpy()
        
        if bisection_divisor < 2:
            raise ValueError('`bisection_divisor` is smaller than 2!')

        def get_bisections(bisections, bisection_divisor):
            remaining = [1.0]
            current = []
            for bs in range(bisections-1):
                current.append(remaining[-1] / bisection_divisor)
                remaining.append(remaining[-1] - current[-1])
            current.extend([remaining[-1]/bisection_divisor] * bisection_divisor)
            current = np.array(current)
            done = current.cumsum()
            remaining = 1-done
            todo = [1, *list(remaining[:-1])]

            #print(pd.DataFrame({'todo': todo, 'current':current, 'done':done, 'remaining':remaining}))

            return zip(todo, current, remaining)

        cell_prior = helper.prep_cell_priors(adata, reads=True)
        total_sums = utils.get_sum(adata.X,axis=1)

        for i, (todo, current, remaining) in enumerate(get_bisections(bisections, bisection_divisor)):

            if verbose > 0:
                print(f'bisection run on {todo}')

            cell_type = annotation_method(adata, reference, annotation_key, annotation_prior, verbose)

            #print('update result with weight', current)
            cell_type *= 1 / cell_type.sum(axis=1).to_numpy()[:, None]
            if i == 0:
                sum_cell_type = current * cell_type
                average_profiles = average_profiles[cell_type.columns].to_numpy()
            else:
                sum_cell_type += current * cell_type

            if i < bisections + bisection_divisor - 2: # dont do preparation for the next round in the last round
                #print('subtract reconstruction with weight', current)

                cell_type = cell_type.to_numpy()

                if scipy.sparse.issparse(adata.X):
                    # the part of the result where there are non-zero entries in the data is the only interesting part, as it is non-negative, will be subtracted from the data and negative values will be set to zero.
                    reconstruction = utils.sparse_result_gemmT(cell_type, average_profiles, adata.X, inplace=False)
                    reconstruction = reconstruction.tocsr()
                else:
                    reconstruction = cell_type @ average_profiles.T
                utils.row_scale(reconstruction, cell_prior)

                if len(adata.obs.index) == 1 and scipy.sparse.issparse(reconstruction): # edgecase bug in scanpy
                    adata.X = adata.X.A
                    reconstruction = reconstruction.A
                adata.X -= current * reconstruction

                del reconstruction

                # eliminate negative counts... they are an artefact of the subtraction and make problems downstream
                if scipy.sparse.issparse(adata.X):
                    adata.X.data *= adata.X.data > 0
                    adata.X.eliminate_zeros()
                else:
                    adata.X *= adata.X > 0

                #print('rescale data to', remaining)

                utils.row_scale(adata.X, remaining * (total_sums / utils.get_sum(adata.X, axis=1)))

        del adata # clean up the copies.
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory    

        return sum_cell_type
    
    return _method

def platform_normalize_annotation_method(
    annotation_method,
    platform_iterations,
    gene_keys=None,
    normalize_to='adata',
    ):

    """\
    Platform normalizes an annotation method. A call to the resulting method
    changes `adata` or `reference` inplace!
    
    Parameters
    ----------
    annotation_method
        A callable of signature
        `method(adata, reference, annotation_key, annotation_prior)`
    platform_iterations
        Number of platform normalization iterations. If `0`, platform
        normalization is done once in the beginning, but no iteration is done.
        If smaller than `0`, no platform normalization is performed at all.
    gene_keys
        String or list of strings specifying additional count-like `.var` and
        `.varm` annotations to scale along with the platform normalization. The
        `annotation_key` is included automatically. If `True`, take all `.var`
        and `.varm` keys.
    normalize_to
        To what expression the adatas should be normalized. Can be one of:
        
        - 'adata': normalize `reference` to conform to `adata`; the resulting
          annotation fractions give how many of the actual reads in `adata` are
          belonging to which annotation.
        - 'reference': normalize `adata` to conform to `reference`; the
          resulting annotation fractions give how many of the reads in `adata`
          would belong to which annotation if they were measured with the same
          platform effects as `reference`.
          
    Returns
    -------
    Returns a callable of signature\
    `method(adata, reference, annotation_key, annotation_prior, verbose)` which\
    performs the actual annotation with platform normalization and normalizes\
    `adata` or `reference` inplace.
    
    """
        
    def _method(adata, reference, annotation_key, annotation_prior, verbose):
        
        nonlocal annotation_method, platform_iterations, gene_keys, normalize_to
        
        if isinstance(gene_keys, str):
            gene_keys = [gene_keys]
        gene_keys = [annotation_key,*[gk for gk in gene_keys]]
        
        if platform_iterations > -1:
            if normalize_to == 'reference':
                adata = preprocessing.normalize_platform(adata=adata, reference=reference, annotation_key=None, reference_annotation_key=None, gene_keys=gene_keys, verbose=verbose)
            else:
                reference = preprocessing.normalize_platform(adata=reference, reference=adata, annotation_key=None, reference_annotation_key=None, gene_keys=gene_keys, verbose=verbose)

        for pi in range(platform_iterations):
            cell_type = annotation_method(adata, reference, annotation_key, annotation_prior, verbose)
            ukey = utils.find_unused_key(adata.obsm)
            adata.obsm[ukey] = cell_type
            if normalize_to == 'reference':
                adata = preprocessing.normalize_platform(adata=adata, reference=reference, annotation_key=ukey, reference_annotation_key=annotation_key, gene_keys=gene_keys, verbose=verbose)
            else:
                reference = preprocessing.normalize_platform(adata=reference, reference=adata, annotation_key=annotation_key, reference_annotation_key=ukey, gene_keys=gene_keys, verbose=verbose)
            del adata.obsm[ukey]
        
        # renormalize profiles as they have been denormalized by platform noramlization
        reference.varm[annotation_key] /= reference.varm[annotation_key].sum(axis=0).to_numpy()
        
        cell_type = annotation_method(adata, reference, annotation_key, annotation_prior, verbose)
        return cell_type
    
    return _method

def multi_center_annotation_method(
    annotation_method,
    multi_center,
    multi_center_amplitudes=True,
    prepare_reconstruction=None,
):

    """\
    Creates an annotation method which internally works not on the annotation
    categories directly, but on multiple automatically generated sub-categories
    using k-means subclusters. This is currently only implemented for
    categorical annotations.
    
    Parameters
    ----------
    annotation_method
        A callable of signature
        `method(adata, reference, annotation_key, annotation_prior)`
    multi_center
        The number of sub-categories per annotation category. If a category has
        less observations than this number, uses all the available observations
        individually. If `None` or smaller than `1`, then the original
        categories are used.
    multi_center_amplitudes
        Whether to run k-means on amplitudes of the observation profiles or on
        the profiles directly.
    prepare_reconstruction
        This is an out-argument providing a dictionary to fill with the data
        necessary for the reconstruction of "denoised" profiles. The necessary
        data is a :class:`~pandas.DataFrame` containing the annotation on
        sub-categories, another :class:`~pandas.DataFrame` containing the
        profiles of the sub-categories, and a mapping of sub-categories to
        their original categories.
        
    Returns
    -------
    Returns a callable of signature\
    `method(adata, reference, annotation_key, annotation_prior, verbose)` which\
    performs the actual annotation on the sub-categories and collects the\
    results afterwards.
    
    """
        
    def _method(adata, reference, annotation_key, annotation_prior, verbose):
        
        nonlocal annotation_method, multi_center, prepare_reconstruction
        
        if multi_center is None or multi_center <= 1:
            cell_type = annotation_method(adata, reference, annotation_key, annotation_prior, verbose)
            if prepare_reconstruction is not None:
                prepare_reconstruction['annotation'] = cell_type.copy()
                prepare_reconstruction['profiles'] = reference.varm[annotation_key].copy()
                prepare_reconstruction['mapping'] = None
            return cell_type
        
        if annotation_key not in reference.obs:
            raise ValueError('`multi_center_annotation_method` needs categorical annotation in the reference!')
        if np.prod(reference.shape) == 0:
            raise ValueError('`multi_center_annotation_method` needs per observation data in the reference!')
        
        new_key = utils.find_unused_key([reference.obs,reference.varm]) # profiles will be generated and stored in varm
        
        reference.obs[new_key] = pd.Series(np.nan, index=reference.obs.index, dtype=str)
            
        preped = reference.copy()
        import scanpy as sc
        if pd.api.types.is_integer_dtype(preped.X):
            preped.X = preped.X.astype(float)
        if multi_center_amplitudes:
            utils.row_scale(preped.X, 1/utils.get_sum(preped.X,axis=1))
            utils.sqrt(preped.X)
        else:
            utils.row_scale(preped.X, 1e4/utils.get_sum(preped.X,axis=1))
            utils.log1p(preped)
            sc.pp.scale(preped)
        sc.pp.pca(preped, random_state=42, n_comps=min(10,min(preped.shape[0],preped.shape[1])-1))
        
        new_cats = []
        for cat, df in reference.obs.groupby(annotation_key):
            _multi_center = min(multi_center, df.shape[0])
            
            X = preped[df.index].obsm['X_pca']
            
            # MiniBatch for speed
            kmeans = MiniBatchKMeans(
                n_clusters=_multi_center,
                random_state=42,
                batch_size=100,
                max_iter=100,
                n_init=3, # avoid FutureWarning about changing the default n_init to 'auto'; possible speedup by using 'auto' - needs evaluation
            ).fit(X)
            
            for c in range(_multi_center):
                new_cats.append(f'{cat}-{c}')
            reference.obs.loc[df.index,new_key] = str(cat) + '-' + pd.Series(kmeans.labels_, index=df.index).astype(str)

        del preped
        
        map_new2orig = pd.Series(new_cats, index=new_cats).str.rsplit('-',n=1).str.get(0)
        # restore dtype
        if hasattr(reference.varm[annotation_key].columns, 'categories'):
            map_new2orig = map_new2orig.astype(reference.varm[annotation_key].columns.categories.dtype)
        map_new2orig = map_new2orig.astype(reference.varm[annotation_key].columns.dtype)
        df_new2orig = pd.get_dummies(map_new2orig)
        
        if isinstance(annotation_prior, pd.Series):
            annotation_prior = pd.Series((df_new2orig.to_numpy() / df_new2orig.sum(axis=0).to_numpy()) @ annotation_prior.reindex(df_new2orig.columns).to_numpy(), index=df_new2orig.index)

        preprocessing.construct_reference_profiles(reference, annotation_key=new_key, inplace=True)
        
        cell_type = annotation_method(adata, reference, new_key, annotation_prior, verbose)
        
        if prepare_reconstruction is not None:
            prepare_reconstruction['annotation'] = cell_type.copy()
            prepare_reconstruction['profiles'] = reference.varm[new_key].copy()
            prepare_reconstruction['mapping'] = map_new2orig
        
        del reference.obs[new_key]
        del reference.varm[new_key]
        
        df_new2orig = df_new2orig.reindex(index=cell_type.columns.astype(df_new2orig.index.dtype)) # annotation_method could reorder the columns - maybe...
        cell_type = pd.DataFrame(cell_type.to_numpy() @ df_new2orig.to_numpy(), index=cell_type.index, columns=df_new2orig.columns)
        
        return cell_type
    
    return _method

def max_annotation_method(
    annotation_method,
    max_annotation,
    prepare_reconstruction
    ):

    """\
    Enforces sparsity on the result of an annotation method.
    
    Parameters
    ----------
    annotation_method
        A callable of signature
        `method(adata, reference, annotation_key, annotation_prior)`
    max_annotation
        Number of different annotations to allow per observation. `1` assigns
        the maximum annotation, higher values assign the top annotations and
        distribute the remaining annotations equally on the top annotations.
        If `None` or smaller than `1`, no restrictions are imposed. 
    prepare_reconstruction
        This is an out-argument providing a dictionary to fill with the data
        necessary for the reconstruction of "denoised" profiles. The necessary
        data is a :class:`~pandas.DataFrame` containing the annotation on
        sub-categories, another :class:`~pandas.DataFrame` containing the
        profiles of the sub-categories, and a mapping of sub-categories to
        their original categories.
        
    Returns
    -------
    Returns a callable of signature\
    `method(adata, reference, annotation_key, annotation_prior, verbose)` which\
    performs the actual annotation with maximum annotation constraint.
    
    """
        
    def _method(adata, reference, annotation_key, annotation_prior, verbose):
        
        nonlocal annotation_method, max_annotation, prepare_reconstruction
        
        cell_type = annotation_method(adata, reference, annotation_key, annotation_prior, verbose)
        
        if max_annotation is not None and max_annotation > 0:
            _cell_type = cell_type.to_numpy()
            nth_largest_values_per_observation = np.partition(_cell_type, -max_annotation, axis=-1)[:,-max_annotation]
            _cell_type[_cell_type<nth_largest_values_per_observation[:,None]] = 0
            _cell_type /= _cell_type.sum(axis=1)[:,None]
            cell_type = pd.DataFrame(_cell_type, index=cell_type.index, columns=cell_type.columns)
            if prepare_reconstruction is not None:
                sub_anno = prepare_reconstruction['annotation']
                sub_map = prepare_reconstruction['mapping']
                if sub_map is not None:
                    sub_anno *= (cell_type != 0).reindex(columns=sub_anno.columns.map(sub_map)).to_numpy()
                    sub_anno /= sub_anno.sum(axis=1).to_numpy()[:,None]
                else:
                    prepare_reconstruction['annotation'] = cell_type.copy()


        return cell_type
    
    return _method

def annotate(
    adata,
    reference,
    annotation_key=None,
    result_key=None,
    counts_location=None,
    method='OT',
    bisections=None,
    bisection_divisor=3,
    platform_iterations=None,
    normalize_to='adata',
    annotation_prior=None,
    multi_center=None,
    multi_center_amplitudes=True,
    reconstruction_key=None,
    max_annotation=None,
    min_counts_per_gene=None,
    min_counts_per_cell=None,
    min_cells_per_gene=None,
    min_genes_per_cell=None,
    remove_constant_genes=True,
    remove_zero_cells=True,
    min_log2foldchange=None,
    min_expression=None,
    remove_mito=False,
    n_hvg=None,
    skip_checks=False,
    assume_valid_counts=False,
    return_reference=False,
    gene_keys=None,
    verbose=1,
    **kw_args,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data.
    
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
    result_key
        The `.obsm` key of `adata` where to store the resulting annotation. If
        `None`, do not write to `adata` and return the annotation as
        :class:`~pandas.DataFrame` instead.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    method
        String selecting the method to use for annotation. Can also be a
        callable of signature
        `method(adata, reference, annotation_key, **kw_args)` returning a
        :class:`~pandas.DataFrame`.
        Possible methods include:
        
        - 'OT', implemented by :func:`~tacco.tools.annotate_OT`
        - 'nnls', implemented by :func:`~tacco.tools.annotate_nnls`
        - 'tangram', implemented by :func:`~tacco.tools.annotate_tangram`
        - 'novosparc', implemented by :func:`~tacco.tools.annotate_novosparc`
        - 'projection', implemented by :func:`~tacco.tools.annotate_projection`
        - 'RCTD', implemented by :func:`~tacco.tools.annotate_RCTD`
        - 'NMFreg', implemented by :func:`~tacco.tools.annotate_NMFreg`
        - 'SingleR', implemented by :func:`~tacco.tools.annotate_SingleR`
        - 'WOT', implemented by :func:`~tacco.tools.annotate_wot`
        - 'svm', implemented by :func:`~tacco.tools.annotate_svm`

    bisections
        If larger than 0, runs a boosted annotator using a basis annotator by
        iteratively running the basis annotator and removing a reconstructed
        fraction of the counts. The parameter gives the number of recursive
        bisections of the annotation. If `None`, defaults to method dependent
        values.
    bisection_divisor
        Number of parts for bisection: First, `bisections` times a fraction of
        1/`bisection_divisor` of the unassigned counts of every observation is
        assigned. The remainder is then split evely in `bisection_divisor`
        parts. E.g. if `bisections` is `2` and `bisection_divisor` is `3`, then
        the assigned fractions per round of typing are 1/3,2/3*(1/3,1/3,1/3),
        if `bisections` is `3` and `bisection_divisor` is `2`, then they are
        1/2,(1/2,(1/2,1/2)/2)/2. Generally, the total number of typing rounds
        is `bisections + bisection_divisor - 1`.
    platform_iterations
        Number of platform normalization iterations before running the
        annotation. If `0`, platform normalization is done once in the
        beginning, but no iteration is done. If smaller than `0`, no platform
        normalization is performed at all. If `None`, defaults to method
        dependent values.
    normalize_to
        To what expression the adatas should be normalized. Can be one of:
        
        - 'adata': normalize `reference` to conform to `adata`; the resulting
          annotation fractions give how many of the actual reads in `adata` are
          belonging to which annotation.
        - 'reference': normalize `adata` to conform to `reference`; the
          resulting annotation fractions give how many of the reads in `adata`
          would belong to which annotation if they were measured with the same
          platform effects as `reference`.
          
    annotation_prior
        A callable of signature `method(adata, reference, annotation_key)`
        which returns priors for the annotation or a :class:`~pandas.Series`
        containing the annotation prior distribution directly.
        This parameter is used only for methods which require such a parameter.
        If `None`, it is determined by summing the annotation in the reference
        data weighted with the counts from `reference.X`.
    multi_center
        The number of sub-categories per annotation category. If a category has
        less observations than this number, uses all the available observations
        individually. If `None` or smaller than `1`, then the original
        categories are used.
    multi_center_amplitudes
        Whether to run k-means on amplitudes of the observation profiles or on
        the profiles directly.
    reconstruction_key
        The key for `.varm`, `.obsm`, and `.uns` where to put information for
        reconstructing "denoised" data: profiles, annotation, and a mapping of
        annotation sub-categories to annotation categories. If
        `multi_center==1`, `reconstruction_key` can be equal to
        `result_key`, as the `annotation` information is identical; a mapping
        is not necessary; just the profiles are additionally stored in `.varm`.
        If `None`, and `multi_center==1`, the `result_key` is used, else
        "{result_key}_mc{multi_center}". If `result_key` is `None`, no
        reconstruction information is returned.
    max_annotation
        Number of different annotations to allow per observation. `1` assigns
        the maximum annotation, higher values assign the top annotations and
        distribute the remaining annotations equally on the top annotations.
        If `None` or smaller than `1`, no restrictions are imposed. 
    min_counts_per_gene
        The minimum number of counts genes must have in both adata and
        reference to be kept.
    min_counts_per_cell
        The minimum number of counts cells must have in both adata and
        reference to be kept.
    min_cells_per_gene
        The minimum number of cells genes must have in both adata and
        reference to be kept.
    min_genes_per_cell
        The minimum number of genes cells must have in both adata and
        reference to be kept.
    remove_constant_genes
        Whether to remove genes which do not show any variation between cells
    remove_zero_cells
        Whether to remove cells without non-zero genes
    min_log2foldchange
        Minimum log2-fold change a gene must have in at least one annotation
        category relative to the mean of the other categories to be kept.
    min_expression
        Minimum expression level relative to all expression a gene must have in
        at least one annotation category to be kept.
    remove_mito
        Whether to remove genes starting with "mt-" and "MT-".
    n_hvg
        The number of highly variable genes to run on. If `None`, use all
        genes.
    skip_checks
        Whether to skip data integrity checks and save time. Only recommended
        for internal use - or people who really know what they are doing.
    assume_valid_counts
        Disable checking for invalid counts (e.g. non-integer or negative).
    return_reference
        Whether to return the platform normalized `reference`.
    gene_keys
        String or list of strings specifying additional count-like `.var` and
        `.varm` annotations to scale along with the platform normalization. The
        `annotation_key` is included automatically. If `True`, take all `.var`
        and `.varm` keys. This makes only sense if `return_reference` is
        `True`.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to the annotation method.
        See the functions mentioned in the documentation of the parameter
        `method` for details.
        
    Returns
    -------
    Depending on `result_key`, either returns the original `adata` with\
    annotation written in the corresponding `.obsm` key, or just the annotation\
    as a new :class:`~pandas.DataFrame`.
    
    """
    
    if adata is None:
        raise ValueError('"adata" cannot be None!')
    if reference is None:
        raise ValueError('"reference" cannot be None!')
        
    annotation_key = _infer_annotation_key(reference, annotation_key)
        
    gene_keys = _get_unique_keys(annotation_key, gene_keys)
        
    if skip_checks:
        tdata = adata
    else:
        if verbose > 0:
            print(f'Starting preprocessing')
        start = time.time()
        
        tdata = get.counts(adata, counts_location=counts_location, annotation=True, copy=False)
        reference = get.counts(reference, counts_location=counts_location, annotation=gene_keys, copy=False)
        
        if annotation_key not in reference.varm:
            if verbose > 0:
                print(f'Annotation profiles were not found in `reference.varm["{annotation_key}"]`. Constructing reference profiles with `tacco.preprocessing.construct_reference_profiles` and default arguments...')
            reference = preprocessing.construct_reference_profiles(reference, annotation_key=annotation_key)
        
        gene_mask = preprocessing.filter_reference_genes(reference, annotation_key=annotation_key, min_log2foldchange=min_log2foldchange, min_expression=min_expression, remove_mito=remove_mito, n_hvg=n_hvg, return_mask=True)
        if not gene_mask.all():
            reference = reference[:,gene_mask].copy()
            tdata = tdata[:,tdata.var.index.intersection(reference.var.index)].copy()
            gc.collect()

#        if annotation_key in reference.varm:
#            reference = preprocessing.filter_profiles(adata=reference, annotation_key=annotation_key, fill_na=0, fill_negative=None) # filter out zero-only genes in the profiles
        if annotation_key in reference.obsm:
            reference = preprocessing.filter_annotation(adata=reference, annotation_key=annotation_key, fill_na=0, fill_negative=None) # filter out zero-only cells in the annotation

        if not assume_valid_counts:
            try:
                preprocessing.check_counts_validity(tdata.X, raise_exception=True)
            except ValueError as e: # as e syntax added in ~python2.5
                raise ValueError(f'{str(e)}\nYou can deactivate checking for invalid counts by specifying `assume_valid_counts=True`.')
        tdata,reference = preprocessing.filter(adata=(tdata,reference), min_counts_per_cell=min_counts_per_cell, min_counts_per_gene=min_counts_per_gene, min_cells_per_gene=min_cells_per_gene, min_genes_per_cell=min_genes_per_cell, remove_constant_genes=remove_constant_genes, remove_zero_cells=remove_zero_cells, assume_valid_counts=True) # ensure consistent gene selection
        if verbose > 0:
            print(f'Finished preprocessing in {np.round(time.time() - start, 2)} seconds.')

    # construct the method to perform the actual annotation
    prior_argument_string = f' annotation_prior={annotation_prior}'
    kw_args_string = ''.join([f' {k}={v}' for k,v in kw_args.items()])
    method_construction_info = [f'+- core: method={method}{prior_argument_string}{kw_args_string}']
    _method = annotation_method(method=method, **kw_args)
    
    if bisections is None:
        if method == 'OT':
            bisections = 4
        else:
            bisections = 0
    if bisections > 0:
        _method = boost_annotation_method(annotation_method=_method, bisections=bisections, bisection_divisor=bisection_divisor)
        method_construction_info = ['   ' + i for i in method_construction_info]
        method_construction_info.append(f'+- bisection boost: bisections={bisections}, bisection_divisor={bisection_divisor}')
    
    prepare_reconstruction = None if result_key is None else {}
    
    if result_key is None and reconstruction_key is not None:
        raise ValueError(f'`result_key` is `None`, indicating that the `adata` will not be changed, but `reconstruction_key` is "{reconstruction_key}" and requires adata to be changed!')
    if result_key is not None and reconstruction_key is None:
        if multi_center is None or multi_center <= 1:
            reconstruction_key = result_key
        else:
            reconstruction_key = f'{result_key}_mc{multi_center}'
    if result_key is not None and result_key == reconstruction_key:
        if multi_center is not None and multi_center > 1:
            raise ValueError(f'`result_key` and `reconstruction_key` are identical "{result_key}", but `multi_center>1`! `result_key` and `reconstruction_key` can only be equal if the annotation is identical to the reconstruction information.')
    if multi_center is not None or reconstruction_key is not None:
        _method = multi_center_annotation_method(annotation_method=_method, multi_center=multi_center, multi_center_amplitudes=multi_center_amplitudes, prepare_reconstruction=prepare_reconstruction)
        method_construction_info = ['   ' + i for i in method_construction_info]
        method_construction_info.append(f'+- multi center: multi_center={multi_center} multi_center_amplitudes={multi_center_amplitudes}')
    
    if platform_iterations is None:
        if method in ['OT','projection']:
            platform_iterations = 0
        else:
            platform_iterations = -1
    if platform_iterations >= 0:
        if np.prod(reference.shape) == 0:
            if verbose > 0:
                print(f'There is no expression data available in the `reference` (`np.prod(reference.shape) == 0`)! Therefore no platform normalization is performed even though `platform_iterations={platform_iterations}`.')
        else:
            if normalize_to == 'reference':
                tdata = tdata.copy() # platform normalization changes tdata inplace
                if not pd.api.types.is_float_dtype(tdata.X.dtype):
                    tdata.X = tdata.X.astype(float)
            else:
                reference = reference.copy() # platform normalization changes reference inplace
                if not pd.api.types.is_float_dtype(reference.X.dtype):
                    reference.X = reference.X.astype(float)
            _method = platform_normalize_annotation_method(annotation_method=_method, platform_iterations=platform_iterations, gene_keys=gene_keys, normalize_to=normalize_to)
            method_construction_info = ['   ' + i for i in method_construction_info]
            method_construction_info.append(f'+- platform normalization: platform_iterations={platform_iterations}, gene_keys={gene_keys}, normalize_to={normalize_to}')
    
    if max_annotation is not None:
        _method = max_annotation_method(annotation_method=_method, max_annotation=max_annotation, prepare_reconstruction=prepare_reconstruction)
        method_construction_info = ['   ' + i for i in method_construction_info]
        method_construction_info.append(f'+- maximum annotation: max_annotation={max_annotation}')
    
    if verbose > 0:
        print(f'Starting annotation of data with shape {tdata.shape} and a reference of shape {reference.shape} using the following wrapped method:')
        print('\n'.join(method_construction_info[::-1]))
    start = time.time()
    cell_type = _method(tdata, reference, annotation_key, annotation_prior, verbose)
    if verbose > 0:
        print(f'Finished annotation in {np.round(time.time() - start, 2)} seconds.')
    
    if reconstruction_key is not None:
        annotation = prepare_reconstruction['annotation']
        profiles = prepare_reconstruction['profiles']
        mapping = prepare_reconstruction['mapping']
        
        if mapping is not None:
            adata.uns[reconstruction_key] = mapping
        
        adata.varm[reconstruction_key] = profiles.reindex(index=adata.var.index)
            
        if result_key != reconstruction_key:
            adata.obsm[reconstruction_key] = annotation.reindex(index=adata.obs.index)
    
    # conserve/define types and names
    if annotation_key in reference.obs:
        if hasattr(reference.obs[annotation_key], 'cat'):
            cell_type.columns = cell_type.columns.astype(reference.obs[annotation_key].cat.categories.dtype)
        cell_type.columns = cell_type.columns.astype(reference.obs[annotation_key].dtype)
    elif annotation_key in reference.obsm:
        cell_type.columns = cell_type.columns.astype(reference.obsm[annotation_key].columns.dtype)
    elif annotation_key in reference.varm:
        cell_type.columns = cell_type.columns.astype(reference.varm[annotation_key].columns.dtype)
    cell_type.columns.name = annotation_key
    cell_type.index = cell_type.index.astype(str)
    
    if result_key is not None:
        adata.obsm[result_key] = cell_type.reindex(adata.obs.index)
        result = adata
    else:
        result = cell_type.reindex(adata.obs.index)
    result = (result, reference) if return_reference else result
    
    del tdata, reference # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    return result

def _estimate_bin_size(coordinates, target_number=250):
    
    # find order of magnitude of extent assuming a hyper-cube of data
    maxs, mins = coordinates.max(axis=0), coordinates.min(axis=0)
    extents = maxs - mins
    volume = np.prod(extents)
    density = len(coordinates) / volume
    
    bin_volume = target_number / density
    
    bin_size = bin_volume**(1/len(extents))
    
    return bin_size

def _enum_shifts(n_shifts,n_dim,shifts=None):
    if n_dim == 0:
        return np.array(shifts)
    if shifts is None:
        shifts = [ [s] for s in range(n_shifts) ]
    else:
        shifts = [ [s,*shift] for s in range(n_shifts) for shift in shifts ]
    return _enum_shifts(n_shifts,n_dim-1,shifts=shifts)

def annotate_single_molecules(
    molecules,
    reference,
    annotation_key,
    result_key=None,
    bin_size=None,
    position_keys=['x','y'],
    gene_key='gene',
    n_shifts=2,
    verbose=1,
    **kw_args,
    ):

    """\
    Annotates single molecules in space using reference data.
    
    Parameters
    ----------
    molecules
        A :class:`~pandas.DataFrame` with columns containing spatial
        coordinates and molecule species annotation.
    reference
        Reference data to get the annotation definition from.
    annotation_key
        The `.obs` and/or `.varm` key where the annotation and/or profiles are
        stored in the `reference`. If `None`, it is inferred from `reference`,
        if possible.
    result_key
        The key of `molecules` where to store the resulting annotation. If
        `None`, do not write to `molecules` and return the annotation as
        :class:`~pandas.Series` instead.
    bin_size
        The spatial size of a bin. Bins are of the same size in all
        dimensions. A bin should be large enough to contain enough molecules
        to enable statements about its type composition. If `None`, use a
        heuristic to choose a value.
    position_keys
        Array-like of column keys which contain the position of the molecules.
    gene_key
        The name of the column which contains the molecule species annotation.
    n_shifts
        An integer giving the number of independent binnings per dimension.
        Larger values give a better majority vote for the molecule annotation.
        But computing time scales with `n_shifts**dimension`.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools.annotate`.
        
    Returns
    -------
    Depending on `result_key`, either returns the original `molecules` with\
    annotation written in the corresponding column, or just the annotation\
    as a new :class:`~pandas.Series`.
    
    """
        
    annotation_key = _infer_annotation_key(reference, annotation_key)
    
    if bin_size is None:
        bin_size = _estimate_bin_size(molecules[position_keys])
        if verbose > 0:
            print(f'estimated bin_size: {bin_size}')
    
    shift_annos = pd.DataFrame(index=molecules.index)
    
    n_dim = len(position_keys)

    shifts = _enum_shifts(n_shifts=n_shifts,n_dim=n_dim)
    
    # do the binning, typing, splitting, mapping for multiple shifted bins
    for shift in shifts:

        bins = utils.bin(molecules, bin_size=bin_size, position_keys=position_keys, shift=-shift*bin_size/n_shifts); # shift bin to negaive directions to not exclude points

        hashes = utils.hash(bins);

        bingene = pd.DataFrame({'bin':hashes,'gene':molecules[gene_key]})

        rdata = utils.dataframe2anndata(bingene, obs_key='bin', var_key='gene')
        
        # annotate bins and get the reference, because the annotation profiles are needed for bead splitting and might have been generated in annotate()
        result, _reference = annotate(adata=rdata, reference=reference, annotation_key=annotation_key, result_key='anno', return_reference=True, verbose=verbose, **kw_args);
    
        rdata.varm['anno'] = _reference.varm[annotation_key].reindex(index=rdata.var.index)
        del _reference # clean up
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
        sdata = split_observations(adata=rdata, annotation_key='anno', min_counts=1, verbose=verbose);
        del rdata # clean up
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory

        ldata = utils.anndata2dataframe(adata=sdata, obs_name='split', obs_keys=['bin', 'anno']);
        del sdata # clean up
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory

        # align dtypes as anndata converted the index to str above
        ldata['bin'] = ldata['bin'].astype(bingene['bin'].dtype)
        if hasattr(bingene['gene'], 'cat'):
            ldata['gene'] = ldata['gene'].astype(bingene['gene'].cat.categories.dtype)
        ldata['gene'] = ldata['gene'].astype(bingene['gene'].dtype)
        utils.hash(bingene, hash_key='bingene', other=ldata);

        bingene = bingene[bingene['bingene'].isin(ldata['bingene'])].copy()
    
        anno = _map_hash_annotation(bingene, abcdata=ldata, annotation_key='anno', hash_key='bingene', count_key='X')
        
        shift_annos[str(shift)] = anno
    
    anno = utils.mode(shift_annos)
    
    # conserve index dtype
    if hasattr(molecules.index, 'categories'): # categorical index
        anno.index = anno.index.astype(type(molecules.index.categories))
    anno.index = anno.index.astype(molecules.index.dtype)
    anno.name = annotation_key
    
    if result_key is not None:
        molecules[result_key] = anno.reindex(molecules.index)
        result = molecules
    else:
        result = anno
    
    return result

def segment(
    molecules,
    distance_scale,
    max_size,
    position_scale='auto',
    position_keys=['x','y'],
    result_key=None,
    max_distance=None,
    annotation_key=None,
    annotation_distance=None,
    distance_kw_args={},
    gene_key=None,
    verbose=1,
    **kw_args,
    ):

    """\
    Segment single molecules in space to get a cell-like annotation.
    
    Parameters
    ----------
    molecules
        A :class:`~pandas.DataFrame` with columns containing spatial
        coordinates and annotation.
    distance_scale
        This is a smooth size of the molecule neighborhood to be considered in
        clustering (the width parameter of the Gaussian for affinity
        calculation).
    max_size
        The most important parameter for the hierarchical spectral clustering in
        :func:`~tacco.tools.spectral_clustering`: The clustering goes on until
        no cluster has more elements than this. Additional arguments can be
        supplied as key word arguments.
    position_scale
        The most important parameter for the hierarchical spectral clustering
        in :func:`~tacco.tools.spectral_clustering` when spatial information is
        provided: The expected feature size to use for splitting the problem
        spatially. If `position_key` or `position_scale` is `None`, do hirarchical
        clustering to iteratively split the problems in smaller subproblems.
        If `position_scale` is "auto", it is estimated based on a heuristic.
    position_keys
        Array-like of column keys which contain the position of the molecules.
    result_key
        The key of `molecules` where to store the resulting annotation. If
        `None`, do not write to `molecules` and return the annotation as
        :class:`~pandas.Series` instead.
    max_distance
        The maximum distance to consider in the distance matrix. This should be
        large enough to capture the wider local connectivity between molecules.
        `max_distance` and `sigma` have similar effects, with `max_distance`
        giving a hard cutoff which is crucial for fast computations, while
        `sigma` gives a smooth cutoff. If `None`, `max_distance` is taken to be
        `2*distance_scale`.
    annotation_key
        The column containing categorical annotation information to support the
        segmentation, e.g. cell type. If `None`, the segmentation is done using
        the molecule distribution without any further annotation.
    annotation_distance
        Specifies the effect of `annotation_key` in adding a distances
        between two observations of different type. It can be:
        
        - a scalar to use for all annotation pairs
        - a :class:`~pandas.DataFrame` to give every annotation pair its own
          finite distance. If some should retain infinite distance, use
          `np.inf`, `np.nan` or negative values
        - `None` to use an infinite distance between different annotations
        - a metric to calculate a distance between the annotation profiles.
          This is forwarded to :func:`~tacco.utils.cdist` as the `metric`
          argument, so everything available there is also posible here, e.g.
          'h2'.
          
    distance_kw_args
        A dictionary of additional keyword arguments to be forwarded to
        :func:`~tacco.tools.distance_matrix`.
    gene_key
        The name of the column which contains the molecule species annotation.
        This is used iff the annotation distance is to be calculated with a
        metric specified by a string via `annotation_distance`.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools.spectral_clustering`.
        
    Returns
    -------
    Depending on `result_key`, either returns the original `molecules` with\
    cell-like annotation written in the corresponding column, or just the\
    cell-like annotation as a new :class:`~pandas.Series`.
    
    """
    
    if max_distance is None:
        max_distance = 2 * distance_scale
    
    if annotation_key is not None and isinstance(annotation_distance, str) and gene_key is None:
        raise ValueError(f'`annotation_key` is not `None`, a metric is supplied for `annotation_distance` "{annotation_distance}", but `gene_key` is `None`, so nothing is available to calculate the distance on!')
    
    fdata = utils.dataframe2anndata(molecules, obs_key=None, var_key=gene_key)
    distance_key = 'distance'
    distance_matrix(
        adata=fdata,
        max_distance=max_distance,
        position_key=position_keys,
        result_key=distance_key,
        annotation_key=annotation_key,
        annotation_distance=annotation_distance,
        distance_scale=distance_scale,
        verbose=verbose,
        **distance_kw_args,
    )
    affinity_key = 'affinity'
    affinity(
        adata=fdata,
        sigma=distance_scale,
        distance_key=distance_key,
        result_key=affinity_key,
    )
    if position_scale == 'auto':
        position_scale = _estimate_bin_size(molecules[position_keys], target_number=max_size)
        if verbose > 0:
            print(f'estimated position_scale: {position_scale}')
    anno = spectral_clustering(
        adata=fdata,
        max_size=max_size,
        affinity_key=affinity_key,
        position_key=position_keys,
        verbose=verbose,
        **kw_args,
    )
    
    # conserve index dtype
    if hasattr(molecules.index, 'categories'): # categorical index
        anno.index = anno.index.astype(type(molecules.index.categories))
    anno.index = anno.index.astype(molecules.index.dtype)
    
    if result_key is not None:
        molecules[result_key] = anno.reindex(index=molecules.index)
        result = molecules
    else:
        result = anno
    
    return result
