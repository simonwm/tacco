import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from ..utils import _split
from .. import utils
from .. import get
from .. import preprocessing
from .. import testing
from . import _helper as helper

def map_obs_obsm(
    adata,
    source_adata,
    obs_index_key,
    map_obs_keys=False,
    map_obsm_keys=False,
):
    """\
    Maps obs and obsm annotation from a source to a target adata.
    
    Parameters
    ----------
    adata
        A target :class:`~anndata.AnnData` to map the annotation to.
    source_adata
        A source :class:`~anndata.AnnData` to map the annotation to.
    annotation
        A :class:`~pandas.DataFrame` containing the obs by category annotation.
    obs_index_key
        The key in `adata.obs` which contains corresponding index values of
        `source_adata.obs.index` to define the mapping.
    map_obs_keys
        List of `.obs` keys to map to the new data. If `True` or `False`, maps
        all or no `.obs` keys.
    map_obsm_keys
        List of `.obsm` keys to map to the new data. If `True` or `False`, maps
        all or no `.obsm` keys.
        
    Returns
    -------
    Returns the updated input `adata`.
    
    """
        
    if isinstance(map_obs_keys, bool):
        if map_obs_keys:
            map_obs_keys = source_adata.obs.columns
        else:
            map_obs_keys = []
    elif isinstance(map_obs_keys, str):
        map_obs_keys = [map_obs_keys]
    if isinstance(map_obsm_keys, bool):
        if map_obsm_keys:
            map_obsm_keys = source_adata.obsm.keys
        else:
            map_obsm_keys = []
    elif isinstance(map_obsm_keys, str):
        map_obsm_keys = [map_obsm_keys]

    if len(map_obs_keys) > 0:
        reindexed_obs = source_adata.obs.reindex(index=adata.obs[obs_index_key]).set_index(adata.obs.index)
    for k in map_obs_keys:
        adata.obs[k] = reindexed_obs[k]
    for k in map_obsm_keys:
        adata.obsm[k] = source_adata.obsm[k].reindex(index=adata.obs[obs_index_key]).set_index(adata.obs.index)
    
    return adata

def reconstruct_expression(
    adata,
    annotation,
    profiles,
    mapping=None,
    min_counts=100,
    rounded=False,
    merge_key=None,
    obs_index_key=None,
    verbose=1,
):
    """\
    Combines annotation and profiles to reconstruct a "denoised" expression
    matrix.
    
    Parameters
    ----------
    adata
        A source :class:`~anndata.AnnData` providing the counts to split
        and annotation to include in the result.
    annotation
        A :class:`~pandas.DataFrame` containing the obs by category annotation.
    profiles
        A :class:`~pandas.DataFrame` containing the gene by category profiles.
    mapping
        A mapping of sub-categories to their original categories. If `None`,
        directly return the result in the categories given in `annotation` and
        `profiles`. If 'bulk', maps everything to a single unsplit expression
        matrix.
    min_counts
        Minimum count per observation to include in the result. As the result
        a dense matrix, this parameter is quite crucial to get a reasonably
        sized result. If `None`, include all non-zero observations.
    rounded
        Whether to round the result to the nearest integer-valued matrix. This
        leads to the result being reported using a sparse matrix.
    merge_key
        The `.obs` key to contain the annotation in the merged data. If `None`,
        tries to guess a reasonable name.
    obs_index_key
        A string specifying the name of the obs column to write the old
        `.obs.index` (i.e. the cell names) to. If `None`, tries to guess a
        reasonable name.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the per annotation category\
    reconstructed expression data.
    
    """
    
    categories = annotation.columns
    if len(categories) != len(profiles.columns) or (~categories.isin(profiles.columns)).any():
        raise ValueError(f'The `annotation` and `profiles` use incompatible annotation categories: {categories} VS {profiles.columns}')
    elif (categories != profiles.columns).any():
        profiles.reindex(columns=categories)
    
    obs_index_key = helper.guess_obs_index_key(obs_index_key, adata)
    
    merge_key = helper.guess_merge_key(merge_key, adata, annotation, profiles)
    
    # filter out observations with missing annotation
    annotation.fillna(0.0, inplace=True)
    no_annotation = annotation.sum(axis=1) == 0.0
    annotation = annotation.loc[~no_annotation]
    # filter out genes without contribution to any profile
    profiles.fillna(0.0, inplace=True)
    no_profiles = profiles.sum(axis=1) == 0.0
    profiles = profiles.loc[~no_profiles]
    # find intersection of annotation and data
    obs_intersection = annotation.index.intersection(adata.obs.index)
    var_intersection = profiles.index.intersection(adata.var.index)
    annotation = annotation.loc[obs_intersection]
    profiles = profiles.loc[var_intersection]
    adata = adata[obs_intersection,var_intersection].copy()
    # normalize annotation: every bead has type fractions that sum to the total number of counts per bead
    annotation = annotation * (utils.get_sum(adata.X, axis=1) / annotation.sum(axis=1).to_numpy())[:,None]
    # normalize profiles: every type has gene fractions that sum to 1
    profiles = profiles / profiles.sum(axis=0).to_numpy()
     
    if mapping is None:
        mapping = pd.Series(categories, index=categories)
    elif isinstance(mapping, str):
        if mapping == 'bulk':
            merge_key = None
            mapping = pd.Series(['bulk']*len(categories), index=categories)
        else:
            raise ValueError(f'`mapping` is the string "{mapping}", but the string "bluk" is the only string option. `mapping` can also be `None` or a mapping.')

    contributions = {}
    for old, new in mapping.items():
        if new not in contributions:
            contributions[new] = []
        if old in annotation.columns:
            contributions[new].append(old)
        else:
            if verbose > 0:
                print(f'`mapping` references sub-category {old!r} which is not available in `annotation.columns`. Continuing without this sub-category...')

    obs = []
    X = []

    for i, (new, olds) in enumerate(contributions.items()):
        if verbose > 1:
            print(f'runnig the reconstruction of {new!r} by {olds!r}')
        _X = utils.gemmT(annotation[olds].to_numpy(), profiles[olds].to_numpy())

        if min_counts is None:
            keep = utils.get_sum(_X, axis=1) > 0
        else:
            keep = utils.get_sum(_X, axis=1) >= min_counts
        _X = _X[keep].copy()
        if rounded:
            _X = scipy.sparse.csr_matrix(np.round(_X))

        X.append(_X)
        _obs = pd.DataFrame({obs_index_key:annotation.index[keep]})
        if merge_key is not None:
            _obs[merge_key] = new
        obs.append(_obs)

    if scipy.sparse.issparse(X[0]):
        X = scipy.sparse.vstack(X)
    else:
        X = np.vstack(X)
    obs = pd.concat(obs)
    obs.reset_index(drop=True,inplace=True)
    var = adata.var.reindex(index=profiles.index)

    if merge_key is not None:
        # conserve dtype
        if hasattr(mapping, 'cat'):
            obs[merge_key] = obs[merge_key].astype(mapping.cat.categories.dtype)
        obs[merge_key] = obs[merge_key].astype(mapping.dtype)

    result = sc.AnnData(X, obs=obs, var=var)

    for k in adata.varm:
        result.varm[k] = adata.varm[k]

    return result

def split_observations(
    adata,
    annotation_key,
    result_key=None,
    counts_location=None,
    mode='exact',
    map_all_genes=False,
    min_counts=None,
    seed=42,
    delta=1e-6,
    scaling_jobs=None,
    scaling_batch_size=1000,
    rounding_batch_size=50000,
    obs_index_key=None,
    map_obs_keys=False,
    map_obsm_keys=False,
    verbose=1,
):
    
    """\
    Splits expression data with a "soft" weights annotation in `.obsm` into
    multiple "virtual" observations per input observation e.g. splits
    expression data for cell type mixtures with annotated cell type fractions
    into expression of single type observations with categorical annotation.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`,
        profiles in `.varm` and annotation in `.obsm`.
    annotation_key
        The `.obsm` and `.varm` key where the annotation and profiles are
        stored. If the `annotation_key` is also available in `.uns`, it
        should contain a mapping of annotation categories from `.obsm` and
        `.varm` to the target ones. Such a triplett of annotation can be
        generated by :func:`~tacco.tools.annotate` with a `reconstruction_key`
        argument.
    result_key
        The name of the `.obs` annotation column of the result to contain the
        split annotation. If `None`, tries to find a sensible value
        automatically: If `annotation_key` is not in `.uns`, then
        `annotation_key` is used, else the name attribute of the mapping is
        used if available.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    mode
        String to switches between the type of split:
        
        - 'exact': All counts in the input are distributed to the splitted
          observations conserving the total number of counts per gene and bead
          and the annotation fractions; see `map_zero_profile_genes`.
        - 'denoise': The counts per gene and bead are ignored and the split
          results from a matrix product of mean expression profiles and the
          annotation. Depending on the input this can be done on a sub-category
          level providing variation within the categories, if `annotation_key`
          is a `.uns` key.
        - 'bulk': Like 'denoise', but summing over all (sub-)categories.
        
    map_all_genes
        Only for `mode` 'exact': Whether to map counts of genes without profile
        information assuming equal probabilities for all profiles.
    min_counts
        Minimum count per observation to include in the splitted data.
        If `None`, include all non-zero observations.
    seed
        Random seed for integerizing the splitted count matrix. If `None`,
        directly return the non-integer valued count matrix.
        If `mode=='denoise'` a non-`None` value leads to plain rounding.
    delta
        The relative error target for the matrix scaling.
        Ignored if `mode=='denoise'`.
    scaling_jobs
        Number of jobs or cores to use for the matrix scaling task.
        Ignored if `mode=='denoise'`.
    scaling_batch_size
        Batch size for the matrix scaling task.
        Ignored if `mode=='denoise'`.
    rounding_batch_size
        Batch size for the rounding task.
        Ignored if `mode=='denoise'`.
    obs_index_key
        A string specifying the name of the obs column to write the old
        `.obs.index` (i.e. the cell names) to. If `None`, tries to guess a
        reasonable name.
    map_obs_keys
        List of `.obs` keys to map to the new data. If `True` or `False`, maps
        all or no `.obs` keys.
    map_obsm_keys
        List of `.obsm` keys to map to the new data. If `True` or `False`, maps
        all or no `.obsm` keys.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the splitted expression\
    data.
    
    """

    adata = get.counts(adata, counts_location=counts_location, annotation=True, copy=True)
    preprocessing.check_counts_validity(adata.X)
    
    if (annotation_key in adata.obsm) and (annotation_key in adata.varm):
        annotation = adata.obsm[annotation_key]
        profiles = adata.varm[annotation_key]
    elif annotation_key not in adata.obsm:
        raise ValueError(f'The key "{annotation_key}" is not available in `.obsm`!')
    else:
        raise ValueError(f'The key "{annotation_key}" is not available in `.varm`!')
    
    mapping = adata.uns[annotation_key] if annotation_key in adata.uns else None
    
    obs_index_key = helper.guess_obs_index_key(obs_index_key, adata)
    
    merge_key = result_key
    if merge_key is None:
        if mapping is None:
            merge_key = annotation_key
        elif mapping.name is not None:
            merge_key = mapping.name
        else:
            raise ValueError(f'An automatic value for `result_key` could not be determined! Specify it explicitly.')
    
    if mode == 'exact':
        
        sdata = _split.split_beads(adata, annotation, profiles, min_counts=(0 if min_counts is None else min_counts), seed=seed, delta=delta, scaling_jobs=scaling_jobs, scaling_batch_size=scaling_batch_size, rounding_batch_size=rounding_batch_size, copy=False, map_all_genes=map_all_genes, verbose=verbose)

        layers = profiles.columns
        if mapping is not None:
            _split.combine_layers(sdata, mapping, remove_old_layers=True)
            layers = layers.map(mapping).unique()

        mdata = _split.merge_layers(sdata, layers, merge_annotation_name=merge_key, bead_annotation_name=obs_index_key, min_reads=min_counts)
    
    elif mode == 'denoise' or mode == 'bulk':
        
        if mode == 'bulk':
            mapping = 'bulk'
        
        mdata = reconstruct_expression(adata, annotation, profiles, mapping, min_counts=min_counts, rounded=(seed is not None), merge_key=merge_key, obs_index_key=obs_index_key, verbose=verbose)
        
    else:
        
        raise ValueError(f'`mode` can only be one of ["exact","denoise","bulk"] but got "{mode}"!')
    
    map_obs_obsm(mdata, adata, obs_index_key, map_obs_keys=map_obs_keys, map_obsm_keys=map_obsm_keys)
    
    return mdata

def merge_observations(
    adata,
    obs_index_key,
    annotation_key=None,
    min_counts=0,
    mean_keys=None,
):

    """\
    Merges equivalent observations of an :class:`~anndata.AnnData` by summing
    over the expression. Equivalence is determined by the values of `.obs`
    columns.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        annotation in `.obs`.
    obs_index_key
        The primary `.obs` key to merge on. May be an original observation
        index before :func:`~split_observations`.
    annotation_key
        An optional second `.obs` key to merge on.
    min_counts
        Minimum count per observation to include in the merged data.
    mean_keys
        The names of the columns containing numerical properties to construct
        mean quantities for `.obs` columns with.
        
    Returns
    -------
    Returns an :class:`~anndata.AnnData` containing the merged expression data.
    
    """

    adata = _split.merge_beads(adata, merge_annotation_name=annotation_key, bead_annotation_name=obs_index_key, min_counts=min_counts, mean_keys=mean_keys)
    
    return adata
