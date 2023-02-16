import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import scipy.spatial
import scipy.sparse.csgraph
from numba import njit
import sklearn.cluster
import sklearn.linear_model
import gc
import tempfile

from .. import utils
from ..utils import _math
from .. import preprocessing
from .. import get
from . import _helper as helper

#@njit(fastmath=True)
def _map_types(cht_indptr, cht_indices, cht_data, point_hashes, dtype):
    point_types = np.full(point_hashes.shape, -1, dtype=dtype)
    i = 0
    for cht_hash in range(len(cht_indptr)-1):
        ptr_start = cht_indptr[cht_hash];
        ptr_end   = cht_indptr[cht_hash+1];
        if ptr_start == ptr_end: # no data for this hash
            continue
        
        while point_hashes[i] < cht_hash:
            raise ValueError('map_types got unfiltered and/or unsorted input??')
            point_types[i] = -1 # -1 means 'no type' # should not happen, due to filter outside
            i += 1
        # point_hashes[i] == cht_hash is now guaranteed, as cht hashes are a subset of rna hashes
        
        for ptr in range(ptr_start, ptr_end):
            cht_type = cht_indices[ptr];
            cht_count = cht_data[ptr];
            
            for c in range(cht_count):
                point_types[i] = cht_type
                i += 1
    
    return point_types

def _map_hash_annotation(
    data,
    abcdata,
    annotation_key,
    hash_key='hash',
    count_key='X',
    result_key=None,
):

    """\
    Maps annotation from annotated binned count data to unbinned data.
    
    Parameters
    ----------
    data
        A :class:`~pandas.DataFrame` containing the unbinned data in long
        format.
    abcdata
        A :class:`~pandas.DataFrame` containing annotated binned count data
        in long format.
    annotation_key
        The key containing the categorical annotation to map in `abcdata`.
    hash_key
        The key which identifies corresponding entries in `data` and
        `abcdata`.
    count_key
        The key corresponding to observation counts/weights in `abcdata`.
    result_key
        The key to contain the mapped annotation in `data`. If `None`, a
        :class:`~pandas.Series` containing the mapped annotation is returned.
        
    Returns
    -------
    Depending on `result_key` returns either a :class:`~pandas.Series` of\
    annotation assignments or the updated input `data` contining the\
    assignments under `result_key`.
    
    """
    
    # make hash keys compatible
    point_hashes = data[hash_key]
    if not hasattr(point_hashes, 'cat'):
        point_hashes = point_hashes.astype('category')
    bin_hashes = abcdata[hash_key]
    if not hasattr(bin_hashes, 'cat'):
        bin_hashes = bin_hashes.astype('category')
    unavailable = bin_hashes.cat.categories[~bin_hashes.cat.categories.isin(point_hashes.cat.categories)]
    if len(unavailable) != 0:
        raise ValueError(f'hashes in `data` and `abcdata` are not compatible: `abcdata["{hash_key}"]` contained the hashes {unavailable}, which were not found in `data["{hash_key}"]`! Maybe the dtypes do not match and e.g. one is str and the other int?')
    point_hashes = point_hashes[point_hashes.isin(bin_hashes.cat.categories)] # it is OK to have some hashes not recieve any annotation
    bin_hashes = bin_hashes.astype(point_hashes.dtype) # use the identical categorical type to be certain that the .cat.codes are compatible
    
    # get bin*annotation anndata with number of annotated entitites in .X
    adata = utils.dataframe2anndata(abcdata, obs_key=bin_hashes, var_key=annotation_key, count_key=count_key) # the hash .cat.codes ~ rows in adata are ordered by construction

    point_hashes = point_hashes.sort_values() # ordered hashes are assumed in the numba part; sorting conserves mapping of entity to hash via the index

    sum0 = utils.get_sum(adata.X,axis=1)
    sum1 = point_hashes.cat.codes.value_counts()[np.arange(len(point_hashes.cat.categories))].to_numpy()
    if ((sum0-sum1)!=0).sum() > 0:
        raise ValueError(f'The hashed in `data` and `absdata` are not 1-to-1 mappable! In case this is the result of `split_observation`, a smaller stopping criterion `delta` could help...')
    
    dtype = utils.min_dtype(adata.shape[1] + 1)
    point_annotation = _map_types(adata.X.indptr, adata.X.indices, adata.X.data, point_hashes.cat.codes.to_numpy(), dtype)

    # map back from numbers to annotation
    point_annotation = adata.var.index[point_annotation]
    if hasattr(abcdata[annotation_key], 'cat'):
        point_annotation = point_annotation.astype(abcdata[annotation_key].cat.categories.dtype)
    point_annotation = point_annotation.astype(abcdata[annotation_key].dtype)

    del adata # clean up
    gc.collect() # anndatas are not well garbage collected and accumulate in memory

    point_annotation = pd.Series(point_annotation, index=point_hashes.index).reindex(index=data.index) # reindex result to conform with the input

    if result_key is None:
        return point_annotation
    else:
        data[result_key] = point_annotation
        return data

@njit(cache=True)
def _strip_distance_matrix(row, col, data, max_distance):
    n = len(row)
    new_row = (row)
    new_col = (col)
    new_data = (data)
    # only take the upper triangular part and mirror it later
    c = 0
    for i in range(n):
        if row[i] > col[i] and data[i] <= max_distance:
            new_row[c] = row[i]
            new_col[c] = col[i]
            new_data[c] = data[i]
            c += 1
    new_row[c:(2*c)] = new_col[:c]
    new_col[c:(2*c)] = new_row[:c]
    new_data[c:(2*c)] = new_data[:c]
    
    return new_row[:2*c].copy(), new_col[:2*c].copy(), new_data[:2*c].copy()
    
def strip_distance_matrix(distance, max_distance):
    if not isinstance(distance, scipy.sparse.coo_matrix):
        raise ValueError('`distance` has to be a `scipy.sparse.coo_matrix`!')
    distance.row, distance.col, distance.data = _strip_distance_matrix(distance.row, distance.col, distance.data, max_distance)

@njit(cache=True)
def _update_submatrix(M, m, indx, indy):
    n_x, n_y = m.shape
    assert(n_x == len(indx))
    assert(n_y == len(indy))
    
    for i in range(n_x):
        for j in range(n_y):
            M[indx[i],indx[j]] = m[i,j]

def distance_matrix(
    adata,
    max_distance,
    position_key=['x','y'],
    base_distance_key=None,
    result_key=None,
    annotation_key=None,
    annotation_distance=None,
    distance_scale=None,
    annotation_distance_scale=None,
    coo_result=False,
    low_mem=False,
    verbose=1,
    **kw_args,
):

    """\
    Calculates a sparse or dense distance matrix.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`
    max_distance
        The maximum distance to calculate. All larger distances are unset in
        the result (which acts as a `0` for sparse matrices...). `None` and
        `np.inf` result in dense distance computation (which can be infeasible
        for larger datasets).
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates
    base_distance_key
        The `.obsp` key containing a precomputed distance matrix to update with
        annotation distance. If `None`, the distances are recomputed with the
        positions found in `position_key`. Otherwise `position_key` is ignored.
        If `.obsp[base_distance_key]` does not exist, the distances are also
        recomputed and then written to `.obsp[base_distance_key]`.
    result_key
        The `.obsp` key to contain the distance matrix. If `None`, a
        :class:`~scipy.sparse.csr_matrix` containing the distances is returned.
    annotation_key
        The `.obs` key for a categorical annotation to split the data before
        calculating distances. If `None`, the distances are calculated on the
        full dataset.
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
          
    distance_scale
        The distance scale of the relevant local neighbourhoods. If supplied,
        `annotation_distance` is scaled such that its mean between different
        types has the same value as this `distance_scale`.
    annotation_distance_scale
        A scalar to facilitate conversion between distances in type-space and
        position-space. This parameter directly specifies the scaling factor of
        `annotation_distance` and overrides the `distance_scale` setting. If
        `None`, the bare annotation distances are used. If `None` and
        `distance_scale` is `None` and `annotation_distance` is a metric
        specification an exception is raised as position distance and
        annotation distance cannot be assumed to be comparable.
    coo_result
        Whether to return the result as :class:`~scipy.sparse.coo_matrix`
        instead of a :class:`~scipy.sparse.csr_matrix`. This is faster as it
        avoids conversion at the end, but if written to an `adata.obsp` key,
        the `adata` cannot be subsetted anymore... Ignored for dense
        distances.
    low_mem
        Whether to use memory optimization which run longer and may use the
        harddisc but have the potential to reduce the memory consumption by
        a factor of 2.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to on-the-fly distance
        calculation if necessary. Depending on `max_distance`, this goes to
        :func:`~tacco.utils.sparse_distance_matrix` or 
        :func:`~tacco.utils.dense_distance_matrix`.
        
    Returns
    -------
    Depending on `result_key` returns either a sparse or dense distance\
    matrix or an updated input `adata` containing the distance matrix under\
    `adata.obsp[result_key]`.
    
    """

    if base_distance_key is None or base_distance_key not in adata.obsp:
        positions = get.positions(adata, position_key)

    annotation_column = None
    if annotation_key is not None:
        annotation_column = adata.obs[annotation_key]
        if not hasattr(annotation_column, 'cat'):
            annotation_column = annotation_column.astype('category')

    if annotation_distance is None:
        full_data_distance = False
    elif isinstance(annotation_distance, str):
        full_data_distance = True
    elif annotation_distance_scale is None or np.any(annotation_distance * annotation_distance_scale <= max_distance):
        full_data_distance = True
    full_data_distance = full_data_distance or annotation_column is None
    full_data_distance = full_data_distance or base_distance_key is not None
    
    def dense_warning():
        if max_distance is None and verbose > 0:
            print(f'distance_matrix: `max_distance` is `None` which leads to a dense distance matrix being calculated which can be much slower and memory intensive than a sparse version. In case this is intended, one can silence this warning by setting `max_distance` to `np.inf`.')

    tempdir = None
    
    if full_data_distance:

        # have to get distances across the full dataset

        if base_distance_key is None or base_distance_key not in adata.obsp:
            if max_distance is None or max_distance == np.inf:
                dense_warning()
                distance = utils.dense_distance_matrix(positions.to_numpy(), **kw_args)
            else:
                distance = utils.sparse_distance_matrix(positions.to_numpy(), max_distance, low_mem=low_mem, verbose=verbose, **kw_args)
        else:
            distance = adata.obsp[base_distance_key]
            
        if base_distance_key is not None and base_distance_key not in adata.obsp:
            if scipy.sparse.issparse(distance) and not coo_result: # cannot implicitly return coo matrix as it collides with adata slicing...
                if low_mem:
                    distance = utils.coo_tocsr_inplace(distance) # use inplace operation to reduce memory footprint
                else:
                    distance = distance.tocsr()
                
            adata.obsp[base_distance_key] = distance
        
        if annotation_column is not None:

            annotation_categories = annotation_column.cat.categories

            if len(np.array(annotation_distance).shape) == 0 and pd.api.types.is_number(annotation_distance): # one distance for all

                annotation_distance = pd.DataFrame(annotation_distance, index=annotation_categories, columns=annotation_categories)
                for ac in annotation_categories:
                    annotation_distance.loc[ac,ac] = 0
            elif isinstance(annotation_distance, pd.DataFrame):
                annotation_distance = annotation_distance.reindex(columns=annotation_categories,index=annotation_categories)
            else:
                if distance_scale is None and annotation_distance_scale is None:
                    raise ValueError(f'`distance_scale` and `annotation_distance_scale` are `None`, but `annotation_distance` is interpreted as the `metric` argument for utils.cdist. In this case it is required to specify a conversion factor between distances in annotation and distances in position using `distance_scale` or `annotation_distance_scale`!')

                average_profiles = utils.get_average_profiles(annotation_key, adata)
                if average_profiles.shape[0] == 0:
                    raise ValueError(f'The average profiles from `adata` have length 0! Therefore no meaningful distances between them can be calculated. Either specify the `annotation_distance` by a number or a DataFrame or supply meaningful average profiles, e.g. by setting profiles directly in `adata.var["{annotation_key}"]` or by supplying expression data in `adata.X`.')
                average_profiles = average_profiles.reindex(columns=annotation_categories)
                
                annotation_distance = pd.DataFrame(
                    utils.cdist(average_profiles.T, average_profiles.T, metric=annotation_distance),
                    index=annotation_categories, columns=annotation_categories,
                )
                
            if annotation_distance_scale is not None:
                annotation_distance *= annotation_distance_scale
            elif distance_scale is not None:
                conversion_factor = distance_scale / annotation_distance.to_numpy()[np.triu_indices(annotation_distance.shape[0],1)].mean()
                if verbose > 0:
                    print(f'annotation_distance_scale from distance_scale {conversion_factor}')
                annotation_distance *= conversion_factor

            annotation_distance.replace(np.nan, np.inf, inplace=True)
            annotation_distance[annotation_distance < 0] = np.inf
            annotation_distance = annotation_distance.to_numpy()

            dummies = pd.get_dummies(annotation_column).to_numpy()

            if scipy.sparse.issparse(distance):
                
                distance = distance.tocoo()
                
                # add distances in squares
                distance.data *= distance.data
                annotation_distance *= annotation_distance
                dummies_ad2 = dummies @ annotation_distance
                utils.sparse_result_gemmT(dummies, dummies_ad2, distance, update_out=True)

                # symmetrize result as there can be numerical errors which break the symmetry and lead to very unsymmetric results if rounding error decides over >max_distance and <max_distance

                distance.data = np.sqrt(distance.data)

                strip_distance_matrix(distance, max_distance)
            
            else:
                
                # add distances in squares
                distance *= distance
                annotation_distance *= annotation_distance
                dummies_ad2 = dummies @ annotation_distance
                distance += utils.gemmT(dummies, dummies_ad2)

                distance.data = np.sqrt(distance.data)
        
    else:

        # split data and combine all distances

        nobs = adata.obs.shape[0]

        # mapping of index of subset to row number of the whole
        whole_row_dtype = utils.min_dtype(max(nobs, 2**30)) # get smallest possible index dtype - but at least int32, as scipy does not support less..
        whole_row = pd.Series(range(nobs), index=positions.index, dtype=whole_row_dtype)
        
        if max_distance is None or max_distance == np.inf:
            
            dense_warning()
            
            distance = None
            
            for anno, obs in adata.obs.groupby(annotation_column):

                _whole_row = whole_row[obs.index].to_numpy()
                _distance = utils.dense_distance_matrix(positions.iloc[_whole_row].to_numpy(), **kw_args)

                if distance is None: # now we know the correct datatype:
                    distance = np.zeros((nobs,nobs), dtype=_distance.dtype)
                
                _update_submatrix(distance, _distance, _whole_row, _whole_row)
        else:
            
            row = []
            col = []
            data = []

            if low_mem:
                tempdir = tempfile.TemporaryDirectory(prefix='temp_distance_matrix_',dir='.')
                buffer_directory = tempdir.name + '/'
            else:
                buffer_directory = None

            try:

                for anno, obs in adata.obs.groupby(annotation_column):

                    _whole_row = whole_row[obs.index].to_numpy()

                    _distance = utils.sparse_distance_matrix(positions.iloc[_whole_row].to_numpy(), max_distance, low_mem=low_mem, verbose=verbose, **kw_args)

                    if scipy.sparse.issparse(_distance):
                        if buffer_directory is None:
                            row.append(_whole_row[_distance.row])
                            col.append(_whole_row[_distance.col])
                            data.append(_distance.data)
                        else:
                            with open(buffer_directory+'row.bin', 'ab') as f:
                                _whole_row[_distance.row].tofile(f)
                            with open(buffer_directory+'col.bin', 'ab') as f:
                                _whole_row[_distance.col].tofile(f)
                            with open(buffer_directory+'data.bin', 'ab') as f:
                                _distance.data.tofile(f)
                    else:
                        data.append(_distance)

                # cleanup data as soon as we dont need it anymore
                distance_dtype = _distance.dtype
                del _distance
                del whole_row

                distance = scipy.sparse.coo_matrix((nobs,nobs))
                if buffer_directory is None:
                    distance.row = np.concatenate(row)
                    distance.col = np.concatenate(col)
                    distance.data = np.concatenate(data)
                else:
                    distance.row = np.fromfile(buffer_directory+'row.bin', dtype=whole_row_dtype)
                    distance.col = np.fromfile(buffer_directory+'col.bin', dtype=whole_row_dtype)
                    distance.data = np.fromfile(buffer_directory+'data.bin', dtype=distance_dtype)

            except:
                if low_mem:
                    tempdir.cleanup()
                raise
                
    if scipy.sparse.issparse(distance) and not coo_result: # cannot implicitly return coo matrix as it collides with adata slicing...
        if low_mem:
            buffer_directory = tempdir.name + '/' if tempdir is not None else None
            distance = utils.coo_tocsr_buffered(distance, buffer_directory=buffer_directory) # use buffered operation to reduce memory footprint
            if tempdir is not None:
                tempdir.cleanup()
        else:
            distance = distance.tocsr()
    
    if result_key is None:
        return distance
    else:
        adata.obsp[result_key] = distance
        return adata

def affinity(
    adata,
    sigma,
    distance_key=None,
    result_key=None,
):

    """\
    Calculates an affinity from a distance matrix.

    The affinity is calculated according to

        `aff(i,j) = exp(-dist(i,j)**2 / (2 * sigma**2))`

    where sigma gives the half width of the Gaussian weight. This can be used
    e.g. for spectral clustering.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData` with distance under `.obsp[distance_key]`
        or a (sparse) distance matrix.
    sigma
        The width parameter of the Gaussian.
    distance_key
        The `.obsp` key with the distances. Ignored if `adata` is the distance
        matrix.
    result_key
        The `.obsp` key to contain the affinities. If `None`, a
        :class:`~scipy.sparse.csr_matrix` containing the affinities is
        returned. Ignored if `adata` is the distance matrix.
        
    Returns
    -------
    Depending on `result_key` returns either the (sparse) affinity matrix or an\
    updated input `adata` contining the affinity matrix under\
    `adata.obsp[result_key]`.
    
    """

    if isinstance(adata, ad.AnnData):
        if distance_key is None:
            raise ValueError('`affinity_key` is required if `adata` is an `AnnData`!')
        affinity = adata.obsp[distance_key]
    else:
        affinity = adata
        result_key = None # the results are returned directly

    if result_key is None or result_key != distance_key:
        affinity = affinity.copy()

    if scipy.sparse.issparse(affinity):
        affinity.data = np.exp(- affinity.data ** 2 / (2. * sigma ** 2))
    else:
        affinity[:] = np.exp(- affinity ** 2 / (2. * sigma ** 2))

    if result_key is None:
        return affinity
    else:
        adata.obsp[result_key] = affinity
        return adata

def sparse_dummies(series):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if not hasattr(series, 'cat'):
        series = series.astype('category')
    
    dummies = scipy.sparse.coo_matrix((len(series),len(series.cat.categories)),dtype=np.int8)
    dummies.row = np.arange(len(series),dtype=utils.min_dtype(len(series)))
    dummies.col = series.cat.codes
    dummies.data = np.ones(len(series),dtype=np.int8)
    
    return dummies, series.cat.categories

def spectral_clustering(
    adata,
    max_size,
    min_size=None,
    affinity_key=None,
    result_key=None,
    dim=None,
    cut_threshold=0.7,
    position_key=['x','y'],
    position_scale=None,
    position_range=3,
    max_aspect_ratio=5,
    verbose=0,
):

    """\
    Performs spectral clustering on an affinity matrix.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData` with affinity under `.obsp[affinity_key]`
        or a (sparse) affinity matrix.
    max_size
        The clustering goes on until no cluster has more elements than this.
    min_size
        The clustering does not subcluster clusters smaller than this number.
        If `None`, uses `max_size/5`
    affinity_key
        The `.obsp` key with the affinities. Ignored if `adata` is the affinity
        matrix.
    result_key
        The `.obs` key to contain the clusters. If `None`, a
        :class:`~pandas.Series` containing the cluster labels is returned.
        Ignored if `adata` is the affinity matrix.
    dim
        The dimensionality of the manifold. If `None`, it is taken from
        supplied position space coordinates (if available) or being inferred on
        the fly. The `dim` is used to decide whether to subcluster a given
        cluster based on the surface to volume ratio.
    cut_threshold
        For every proposed subclustering a certain amount of affinity has to be
        cut. This number scales the decision threshold: higher values mean more
        cuts, lower values mean less cuts. The threshold itself scales also
        with the `(1/dim)`-th root of the cluster size.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. This is used to efficiently get small subproblems by
        spatial binning. If `position_key` or `position_scale` is `None`, do
        hirarchical clustering to iteratively split the problems in smaller
        subproblems. Ignored if `adata` is the affinity matrix.
    position_scale
        The expected feature size to use for splitting the problem spatially.
        If `position_key` or `position_scale` is `None`, do hirarchical
        clustering to iteratively split the problems in smaller subproblems.
        Ignored if `adata` is the affinity matrix.
    position_range
        A cluster is subclustered when it has a spatial size (defined as twice
        the standard deviation in the largest spatial PCA direction) of more
        than `position_scale*position_range*` and it is not subclustered if its
        spatial size is smaller than `position_scale/position_range`. Ignored
        if `adata` is the affinity matrix.
    max_aspect_ratio
        A cluster is subclustered when it has a larger aspect ratio (defined as
        the ratio of the standard deviations in the largest and smallest
        spatial PCA direction) of more than `max_aspect_ratio`. Ignored if
        `adata` is the affinity matrix.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
        
    Returns
    -------
    Depending on `result_key` returns either a :class:`~pandas.Series` with\
    the cluster labels or an updated input `adata` contining the cluster\
    labels under `adata.obs[result_key]`.
    
    """

    if max_size <= 1:
        raise ValueError('`max_size` is 1 (or less), which does not make sense for clustering!')
        
    if min_size is None:
        min_size = max_size / 5
    if min_size < 1:
        min_size = 1

    if isinstance(adata, ad.AnnData):
        if affinity_key is None:
            raise ValueError('`affinity_key` is required if `adata` is an `anndata.AnnData`!')
        affinity = adata.obsp[affinity_key]
        obs_index = adata.obs.index
    else:
        affinity = adata
        result_key = None # the results are returned directly
        obs_index = pd.RangeIndex(affinity.shape[0])
        
    if scipy.sparse.issparse(affinity) and not isinstance(affinity, (scipy.sparse.csr_matrix,scipy.sparse.csc_matrix)):
        affinity = affinity.tocsr() # spectral clustering wants high precision
    affinity = affinity.astype(np.float64) # spectral clustering wants high precision
    
    if position_key is None or not isinstance(adata, ad.AnnData):
        positions = None
    else:
        positions = get.positions(adata, position_key)
    
    if dim is None:
        # get dimensionality if possible
        if positions is not None:
            dim = positions.shape[1]
    # otherwise prepare to estimate it
    if dim is None:
        dim_data = []
    else:
        dim_data = None
    
    if position_scale is None: # do not use positions, if it is not specified how.
        positions = None

    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=affinity, directed=False, return_labels=True)
    new_components = pd.Series(labels, index=obs_index)

    max_n_clusters_per_level = 2
    min_n_clusters_per_level = 2

    def perform_clustering(n_clusters, affinity):
        if affinity.shape[0] == n_clusters:
            return np.arange(n_clusters)
        n_unique_labels = -1
        random_state=42
        while n_unique_labels != n_clusters: # This should loop only once. But in rare cases it does that multiple times: https://github.com/scikit-learn/scikit-learn/issues/18083
            labels = sklearn.cluster.SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels="discretize",
                eigen_solver="amg",
                random_state=random_state
            ).fit(affinity).labels_
            random_state += 1
            n_unique_labels = len(np.unique(labels))
        return labels

    # handle (possibly many) small components right away
    components = new_components
    components_sizes = components.value_counts()
    components_sizes = components_sizes[components_sizes <= min_size]
    component_cluster = components[components.isin(components_sizes.index)].astype('category').cat.codes
    cluster = pd.Series(-1, index=obs_index)
    cluster.loc[component_cluster.index] = component_cluster
    if len(component_cluster) > 0:
        cluster_i = component_cluster.max() + 1
    else:
        cluster_i = 0

    new_components.loc[component_cluster.index] = np.nan

    counter = 0
    last_remaining_points = -1
    last_remaining_clusters = -1
    last_dim = -1
    # iterate until there are no new components left
    while not new_components.isna().all():
        components = new_components

        components_sizes = components.value_counts()

        remaining_points = (~components.isna()).sum()
        remaining_clusters = len(components_sizes)
        if verbose > 0:
            if counter != 0:
                print('\n', end='')
            print(f'new component round with {remaining_points}/{len(components)} ({np.round(remaining_points/len(components)*100,1)}%) remaining points in {remaining_clusters} clusters of mean size {np.round(components_sizes.mean(),1)}', end='')
            last_message = 'comp'
        counter += 1
        if last_remaining_points == remaining_points and last_remaining_clusters == remaining_clusters and last_dim == dim:
            raise RuntimeError('The hirarchical clustering is caught in a loop without progress...')
        last_remaining_points = remaining_points
        last_remaining_clusters = remaining_clusters
        last_dim = dim
        
        surface_criterion_stats = {True:0,False:0}
        position_criterion_stats = {True:0,False:0,None:0}

        new_components = pd.Series(np.nan, index=components.index)
        new_component_i = 0

        last_message = ''
        for c, c_size in components_sizes.items():

            _selection = components==c
            aff = affinity[_selection][:,_selection]

            # Sometimes there are disconnected components in a single "component"... Probably due to an instability in the clustering.
            n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=aff, directed=False, return_labels=True)
            if n_components > 1:

                if verbose > 1:
                    if last_message == 'disco':
                        print(c_size, end=' ')
                    else:
                        print(f'\nsplitting disconnected component of size {c_size}', end=' ')
                    last_message = 'disco'

                labels += new_component_i
                new_components.loc[_selection.to_numpy()] = labels
                new_component_i += n_components
                continue

            approximate_cluster_number = c_size / max_size
            
            # check whether this cluster should be subclustered
            subclustering = None # `None` means undecided
            if c_size <= min_size:
                subclustering = False
            if c_size > max_size:
                subclustering = True
                
            if subclustering is None and positions is not None:
                _positions = positions[_selection]
                _X = _positions - np.mean(_positions, axis=0)
                if scipy.sparse.issparse(_X) and _X.shape[0] == _X.shape[1]:
                    _X = _X.A
                _sizes = scipy.linalg.svd(_X, full_matrices=False, compute_uv=False, check_finite=False)
                _sizes *= 2 / np.sqrt(c_size) # these are now twice the standard deviation, i.e. some kind of diameter
                _sizes.sort() # sorts ascending
                aspect_ratio = _sizes[-1] / _sizes[0]
                _largest_size = _sizes[-1]

                if _largest_size > position_scale * position_range:
                    subclustering = True
                elif _largest_size < position_scale / position_range:
                    subclustering = False
                elif aspect_ratio > max_aspect_ratio:
                    subclustering = True
                
                position_criterion_stats[subclustering] += 1
                
                if verbose > 2:
                    print('\nposition criteria',_largest_size,aspect_ratio, subclustering, end=' ')
                    last_message = 'pos'
                
            if subclustering == False:
                # This is already a good cluster

                cluster.loc[_selection.to_numpy()] = cluster_i
                cluster_i += 1
                
            elif subclustering is None and dim is None:
                # We need an estimate of the dimension before we go on with this size cluster, save that for later

                new_components.loc[_selection.to_numpy()] = new_component_i
                new_component_i += 1

            else:
                # Continue subclustering
                
                positon_splitted = False
                if positions is not None:
                    
                    _positions = positions[_selection]
                    
                    # find the direction with the biggest extent
                    maxs, mins = _positions.max(axis=0), _positions.min(axis=0)
                    extents = maxs - mins
                    dmax = extents.argmax()
                    extent = extents.iloc[dmax]
                    _min = mins.iloc[dmax]
                    _max = maxs.iloc[dmax]

                    boundary_layer_width = 4 * position_scale
                    if extent > 2 * boundary_layer_width: # otherwise it makes not much sense

                        coords = _positions.iloc[:,dmax]

                        mid = 0.5 * (_max + _min)

                        left_mid_right = pd.cut(coords,bins=[_min,mid-0.5*boundary_layer_width,mid+0.5*boundary_layer_width,_max], labels=['left','mid','right'])
                        vc = pd.Series(left_mid_right).value_counts()

                        n_points = len(coords)
                        merged = np.arange(n_points,dtype=utils.min_dtype(n_points+2))
                        merged[left_mid_right == 'left'] = n_points
                        merged[left_mid_right == 'right'] = n_points+1

                        dummies, cats = sparse_dummies(merged)
                        dummies = dummies.T.tocsr().astype(np.float64)
                        d_aff = _math.gemmT(dummies, aff, sparse_result=True)
                        d_aff_d = _math.gemmT(d_aff, dummies, sparse_result=True)

                        # allow for extra clusters in the boundary layer for two reasons:
                        # - it is set up already, so additional custers should be relatively cheap
                        # - often the boundary layer contains loosely coupled clusters, which makes it less likey to succed with the split into two big chunks
                        sol_n_clusters = min(max_n_clusters_per_level, 2 + int(0.2 * vc['mid']/max_size), c_size)
                        
                        if verbose > 1:
                            if last_message == 'spatial':
                                print(c_size, end=' ')
                            else:
                                print(f'\nspatial split of component of size {c_size}', end=' ')
                            last_message = 'spatial'
                        
                        split = perform_clustering(n_clusters=sol_n_clusters,affinity=d_aff_d)

                        labels = pd.Series(split,index=cats)[merged].to_numpy()
                        
                        positon_splitted = True

                if not positon_splitted:

                    sol_n_clusters = min(max_n_clusters_per_level, max(int(approximate_cluster_number - 2),2), c_size)
                    
                    if verbose > 1:
                        if last_message == 'reclustering':
                            print(c_size, end=' ')
                        else:
                            print(f'\nreclustering of component of size {c_size}', end=' ')
                        last_message = 'reclustering'
                    
                    labels = perform_clustering(sol_n_clusters, aff)
                
                # get data about the proposed split
                pre_sum = aff.sum()
                if pre_sum <= 0:
                    raise ValueError('Got cluster without connectons! Something must be wrong with the supplied affinity!')
                post_sum = 0.0
                for l in [0,1]:
                    _sel = labels == l
                    _aff = aff[_sel][:,_sel]
                    post_sum += _aff.sum()
                delta = pre_sum - post_sum
                
                if dim_data is not None:
                    dim_data.append((counter,delta,pre_sum,c_size))
                
                if subclustering is None: # if still undecided
                    if dim is not None:
                        rdelta = delta / pre_sum # (cut affinity)/affinity
                        # (cut affinity)/affinity for a cut through a c_size=N**(dim-1)*N cartesian hyperblock with nearest neighbour affinity of 1:
                        # affinity for a N**dim cube: Ac = dim*(N-1)*N**(dim-1)
                        # affinity for a N**(dim-1)*N block: Ab = 2*Ac + N**(dim-1)
                        # (cut affinity)/affinity = 1 / (1 + 2 * dim * (N - 1))
                        cdelta = 1 / (1 + 2 * dim * ((0.5 * c_size)**(1/dim) - 1))
                        sdelta = rdelta / cdelta
                        if sdelta > cut_threshold:
                            subclustering = False
                        else:
                            subclustering = True
                        surface_criterion_stats[subclustering] += 1
                        if verbose > 2:
                            print('\nsurface criteria',sdelta, subclustering, end=' ')
                            last_message = 'surf'
                    else: # if we have no idea about the dimension we just go on subclustering...
                        subclustering = True

                if subclustering:
                    # accept the new clusters
                    labels += new_component_i
                    new_components.loc[_selection.to_numpy()] = labels
                    new_component_i += sol_n_clusters
                else:
                    # reject the new clusters and add the current cluster to the finished clusters
                    cluster.loc[_selection.to_numpy()] = cluster_i
                    cluster_i += 1
        
        if verbose > 1:
            if sum(surface_criterion_stats.values()) > 0:
                print(f'\nsurface criteria stats: {surface_criterion_stats[True]}/{sum(surface_criterion_stats.values())}={np.round(surface_criterion_stats[True]/sum(surface_criterion_stats.values())*100,1)}% have been splitted', end=' ')
            if sum(position_criterion_stats.values()) > 0:
                print(f'\nposition criteria stats: {position_criterion_stats[True]}/{sum(position_criterion_stats.values())}={np.round(position_criterion_stats[True]/sum(position_criterion_stats.values())*100,1)}% have been splitted', end=' ')
            last_message = 'stats'
        
        if dim_data is not None:
            enough_data = len(dim_data) > 50
            need_dim_urgently = components_sizes.max() <= max_size
            if enough_data or need_dim_urgently: # only dare to estimate the dimension if some data is available.
                if not enough_data and need_dim_urgently:
                    if verbose > 0:
                        print('\nnot enough_data and need_dim_urgently',end=' ')
                        last_message = 'dimension'

                dimdf = pd.DataFrame(dim_data,columns=['round','delta','sigma','N'])
                # calculate the effecive dimension for every recorded cut from the scaling of surface VS volume terms
                dimdf['dim'] = np.log(dimdf['N'])/np.log(dimdf['sigma']/dimdf['delta'])

                # fit the effective dimension in windows of log(cluster_size) and report the intercept:
                # Small clusters are cut hard through the volume instead of at loose connections of subclusters
                dr = 0.5 # half window size
                q = np.log(dimdf['N'].to_numpy())/np.log(10)
                d = dimdf['dim'].to_numpy()
                fits = [2.0] # better than nothing... and stabilizes the estimate.
                if len(q) > 0:
                    for r in np.arange(q.max()-dr,q.min()+dr,-0.1):
                        sub = (q<=r+dr)&(q>r-dr)
                        X = q[sub]
                        if len(X) > 1:
                            y = d[sub]
                            _dim = sklearn.linear_model.LinearRegression().fit(X.reshape(-1,1), y).intercept_
                            fits.append(_dim)

                # final dimension is the average of all these intercepts:
                # There is some systematic variation in the result wrt the cluster size e.g. as small clusters show e.g. noise and 3D structure while larger clusters reveal properties of the effective manifold but have less measurements and more cuts through loose connections.
                dim = np.mean(fits)
                if verbose > 0:
                    print(f'\nestimated dimension is {dim} ; len(fits)={len(fits)} ; len(dim_data)={len(dim_data)}',end=' ')
                    last_message = 'dimension'

    if verbose > 0:
        print()
    
    # normalize the cluster numbers such that the n-th occuring cluster has number n
    cluster = cluster.astype('category')
    cluster = pd.Series(np.arange(len(cluster.cat.categories)),index=cluster.cat.codes.unique())[cluster].to_numpy()
    cluster = pd.Series(cluster, index=obs_index).astype('category')

    if result_key is None:
        return cluster
    else:
        adata.obs[result_key] = cluster
        return adata

@njit(fastmath=True,cache=True)
def _distribute_molecules_sparse(coords, X_indptr, X_indices, X_data, genes, index, positions):
    n_dim = coords.shape[1]
    n_obs = len(X_indptr)-1
    
    mol = 0
    for obs in range(n_obs):
        center = coords[obs]
        ptr0 = X_indptr[obs]
        ptr1 = X_indptr[obs+1]
        indices = X_indices[ptr0:ptr1]
        data = X_data[ptr0:ptr1]
        for i in range(len(indices)):
            gene = indices[i]
            count = int(data[i])
            genes[mol:(mol+count)] = gene
            index[mol:(mol+count)] = obs
            positions[mol:(mol+count)] = center
            mol += count

@njit(fastmath=True,cache=True)
def _distribute_molecules_dense(coords, X, genes, index, positions):
    n_dim = coords.shape[1]
    n_obs, n_gene = X.shape
    
    mol = 0
    for obs in range(n_obs):
        center = coords[obs]
        row = X[obs]
        for gene in range(n_gene):
            count = int(row[gene])
            if count > 0:
                genes[mol:(mol+count)] = gene
                index[mol:(mol+count)] = obs
                positions[mol:(mol+count)] = center
                mol += count

def distribute_molecules(
    adata,
    width,
    position_key=('x','y'),
    obs_index_key=None,
    var_index_key=None,
    verbose=1,
    seed=42,
):
    """\
    Distributes the counts of observations randomly in space.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with counts in `.X`.
    width
        The width of the Gaussian to use for randomly placing molecules around
        the central position.
    position_key
        The `.obsm` key or a tuple of `.obs` keys with the position space
        coordinates.
    obs_index_key
        A string specifying the name of the column to write the old
        `.obs.index` (i.e. the cell names) to. If `None`, tries to guess a
        reasonable name.
    var_index_key
        A string specifying the name of the column to write the old
        `.var.index` (i.e. the gene names) to. If `None`, tries to guess a
        reasonable name.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    seed
        The seed for the randomness.
        
    Returns
    -------
    A :class:`~pandas.DataFrame` with a single molecule per row and with\
    scattered position space coordinates.
    
    """
    
    obs_index_key = helper.guess_obs_index_key(obs_index_key, adata)
    var_index_key = helper.guess_var_index_key(var_index_key, adata)
    
    preprocessing.check_counts_validity(adata.X)
    
    coords = get.positions(adata, position_key)
    coords_names = coords.columns
    coords = coords.to_numpy()
    
    n_dim = coords.shape[1]
    
    n_obs, n_gene = adata.shape
    
    n_mol = int(adata.X.astype(int).sum())
    
    pos_dtype = coords.dtype
    geneid_dtype = utils.min_dtype(n_gene)
    obsid_dtype = utils.min_dtype(n_obs)
    
    positions = np.empty((n_mol,n_dim), dtype=pos_dtype)
    genes = np.empty(n_mol, dtype=geneid_dtype)
    index = np.empty(n_mol, dtype=obsid_dtype)
    
    if scipy.sparse.issparse(adata.X):
        if isinstance(adata.X, scipy.sparse.csr_matrix):
            X = adata.X
        else:
            if verbose > 0:
                print(f'`adata.X` is a sparse matrix but not in csr format, and has to be converted first which takes time. To avoid that convert it before.')
            X = adata.X.tocsr()
        _distribute_molecules_sparse(coords, X.indptr, X.indices, X.data, genes, index, positions)
    else:
        X = adata.X
        _distribute_molecules_dense(coords, X, genes, index, positions)
    
    random_dtype = np.float32 if positions.dtype == np.float32 else np.float64
    np.random.seed(seed)
    positions = positions + width * np.random.normal(size=positions.shape).astype(random_dtype)
    
    molecules = pd.DataFrame(positions, columns=coords_names)
    cat_var_index = adata.var.index.astype('category')
    cat_obs_index = adata.obs.index.astype('category')
    molecules[var_index_key] = cat_var_index[genes]
    molecules[obs_index_key] = cat_obs_index[index]
    
    return molecules
