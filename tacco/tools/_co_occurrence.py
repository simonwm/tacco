import numpy as np
import pandas as pd
import scipy.sparse
from numba import njit, prange, get_num_threads
import warnings
import ast
from statsmodels.stats.multitest import multipletests
from ._points import distance_matrix
from ._enrichments import enrichments
from .. import utils
from .. import get

@njit(cache=True)
def _make_intervals(delta_distance, max_distance):
    intervals_list = []
    distance = max_distance
    while distance > 0:
        intervals_list.append(distance)
        distance -= delta_distance
    intervals_list.append(max_distance * 0)
    intervals = np.array(intervals_list)[::-1]
    return intervals

@njit(cache=True)
def _reshape_counts(counts):
    # reorder from memory access efficient order to squidpy compatible order...
    n_codes,n_intervals,n_reference_codes = counts.shape
    reshaped = np.empty(shape=(n_codes,n_reference_codes,n_intervals),dtype=counts.dtype)
    for i in prange(n_codes):
        for j in range(n_intervals):
            for k in range(n_reference_codes):
                reshaped[i,k,j] = counts[i,j,k]
    return reshaped
    
@njit(cache=True, parallel=True, fastmath=True)
def _count_co_occurences_sparse(dist_indptr, dist_indices, dist_data, codes, n_codes, reference_codes, n_reference_codes, delta_distance, max_distance, min_distance, weights, reference_weights, n_threads):
    
    intervals = _make_intervals(delta_distance, max_distance)
    n_intervals = len(intervals)-1
    counts = np.zeros((n_codes,n_intervals,n_reference_codes),dtype=np.float64)
    
    n = len(dist_indptr) - 1
    counts_per_thread = np.zeros((n_threads,n_codes,n_intervals,n_reference_codes),dtype=np.float64)
    # parallel per-thread accumulation
    n_per_thread = (n+n_threads-1) // n_threads
    for thread in prange(n_threads):
        istart = thread * n_per_thread
        iend = (thread+1) * n_per_thread
        iend = min(iend,n)
        for i in range(istart, iend):
            weights_i = weights[i]
            ptr0 = dist_indptr[i]
            ptr1 = dist_indptr[i+1]
            for ptr in range(ptr0,ptr1):
                j = dist_indices[ptr]
                dist = dist_data[ptr]
                if dist > 0 and dist >= min_distance:
                    k = n_intervals - 1 - int(np.floor((max_distance - dist) / delta_distance))
                    if k >= 0 and k < n_intervals:
                        counts_per_thread[thread,codes[i], k, reference_codes[j]] += weights_i * reference_weights[j]
    # serial aggregation across threads
    for thread in range(1,n_threads):
        counts_per_thread[0] += counts_per_thread[thread]
    
    return _reshape_counts(counts_per_thread[0]), intervals

@njit(cache=True, parallel=True, fastmath=True)
def _count_co_occurences_dense(dist_dense, codes, n_codes, reference_codes, n_reference_codes, delta_distance, max_distance, min_distance, weights, reference_weights, n_threads):
    
    intervals = _make_intervals(delta_distance, max_distance)
    n_intervals = len(intervals)-1
    
    n = dist_dense.shape[0]
    counts_per_thread = np.zeros((n_threads,n_codes,n_intervals,n_reference_codes),dtype=np.float64)
    # parallel per-thread accumulation
    n_per_thread = (n+n_threads-1) // n_threads
    for thread in prange(n_threads):
        istart = thread * n_per_thread
        iend = (thread+1) * n_per_thread
        iend = min(iend,n)
        for i in range(istart, iend):
            weights_i = weights[i]
            for j in range(n):
                dist = dist_dense[i,j]
                if dist > 0 and dist >= min_distance:
                    k = n_intervals - 1 - int(np.floor((max_distance - dist) / delta_distance))
                    if k >= 0 and k < n_intervals:
                        counts_per_thread[thread,codes[i], k, reference_codes[j]] += weights_i * reference_weights[j]
    # serial aggregation across threads
    for thread in range(1,n_threads):
        counts_per_thread[0] += counts_per_thread[thread]
    
    return _reshape_counts(counts_per_thread[0]), intervals
    
@njit(cache=True, parallel=True, fastmath=True)
def _count_soft_co_occurences_sparse(dist_indptr, dist_indices, dist_data, contributions, reference_contributions, delta_distance, max_distance, min_distance, weights, reference_weights, n_threads):
    
    n_codes = contributions.shape[1]
    n_reference_codes = reference_contributions.shape[1]
    
    intervals = _make_intervals(delta_distance, max_distance)
    n_intervals = len(intervals)-1
    
    n = len(dist_indptr) - 1
    counts_per_thread = np.zeros((n_threads,n_codes,n_intervals,n_reference_codes),dtype=np.float64)
    # parallel per-thread accumulation
    n_per_thread = (n+n_threads-1) // n_threads
    for thread in prange(n_threads):
        istart = thread * n_per_thread
        iend = (thread+1) * n_per_thread
        iend = min(iend,n)
        for i in range(istart, iend):
            weights_i = weights[i]
            contributions_i = contributions[i,:] * weights_i
            temp_i = np.zeros((n_intervals,n_reference_codes),dtype=np.float64)
            ptr0 = dist_indptr[i]
            ptr1 = dist_indptr[i+1]
            for ptr in range(ptr0,ptr1):
                j = dist_indices[ptr]
                dist = dist_data[ptr]
                if dist > 0 and dist >= min_distance:
                    k = n_intervals - 1 - int(np.floor((max_distance - dist) / delta_distance))
                    if k >= 0 and k < n_intervals:
                        reference_contributions_j = reference_weights[j] * reference_contributions[j,:]
                        for rc in range(n_reference_codes):
                            temp_i[k, rc] += reference_contributions_j[rc]
            for k in range(n_intervals):
                for ac in range(n_codes):
                    for rc in range(n_reference_codes):
                        counts_per_thread[thread,ac,k,rc] += contributions_i[ac] * temp_i[k,rc]
    # serial aggregation across threads
    for thread in range(1,n_threads):
        counts_per_thread[0] += counts_per_thread[thread]

    return _reshape_counts(counts_per_thread[0]), intervals

@njit(cache=True, parallel=True, fastmath=True)
def _count_soft_co_occurences_dense(dist_dense, contributions, reference_contributions, delta_distance, max_distance, min_distance, weights, reference_weights, n_threads):
    
    n_codes = contributions.shape[1]
    n_reference_codes = reference_contributions.shape[1]
    
    intervals = _make_intervals(delta_distance, max_distance)
    n_intervals = len(intervals)-1
    
    n = dist_dense.shape[0]
    counts_per_thread = np.zeros((n_threads,n_codes,n_intervals,n_reference_codes),dtype=np.float64)
    # parallel per-thread accumulation
    n_per_thread = (n+n_threads-1) // n_threads
    for thread in prange(n_threads):
        istart = thread * n_per_thread
        iend = (thread+1) * n_per_thread
        iend = min(iend,n)
        for i in range(istart, iend):
            weights_i = weights[i]
            contributions_i = contributions[i,:] * weights_i
            temp_i = np.zeros((n_intervals,n_reference_codes),dtype=np.float64)
            for j in range(n):
                dist = dist_dense[i,j]
                if dist > 0 and dist >= min_distance:
                    k = n_intervals - 1 - int(np.floor((max_distance - dist) / delta_distance))
                    if k >= 0 and k < n_intervals:
                        reference_contributions_j = reference_weights[j] * reference_contributions[j,:]
                        for rc in range(n_reference_codes):
                            temp_i[k, rc] += reference_contributions_j[rc]
            for k in range(n_intervals):
                for ac in range(n_codes):
                    for rc in range(n_reference_codes):
                        counts_per_thread[thread,ac,k,rc] += contributions_i[ac] * temp_i[k,rc]
    # serial aggregation across threads
    for thread in range(1,n_threads):
        counts_per_thread[0] += counts_per_thread[thread]
    
    return _reshape_counts(counts_per_thread[0]), intervals

def co_occurrence(
    adata,
    annotation_key,
    center_key=None,
    sample_key=None,
    distance_key=None,
    position_key=('x','y'),
    result_key=None,
    max_distance=None,
    sparse=True,
    min_distance=None,
    delta_distance=None,
    reads=True,
    counts_location=None,
    p_corr='fdr_bh',
    n_permutation=0,
    n_boot=0,
    position_split=4,
    seed=42,
    verbose=1,
    **kw_args,
):
    
    """\
    Calculates a spatial co-occurence score given by the conditional
    probability to find an annotation `a` at some distance `d` from an
    observation with annotation `b`, normalized by the probability to find an
    annotation `a` at distance `d` from an observation disregarding the value
    of its annotation: `p(a|bd)/p(a|d)`.
    This is a more general, more accurate, and faster alternative to the
    function of the same name in squidpy :func:`~sq.gr.co_occurrence`. For
    `center_key==None` the result is compatible with the corresponding squidpy
    code. The result is not identical to squidpy, as the parametrization and
    heuristics are different. 
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`
    annotation_key
        The `.obs` or `.obsm` key for the annotation `a` in `p(a|bd)/p(a|d)`.
    center_key
        The `.obs` or `.obsm` key for the annotation `b` in `p(a|bd)/p(a|d)`.
        If `None`, takes the `annotation_key`.
    sample_key
        A categorical sample annotation. The result from different samples is
        averaged for the final result and their standard deviation gives an
        estimation of the error. If `None`, all observations are assumed to be
        on the same sample.
    distance_key
        The `.obsp` key containing a precomputed distance matrix to use. If
        `None`, the distances are computed on the fly with the positions found
        in `position_key`. Otherwise `position_key` is ignored.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates
    result_key
        The `.uns` key to contain the result. If `None`, the result is returned
        as a dictionary containing the keys:
        
        - "occ": mean over samples of the `p(a|bd)/p(a|d)` scores as a
          :class:`~numpy.ndarray` with dimensions according to `a`, `b`, and
          `d`,
        - "log_occ": like "occ", but with the sample mean taken over the
          logarithms of the scores,
        - "p_t": p-values for "log_occ", see `p_corr`,
        - "z": z_scores, see `n_permutation`,
        - "composition": like "occ", but with the sample mean taken over
          `p(a|bd)`,
        - "log_composition": like "occ", but with the sample mean taken over
          `log(p(a|bd))`,
        - "sample_counts": the neighbourship counts per sample as a
          :class:`~numpy.ndarray` with dimensions according to samples, `a`,
          `b`, and `d`, see also `n_boot`,
        - "permutation_counts": the neighbourship counts per permutation sample
          as a :class:`~numpy.ndarray` with dimensions according to permutation
          samples, `a`, `b`, and `d`, see also `n_permutation`,
        - "interval": the boundaries of the distance bins,
        - "annotation": containing the order of the `a` annotations,
        - "center": containing the order of the `b` annotations,
        - "n_boot": the number of added bootstrap samples, see `n_permutation`.
    max_distance
        The maximum distance to use. If `None` or `np.inf`, uses the maximum
        distance in the data (if there are multiple samples, then only the
        first sample is used). If the distance matrix is not precomputed (see
        `distance_key`), `None` and `np.inf` result in dense distance
        computation (which can be infeasible for larger datasets).
    sparse
        Whether to calculate a sparse or dense distance matrix, if it is not
        precomputed. If `None`, this is determined by the value of
        `max_distance`.
    min_distance
        The minimum distance to use. If `None`, uses a heuristic to find a
        sensible low distance cutoff which excludes distances with deviations
        from uniform distribution (e.g. cell-size effects).
    delta_distance
        The width in distance for distance discretization. If `None`, takes
        `max_distance/100`.
    reads
        Whether to weight the co-occurence counts with the counts of the two
        participating observations.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tc.get.counts`. This is only rlelevant if `reads==True`.
    p_corr
        The name of the p-value correction method to use for calculating the
        significance of the deviation from random distribution. The
        significance is calculated over the samples with "1-sample"
        t-test and only available if the co-occurrence was calculated for many
        samples. Possible values are the ones available in
        :func:`~statsmodels.stats.multitest.multipletests`. If `None`, only
        uncorrected p-values are used.
    n_permutation
        The number of permutation samples to generate with randomly permuted
        annotations at fixed centers. This is used only for the calculation of
        the z-score. If `0`, the z-score is not calculated.
    n_boot
        The number of bootstrap samples to add to the real samples. To account
        for spatial correlations, the bootstrap samples are generated by
        spatially blocking the data per sample and resampling the spatial
        blocks per sample.
    position_split
        The number of splits per spatial dimension to use for the bootstrap
        resampling. Can be a tuple with the spatial dimension as length to
        assign a different split per dimension.
    seed
        A random seed for the bootstrapping. See `n_boot`.
    verbose
        Level of verbosity, with `0` (not output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to on-the-fly distance
        calculation with :func:`~tacco.utils.distance_matrix` if necessary.
        
    Returns
    -------
    Depending on `result_key` returns either the updated input `adata` or the\
    result directly in the format described under `result_key`.
    
    """
    
    isna = np.full(shape=len(adata.obs.index),fill_value=False)
    if annotation_key in adata.obs:
        labels = adata.obs[annotation_key]
        isna = isna | labels.isna()
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        if not hasattr(labels, 'cat'):
            labels = labels.astype('category')
    elif annotation_key in adata.obsm:
        labels = adata.obsm[annotation_key].copy()
        labels = labels / labels.sum(axis=1).to_numpy()[:,None] # normalize the annotation - just to be on the safe side
        isna = isna | labels.isna().any(axis=1)
    else:
        raise ValueError(f'`annotation_key` {annotation_key} was not found in `adata.obs` or `adata.obsm`!')
    
    if center_key is None:
        reference_labels = labels
        center_key = annotation_key
    else:
        if center_key in adata.obs:
            reference_labels = adata.obs[center_key]
            isna = isna | reference_labels.isna()
            if not isinstance(reference_labels, pd.Series):
                reference_labels = pd.Series(reference_labels)
            if not hasattr(reference_labels, 'cat'):
                reference_labels = reference_labels.astype('category')
            if len(labels.shape) == 2: # we have to work with soft annotation
                reference_labels = pd.get_dummies(reference_labels)
        elif center_key in adata.obsm:
            reference_labels = adata.obsm[center_key].copy()
            reference_labels = reference_labels / reference_labels.sum(axis=1).to_numpy()[:,None] # normalize the annotation - just to be on the safe side
            isna = isna | reference_labels.isna().any(axis=1)
            if len(labels.shape) == 1: # we have to work with soft annotation
                labels = pd.get_dummies(labels)
        else:
            raise ValueError(f'`center_key` {center_key} was not found in `adata.obs` or `adata.obsm`!')

    original_adata = adata # keep reference to original adata to write result to later
    adata = get.counts(adata, counts_location=counts_location, annotation=True)
    if isna.any(): # remove nans
        adata = adata[~isna]
        
    split = None
    if n_boot > 0:
        split = utils.spatial_split(adata, position_key=position_key, position_split=position_split, sample_key=sample_key, min_obs=0, result_key=None)
    
    if sample_key is not None:
        sample_adatas = { sample: adata[df.index] for sample, df in adata.obs.groupby(sample_key) }
        samples = list(sample_adatas.keys())
        sample_adatas = list(sample_adatas.values())
        sample_labels = [ labels.loc[_adata.obs.index] for _adata in sample_adatas ]
        sample_reference_labels = [ reference_labels.loc[_adata.obs.index] for _adata in sample_adatas ]
    else:
        samples = ['all']
        sample_adatas = [ adata ]
        sample_labels = [ labels ]
        sample_reference_labels = [ reference_labels ]
    
    if distance_key is None:
        if verbose > 0:
            print('co_occurrence: The argument `distance_key` is `None`, meaning that the distance which is now calculated on the fly will not be saved. Providing a precalculated distance saves time in multiple calls to this function.')
    elif distance_key not in adata.obsp:
        raise ValueError(f'The argument `distance_key` is "{distance_key}", but there is no `adata.obsp["{distance_key}"]`!')
    
    _max_distance = max_distance # buffer original max_distance value to trigger the correct path (dense/sparse) for sample number 2+
    if not sparse: # force dense calculation if explicitly asked for
        _max_distance = np.inf
    def _get_dist(si):
        if distance_key is not None:
            if adata is sample_adatas[0]:
                dist = adata.obsp[distance_key]
            else:
                sample_index = adata.obs.index.isin(sample_adatas[si].obs.index)
                dist = adata.obsp[distance_key][sample_index][:,sample_index]
        else:
            if verbose > 0:
                print(f'calculating distance for sample {si+1}/{len(samples)}')
            dist = distance_matrix(sample_adatas[si], max_distance=_max_distance, position_key=position_key, verbose=verbose, **kw_args)
        return dist
    dist = _get_dist(0)

    if max_distance is None or max_distance == np.inf:
        if scipy.sparse.issparse(dist):
            max_distance = dist.data.max()
        else:
            max_distance = dist.max()
    max_distance = float(max_distance)
    
    if delta_distance is None:
        delta_distance = max_distance / 100
    n_intervals = int(max_distance / delta_distance)
    
    intervals = None
    sample_counts = []
    permutation_counts = None
    for si,sn in enumerate(samples): # do not parallelize over samples as the memory for a single sample distance matrix can be already quite large

        if reads:
            weights = utils.get_sum(sample_adatas[si].X, axis=1).astype(float)
        else:
            weights = np.ones(len(sample_adatas[si].obs.index),dtype=float)
        reference_weights = weights.copy()
        
        common_args = [delta_distance, max_distance, (0 if min_distance is None else min_distance), weights, reference_weights, get_num_threads()]
        
        if si > 0: # already have the first sample distance from above
            dist = _get_dist(si)

        if scipy.sparse.issparse(dist):
            if not isinstance(dist, scipy.sparse.csr_matrix):
                if verbose > 0:
                    print(f'co_occurrence: distance is not in csr format and has to be transformed to csr on the fly')
                dist = scipy.sparse.csr_matrix(dist)
            dist_args = [dist.indptr, dist.indices, dist.data]
        else:
            dist_args = [dist]
        
        if len(labels.shape) == 1: # categorical annotation

            codes = sample_labels[si].cat.codes.to_numpy().copy()
            n_codes = len(sample_labels[si].cat.categories)
            reference_codes = sample_reference_labels[si].cat.codes.to_numpy().copy()
            n_reference_codes = len(sample_reference_labels[si].cat.categories)
            
            anno_args = [codes, n_codes, reference_codes, n_reference_codes]
            
            if scipy.sparse.issparse(dist):
                _count_co_occurences = _count_co_occurences_sparse
            else:
                _count_co_occurences = _count_co_occurences_dense
        
        else: # soft annotation
        
            contributions = sample_labels[si].to_numpy()
            reference_contributions = sample_reference_labels[si].to_numpy()
            
            anno_args = [contributions, reference_contributions]
            
            if scipy.sparse.issparse(dist):
                _count_co_occurences = _count_soft_co_occurences_sparse
            else:
                _count_co_occurences = _count_soft_co_occurences_dense
        
        counts, _intervals = _count_co_occurences(*dist_args, *anno_args, *common_args)
        
        sample_counts.append(counts)
        
        # add bootstrap samples
        if n_boot > 0:
            np.random.seed(seed + si)
            base_weights = weights.copy()
            sample_split = split[sample_adatas[si].obs.index]
            sample_splits = sample_split.unique()
            for i_boot in range(n_boot):
                
                boot_splits = np.random.choice(sample_splits, size=len(sample_splits), replace=True)
                reweights = pd.Series(boot_splits).value_counts().reindex(sample_splits).fillna(0)
                reweights = np.sqrt(reweights) # sqrt necessary to mimic bootstrap behaviour as the weights are used as products in the co-occurrence counts
                
                weights[:] = base_weights * sample_split.map(reweights).to_numpy() # update the weights in place to update common_args along
                reference_weights[:] = weights
                
                counts, _intervals = _count_co_occurences(*dist_args, *anno_args, *common_args)
                
                sample_counts.append(counts)
            # reproduce initial state
            weights[:] = base_weights
            reference_weights[:] = base_weights
        
        # add permuted samples
        if n_permutation > 0:
            if permutation_counts is None:
                permutation_counts = []
            np.random.seed(seed + si)
            i_perm = 0
            for _ in range(n_permutation):
                permutation = np.arange(len(weights))
                np.random.shuffle(permutation)
                # update in place to update anno_args along
                weights[:] = weights[permutation]
                if len(labels.shape) == 1: # categorical annotation
                    codes[:] = codes[permutation]
                else: # soft annotation
                    contributions[:] = contributions[permutation]
                
                counts, _intervals = _count_co_occurences(*dist_args, *anno_args, *common_args)
                
                permutation_counts.append(counts)
                i_perm = i_perm + 1
                if i_perm == n_permutation:
                    break
                # if we have a symmetric case, use it to decrease computational load - even though this results in half permutation statistics in the diagonal...
                if annotation_key == center_key:
                    permutation_counts.append(counts.transpose(1,0,2).copy())
                    i_perm = i_perm + 1
                if i_perm == n_permutation:
                    break

            # reproduction of initial state not necessary as a new sample comes next
        
        if intervals is None:
            intervals = _intervals
        else:
            assert((intervals == _intervals).all())
    
    if min_distance is None:
        total_counts = np.sum(sample_counts,axis=0)
        total_pairs_per_interval = total_counts.sum(axis=(0,1))
        integrated_total_pairs = np.cumsum(total_pairs_per_interval)
        densities = integrated_total_pairs / (intervals[1:]**2 * np.pi)
        offset = max(1,n_intervals // 4)
        density_ratios = densities[:-offset] / densities[offset:]
        threshold = min(density_ratios.max() * 0.9, 1.0)
        start_index = np.argwhere(density_ratios > threshold)[0].flatten()[0]
    else:
        start_index = np.argwhere(intervals >= min_distance)[0].flatten()[0]
        
    intervals = intervals[start_index:]
    
    pseudo_count = 1 # add pseudo count to avoid division by 0 and log of 0 issues
    
    sample_counts = np.array(sample_counts)
    sample_counts = sample_counts[...,start_index:]
    sample_counts += pseudo_count
    
    if permutation_counts is not None:
        permutation_counts = np.array(permutation_counts)
        permutation_counts = permutation_counts[...,start_index:]
        permutation_counts += pseudo_count

    # score is conditional probability p(i|j) scaled by p(i)
    with warnings.catch_warnings(): # infinities resulting from division by zeros are wanted here, so ignore associated warnings
        warnings.simplefilter("ignore")

        p_i_j = sample_counts / sample_counts.sum(axis=1)[:,None,:,:]
        p_i = sample_counts.sum(axis=2)[:,:,None,:] / sample_counts.sum(axis=(1,2))[:,None,None,:]

        sample_scores = p_i_j / p_i
        # equivalent alternative:
        # sample_scores = sample_counts * sample_counts.sum(axis=(1,2))[:,None,None,:] / (sample_counts.sum(axis=1)[:,None,:,:] * sample_counts.sum(axis=2)[:,:,None,:])

        # alternative score, similar to "z-scores": (value - random expectation) / standard deviation
        if permutation_counts is not None:
            sample_log_counts = np.log(sample_counts)
            permutation_log_counts = np.log(permutation_counts)
            random_expected_log_counts = np.mean(permutation_log_counts,axis=0)
            #std_log_counts = np.sqrt(np.var(sample_log_counts,axis=0) + np.var(permutation_log_counts,axis=0))
            std_log_counts = np.std(permutation_log_counts,axis=0)
            sample_z_scores = (sample_log_counts - random_expected_log_counts) / std_log_counts
        #else:
        #    random_expected_counts = (sample_counts.sum(axis=1)[:,None,:,:] * sample_counts.sum(axis=2)[:,:,None,:]) / sample_counts.sum(axis=(1,2))[:,None,None,:]
        #    random_expected_log_counts = np.log(random_expected_counts)
        #    std_log_counts = np.std(sample_log_counts,axis=0)
        #    sample_z_scores = (sample_log_counts - random_expected_log_counts) / std_log_counts
        
        # plain neighbourhood composition
        sample_composition = p_i_j
        
        # distance distribution
        sample_distance_distribution = sample_counts / sample_counts.sum(axis=3)[:,:,:,None]
        
        # relative distance distribution
        sample_relative_distance_distribution = sample_counts * sample_counts.sum(axis=(1,3))[:,None,:,None] / (sample_counts.sum(axis=3)[:,:,:,None] * sample_counts.sum(axis=1)[:,None,:,:])
    
    sample_counts -= pseudo_count # remove pseudo count
    if permutation_counts is not None:
        permutation_counts -= pseudo_count # remove pseudo count
    
    mean_scores = sample_scores.mean(axis=0)
    sample_scores = np.log(sample_scores)
    mean_log_scores = sample_scores.mean(axis=0)
    mean_z_scores = sample_z_scores.mean(axis=0) if permutation_counts is not None else None
    mean_composition = sample_composition.mean(axis=0)
    sample_composition = np.log(sample_composition)
    mean_log_composition = sample_composition.mean(axis=0)
    mean_distance_distribution = sample_distance_distribution.mean(axis=0)
    sample_distance_distribution = np.log(sample_distance_distribution)
    mean_log_distance_distribution = sample_distance_distribution.mean(axis=0)
    mean_relative_distance_distribution = sample_relative_distance_distribution.mean(axis=0)
    sample_relative_distance_distribution = np.log(sample_relative_distance_distribution)
    mean_log_relative_distance_distribution = sample_relative_distance_distribution.mean(axis=0)
    
    def ttest_1samp_bootstrapped(a, popmean, n_boot=0, alternative='two-sided'):
        # rescales and samples input for scipy.stats.ttest_1samp such that it accounts for the bootstrapped nature of the data
        
        if n_boot == 0:
            
            modified_a = a
            
        else:

            # use as many samples as we would have without bootstrapping
            n = a.shape[0]
            modified_n = n // (1+n_boot)

            mean = np.mean(a, axis=0)
            std = np.std(a, axis=0)
            rest = a - mean

            # make data have exactly the same mean
            modified_a = np.empty(shape=(modified_n, *a.shape[1:]))
            modified_a[:] = mean[None,...]

            # make data have exactly the same standarddeviation; all other properties will be wrong in general; but only number of samples, mean and standarddeviation are used in the ttest
            deviation = std * np.sqrt(0.5*modified_n)
            modified_a[0] -= deviation
            modified_a[1] += deviation
        
        return scipy.stats.ttest_1samp(modified_a, popmean, axis=0, alternative=alternative)
    
    p_val_key = 'p_t'
    if len(samples) > 1:
        
        pvals_g = ttest_1samp_bootstrapped(sample_scores,0,n_boot=n_boot,alternative='greater')[1]
        pvals_l = ttest_1samp_bootstrapped(sample_scores,0,n_boot=n_boot,alternative='less')[1]
        direction = pvals_g < pvals_l
        sign = direction * 2 - 1
        if p_corr is not None:
            p_val_key = f'{p_val_key}_{p_corr}'
            pvals = np.array([pvals_g,pvals_l])
            flat_pvals = pvals.flatten()
            flat_pvals = multipletests(flat_pvals, alpha=0.05, method=p_corr)[1]
            pvals = flat_pvals.reshape(pvals.shape)
        pvals = np.where(direction, pvals[0], pvals[1])
        pvals = sign * pvals
    else:
        pvals = None
    
    if len(labels.shape) == 1: # categorical annotation
        annotation = labels.cat.categories
        center = reference_labels.cat.categories
    else: # soft annotation
        annotation = labels.columns
        if isinstance(annotation, pd.CategoricalIndex):
            annotation = annotation.astype(annotation.categories.dtype)
        center = reference_labels.columns
        if isinstance(center, pd.CategoricalIndex):
            center = center.astype(center.categories.dtype)
        
    annotation.name = annotation_key
    center.name = center_key
    result = {
        "occ": mean_scores,
        "log_occ": mean_log_scores,
        p_val_key: pvals,
        "z": mean_z_scores,
        "composition": mean_composition,
        "log_composition": mean_log_composition,
        "distance_distribution": mean_distance_distribution,
        "log_distance_distribution": mean_log_distance_distribution,
        "relative_distance_distribution": mean_relative_distance_distribution,
        "log_relative_distance_distribution": mean_log_relative_distance_distribution,
        "sample_counts": sample_counts,
        "permutation_counts": permutation_counts,
        "interval": intervals,
        "annotation": annotation,
        "center": center,
        "n_boot": n_boot,
    }
    if result_key is not None:
        original_adata.uns[result_key] = result
        result = original_adata
    
    return result

def co_occurrence_matrix(
    adata,
    annotation_key,
    center_key=None,
    sample_key=None,
    position_key=('x','y'),
    result_key=None,
    max_distance=None,
    **kw_args,
):
    
    """\
    Calculates a spatial co-occurence score given by the conditional
    probability to find an annotation `a` at some distance `d` from an
    observation with annotation `b`, normalized by the probability to find an
    annotation `a` at distance `d` from an observation disregarding the value
    of its annotation: `p(a|bd)/p(a|d)`.
    This is a convenience wrapper around :func:`~tc.tl.co_occurrence` which
    just calculates the co-occurence score for the first distance bin.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`
    annotation_key
        The `.obs` or `.obsm` key for the annotation `a` in `p(a|bd)/p(a|d)`.
    center_key
        The `.obs` or `.obsm` key for the annotation `b` in `p(a|bd)/p(a|d)`.
        If `None`, takes the `annotation_key`.
    sample_key
        A categorical sample annotation. The result from different samples is
        averaged for the final result and their standard deviation gives an
        estimation of the error. If `None`, all observations are assumed to be
        on the same sample.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates
    result_key
        The `.uns` key to contain the result. If `None`, the result is returned
        as a dictionary `{ "occ": scores, "interval": intervals, "annotation":
        annotations, "center": center }`, with `scores` containing the
        `p(a|bd)/p(a|d)` values as a :class:`~numpy.ndarray` with dimensions
        according to `a`, `b`, and `d`, `intervals` containing the boundaries
        of the distance bins,  `annotations` containing the order of the `a`
        annotations in `scores`, and `center` the order of the `b` annotations.
    max_distance
        The maximum distance to use. If `None` or `np.inf`, uses the maximum
        distance in the data (if there are multiple samples, then only the
        first sample is used). If the distance matrix is not precomputed (see
        `distance_key`), `None` and `np.inf` result in dense distance
        computation (which can be infeasible for larger datasets).
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tc.tl.co_occurrence`.
        
    Returns
    -------
    Depending on `result_key` returns either the updated input `adata` or the\
    result directly in the format described under `result_key`.
    
    """
    return co_occurrence(
        adata=adata,
        annotation_key=annotation_key,
        center_key=center_key,
        sample_key=sample_key,
        position_key=position_key,
        result_key=result_key,
        max_distance=max_distance,
        min_distance=0,
        delta_distance=max_distance,
        **kw_args,
    )

def co_occurrence_comparison(
    adatas,
    analysis_key,
    result_key=None,
    p_corr='fdr_bh',
    restrict_annotation=None,
    restrict_center=None,
    restrict_intervals=None,
):
    """\
    Find significant changes in the co-occurrence determined by
    :func:`~tc.tl.co_occurrence` between data sets.
    
    Parameters
    ----------
    adatas
        A mapping of labels to :class:`~anndata.AnnData` to specify multiple
        datasets, all with the co-occurence analysis in `.uns`.
    analysis_key
        The `.uns` key with the co-occurence analysis result.
    result_key
        The `.uns[analysis_key]['comparisons']` sub-key to store the comparison
        normalized co-occurrence values to. If `None`, use all the labels from
        `adatas` joined with an `'_'`.
    p_corr
        The name of the p-value correction method to use for calculating the
        significance of the deviation from random distribution. The
        significance is calculated over the samples with a two sample
        t-test and only available if the co-occurrence was calculated for many
        samples. Possible values are the ones available in
        :func:`~statsmodels.stats.multitest.multipletests`. If `None`, only
        uncorrected p-values are used.
    restrict_annotation
        A list-like containing the annotation values to use. If `None`, all
        values are included.
    restrict_center
        A list-like containing the center annotation values to use. If
        `None`, all values are included.
    restrict_intervals
        A list-like containing the indices of the intervals to check. If
        `None`, all intervals are included. Running this separately for the
        intervals generally gives different results due to multiple testing
        correction.
        
    Returns
    -------
    `None`. A call to this function updates the input adatas to contain the\
    result of the comparison in\
    `.uns[analysis_key]['comparisons'][result_key]`.
    
    """
    
    if result_key is None:
        result_key = '_'.join([str(key) for key in adatas])
    
    if restrict_intervals is not None:
        if len(np.array(restrict_intervals).shape) == 0:
            restrict_intervals = [restrict_intervals]
        restrict_intervals = pd.Index(np.array(restrict_intervals))

    values = {}
    groups = []
    n_boot = None
    for key in adatas:
        
        analysis = adatas[key].uns[analysis_key]
        
        if n_boot is None:
            n_boot = analysis['n_boot']
        elif n_boot != analysis['n_boot']:
            raise ValueError('Only co-occurrence analyses with the same number of bootstrap samples can be compared!')
        
        interval_IDs = list(range(len(analysis['interval'])-1))
        if restrict_intervals is not None:
            interval_IDs = restrict_intervals.intersection(interval_IDs)
            if len(restrict_intervals) > len(interval_IDs):
                raise ValueError(f'`restrict_intervals` contained the indices {restrict_intervals[~restrict_intervals.isin(interval_IDs)].tolist()!r} which are not available in the adata with key {key!r}')
                
        pseudo_count = 1 # add pseudo count to avoid division by 0 and log of 0 issues
        
        sample_scores = analysis['sample_counts'] + pseudo_count
        sample_scores = sample_scores * sample_scores.sum(axis=(1,2))[:,None,None,:] / (sample_scores.sum(axis=1)[:,None,:,:] * sample_scores.sum(axis=2)[:,:,None,:])
        for ii in interval_IDs:
            for rj, ref in enumerate(analysis['center']):
                for ai, anno in enumerate(analysis['annotation']):
                    # stringify the combination identifier to avoid issues with multiindices
                    designation = f'{(ai,rj,ii)!r}'
                    
                    if designation not in values:
                        values[designation] = []
                    values[designation].append(sample_scores[:,ai,rj,ii])

        groups.append([key]*len(analysis['sample_counts']))
    
    for designation in values:
        values[designation] = np.concatenate(values[designation])
        
    df = pd.DataFrame({'groups':np.concatenate(groups), **values})
    
    df['groups'] = df['groups'].astype('category')
    
    annotations = list(values.keys())
    
    group_means = df.groupby('groups')[annotations].mean() # adata by designation
    
    # log transform the data to work with fold changes
    df[annotations] = np.log(df[annotations])
    group_means = np.log(group_means)
    
    # normalize the values to be of the same magnitude across designations to compare against the mean logs / log geometric means
    means = group_means.mean(axis=0)
    
    df[annotations] -= means
    group_means -= means
    
    group_means = group_means.T # designation by adata
    group_means.index.name = 'value'
    group_means.columns = group_means.columns.astype(str)
    group_means = group_means.reset_index()
    
    if restrict_annotation is not None:
        if len(np.array(restrict_annotation).shape) == 0:
            restrict_annotation = [restrict_annotation]
        restrict_annotation = pd.Index(np.array([restrict_annotation]))
        annotations = pd.Index(annotations).intersection(restrict_annotation)
        if len(restrict_annotation) > len(annotations):
            raise ValueError(f'`restrict_center` contained the values {restrict_annotation[~restrict_annotation.isin(annotations)].tolist()!r} which are not available in the adata with key {key!r}')
            
    results = enrichments(df, annotations, 'groups', method='t', p_corr=p_corr, restrict_groups=restrict_center, n_boot=n_boot)
    
    if len(results) == 0:
        raise ValueError('There were no enrichment results!')
        #results[['annotation','center','interval']] = np.zeros((0,3))
    else:
    
        for key, result in results.groupby('groups'):
            
            # destringify the combination identifiers for the significances
            ais,rjs,iis = np.array([ast.literal_eval(v) for v in result['value']]).T
        
            # take the last entry of the significance result: this is either the bare p value or the multiple testing corrected one
            pvals_g = np.full_like(adatas[key].uns[analysis_key]['occ'], fill_value=np.nan, dtype=np.float64)
            pvals_l = np.full_like(adatas[key].uns[analysis_key]['occ'], fill_value=np.nan, dtype=np.float64)
            enriched = result['enrichment'] == 'enriched'
            pvals_g[ais[ enriched],rjs[ enriched],iis[ enriched]] = result.iloc[:,-1][ enriched]
            pvals_l[ais[~enriched],rjs[~enriched],iis[~enriched]] = result.iloc[:,-1][~enriched]
            
            # merge the significance pvalues to a single matrix with the sign determining enrichment/depletion
            direction = pvals_g < pvals_l
            sign = direction * 2 - 1
            pvals = np.where(direction, pvals_g, pvals_l)
            pvals = sign * pvals
            
            # destringify the combination identifiers for the means
            group_mean = group_means[key]
            ais,rjs,iis = np.array([ast.literal_eval(v) for v in group_means['value']]).T
        
            # take the last entry of the significance result: this is either the bare p value or the multiple testing corrected one
            group_mean_result = np.full_like(adatas[key].uns[analysis_key]['occ'], fill_value=np.nan, dtype=np.float64)
            group_mean_result[ais,rjs,iis] = group_mean
            
            if 'comparisons' not in adatas[key].uns[analysis_key]:
                adatas[key].uns[analysis_key]['comparisons'] = {}
            adatas[key].uns[analysis_key]['comparisons'][result_key] = {
                result.columns[-1]: pvals,
                'rel_occ': group_mean_result,
            }

@njit(parallel=True,fastmath=True,cache=True)
def _dist_hist_dense(distances, max_distance, delta_distance, weights, ref_weights):
    Ni,Nj = distances.shape
    _Nj,Nw = weights.shape
    assert(Nj==_Nj)
    assert(Nj==len(ref_weights))
    Nd = int(np.ceil(max_distance / delta_distance))
    hist = np.zeros((Ni,Nd,Nw))
    ref_hist = np.zeros((Ni,Nd))
    for i in prange(Ni):
        for j in range(Nj):
            dij = distances[i,j]
            _dij = int(dij / delta_distance)
            if dij <= max_distance:
                hist[i,_dij] += weights[j]
                ref_hist[i,_dij] += ref_weights[j]
    return hist,ref_hist

@njit(parallel=True,fastmath=True,cache=True)
def _dist_hist_sparse(dist_indptr, dist_indices, dist_data, max_distance, delta_distance, weights, ref_weights):
    Ni = len(dist_indptr)-1
    Nj,Nw = weights.shape
    assert(Nj==len(ref_weights))
    Nd = int(np.ceil(max_distance / delta_distance))
    hist = np.zeros((Ni,Nd,Nw))
    ref_hist = np.zeros((Ni,Nd))
    for i in prange(Ni):
        ptr0 = dist_indptr[i]
        ptr1 = dist_indptr[i+1]
        for ptr in range(ptr0,ptr1):
            j = dist_indices[ptr]
            dij = dist_data[ptr]
            _dij = int(dij / delta_distance)
            if dij <= max_distance:
                hist[i,_dij] += weights[j]
                ref_hist[i,_dij] += ref_weights[j]
    return hist,ref_hist

def annotation_coordinate(
    adata,
    annotation_key,
    sample_key=None,
    distance_key=None,
    position_key=('x','y'),
    result_key=None,
    critical_neighbourhood_size=100,
    reference_key=None,
    max_distance=None,
    sparse=True,
    delta_distance=None,
    verbose=1,
    **kw_args,
):
    """\
    Calculates a distance-like quantity from an annotation in space.
    
    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`.
    annotation_key
        The `.obs` or `.obsm` key to calculate distances from. For categorical
        `.obs` keys and for `.obsm` keys, the distances from all components are
        calculated. The numerical value of the annotation is used as a weight
        for distance calculation, see `critical_neighbourhood_size`.
    sample_key
        A categorical sample annotation. The calculation is performed
        separately per sample. If `None`, all observations are assumed to be
        on the same sample.
    distance_key
        The `.obsp` key containing a precomputed distance matrix to use. If
        `None`, the distances are computed on the fly with the positions found
        in `position_key`. Otherwise `position_key` is ignored.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates
    result_key
        The `.obsm` key to contain the result. If `None`, the result is
        returned as a :class:`~pandas.DataFrame`.
    critical_neighbourhood_size
        The aggregated weight of a observations within a certain distance in
        order to call this distance THE distance (see `reference_key`). The
        weight used here is determined by numerical annotation values.
    reference_key
        The `.obs` key to use as reference weights (i.e. the maximum weight
        possible per observation) for neighbourhood size distance correction.
        If `None`, use `1` per observation, which makes sense if the annotation
        is categorical or fractional annotations which should sum to `1`. If
        `False`, dont perform the correction.
    max_distance
        The maximum distance to use. If `None` or `np.inf`, uses the maximum
        distance in the data (if there are multiple samples, then only the
        first sample is used). If the distance matrix is not precomputed (see
        `distance_key`), `None` and `np.inf` result in dense distance
        computation (which can be infeasible for larger datasets).
    sparse
        Whether to calculate a sparse or dense distance matrix, if it is not
        precomputed. If `None`, this is determined by the value of
        `max_distance`.
    delta_distance
        The width in distance for distance discretization. If `None`, takes
        `max_distance/100`.
    verbose
        Level of verbosity, with `0` (not output), `1` (some output), ...
    **kw_args
        Additional keyword arguments are forwarded to on-the-fly distance
        calculation with :func:`~tacco.utils.distance_matrix` if necessary.
        
    Returns
    -------
    Depending on `result_key` returns either the updated input `adata` or the\
    result directly as :class:`~pandas.DataFrame`.
    
    """
    
    if sample_key is None:
        sample_adatas = [adata]
    else:
        sample_adatas = [adata[df.index] for sample,df in adata.obs.groupby(sample_key)]
    
    if distance_key is None:
        if verbose > 0:
            print('annotation_distance: The argument `distance_key` is `None`, meaning that the distance which is now calculated on the fly will not be saved. Providing a precalculated distance saves time in multiple calls to this function.')
    elif distance_key not in adata.obsp:
        raise ValueError(f'The argument `distance_key` is "{distance_key}", but there is no `adata.obsp["{distance_key}"]`!')
    
    _max_distance = max_distance # buffer original max_distance value to trigger the correct path (dense/sparse) for sample number 2+
    if not sparse: # force dense calculation if explicitly asked for
        _max_distance = np.inf
    def _get_dist(si):
        if distance_key is not None:
            if adata is sample_adatas[0]:
                dist = adata.obsp[distance_key]
            else:
                sample_index = adata.obs.index.isin(sample_adatas[si].obs.index)
                dist = adata.obsp[distance_key][sample_index][:,sample_index]
        else:
            if verbose > 0:
                print(f'calculating distance for sample {si+1}/{len(sample_adatas)}')
            dist = distance_matrix(sample_adatas[si], max_distance=_max_distance, position_key=position_key, verbose=verbose, **kw_args)
        return dist
    dist = _get_dist(0)

    if max_distance is None or max_distance == np.inf:
        if scipy.sparse.issparse(dist):
            max_distance = dist.data.max()
        else:
            max_distance = dist.max()
    max_distance = float(max_distance)
    
    if delta_distance is None:
        delta_distance = max_distance / 100
    n_intervals = int(max_distance / delta_distance)
    max_distance = n_intervals * delta_distance
    
    results = []
    for si,sn in enumerate(sample_adatas): # do not parallelize over samples as the memory for a single sample distance matrix can be already quite large
        
        if annotation_key in sample_adatas[si].obs:
            weights = sample_adatas[si].obs[annotation_key]
            if hasattr(weights, 'cat'):
                weights = pd.get_dummies(weights)
            else:
                weights = pd.DataFrame(weights)
        elif annotation_key in sample_adatas[si].obsm:
            weights = sample_adatas[si].obsm[annotation_key].copy()
        else:
            raise ValueError(f'The `annotation_key` {annotation_key!r} is neither in `adata.obs` nor `adata.obsm`!')
        
        columns = weights.columns
        weights = weights.to_numpy()
        
        # consider only not-nan data
        weights[np.isnan(weights).any(axis=1)] = 0

        if reference_key is None or reference_key == False:
            reference_weights = np.ones(len(sample_adatas[si].obs.index),dtype=float)
        elif reference_key in sample_adatas[si].obs:
            reference_weights = sample_adatas[si].obs[reference_key].to_numpy()
        else:
            raise ValueError(f'The `reference_key` {reference_key!r} is not in `adata.obs`!')
        
        common_args = [max_distance, delta_distance, weights, reference_weights]
        
        if si > 0: # already have the first sample distance from above
            dist = _get_dist(si)

        if scipy.sparse.issparse(dist):
            if not isinstance(dist, scipy.sparse.csr_matrix):
                if verbose > 0:
                    print(f'annotation_distance: distance is not in csr format and has to be transformed to csr on the fly')
                dist = scipy.sparse.csr_matrix(dist)
            dist_args = [dist.indptr, dist.indices, dist.data]
            _dist_hist = _dist_hist_sparse
        else:
            dist_args = [dist]
            _dist_hist = _dist_hist_dense
        
        hist,ref_hist = _dist_hist(*dist_args, *common_args)
        
        ref_chist = ref_hist.cumsum(axis=1)
    
        result_c = {}
        for ci,c in enumerate(columns): # go over annotation categories
            chist = hist[...,ci].cumsum(axis=1) # sum histogram over distances
            gg = chist>critical_neighbourhood_size # find the bins which have more counts than the threshhold
            good = gg.any(axis=1) # find the points with at least one distance bin over threshold
            mist = np.argmax(gg,axis=1) # find the first bin over threshold per point - or 0 if none is over threshold
            if reference_key == False:
                sist = mist
                sist = np.where(good, mist, max_distance / delta_distance + 0.5) # clip max distance at the maximum
            else: # correct for threshold effect
                shift = ref_chist[np.arange(len(mist)),mist.astype(int)]-critical_neighbourhood_size # max possible bin values at the distance over threshold, corrected by the threshold value
                gs = ref_chist>=shift[:,None] # find minimum distance to yield the corrected max possible bin value at distance over threshold
                sist = np.where(gs.any(axis=1) & good, np.argmax(gs,axis=1), max_distance / delta_distance + 0.5) # clip max distance at the maximum
            sist = (sist + 0.5) * delta_distance # account for finite bin width and convert from bin units to physical units
            result_c[c] = sist
        
        result = pd.DataFrame(result_c, index=sample_adatas[si].obs.index)

        results.append(result)
    
    result = pd.concat(results)
    
    if result_key is not None:
        adata.obsm[result_key] = result.reindex(index=adata.obs.index)
        result = adata
    
    return result
